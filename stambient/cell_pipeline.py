"""Cell-based pipeline: cells as intact units, empty space binned separately.

Hybrid model:
  - Cells: centroid (x,y), area (DNB count), expression per area
  - Empty bins: centroid (x,y), area, expression per area (from empty DNBs)
  - Ambient from other cells only (natural self-exclusion)

Key advantage over pure bin-level:
  Cells are never split → no "mixed bin" → sstIN's expression isn't
  counted as ambient by neighboring sstIN bins.

This module is the pipeline orchestrator only. The stages it coordinates
live in dedicated modules:

  weights           → expression / self-confidence weighting helpers
  binning           → empty-DNB spatial binning
  aggregation       → cell & empty-bin aggregation, gene selection
  graphs            → empty→cell and cell→cell distance graphs
  lambda_search     → lambda grid search
  alpha_estimation  → per-gene leakage rates alpha and R²
  correction        → cell-level ambient subtraction

The orchestrator resolves the GPU context once, times every stage, and
assembles the diagnostics dict.
"""

import numpy as np
from scipy.sparse import csr_matrix
from typing import Dict, Tuple, Optional, List
from time import perf_counter

from .spatial import DistanceMetric
from .gpu import resolve_gpu
from .binning import bin_empty_dnbs, _bin_empty_dnbs
from .weights import expression_weight, self_confidence_weight
from .weights import _expression_weight, _self_confidence_weight
from .aggregation import (
    aggregate_empty_bin_expression,
    extract_cells,
    select_high_expression_genes,
)
from .graphs import build_cell_to_cell_graph, build_empty_to_cell_graph
from .lambda_search import estimate_lambda
from .alpha_estimation import estimate_alphas
from .correction import correct_cells
from .latent_correction import correct_cells_latent


def cell_pipeline_fit(
    dnb_expr: np.ndarray,
    dnb_coords: np.ndarray,
    dnb_labels: np.ndarray,
    bin_size: float = 50.0,
    distance_metric: DistanceMetric = "exponential",
    max_radius: float = 200.0,
    n_high_genes: Optional[int] = None,
    n_lambda_genes: int = 50,
    r2_threshold: float = 0.01,
    lambda_grid: Optional[List[float]] = None,
    use_expr_weight: bool = False,
    self_confidence_penalty: bool = True,
    penalty_mode: str = "1/(1+(s/p90)²)",
    verbose: bool = True,
    use_gpu: bool = False,
    gpu_gene_batch_size: Optional[int] = None,
    gpu_dtype: str = "float64",
    inference_mode: str = "legacy",
    latent_eta: float = 0.5,
    latent_max_iter: int = 20,
    latent_tol: float = 1e-4,
) -> Tuple[np.ndarray, Dict]:
    """Run cell-based SPARKLE pipeline.

    Args:
        dnb_expr: [genes × DNBs] DNB-level expression.
        dnb_coords: [DNBs × 2] coordinates in micrometres.
        dnb_labels: [DNBs] cell IDs (-1 for empty).
        bin_size: Side length of each square empty-space bin in micrometres.
        max_radius: Spatial-graph truncation radius in micrometres.
        lambda_grid: Candidate spatial-decay lengths in micrometres. If None,
            use the default physical-length grid. Ignored when
            ``distance_metric='inverse'`` (that kernel does not use λ, so the
            grid search is skipped and ``lambda_estimated`` is None).
        n_high_genes: Number of top high-expression genes to select for
            correction. If None, all genes are used.
        ... (standard SPARKLE params)
        use_gpu: Use PyTorch CUDA for batched sparse/dense computations.
            Falls back to CPU with a warning if CUDA is unavailable.
        gpu_gene_batch_size: Number of genes processed per CUDA batch. If None,
            choose it adaptively from currently available GPU memory.
        gpu_dtype: GPU precision mode: ``float64`` for strict reproducibility,
            ``mixed`` for float32 sparse products plus float64 reductions, or
            ``float32`` for maximum throughput.
        inference_mode: ``legacy`` (SPARKLE 1.x, leakage source = observed Y)
            or ``latent`` (2.0-alpha1: damped fixed-point solve for latent
            clean expression X as the leakage source).  λ, α and R² gating
            are identical in both modes.
        latent_eta: damping factor of the latent fixed-point update.
        latent_max_iter: maximum latent fixed-point iterations.
        latent_tol: relative L1 convergence tolerance of the latent solve.

    Returns:
        corrected_cell_expr: [genes × n_cells] corrected per-cell expression.
        diagnostics: dict with λ, α, R², etc.
    """
    total_started = perf_counter()
    timings = {}
    if gpu_dtype not in {"float64", "mixed", "float32"}:
        raise ValueError("gpu_dtype must be one of: float64, mixed, float32")
    if inference_mode not in {"legacy", "latent"}:
        raise ValueError(
            f"inference_mode must be 'legacy' or 'latent'; got {inference_mode!r}"
        )

    gpu, gpu_fallback_reason = resolve_gpu(use_gpu)
    if gpu is not None:
        storage_dtype = (
            gpu.torch.float64 if gpu_dtype == "float64" else gpu.torch.float32
        )
        reduction_dtype = (
            gpu.torch.float32 if gpu_dtype == "float32" else gpu.torch.float64
        )
    else:
        storage_dtype = None
        reduction_dtype = None
    if verbose:
        if gpu is not None:
            print(f"Using GPU acceleration: {gpu.name}")
        elif use_gpu:
            print("Using CPU fallback")

    if lambda_grid is None:
        lambda_grid = [10, 20, 30, 50, 70, 100, 150, 200]

    n_genes, n_dnbs = dnb_expr.shape
    empty_mask = dnb_labels < 0

    # Keep sparse, convert to CSC for efficient column slicing
    if hasattr(dnb_expr, 'tocsc'):
        dnb_csc = dnb_expr.tocsc()
    else:
        dnb_csc = csr_matrix(dnb_expr).tocsc()

    # ── 1. Extract cells ───────────────────────────────────────
    stage_started = perf_counter()
    unique_cells, cell_areas, cell_centroids, cell_expr = extract_cells(
        dnb_csc, dnb_coords, dnb_labels
    )
    n_cells = len(unique_cells)
    if verbose:
        print(f"Extracting {n_cells} cells...")
        print(
            f"  Areas: {cell_areas.min()}-{cell_areas.max()} DNBs "
            f"(mean {cell_areas.mean():.0f})"
        )
    timings["cell_aggregation_sec"] = perf_counter() - stage_started

    # ── 2. Bin empty DNBs ──────────────────────────────────────
    stage_started = perf_counter()
    if verbose:
        print(f"Binning {empty_mask.sum()} empty DNBs...")

    empty_bin_coords, empty_bin_areas, empty_dnb_indices, empty_bin_indices = (
        bin_empty_dnbs(dnb_coords, dnb_labels, bin_size)
    )
    timings["empty_bin_assignment_sec"] = perf_counter() - stage_started
    n_empty_bins = len(empty_bin_areas)

    if verbose:
        print(
            f"  → {n_empty_bins} empty bins "
            f"(areas {empty_bin_areas.min()}-{empty_bin_areas.max()})"
        )

    # Empty bin expression via sparse indicator matrix
    expression_started = perf_counter()
    empty_bin_expr = aggregate_empty_bin_expression(
        dnb_csc,
        empty_dnb_indices,
        empty_bin_indices,
        n_empty_bins,
        n_genes,
    )
    timings["empty_expression_aggregation_sec"] = (
        perf_counter() - expression_started
    )
    timings["empty_binning_sec"] = perf_counter() - stage_started

    # ── 3. Select high-expression genes ────────────────────────
    stage_started = perf_counter()
    if n_empty_bins == 0:
        raise RuntimeError("No empty bins available for ambient estimation.")
    gene_indices = select_high_expression_genes(
        empty_bin_expr, empty_bin_areas, n_high_genes
    )
    if verbose:
        print(f"Selected {len(gene_indices)} high-expression genes")
    timings["gene_selection_sec"] = perf_counter() - stage_started

    # ── 4. Prepare empty→cell spatial graph ────────────────────
    # Distances from empty bins to cells drive the λ/α estimation.  The
    # full combined matrix is avoided by building the cross-graph only.
    stage_started = perf_counter()
    (
        empty_to_cell_distances,
        empty_to_cell_graph_nnz,
        empty_distance_gpu,
    ) = build_empty_to_cell_graph(
        empty_bin_coords, cell_centroids, max_radius, gpu, storage_dtype
    )
    timings["empty_graph_build_sec"] = perf_counter() - stage_started

    # ── 5. Estimate λ ──────────────────────────────────────────
    stage_started = perf_counter()
    best_lam, best_rss, lambda_search_rss, W_empty = estimate_lambda(
        lambda_grid=lambda_grid,
        distance_metric=distance_metric,
        gene_indices=gene_indices,
        n_lambda_genes=n_lambda_genes,
        cell_expr=cell_expr,
        cell_areas=cell_areas,
        empty_bin_expr=empty_bin_expr,
        empty_bin_areas=empty_bin_areas,
        empty_to_cell_distances=empty_to_cell_distances,
        empty_distance_gpu=empty_distance_gpu,
        use_expr_weight=use_expr_weight,
        gpu=gpu,
        storage_dtype=storage_dtype,
        reduction_dtype=reduction_dtype,
        verbose=verbose,
    )
    # λ is unused by the inverse kernel; any value yields the same weights.
    lam_weights = best_lam if best_lam is not None else 1.0
    timings["lambda_search_sec"] = perf_counter() - stage_started

    # ── 6. Estimate α per gene ─────────────────────────────────
    stage_started = perf_counter()
    if verbose:
        print("Estimating gene-specific leakage rates α...")

    alphas, r2_scores, effective_gpu_gene_batch_size = estimate_alphas(
        gene_indices=gene_indices,
        cell_expr=cell_expr,
        cell_areas=cell_areas,
        empty_bin_expr=empty_bin_expr,
        empty_bin_areas=empty_bin_areas,
        W_empty=W_empty,
        use_expr_weight=use_expr_weight,
        gpu=gpu,
        storage_dtype=storage_dtype,
        reduction_dtype=reduction_dtype,
        gpu_gene_batch_size=gpu_gene_batch_size,
    )
    timings["alpha_estimation_sec"] = perf_counter() - stage_started

    n_genes_use = len(gene_indices)
    n_corrected = int((r2_scores >= r2_threshold).sum())
    if verbose:
        print(
            f"  → {n_corrected}/{n_genes_use} genes pass R² threshold "
            f"({r2_threshold})"
        )

    # ── 7. Correct cells ───────────────────────────────────────
    if verbose:
        print("Applying cell-level ambient correction...")

    stage_started = perf_counter()
    cell_to_cell_distances, cell_to_cell_graph_nnz, cell_distance_gpu = (
        build_cell_to_cell_graph(cell_centroids, max_radius, gpu, storage_dtype)
    )
    timings["cell_graph_build_sec"] = perf_counter() - stage_started

    stage_started = perf_counter()
    latent_info = None
    correction_kwargs = dict(
        gene_indices=gene_indices,
        r2_scores=r2_scores,
        r2_threshold=r2_threshold,
        n_genes=n_genes,
        n_cells=n_cells,
        cell_expr=cell_expr,
        cell_areas=cell_areas,
        alphas=alphas,
        cell_distance_graph=(
            cell_distance_gpu if gpu is not None else cell_to_cell_distances
        ),
        lam_weights=lam_weights,
        distance_metric=distance_metric,
        use_expr_weight=use_expr_weight,
        self_confidence_penalty=self_confidence_penalty,
        penalty_mode=penalty_mode,
        gpu=gpu,
        storage_dtype=storage_dtype,
        reduction_dtype=reduction_dtype,
        effective_gpu_gene_batch_size=effective_gpu_gene_batch_size,
    )
    if inference_mode == "latent":
        corrected_expr, latent_info = correct_cells_latent(
            eta=latent_eta,
            max_iter=latent_max_iter,
            tol=latent_tol,
            **correction_kwargs,
        )
        if verbose:
            print(
                f"  → latent-X solve converged={latent_info['converged']} "
                f"in {max(latent_info['n_iterations'], default=0)} iterations "
                f"(eta={latent_eta}, tol={latent_tol})"
            )
    else:
        corrected_expr = correct_cells(**correction_kwargs)
    timings["correction_sec"] = perf_counter() - stage_started
    timings["total_sec"] = perf_counter() - total_started

    diagnostics = {
        "lambda_estimated": best_lam,
        "lambda_grid": lambda_grid,
        "lambda_search_rss": lambda_search_rss,
        "lambda_search_best_rss": float(best_rss),
        "spatial_unit": "micrometre",
        "bin_size": float(bin_size),
        "max_radius": float(max_radius),
        "n_genes_corrected": n_corrected,
        "n_high_genes_selected": n_genes_use,
        "r2_threshold": r2_threshold,
        "inference_mode": inference_mode,
        "latent": latent_info,
        "alpha_mean": float(alphas.mean()) if n_genes_use > 0 else 0.0,
        "r2_mean": float(r2_scores.mean()) if n_genes_use > 0 else 0.0,
        "gene_indices": gene_indices,
        "r2_scores": r2_scores,
        "alphas": alphas,
        "n_cells": n_cells,
        "n_empty_bins": n_empty_bins,
        "empty_bin_areas_mean": float(empty_bin_areas.mean()) if n_empty_bins > 0 else 0,
        "empty_bin_dnb_count_min": int(empty_bin_areas.min()) if n_empty_bins > 0 else 0,
        "empty_bin_dnb_count_median": float(np.median(empty_bin_areas)) if n_empty_bins > 0 else 0,
        "empty_bin_dnb_count_max": int(empty_bin_areas.max()) if n_empty_bins > 0 else 0,
        "cell_areas_mean": float(cell_areas.mean()),
        "compute_backend": "gpu" if gpu is not None else "cpu",
        "gpu_requested": bool(use_gpu),
        "gpu_device": gpu.name if gpu is not None else None,
        "gpu_fallback_reason": gpu_fallback_reason,
        "gpu_gene_batch_size": effective_gpu_gene_batch_size if gpu is not None else None,
        "gpu_dtype": gpu_dtype if gpu is not None else None,
        "gpu_sparse_format": "csr" if gpu is not None else None,
        "empty_to_cell_graph_nnz": empty_to_cell_graph_nnz,
        "cell_to_cell_graph_nnz": cell_to_cell_graph_nnz,
        "timings_sec": timings,
    }

    return corrected_expr, diagnostics
