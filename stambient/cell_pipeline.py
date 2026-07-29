"""Cell-based pipeline: cells as intact units, empty space binned separately.

Hybrid model:
  - Cells: centroid (x,y), area (DNB count), expression per area
  - Empty bins: centroid (x,y), area, expression per area (from empty DNBs)
  - Ambient from other cells only (natural self-exclusion)

Key advantage over pure bin-level:
  Cells are never split → no "mixed bin" → sstIN's expression isn't
  counted as ambient by neighboring sstIN bins.
"""

import numpy as np
from scipy.sparse import csr_matrix
from typing import Dict, Tuple, Optional, List
from time import perf_counter

from .spatial import (
    build_spatial_distance_graph,
    build_spatial_distance_graph_between,
    distance_graph_to_weights,
    DistanceMetric,
)
from .gpu import (
    resolve_gpu,
    choose_gpu_gene_batch_size,
    sparse_mm,
    sparse_distance_graph_to_gpu_csr,
    weighted_gpu_csr,
    to_cpu,
    to_gpu,
)


def _expression_weight(source: np.ndarray) -> np.ndarray:
    """Expression-weight ambient source to suppress non-expressing cells.

    W(s) = s / (s + s_median) where s_median is the median of positive rates.
    This is a soft sigmoid: high-expressors weight → 1, low-expressors → 0.5
    or lower, preventing non-expressing cells from diluting the signal.

    For cell-type-specific genes (e.g., SST): non-expressing cells
    (rate ~0.005) get w ≈ 0.5 when median = 0.005; expressing cells
    (rate ~0.17) get w ≈ 0.97 → 2× relative weight boost.
    For housekeeping genes (uniform ~0.15): all cells get w ≈ 0.5.
    """
    positive = source[source > 0]
    if len(positive) == 0:
        return np.zeros_like(source)
    s_median = np.median(positive)
    if s_median <= 0:
        return source
    weight = source / (source + s_median)
    return source * weight  # s * s/(s+median) = s²/(s+median)


def _self_confidence_weight(source_c: np.ndarray, mode: str = "1/(1+s/p90)") -> np.ndarray:
    """Per-cell penalty factor. Modes:
    - '1/(1+s/p90)': sigmoid, p90 ref (default)
    - '1/(1+s/p50)': sigmoid, median ref (stronger)
    - '1/(1+(s/p90)²)': quadratic sigmoid (sharp cutoff)
    - 'exp(-s/p90)': exponential decay
    - 'p50/s': linear ramp above median, 1 below

    Supports both 1D (single gene) and 2D (batch of genes, last axis = cells).
    """
    if source_c.ndim == 2:
        return np.array([_self_confidence_weight(row, mode=mode) for row in source_c])

    positive = source_c[source_c > 0]
    if len(positive) == 0:
        return np.ones_like(source_c)
    p50 = np.percentile(positive, 50)
    p90 = np.percentile(positive, 90)
    if p50 <= 0 or p90 <= 0:
        return np.ones_like(source_c)

    if mode == "1/(1+s/p50)":
        return 1.0 / (1.0 + source_c / p50)
    elif mode == "1/(1+(s/p90)²)":
        return 1.0 / (1.0 + (source_c / p90) ** 2)
    elif mode == "exp(-s/p90)":
        return np.exp(-source_c / p90)
    elif mode == "p50/s":
        return np.where(source_c < p50, 1.0, p50 / source_c)
    else:  # default: 1/(1+s/p90)
        return 1.0 / (1.0 + source_c / p90)


def _bin_empty_dnbs(
    dnb_coords: np.ndarray,
    dnb_labels: np.ndarray,
    bin_size: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Assign empty DNBs to square spatial bins in one pass.

    ``dnb_coords`` and ``bin_size`` must use the same spatial unit.  The
    public API requires micrometres, but keeping this helper unit-agnostic is
    useful for adapters that scale the resulting centroids afterwards.

    Returns bin centroids/areas plus two aligned arrays describing the sparse
    DNB-to-bin mapping.  Returning the mapping directly avoids the former
    per-bin boolean scan, whose complexity was O(n_empty * n_bins).
    """
    if not np.isfinite(bin_size) or bin_size <= 0:
        raise ValueError("bin_size must be a positive finite spatial length")

    empty_mask = dnb_labels < 0
    n_empty = empty_mask.sum()
    if n_empty == 0:
        empty = np.zeros(0, dtype=np.int64)
        return np.zeros((0, 2)), empty, empty, empty

    empty_orig_idx = np.flatnonzero(empty_mask)
    empty_coords = dnb_coords[empty_orig_idx]

    x_min, y_min = empty_coords.min(axis=0)
    x_max, y_max = empty_coords.max(axis=0)

    n_bins_x = max(1, int(np.ceil((x_max - x_min) / bin_size)))
    n_bins_y = max(1, int(np.ceil((y_max - y_min) / bin_size)))

    bin_x = np.floor((empty_coords[:, 0] - x_min) / bin_size).astype(np.int64)
    bin_y = np.floor((empty_coords[:, 1] - y_min) / bin_size).astype(np.int64)
    bin_x = np.clip(bin_x, 0, n_bins_x - 1)
    bin_y = np.clip(bin_y, 0, n_bins_y - 1)
    bin_ids = bin_x * np.int64(n_bins_y) + bin_y

    _, inverse = np.unique(bin_ids, return_inverse=True)
    inverse = inverse.astype(np.int64, copy=False)
    n_bins = int(inverse.max()) + 1

    bin_areas = np.bincount(inverse, minlength=n_bins).astype(
        np.int64, copy=False
    )
    bin_coords = np.column_stack(
        (
            np.bincount(inverse, weights=empty_coords[:, 0], minlength=n_bins),
            np.bincount(inverse, weights=empty_coords[:, 1], minlength=n_bins),
        )
    )
    bin_coords /= bin_areas[:, None]

    return bin_coords, bin_areas, empty_orig_idx, inverse


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

    Returns:
        corrected_cell_expr: [genes × n_cells] corrected per-cell expression.
        diagnostics: dict with λ, α, R², etc.
    """
    total_started = perf_counter()
    timings = {}
    if gpu_dtype not in {"float64", "mixed", "float32"}:
        raise ValueError("gpu_dtype must be one of: float64, mixed, float32")

    gpu, gpu_fallback_reason = resolve_gpu(use_gpu)
    if gpu is not None:
        storage_dtype = (
            gpu.torch.float64 if gpu_dtype == "float64" else gpu.torch.float32
        )
        reduction_dtype = (
            gpu.torch.float32 if gpu_dtype == "float32" else gpu.torch.float64
        )
    if verbose:
        if gpu is not None:
            print(f"Using GPU acceleration: {gpu.name}")
        elif use_gpu:
            print("Using CPU fallback")

    if lambda_grid is None:
        lambda_grid = [10, 20, 30, 50, 70, 100, 150, 200]

    n_genes, n_dnbs = dnb_expr.shape
    empty_mask = dnb_labels < 0
    cell_mask_labels = dnb_labels >= 0

    # Keep sparse, convert to CSC for efficient column slicing
    if hasattr(dnb_expr, 'tocsc'):
        dnb_csc = dnb_expr.tocsc()
    else:
        dnb_csc = csr_matrix(dnb_expr).tocsc()

    # ── 1. Extract cells ───────────────────────────────────────
    stage_started = perf_counter()
    unique_cells = np.unique(dnb_labels[cell_mask_labels])
    n_cells = len(unique_cells)
    cell_id_to_idx = {cid: i for i, cid in enumerate(unique_cells)}

    if verbose:
        print(f"Extracting {n_cells} cells...")

    # Vectorized cell area and centroid computation
    cell_indices = np.array([cell_id_to_idx[l] for l in dnb_labels[cell_mask_labels]])
    cell_areas = np.bincount(cell_indices, minlength=n_cells)

    cell_centroids = np.zeros((n_cells, 2))
    np.add.at(cell_centroids[:, 0], cell_indices, dnb_coords[cell_mask_labels, 0])
    np.add.at(cell_centroids[:, 1], cell_indices, dnb_coords[cell_mask_labels, 1])
    valid = cell_areas > 0
    cell_centroids[valid, 0] /= cell_areas[valid]
    cell_centroids[valid, 1] /= cell_areas[valid]

    if verbose:
        print(f"  Areas: {cell_areas.min()}-{cell_areas.max()} DNBs "
              f"(mean {cell_areas.mean():.0f})")

    # Build cell expression via sparse matrix multiplication
    cell_dnb_idx = np.where(cell_mask_labels)[0]
    C = csr_matrix(
        (np.ones(len(cell_dnb_idx), dtype=np.float64), (cell_dnb_idx, cell_indices)),
        shape=(n_dnbs, n_cells)
    )
    cell_expr = np.asarray((dnb_csc @ C).sum(axis=1) if False else (dnb_csc @ C).toarray())
    # Actually: dnb_csc @ C gives [genes × cells] directly
    cell_expr_sp = dnb_csc @ C
    cell_expr = np.asarray(cell_expr_sp.todense()) if hasattr(cell_expr_sp, 'todense') else cell_expr_sp.toarray()
    timings["cell_aggregation_sec"] = perf_counter() - stage_started

    # ── 2. Bin empty DNBs ──────────────────────────────────────
    stage_started = perf_counter()
    if verbose:
        print(f"Binning {empty_mask.sum()} empty DNBs...")

    empty_bin_coords, empty_bin_areas, empty_dnb_indices, empty_bin_indices = _bin_empty_dnbs(
        dnb_coords, dnb_labels, bin_size
    )
    timings["empty_bin_assignment_sec"] = perf_counter() - stage_started
    n_empty_bins = len(empty_bin_areas)

    if verbose:
        print(f"  → {n_empty_bins} empty bins "
              f"(areas {empty_bin_areas.min()}-{empty_bin_areas.max()})")

    # Empty bin expression via sparse indicator matrix
    expression_started = perf_counter()
    if len(empty_dnb_indices) > 0:
        C_empty = csr_matrix(
            (
                np.ones(len(empty_dnb_indices), dtype=np.float64),
                (empty_dnb_indices, empty_bin_indices),
            ),
            shape=(n_dnbs, n_empty_bins)
        )
        empty_bin_expr_sp = dnb_csc @ C_empty
        empty_bin_expr = np.asarray(empty_bin_expr_sp.todense()) if hasattr(empty_bin_expr_sp, 'todense') else empty_bin_expr_sp.toarray()
    else:
        empty_bin_expr = np.zeros((n_genes, 0), dtype=np.float64)
    timings["empty_expression_aggregation_sec"] = (
        perf_counter() - expression_started
    )
    timings["empty_binning_sec"] = perf_counter() - stage_started

    # ── 3. Select high-expression genes ────────────────────────
    stage_started = perf_counter()
    if n_empty_bins == 0:
        raise RuntimeError("No empty bins available for ambient estimation.")

    total_empty = empty_bin_areas.sum()
    mean_empty = empty_bin_expr.sum(axis=1) / total_empty
    if n_high_genes is None:
        n_select = n_genes
    else:
        n_select = min(n_high_genes, n_genes)
    gene_indices = np.argsort(mean_empty)[::-1][:n_select]

    if verbose:
        print(f"Selected {len(gene_indices)} high-expression genes")
    timings["gene_selection_sec"] = perf_counter() - stage_started

    # ── 4. Prepare spatial graphs (cells → empty bins) ─────────
    # We need distances from empty bins to cells (for λ/α estimation)
    # and from cells to cells (for correction).
    # Build them as separate cross-graphs to avoid the full combined matrix.
    stage_started = perf_counter()
    empty_to_cell_distances = build_spatial_distance_graph_between(
        empty_bin_coords, cell_centroids, max_radius
    )
    empty_to_cell_graph_nnz = int(empty_to_cell_distances.nnz)
    if gpu is not None:
        empty_distance_gpu = sparse_distance_graph_to_gpu_csr(
            empty_to_cell_distances, gpu, dtype=storage_dtype
        )
    timings["empty_graph_build_sec"] = perf_counter() - stage_started

    # ── 5. Estimate λ ──────────────────────────────────────────
    stage_started = perf_counter()

    if distance_metric == "inverse":
        # The inverse kernel 1/(d+eps) has no λ dependence: every candidate
        # would produce identical weights, so a grid search is meaningless.
        best_lam = None
        best_rss = np.nan
        if verbose:
            print(
                "distance_metric='inverse' does not use λ; "
                "skipping the λ grid search."
            )
    else:
        if verbose:
            print(f"Estimating λ via grid search...")

        n_lambda_use = min(n_lambda_genes, len(gene_indices))
        lambda_gene_indices = gene_indices[:n_lambda_use]

        # Precompute cell source strengths: expression / area
        cell_source = np.zeros((n_lambda_use, n_cells), dtype=np.float64)
        for i, g_idx in enumerate(lambda_gene_indices):
            cell_source[i] = np.divide(cell_expr[g_idx], cell_areas,
                                        out=np.zeros(n_cells), where=cell_areas > 0)
            if use_expr_weight:
                cell_source[i] = _expression_weight(cell_source[i])

        best_lam = lambda_grid[0]
        best_rss = np.inf
        w = empty_bin_areas.astype(np.float64)
        y_obs_lambda = empty_bin_expr[lambda_gene_indices].T  # [n_empty_bins × n_lambda_use]

        if gpu is not None:
            cell_source_gpu = to_gpu(cell_source.T, gpu, dtype=storage_dtype)
            w_gpu = to_gpu(w[:, None], gpu, dtype=reduction_dtype)
            y_obs_lambda_gpu = to_gpu(y_obs_lambda, gpu, dtype=reduction_dtype)
            areas_gpu = to_gpu(
                empty_bin_areas[:, None], gpu, dtype=reduction_dtype
            )

        for lam in lambda_grid:
            if gpu is not None:
                W_empty_gpu = weighted_gpu_csr(
                    empty_distance_gpu, lam, distance_metric, gpu
                )
                weighted_sums_gpu = sparse_mm(
                    W_empty_gpu, cell_source_gpu, gpu
                ).to(dtype=reduction_dtype)
                N_gb_gpu = areas_gpu * weighted_sums_gpu
                denom_gpu = (w_gpu * N_gb_gpu.square()).sum(dim=0)
                numer_gpu = (w_gpu * y_obs_lambda_gpu * N_gb_gpu).sum(dim=0)
                alpha_gpu = gpu.torch.where(
                    denom_gpu > 0, numer_gpu / denom_gpu, gpu.torch.zeros_like(denom_gpu)
                ).clamp_min(0.0)
                residuals_gpu = y_obs_lambda_gpu - alpha_gpu.unsqueeze(0) * N_gb_gpu
                total_rss = float((w_gpu * residuals_gpu.square()).sum().item())
            else:
                W_empty_to_cell = distance_graph_to_weights(
                    empty_to_cell_distances, lam, distance_metric
                )
                # Batched weighted sums for all lambda genes at once
                weighted_sums = W_empty_to_cell.dot(cell_source.T)
                N_gb_all = empty_bin_areas[:, None] * weighted_sums

                denom = (w[:, None] * N_gb_all ** 2).sum(axis=0)
                alpha_g = np.divide(
                    (w[:, None] * y_obs_lambda * N_gb_all).sum(axis=0),
                    denom,
                    out=np.zeros(n_lambda_use, dtype=np.float64),
                    where=denom > 0,
                )
                alpha_g = np.maximum(alpha_g, 0.0)

                residuals = y_obs_lambda - alpha_g[None, :] * N_gb_all
                rss_per_gene = (w[:, None] * residuals ** 2).sum(axis=0)
                total_rss = rss_per_gene.sum()

            if total_rss < best_rss:
                best_rss = total_rss
                best_lam = lam

        if verbose:
            print(f"  → Optimal λ = {best_lam:.1f} μm (RSS = {best_rss:.2f})")

    # λ is unused by the inverse kernel; any value yields the same weights.
    lam_weights = best_lam if best_lam is not None else 1.0
    if gpu is not None:
        best_W_empty_gpu = weighted_gpu_csr(
            empty_distance_gpu, lam_weights, distance_metric, gpu
        )
    else:
        W_empty_to_cell = distance_graph_to_weights(
            empty_to_cell_distances, lam_weights, distance_metric
        )
    timings["lambda_search_sec"] = perf_counter() - stage_started

    # ── 6. Estimate α per gene ──────────────────────────────────
    stage_started = perf_counter()
    if verbose:
        print("Estimating gene-specific leakage rates α...")

    n_genes_use = len(gene_indices)
    alphas = np.zeros(n_genes_use, dtype=np.float64)
    r2_scores = np.zeros(n_genes_use, dtype=np.float64)
    w_empty = empty_bin_areas.astype(np.float64)

    eps = 1e-15
    if gpu is not None:
        w_empty_gpu = to_gpu(w_empty[:, None], gpu, dtype=reduction_dtype)
        if gpu_gene_batch_size is None:
            effective_gpu_gene_batch_size = choose_gpu_gene_batch_size(
                gpu,
                n_rows=n_empty_bins,
                n_cells=n_cells,
                dtype=storage_dtype,
                max_batch_size=n_genes_use,
            )
        else:
            effective_gpu_gene_batch_size = max(1, int(gpu_gene_batch_size))
        for start in range(0, n_genes_use, effective_gpu_gene_batch_size):
            stop = min(start + effective_gpu_gene_batch_size, n_genes_use)
            batch_gene_indices = gene_indices[start:stop]
            sources_batch = np.divide(
                cell_expr[batch_gene_indices],
                cell_areas[None, :],
                out=np.zeros((stop - start, n_cells), dtype=np.float64),
                where=cell_areas[None, :] > 0,
            )
            if use_expr_weight:
                sources_batch = np.array([_expression_weight(s) for s in sources_batch])
            y_obs_batch = empty_bin_expr[batch_gene_indices].T

            sources_gpu = to_gpu(sources_batch.T, gpu, dtype=storage_dtype)
            y_obs_gpu = to_gpu(y_obs_batch, gpu, dtype=reduction_dtype)
            weighted_sums_gpu = sparse_mm(
                best_W_empty_gpu, sources_gpu, gpu
            ).to(dtype=reduction_dtype)
            N_gb_gpu = areas_gpu * weighted_sums_gpu
            denom_gpu = (w_empty_gpu * N_gb_gpu.square()).sum(dim=0)
            numer_gpu = (w_empty_gpu * y_obs_gpu * N_gb_gpu).sum(dim=0)
            alphas_gpu = gpu.torch.where(
                denom_gpu > 0,
                numer_gpu / denom_gpu,
                gpu.torch.zeros_like(denom_gpu),
            ).clamp_min(0.0)
            ss_res_gpu = (
                w_empty_gpu * (y_obs_gpu - alphas_gpu.unsqueeze(0) * N_gb_gpu).square()
            ).sum(dim=0)
            if w_empty.sum() > 0:
                y_mean_gpu = (w_empty_gpu * y_obs_gpu).sum(dim=0) / w_empty.sum()
            else:
                y_mean_gpu = gpu.torch.zeros(
                    stop - start, dtype=reduction_dtype, device=gpu.device
                )
            ss_tot_gpu = (
                w_empty_gpu * (y_obs_gpu - y_mean_gpu.unsqueeze(0)).square()
            ).sum(dim=0)
            r2_gpu = gpu.torch.where(
                ss_tot_gpu > eps,
                1.0 - ss_res_gpu / ss_tot_gpu,
                gpu.torch.where(
                    ss_res_gpu > eps,
                    gpu.torch.zeros_like(ss_res_gpu),
                    gpu.torch.ones_like(ss_res_gpu),
                ),
            )
            alphas[start:stop] = to_cpu(alphas_gpu, gpu)
            r2_scores[start:stop] = to_cpu(r2_gpu, gpu)
    else:
        # Precompute source strengths for all high-expression genes in one matrix
        sources = np.divide(
            cell_expr[gene_indices], cell_areas[None, :],
            out=np.zeros((n_genes_use, n_cells), dtype=np.float64),
            where=cell_areas[None, :] > 0,
        )
        if use_expr_weight:
            sources = np.array([_expression_weight(s) for s in sources])
        y_obs_all = empty_bin_expr[gene_indices].T

        # Batched weighted sums and OLS for all genes
        weighted_sums = W_empty_to_cell.dot(sources.T)
        N_gb_all = empty_bin_areas[:, None] * weighted_sums

        denom = (w_empty[:, None] * N_gb_all ** 2).sum(axis=0)
        alphas = np.divide(
            (w_empty[:, None] * y_obs_all * N_gb_all).sum(axis=0),
            denom,
            out=np.zeros(n_genes_use, dtype=np.float64),
            where=denom > 0,
        )
        alphas = np.maximum(alphas, 0.0)
        alphas[denom == 0] = 0.0

        ss_res = (w_empty[:, None] * (y_obs_all - alphas[None, :] * N_gb_all) ** 2).sum(axis=0)
        y_mean = (w_empty[:, None] * y_obs_all).sum(axis=0) / w_empty.sum() if w_empty.sum() > 0 else 0.0
        ss_tot = (w_empty[:, None] * (y_obs_all - y_mean[None, :]) ** 2).sum(axis=0)
        ss_ratio = np.divide(
            ss_res,
            ss_tot,
            out=np.zeros_like(ss_res),
            where=ss_tot > eps,
        )
        r2_scores = np.where(
            ss_tot > eps,
            1.0 - ss_ratio,
            np.where(ss_res > eps, 0.0, 1.0),
        )
        effective_gpu_gene_batch_size = None

    timings["alpha_estimation_sec"] = perf_counter() - stage_started

    n_corrected = int((r2_scores >= r2_threshold).sum())
    if verbose:
        print(f"  → {n_corrected}/{n_genes_use} genes pass R² threshold ({r2_threshold})")

    # ── 7. Correct cells ────────────────────────────────────────
    if verbose:
        print("Applying cell-level ambient correction...")

    stage_started = perf_counter()
    cell_to_cell_distances = build_spatial_distance_graph(
        cell_centroids, max_radius
    )
    cell_to_cell_graph_nnz = int(cell_to_cell_distances.nnz)
    if gpu is not None:
        cell_distance_gpu = sparse_distance_graph_to_gpu_csr(
            cell_to_cell_distances, gpu, dtype=storage_dtype
        )
        W_cell_gpu = weighted_gpu_csr(
            cell_distance_gpu, lam_weights, distance_metric, gpu
        )
    else:
        W_cell_to_cell = distance_graph_to_weights(
            cell_to_cell_distances, lam_weights, distance_metric
        )
    timings["cell_graph_build_sec"] = perf_counter() - stage_started
    stage_started = perf_counter()

    corrected_expr = np.zeros((n_genes, n_cells), dtype=np.float64)

    gene_idx_set = set(gene_indices)
    gene_idx_to_pos = {int(g_idx): i for i, g_idx in enumerate(gene_indices)}

    # Identify genes to correct
    corrected_gene_indices = []
    corrected_positions = []
    for g_idx in range(n_genes):
        if g_idx not in gene_idx_set:
            corrected_expr[g_idx] = cell_expr[g_idx]
            continue
        pos = gene_idx_to_pos[g_idx]
        if r2_scores[pos] < r2_threshold:
            corrected_expr[g_idx] = cell_expr[g_idx]
            continue
        corrected_gene_indices.append(g_idx)
        corrected_positions.append(pos)

    if corrected_gene_indices:
        cg_idx = np.array(corrected_gene_indices, dtype=np.int64)
        cpos = np.array(corrected_positions, dtype=np.int64)

        if gpu is not None:
            cell_areas_gpu = to_gpu(
                cell_areas[:, None], gpu, dtype=reduction_dtype
            )
            for start in range(0, len(cg_idx), effective_gpu_gene_batch_size):
                stop = min(start + effective_gpu_gene_batch_size, len(cg_idx))
                batch_gene_indices = cg_idx[start:stop]
                batch_positions = cpos[start:stop]
                corr_sources = np.divide(
                    cell_expr[batch_gene_indices],
                    cell_areas[None, :],
                    out=np.zeros((stop - start, n_cells), dtype=np.float64),
                    where=cell_areas[None, :] > 0,
                )
                if use_expr_weight:
                    corr_sources = np.array([_expression_weight(s) for s in corr_sources])
                corr_sources_gpu = to_gpu(
                    corr_sources.T, gpu, dtype=storage_dtype
                )
                neighbor_gpu = sparse_mm(
                    W_cell_gpu, corr_sources_gpu, gpu
                ).to(dtype=reduction_dtype)
                ambient_gpu = (
                    to_gpu(
                        alphas[batch_positions][None, :],
                        gpu,
                        dtype=reduction_dtype,
                    )
                    * cell_areas_gpu
                    * neighbor_gpu
                )
                if self_confidence_penalty:
                    penalties = _self_confidence_weight(corr_sources, mode=penalty_mode)
                    ambient_gpu *= to_gpu(
                        penalties.T, gpu, dtype=reduction_dtype
                    )
                corrected_gpu = (
                    to_gpu(
                        cell_expr[batch_gene_indices].T,
                        gpu,
                        dtype=reduction_dtype,
                    )
                    - ambient_gpu
                ).clamp_min(0.0)
                corrected_expr[batch_gene_indices] = to_cpu(corrected_gpu.T, gpu)
        else:
            corr_sources = np.divide(
                cell_expr[cg_idx], cell_areas[None, :],
                out=np.zeros((len(cg_idx), n_cells), dtype=np.float64),
                where=cell_areas[None, :] > 0,
            )
            if use_expr_weight:
                corr_sources = np.array([_expression_weight(s) for s in corr_sources])
            # Single sparse-dense matrix multiply for all corrected genes
            neighbor_contribs = W_cell_to_cell.dot(corr_sources.T)
            ambient = alphas[cpos][None, :] * cell_areas[:, None] * neighbor_contribs
            if self_confidence_penalty:
                penalties = _self_confidence_weight(corr_sources, mode=penalty_mode)
                ambient *= penalties.T
            corrected_expr[cg_idx] = np.maximum(cell_expr[cg_idx] - ambient.T, 0.0)

    timings["correction_sec"] = perf_counter() - stage_started
    timings["total_sec"] = perf_counter() - total_started

    diagnostics = {
        "lambda_estimated": best_lam,
        "lambda_grid": lambda_grid,
        "spatial_unit": "micrometre",
        "bin_size": float(bin_size),
        "max_radius": float(max_radius),
        "n_genes_corrected": n_corrected,
        "n_high_genes_selected": n_genes_use,
        "r2_threshold": r2_threshold,
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
