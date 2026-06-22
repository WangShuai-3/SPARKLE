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
from scipy.sparse import csr_matrix, lil_matrix, coo_matrix
from typing import Dict, Tuple, Optional, List

from .spatial import build_spatial_graph, DistanceMetric


def _self_confidence_weight(source_c: np.ndarray, mode: str = "1/(1+s/p90)") -> np.ndarray:
    """Per-cell penalty factor. Modes:
    - '1/(1+s/p90)': sigmoid, p90 ref (default)
    - '1/(1+s/p50)': sigmoid, median ref (stronger)
    - '1/(1+(s/p90)²)': quadratic sigmoid (sharp cutoff)
    - 'exp(-s/p90)': exponential decay
    - 'p50/s': linear ramp above median, 1 below
    """
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
    bin_size: int,
) -> Tuple[np.ndarray, np.ndarray, list]:
    """Bin only empty DNBs (label = -1) into spatial bins."""
    empty_mask = dnb_labels < 0
    n_empty = empty_mask.sum()
    if n_empty == 0:
        return np.zeros((0, 2)), np.zeros(0, dtype=np.int64), []

    empty_coords = dnb_coords[empty_mask]
    empty_orig_idx = np.where(empty_mask)[0]

    x_min, y_min = empty_coords.min(axis=0)
    x_max, y_max = empty_coords.max(axis=0)

    n_bins_x = max(1, int(np.ceil((x_max - x_min) / bin_size)))

    bin_x = np.floor((empty_coords[:, 0] - x_min) / bin_size).astype(np.int64)
    bin_y = np.floor((empty_coords[:, 1] - y_min) / bin_size).astype(np.int64)
    bin_x = np.clip(bin_x, 0, n_bins_x - 1)
    bin_y = np.clip(bin_y, 0, n_bins_x - 1)  # reuse n_bins_x for both dims
    bin_ids = bin_x.astype(np.int64) * 100000 + bin_y.astype(np.int64)

    unique_bins, inverse = np.unique(bin_ids, return_inverse=True)
    n_bins = len(unique_bins)

    bin_areas = np.bincount(inverse, minlength=n_bins)
    bin_coords = np.zeros((n_bins, 2))
    np.add.at(bin_coords[:, 0], inverse, empty_coords[:, 0])
    np.add.at(bin_coords[:, 1], inverse, empty_coords[:, 1])
    bin_coords[:, 0] /= bin_areas
    bin_coords[:, 1] /= bin_areas

    bin_dnb_idx = [empty_orig_idx[inverse == i] for i in range(n_bins)]

    return bin_coords, bin_areas, bin_dnb_idx


def cell_pipeline_fit(
    dnb_expr: np.ndarray,
    dnb_coords: np.ndarray,
    dnb_labels: np.ndarray,
    bin_size: int = 50,
    distance_metric: DistanceMetric = "exponential",
    max_radius: float = 200.0,
    n_high_genes: int = 500,
    n_lambda_genes: int = 50,
    r2_threshold: float = 0.05,
    lambda_grid: Optional[List[float]] = None,
    use_expr_weight: bool = False,
    self_confidence_penalty: bool = True,
    penalty_mode: str = "1/(1+(s/p90)²)",
    verbose: bool = True,
) -> Tuple[np.ndarray, Dict]:
    """Run cell-based SPARKLE pipeline.

    Args:
        dnb_expr: [genes × DNBs] DNB-level expression.
        dnb_coords: [DNBs × 2] coordinates.
        dnb_labels: [DNBs] cell IDs (-1 for empty).
        ... (standard SPARKLE params)

    Returns:
        corrected_cell_expr: [genes × n_cells] corrected per-cell expression.
        diagnostics: dict with λ, α, R², etc.
    """
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

    # ── 2. Bin empty DNBs ──────────────────────────────────────
    if verbose:
        print(f"Binning {empty_mask.sum()} empty DNBs...")

    empty_bin_coords, empty_bin_areas, empty_bin_dnb_idx = _bin_empty_dnbs(
        dnb_coords, dnb_labels, bin_size
    )
    n_empty_bins = len(empty_bin_areas)

    if verbose:
        print(f"  → {n_empty_bins} empty bins "
              f"(areas {empty_bin_areas.min()}-{empty_bin_areas.max()})")

    # Empty bin expression via sparse indicator matrix
    empty_dnb_list = np.concatenate(empty_bin_dnb_idx) if empty_bin_dnb_idx else np.array([], dtype=np.int64)
    empty_bin_list = np.concatenate([np.full(len(idx), i, dtype=np.int64) for i, idx in enumerate(empty_bin_dnb_idx)]) if empty_bin_dnb_idx else np.array([], dtype=np.int64)
    if len(empty_dnb_list) > 0:
        C_empty = csr_matrix(
            (np.ones(len(empty_dnb_list), dtype=np.float64), (empty_dnb_list, empty_bin_list)),
            shape=(n_dnbs, n_empty_bins)
        )
        empty_bin_expr_sp = dnb_csc @ C_empty
        empty_bin_expr = np.asarray(empty_bin_expr_sp.todense()) if hasattr(empty_bin_expr_sp, 'todense') else empty_bin_expr_sp.toarray()
    else:
        empty_bin_expr = np.zeros((n_genes, 0), dtype=np.float64)

    # ── 3. Select high-expression genes ────────────────────────
    if n_empty_bins == 0:
        raise RuntimeError("No empty bins available for ambient estimation.")

    total_empty = empty_bin_areas.sum()
    mean_empty = empty_bin_expr.sum(axis=1) / total_empty
    n_select = min(n_high_genes, n_genes)
    gene_indices = np.argsort(mean_empty)[::-1][:n_select]

    if verbose:
        print(f"Selected {len(gene_indices)} high-expression genes")

    # ── 4. Build spatial graph (cells → empty bins) ────────────
    # We need distances from empty bins to cells (for λ estimation)
    # and from cells to cells (for correction).
    # Build a combined coordinate set: [cells, empty_bins]
    all_coords = np.vstack([cell_centroids, empty_bin_coords]) if n_empty_bins > 0 else cell_centroids
    n_total = n_cells + n_empty_bins

    # ── 5. Estimate λ ──────────────────────────────────────────
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
            cell_source[i] = _ewap_source(cell_source[i])

    best_lam = lambda_grid[0]
    best_rss = np.inf

    for lam in lambda_grid:
        # Build graph: distances from empty bins to cells
        W = build_spatial_graph(all_coords, max_radius, lam, distance_metric)
        # Extract empty-bin → cell submatrix
        W_empty_to_cell = W[n_cells:, :n_cells]  # [n_empty_bins × n_cells]

        total_rss = 0.0
        for i, g_idx in enumerate(lambda_gene_indices):
            source = cell_source[i]
            weighted_sum = W_empty_to_cell.dot(source)  # Σ_c w(d) × source_c
            N_gb = empty_bin_areas * weighted_sum
            y_obs = empty_bin_expr[g_idx]
            w = empty_bin_areas.astype(np.float64)

            denom = (w * N_gb ** 2).sum()
            if denom == 0:
                continue
            alpha_g = max((w * y_obs * N_gb).sum() / denom, 0.0)
            residuals = y_obs - alpha_g * N_gb
            total_rss += (w * residuals ** 2).sum()

        if total_rss < best_rss:
            best_rss = total_rss
            best_lam = lam

    if verbose:
        print(f"  → Optimal λ = {best_lam:.1f} μm (RSS = {best_rss:.2f})")

    # ── 6. Estimate α per gene ──────────────────────────────────
    if verbose:
        print("Estimating gene-specific leakage rates α...")

    W = build_spatial_graph(all_coords, max_radius, best_lam, distance_metric)
    W_empty_to_cell = W[n_cells:, :n_cells]

    n_genes_use = len(gene_indices)
    alphas = np.zeros(n_genes_use, dtype=np.float64)
    r2_scores = np.zeros(n_genes_use, dtype=np.float64)
    w_empty = empty_bin_areas.astype(np.float64)

    for i, g_idx in enumerate(gene_indices):
        source = np.divide(cell_expr[g_idx], cell_areas,
                           out=np.zeros(n_cells), where=cell_areas > 0)
        if use_expr_weight:
            source = _ewap_source(source)
        weighted_sum = W_empty_to_cell.dot(source)
        N_gb = empty_bin_areas * weighted_sum
        y_obs = empty_bin_expr[g_idx]

        denom = (w_empty * N_gb ** 2).sum()
        if denom == 0:
            alphas[i] = 0.0
            r2_scores[i] = 0.0
            continue

        alpha_g = max((w_empty * y_obs * N_gb).sum() / denom, 0.0)
        alphas[i] = alpha_g

        ss_res = (w_empty * (y_obs - alpha_g * N_gb) ** 2).sum()
        y_mean = (w_empty * y_obs).sum() / w_empty.sum() if w_empty.sum() > 0 else 0
        ss_tot = (w_empty * (y_obs - y_mean) ** 2).sum()
        if ss_tot > 1e-15:
            r2_scores[i] = 1.0 - ss_res / ss_tot
        elif ss_res <= 1e-15:
            r2_scores[i] = 1.0
        else:
            r2_scores[i] = 0.0

    n_corrected = int((r2_scores >= r2_threshold).sum())
    if verbose:
        print(f"  → {n_corrected}/{n_genes_use} genes pass R² threshold ({r2_threshold})")

    # ── 7. Correct cells ────────────────────────────────────────
    if verbose:
        print("Applying cell-level ambient correction...")

    # Build cell-to-cell distance matrix (excluding self)
    W_cell_to_cell = W[:n_cells, :n_cells]  # [n_cells × n_cells]
    # Zero out self-connections (diagonal)
    W_cell_to_cell = W_cell_to_cell.tolil()
    W_cell_to_cell.setdiag(0)
    W_cell_to_cell = W_cell_to_cell.tocsr()

    corrected_expr = np.zeros((n_genes, n_cells), dtype=np.float64)

    for g_idx in range(n_genes):
        if g_idx not in set(gene_indices):
            corrected_expr[g_idx] = cell_expr[g_idx]
            continue

        pos = list(gene_indices).index(g_idx)
        if r2_scores[pos] < r2_threshold:
            corrected_expr[g_idx] = cell_expr[g_idx]
            continue

        alpha_g = alphas[pos]
        source = np.divide(cell_expr[g_idx], cell_areas,
                           out=np.zeros(n_cells), where=cell_areas > 0)
        if use_expr_weight:
            source = _ewap_source(source)
        # Ambient at each cell from OTHER cells, optionally with self-confidence penalty
        neighbor_contrib = W_cell_to_cell.dot(source)
        if self_confidence_penalty:
            penalty = _self_confidence_weight(source, mode=penalty_mode)
            ambient = alpha_g * cell_areas * neighbor_contrib * penalty
        else:
            ambient = alpha_g * cell_areas * neighbor_contrib
        corrected_expr[g_idx] = np.maximum(cell_expr[g_idx] - ambient, 0.0)

    diagnostics = {
        "lambda_estimated": best_lam,
        "lambda_grid": lambda_grid,
        "n_genes_corrected": n_corrected,
        "n_high_genes_selected": n_genes_use,
        "r2_threshold": r2_threshold,
        "alpha_mean": float(alphas.mean()) if n_genes_use > 0 else 0.0,
        "r2_mean": float(r2_scores.mean()) if n_genes_use > 0 else 0.0,
        "n_cells": n_cells,
        "n_empty_bins": n_empty_bins,
        "empty_bin_areas_mean": float(empty_bin_areas.mean()) if n_empty_bins > 0 else 0,
        "cell_areas_mean": float(cell_areas.mean()),
    }

    return corrected_expr, diagnostics
