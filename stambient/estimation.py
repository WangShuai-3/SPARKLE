"""Parameter estimation: λ grid search, α weighted OLS, β local density."""

import numpy as np
from scipy.sparse import csr_matrix
from typing import List, Tuple, Optional

from .spatial import (
    build_spatial_graph,
    compute_distance_weights,
    compute_local_density,
    DistanceMetric,
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


def _compute_source_strength(
    y_cell_row: np.ndarray,
    n_cell: np.ndarray,
    cell_mask: np.ndarray,
    use_expr_weight: bool = False,
) -> np.ndarray:
    """Compute per-DNB source strength, optionally expression-weighted."""
    source = np.divide(
        y_cell_row, n_cell,
        out=np.zeros_like(y_cell_row, dtype=np.float64),
        where=cell_mask
    )
    if use_expr_weight:
        source = _expression_weight(source)
    return source


def select_high_expression_genes(
    Y_empty: csr_matrix,
    n_empty: np.ndarray,
    n_high: Optional[int] = None,
    empty_bin_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Select top high-expression genes based on per-DNB empty expression.

    Per-DNB empty expression rate = total empty expression / total empty DNBs.

    Args:
        Y_empty: [genes × bins] empty DNB expression.
        n_empty: [bins] empty DNB counts per bin.
        n_high: Number of top genes to select. If None, use all genes.
        empty_bin_mask: If provided, only consider bins where mask is True.

    Returns:
        Array of gene indices, sorted descending by mean empty expression.
    """
    if empty_bin_mask is None:
        empty_bin_mask = n_empty > 0

    total_empty_dnb = n_empty[empty_bin_mask].sum()
    if total_empty_dnb == 0:
        return np.array([], dtype=np.int64)

    # Sum empty expression for each gene across all valid bins
    if hasattr(Y_empty, 'toarray'):
        empty_sum = Y_empty[:, empty_bin_mask].sum(axis=1)
        if hasattr(empty_sum, 'A1'):
            empty_sum = empty_sum.A1
        else:
            empty_sum = np.asarray(empty_sum).ravel()
    else:
        empty_sum = Y_empty[:, empty_bin_mask].sum(axis=1).ravel()

    mean_empty = empty_sum / total_empty_dnb

    # Get top N (or all genes if n_high is None)
    if n_high is None:
        n_select = len(mean_empty)
    else:
        n_select = min(n_high, len(mean_empty))
    top_genes = np.argsort(mean_empty)[::-1][:n_select]
    return top_genes


def estimate_lambda_grid_search(
    Y_empty: csr_matrix,
    Y_cell: csr_matrix,
    n_empty: np.ndarray,
    n_cell: np.ndarray,
    n_total: np.ndarray,
    bin_coords: np.ndarray,
    gene_indices: np.ndarray,
    empty_bin_mask: np.ndarray,
    lambda_grid: List[float],
    max_radius: float,
    metric: DistanceMetric = "exponential",
    use_expr_weight: bool = False,
) -> Tuple[float, float, List[float], np.ndarray]:
    """Estimate global distance decay λ by grid search.

    For each λ in the grid, compute weighted RSS over the top genes
    using only empty-bin observations. Choose λ that minimizes total RSS.

    Args:
        ...
        use_expr_weight: If True, use expression-weighted source
            (quadratic: s²/s_max) to suppress non-expressing cells.
    """
    best_lambda = lambda_grid[0]
    best_rss = np.inf
    rss_per_lambda = []

    n_genes_use = len(gene_indices)
    cell_mask = n_cell > 0
    source_strengths = np.zeros((n_genes_use, Y_cell.shape[1]), dtype=np.float64)

    # Precompute source strengths: Y_cell / n_cell (per-DNB expression rate)
    for i, g_idx in enumerate(gene_indices):
        if hasattr(Y_cell, 'toarray'):
            y_row = Y_cell[g_idx].toarray().ravel()
        else:
            y_row = np.asarray(Y_cell[g_idx].todense()).ravel()
        source_strengths[i] = _compute_source_strength(
            y_row, n_cell, cell_mask, use_expr_weight
        )

    all_alphas_for_best = None

    for lam in lambda_grid:
        # Build spatial weight matrix
        W = build_spatial_graph(bin_coords, max_radius, lam, metric)

        total_rss = 0.0
        alphas_this = np.zeros(n_genes_use, dtype=np.float64)

        for i, g_idx in enumerate(gene_indices):
            if hasattr(Y_empty, 'toarray'):
                y_empty_row = Y_empty[g_idx].toarray().ravel()
            else:
                y_empty_row = np.asarray(Y_empty[g_idx].todense()).ravel()

            # Compute N_gb for empty bins: n_empty[b] * Σ w(d) * source_strength
            source = source_strengths[i]
            weighted_sum = W.dot(source)  # Σ_{b'} w(d) * Y_cell / n_cell
            N_gb = n_empty * weighted_sum

            # Weighted OLS through origin on empty observations
            w = n_empty[empty_bin_mask].astype(np.float64)
            y_obs = y_empty_row[empty_bin_mask]
            n_obs = N_gb[empty_bin_mask]

            if w.sum() == 0 or (w * n_obs ** 2).sum() == 0:
                alpha_g = 0.0
            else:
                alpha_g = max((w * y_obs * n_obs).sum() / (w * n_obs ** 2).sum(), 0.0)

            alphas_this[i] = alpha_g

            # Weighted RSS
            residuals = y_obs - alpha_g * n_obs
            rss_g = (w * residuals ** 2).sum()
            total_rss += rss_g

        rss_per_lambda.append(total_rss)

        if total_rss < best_rss:
            best_rss = total_rss
            best_lambda = lam
            all_alphas_for_best = alphas_this.copy()

    return best_lambda, best_rss, rss_per_lambda, all_alphas_for_best


def estimate_alpha_per_gene(
    Y_empty: csr_matrix,
    Y_cell: csr_matrix,
    n_empty: np.ndarray,
    n_cell: np.ndarray,
    bin_coords: np.ndarray,
    gene_indices: np.ndarray,
    empty_bin_mask: np.ndarray,
    lam: float,
    max_radius: float,
    metric: DistanceMetric = "exponential",
    use_expr_weight: bool = False,
    per_gene_lambda: bool = False,
    lambda_grid: Optional[List[float]] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Estimate gene-specific leakage rate α_g via weighted OLS.

    Uses only empty-bin observations. Weight = n_empty[b] (more DNBs = more reliable).

    Args:
        ...
        use_expr_weight: If True, expression-weight the source (s²/s_max).
        per_gene_lambda: If True, search λ_grid per gene for optimal λ_g.
        lambda_grid: λ candidates for per-gene search (required if per_gene_lambda=True).

    Returns:
        alpha: [n_genes] estimated α per gene.
        r2: [n_genes] weighted R² per gene.
        lambdas: [n_genes] per-gene λ (None if per_gene_lambda=False).
    """
    n_genes_use = len(gene_indices)
    n_bins = Y_cell.shape[1]
    cell_mask = n_cell > 0
    w_empty = n_empty[empty_bin_mask].astype(np.float64)

    # Pre-load Y_empty dense
    if hasattr(Y_empty, 'toarray'):
        y_empty_all = Y_empty.toarray()
    else:
        y_empty_all = np.asarray(Y_empty.todense())

    y_empty_mean = np.zeros(n_genes_use, dtype=np.float64)
    for i, g_idx in enumerate(gene_indices):
        y_row = y_empty_all[g_idx]
        if w_empty.sum() > 0:
            y_empty_mean[i] = (w_empty * y_row[empty_bin_mask]).sum() / w_empty.sum()

    alphas = np.zeros(n_genes_use, dtype=np.float64)
    r2_scores = np.zeros(n_genes_use, dtype=np.float64)
    lambdas = np.full(n_genes_use, lam, dtype=np.float64)

    if per_gene_lambda and lambda_grid is not None and len(lambda_grid) > 0:
        # ── Per-gene λ grid search ──────────────────────────
        # Pre-load Y_cell for all genes
        if hasattr(Y_cell, 'toarray'):
            y_cell_all = Y_cell.toarray()
        else:
            y_cell_all = np.asarray(Y_cell.todense())

        # Cache W matrices per λ
        W_cache = {}
        source_cache = {}

        for i, g_idx in enumerate(gene_indices):
            y_cell_row = y_cell_all[g_idx]
            y_empty_row = y_empty_all[g_idx]

            source_raw = _compute_source_strength(
                y_cell_row, n_cell, cell_mask, use_expr_weight
            )

            best_rss_g = np.inf
            best_lam_g = lam
            best_alpha_g = 0.0

            for lam_cand in lambda_grid:
                if lam_cand not in W_cache:
                    W_cache[lam_cand] = build_spatial_graph(
                        bin_coords, max_radius, lam_cand, metric
                    )
                W = W_cache[lam_cand]

                weighted_sum = W.dot(source_raw)
                N_gb = n_empty * weighted_sum

                y_obs = y_empty_row[empty_bin_mask]
                n_obs = N_gb[empty_bin_mask]

                denom = (w_empty * n_obs ** 2).sum()
                if denom == 0:
                    continue
                alpha_cand = max((w_empty * y_obs * n_obs).sum() / denom, 0.0)
                residuals = y_obs - alpha_cand * n_obs
                rss_g = (w_empty * residuals ** 2).sum()

                if rss_g < best_rss_g:
                    best_rss_g = rss_g
                    best_lam_g = lam_cand
                    best_alpha_g = alpha_cand

            lambdas[i] = best_lam_g
            alphas[i] = best_alpha_g

            # R² with best λ
            if best_lam_g not in W_cache:
                W_cache[best_lam_g] = build_spatial_graph(
                    bin_coords, max_radius, best_lam_g, metric
                )
            W_best = W_cache[best_lam_g]
            weighted_sum = W_best.dot(source_raw)
            N_gb = n_empty * weighted_sum
            y_obs = y_empty_row[empty_bin_mask]
            n_obs = N_gb[empty_bin_mask]
            ss_res = (w_empty * (y_obs - best_alpha_g * n_obs) ** 2).sum()
            ss_tot = (w_empty * (y_obs - y_empty_mean[i]) ** 2).sum()
            eps = 1e-15
            if ss_tot > eps:
                r2_scores[i] = 1.0 - ss_res / ss_tot
            elif ss_res > eps:
                r2_scores[i] = 0.0
            else:
                r2_scores[i] = 1.0

        return alphas, r2_scores, lambdas

    # ── Single global λ (standard path) ────────────────────
    W = build_spatial_graph(bin_coords, max_radius, lam, metric)

    for i, g_idx in enumerate(gene_indices):
        y_row = y_empty_all[g_idx]

        if hasattr(Y_cell, 'toarray'):
            y_cell_row = Y_cell[g_idx].toarray().ravel()
        else:
            y_cell_row = np.asarray(Y_cell[g_idx].todense()).ravel()

        source = _compute_source_strength(
            y_cell_row, n_cell, cell_mask, use_expr_weight
        )

        weighted_sum = W.dot(source)
        N_gb = n_empty * weighted_sum

        y_obs = y_row[empty_bin_mask]
        n_obs = N_gb[empty_bin_mask]

        denom = (w_empty * n_obs ** 2).sum()
        if denom == 0:
            alphas[i] = 0.0
            r2_scores[i] = 0.0
        else:
            alpha_g = max((w_empty * y_obs * n_obs).sum() / denom, 0.0)
            alphas[i] = alpha_g

            ss_res = (w_empty * (y_obs - alpha_g * n_obs) ** 2).sum()
            ss_tot = (w_empty * (y_obs - y_empty_mean[i]) ** 2).sum()
            eps = 1e-15
            if ss_tot > eps:
                r2_scores[i] = 1.0 - ss_res / ss_tot
            elif ss_res > eps:
                r2_scores[i] = 0.0
            else:
                r2_scores[i] = 1.0

    return alphas, r2_scores, None
