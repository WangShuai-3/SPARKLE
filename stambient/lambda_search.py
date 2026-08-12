"""Lambda selection: grid search for the spatial decay length.

The kernel weight w(d) = f(d / lambda) turns the empty-bin → cell distance
graph into an ambient predictor. For each candidate lambda the pipeline fits
a closed-form per-gene alpha by area-weighted least squares and scores the
fit by the area-weighted residual sum of squares summed across genes.

The 'inverse' kernel 1/(d + eps) has no lambda dependence, so the grid search
is skipped and ``lambda_estimated`` stays None.  This module owns both the
CPU (SciPy) and GPU (PyTorch) branches of the search.
"""

import numpy as np
from typing import List, Optional, Tuple

from .gpu import (
    GPUContext,
    sparse_mm,
    to_gpu,
    weighted_gpu_csr,
)
from .spatial import DistanceMetric, distance_graph_to_weights
from .weights import expression_weight


def estimate_lambda(
    lambda_grid: List[float],
    distance_metric: DistanceMetric,
    gene_indices: np.ndarray,
    n_lambda_genes: int,
    cell_expr: np.ndarray,
    cell_areas: np.ndarray,
    empty_bin_expr: np.ndarray,
    empty_bin_areas: np.ndarray,
    empty_to_cell_distances,
    empty_distance_gpu,
    use_expr_weight: bool,
    gpu: Optional[GPUContext],
    storage_dtype,
    reduction_dtype,
    verbose: bool,
) -> Tuple[Optional[float], float, List[float], object]:
    """Grid-search lambda and return the winning empty→cell weight matrix.

    Args:
        lambda_grid: candidate spatial-decay lengths in micrometres.
        distance_metric: kernel name ('exponential', 'gaussian', 'inverse').
        gene_indices: high-expression genes (ranked, descending).
        n_lambda_genes: number of top genes to use for the search.
        cell_expr: [genes × n_cells] dense per-cell expression.
        cell_areas: [n_cells] DNB count per cell.
        empty_bin_expr: [genes × n_empty_bins] dense bin expression.
        empty_bin_areas: [n_empty_bins] DNB count per bin.
        empty_to_cell_distances: CPU CSR of empty-bin → cell distances.
        empty_distance_gpu: GPU CSR of the same graph, or None on CPU.
        use_expr_weight: expression-weight the cell sources.
        gpu: resolved GPU context, or None for CPU.
        storage_dtype: torch dtype for sparse/dense products.
        reduction_dtype: torch dtype for reductions.
        verbose: print progress messages.

    Returns:
        (best_lam, best_rss, lambda_search_rss, W_empty) where W_empty is the
        final empty→cell weight matrix evaluated at the winning lambda (GPU
        tensor when ``gpu`` is set, otherwise a CPU SciPy CSR).
    """
    if distance_metric == "inverse":
        # The inverse kernel 1/(d+eps) has no lambda dependence: every
        # candidate would produce identical weights, so a grid search is
        # meaningless.
        best_lam = None
        best_rss = np.nan
        lambda_search_rss: List[float] = []
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
        n_cells = cell_expr.shape[1]
        cell_source = np.zeros((n_lambda_use, n_cells), dtype=np.float64)
        for i, g_idx in enumerate(lambda_gene_indices):
            cell_source[i] = np.divide(
                cell_expr[g_idx],
                cell_areas,
                out=np.zeros(n_cells),
                where=cell_areas > 0,
            )
            if use_expr_weight:
                cell_source[i] = expression_weight(cell_source[i])

        best_lam = lambda_grid[0]
        best_rss = np.inf
        lambda_search_rss = []
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
                    denom_gpu > 0,
                    numer_gpu / denom_gpu,
                    gpu.torch.zeros_like(denom_gpu),
                ).clamp_min(0.0)
                residuals_gpu = (
                    y_obs_lambda_gpu - alpha_gpu.unsqueeze(0) * N_gb_gpu
                )
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

            lambda_search_rss.append(float(total_rss))

            if total_rss < best_rss:
                best_rss = total_rss
                best_lam = lam

        if verbose:
            print(f"  → Optimal λ = {best_lam:.1f} μm (RSS = {best_rss:.2f})")

    # Lambda is unused by the inverse kernel; any value yields the same weights.
    lam_weights = best_lam if best_lam is not None else 1.0
    if gpu is not None:
        W_empty = weighted_gpu_csr(
            empty_distance_gpu, lam_weights, distance_metric, gpu
        )
    else:
        W_empty = distance_graph_to_weights(
            empty_to_cell_distances, lam_weights, distance_metric
        )
    return best_lam, best_rss, lambda_search_rss, W_empty
