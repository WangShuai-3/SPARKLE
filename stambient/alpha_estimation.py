"""Alpha estimation: per-gene ambient leakage rates with weighted R².

For each high-expression gene the pipeline solves a one-parameter weighted
least-squares fit of the observed empty-bin expression against the predicted
ambient inflow from neighbouring cells:

    y_bin ≈ alpha_g * N_bin,   N_bin = area_bin * Σ_cells w(d) * s_cell

The closed-form alpha is area-weighted and clamped to non-negative values.
The weighted R² measures how much of the empty-bin expression variance the
spatial predictor explains and gates whether correction is applied.

Both the batched CPU (SciPy) and GPU (PyTorch) branches live here.
"""

import numpy as np
from typing import Optional, Tuple

from .gpu import (
    GPUContext,
    choose_gpu_gene_batch_size,
    sparse_mm,
    to_cpu,
    to_gpu,
)
from .weights import expression_weight


def estimate_alphas(
    gene_indices: np.ndarray,
    cell_expr: np.ndarray,
    cell_areas: np.ndarray,
    empty_bin_expr: np.ndarray,
    empty_bin_areas: np.ndarray,
    W_empty,
    use_expr_weight: bool,
    gpu: Optional[GPUContext],
    storage_dtype,
    reduction_dtype,
    gpu_gene_batch_size: Optional[int],
) -> Tuple[np.ndarray, np.ndarray, Optional[int]]:
    """Estimate gene-specific leakage rates alpha and their R² scores.

    Args:
        gene_indices: high-expression genes (ranked, descending).
        cell_expr: [genes × n_cells] dense per-cell expression.
        cell_areas: [n_cells] DNB count per cell.
        empty_bin_expr: [genes × n_empty_bins] dense bin expression.
        empty_bin_areas: [n_empty_bins] DNB count per bin.
        W_empty: empty→cell weight matrix evaluated at the winning lambda
            (GPU tensor or CPU SciPy CSR).
        use_expr_weight: expression-weight the cell sources.
        gpu: resolved GPU context, or None for CPU.
        storage_dtype: torch dtype for sparse/dense products.
        reduction_dtype: torch dtype for reductions.
        gpu_gene_batch_size: genes per GPU batch; None selects adaptively.

    Returns:
        (alphas, r2_scores, effective_gpu_gene_batch_size). The batch size is
        returned because the correction stage reuses it for its own batching.
    """
    n_genes_use = len(gene_indices)
    n_cells = cell_expr.shape[1]
    n_empty_bins = empty_bin_expr.shape[1]
    alphas = np.zeros(n_genes_use, dtype=np.float64)
    r2_scores = np.zeros(n_genes_use, dtype=np.float64)
    w_empty = empty_bin_areas.astype(np.float64)
    effective_gpu_gene_batch_size: Optional[int] = None

    eps = 1e-15
    if gpu is not None:
        w_empty_gpu = to_gpu(w_empty[:, None], gpu, dtype=reduction_dtype)
        areas_gpu = to_gpu(
            empty_bin_areas[:, None], gpu, dtype=reduction_dtype
        )
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
                sources_batch = np.array(
                    [expression_weight(s) for s in sources_batch]
                )
            y_obs_batch = empty_bin_expr[batch_gene_indices].T

            sources_gpu = to_gpu(sources_batch.T, gpu, dtype=storage_dtype)
            y_obs_gpu = to_gpu(y_obs_batch, gpu, dtype=reduction_dtype)
            weighted_sums_gpu = sparse_mm(
                W_empty, sources_gpu, gpu
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
                w_empty_gpu
                * (y_obs_gpu - alphas_gpu.unsqueeze(0) * N_gb_gpu).square()
            ).sum(dim=0)
            if w_empty.sum() > 0:
                y_mean_gpu = (
                    w_empty_gpu * y_obs_gpu
                ).sum(dim=0) / w_empty.sum()
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
            cell_expr[gene_indices],
            cell_areas[None, :],
            out=np.zeros((n_genes_use, n_cells), dtype=np.float64),
            where=cell_areas[None, :] > 0,
        )
        if use_expr_weight:
            sources = np.array([expression_weight(s) for s in sources])
        y_obs_all = empty_bin_expr[gene_indices].T

        # Batched weighted sums and OLS for all genes
        weighted_sums = W_empty.dot(sources.T)
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

        ss_res = (
            w_empty[:, None] * (y_obs_all - alphas[None, :] * N_gb_all) ** 2
        ).sum(axis=0)
        y_mean = (
            (w_empty[:, None] * y_obs_all).sum(axis=0) / w_empty.sum()
            if w_empty.sum() > 0
            else 0.0
        )
        ss_tot = (
            w_empty[:, None] * (y_obs_all - y_mean[None, :]) ** 2
        ).sum(axis=0)
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

    return alphas, r2_scores, effective_gpu_gene_batch_size
