"""Cell-level ambient correction.

For each gene that passed the R² gate the predicted ambient inflow into a
cell is the sum of weighted neighbour source strengths scaled by the cell's
area and the gene's leakage rate alpha:

    ambient_i = alpha_g * area_i * Σ_j w(d_ij) * s_j

Optionally each neighbour's contribution is penalised by the neighbour's own
expression (``self_confidence_weight``) so high-expressors are not
over-subtracted.  The corrected value is clamped at zero.

Both the batched CPU (SciPy) and GPU (PyTorch) branches live here.
"""

import numpy as np
from typing import Optional

from .gpu import (
    GPUContext,
    sparse_mm,
    to_cpu,
    to_gpu,
    weighted_gpu_csr,
)
from .spatial import DistanceMetric, distance_graph_to_weights
from .weights import expression_weight, self_confidence_weight


def correct_cells(
    gene_indices: np.ndarray,
    r2_scores: np.ndarray,
    r2_threshold: float,
    n_genes: int,
    n_cells: int,
    cell_expr: np.ndarray,
    cell_areas: np.ndarray,
    alphas: np.ndarray,
    cell_distance_graph,
    lam_weights: float,
    distance_metric: DistanceMetric,
    use_expr_weight: bool,
    self_confidence_penalty: bool,
    penalty_mode: str,
    gpu: Optional[GPUContext],
    storage_dtype,
    reduction_dtype,
    effective_gpu_gene_batch_size: Optional[int],
) -> np.ndarray:
    """Subtract predicted ambient RNA from per-cell expression.

    Args:
        gene_indices: high-expression genes (ranked, descending).
        r2_scores: [n_genes_use] weighted R² per selected gene.
        r2_threshold: minimum R² for a gene to be corrected.
        n_genes: total number of genes in the expression matrix.
        n_cells: number of cells.
        cell_expr: [genes × n_cells] dense per-cell expression.
        cell_areas: [n_cells] DNB count per cell.
        alphas: [n_genes_use] leakage rates aligned with ``gene_indices``.
        cell_distance_graph: cells → cells distance graph: GPU-resident CSR
            when ``gpu`` is set, otherwise the CPU SciPy CSR.
        lam_weights: spatial decay length for the kernel.
        distance_metric: kernel name ('exponential', 'gaussian', 'inverse').
        use_expr_weight: expression-weight the cell sources.
        self_confidence_penalty: apply the self-confidence penalty.
        penalty_mode: penalty functional form (see
            :func:`stambient.weights.self_confidence_weight`).
        gpu: resolved GPU context, or None for CPU.
        storage_dtype: torch dtype for sparse/dense products.
        reduction_dtype: torch dtype for reductions.
        effective_gpu_gene_batch_size: GPU batch size chosen during alpha
            estimation; reused here for consistency.

    Returns:
        corrected_expr: [genes × n_cells] corrected per-cell expression.
    """
    if gpu is not None:
        W_cell = weighted_gpu_csr(
            cell_distance_graph, lam_weights, distance_metric, gpu
        )
    else:
        W_cell = distance_graph_to_weights(
            cell_distance_graph, lam_weights, distance_metric
        )

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
                    corr_sources = np.array(
                        [expression_weight(s) for s in corr_sources]
                    )
                corr_sources_gpu = to_gpu(
                    corr_sources.T, gpu, dtype=storage_dtype
                )
                neighbor_gpu = sparse_mm(
                    W_cell, corr_sources_gpu, gpu
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
                    penalties = self_confidence_weight(
                        corr_sources, mode=penalty_mode
                    )
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
                cell_expr[cg_idx],
                cell_areas[None, :],
                out=np.zeros((len(cg_idx), n_cells), dtype=np.float64),
                where=cell_areas[None, :] > 0,
            )
            if use_expr_weight:
                corr_sources = np.array(
                    [expression_weight(s) for s in corr_sources]
                )
            # Single sparse-dense matrix multiply for all corrected genes
            neighbor_contribs = W_cell.dot(corr_sources.T)
            ambient = (
                alphas[cpos][None, :]
                * cell_areas[:, None]
                * neighbor_contribs
            )
            if self_confidence_penalty:
                penalties = self_confidence_weight(
                    corr_sources, mode=penalty_mode
                )
                ambient *= penalties.T
            corrected_expr[cg_idx] = np.maximum(
                cell_expr[cg_idx] - ambient.T, 0.0
            )

    return corrected_expr
