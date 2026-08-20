"""Spatial cross-validated evidence for local leakage (2.x Phase 3).

Replaces the hard weighted-R^2 gate with out-of-sample evidence: for every
panel gene we ask whether the local-leakage model

    M1:  mu_b = A_b * (beta + rho * S_b)

predicts *held-out* background bins better than the diffuse-only null

    M0:  mu_b = A_b * beta0         (beta0 = sum(y) / sum(A), closed form)

Background bins are split into spatial blocks (a 2x2 checkerboard of large
tiles) so that train/validation bins are spatially separated; plain random
splits would leak through spatial autocorrelation.  The per-gene evidence
score is the normalised held-out deviance gain

    E_g = (D0_g - D1_g) / (D0_g + eps)

which is > 0 exactly when the spatial component improves out-of-sample
prediction.  Correction gates/weights are derived from E_g by the pipeline
(hard threshold or continuous saturation weight), never from the in-sample
fit.  All heavy lifting reuses the count-model projected Newton solver, so
CPU and GPU paths agree numerically with ``estimate_leakage_poisson``.
"""

from typing import Dict, Optional

import numpy as np

from .count_model import _fit_poisson_batch_cpu, _fit_poisson_batch_gpu
from .gpu import (
    GPUContext,
    choose_gpu_gene_batch_size,
    sparse_distance_graph_to_gpu_csr,
    sparse_mm,
    to_cpu,
    to_gpu,
    weighted_gpu_csr,
)
from .spatial import distance_graph_to_weights
from .weights import expression_weight

_MU_FLOOR = 1e-12


def spatial_block_fold_ids(
    bin_coords: np.ndarray, block_size: float, n_splits: int = 2
) -> np.ndarray:
    """Assign each bin to one of ``n_splits**2`` checkerboard folds.

    Bins within a block tile are contiguous in space; holdout folds are
    therefore spatially separated from their training complement.
    """
    xy = (bin_coords - bin_coords.min(axis=0)) / float(block_size)
    bx = np.floor(xy[:, 0]).astype(np.int64) % n_splits
    by = np.floor(xy[:, 1]).astype(np.int64) % n_splits
    return bx + n_splits * by


def poisson_deviance(y, mu):
    """Poisson deviance D(y, mu) = 2*sum[y log(y/mu) - (y - mu)] (>= 0).

    ``y``/``mu``: [n_bins x batch]; returns [batch].
    """
    y = np.asarray(y, dtype=np.float64)
    mu = np.maximum(np.asarray(mu, dtype=np.float64), _MU_FLOOR)
    ratio_term = np.where(y > 0, y * np.log(np.maximum(y / mu, _MU_FLOOR)), 0.0)
    return 2.0 * (ratio_term - (y - mu)).sum(axis=0)


def _prepare_sources(cell_expr, cell_areas, use_expr_weight):
    sources = np.divide(
        cell_expr,
        cell_areas[None, :],
        out=np.zeros_like(cell_expr, dtype=np.float64),
        where=cell_areas[None, :] > 0,
    )
    if use_expr_weight:
        sources = np.array([expression_weight(s) for s in sources])
    return sources


def spatial_cv_evidence(
    *,
    gene_indices: np.ndarray,
    cell_expr: np.ndarray,
    cell_areas: np.ndarray,
    empty_bin_expr: np.ndarray,
    empty_bin_areas: np.ndarray,
    empty_bin_coords: np.ndarray,
    empty_to_cell_distances,
    lam_weights: float,
    distance_metric: str = "exponential",
    fit_diffuse: bool = True,
    use_expr_weight: bool = False,
    n_splits: int = 2,
    block_size: float = 50.0,
    gpu: Optional[GPUContext] = None,
    storage_dtype=None,
    reduction_dtype=None,
    gpu_gene_batch_size: Optional[int] = None,
    max_newton_iter: int = 25,
    newton_tol: float = 1e-6,
) -> Dict[str, np.ndarray]:
    """Per-gene spatial-CV evidence for local leakage.

    Args mirror :func:`stambient.count_model.estimate_leakage_poisson`, plus:

        empty_bin_coords: [n_bins x 2] coordinates (for the block split).
        empty_to_cell_distances: [n_bins x n_cells] CSR of distances, used
            to build fold-specific W matrices at ``lam_weights``.
        lam_weights: winning lambda of the shared decay kernel.
        n_splits: checkerboard splits per axis (n_splits**2 folds).
        block_size: edge length of one CV block tile (spatial units).

    Returns:
        dict with [n_genes_use] arrays:
        ``deviance_null`` / ``deviance_model``: summed held-out Poisson
            deviances of M0 / M1 over all folds.
        ``delta_deviance``: D0 - D1 (positive = local model generalises).
        ``evidence``: delta / (D0 + eps), the normalised evidence score E_g.
    """
    gene_indices = np.asarray(gene_indices, dtype=int)
    n_genes_use = len(gene_indices)
    n_bins = len(empty_bin_areas)
    folds = spatial_block_fold_ids(empty_bin_coords, block_size, n_splits)
    n_folds = n_splits * n_splits

    sources = _prepare_sources(
        cell_expr[gene_indices], cell_areas, use_expr_weight
    )

    d0 = np.zeros(n_genes_use, dtype=np.float64)
    d1 = np.zeros(n_genes_use, dtype=np.float64)

    if gpu is None:
        W_full = distance_graph_to_weights(
            empty_to_cell_distances, lam_weights, distance_metric
        ).tocsr()
        S_full = W_full.dot(sources.T)  # [n_bins x n_genes_use]
        y_all = empty_bin_expr[gene_indices].T
        for fold in range(n_folds):
            tr = folds != fold
            va = folds == fold
            W_tr = distance_graph_to_weights(
                empty_to_cell_distances[tr], lam_weights, distance_metric
            ).tocsr()
            S_tr = W_tr.dot(sources.T)
            y_tr = y_all[tr]
            A_tr = empty_bin_areas[tr]
            beta0 = y_tr.sum(axis=0) / max(A_tr.sum(), _MU_FLOOR)
            if fit_diffuse:
                betas, rhos, ll_full, _ = _fit_poisson_batch_cpu(
                    y_tr, A_tr, S_tr, True, max_newton_iter, newton_tol
                )
                # Nested-model safeguard, as in estimate_leakage_poisson.
                _, rhos_local, ll_local, _ = _fit_poisson_batch_cpu(
                    y_tr, A_tr, S_tr, False, max_newton_iter, newton_tol
                )
                better = ll_local > ll_full
                betas = np.where(better, 0.0, betas)
                rhos = np.where(better, rhos_local, rhos)
            else:
                betas = np.zeros(n_genes_use, dtype=np.float64)
                _, rhos, _, _ = _fit_poisson_batch_cpu(
                    y_tr, A_tr, S_tr, False, max_newton_iter, newton_tol
                )
            y_va = y_all[va]
            A_va = empty_bin_areas[va][:, None]
            S_va = S_full[va]
            d0 += poisson_deviance(y_va, A_va * beta0[None, :])
            d1 += poisson_deviance(
                y_va, A_va * (betas[None, :] + rhos[None, :] * S_va)
            )
    else:
        torch = gpu.torch
        W_full_gpu = weighted_gpu_csr(
            sparse_distance_graph_to_gpu_csr(
                empty_to_cell_distances, gpu, dtype=storage_dtype
            ),
            lam_weights,
            distance_metric,
            gpu,
        )
        if gpu_gene_batch_size is None:
            gpu_gene_batch_size = choose_gpu_gene_batch_size(
                gpu,
                n_rows=n_bins,
                n_cells=sources.shape[1],
                dtype=storage_dtype,
                max_batch_size=n_genes_use,
            )
        batch = max(1, int(gpu_gene_batch_size))
        for start in range(0, n_genes_use, batch):
            stop = min(start + batch, n_genes_use)
            S_full_b = sparse_mm(
                W_full_gpu,
                to_gpu(sources[start:stop].T, gpu, dtype=storage_dtype),
                gpu,
            ).to(dtype=reduction_dtype)
            y_b = to_gpu(
                empty_bin_expr[gene_indices[start:stop]].T,
                gpu,
                dtype=reduction_dtype,
            )
            for fold in range(n_folds):
                va_rows = np.flatnonzero(folds == fold)
                tr_rows = np.flatnonzero(folds != fold)
                y_tr = y_b[tr_rows]
                A_tr = to_gpu(
                    empty_bin_areas[tr_rows][:, None].astype(np.float64),
                    gpu, dtype=reduction_dtype,
                )
                S_tr = S_full_b[tr_rows]
                beta0 = y_tr.sum(dim=0) / A_tr.sum().clamp_min(_MU_FLOOR)
                if fit_diffuse:
                    betas, rhos, ll_full, _ = _fit_poisson_batch_gpu(
                        y_tr, A_tr, S_tr, gpu, True,
                        max_newton_iter, newton_tol,
                    )
                    _, rhos_local, ll_local, _ = _fit_poisson_batch_gpu(
                        y_tr, A_tr, S_tr, gpu, False,
                        max_newton_iter, newton_tol,
                    )
                    better = ll_local > ll_full
                    betas = torch.where(
                        better, torch.zeros_like(betas), betas
                    )
                    rhos = torch.where(better, rhos_local, rhos)
                else:
                    betas = torch.zeros_like(beta0)
                    _, rhos, _, _ = _fit_poisson_batch_gpu(
                        y_tr, A_tr, S_tr, gpu, False,
                        max_newton_iter, newton_tol,
                    )
                A_va = to_gpu(
                    empty_bin_areas[va_rows][:, None].astype(np.float64),
                    gpu, dtype=reduction_dtype,
                )
                y_va = y_b[va_rows]
                S_va = S_full_b[va_rows]
                mu0 = (A_va * beta0.unsqueeze(0)).clamp_min(_MU_FLOOR)
                mu1 = (
                    A_va * (betas.unsqueeze(0) + rhos.unsqueeze(0) * S_va)
                ).clamp_min(_MU_FLOOR)
                dev0 = 2.0 * (
                    torch.where(
                        y_va > 0, y_va * torch.log(y_va / mu0),
                        torch.zeros_like(y_va),
                    ) - (y_va - mu0)
                ).sum(dim=0)
                dev1 = 2.0 * (
                    torch.where(
                        y_va > 0, y_va * torch.log(y_va / mu1),
                        torch.zeros_like(y_va),
                    ) - (y_va - mu1)
                ).sum(dim=0)
                d0[start:stop] += to_cpu(dev0, gpu)
                d1[start:stop] += to_cpu(dev1, gpu)

    delta = d0 - d1
    evidence = delta / (d0 + 1e-12)
    return {
        "deviance_null": d0,
        "deviance_model": d1,
        "delta_deviance": delta,
        "evidence": evidence,
        "folds": folds,
    }
