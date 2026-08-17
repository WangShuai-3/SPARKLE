"""Poisson count model for background leakage (SPARKLE 2.x Phase 2).

SPARKLE 1.x/2.0-alpha fits the gene-specific leakage rate α by area-weighted
least squares on background bins.  Background bin counts are sequencing
counts, so their variance scales with the mean, not a constant: the Poisson
likelihood is the statistically appropriate observation model.

Generative background model (per gene g, bin b)::

    Y_b ~ Poisson(μ_b),   μ_b = A_b * (β + ρ * S_b)

* ``A_b``: effective area of the background bin;
* ``β ≥ 0``: non-local / diffuse ambient component;
* ``ρ ≥ 0``: local leakage strength (the count-model successor of α);
* ``S_b``: local source signal, the empty→cell kernel-weighted neighbour
  source strength (identical to the regressor used by the weighted OLS).

The additive non-negative mean keeps the physical decomposition
``background = diffuse RNA + local leakage`` explicit (a log-link GLM would
lose it).  Per gene the model has only the two non-negative parameters
(β, ρ); the negative Poisson log-likelihood is convex in them, so a
backtracking projected Newton solves it in a handful of iterations.  All
genes of a batch are optimised simultaneously with vectorised 2×2 Newton
systems — no per-gene scipy optimizer calls.

Both CPU (NumPy) and GPU (PyTorch) branches live here, mirroring
:mod:`stambient.alpha_estimation`.
"""

from typing import Dict, Optional, Tuple

import numpy as np

from .gpu import (
    GPUContext,
    choose_gpu_gene_batch_size,
    sparse_mm,
    to_cpu,
    to_gpu,
)
from .weights import expression_weight

_MU_FLOOR = 1e-12
_HESS_RIDGE = 1e-12


def _poisson_nll(y, mu):
    """Negative Poisson log-likelihood up to the constant log(y!) term."""
    return (mu - y * np.log(np.maximum(mu, _MU_FLOOR))).sum(axis=0)


def _fit_poisson_batch_cpu(y, A, S, fit_diffuse, max_iter, tol):
    """Projected Newton for one batch of genes (NumPy).

    ``y``, ``S``: [n_bins × batch]; ``A``: [n_bins] (or [n_bins × 1]).
    Returns (betas, rhos, loglik, null_loglik, n_iter) as [batch] arrays.
    """
    A = np.asarray(A, dtype=np.float64).reshape(-1, 1)
    y = np.asarray(y, dtype=np.float64)
    S = np.asarray(S, dtype=np.float64)
    batch = y.shape[1]

    total_y = y.sum(axis=0)
    has_counts = total_y > 0
    has_signal = (S * A).sum(axis=0) > 0

    # Init: null-model diffuse rate; zero local component.
    beta = np.divide(
        total_y,
        A.sum(),
        out=np.zeros(batch, dtype=np.float64),
        where=np.full(batch, A.sum() > 0),
    )
    if not fit_diffuse:
        beta = np.zeros(batch, dtype=np.float64)
    # Closed-form weighted-OLS init for rho: without it a zero start puts
    # mu at the floor and Newton steps crawl at the mu scale.
    AS = A * S
    ols_denom = (A * AS**2).sum(axis=0)
    rho = np.divide(
        (A * AS * y).sum(axis=0),
        ols_denom,
        out=np.zeros(batch, dtype=np.float64),
        where=ols_denom > 0,
    )
    rho = np.maximum(rho, 0.0)

    # Unidentifiable local component: no spatial signal -> rho stays 0.
    free = has_counts & (has_signal if fit_diffuse else (S > 0).any(axis=0))
    nll = np.full(batch, np.inf)

    for _ in range(max_iter):
        mu = A * (beta[None, :] + rho[None, :] * S)
        np.maximum(mu, _MU_FLOOR, out=mu)
        r = 1.0 - y / mu  # [bins × batch]
        ymu2 = y / mu**2
        g_b = (A * r).sum(axis=0)
        g_r = (A * S * r).sum(axis=0)
        h_bb = (A**2 * ymu2).sum(axis=0)
        h_br = (A**2 * S * ymu2).sum(axis=0)
        h_rr = (A**2 * S**2 * ymu2).sum(axis=0)
        nll = _poisson_nll(y, mu)

        # KKT-aware stopping: at a lower bound, a gradient component that
        # points further out of the feasible set is already satisfied.
        g_b_eff = np.where(beta > 0.0, g_b, np.minimum(g_b, 0.0))
        g_r_eff = np.where(rho > 0.0, g_r, np.minimum(g_r, 0.0))
        grad_norm = np.maximum(np.abs(g_b_eff), np.abs(g_r_eff))
        done = grad_norm < tol * (total_y + 1.0)
        active = free & ~done
        if not active.any():
            break

        det = h_bb * h_rr - h_br**2
        use_newton = active & (det > _HESS_RIDGE * h_bb * (h_rr + _HESS_RIDGE))
        d_b = np.zeros(batch, dtype=np.float64)
        d_r = np.zeros(batch, dtype=np.float64)
        idx = use_newton
        d_b[idx] = (h_rr[idx] * g_b[idx] - h_br[idx] * g_r[idx]) / det[idx]
        d_r[idx] = (-h_br[idx] * g_b[idx] + h_bb[idx] * g_r[idx]) / det[idx]
        # Singular Hessian (e.g. rho unidentifiable): gradient descent fallback.
        idx = active & ~use_newton
        scale = np.maximum(h_bb[idx] + h_rr[idx], _HESS_RIDGE)
        d_b[idx] = g_b[idx] / scale
        d_r[idx] = g_r[idx] / scale
        if not fit_diffuse:
            d_b[:] = 0.0
        d_r[~has_signal] = 0.0
        # Active set: a parameter at its lower bound whose unconstrained
        # step (x - step*d) would leave the non-negative orthant is dropped
        # from the system; the free coordinate is re-solved on the reduced
        # block.  Without this, boundary genes stall: the clipped joint
        # direction is not a descent direction for the constrained problem.
        bind_b = active & (beta <= 0.0) & (d_b > 0.0)
        bind_r = active & (rho <= 0.0) & (d_r > 0.0)
        only_r = bind_b & ~bind_r
        only_b = bind_r & ~bind_b
        d_b = np.where(bind_b, 0.0, d_b)
        d_r = np.where(bind_r, 0.0, d_r)
        d_r[only_r] = g_r[only_r] / np.maximum(h_rr[only_r], _HESS_RIDGE)
        if fit_diffuse:
            d_b[only_b] = g_b[only_b] / np.maximum(h_bb[only_b], _HESS_RIDGE)

        # Backtracking line search with non-negativity projection.
        step = np.ones(batch, dtype=np.float64)
        for _ls in range(30):
            new_beta = np.maximum(beta - step * d_b, 0.0)
            new_rho = np.maximum(rho - step * d_r, 0.0)
            mu_new = A * (new_beta[None, :] + new_rho[None, :] * S)
            np.maximum(mu_new, _MU_FLOOR, out=mu_new)
            nll_new = _poisson_nll(y, mu_new)
            accept = (nll_new <= nll + 1e-10) | ~active
            if accept.all():
                break
            step = np.where(accept, step, step * 0.5)
        beta = np.maximum(beta - step * d_b, 0.0)
        rho = np.maximum(rho - step * d_r, 0.0)

    beta[~has_counts] = 0.0
    rho[~has_counts] = 0.0
    rho[~has_signal] = 0.0

    mu = A * (beta[None, :] + rho[None, :] * S)
    loglik = -_poisson_nll(y, mu)
    # Null model: diffuse-only MLE has the closed form beta0 = Σy / ΣA.
    beta0 = np.divide(
        total_y, A.sum(), out=np.zeros(batch), where=np.full(batch, A.sum() > 0)
    )
    null_loglik = -_poisson_nll(y, A * beta0[None, :])
    return beta, rho, loglik, null_loglik


def _fit_poisson_batch_gpu(y_gpu, A_gpu, S_gpu, gpu, fit_diffuse, max_iter, tol):
    """Projected Newton for one batch of genes (PyTorch, GPU-resident)."""
    torch = gpu.torch
    y, A, S = y_gpu, A_gpu, S_gpu
    batch = y.shape[1]
    fzero = torch.zeros((), dtype=y.dtype, device=y.device)

    total_y = y.sum(dim=0)
    has_counts = total_y > 0
    A_col_sum = A.sum()
    has_signal = (S * A).sum(dim=0) > 0

    beta = torch.where(has_counts, total_y / A_col_sum, torch.zeros_like(total_y))
    if not fit_diffuse:
        beta = torch.zeros_like(beta)
    # Closed-form weighted-OLS init for rho (see the CPU branch).
    AS = A * S
    ols_denom = (A * AS**2).sum(dim=0)
    rho = torch.where(
        ols_denom > 0,
        ((A * AS * y).sum(dim=0) / ols_denom.clamp_min(_MU_FLOOR)).clamp_min(0.0),
        torch.zeros_like(total_y),
    )
    free = has_counts & (has_signal if fit_diffuse else (S > 0).any(dim=0))

    nll = torch.full((batch,), float("inf"), dtype=y.dtype, device=y.device)
    for _ in range(max_iter):
        mu = A * (beta.unsqueeze(0) + rho.unsqueeze(0) * S)
        mu = mu.clamp_min(_MU_FLOOR)
        r = 1.0 - y / mu
        ymu2 = y / mu**2
        g_b = (A * r).sum(dim=0)
        g_r = (A * S * r).sum(dim=0)
        h_bb = (A**2 * ymu2).sum(dim=0)
        h_br = (A**2 * S * ymu2).sum(dim=0)
        h_rr = (A**2 * S**2 * ymu2).sum(dim=0)
        nll = (mu - y * torch.log(mu)).sum(dim=0)

        # KKT-aware stopping (see the CPU branch).
        g_b_eff = torch.where(beta > 0, g_b, g_b.clamp_max(0.0))
        g_r_eff = torch.where(rho > 0, g_r, g_r.clamp_max(0.0))
        grad_norm = torch.maximum(g_b_eff.abs(), g_r_eff.abs())
        active = free & (grad_norm >= tol * (total_y + 1.0))
        if not bool(active.any()):
            break

        det = h_bb * h_rr - h_br**2
        use_newton = active & (det > _HESS_RIDGE * h_bb * (h_rr + _HESS_RIDGE))
        d_b = torch.zeros_like(beta)
        d_r = torch.zeros_like(rho)
        d_b[use_newton] = (
            h_rr[use_newton] * g_b[use_newton]
            - h_br[use_newton] * g_r[use_newton]
        ) / det[use_newton]
        d_r[use_newton] = (
            -h_br[use_newton] * g_b[use_newton]
            + h_bb[use_newton] * g_r[use_newton]
        ) / det[use_newton]
        idx = active & ~use_newton
        scale = (h_bb[idx] + h_rr[idx]).clamp_min(_HESS_RIDGE)
        d_b[idx] = g_b[idx] / scale
        d_r[idx] = g_r[idx] / scale
        if not fit_diffuse:
            d_b = torch.zeros_like(d_b)
        d_r = torch.where(has_signal, d_r, fzero)
        # Active set (see the CPU branch).
        bind_b = active & (beta <= 0) & (d_b > 0)
        bind_r = active & (rho <= 0) & (d_r > 0)
        only_r = bind_b & ~bind_r
        only_b = bind_r & ~bind_b
        d_b = torch.where(bind_b, torch.zeros_like(d_b), d_b)
        d_r = torch.where(bind_r, torch.zeros_like(d_r), d_r)
        d_r[only_r] = g_r[only_r] / h_rr[only_r].clamp_min(_HESS_RIDGE)
        if fit_diffuse:
            d_b[only_b] = g_b[only_b] / h_bb[only_b].clamp_min(_HESS_RIDGE)

        step = torch.ones_like(beta)
        for _ls in range(30):
            new_beta = (beta - step * d_b).clamp_min(0.0)
            new_rho = (rho - step * d_r).clamp_min(0.0)
            mu_new = (A * (new_beta.unsqueeze(0) + new_rho.unsqueeze(0) * S)).clamp_min(_MU_FLOOR)
            nll_new = (mu_new - y * torch.log(mu_new)).sum(dim=0)
            accept = (nll_new <= nll + 1e-10) | ~active
            if bool(accept.all()):
                break
            step = torch.where(accept, step, step * 0.5)
        beta = (beta - step * d_b).clamp_min(0.0)
        rho = (rho - step * d_r).clamp_min(0.0)

    beta = torch.where(has_counts, beta, fzero)
    rho = torch.where(has_counts & has_signal, rho, fzero)

    mu = (A * (beta.unsqueeze(0) + rho.unsqueeze(0) * S)).clamp_min(_MU_FLOOR)
    loglik = -(mu - y * torch.log(mu)).sum(dim=0)
    beta0 = torch.where(has_counts, total_y / A_col_sum, torch.zeros_like(total_y))
    mu0 = (A * beta0.unsqueeze(0)).clamp_min(_MU_FLOOR)
    null_loglik = -(mu0 - y * torch.log(mu0)).sum(dim=0)
    return beta, rho, loglik, null_loglik


def estimate_leakage_poisson(
    gene_indices: np.ndarray,
    cell_expr: np.ndarray,
    cell_areas: np.ndarray,
    empty_bin_expr: np.ndarray,
    empty_bin_areas: np.ndarray,
    W_empty,
    use_expr_weight: bool = False,
    fit_diffuse: bool = True,
    max_newton_iter: int = 25,
    newton_tol: float = 1e-6,
    gpu: Optional[GPUContext] = None,
    storage_dtype=None,
    reduction_dtype=None,
    gpu_gene_batch_size: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], Optional[int]]:
    """Fit the Poisson background model for every selected gene.

    Args mirror :func:`stambient.alpha_estimation.estimate_alphas`, plus:

        fit_diffuse: include the non-local diffuse component β.  ``False``
            gives the local-only Poisson model μ = A·ρ·S (ablation B1).
        max_newton_iter: maximum projected-Newton iterations per gene.
        newton_tol: stop when |gradient| < tol × (total counts + 1).

    Returns:
        betas: [n_genes_use] diffuse ambient rates (0 when ``fit_diffuse``
            is False).
        rhos: [n_genes_use] local leakage strengths (Poisson successor of α).
        stats: dict with per-gene ``loglik``, ``null_loglik`` (diffuse-only
            reference) and ``rho_positive_rate`` summary for diagnostics.
        effective_gpu_gene_batch_size: batch size used on the GPU path
            (``None`` on CPU), reused by the correction stage.
    """
    n_genes_use = len(gene_indices)
    n_cells = cell_expr.shape[1]
    betas = np.zeros(n_genes_use, dtype=np.float64)
    rhos = np.zeros(n_genes_use, dtype=np.float64)
    logliks = np.zeros(n_genes_use, dtype=np.float64)
    null_logliks = np.zeros(n_genes_use, dtype=np.float64)
    effective_gpu_gene_batch_size: Optional[int] = None

    sources = np.divide(
        cell_expr[gene_indices],
        cell_areas[None, :],
        out=np.zeros((n_genes_use, n_cells), dtype=np.float64),
        where=cell_areas[None, :] > 0,
    )
    if use_expr_weight:
        sources = np.array([expression_weight(s) for s in sources])

    if gpu is not None:
        if gpu_gene_batch_size is None:
            effective_gpu_gene_batch_size = choose_gpu_gene_batch_size(
                gpu,
                n_rows=empty_bin_expr.shape[1],
                n_cells=n_cells,
                dtype=storage_dtype,
                max_batch_size=n_genes_use,
            )
        else:
            effective_gpu_gene_batch_size = max(1, int(gpu_gene_batch_size))
        A_gpu = to_gpu(
            empty_bin_areas[:, None], gpu, dtype=reduction_dtype
        )
        for start in range(0, n_genes_use, effective_gpu_gene_batch_size):
            stop = min(start + effective_gpu_gene_batch_size, n_genes_use)
            S_gpu = sparse_mm(
                W_empty,
                to_gpu(sources[start:stop].T, gpu, dtype=storage_dtype),
                gpu,
            ).to(dtype=reduction_dtype)
            y_gpu = to_gpu(
                empty_bin_expr[gene_indices[start:stop]].T,
                gpu,
                dtype=reduction_dtype,
            )
            b, r, ll, ll0 = _fit_poisson_batch_gpu(
                y_gpu, A_gpu, S_gpu, gpu, fit_diffuse,
                max_newton_iter, newton_tol,
            )
            if fit_diffuse:
                # Nested-model safeguard: the local-only MLE is always
                # feasible for the diffuse+local model, so keep the
                # pointwise better solution (beta -> 0 there).
                b1, r1, ll1, _ = _fit_poisson_batch_gpu(
                    y_gpu, A_gpu, S_gpu, gpu, False,
                    max_newton_iter, newton_tol,
                )
                better = ll1 > ll
                b = gpu.torch.where(better, gpu.torch.zeros_like(b), b)
                r = gpu.torch.where(better, r1, r)
                ll = gpu.torch.where(better, ll1, ll)
            betas[start:stop] = to_cpu(b, gpu)
            rhos[start:stop] = to_cpu(r, gpu)
            logliks[start:stop] = to_cpu(ll, gpu)
            null_logliks[start:stop] = to_cpu(ll0, gpu)
    else:
        S_all = W_empty.dot(sources.T)  # [n_bins × n_genes_use]
        # CPU processes all genes in one dense Newton batch.
        y_all = empty_bin_expr[gene_indices].T
        b, r, ll, ll0 = _fit_poisson_batch_cpu(
            y_all,
            empty_bin_areas,
            S_all,
            fit_diffuse,
            max_newton_iter,
            newton_tol,
        )
        if fit_diffuse:
            # Nested-model safeguard (see the GPU branch).
            b1, r1, ll1, _ = _fit_poisson_batch_cpu(
                y_all,
                empty_bin_areas,
                S_all,
                False,
                max_newton_iter,
                newton_tol,
            )
            better = ll1 > ll
            b = np.where(better, 0.0, b)
            r = np.where(better, r1, r)
            ll = np.where(better, ll1, ll)
        betas, rhos, logliks, null_logliks = b, r, ll, ll0

    stats = {
        "loglik": logliks,
        "null_loglik": null_logliks,
        "rho_positive_rate": float((rhos > 0).mean()) if n_genes_use else 0.0,
        "beta_positive_rate": float((betas > 0).mean()) if n_genes_use else 0.0,
    }
    return betas, rhos, stats, effective_gpu_gene_batch_size
