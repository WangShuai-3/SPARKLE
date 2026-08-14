"""Latent-X self-consistent ambient correction (SPARKLE 2.0-alpha1).

SPARKLE 1.x predicts each cell's ambient inflow from the *observed* (already
contaminated) neighbour expression Y.  The latent mode instead treats the
clean expression X as the variable to recover and solves the self-consistent
system

    X = max(Y - L(X), 0),   L(X) = alpha_g * A_c * (W_cell @ (X / A))

with a damped fixed-point iteration:

    X^(0)   = Y
    L^(t)   = L(X^(t))            (optionally self-confidence-penalised)
    X~      = max(Y - L^(t), 0)
    X^(t+1) = (1 - eta) X^(t) + eta X~

until the relative L1 change falls below ``tol`` or ``max_iter`` is reached.

Only the leakage source changes (X instead of Y); lambda, alpha, R² gating
and the penalty function are inherited from the 1.x pipeline unchanged.
Both batched CPU (SciPy) and GPU (PyTorch) branches live here, mirroring
:mod:`stambient.correction`.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from .gpu import (
    GPUContext,
    sparse_mm,
    to_cpu,
    to_gpu,
    weighted_gpu_csr,
)
from .spatial import DistanceMetric, distance_graph_to_weights
from .weights import expression_weight, self_confidence_weight

_L1_EPS = 1e-12


def _divide_by_area(values: np.ndarray, cell_areas: np.ndarray) -> np.ndarray:
    """Row-wise values / area with a zero guard, [batch × n_cells]."""
    return np.divide(
        values,
        cell_areas[None, :],
        out=np.zeros_like(values, dtype=np.float64),
        where=cell_areas[None, :] > 0,
    )


def _latent_fixed_point_cpu(
    Y: np.ndarray,
    alphas: np.ndarray,
    cell_areas: np.ndarray,
    W_cell,
    use_expr_weight: bool,
    self_confidence_penalty: bool,
    penalty_mode: str,
    eta: float,
    max_iter: int,
    tol: float,
) -> Tuple[np.ndarray, List[float]]:
    """Damped fixed-point solve for one batch of genes (CPU, SciPy)."""
    X = Y.astype(np.float64, copy=True)
    rel_history: List[float] = []
    for _ in range(max_iter):
        sources = _divide_by_area(X, cell_areas)
        weighted_sources = (
            np.array([expression_weight(s) for s in sources])
            if use_expr_weight
            else sources
        )
        neighbor = W_cell.dot(weighted_sources.T)  # [n_cells × batch]
        ambient = alphas[None, :] * cell_areas[:, None] * neighbor
        if self_confidence_penalty:
            penalties = self_confidence_weight(sources, mode=penalty_mode)
            ambient *= penalties.T
        X_raw = np.maximum(Y - ambient.T, 0.0)
        X_new = (1.0 - eta) * X + eta * X_raw
        rel = float(
            np.abs(X_new - X).sum() / (np.abs(X).sum() + _L1_EPS)
        )
        rel_history.append(rel)
        X = X_new
        if rel < tol:
            break
    return X, rel_history


def _positive_quantile_gpu(x, q: float, torch):
    """Per-column quantile of positive entries, linear interpolation.

    ``x`` is [n_cells × batch]; returns [batch].  Columns without any
    positive entry yield 0 (callers map that case to the NumPy behaviour,
    where the corresponding weight degrades to a neutral value).
    """
    n_cells, batch = x.shape
    sorted_x, _ = torch.sort(x, dim=0, descending=True)
    n_pos = (x > 0).sum(dim=0).to(x.dtype)
    # Descending order: numpy's ascending linear-interpolated rank r maps to
    # (n_pos - 1) - r.
    rank = (1.0 - q / 100.0) * torch.clamp(n_pos - 1.0, min=0.0)
    lo = torch.floor(rank).long()
    hi = torch.clamp(lo + 1, max=n_cells - 1)
    frac = (rank - lo.to(x.dtype)).unsqueeze(0)  # [1 × batch]
    cols = torch.arange(batch, device=x.device)
    v_lo = sorted_x[lo, cols]
    v_hi = sorted_x[hi, cols]
    return v_lo * (1.0 - frac.squeeze(0)) + v_hi * frac.squeeze(0)


def _expression_weight_gpu(sources, torch):
    """Torch equivalent of :func:`stambient.weights.expression_weight`."""
    s_median = _positive_quantile_gpu(sources, 50.0, torch)
    safe_median = torch.where(
        s_median > 0, s_median, torch.ones_like(s_median)
    )
    weighted = sources * sources / (sources + safe_median.unsqueeze(0))
    # No-positive columns: NumPy returns zeros; s_median<=0 columns (already
    # zero everywhere) stay zero through the formula above.
    return weighted


def _self_confidence_weight_gpu(sources, mode: str, torch):
    """Torch equivalent of :func:`stambient.weights.self_confidence_weight`."""
    n_pos = (sources > 0).sum(dim=0)

    def _neutral_like():
        return torch.ones_like(sources)

    if mode in ("1/(1+s/p50)", "p50/s"):
        p = _positive_quantile_gpu(sources, 50.0, torch)
        valid = (n_pos > 0) & (p > 0)
        safe_p = torch.where(p > 0, p, torch.ones_like(p)).unsqueeze(0)
        if mode == "1/(1+s/p50)":
            w = 1.0 / (1.0 + sources / safe_p)
        else:
            w = torch.where(sources < safe_p, torch.ones_like(sources), safe_p / torch.clamp(sources, min=1e-300))
        return torch.where(valid.unsqueeze(0), w, _neutral_like())

    p90 = _positive_quantile_gpu(sources, 90.0, torch)
    valid = (n_pos > 0) & (p90 > 0)
    safe_p90 = torch.where(p90 > 0, p90, torch.ones_like(p90)).unsqueeze(0)
    if mode == "1/(1+(s/p90)²)":
        w = 1.0 / (1.0 + (sources / safe_p90) ** 2)
    elif mode == "exp(-s/p90)":
        w = torch.exp(-sources / safe_p90)
    else:  # default: 1/(1+s/p90)
        w = 1.0 / (1.0 + sources / safe_p90)
    return torch.where(valid.unsqueeze(0), w, _neutral_like())


def _latent_fixed_point_gpu(
    Y,
    alphas_gpu,
    areas_gpu,
    W_cell,
    gpu: GPUContext,
    storage_dtype,
    reduction_dtype,
    use_expr_weight: bool,
    self_confidence_penalty: bool,
    penalty_mode: str,
    eta: float,
    max_iter: int,
    tol: float,
):
    """Damped fixed-point solve for one batch of genes (GPU, PyTorch).

    ``Y`` is [n_cells × batch] in ``reduction_dtype`` and stays resident on
    the device across iterations.
    """
    torch = gpu.torch
    zero = torch.zeros((), dtype=reduction_dtype, device=gpu.device)
    X = Y.clone()
    rel_history: List[float] = []
    for _ in range(max_iter):
        sources = torch.where(
            areas_gpu > 0,
            X / torch.where(areas_gpu > 0, areas_gpu, torch.ones_like(areas_gpu)),
            zero,
        )
        if use_expr_weight:
            weighted_sources = _expression_weight_gpu(sources, torch)
        else:
            weighted_sources = sources
        neighbor = sparse_mm(
            W_cell, weighted_sources.to(dtype=storage_dtype), gpu
        ).to(dtype=reduction_dtype)
        ambient = alphas_gpu * areas_gpu * neighbor
        if self_confidence_penalty:
            ambient = ambient * _self_confidence_weight_gpu(
                sources, penalty_mode, torch
            )
        X_raw = (Y - ambient).clamp_min(0.0)
        X_new = (1.0 - eta) * X + eta * X_raw
        rel = float(
            (X_new - X).abs().sum() / (X.abs().sum() + _L1_EPS)
        )
        rel_history.append(rel)
        X = X_new
        if rel < tol:
            break
    return X, rel_history


def correct_cells_latent(
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
    eta: float = 0.5,
    max_iter: int = 20,
    tol: float = 1e-4,
) -> Tuple[np.ndarray, Dict]:
    """Correct cells by solving the latent-X self-consistent system.

    Args identical to :func:`stambient.correction.correct_cells`, plus:

        eta: damping factor of the fixed-point update (0 < eta <= 1).
        max_iter: maximum fixed-point iterations per gene batch.
        tol: stop when ||ΔX||₁ / (||X||₁ + ε) falls below this value.

    Returns:
        corrected_expr: [genes × n_cells] corrected per-cell expression.
        latent_info: convergence diagnostics dict with keys
            ``eta``, ``max_iter``, ``tol``, ``n_iterations`` (per batch),
            ``final_rel_change`` (per batch), ``converged`` (bool),
            ``rel_change_history_max`` (longest per-iteration trace).
    """
    if not (0.0 < eta <= 1.0):
        raise ValueError(f"latent eta must be in (0, 1]; got {eta!r}")
    if max_iter < 1:
        raise ValueError(f"latent max_iter must be >= 1; got {max_iter!r}")
    if tol <= 0:
        raise ValueError(f"latent tol must be positive; got {tol!r}")

    if gpu is not None:
        W_cell = weighted_gpu_csr(
            cell_distance_graph, lam_weights, distance_metric, gpu
        )
    else:
        W_cell = distance_graph_to_weights(
            cell_distance_graph, lam_weights, distance_metric
        )

    corrected_expr = cell_expr.astype(np.float64, copy=True)

    gene_idx_set = set(gene_indices)
    gene_idx_to_pos = {int(g_idx): i for i, g_idx in enumerate(gene_indices)}
    corrected_gene_indices = []
    corrected_positions = []
    for g_idx in range(n_genes):
        if g_idx not in gene_idx_set:
            continue
        pos = gene_idx_to_pos[g_idx]
        if r2_scores[pos] < r2_threshold:
            continue
        corrected_gene_indices.append(g_idx)
        corrected_positions.append(pos)

    batch_iterations: List[int] = []
    batch_final_rel: List[float] = []
    longest_history: List[float] = []

    if corrected_gene_indices:
        cg_idx = np.array(corrected_gene_indices, dtype=np.int64)
        cpos = np.array(corrected_positions, dtype=np.int64)

        if gpu is not None:
            batch_size = effective_gpu_gene_batch_size or 512
            areas_gpu = to_gpu(
                cell_areas[:, None], gpu, dtype=reduction_dtype
            )
            for start in range(0, len(cg_idx), batch_size):
                stop = min(start + batch_size, len(cg_idx))
                batch_gene_indices = cg_idx[start:stop]
                batch_positions = cpos[start:stop]
                Y_gpu = to_gpu(
                    cell_expr[batch_gene_indices].T,
                    gpu,
                    dtype=reduction_dtype,
                )
                alphas_gpu = to_gpu(
                    alphas[batch_positions][None, :],
                    gpu,
                    dtype=reduction_dtype,
                )
                X_gpu, history = _latent_fixed_point_gpu(
                    Y_gpu,
                    alphas_gpu,
                    areas_gpu,
                    W_cell,
                    gpu,
                    storage_dtype,
                    reduction_dtype,
                    use_expr_weight,
                    self_confidence_penalty,
                    penalty_mode,
                    eta,
                    max_iter,
                    tol,
                )
                corrected_expr[batch_gene_indices] = to_cpu(X_gpu.T, gpu)
                batch_iterations.append(len(history))
                batch_final_rel.append(history[-1])
                if len(history) > len(longest_history):
                    longest_history = history
        else:
            X, history = _latent_fixed_point_cpu(
                Y=cell_expr[cg_idx],
                alphas=alphas[cpos],
                cell_areas=cell_areas,
                W_cell=W_cell,
                use_expr_weight=use_expr_weight,
                self_confidence_penalty=self_confidence_penalty,
                penalty_mode=penalty_mode,
                eta=eta,
                max_iter=max_iter,
                tol=tol,
            )
            corrected_expr[cg_idx] = X
            batch_iterations.append(len(history))
            batch_final_rel.append(history[-1])
            longest_history = history

    converged = bool(batch_iterations) and all(
        rel < tol for rel in batch_final_rel
    )
    latent_info = {
        "eta": float(eta),
        "max_iter": int(max_iter),
        "tol": float(tol),
        "n_iterations": batch_iterations,
        "final_rel_change": batch_final_rel,
        "converged": converged,
        "rel_change_history_max": longest_history,
    }
    return corrected_expr, latent_info
