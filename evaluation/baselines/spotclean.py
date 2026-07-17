"""Two SpotClean adaptations for the synthetic benchmark.

This module implements the published SpotClean redistribution model at the
segmented-cell level.  The cell-only mode treats each segmented cell as one
tissue spot and excludes empty DNBs.  The background-aware mode concatenates
cells with binned empty DNBs and uses the latter as background spots.

This is not a call to the official R package.  The standard SpotClean workflow
uses background spots to identify the global bleeding and distal rates.  The
cell-only adapter therefore keeps them fixed, while the background-aware
adapter estimates them from concatenated total counts.  Both use SpotClean EM
expression redistribution.  These are independent Python implementations,
not calls to the official R package.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.sparse import csr_matrix, issparse
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

from stambient.cell_pipeline import _bin_empty_dnbs


def _aggregate_cells(
    dnb_expr,
    dnb_coords: np.ndarray,
    dnb_labels: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate labelled DNBs into gene-by-cell counts and cell centroids."""
    labels = np.asarray(dnb_labels)
    coords = np.asarray(dnb_coords, dtype=np.float64)
    cell_mask = labels >= 0
    cell_ids = np.unique(labels[cell_mask])
    if cell_ids.size == 0:
        raise ValueError("Cell-adapted SpotClean requires at least one labelled cell.")

    id_to_col = {cell_id: i for i, cell_id in enumerate(cell_ids)}
    cell_columns = np.fromiter(
        (id_to_col[cell_id] for cell_id in labels[cell_mask]),
        dtype=np.int64,
        count=int(cell_mask.sum()),
    )
    dnb_columns = np.flatnonzero(cell_mask)
    assignment = csr_matrix(
        (
            np.ones(dnb_columns.size, dtype=np.float64),
            (dnb_columns, cell_columns),
        ),
        shape=(labels.size, cell_ids.size),
    )

    expr = dnb_expr.tocsr() if issparse(dnb_expr) else csr_matrix(dnb_expr)
    cell_expr = np.asarray((expr @ assignment).toarray(), dtype=np.float64)

    areas = np.bincount(cell_columns, minlength=cell_ids.size).astype(np.float64)
    centroids = np.column_stack(
        [
            np.bincount(
                cell_columns,
                weights=coords[cell_mask, axis],
                minlength=cell_ids.size,
            )
            for axis in range(2)
        ]
    )
    centroids /= areas[:, None]
    return cell_expr, centroids, cell_ids


def _cell_to_cell_weight(
    centroids: np.ndarray,
    bandwidth: float,
    distal_rate: float,
) -> np.ndarray:
    """Return a source-by-destination SpotClean swapping matrix.

    The diagonal is zero because this adapter models swapping between distinct
    cells only.  Each source row sums to one before multiplication by the global
    bleeding rate.
    """
    n_cells = centroids.shape[0]
    if n_cells == 1:
        return np.zeros((1, 1), dtype=np.float64)

    distances = cdist(centroids, centroids, metric="euclidean")
    local = np.exp(-(distances ** 2) / (2.0 * float(bandwidth) ** 2))
    np.fill_diagonal(local, 0.0)
    local_sums = local.sum(axis=1, keepdims=True)
    local = np.divide(
        local,
        local_sums,
        out=np.zeros_like(local),
        where=local_sums > 0,
    )

    distal = np.ones((n_cells, n_cells), dtype=np.float64)
    np.fill_diagonal(distal, 0.0)
    distal /= n_cells - 1
    return (1.0 - distal_rate) * local + distal_rate * distal


def _aggregate_empty_bins(
    dnb_expr,
    dnb_coords: np.ndarray,
    dnb_labels: np.ndarray,
    bin_size: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate empty DNBs using the same fixed-grid mapping as SPARKLE."""
    bin_coords, bin_areas, dnb_indices, bin_indices = _bin_empty_dnbs(
        np.asarray(dnb_coords, dtype=np.float64),
        np.asarray(dnb_labels),
        bin_size,
    )
    if bin_areas.size == 0:
        raise ValueError("Background-aware SpotClean requires empty DNBs.")

    assignment = csr_matrix(
        (
            np.ones(dnb_indices.size, dtype=np.float64),
            (dnb_indices, bin_indices),
        ),
        shape=(len(dnb_labels), bin_areas.size),
    )
    expr = dnb_expr.tocsr() if issparse(dnb_expr) else csr_matrix(dnb_expr)
    empty_expr = np.asarray((expr @ assignment).toarray(), dtype=np.float64)
    return empty_expr, bin_coords, bin_areas.astype(np.float64)


def _source_destination_weights(
    source_coords: np.ndarray,
    destination_coords: np.ndarray,
    bandwidth: float,
) -> np.ndarray:
    """Normalized Gaussian weights from cell sources to all destinations."""
    distances = cdist(source_coords, destination_coords, metric="euclidean")
    weights = np.exp(-(distances ** 2) / (2.0 * float(bandwidth) ** 2))

    # Cells occupy the first columns and have the same ordering as sources.
    # SpotClean defines swapping between distinct spots; the non-swapped part
    # is represented separately by (1 - bleeding_rate).
    n_sources = source_coords.shape[0]
    weights[np.arange(n_sources), np.arange(n_sources)] = 0.0
    row_sums = weights.sum(axis=1, keepdims=True)
    return np.divide(
        weights,
        row_sums,
        out=np.zeros_like(weights),
        where=row_sums > 0,
    )


def _initial_background_rates(
    totals: np.ndarray,
    n_tissue: int,
) -> Tuple[float, float, float]:
    """SpotClean-style initial rates and bleeding-rate lower bound."""
    background = totals[n_tissue:]
    total_sum = float(totals.sum())
    background_sum = float(background.sum())
    if background.size == 0 or total_sum <= 0:
        raise ValueError("Non-empty background observations are required.")

    bleed_init = float(background.mean() * totals.size / total_sum)
    bleed_lower = float(background_sum / total_sum)

    if background_sum <= 0:
        distal_init = 0.1
    else:
        q25, q50 = np.quantile(background, [0.25, 0.50])
        trimmed = background[(background >= q25) & (background <= q50)]
        uniform_contamination = float(trimmed.mean()) if trimmed.size else 0.0
        distal_init = uniform_contamination / background_sum * background.size
    distal_init = float(np.clip(distal_init, 0.1, 0.5))
    return float(np.clip(bleed_init, bleed_lower, 1.0)), distal_init, bleed_lower


def _estimate_contamination_parameters(
    totals: np.ndarray,
    n_tissue: int,
    proximal_weight: np.ndarray,
) -> Dict:
    """Fit SpotClean global parameters by RSS minimization on total counts."""
    totals = np.asarray(totals, dtype=np.float64)
    n_spots = totals.size
    tissue_totals = totals[:n_tissue]
    positive = tissue_totals > 0
    if not positive.any():
        raise ValueError("At least one tissue cell must have positive counts.")

    bleed_init, distal_init, bleed_lower = _initial_background_rates(
        totals, n_tissue
    )
    mu_init = tissue_totals[positive].astype(np.float64)
    mu_init *= totals.sum() / mu_init.sum()

    stay = np.zeros((n_spots, positive.sum()), dtype=np.float64)
    positive_indices = np.flatnonzero(positive)
    stay[positive_indices, np.arange(positive.sum())] = 1.0
    proximal = proximal_weight[positive].T
    uniform = np.full_like(proximal, 1.0 / n_spots)

    def objective_and_gradient(parameters):
        bleed, distal = parameters[:2]
        mu = parameters[2:]
        swapping = (1.0 - distal) * proximal + distal * uniform
        design = (1.0 - bleed) * stay + bleed * swapping
        expected = design @ mu
        residual = expected - totals

        gradient = np.empty_like(parameters)
        d_bleed = (swapping - stay) @ mu
        d_distal = bleed * ((uniform - proximal) @ mu)
        gradient[0] = 2.0 * residual.dot(d_bleed)
        gradient[1] = 2.0 * residual.dot(d_distal)
        gradient[2:] = 2.0 * design.T.dot(residual)
        return float(residual.dot(residual)), gradient

    bounds = [(bleed_lower, 1.0), (0.1, 1.0)]
    bounds.extend([(0.0, None)] * int(positive.sum()))
    starts = [
        (max(bleed_init, 0.1), max(distal_init, 0.1)),
        (max(bleed_lower, 0.3), 0.3),
        (max(bleed_lower, 0.5), 0.3),
    ]
    solutions = []
    for bleed_start, distal_start in starts:
        initial = np.concatenate(([bleed_start, distal_start], mu_init))
        solution = minimize(
            objective_and_gradient,
            initial,
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={"maxiter": 100},
        )
        solutions.append(solution)
    best = min(solutions, key=lambda result: float(result.fun))

    mu = np.zeros(n_tissue, dtype=np.float64)
    mu[positive] = best.x[2:]
    return {
        "rss": float(best.fun),
        "bleeding_rate": float(best.x[0]),
        "distal_rate": float(best.x[1]),
        "mu_total": mu,
        "converged": bool(best.success),
        "optimizer_message": str(best.message),
        "bleeding_rate_lower_bound": bleed_lower,
    }


def _spotclean_background_em(
    observed_all: np.ndarray,
    n_tissue: int,
    proximal_weight: np.ndarray,
    bleed_rate: float,
    distal_rate: float,
    maxit: int,
    tol: float,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """SpotClean EM with tissue/cells as sources and empty bins as receivers."""
    n_spots = observed_all.shape[1]
    tissue_observed = observed_all[:, :n_tissue]
    gene_all = observed_all.sum(axis=1)
    gene_tissue = tissue_observed.sum(axis=1)
    scale = np.divide(
        gene_all,
        gene_tissue,
        out=np.ones_like(gene_all),
        where=gene_tissue > 0,
    )
    latent = tissue_observed * scale[:, None]

    uniform = np.full_like(proximal_weight, 1.0 / n_spots)
    swapping = (1.0 - distal_rate) * proximal_weight + distal_rate * uniform
    eps = np.finfo(np.float64).tiny
    loglik = []

    for iteration in range(1, maxit + 1):
        expected = bleed_rate * (latent @ swapping)
        expected[:, :n_tissue] += (1.0 - bleed_rate) * latent
        expected_safe = np.maximum(expected, eps)
        ratio = np.divide(
            observed_all,
            expected_safe,
            out=np.zeros_like(observed_all),
            where=expected_safe > 0,
        )
        stayed = (
            (1.0 - bleed_rate)
            * tissue_observed
            * latent
            / expected_safe[:, :n_tissue]
        )
        swapped = bleed_rate * latent * (ratio @ swapping.T)
        updated = stayed + swapped

        loglik.append(_poisson_loglik(observed_all, expected_safe))
        max_difference = float(np.max(np.abs(updated - latent)))
        latent = updated
        if iteration > 1 and max_difference < tol:
            break
    return latent, np.asarray(loglik, dtype=np.float64), iteration


def _spotclean_em(
    observed: np.ndarray,
    swap_weight: np.ndarray,
    bleed_rate: float,
    maxit: int,
    tol: float,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Estimate latent gene-by-cell expression with SpotClean EM updates."""
    latent = np.asarray(observed, dtype=np.float64).copy()
    if latent.shape[1] == 1 or bleed_rate == 0:
        return latent, np.array([_poisson_loglik(observed, latent)]), 0

    eps = np.finfo(np.float64).tiny
    loglik = []
    for iteration in range(1, maxit + 1):
        expected = (1.0 - bleed_rate) * latent
        expected += bleed_rate * (latent @ swap_weight)
        expected_safe = np.maximum(expected, eps)

        ratio = np.divide(
            observed,
            expected_safe,
            out=np.zeros_like(observed, dtype=np.float64),
            where=expected_safe > 0,
        )
        stayed = (1.0 - bleed_rate) * observed * latent / expected_safe
        swapped = bleed_rate * latent * (ratio @ swap_weight.T)
        updated = stayed + swapped

        loglik.append(_poisson_loglik(observed, expected_safe))
        max_difference = float(np.max(np.abs(updated - latent)))
        latent = updated
        if iteration > 1 and max_difference < tol:
            break

    return latent, np.asarray(loglik, dtype=np.float64), iteration


def _poisson_loglik(observed: np.ndarray, expected: np.ndarray) -> float:
    """Poisson log-likelihood up to the constant log-factorial term."""
    expected_safe = np.maximum(expected, np.finfo(np.float64).tiny)
    return float(np.sum(observed * np.log(expected_safe) - expected_safe))


def _knn_local_weight(
    cell_coords: np.ndarray,
    bandwidth: float,
    n_neighbors: int,
) -> csr_matrix:
    """Build a row-normalized Gaussian source-to-destination kNN graph."""
    n_cells = len(cell_coords)
    if n_cells <= 1:
        return csr_matrix((n_cells, n_cells), dtype=np.float64)
    k = min(int(n_neighbors), n_cells - 1)
    distances, indices = cKDTree(cell_coords).query(cell_coords, k=k + 1)
    distances = np.asarray(distances[:, 1:], dtype=np.float64)
    indices = np.asarray(indices[:, 1:], dtype=np.int64)
    weights = np.exp(-(distances ** 2) / (2.0 * float(bandwidth) ** 2))
    row_sums = weights.sum(axis=1, keepdims=True)
    weights = np.divide(weights, row_sums, out=np.zeros_like(weights), where=row_sums > 0)
    rows = np.repeat(np.arange(n_cells, dtype=np.int64), k)
    return csr_matrix(
        (weights.ravel(), (rows, indices.ravel())),
        shape=(n_cells, n_cells),
    )


def _spotclean_sparse_local_em(
    observed: np.ndarray,
    local_weight: csr_matrix,
    bleed_rate: float,
    distal_rate: float,
    maxit: int,
    tol: float,
) -> Tuple[np.ndarray, float, int]:
    """Batched SpotClean EM without materializing the distal dense matrix."""
    observed = np.asarray(observed, dtype=np.float64)
    latent = observed.copy()
    n_cells = observed.shape[1]
    if n_cells <= 1 or bleed_rate == 0:
        return latent, _poisson_loglik(observed, latent), 0

    local_scale = 1.0 - distal_rate
    uniform_scale = distal_rate / (n_cells - 1)
    eps = np.finfo(np.float64).tiny
    final_loglik = float("-inf")

    for iteration in range(1, maxit + 1):
        local_forward = np.asarray(local_weight.T @ latent.T).T
        uniform_forward = latent.sum(axis=1, keepdims=True) - latent
        expected = (1.0 - bleed_rate) * latent
        expected += bleed_rate * (
            local_scale * local_forward + uniform_scale * uniform_forward
        )
        expected_safe = np.maximum(expected, eps)
        ratio = np.divide(
            observed,
            expected_safe,
            out=np.zeros_like(observed),
            where=expected_safe > 0,
        )
        stayed = (1.0 - bleed_rate) * observed * latent / expected_safe
        local_backward = np.asarray(local_weight @ ratio.T).T
        uniform_backward = ratio.sum(axis=1, keepdims=True) - ratio
        updated = stayed + bleed_rate * latent * (
            local_scale * local_backward + uniform_scale * uniform_backward
        )
        final_loglik = _poisson_loglik(observed, expected_safe)
        max_difference = float(np.max(np.abs(updated - latent)))
        latent = updated
        if iteration > 1 and max_difference < tol:
            break

    local_forward = np.asarray(local_weight.T @ latent.T).T
    uniform_forward = latent.sum(axis=1, keepdims=True) - latent
    final_expected = (1.0 - bleed_rate) * latent + bleed_rate * (
        local_scale * local_forward + uniform_scale * uniform_forward
    )
    final_loglik = _poisson_loglik(observed, final_expected)
    return latent, final_loglik, iteration


def run_spotclean_sparse_cells(
    cell_expr,
    cell_coords: np.ndarray,
    candidate_bandwidths: Optional[Sequence[float]] = None,
    bleed_rate: float = 0.10,
    distal_rate: float = 0.10,
    n_neighbors: int = 32,
    n_selection_genes: int = 50,
    gene_batch_size: int = 64,
    maxit: int = 3,
    tol: float = 1.0,
    n_jobs: int = 8,
    batch_writer: Optional[Callable[[int, int, np.ndarray], None]] = None,
    verbose: bool = False,
) -> Tuple[Optional[np.ndarray], Dict]:
    """Scalable cell-only SpotClean for large real-data cell matrices.

    The Gaussian component is restricted to the nearest ``n_neighbors`` cells;
    the distal uniform component remains exact and is evaluated algebraically.
    Bandwidth is selected on the most abundant genes, after which all genes are
    corrected in independent batches. ``batch_writer`` permits streaming the
    result directly into a large h5ad without retaining a dense copy in memory.
    """
    if not 0.0 <= bleed_rate < 1.0:
        raise ValueError("bleed_rate must be in [0, 1).")
    if not 0.0 <= distal_rate <= 1.0:
        raise ValueError("distal_rate must be in [0, 1].")
    if n_neighbors < 1 or n_selection_genes < 1 or gene_batch_size < 1:
        raise ValueError("neighbor, selection-gene, and batch sizes must be positive.")
    if maxit < 1 or tol < 0 or n_jobs < 1:
        raise ValueError("maxit/n_jobs must be positive and tol non-negative.")
    if candidate_bandwidths is None:
        candidate_bandwidths = (10, 20, 30, 50, 70, 100, 150, 200, 300)
    bandwidths = sorted({float(x) for x in candidate_bandwidths if float(x) > 0})
    if not bandwidths:
        raise ValueError("candidate_bandwidths must contain a positive value.")

    expression = cell_expr.tocsr() if issparse(cell_expr) else csr_matrix(cell_expr)
    coords = np.asarray(cell_coords, dtype=np.float64)
    n_genes, n_cells = expression.shape
    if coords.shape != (n_cells, 2):
        raise ValueError("cell_coords must have shape [n_cells, 2].")

    gene_totals = np.asarray(expression.sum(axis=1)).ravel()
    n_select = min(int(n_selection_genes), n_genes)
    selected = np.argsort(gene_totals)[-n_select:]
    selected_observed = expression[selected].toarray()

    candidate_loglik = []
    best = None
    for bandwidth in bandwidths:
        local_weight = _knn_local_weight(coords, bandwidth, n_neighbors)
        _, loglik, iterations = _spotclean_sparse_local_em(
            selected_observed,
            local_weight,
            bleed_rate,
            distal_rate,
            maxit=maxit,
            tol=tol,
        )
        candidate_loglik.append(float(loglik))
        if best is None or loglik > best[0]:
            best = (float(loglik), bandwidth, local_weight, iterations)
        if verbose:
            print(f"  bandwidth={bandwidth:g}: selection loglik={loglik:.3e}")

    _, best_bandwidth, local_weight, selection_iterations = best
    corrected_full = None
    if batch_writer is None:
        corrected_full = np.empty((n_genes, n_cells), dtype=np.float64)

    starts = list(range(0, n_genes, gene_batch_size))

    def correct_batch(start):
        end = min(start + gene_batch_size, n_genes)
        observed = expression[start:end].toarray()
        corrected, _, iterations = _spotclean_sparse_local_em(
            observed,
            local_weight,
            bleed_rate,
            distal_rate,
            maxit=maxit,
            tol=tol,
        )
        count_error = float(
            np.max(np.abs(corrected.sum(axis=1) - observed.sum(axis=1)))
        )
        return start, end, corrected, iterations, count_error

    max_count_error = 0.0
    max_iterations = 0
    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        for batch_number, result in enumerate(executor.map(correct_batch, starts), start=1):
            start, end, corrected, iterations, count_error = result
            if batch_writer is None:
                corrected_full[start:end] = corrected
            else:
                batch_writer(start, end, corrected)
            max_count_error = max(max_count_error, count_error)
            max_iterations = max(max_iterations, iterations)
            if verbose and (batch_number == 1 or batch_number % 25 == 0 or end == n_genes):
                print(f"  corrected genes {end:,}/{n_genes:,}")

    diagnostics = {
        "method": "CellSpotClean-sparse",
        "adaptation": "segmented cells treated as tissue spots",
        "official_r_package": False,
        "background_used": False,
        "empty_dnbs_used": False,
        "parameter_identification": "fixed rates because no background spots",
        "spatial_approximation": "Gaussian component truncated to k nearest cells",
        "bleeding_rate": float(bleed_rate),
        "distal_rate": float(distal_rate),
        "contamination_bandwidth": float(best_bandwidth),
        "candidate_bandwidths": bandwidths,
        "candidate_loglik": candidate_loglik,
        "n_neighbors": int(min(n_neighbors, max(n_cells - 1, 0))),
        "n_selection_genes": int(n_select),
        "gene_batch_size": int(gene_batch_size),
        "maxit": int(maxit),
        "max_iterations_used": int(max(max_iterations, selection_iterations)),
        "n_jobs": int(n_jobs),
        "n_genes": int(n_genes),
        "n_cells": int(n_cells),
        "max_gene_count_conservation_error": max_count_error,
    }
    return corrected_full, diagnostics


def run_spotclean(
    dnb_expr,
    dnb_coords: np.ndarray,
    dnb_labels: np.ndarray,
    candidate_bandwidths: Optional[Sequence[float]] = None,
    bleed_rate: float = 0.10,
    distal_rate: float = 0.10,
    maxit: int = 30,
    tol: float = 1.0,
    verbose: bool = False,
) -> Tuple[np.ndarray, Dict]:
    """Run cell-adapted SpotClean without empty DNB/background observations.

    Parameters
    ----------
    dnb_expr, dnb_coords, dnb_labels
        DNB-level expression, coordinates, and segmentation labels.  Labels
        below zero are ignored completely.
    candidate_bandwidths
        Gaussian bandwidth candidates in the same physical unit as coordinates.
    bleed_rate, distal_rate
        Fixed SpotClean global parameters.  They cannot be estimated from an
        all-cell input without background spots.

    Returns
    -------
    corrected, diagnostics
        Corrected gene-by-cell matrix and an explicit record of the adapted
        model assumptions.
    """
    if not 0.0 <= bleed_rate < 1.0:
        raise ValueError("bleed_rate must be in [0, 1).")
    if not 0.0 <= distal_rate <= 1.0:
        raise ValueError("distal_rate must be in [0, 1].")
    if maxit < 1:
        raise ValueError("maxit must be at least 1.")
    if tol < 0:
        raise ValueError("tol must be non-negative.")

    if candidate_bandwidths is None:
        candidate_bandwidths = (10, 20, 30, 50, 70, 100, 150, 200, 300)
    bandwidths = sorted({float(x) for x in candidate_bandwidths if float(x) > 0})
    if not bandwidths:
        raise ValueError("candidate_bandwidths must contain a positive value.")

    observed, centroids, cell_ids = _aggregate_cells(
        dnb_expr, dnb_coords, dnb_labels
    )

    best = None
    candidate_loglik = []
    for bandwidth in bandwidths:
        swap_weight = _cell_to_cell_weight(centroids, bandwidth, distal_rate)
        corrected, loglik, iterations = _spotclean_em(
            observed, swap_weight, bleed_rate, maxit=maxit, tol=tol
        )
        final_expected = (1.0 - bleed_rate) * corrected
        final_expected += bleed_rate * (corrected @ swap_weight)
        final_loglik = _poisson_loglik(observed, final_expected)
        loglik = np.append(loglik, final_loglik)
        candidate_loglik.append(final_loglik)
        if best is None or final_loglik > best[0]:
            best = (final_loglik, bandwidth, corrected, loglik, iterations)

    _, best_bandwidth, corrected, loglik, iterations = best
    count_error = float(
        np.max(np.abs(corrected.sum(axis=1) - observed.sum(axis=1)))
    )
    diagnostics = {
        "method": "CellSpotClean",
        "adaptation": "segmented cells treated as tissue spots",
        "official_r_package": False,
        "background_used": False,
        "empty_dnbs_used": False,
        "parameter_identification": "fixed rates because no background spots",
        "bleeding_rate": float(bleed_rate),
        "distal_rate": float(distal_rate),
        "contamination_bandwidth": float(best_bandwidth),
        "candidate_bandwidths": bandwidths,
        "candidate_loglik": candidate_loglik,
        "loglik": loglik.tolist(),
        "iterations": int(iterations),
        "n_cells": int(observed.shape[1]),
        "cell_ids": cell_ids,
        "max_gene_count_conservation_error": count_error,
    }
    if verbose:
        print(
            "CellSpotClean: "
            f"bandwidth={best_bandwidth:g}, bleed={bleed_rate:.3f}, "
            f"distal={distal_rate:.3f}, iterations={iterations}, "
            f"count_error={count_error:.3e}"
        )
    return corrected, diagnostics


def run_spotclean_with_background(
    dnb_expr,
    dnb_coords: np.ndarray,
    dnb_labels: np.ndarray,
    empty_bin_size: float = 25.0,
    normalize_empty_exposure: bool = True,
    candidate_bandwidths: Optional[Sequence[float]] = None,
    maxit: int = 30,
    tol: float = 1.0,
    verbose: bool = False,
) -> Tuple[np.ndarray, Dict]:
    """Run SpotClean with cells and empty bins concatenated as input spots.

    Segmented cells are tissue spots and the only expression sources. Empty
    bins are background receivers used to identify the global bleeding rate,
    distal rate, and Gaussian bandwidth. Only the corrected cell matrix is
    returned.
    """
    if empty_bin_size <= 0:
        raise ValueError("empty_bin_size must be positive.")
    if maxit < 1:
        raise ValueError("maxit must be at least 1.")
    if tol < 0:
        raise ValueError("tol must be non-negative.")
    if candidate_bandwidths is None:
        candidate_bandwidths = (10, 20, 30, 50, 70, 100, 150, 200, 300)
    bandwidths = sorted({float(x) for x in candidate_bandwidths if float(x) > 0})
    if not bandwidths:
        raise ValueError("candidate_bandwidths must contain a positive value.")

    cell_expr, cell_coords, cell_ids = _aggregate_cells(
        dnb_expr, dnb_coords, dnb_labels
    )
    empty_expr, empty_coords, empty_areas = _aggregate_empty_bins(
        dnb_expr, dnb_coords, dnb_labels, empty_bin_size
    )
    labels = np.asarray(dnb_labels)
    cell_areas = np.asarray(
        [(labels == cell_id).sum() for cell_id in cell_ids], dtype=np.float64
    )
    target_area = float(np.median(cell_areas))
    if normalize_empty_exposure:
        exposure_scale = np.divide(
            target_area,
            empty_areas,
            out=np.ones_like(empty_areas),
            where=empty_areas > 0,
        )
        empty_expr = empty_expr * exposure_scale[None, :]
    else:
        exposure_scale = np.ones_like(empty_areas)
    observed_all = np.concatenate((cell_expr, empty_expr), axis=1)
    destination_coords = np.vstack((cell_coords, empty_coords))
    n_cells = cell_expr.shape[1]
    totals = observed_all.sum(axis=0)

    candidates = []
    for bandwidth in bandwidths:
        proximal = _source_destination_weights(
            cell_coords, destination_coords, bandwidth
        )
        fitted = _estimate_contamination_parameters(totals, n_cells, proximal)
        fitted["bandwidth"] = bandwidth
        fitted["proximal_weight"] = proximal
        candidates.append(fitted)
    best = min(candidates, key=lambda result: result["rss"])

    corrected, loglik, iterations = _spotclean_background_em(
        observed_all,
        n_cells,
        best["proximal_weight"],
        best["bleeding_rate"],
        best["distal_rate"],
        maxit=maxit,
        tol=tol,
    )
    all_gene_totals = observed_all.sum(axis=1)
    count_error = float(np.max(np.abs(corrected.sum(axis=1) - all_gene_totals)))
    mean_area = float(empty_areas.mean())
    diagnostics = {
        "method": "CellSpotClean-bg",
        "adaptation": "segmented cells as tissue spots plus empty bins as background spots",
        "official_r_package": False,
        "background_used": True,
        "empty_dnbs_used": True,
        "parameter_identification": "estimated from concatenated cell and empty-bin totals",
        "bleeding_rate": best["bleeding_rate"],
        "bleeding_rate_lower_bound": best["bleeding_rate_lower_bound"],
        "distal_rate": best["distal_rate"],
        "contamination_bandwidth": float(best["bandwidth"]),
        "candidate_bandwidths": bandwidths,
        "candidate_rss": [float(result["rss"]) for result in candidates],
        "optimizer_converged": bool(best["converged"]),
        "optimizer_message": best["optimizer_message"],
        "loglik": loglik.tolist(),
        "iterations": int(iterations),
        "n_cells": int(n_cells),
        "n_empty_bins": int(empty_expr.shape[1]),
        "empty_bin_size": float(empty_bin_size),
        "empty_bin_area_mean": mean_area,
        "empty_bin_area_cv": (
            float(empty_areas.std() / mean_area) if mean_area > 0 else 0.0
        ),
        "empty_exposure_normalized": bool(normalize_empty_exposure),
        "empty_exposure_target_dnb_area": target_area,
        "empty_exposure_scale_min": float(exposure_scale.min()),
        "empty_exposure_scale_max": float(exposure_scale.max()),
        "cell_ids": cell_ids,
        "max_gene_count_conservation_error": count_error,
    }
    if verbose:
        print(
            "CellSpotClean-bg: "
            f"cells={n_cells}, empty_bins={empty_expr.shape[1]}, "
            f"bandwidth={best['bandwidth']:g}, "
            f"bleed={best['bleeding_rate']:.3f}, "
            f"distal={best['distal_rate']:.3f}, iterations={iterations}, "
            f"count_error={count_error:.3e}"
        )
    return corrected, diagnostics
