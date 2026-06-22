"""Spatial SoupX baseline: SoupX enhanced with SPARKLE's spatial kernel.

Regular SoupX: ambient = ρ × soup_profile × nUMI (global, uniform)
Spatial SoupX: ambient = ρ × Σ w(d) × source (spatially weighted)

Uses SPARKLE's binning and spatial graph, but estimates a single global
contamination fraction ρ instead of per-gene α.
"""

import numpy as np
from scipy.sparse import csr_matrix
from typing import Dict, Tuple, Optional

from stambient.spatial import build_spatial_graph, DistanceMetric
from stambient.binning import dnb_to_bins


def run_spatial_soupx(
    dnb_expr: np.ndarray,
    dnb_coords: np.ndarray,
    dnb_labels: np.ndarray,
    bin_size: int = 50,
    max_radius: float = 200.0,
    lambda_grid: list = None,
    metric: DistanceMetric = "exponential",
    verbose: bool = False,
) -> Tuple[np.ndarray, float, float]:
    """Run Spatial SoupX: SoupX with SPARKLE's spatial kernel.

    Args:
        dnb_expr: [genes × DNBs] DNB-level expression.
        dnb_coords: [DNBs × 2] coordinates.
        dnb_labels: [DNBs] cell IDs (-1 for empty).
        bin_size: DNBs per bin edge.
        max_radius: max neighbor search radius.
        lambda_grid: λ candidates for grid search.
        metric: distance weight function.

    Returns:
        corrected_expr: [genes × n_cells] corrected per-cell expression.
        rho: global contamination fraction.
        best_lam: estimated λ.
    """
    if lambda_grid is None:
        lambda_grid = [10, 20, 30, 50, 70, 100, 150, 200]

    # ── 1. Binning (same as SPARKLE) ──────────────────────────
    (Y_cell, Y_empty, n_cell, n_empty, bin_coords,
     bin_cell_assignment, classification, n_total) = dnb_to_bins(
        dnb_expr, dnb_coords, dnb_labels, bin_size
    )

    if hasattr(Y_cell, 'toarray'):
        Y_cell_d = Y_cell.toarray()
    else:
        Y_cell_d = np.asarray(Y_cell.todense())

    if hasattr(Y_empty, 'toarray'):
        Y_empty_d = Y_empty.toarray()
    else:
        Y_empty_d = np.asarray(Y_empty.todense())

    n_genes = Y_cell.shape[0]
    n_bins = Y_cell.shape[1]
    empty_bin_mask = n_empty > 0
    cell_mask = n_cell > 0

    # ── 2. λ grid search (minimize empty-bin RSS with global ρ) ──
    best_lam = lambda_grid[0]
    best_rss = np.inf

    for lam in lambda_grid:
        W = build_spatial_graph(bin_coords, max_radius, lam, metric)
        total_rss = 0.0

        for g in range(n_genes):
            source = np.divide(Y_cell_d[g], n_cell,
                               out=np.zeros(n_bins), where=cell_mask)
            weighted_sum = W.dot(source)
            N_gb = n_empty * weighted_sum
            y_obs = Y_empty_d[g][empty_bin_mask]
            n_obs = N_gb[empty_bin_mask]

            denom = (n_obs ** 2).sum()
            if denom == 0:
                continue
            rho_g = max((y_obs * n_obs).sum() / denom, 0.0)
            residuals = y_obs - rho_g * n_obs
            total_rss += (residuals ** 2).sum()

        if total_rss < best_rss:
            best_rss = total_rss
            best_lam = lam

    if verbose:
        print(f"  Optimal λ: {best_lam:.1f} μm")

    # ── 3. Estimate global ρ with best λ ─────────────────────
    W = build_spatial_graph(bin_coords, max_radius, best_lam, metric)

    rho_values = []
    for g in range(n_genes):
        source = np.divide(Y_cell_d[g], n_cell,
                           out=np.zeros(n_bins), where=cell_mask)
        weighted_sum = W.dot(source)
        N_gb = n_empty * weighted_sum
        y_obs = Y_empty_d[g][empty_bin_mask]
        n_obs = N_gb[empty_bin_mask]

        denom = (n_obs ** 2).sum()
        if denom == 0 or y_obs.sum() == 0:
            continue
        rho_g = max((y_obs * n_obs).sum() / denom, 0.0)
        rho_values.append(rho_g)

    rho = float(np.median(rho_values)) if rho_values else 0.01
    rho = max(0.001, min(rho, 0.8))

    if verbose:
        print(f"  Global ρ: {rho:.4f} (from {len(rho_values)} genes)")

    # ── 4. Correct expression ──────────────────────────────────
    # Collect all unique cell IDs
    all_cell_ids = set()
    for d in bin_cell_assignment.values():
        all_cell_ids.update(d.keys())
    cell_id_list = sorted(all_cell_ids)
    cell_id_to_idx = {cid: i for i, cid in enumerate(cell_id_list)}
    n_cells = len(cell_id_list)

    corrected = np.zeros((n_genes, n_cells), dtype=np.float64)

    for g in range(n_genes):
        source = np.divide(Y_cell_d[g], n_cell,
                           out=np.zeros(n_bins), where=cell_mask)
        neighbor_contrib = W.dot(source)
        ambient = rho * n_cell * neighbor_contrib
        s_gb = np.maximum(Y_cell_d[g] - ambient, 0.0)

        for b in range(n_bins):
            if not cell_mask[b] or n_cell[b] == 0 or s_gb[b] <= 0:
                continue
            assignment = bin_cell_assignment[b]
            for cid, n_dnb in assignment.items():
                cidx = cell_id_to_idx[cid]
                corrected[g, cidx] += s_gb[b] * (n_dnb / n_cell[b])

    return corrected, rho, best_lam
