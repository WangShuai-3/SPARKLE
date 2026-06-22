"""Optimized synthetic data generator for testing SPARKLE.

Uses vectorized operations where possible for speed.
"""

import numpy as np
from typing import Tuple


def generate_synthetic_data_fast(
    n_cells: int = 50,
    grid_width: int = 80,
    grid_height: int = 80,
    dnb_pitch: float = 0.5,
    cell_radius: float = 5.0,
    n_genes: int = 300,
    n_high_genes: int = 80,
    ambient_lambda: float = 50.0,
    ambient_alpha: float = 0.01,
    empty_fraction: float = 0.3,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic data efficiently.

    Returns:
        dnb_expr: [genes × DNBs] expression.
        dnb_coords: [DNBs × 2] coordinates.
        dnb_labels: [DNBs] cell IDs (-1 for empty).
        true_expr: [genes × cells] ground truth.
        gene_is_high: [genes] boolean.
    """
    rng = np.random.RandomState(seed)

    # Generate DNB grid
    x_coords = np.arange(grid_width) * dnb_pitch
    y_coords = np.arange(grid_height) * dnb_pitch
    xx, yy = np.meshgrid(x_coords, y_coords)
    all_coords = np.column_stack([xx.ravel(), yy.ravel()])
    n_dnbs = all_coords.shape[0]
    width_um = grid_width * dnb_pitch
    height_um = grid_height * dnb_pitch

    # Generate cell centers on a jittered grid
    cols = int(np.sqrt(n_cells * width_um / height_um))
    rows = int(n_cells / cols)
    cx = np.linspace(cell_radius, width_um - cell_radius, cols)
    cy = np.linspace(cell_radius, height_um - cell_radius, rows)
    cx_grid, cy_grid = np.meshgrid(cx, cy)
    cell_centers = np.column_stack([cx_grid.ravel(), cy_grid.ravel()])
    cell_centers = cell_centers[:n_cells]
    # Jitter
    cell_centers += rng.uniform(-cell_radius * 0.3, cell_radius * 0.3, cell_centers.shape)

    # Assign DNBs to cells: each DNB → nearest cell within radius
    dnb_labels = np.full(n_dnbs, -1, dtype=np.int64)
    for i in range(0, n_dnbs, 1000):
        end = min(i + 1000, n_dnbs)
        chunk = all_coords[i:end]
        # Compute distances to all cell centers
        dists = np.linalg.norm(chunk[:, None, :] - cell_centers[None, :, :], axis=2)
        nearest = np.argmin(dists, axis=1)
        min_dists = dists[np.arange(len(chunk)), nearest]
        valid = min_dists <= cell_radius
        dnb_labels[i:end][valid] = nearest[valid]

    # Make some cells empty
    n_empty_cells = int(n_cells * empty_fraction)
    empty_cell_indices = rng.choice(n_cells, n_empty_cells, replace=False)
    for c in empty_cell_indices:
        dnb_labels[dnb_labels == c] = -1

    # Generate ground truth per-cell expression (Poisson)
    gene_is_high = np.zeros(n_genes, dtype=bool)
    gene_is_high[:n_high_genes] = True

    true_expr = rng.poisson(5.0, (n_genes, n_cells)).astype(np.float64)
    # High-expr genes: multiply by higher factor
    true_expr[:n_high_genes] = rng.poisson(50.0, (n_high_genes, n_cells)).astype(np.float64)

    # Generate DNB-level clean expression (Poisson per-DNB from cell total)
    dnb_expr_clean = np.zeros((n_genes, n_dnbs), dtype=np.float64)
    cell_dnb_counts = np.bincount(dnb_labels[dnb_labels >= 0], minlength=n_cells)

    for g in range(n_genes):
        for c in range(n_cells):
            n_dnbs_c = cell_dnb_counts[c]
            if n_dnbs_c == 0:
                continue
            per_dnb_rate = true_expr[g, c] / n_dnbs_c
            mask = dnb_labels == c
            dnb_expr_clean[g, mask] = rng.poisson(max(per_dnb_rate, 0.01), int(n_dnbs_c)).astype(np.float64)

    # Add ambient RNA efficiently using a KDTree
    from scipy.spatial import cKDTree
    tree = cKDTree(all_coords)

    dnb_expr = dnb_expr_clean.copy()
    cell_dnb_mask = dnb_labels >= 0

    # For each gene, add ambient
    for g in range(n_genes):
        if not gene_is_high[g]:
            continue
        alpha_g = ambient_alpha * (0.5 + rng.random())

        # Cell DNB source strengths (per-DNB clean expression)
        source = dnb_expr_clean[g, cell_dnb_mask]

        # For empty DNBs: sum over all nearby cell DNBs
        empty_mask = dnb_labels < 0
        empty_coords = all_coords[empty_mask]

        if empty_coords.shape[0] == 0:
            continue

        cell_coords = all_coords[cell_dnb_mask]

        # Find neighbors of each empty DNB among cell DNBs
        # Use query_ball_point for each empty DNB
        for idx_in_empty, empty_idx in enumerate(np.where(empty_mask)[0]):
            empty_pos = all_coords[empty_idx]
            # Find cell DNBs within 3*ambient_lambda
            neighbors = tree.query_ball_point(empty_pos, 3 * ambient_lambda)
            # Filter to cell DNBs only
            cell_neighbors = [n for n in neighbors if cell_dnb_mask[n]]
            if not cell_neighbors:
                continue

            for nbr in cell_neighbors:
                d = np.linalg.norm(empty_pos - all_coords[nbr])
                w = np.exp(-d / ambient_lambda)
                if w < 0.001:
                    continue
                contribution = alpha_g * w * dnb_expr_clean[g, nbr]
                dnb_expr[g, empty_idx] += rng.poisson(max(contribution, 0))

        # Also add ambient to cell DNBs from other cells
        for idx_in_cell, cell_idx in enumerate(np.where(cell_dnb_mask)[0]):
            their_cell = dnb_labels[cell_idx]
            cell_pos = all_coords[cell_idx]
            neighbors = tree.query_ball_point(cell_pos, 3 * ambient_lambda)
            cell_neighbors = [n for n in neighbors if cell_dnb_mask[n] and dnb_labels[n] != their_cell]
            for nbr in cell_neighbors:
                d = np.linalg.norm(cell_pos - all_coords[nbr])
                w = np.exp(-d / ambient_lambda)
                if w < 0.001:
                    continue
                contribution = alpha_g * w * dnb_expr_clean[g, nbr]
                dnb_expr[g, cell_idx] += rng.poisson(max(contribution, 0))

    return dnb_expr, all_coords, dnb_labels, true_expr, gene_is_high
