"""Vectorized synthetic spatial transcriptomics data generator.

The generator produces DNB-level expression with spatially-decaying ambient
RNA contamination, plus per-cell ground-truth expression for benchmarking.

Ambient RNA is modeled consistently with SPARKLE's cell-based correction:
  - source strength of cell c for gene g = true_expr[g, c] / cell_area[c]
  - ambient received by a target with area A_t from cell c =
      alpha_g * A_t * exp(-d / lambda) * source strength[c]
  - for a cell target, A_t = cell_area
  - for an empty DNB target, A_t = 1
"""

from typing import Dict
import numpy as np
from scipy.spatial import cKDTree


def _jittered_grid_centers(
    n_cells: int,
    width_um: float,
    height_um: float,
    cell_radius: float,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Place cell centers on a jittered grid that fits inside the FOV."""
    aspect = width_um / height_um
    cols = max(1, int(np.round(np.sqrt(n_cells * aspect))))
    rows = max(1, int(np.ceil(n_cells / cols)))

    jitter = cell_radius * 0.35
    safe_x_min = cell_radius + jitter
    safe_x_max = width_um - cell_radius - jitter
    safe_y_min = cell_radius + jitter
    safe_y_max = height_um - cell_radius - jitter

    if safe_x_min >= safe_x_max or safe_y_min >= safe_y_max:
        centers = np.column_stack([
            rng.uniform(cell_radius, width_um - cell_radius, n_cells),
            rng.uniform(cell_radius, height_um - cell_radius, n_cells),
        ])
        return centers

    x_grid = np.linspace(safe_x_min, safe_x_max, cols)
    y_grid = np.linspace(safe_y_min, safe_y_max, rows)
    xx, yy = np.meshgrid(x_grid, y_grid)
    centers = np.column_stack([xx.ravel(), yy.ravel()])[:n_cells]

    centers += rng.uniform(-jitter, jitter, centers.shape)
    centers[:, 0] = np.clip(centers[:, 0], cell_radius, width_um - cell_radius)
    centers[:, 1] = np.clip(centers[:, 1], cell_radius, height_um - cell_radius)
    return centers


def generate_synthetic_data(
    n_cells: int = 200,
    grid_width: int = 200,
    grid_height: int = 200,
    dnb_pitch: float = 0.5,
    cell_radius: float = 5.0,
    n_genes: int = 500,
    n_high_genes: int = 80,
    ambient_lambda: float = 50.0,
    ambient_alpha: float = 0.01,
    empty_fraction: float = 0.30,
    n_cell_types: int = 1,
    marker_fraction: float = 0.0,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Generate synthetic data efficiently.

    Returns a dict with keys:
        dnb_expr:    [n_genes x n_dnbs] float64, DNB-level contaminated expression.
        dnb_coords:  [n_dnbs x 2] float64, DNB spatial coordinates.
        dnb_labels:  [n_dnbs] int64, cell ID (-1 for empty DNBs).
        true_expr:   [n_genes x n_cells] float64, per-cell ground-truth expression.
        gene_is_high:[n_genes] bool, high-expression gene flag.
        cell_types:  [n_cells] int64, cell-type assignment.
        params:      dict of generation parameters (including ground-truth lambda/alpha).
    """
    rng = np.random.RandomState(seed)

    # ── 1. DNB grid ──────────────────────────────────────────────────
    x_coords = np.arange(grid_width) * dnb_pitch
    y_coords = np.arange(grid_height) * dnb_pitch
    xx, yy = np.meshgrid(x_coords, y_coords)
    all_coords = np.column_stack([xx.ravel(), yy.ravel()])
    n_dnbs = all_coords.shape[0]
    width_um = grid_width * dnb_pitch
    height_um = grid_height * dnb_pitch

    # ── 2. Cell centers on jittered grid ─────────────────────────────
    cell_centers = _jittered_grid_centers(n_cells, width_um, height_um, cell_radius, rng)

    # Assign DNBs to nearest cell center within radius
    dnb_labels = np.full(n_dnbs, -1, dtype=np.int64)
    chunk = 2000
    for i in range(0, n_dnbs, chunk):
        end = min(i + chunk, n_dnbs)
        coords_chunk = all_coords[i:end]
        diff = coords_chunk[:, None, :] - cell_centers[None, :, :]
        dists = np.linalg.norm(diff, axis=2)
        nearest = np.argmin(dists, axis=1)
        min_dists = dists[np.arange(len(coords_chunk)), nearest]
        valid = min_dists <= cell_radius
        dnb_labels[i:end][valid] = nearest[valid]

    # ── 3. Empty cells ───────────────────────────────────────────────
    if empty_fraction > 0:
        n_empty_cells = max(1, int(round(n_cells * empty_fraction)))
        empty_cell_indices = rng.choice(n_cells, n_empty_cells, replace=False)
        for c in empty_cell_indices:
            dnb_labels[dnb_labels == c] = -1
    else:
        empty_cell_indices = np.array([], dtype=np.int64)

    # Compact cell IDs so that only cells with DNBs are kept (matches SPARKLE output)
    kept_cells = sorted(set(int(x) for x in dnb_labels[dnb_labels >= 0]))
    n_kept = len(kept_cells)
    old_to_new = {old_id: i for i, old_id in enumerate(kept_cells)}
    dnb_labels = np.array([old_to_new.get(l, -1) for l in dnb_labels], dtype=np.int64)
    cell_centers = cell_centers[kept_cells]

    # ── 4. Cell types ────────────────────────────────────────────────
    cell_types = np.zeros(n_kept, dtype=np.int64)
    if n_cell_types > 1:
        old_cell_types = np.zeros(n_cells, dtype=np.int64)
        cells_per_type = np.array_split(np.arange(n_cells), n_cell_types)
        for t, idxs in enumerate(cells_per_type):
            old_cell_types[idxs] = t
        cell_types = old_cell_types[kept_cells]

    # ── 5. Ground-truth per-cell expression ──────────────────────────
    gene_is_high = np.zeros(n_genes, dtype=bool)
    gene_is_high[:n_high_genes] = True

    true_expr = np.zeros((n_genes, n_kept), dtype=np.float64)

    # Background expression for all genes (low Poisson)
    true_expr[:] = rng.poisson(2.0, (n_genes, n_kept)).astype(np.float64)

    if n_cell_types <= 1:
        # Shared high-expression program
        true_expr[:n_high_genes] = rng.poisson(50.0, (n_high_genes, n_kept)).astype(np.float64)
    else:
        # Each cell type expresses a distinct subset of high genes
        high_genes_per_type = np.array_split(np.arange(n_high_genes), n_cell_types)
        for t in range(n_cell_types):
            gidx = high_genes_per_type[t]
            cells_t = np.where(cell_types == t)[0]
            true_expr[gidx[:, None], cells_t] = rng.poisson(
                50.0, (len(gidx), len(cells_t))
            ).astype(np.float64)

    # Marker genes: additional strong expression in one cell type only
    n_markers = int(round(n_high_genes * marker_fraction))
    if n_markers > 0 and n_cell_types > 1:
        marker_pool = np.arange(n_high_genes)
        rng.shuffle(marker_pool)
        for t in range(n_cell_types):
            start = t * n_markers
            end = min(start + n_markers, len(marker_pool))
            if start >= len(marker_pool):
                break
            marker_genes = marker_pool[start:end]
            cells_t = np.where(cell_types == t)[0]
            true_expr[marker_genes[:, None], cells_t] += rng.poisson(
                80.0, (len(marker_genes), len(cells_t))
            ).astype(np.float64)

    # ── 6. DNB-level clean expression (Poisson per-DNB from cell total)
    dnb_expr_clean = np.zeros((n_genes, n_dnbs), dtype=np.float64)
    cell_dnb_counts = np.bincount(dnb_labels[dnb_labels >= 0], minlength=n_kept)

    nonzero_cells = np.where(cell_dnb_counts > 0)[0]
    for c in nonzero_cells:
        mask = dnb_labels == c
        n_dnbs_c = int(mask.sum())
        if n_dnbs_c == 0:
            continue
        per_dnb_rate = true_expr[:, c] / n_dnbs_c
        per_dnb_rate = np.maximum(per_dnb_rate, 0.01)
        dnb_expr_clean[:, mask] = rng.poisson(per_dnb_rate[:, None], (n_genes, n_dnbs_c)).astype(np.float64)

    # ── 7. Ambient RNA injection (SPARKLE-consistent model) ──────────
    dnb_expr = dnb_expr_clean.copy()
    cell_areas = cell_dnb_counts.astype(np.float64)
    cell_areas_safe = np.maximum(cell_areas, 1.0)

    cell_tree = cKDTree(cell_centers)
    max_neigh_dist = 3 * ambient_lambda

    # Cell-to-cell ambient contributions
    cell_dist_coo = cell_tree.sparse_distance_matrix(cell_tree, max_neigh_dist, output_type="coo_matrix")
    cell_dists = cell_dist_coo.data
    cell_src = cell_dist_coo.col
    cell_tgt = cell_dist_coo.row
    same_cell_mask = cell_src != cell_tgt
    cell_dists = cell_dists[same_cell_mask]
    cell_src = cell_src[same_cell_mask]
    cell_tgt = cell_tgt[same_cell_mask]
    weights = np.exp(-cell_dists / ambient_lambda)
    weights[weights < 0.001] = 0.0

    for g in range(n_genes):
        if not gene_is_high[g]:
            continue
        alpha_g = ambient_alpha * (0.5 + rng.random())

        # source rate = true_expr / area for source cells
        source_rate = true_expr[g, cell_src] / cell_areas_safe[cell_src]
        contrib_per_target = alpha_g * weights * source_rate * cell_areas_safe[cell_tgt]
        ambient_per_cell = np.bincount(cell_tgt, weights=contrib_per_target, minlength=n_kept)

        # Distribute ambient uniformly over each cell's DNBs
        for c in nonzero_cells:
            n_dnbs_c = cell_dnb_counts[c]
            if n_dnbs_c == 0:
                continue
            mask = dnb_labels == c
            per_dnb_ambient = ambient_per_cell[c] / n_dnbs_c
            dnb_expr[g, mask] += rng.poisson(max(per_dnb_ambient, 0.0), int(n_dnbs_c)).astype(np.float64)

    # Empty DNB ambient contributions
    empty_mask = dnb_labels < 0
    empty_coords = all_coords[empty_mask]
    if empty_coords.shape[0] > 0:
        empty_tree = cKDTree(empty_coords)
        empty_dist_coo = empty_tree.sparse_distance_matrix(cell_tree, max_neigh_dist, output_type="coo_matrix")
        empty_dists = empty_dist_coo.data
        empty_src = empty_dist_coo.col  # cell index
        empty_tgt_local = empty_dist_coo.row  # index within empty_coords
        weights_e = np.exp(-empty_dists / ambient_lambda)
        weights_e[weights_e < 0.001] = 0.0

        empty_global_idx = np.where(empty_mask)[0]
        for g in range(n_genes):
            if not gene_is_high[g]:
                continue
            alpha_g = ambient_alpha * (0.5 + rng.random())

            source_rate = true_expr[g, empty_src] / cell_areas_safe[empty_src]
            contrib = alpha_g * weights_e * source_rate  # A_t = 1 for a single empty DNB
            ambient_per_empty = np.bincount(empty_tgt_local, weights=contrib, minlength=empty_coords.shape[0])
            dnb_expr[g, empty_global_idx] += rng.poisson(np.maximum(ambient_per_empty, 0)).astype(np.float64)

    return {
        "dnb_expr": dnb_expr,
        "dnb_coords": all_coords,
        "dnb_labels": dnb_labels,
        "true_expr": true_expr,
        "gene_is_high": gene_is_high,
        "cell_types": cell_types,
        "params": {
            "n_cells": n_kept,
            "n_cells_requested": n_cells,
            "grid_width": grid_width,
            "grid_height": grid_height,
            "dnb_pitch": dnb_pitch,
            "cell_radius": cell_radius,
            "n_genes": n_genes,
            "n_high_genes": n_high_genes,
            "ambient_lambda": ambient_lambda,
            "ambient_alpha": ambient_alpha,
            "empty_fraction": empty_fraction,
            "n_cell_types": n_cell_types,
            "marker_fraction": marker_fraction,
            "empty_cell_indices": empty_cell_indices,
            "seed": seed,
        },
    }
