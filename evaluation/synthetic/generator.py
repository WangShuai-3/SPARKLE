"""Physically-motivated synthetic spatial transcriptomics data generator.

Design goals:
  1. Mimic real Stereo-seq/Visium HD data: cells occupy part of the FOV,
     inter-cellular space is empty.
  2. Match SPARKLE's cell-based diffusion model, so the correction target is
     exactly the model SPARKLE assumes.
  3. Output is DNB-level data compatible with the Axolotl pipeline:
       dnb_expr  : [genes x DNBs]
       dnb_coords: [DNBs x 2]
       dnb_labels: [DNBs] cell ID (-1 for empty DNBs)

Ambient model (SPARKLE-consistent):
  - Cell c source strength for gene g:  s_{gc} = true_expr[g, c] / A_c
    where A_c is the observed number of DNBs assigned to cell c.
  - Empty DNB e receives ambient:       a_{ge} = alpha_g * sum_c w(d_{e,c}) * s_{gc}
  - Cell c receives total ambient:      A_c * alpha_g * sum_{c'!=c} w(d_{c,c'}) * s_{gc'}
    distributed uniformly over its DNBs.
  - Weight: w(d) = exp(-d / lambda), truncated at 3*lambda.

Cells are placed on a jittered grid and occupy circular regions with variable
radii. The requested `empty_fraction` is enforced globally: if natural
inter-cellular space already exceeds the target, the uncovered DNBs closest to
cell centres are assigned to those cells; otherwise covered DNBs are randomly
removed to create empty space.
"""

from typing import Dict, Optional, Tuple
import numpy as np
from scipy.spatial import cKDTree


def _place_cell_centers(
    n_cells: int,
    width_um: float,
    height_um: float,
    mean_radius: float,
    rng: np.random.RandomState,
    n_cell_types: int = 1,
    cluster_strength: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Place cell centers on a jittered grid with optional type clusters."""
    aspect = width_um / height_um
    cols = max(1, int(np.round(np.sqrt(n_cells * aspect))))
    rows = max(1, int(np.ceil(n_cells / cols)))

    # Grid spacing
    x_grid = np.linspace(mean_radius * 2, width_um - mean_radius * 2, cols)
    y_grid = np.linspace(mean_radius * 2, height_um - mean_radius * 2, rows)
    xx, yy = np.meshgrid(x_grid, y_grid)
    centers = np.column_stack([xx.ravel(), yy.ravel()])[:n_cells]

    # Random jitter (up to 35% of mean radius)
    jitter = mean_radius * 0.35
    centers += rng.uniform(-jitter, jitter, centers.shape)
    centers[:, 0] = np.clip(centers[:, 0], mean_radius, width_um - mean_radius)
    centers[:, 1] = np.clip(centers[:, 1], mean_radius, height_um - mean_radius)

    # Cell types: random with optional spatial clustering
    cell_types = np.zeros(n_cells, dtype=np.int64)
    if n_cell_types > 1:
        if cluster_strength > 0:
            # Cluster centers in space, assign nearby cells to the same type
            type_centers = rng.uniform(
                low=[mean_radius * 2, mean_radius * 2],
                high=[width_um - mean_radius * 2, height_um - mean_radius * 2],
                size=(n_cell_types, 2),
            )
            tree = cKDTree(type_centers)
            _, cell_types = tree.query(centers)
        else:
            cell_types = np.arange(n_cells, dtype=np.int64) % n_cell_types
        rng.shuffle(cell_types)

    return centers, cell_types


def _assign_dnbs_to_cells(
    all_coords: np.ndarray,
    cell_centers: np.ndarray,
    cell_radii: np.ndarray,
) -> np.ndarray:
    """Assign each DNB to the nearest cell center within that cell's radius."""
    n_dnbs = all_coords.shape[0]
    dnb_labels = np.full(n_dnbs, -1, dtype=np.int64)
    tree = cKDTree(cell_centers)

    # Query nearest cell center for each DNB
    dists, nearest = tree.query(all_coords, k=1)
    valid = dists <= cell_radii[nearest]
    dnb_labels[valid] = nearest[valid]
    return dnb_labels


def generate_synthetic_data(
    n_cells: int = 200,
    grid_width: int = 200,
    grid_height: int = 200,
    dnb_pitch: float = 0.5,
    cell_radius: float = 5.0,
    cell_radius_cv: float = 0.2,
    n_genes: int = 500,
    n_high_genes: int = 80,
    ambient_lambda: float = 50.0,
    ambient_alpha: float = 0.01,
    empty_fraction: float = 0.30,
    n_cell_types: int = 1,
    marker_fraction: float = 0.0,
    cluster_strength: float = 0.5,
    dropout_rate: float = 0.0,
    bg_rate_range: Tuple[float, float] = (0.1, 10.0),
    high_rate_range: Tuple[float, float] = (20.0, 100.0),
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Generate synthetic data.

    Args:
        n_cells: Target number of cells to place in the FOV.
        grid_width/height: Number of DNBs along each axis.
        dnb_pitch: Physical spacing between DNBs (µm).
        cell_radius: Mean cell radius (µm).
        cell_radius_cv: Coefficient of variation of cell radii.
        n_genes: Total number of genes.
        n_high_genes: Number of high-expression genes.
        ambient_lambda: Spatial decay length of ambient RNA (µm).
        ambient_alpha: Global ambient leakage coefficient.
        empty_fraction: Target fraction of DNBs that are empty (inter-cellular
            space). The generator enforces this globally by resurrecting the
            uncovered DNBs closest to a cell centre when there is too much
            natural empty space, or by randomly removing covered DNBs when
            there is too little.
        n_cell_types: Number of cell types.
        marker_fraction: Fraction of high genes that are cell-type-specific markers.
        cluster_strength: If >0 and n_cell_types>1, spatially cluster cell types.
        dropout_rate: Fraction of UMIs randomly lost (binomial thinning),
            simulating capture loss. Applied to the clean and ambient parts
            independently; the ground truth is then defined as the observed
            (thinned) clean expression. Default 0 (no dropout).
        bg_rate_range: (min, max) of the log-spaced per-gene background
            expression gradient; most genes sit near min (very low).
        high_rate_range: (min, max) of the log-spaced per-gene gradient of
            the high-expression program in owning cells.
        seed: Random seed.

    Returns:
        dict with keys:
            dnb_expr:     [n_genes x n_dnbs] contaminated DNB-level expression.
            dnb_coords:   [n_dnbs x 2] coordinates.
            dnb_labels:   [n_dnbs] cell ID (-1 for empty).
            true_expr:    [n_genes x n_kept_cells] ground-truth per-cell expression.
            gene_is_high: [n_genes] bool.
            cell_types:   [n_kept_cells] int.
            true_alpha:   [n_genes] per-gene ambient coefficient.
            true_lambda:  float, ground-truth lambda.
            params:       dict of generation parameters.
    """
    rng = np.random.RandomState(seed)
    if not 0.0 <= dropout_rate < 1.0:
        raise ValueError(
            f"dropout_rate must be in [0, 1); got {dropout_rate}"
        )

    # ── 1. DNB grid ──────────────────────────────────────────────────
    x_coords = np.arange(grid_width) * dnb_pitch
    y_coords = np.arange(grid_height) * dnb_pitch
    xx, yy = np.meshgrid(x_coords, y_coords)
    all_coords = np.column_stack([xx.ravel(), yy.ravel()])
    n_dnbs = all_coords.shape[0]
    width_um = grid_width * dnb_pitch
    height_um = grid_height * dnb_pitch

    # ── 2. Cell centers and types ────────────────────────────────────
    cell_centers, cell_types = _place_cell_centers(
        n_cells, width_um, height_um, cell_radius, rng,
        n_cell_types=n_cell_types, cluster_strength=cluster_strength,
    )

    # Variable cell radii
    cell_radii = rng.lognormal(
        mean=np.log(cell_radius),
        sigma=cell_radius_cv,
        size=n_cells,
    )
    cell_radii = np.clip(cell_radii, cell_radius * 0.5, cell_radius * 1.5)

    # ── 3. Assign DNBs to cells ──────────────────────────────────────
    dnb_labels = _assign_dnbs_to_cells(all_coords, cell_centers, cell_radii)
    n_covered = (dnb_labels >= 0).sum()

    # ── 4. Create empty DNBs to match requested empty_fraction ────────
    # We enforce the requested global empty_fraction.  DNBs not covered by any
    # cell are natural empty space; if there are too many we resurrect the
    # uncovered DNBs closest to a cell centre so that dense scenarios can really
    # reach the intended low empty fraction.  If there are too few natural empty
    # DNBs we randomly remove covered DNBs to model inter-cellular space.
    if n_covered > 0:
        target_empty = int(round(n_dnbs * empty_fraction))
        target_empty = max(0, min(target_empty, n_dnbs - 1))
        already_empty = n_dnbs - n_covered

        if already_empty > target_empty:
            # Too many uncovered DNBs: assign the closest ones to their nearest
            # cell so that the final empty fraction matches the scenario.
            n_resurrect = already_empty - target_empty
            uncovered_idx = np.where(dnb_labels < 0)[0]
            cell_tree_for_empty = cKDTree(cell_centers)
            dists, nearest = cell_tree_for_empty.query(
                all_coords[uncovered_idx], k=1
            )
            # Prefer resurrecting DNBs that are close to a cell centre.
            order = np.argsort(dists)
            resurrect_idx = uncovered_idx[order[:n_resurrect]]
            dnb_labels[resurrect_idx] = nearest[order[:n_resurrect]]
        elif already_empty < target_empty:
            need_to_remove = target_empty - already_empty
            covered_idx = np.where(dnb_labels >= 0)[0]
            need_to_remove = min(need_to_remove, len(covered_idx))
            if need_to_remove > 0:
                remove_idx = rng.choice(
                    covered_idx, need_to_remove, replace=False
                )
                dnb_labels[remove_idx] = -1

    # Compact cell IDs: only cells with at least one DNB are kept
    kept_cells = sorted(set(int(x) for x in dnb_labels[dnb_labels >= 0]))
    n_kept = len(kept_cells)
    old_to_new = {old_id: i for i, old_id in enumerate(kept_cells)}
    dnb_labels = np.array([old_to_new.get(l, -1) for l in dnb_labels], dtype=np.int64)
    cell_centers = cell_centers[kept_cells]
    cell_types = cell_types[kept_cells]
    cell_radii = cell_radii[kept_cells]

    # ── 5. Ground-truth per-cell expression ──────────────────────────
    gene_is_high = np.zeros(n_genes, dtype=bool)
    gene_is_high[:n_high_genes] = True

    true_expr = np.zeros((n_genes, n_kept), dtype=np.float64)

    # Background expression for all genes: per-gene baseline rates follow a
    # log-spaced gradient, so most genes are expressed very low and only a
    # few are abundant (long-tailed, like real data).
    bg_rates = np.logspace(
        np.log10(bg_rate_range[1]), np.log10(bg_rate_range[0]), n_genes
    )
    rng.shuffle(bg_rates)
    true_expr[:] = rng.poisson(bg_rates[:, None], (n_genes, n_kept)).astype(np.float64)

    # High-expression program: rates likewise log-graded across high genes
    # (shuffled so no cell type systematically gets the highest genes).
    high_rates = np.logspace(
        np.log10(high_rate_range[1]), np.log10(high_rate_range[0]), n_high_genes
    )
    rng.shuffle(high_rates)
    if n_cell_types <= 1:
        # Shared high-expression program
        true_expr[:n_high_genes] = rng.poisson(
            high_rates[:, None], (n_high_genes, n_kept)
        ).astype(np.float64)
    else:
        # Each cell type expresses a distinct subset of high genes
        high_genes_per_type = np.array_split(np.arange(n_high_genes), n_cell_types)
        for t in range(n_cell_types):
            gidx = high_genes_per_type[t]
            cells_t = np.where(cell_types == t)[0]
            true_expr[gidx[:, None], cells_t] = rng.poisson(
                high_rates[gidx][:, None], (len(gidx), len(cells_t))
            ).astype(np.float64)

    # Marker genes: strong, cell-type-specific expression.  Markers are
    # absent from every other cell type (exact zero ground truth), giving a
    # present-vs-absent frequency contrast that marker-based detection
    # methods (e.g. SoupX quickMarkers) can pick up.
    marker_absent = np.zeros((n_genes, n_kept), dtype=bool)
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
            not_t = np.where(cell_types != t)[0]
            true_expr[marker_genes[:, None], not_t] = 0.0
            marker_absent[marker_genes[:, None], not_t] = True

    # ── 6. DNB-level clean expression ────────────────────────────────
    dnb_expr_clean = np.zeros((n_genes, n_dnbs), dtype=np.float64)
    cell_dnb_counts = np.bincount(dnb_labels[dnb_labels >= 0], minlength=n_kept)
    cell_areas = cell_dnb_counts.astype(np.float64)
    cell_areas_safe = np.maximum(cell_areas, 1.0)

    nonzero_cells = np.where(cell_dnb_counts > 0)[0]
    for c in nonzero_cells:
        mask = dnb_labels == c
        n_dnbs_c = int(mask.sum())
        if n_dnbs_c == 0:
            continue
        per_dnb_rate = true_expr[:, c] / n_dnbs_c
        per_dnb_rate = np.maximum(per_dnb_rate, 0.01)
        # Keep absent markers at exactly zero: the 0.01 floor would
        # otherwise re-introduce ~2 background counts per cell.
        per_dnb_rate[marker_absent[:, c]] = 0.0
        dnb_expr_clean[:, mask] = rng.poisson(
            per_dnb_rate[:, None], (n_genes, n_dnbs_c)
        ).astype(np.float64)

    # ── 7. Ambient RNA injection (SPARKLE-consistent) ────────────────
    dnb_expr = dnb_expr_clean.copy()
    max_neigh_dist = 3 * ambient_lambda
    cell_tree = cKDTree(cell_centers)

    # Per-gene ambient coefficients
    true_alpha = np.zeros(n_genes, dtype=np.float64)
    true_alpha[:] = ambient_alpha * (0.5 + rng.random(n_genes))

    # Pre-compute cell-to-cell distances and weights
    cell_dist_coo = cell_tree.sparse_distance_matrix(
        cell_tree, max_neigh_dist, output_type="coo_matrix"
    )
    cell_src = cell_dist_coo.col
    cell_tgt = cell_dist_coo.row
    same_cell_mask = cell_src != cell_tgt
    cell_dists = cell_dist_coo.data[same_cell_mask]
    cell_src = cell_src[same_cell_mask]
    cell_tgt = cell_tgt[same_cell_mask]
    cell_weights = np.exp(-cell_dists / ambient_lambda)
    cell_weights[cell_weights < 0.001] = 0.0

    # Pre-compute empty DNB to cell distances and weights
    empty_mask = dnb_labels < 0
    empty_coords = all_coords[empty_mask]
    empty_global_idx = np.where(empty_mask)[0]

    if empty_coords.shape[0] > 0:
        empty_tree = cKDTree(empty_coords)
        empty_dist_coo = empty_tree.sparse_distance_matrix(
            cell_tree, max_neigh_dist, output_type="coo_matrix"
        )
        empty_src = empty_dist_coo.col
        empty_tgt_local = empty_dist_coo.row
        empty_dists = empty_dist_coo.data
        empty_weights = np.exp(-empty_dists / ambient_lambda)
        empty_weights[empty_weights < 0.001] = 0.0
    else:
        empty_src = np.array([], dtype=np.int64)
        empty_tgt_local = np.array([], dtype=np.int64)
        empty_weights = np.array([], dtype=np.float64)

    for g in range(n_genes):
        if not gene_is_high[g]:
            continue

        alpha_g = true_alpha[g]
        source_rate = true_expr[g, :] / cell_areas_safe

        # Cell-to-cell ambient: total ambient received by each cell
        contrib_to_cell = alpha_g * cell_weights * source_rate[cell_src]
        ambient_total_cell = np.bincount(
            cell_tgt, weights=contrib_to_cell, minlength=n_kept
        )

        # Distribute uniformly over each cell's DNBs.
        # ambient_total_cell[c] is already the per-DNB ambient rate for cell c.
        for c in nonzero_cells:
            n_dnbs_c = cell_dnb_counts[c]
            if n_dnbs_c == 0:
                continue
            mask = dnb_labels == c
            per_dnb_ambient = ambient_total_cell[c]
            dnb_expr[g, mask] += rng.poisson(
                max(per_dnb_ambient, 0.0), int(n_dnbs_c)
            ).astype(np.float64)

        # Empty DNB ambient
        if empty_coords.shape[0] > 0:
            contrib_to_empty = alpha_g * empty_weights * source_rate[empty_src]
            ambient_per_empty = np.bincount(
                empty_tgt_local, weights=contrib_to_empty, minlength=empty_coords.shape[0]
            )
            dnb_expr[g, empty_global_idx] += rng.poisson(
                np.maximum(ambient_per_empty, 0.0)
            ).astype(np.float64)

    # ── 8. Dropout (random UMI loss) ─────────────────────────────────
    # Real capture is lossy: abundant true molecules survive detection while
    # sparse leaked counts often drop to zero.  Thin the clean and ambient
    # parts independently (binomial per UMI) so that the ground truth can be
    # defined as the *observed* clean expression; the RMSE evaluation then
    # compares methods against a recoverable target.
    if dropout_rate > 0:
        keep_prob = 1.0 - dropout_rate
        clean_observed = rng.binomial(
            dnb_expr_clean.astype(np.int64), keep_prob
        ).astype(np.float64)
        ambient_observed = rng.binomial(
            (dnb_expr - dnb_expr_clean).astype(np.int64), keep_prob
        ).astype(np.float64)
        dnb_expr = clean_observed + ambient_observed
        for c in nonzero_cells:
            mask = dnb_labels == c
            true_expr[:, c] = clean_observed[:, mask].sum(axis=1)

    n_empty_final = int(empty_mask.sum())
    print(f"  Generated: {n_kept} cells, {n_dnbs} DNBs "
          f"({n_empty_final} empty {100*n_empty_final/n_dnbs:.1f}%, "
          f"{n_dnbs - n_empty_final} cell), {n_genes} genes")

    return {
        "dnb_expr": dnb_expr,
        "dnb_coords": all_coords,
        "dnb_labels": dnb_labels,
        "true_expr": true_expr,
        "gene_is_high": gene_is_high,
        "cell_types": cell_types,
        "true_alpha": true_alpha,
        "true_lambda": float(ambient_lambda),
        "params": {
            "n_cells_requested": n_cells,
            "n_cells_kept": n_kept,
            "grid_width": grid_width,
            "grid_height": grid_height,
            "dnb_pitch": dnb_pitch,
            "cell_radius": float(cell_radius),
            "cell_radius_cv": float(cell_radius_cv),
            "n_genes": n_genes,
            "n_high_genes": n_high_genes,
            "ambient_lambda": float(ambient_lambda),
            "ambient_alpha": float(ambient_alpha),
            "empty_fraction": float(empty_fraction),
            "n_cell_types": n_cell_types,
            "marker_fraction": float(marker_fraction),
            "cluster_strength": float(cluster_strength),
            "dropout_rate": float(dropout_rate),
            "bg_rate_range": tuple(float(v) for v in bg_rate_range),
            "high_rate_range": tuple(float(v) for v in high_rate_range),
            "seed": seed,
        },
    }
