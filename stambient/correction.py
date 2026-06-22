"""Ambient RNA subtraction and per-cell aggregation."""

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from typing import Dict, Tuple, Optional

from .spatial import build_spatial_graph, DistanceMetric
from .estimation import _compute_source_strength


def correct_expression(
    Y_cell: csr_matrix,
    Y_empty: csr_matrix,
    n_cell: np.ndarray,
    n_empty: np.ndarray,
    n_total: np.ndarray,
    bin_coords: np.ndarray,
    bin_cell_assignment: Dict[int, Dict[int, int]],
    gene_indices: np.ndarray,
    alphas: np.ndarray,
    r2_scores: np.ndarray,
    r2_threshold: float,
    lam: float,
    max_radius: float,
    beta: Optional[np.ndarray],
    metric: DistanceMetric = "exponential",
    use_expr_weight: bool = False,
    lambdas_per_gene: Optional[np.ndarray] = None,
    cell_level: bool = False,
) -> Tuple[csr_matrix, np.ndarray, np.ndarray]:
    """Apply ambient RNA correction and aggregate to per-cell expression.

    For each high-expression gene with R² >= threshold:
      A_gb = α_g * n_cell[b] * β_b * Σ_{b'} w(d) * (Y_cell[g,b'] / n_cell[b'])
      S_gb = max(Y_cell[g,b] - A_gb, 0)

    Then aggregate per cell:
      S_gc = Σ_b S_gb * (n_cell_c[b] / n_cell[b])

    Args:
        Y_cell: [genes × bins] cell DNB expression.
        Y_empty: [genes × bins] empty DNB expression.
        n_cell: [bins] cell DNB counts.
        n_empty: [bins] empty DNB counts.
        n_total: [bins] total DNB counts.
        bin_coords: [bins × 2] bin centers.
        bin_cell_assignment: bin_id → {cell_id: n_dnbs}.
        gene_indices: Indices of high-expression genes.
        alphas: [n_genes] estimated α per gene.
        r2_scores: [n_genes] R² per gene.
        r2_threshold: Minimum R² to apply correction.
        lam: Distance decay parameter.
        max_radius: Max neighbor search radius.
        beta: [bins] local density. If None, set to 1.
        metric: Weight function.

    Returns:
        corrected_cell_expr: [genes × cells] corrected per-cell expression (sparse).
        ambient_fraction: [n_genes] estimated ambient fraction per gene.
        correction_magnitude: [n_genes] mean correction magnitude per gene.
    """
    n_genes_total = Y_cell.shape[0]
    n_bins = Y_cell.shape[1]
    n_genes_use = len(gene_indices)

    if beta is None:
        beta = np.ones(n_bins, dtype=np.float64)

    # Collect all unique cell IDs
    all_cell_ids = set()
    for d in bin_cell_assignment.values():
        all_cell_ids.update(d.keys())
    cell_id_list = sorted(all_cell_ids)
    cell_id_to_idx = {cid: i for i, cid in enumerate(cell_id_list)}
    n_cells = len(cell_id_list)

    # Build spatial weight matrix once (or per gene if per-gene λ)
    W_global = build_spatial_graph(bin_coords, max_radius, lam, metric)
    W_cache = {}  # cache per unique λ

    cell_mask = n_cell > 0

    # Precompute source strengths for ALL bins (we'll use as needed)
    # Store Y_cell as dense for the genes we care about
    if hasattr(Y_cell, 'toarray'):
        Y_cell_dense = Y_cell.toarray()
    else:
        Y_cell_dense = np.asarray(Y_cell.todense())

    if hasattr(Y_empty, 'toarray'):
        Y_empty_dense = Y_empty.toarray()
    else:
        Y_empty_dense = np.asarray(Y_empty.todense())

    # Build gene index map
    gene_idx_to_pos = {g_idx: i for i, g_idx in enumerate(gene_indices)}
    gene_idx_set = set(gene_indices)

    # Initialize corrected sparse matrix
    corrected = lil_matrix((n_genes_total, n_cells), dtype=np.float64)

    ambient_fraction = np.zeros(n_genes_total, dtype=np.float64)
    correction_magnitude = np.zeros(n_genes_total, dtype=np.float64)

    # For genes NOT in high-expr list or with low R²: copy original expression
    for g_idx in range(n_genes_total):
        if g_idx not in gene_idx_set:
            # Non-high-expr gene: keep original
            _aggregate_gene_to_cells(
                Y_cell_dense[g_idx], n_cell, cell_mask,
                bin_cell_assignment, cell_id_to_idx, corrected, g_idx
            )
            ambient_fraction[g_idx] = 0.0
            correction_magnitude[g_idx] = 0.0
            continue

        pos = gene_idx_to_pos[g_idx]
        if r2_scores[pos] < r2_threshold:
            _aggregate_gene_to_cells(
                Y_cell_dense[g_idx], n_cell, cell_mask,
                bin_cell_assignment, cell_id_to_idx, corrected, g_idx
            )
            ambient_fraction[g_idx] = 0.0
            correction_magnitude[g_idx] = 0.0
            continue

        # ── Cell-level correction ──────────────────────────────
        if cell_level:
            # Compute dominant cell per bin and overlap matrix
            bin_dominant = np.full(n_bins, -1, dtype=np.int64)
            for b in bin_cell_assignment:
                if n_cell[b] > 0:
                    dom_cid = max(bin_cell_assignment[b], key=bin_cell_assignment[b].get)
                    bin_dominant[b] = cell_id_to_idx.get(dom_cid, -1)

            # For each pair of bins (b, b2) where W[b,b2] > 0, compute
            # overlap. If b2 is dominated by b's dominant cell, zero the weight.
            W_coo = W_global.tocoo()
            rows, cols, data = W_coo.row, W_coo.col, W_coo.data.copy()
            for idx in range(len(rows)):
                b, b2 = rows[idx], cols[idx]
                if bin_dominant[b] < 0 or bin_dominant[b2] < 0:
                    continue
                if bin_dominant[b] == bin_dominant[b2]:
                    # b2 is dominated by same cell as b → exclude
                    data[idx] = 0.0
            W_corrected = csr_matrix((data, (rows, cols)), shape=W_global.shape)

            for g_idx in range(n_genes_total):
                if g_idx not in gene_idx_set:
                    _aggregate_gene_to_cells(Y_cell_dense[g_idx], n_cell, cell_mask,
                        bin_cell_assignment, cell_id_to_idx, corrected, g_idx)
                    continue
                pos = gene_idx_to_pos[g_idx]
                if r2_scores[pos] < r2_threshold:
                    _aggregate_gene_to_cells(Y_cell_dense[g_idx], n_cell, cell_mask,
                        bin_cell_assignment, cell_id_to_idx, corrected, g_idx)
                    continue

                alpha_g = alphas[pos]
                y_cell_row = Y_cell_dense[g_idx]
                source = _compute_source_strength(y_cell_row, n_cell, cell_mask, use_expr_weight)
                neighbor_contrib = W_corrected.dot(source)
                ambient = alpha_g * n_cell * beta * neighbor_contrib
                s_gb = np.maximum(y_cell_row - ambient, 0.0)
                _aggregate_gene_to_cells(s_gb, n_cell, cell_mask,
                    bin_cell_assignment, cell_id_to_idx, corrected, g_idx)

                total_y = y_cell_row[cell_mask].sum()
                total_ambient = ambient[cell_mask].sum()
                if total_y > 0:
                    ambient_fraction[g_idx] = min(total_ambient / total_y, 1.0)

            return corrected.tocsr(), ambient_fraction, correction_magnitude

        # ── Bin-level correction ────────────────────────────────
        alpha_g = alphas[pos]
        y_cell_row = Y_cell_dense[g_idx]

        # Source strength: Y_cell / n_cell (optionally expression-weighted)
        source_strength = _compute_source_strength(
            y_cell_row, n_cell, cell_mask, use_expr_weight
        )

        # Ambient per bin: α * n_cell * β * Σ w(d) * source
        # Use per-gene W if available
        gene_lam = lam
        if lambdas_per_gene is not None and pos < len(lambdas_per_gene):
            gene_lam = lambdas_per_gene[pos]
        if gene_lam not in W_cache:
            if gene_lam == lam:
                W_cache[gene_lam] = W_global
            else:
                W_cache[gene_lam] = build_spatial_graph(
                    bin_coords, max_radius, gene_lam, metric
                )
        W = W_cache[gene_lam]
        neighbor_contrib = W.dot(source_strength)
        ambient = alpha_g * n_cell * beta * neighbor_contrib

        # Corrected cell expression per bin
        s_gb = np.maximum(y_cell_row - ambient, 0.0)

        # Aggregate to cells
        _aggregate_gene_to_cells(
            s_gb, n_cell, cell_mask,
            bin_cell_assignment, cell_id_to_idx, corrected, g_idx
        )

        # Diagnostics
        total_y = y_cell_row[cell_mask].sum()
        total_ambient = ambient[cell_mask].sum()
        if total_y > 0:
            ambient_fraction[g_idx] = min(total_ambient / total_y, 1.0)
        correction_magnitude[g_idx] = ambient[cell_mask].mean() if cell_mask.any() else 0.0

    return corrected.tocsr(), ambient_fraction, correction_magnitude


def _aggregate_gene_to_cells(
    values: np.ndarray,
    n_cell: np.ndarray,
    cell_mask: np.ndarray,
    bin_cell_assignment: Dict[int, Dict[int, int]],
    cell_id_to_idx: Dict[int, int],
    corrected: lil_matrix,
    g_idx: int,
):
    """Aggregate per-bin values for one gene to per-cell expression."""
    for b in range(len(values)):
        if not cell_mask[b]:
            continue
        n_cell_b = n_cell[b]
        if n_cell_b == 0:
            continue
        val_b = values[b]
        if val_b <= 0:
            continue
        assignment = bin_cell_assignment[b]
        for cid, n_dnb in assignment.items():
            cidx = cell_id_to_idx[cid]
            # Proportional allocation
            contribution = val_b * (n_dnb / n_cell_b)
            corrected[g_idx, cidx] += contribution
