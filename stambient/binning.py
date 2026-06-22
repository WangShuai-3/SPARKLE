"""DNB-level data → bin-level aggregation with cell/empty separation."""

import numpy as np
from scipy.sparse import csr_matrix, issparse
from typing import Dict, List, Tuple


def dnb_to_bins(
    dnb_expr: np.ndarray,
    dnb_coords: np.ndarray,
    dnb_labels: np.ndarray,
    bin_size: int,
) -> Tuple[
    csr_matrix,     # Y_cell: [genes × bins] cell DNB expression
    csr_matrix,     # Y_empty: [genes × bins] empty DNB expression
    np.ndarray,     # n_cell: [bins] cell DNB counts
    np.ndarray,     # n_empty: [bins] empty DNB counts
    np.ndarray,     # bin_coords: [bins × 2] bin center coordinates
    Dict,           # bin_cell_assignment: bin_id → {cell_id: n_dnbs}
    np.ndarray,     # bin_classification: [bins] 0=empty, 1=cell, 2=mixed
    np.ndarray,     # bin_sizes: [bins] n_total per bin
]:
    """Aggregate DNB-level data into regular grid bins.

    Each bin separately tracks:
      - Expression from cell-associated DNBs (Y_cell)
      - Expression from empty DNBs (Y_empty)
      - DNB counts for each category

    Args:
        dnb_expr: [genes × DNBs] expression matrix (dense or sparse).
        dnb_coords: [DNBs × 2] coordinates in μm.
        dnb_labels: [DNBs] cell_id per DNB; -1 for empty.
        bin_size: Number of DNBs per bin edge (bin_size × bin_size DNBs per bin).

    Returns:
        Y_cell: [genes × bins] sparse cell DNB expression.
        Y_empty: [genes × bins] sparse empty DNB expression.
        n_cell: [bins] cell DNB counts.
        n_empty: [bins] empty DNB counts.
        bin_coords: [bins × 2] bin centers.
        bin_cell_assignment: dict bin_id → {cell_id: n_dnbs_in_bin}.
        bin_classification: [bins] 0=pure_empty, 1=pure_cell, 2=mixed.
        bin_sizes: [bins] total DNBs per bin.
    """
    n_dnb = dnb_coords.shape[0]
    n_genes = dnb_expr.shape[0]

    # Determine grid origin from min coordinates
    x_min, y_min = dnb_coords.min(axis=0)

    # Sort DNBs by x, y to estimate DNB pitch
    # Assume DNBs are on a regular grid; estimate spacing from sorted unique coords
    x_coords_sorted = np.sort(np.unique(dnb_coords[:, 0]))
    y_coords_sorted = np.sort(np.unique(dnb_coords[:, 1]))
    if len(x_coords_sorted) > 1:
        x_pitch = np.median(np.diff(x_coords_sorted))
    else:
        x_pitch = 500.0  # default for Stereo-seq (nm)
    if len(y_coords_sorted) > 1:
        y_pitch = np.median(np.diff(y_coords_sorted))
    else:
        y_pitch = 500.0

    # Bin dimensions in μm
    bin_width = bin_size * x_pitch
    bin_height = bin_size * y_pitch

    # Assign each DNB to a bin
    x_bin_idx = np.floor((dnb_coords[:, 0] - x_min) / bin_width).astype(np.int64)
    y_bin_idx = np.floor((dnb_coords[:, 1] - y_min) / bin_height).astype(np.int64)

    # Create unique bin IDs
    # Use (x_idx, y_idx) as key
    bin_keys = list(zip(x_bin_idx.tolist(), y_bin_idx.tolist()))
    unique_bins = sorted(set(bin_keys))
    bin_to_id = {k: i for i, k in enumerate(unique_bins)}
    n_bins = len(unique_bins)

    # Map each DNB to bin_id
    bin_ids = np.array([bin_to_id[k] for k in bin_keys], dtype=np.int64)

    # Initialize per-bin accumulators
    n_cell = np.zeros(n_bins, dtype=np.int64)
    n_empty = np.zeros(n_bins, dtype=np.int64)
    bin_cell_assignment: Dict[int, Dict[int, int]] = {i: {} for i in range(n_bins)}

    # Aggregate DNB counts
    is_empty = (dnb_labels < 0)
    is_cell = ~is_empty

    for i in range(n_dnb):
        bid = bin_ids[i]
        if is_empty[i]:
            n_empty[bid] += 1
        else:
            n_cell[bid] += 1
            cell_id = int(dnb_labels[i])
            d = bin_cell_assignment[bid]
            d[cell_id] = d.get(cell_id, 0) + 1

    n_total = n_cell + n_empty

    # Aggregate expression into sparse matrices
    if issparse(dnb_expr):
        dnb_expr = dnb_expr.tocsc()
        Y_cell = _aggregate_sparse(dnb_expr, bin_ids, is_cell, n_bins, n_genes)
        Y_empty = _aggregate_sparse(dnb_expr, bin_ids, is_empty, n_bins, n_genes)
    else:
        Y_cell = _aggregate_dense(dnb_expr, bin_ids, is_cell, n_bins, n_genes)
        Y_empty = _aggregate_dense(dnb_expr, bin_ids, is_empty, n_bins, n_genes)

    # Bin center coordinates (center of the bin grid cell)
    x_centers = np.array([x_min + (x_idx + 0.5) * bin_width for x_idx, _ in unique_bins])
    y_centers = np.array([y_min + (y_idx + 0.5) * bin_height for _, y_idx in unique_bins])
    bin_coords = np.column_stack([x_centers, y_centers])

    # Classify bins
    empty_purity = 0.95
    cell_purity = 0.80

    empty_ratio = np.zeros(n_bins, dtype=np.float64)
    mask = n_total > 0
    empty_ratio[mask] = n_empty[mask] / n_total[mask]

    classification = np.full(n_bins, 2, dtype=np.int64)  # default: mixed

    for i in range(n_bins):
        if n_total[i] == 0:
            classification[i] = 0  # treat empty bins as pure empty
            continue
        if empty_ratio[i] >= empty_purity:
            classification[i] = 0  # pure empty
        else:
            # Check for pure cell
            if bin_cell_assignment[i]:
                max_cell_ratio = max(bin_cell_assignment[i].values()) / n_total[i]
                if max_cell_ratio >= cell_purity:
                    classification[i] = 1  # pure cell

    return (
        Y_cell,
        Y_empty,
        n_cell,
        n_empty,
        bin_coords,
        bin_cell_assignment,
        classification,
        n_total,
    )


def _aggregate_dense(
    expr: np.ndarray,
    bin_ids: np.ndarray,
    mask: np.ndarray,
    n_bins: int,
    n_genes: int,
) -> csr_matrix:
    """Aggregate dense expression per bin for masked DNBs."""
    result = np.zeros((n_genes, n_bins), dtype=np.float64)
    for g in range(n_genes):
        np.add.at(result[g], bin_ids[mask], expr[g, mask])
    # Convert to sparse (most entries may be zero)
    result[result < 0] = 0  # safety
    return csr_matrix(result)


def _aggregate_sparse(
    expr,
    bin_ids: np.ndarray,
    mask: np.ndarray,
    n_bins: int,
    n_genes: int,
) -> csr_matrix:
    """Aggregate sparse expression per bin for masked DNBs."""
    from scipy.sparse import coo_matrix

    expr = expr.tocoo()
    # Filter to masked DNBs
    keep = mask[expr.col]
    rows = expr.row[keep]
    cols = bin_ids[expr.col[keep]]
    data = expr.data[keep]

    result = coo_matrix((data, (rows, cols)), shape=(n_genes, n_bins))
    return result.tocsr()
