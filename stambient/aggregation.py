"""Data-aggregation stages: turn DNB-level input into the pipeline's objects.

Three responsibilities live here:

- ``extract_cells``: unique cell IDs, per-cell DNB counts (areas), centroids,
  and the sparse gene × cell expression matrix.
- ``aggregate_empty_bin_expression``: sparse gene × bin expression from the
  DNB-to-bin mapping produced by :func:`stambient.binning.bin_empty_dnbs`.
- ``select_high_expression_genes``: rank genes by mean empty-bin expression
  and keep the top ``n_high_genes`` for correction.

The aggregated matrices stay sparse (SciPy CSR, genes as rows) so
full-transcriptome runs never materialize a dense genes × cells / genes ×
bins copy; downstream stages densify one gene batch at a time via
:func:`stambient.batching.densify_gene_rows`.

The public pipeline orchestrator times these steps individually.
"""

import numpy as np
from scipy.sparse import csr_matrix
from typing import Tuple


def extract_cells(
    dnb_csc,
    dnb_coords: np.ndarray,
    dnb_labels: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Group labelled DNBs into intact cells.

    Args:
        dnb_csc: [genes × DNBs] expression matrix in CSC form.
        dnb_coords: [DNBs × 2] coordinates in micrometres.
        dnb_labels: [DNBs] cell IDs (-1 for empty).

    Returns:
        unique_cells: sorted unique cell IDs.
        cell_areas: [n_cells] DNB count per cell.
        cell_centroids: [n_cells × 2] mean DNB coordinates per cell.
        cell_expr: [genes × n_cells] sparse CSR per-cell expression.
    """
    cell_mask_labels = dnb_labels >= 0
    unique_cells = np.unique(dnb_labels[cell_mask_labels])
    n_cells = len(unique_cells)
    cell_id_to_idx = {cid: i for i, cid in enumerate(unique_cells)}

    # Vectorized cell area and centroid computation
    cell_indices = np.array(
        [cell_id_to_idx[l] for l in dnb_labels[cell_mask_labels]]
    )
    cell_areas = np.bincount(cell_indices, minlength=n_cells)

    cell_centroids = np.zeros((n_cells, 2))
    np.add.at(cell_centroids[:, 0], cell_indices, dnb_coords[cell_mask_labels, 0])
    np.add.at(cell_centroids[:, 1], cell_indices, dnb_coords[cell_mask_labels, 1])
    valid = cell_areas > 0
    cell_centroids[valid, 0] /= cell_areas[valid]
    cell_centroids[valid, 1] /= cell_areas[valid]

    # Build cell expression via sparse matrix multiplication; the product
    # stays sparse (genes as rows) and is densified per gene batch later.
    cell_dnb_idx = np.where(cell_mask_labels)[0]
    C = csr_matrix(
        (np.ones(len(cell_dnb_idx), dtype=np.float64), (cell_dnb_idx, cell_indices)),
        shape=(len(dnb_coords), n_cells),
    )
    cell_expr = (dnb_csc @ C).tocsr()
    return unique_cells, cell_areas, cell_centroids, cell_expr


def aggregate_empty_bin_expression(
    dnb_csc,
    empty_dnb_indices: np.ndarray,
    empty_bin_indices: np.ndarray,
    n_empty_bins: int,
    n_genes: int,
):
    """Sum empty DNB expression into bins using a sparse indicator matrix.

    Args:
        dnb_csc: [genes × DNBs] expression matrix in CSC form.
        empty_dnb_indices: DNB positions of the empty DNBs (from
            :func:`stambient.binning.bin_empty_dnbs`).
        empty_bin_indices: aligned bin id for each empty DNB.
        n_empty_bins: number of bins (columns of the output).
        n_genes: number of genes (rows of the output).

    Returns:
        empty_bin_expr: [genes × n_empty_bins] sparse CSR bin expression.
    """
    if len(empty_dnb_indices) > 0:
        C_empty = csr_matrix(
            (
                np.ones(len(empty_dnb_indices), dtype=np.float64),
                (empty_dnb_indices, empty_bin_indices),
            ),
            shape=(dnb_csc.shape[1], n_empty_bins),
        )
        empty_bin_expr = (dnb_csc @ C_empty).tocsr()
    else:
        empty_bin_expr = csr_matrix((n_genes, 0), dtype=np.float64)
    return empty_bin_expr


def select_high_expression_genes(
    empty_bin_expr,
    empty_bin_areas: np.ndarray,
    n_high_genes: int,
) -> np.ndarray:
    """Rank genes by mean empty-bin expression and keep the top ones.

    Args:
        empty_bin_expr: [genes × n_empty_bins] bin expression, sparse CSR or
            dense; only row sums are needed, so no densification occurs.
        empty_bin_areas: [n_empty_bins] DNB count per bin (area weighting).
        n_high_genes: number of top genes to select; None means all genes.

    Returns:
        gene_indices: [n_select] indices sorted by descending mean expression.
    """
    n_genes = empty_bin_expr.shape[0]
    total_empty = empty_bin_areas.sum()
    mean_empty = np.asarray(empty_bin_expr.sum(axis=1)).ravel() / total_empty
    n_select = n_genes if n_high_genes is None else min(n_high_genes, n_genes)
    return np.argsort(mean_empty)[::-1][:n_select]
