"""Gene batching helpers shared by the CPU and GPU estimation stages.

Full-transcriptome runs cannot hold one dense [bins × genes] (or
[cells × genes]) temporary per stage in memory.  Both backends therefore
process genes in batches: the sparse gene × cell / gene × bin matrices
produced by :mod:`stambient.aggregation` stay sparse, and only the rows of
the current batch are densified.

- ``densify_gene_rows``: slice a row subset from a dense or sparse matrix
  into a dense float64 block.
- ``choose_cpu_gene_batch_size``: pick a batch size that keeps the dense
  per-batch working set within a fixed RAM budget (the CPU counterpart of
  ``stambient.gpu.choose_gpu_gene_batch_size``).
"""

import numpy as np

# Target size of the dense float64 temporaries a gene batch may allocate at
# once (sources, observations, sparse products, and residual temporaries).
CPU_GENE_BATCH_WORKING_SET_BYTES = 1 << 30  # 1 GiB


def densify_gene_rows(matrix, gene_indices: np.ndarray) -> np.ndarray:
    """Return dense ``[len(gene_indices) × n_cols]`` rows of ``matrix``.

    Args:
        matrix: dense ndarray or SciPy sparse matrix with genes as rows.
        gene_indices: row indices (genes) to extract.

    Returns:
        Dense float64 ndarray holding only the requested rows, so callers
        with a sparse ``matrix`` never densify more than one batch.
    """
    rows = matrix[gene_indices]
    if isinstance(rows, np.ndarray):
        return rows
    return (
        np.asarray(rows.todense())
        if hasattr(rows, "todense")
        else rows.toarray()
    )


def choose_cpu_gene_batch_size(
    n_rows: int,
    n_cols: int,
    max_batch_size: int,
    dtype_size: int = 8,
    working_set_bytes: int = CPU_GENE_BATCH_WORKING_SET_BYTES,
) -> int:
    """Choose a gene batch size that bounds the per-batch RAM working set.

    A batch allocates a handful of dense ``[n_rows × batch]`` and
    ``[n_cols × batch]`` arrays (sources, observations, sparse-matrix
    products, and residual temporaries), so the cost per gene is taken to
    be ``6 * max(n_rows, n_cols) * dtype_size`` bytes.

    Args:
        n_rows: rows of the largest per-batch product (e.g. n_empty_bins).
        n_cols: columns of the largest per-batch product (e.g. n_cells).
        max_batch_size: never exceed this many genes (the stage's gene
            count); values below 1 are clamped to 1.
        dtype_size: bytes per element of the dense temporaries.
        working_set_bytes: RAM budget for one batch of dense temporaries.

    Returns:
        A batch size in ``[1, max_batch_size]``.
    """
    max_batch_size = max(1, int(max_batch_size))
    bytes_per_gene = max(1, 6 * max(int(n_rows), int(n_cols)) * int(dtype_size))
    estimate = int(working_set_bytes) // bytes_per_gene
    return max(1, min(max_batch_size, estimate))
