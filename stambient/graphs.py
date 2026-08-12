"""Spatial-graph preparation for the pipeline's estimation stages.

The KDTree neighbour search lives in :mod:`stambient.spatial`; this module
wraps it for the two graphs the cell pipeline needs:

- empty bins → cells: the ambient "predictor" graph used for lambda and alpha
  estimation;
- cells → cells: the neighbour graph used to compute per-cell ambient inflow
  during correction.

Both wrappers optionally upload the distance CSR to the GPU so the weight
kernels can be applied there without repeating the neighbour search.
"""

import numpy as np
from scipy.sparse import csr_matrix
from typing import Optional, Tuple

from .gpu import GPUContext, sparse_distance_graph_to_gpu_csr
from .spatial import build_spatial_distance_graph, build_spatial_distance_graph_between


def build_empty_to_cell_graph(
    empty_bin_coords: np.ndarray,
    cell_centroids: np.ndarray,
    max_radius: float,
    gpu: Optional[GPUContext],
    storage_dtype,
) -> Tuple[csr_matrix, int, Optional[object]]:
    """Build the empty-bins → cells distance graph (CPU and optionally GPU).

    Returns:
        distances: [n_empty_bins × n_cells] CSR of pairwise distances.
        nnz: number of stored edges (for diagnostics).
        gpu_csr: GPU-resident distance graph, or None on CPU.
    """
    distances = build_spatial_distance_graph_between(
        empty_bin_coords, cell_centroids, max_radius
    )
    nnz = int(distances.nnz)
    gpu_csr = None
    if gpu is not None:
        gpu_csr = sparse_distance_graph_to_gpu_csr(
            distances, gpu, dtype=storage_dtype
        )
    return distances, nnz, gpu_csr


def build_cell_to_cell_graph(
    cell_centroids: np.ndarray,
    max_radius: float,
    gpu: Optional[GPUContext],
    storage_dtype,
) -> Tuple[csr_matrix, int, Optional[object]]:
    """Build the cells → cells distance graph (CPU and optionally GPU).

    Returns:
        distances: [n_cells × n_cells] CSR of pairwise distances (diagonal
            excluded).
        nnz: number of stored edges (for diagnostics).
        gpu_csr: GPU-resident distance graph, or None on CPU.
    """
    distances = build_spatial_distance_graph(cell_centroids, max_radius)
    nnz = int(distances.nnz)
    gpu_csr = None
    if gpu is not None:
        gpu_csr = sparse_distance_graph_to_gpu_csr(
            distances, gpu, dtype=storage_dtype
        )
    return distances, nnz, gpu_csr
