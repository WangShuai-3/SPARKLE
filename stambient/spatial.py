"""Spatial computation: KDTree, neighbor search, distance weight functions."""

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from typing import Optional, Tuple, Literal

DistanceMetric = Literal["exponential", "gaussian", "inverse"]


def _weight_exponential(distances: np.ndarray, lam: float) -> np.ndarray:
    """Exponential decay: w(d) = exp(-d / lambda)."""
    return np.exp(-distances / lam)


def _weight_gaussian(distances: np.ndarray, lam: float) -> np.ndarray:
    """Gaussian decay: w(d) = exp(-d² / (2 * lambda²))."""
    return np.exp(-distances ** 2 / (2 * lam ** 2))


def _weight_inverse(distances: np.ndarray, lam: float, eps: float = 1e-6) -> np.ndarray:
    """Inverse distance: w(d) = 1 / (d + eps). lam is unused but kept for API consistency."""
    return 1.0 / (distances + eps)


_WEIGHT_FUNCTIONS = {
    "exponential": _weight_exponential,
    "gaussian": _weight_gaussian,
    "inverse": _weight_inverse,
}


def compute_distance_weights(
    distances: np.ndarray,
    lam: float,
    metric: DistanceMetric = "exponential",
) -> np.ndarray:
    """Compute distance-based weights.

    Args:
        distances: Array of distances.
        lam: Distance decay parameter (lambda).
        metric: Weight function type.

    Returns:
        Weight array of same shape.
    """
    fn = _WEIGHT_FUNCTIONS[metric]
    return fn(distances, lam)


def build_spatial_graph(
    coords: np.ndarray,
    max_radius: float,
    lam: float,
    metric: DistanceMetric = "exponential",
) -> csr_matrix:
    """Build a sparse spatial adjacency matrix with distance-based weights.

    For each pair of bins (i, j) within max_radius, stores w(d(i, j)).

    Args:
        coords: [N × 2] array of bin center coordinates in μm.
        max_radius: Maximum neighbor distance (μm).
        lam: Distance decay parameter.
        metric: Weight function type.

    Returns:
        Sparse [N × N] CSR matrix of distance weights. Diagonal is 0.
    """
    return distance_graph_to_weights(
        build_spatial_distance_graph(coords, max_radius), lam, metric
    )


def build_spatial_distance_graph(
    coords: np.ndarray,
    max_radius: float,
) -> csr_matrix:
    """Build a CSR graph whose values are distances, excluding its diagonal.

    Keeping topology/distances separate from weights lets lambda grid search
    run without repeating the KDTree neighbor search.
    """
    tree = cKDTree(coords)
    n = len(coords)
    dist_coo = tree.sparse_distance_matrix(
        tree, max_radius, output_type="coo_matrix"
    )
    mask = dist_coo.row != dist_coo.col
    return csr_matrix(
        (
            dist_coo.data[mask].astype(np.float64, copy=False),
            (dist_coo.row[mask], dist_coo.col[mask]),
        ),
        shape=(n, n),
        dtype=np.float64,
    )


def distance_graph_to_weights(
    distance_graph: csr_matrix,
    lam: float,
    metric: DistanceMetric = "exponential",
) -> csr_matrix:
    """Apply a distance kernel while reusing a CSR graph's topology."""
    distances = distance_graph.tocsr()
    return csr_matrix(
        (
            compute_distance_weights(distances.data, lam, metric),
            distances.indices.copy(),
            distances.indptr.copy(),
        ),
        shape=distances.shape,
        dtype=np.float64,
    )


def compute_neighbor_weighted_sum(
    W: csr_matrix,
    source_values: np.ndarray,
) -> np.ndarray:
    """Compute the neighbor-weighted sum N_i = Σ_j W_{ij} * source_values_j.

    Args:
        W: [N × N] sparse spatial weight matrix.
        source_values: [N] array of source strengths.

    Returns:
        [N] array of weighted sums.
    """
    return W.dot(source_values)


def build_spatial_graph_between(
    coords_a: np.ndarray,
    coords_b: np.ndarray,
    max_radius: float,
    lam: float,
    metric: DistanceMetric = "exponential",
) -> csr_matrix:
    """Build a sparse spatial weight matrix from coords_a to coords_b.

    For each pair (i in A, j in B) within max_radius, stores w(d(i, j)).
    This avoids building the full (|A|+|B|)^2 matrix when only the
    cross-block is needed.

    Args:
        coords_a: [M × 2] source coordinates.
        coords_b: [N × 2] target coordinates.
        max_radius: Maximum neighbor distance (μm).
        lam: Distance decay parameter.
        metric: Weight function type.

    Returns:
        Sparse [M × N] CSR matrix of distance weights.
    """
    return distance_graph_to_weights(
        build_spatial_distance_graph_between(coords_a, coords_b, max_radius),
        lam,
        metric,
    )


def build_spatial_distance_graph_between(
    coords_a: np.ndarray,
    coords_b: np.ndarray,
    max_radius: float,
) -> csr_matrix:
    """Build a cross-CSR graph whose values are pairwise distances."""
    tree_a = cKDTree(coords_a)
    tree_b = cKDTree(coords_b)
    dist_coo = tree_a.sparse_distance_matrix(
        tree_b, max_radius, output_type="coo_matrix"
    )
    return csr_matrix(
        (
            dist_coo.data.astype(np.float64, copy=False),
            (dist_coo.row, dist_coo.col),
        ),
        shape=(len(coords_a), len(coords_b)),
        dtype=np.float64,
    )


def compute_local_density(
    coords: np.ndarray,
    n_cell: np.ndarray,
    n_total: np.ndarray,
    local_radius: float,
) -> np.ndarray:
    """Compute local cell density β for each bin.

    β_b = Σ_{b' in window} n_cell[b'] / Σ_{b' in window} n_total[b']

    Args:
        coords: [N × 2] bin center coordinates.
        n_cell: [N] cell DNB counts per bin.
        n_total: [N] total DNB counts per bin (n_cell + n_empty).
        local_radius: Radius of local density window (μm).

    Returns:
        [N] array of local density β values.
    """
    tree = cKDTree(coords)
    # Vectorized neighborhood aggregation via a binary adjacency matrix.
    adj = tree.sparse_distance_matrix(tree, local_radius, output_type="coo_matrix")
    # Include self in the local window (the legacy loop included i itself).
    adj.data = np.ones_like(adj.data)
    adj = adj.tocsr()

    sum_cell = adj.dot(n_cell.astype(np.float64))
    sum_total = adj.dot(n_total.astype(np.float64))
    beta = sum_cell / np.maximum(sum_total, 1.0)
    return beta
