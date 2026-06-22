"""Spatial computation: KDTree, neighbor search, distance weight functions."""

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix, lil_matrix
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
    tree = cKDTree(coords)
    # Query all pairs within max_radius; returns indices as list-of-arrays
    pairs = tree.query_ball_tree(tree, max_radius)

    n = len(coords)
    W = lil_matrix((n, n), dtype=np.float64)

    for i, neighbors in enumerate(pairs):
        if len(neighbors) == 0:
            continue
        # Remove self
        neighbors_arr = np.array(neighbors)
        neighbors_arr = neighbors_arr[neighbors_arr != i]
        if len(neighbors_arr) == 0:
            continue
        dists = np.linalg.norm(coords[neighbors_arr] - coords[i], axis=1)
        weights = compute_distance_weights(dists, lam, metric)
        W[i, neighbors_arr] = weights

    return W.tocsr()


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
    neighborhood = tree.query_ball_tree(tree, local_radius)

    beta = np.zeros(len(coords), dtype=np.float64)
    for i, neighbors in enumerate(neighborhood):
        if len(neighbors) == 0:
            beta[i] = 0.0
            continue
        idx = np.array(neighbors)
        sum_cell = n_cell[idx].sum()
        sum_total = n_total[idx].sum()
        beta[i] = sum_cell / max(sum_total, 1)
    return beta
