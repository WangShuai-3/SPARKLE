"""Spatial computation: KDTree, neighbor search, distance weight functions."""

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from typing import Literal

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


def build_spatial_distance_graph_between(
    coords_a: np.ndarray,
    coords_b: np.ndarray,
    max_radius: float,
) -> csr_matrix:
    """Build a cross-CSR graph whose values are pairwise distances.

    This avoids building the full (|A|+|B|)^2 matrix when only the
    cross-block is needed.
    """
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
