"""Tests for spatial module."""

import numpy as np
import pytest
from stambient.spatial import (
    build_spatial_distance_graph,
    build_spatial_graph,
    compute_distance_weights,
    compute_local_density,
    compute_neighbor_weighted_sum,
    distance_graph_to_weights,
)


class TestDistanceWeights:
    def test_exponential(self):
        d = np.array([0.0, 50.0, 100.0])
        w = compute_distance_weights(d, 50.0, "exponential")
        expected = np.exp(-d / 50.0)
        np.testing.assert_array_almost_equal(w, expected)

    def test_gaussian(self):
        d = np.array([0.0, 50.0, 100.0])
        w = compute_distance_weights(d, 50.0, "gaussian")
        expected = np.exp(-d**2 / (2 * 50.0**2))
        np.testing.assert_array_almost_equal(w, expected)

    def test_inverse(self):
        d = np.array([1.0, 10.0, 100.0])
        w = compute_distance_weights(d, 50.0, "inverse")
        expected = 1.0 / (d + 1e-6)
        np.testing.assert_array_almost_equal(w, expected)


class TestSpatialGraph:
    @pytest.mark.parametrize("metric", ["exponential", "gaussian", "inverse"])
    def test_distance_topology_reweight_matches_direct_graph(self, metric):
        coords = np.array([[0.0, 0.0], [3.0, 4.0], [20.0, 0.0]])
        distances = build_spatial_distance_graph(coords, max_radius=10.0)
        reused = distance_graph_to_weights(distances, lam=7.0, metric=metric)
        direct = build_spatial_graph(
            coords, max_radius=10.0, lam=7.0, metric=metric
        )
        np.testing.assert_array_equal(reused.indptr, direct.indptr)
        np.testing.assert_array_equal(reused.indices, direct.indices)
        np.testing.assert_allclose(reused.data, direct.data, rtol=0, atol=0)

    def test_build_and_weighted_sum(self):
        coords = np.array([
            [0.0, 0.0],
            [10.0, 0.0],
            [0.0, 10.0],
            [100.0, 100.0],  # far away
        ])
        W = build_spatial_graph(coords, max_radius=20.0, lam=10.0, metric="exponential")

        # Should be symmetric, no diagonal
        assert W.shape == (4, 4)
        assert W[0, 0] == 0
        assert W[0, 1] > 0  # distance 10
        assert W[0, 2] > 0  # distance 10
        assert W[0, 3] == 0  # distance > 20
        assert W[1, 2] > 0  # distance ~14.1

    def test_weighted_sum(self):
        coords = np.array([
            [0.0, 0.0],
            [10.0, 0.0],
        ])
        W = build_spatial_graph(coords, max_radius=20.0, lam=10.0, metric="exponential")
        source = np.array([5.0, 3.0])
        result = compute_neighbor_weighted_sum(W, source)

        # For bin 0: w(10) * 3.0 = exp(-1) * 3.0
        assert result[0] == pytest.approx(np.exp(-1.0) * 3.0)
        # For bin 1: w(10) * 5.0 = exp(-1) * 5.0
        assert result[1] == pytest.approx(np.exp(-1.0) * 5.0)


class TestLocalDensity:
    def test_uniform_density(self):
        coords = np.array([
            [0.0, 0.0],
            [5.0, 0.0],
            [10.0, 0.0],
        ])
        n_cell = np.array([10, 10, 10])
        n_total = np.array([20, 20, 20])
        beta = compute_local_density(coords, n_cell, n_total, local_radius=20.0)

        # All bins see all others → same density
        assert np.allclose(beta, 0.5)

    def test_sparse_density(self):
        coords = np.array([
            [0.0, 0.0],
            [100.0, 0.0],
        ])
        n_cell = np.array([10, 0])
        n_total = np.array([10, 10])
        beta = compute_local_density(coords, n_cell, n_total, local_radius=50.0)

        # bin 0: only sees itself → 10/10 = 1.0
        # bin 1: only sees itself → 0/10 = 0.0
        assert beta[0] == 1.0
        assert beta[1] == 0.0
