"""Tests for spatial module."""

import numpy as np
import pytest
from stambient.spatial import (
    build_spatial_distance_graph,
    build_spatial_distance_graph_between,
    compute_distance_weights,
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
    def test_reweight_applies_kernel_to_distances(self, metric):
        coords = np.array([[0.0, 0.0], [3.0, 4.0], [20.0, 0.0]])
        distances = build_spatial_distance_graph(coords, max_radius=10.0)
        W = distance_graph_to_weights(distances, lam=7.0, metric=metric)
        np.testing.assert_array_equal(W.indptr, distances.indptr)
        np.testing.assert_array_equal(W.indices, distances.indices)
        np.testing.assert_allclose(
            W.data,
            compute_distance_weights(distances.data, 7.0, metric),
            rtol=0,
            atol=0,
        )

    def test_build_distance_graph(self):
        coords = np.array([
            [0.0, 0.0],
            [10.0, 0.0],
            [0.0, 10.0],
            [100.0, 100.0],  # far away
        ])
        distances = build_spatial_distance_graph(coords, max_radius=20.0)

        # Should be symmetric, no diagonal, values are raw distances
        assert distances.shape == (4, 4)
        assert distances[0, 0] == 0
        assert distances[0, 1] == pytest.approx(10.0)
        assert distances[0, 2] == pytest.approx(10.0)
        assert distances[0, 3] == 0  # distance > 20
        assert distances[1, 2] == pytest.approx(np.sqrt(200.0))

        W = distance_graph_to_weights(distances, lam=10.0, metric="exponential")
        assert W[0, 1] == pytest.approx(np.exp(-1.0))

    def test_neighbor_weighted_sum_via_dot(self):
        coords = np.array([
            [0.0, 0.0],
            [10.0, 0.0],
        ])
        distances = build_spatial_distance_graph(coords, max_radius=20.0)
        W = distance_graph_to_weights(distances, lam=10.0, metric="exponential")
        source = np.array([5.0, 3.0])
        result = W.dot(source)

        # For bin 0: w(10) * 3.0 = exp(-1) * 3.0
        assert result[0] == pytest.approx(np.exp(-1.0) * 3.0)
        # For bin 1: w(10) * 5.0 = exp(-1) * 5.0
        assert result[1] == pytest.approx(np.exp(-1.0) * 5.0)


class TestCrossGraph:
    def test_between_shapes_and_distances(self):
        coords_a = np.array([[0.0, 0.0], [10.0, 0.0]])
        coords_b = np.array([[5.0, 0.0], [100.0, 0.0]])
        distances = build_spatial_distance_graph_between(
            coords_a, coords_b, max_radius=20.0
        )

        assert distances.shape == (2, 2)
        assert distances[0, 0] == pytest.approx(5.0)
        assert distances[1, 0] == pytest.approx(5.0)
        assert distances[0, 1] == 0  # distance 100 > 20
        assert distances[1, 1] == 0  # distance 90 > 20
