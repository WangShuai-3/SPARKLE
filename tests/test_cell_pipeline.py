"""Tests for cell-pipeline empty-space preprocessing."""

import numpy as np
import pytest

from stambient.cell_pipeline import _bin_empty_dnbs


def _rectangular_grid(width, height, pitch=5.0):
    x = np.arange(0.0, width, pitch)
    y = np.arange(0.0, height, pitch)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    return np.column_stack((xx.ravel(), yy.ravel()))


def test_empty_mapping_preserves_portrait_grid_rows():
    coords = _rectangular_grid(width=20.0, height=50.0)
    labels = np.full(len(coords), -1, dtype=np.int64)

    centroids, areas, rows, columns = _bin_empty_dnbs(
        coords, labels, bin_size=10.0
    )

    # Two x bins and five y bins. The former implementation clipped y with
    # n_bins_x and incorrectly collapsed this portrait grid to four bins.
    assert len(areas) == 10
    np.testing.assert_array_equal(areas, np.full(10, 4))
    assert len(np.unique(centroids[:, 0])) == 2
    assert len(np.unique(centroids[:, 1])) == 5
    np.testing.assert_array_equal(np.sort(rows), np.arange(len(coords)))
    assert columns.min() == 0
    assert columns.max() == 9


def test_empty_mapping_is_invariant_to_axis_transposition():
    portrait = _rectangular_grid(width=20.0, height=50.0)
    landscape = portrait[:, ::-1]
    labels = np.full(len(portrait), -1, dtype=np.int64)

    portrait_centroids, portrait_areas, _, _ = _bin_empty_dnbs(
        portrait, labels, bin_size=10.0
    )
    landscape_centroids, landscape_areas, _, _ = _bin_empty_dnbs(
        landscape, labels, bin_size=10.0
    )

    np.testing.assert_array_equal(
        np.sort(portrait_areas), np.sort(landscape_areas)
    )
    expected = portrait_centroids[:, ::-1]
    expected = expected[np.lexsort((expected[:, 1], expected[:, 0]))]
    actual = landscape_centroids[
        np.lexsort((landscape_centroids[:, 1], landscape_centroids[:, 0]))
    ]
    np.testing.assert_allclose(actual, expected)


def test_empty_mapping_ignores_cell_associated_locations():
    coords = np.array(
        [[0.0, 0.0], [5.0, 5.0], [10.0, 10.0], [100.0, 100.0]]
    )
    labels = np.array([-1, -1, -1, 0])

    centroids, areas, rows, columns = _bin_empty_dnbs(
        coords, labels, bin_size=10.0
    )

    assert areas.sum() == 3
    np.testing.assert_array_equal(rows, np.array([0, 1, 2]))
    assert columns.shape == (3,)
    assert np.max(centroids) <= 10.0


def test_empty_mapping_handles_no_empty_dnbs():
    coords = np.array([[0.0, 0.0], [1.0, 1.0]])
    labels = np.array([0, 1])
    bin_coords, areas, rows, columns = _bin_empty_dnbs(coords, labels, 5.0)
    assert bin_coords.shape == (0, 2)
    assert areas.size == rows.size == columns.size == 0


@pytest.mark.parametrize("bin_size", [0.0, -1.0, np.inf, np.nan])
def test_empty_mapping_rejects_invalid_bin_size(bin_size):
    coords = np.array([[0.0, 0.0]])
    labels = np.array([-1])
    with pytest.raises(ValueError, match="positive finite"):
        _bin_empty_dnbs(coords, labels, bin_size)
