"""Tests for vectorized cell-pipeline preprocessing."""

import numpy as np
from scipy.sparse import csr_matrix

from stambient.cell_pipeline import _bin_empty_dnbs


def _legacy_empty_mapping(coords, labels, bin_size):
    empty_mask = labels < 0
    empty_coords = coords[empty_mask]
    empty_orig_idx = np.where(empty_mask)[0]
    x_min, y_min = empty_coords.min(axis=0)
    x_max, _ = empty_coords.max(axis=0)
    n_bins_x = max(1, int(np.ceil((x_max - x_min) / bin_size)))
    bin_x = np.floor((empty_coords[:, 0] - x_min) / bin_size).astype(np.int64)
    bin_y = np.floor((empty_coords[:, 1] - y_min) / bin_size).astype(np.int64)
    bin_x = np.clip(bin_x, 0, n_bins_x - 1)
    bin_y = np.clip(bin_y, 0, n_bins_x - 1)
    bin_ids = bin_x * 100000 + bin_y
    unique_bins, inverse = np.unique(bin_ids, return_inverse=True)
    areas = np.bincount(inverse, minlength=len(unique_bins))
    centroids = np.zeros((len(unique_bins), 2))
    np.add.at(centroids[:, 0], inverse, empty_coords[:, 0])
    np.add.at(centroids[:, 1], inverse, empty_coords[:, 1])
    centroids /= areas[:, None]
    grouped = [empty_orig_idx[inverse == i] for i in range(len(unique_bins))]
    rows = np.concatenate(grouped)
    columns = np.concatenate(
        [np.full(len(indices), i, dtype=np.int64) for i, indices in enumerate(grouped)]
    )
    return centroids, areas, rows, columns


def test_vectorized_empty_mapping_matches_legacy_result():
    rng = np.random.RandomState(12)
    coords = rng.randint(0, 80, size=(500, 2)).astype(np.float64)
    labels = rng.randint(-1, 5, size=500)
    labels[:50] = -1

    expected_coords, expected_areas, old_rows, old_columns = (
        _legacy_empty_mapping(coords, labels, bin_size=7)
    )
    actual_coords, actual_areas, new_rows, new_columns = _bin_empty_dnbs(
        coords, labels, bin_size=7
    )

    np.testing.assert_allclose(actual_coords, expected_coords, rtol=0, atol=0)
    np.testing.assert_array_equal(actual_areas, expected_areas)
    old_mapping = csr_matrix(
        (np.ones(len(old_rows)), (old_rows, old_columns)),
        shape=(len(labels), len(expected_areas)),
    )
    new_mapping = csr_matrix(
        (np.ones(len(new_rows)), (new_rows, new_columns)),
        shape=(len(labels), len(actual_areas)),
    )
    assert (old_mapping != new_mapping).nnz == 0


def test_empty_mapping_handles_no_empty_dnbs():
    coords = np.array([[0.0, 0.0], [1.0, 1.0]])
    labels = np.array([0, 1])
    bin_coords, areas, rows, columns = _bin_empty_dnbs(coords, labels, 5)
    assert bin_coords.shape == (0, 2)
    assert areas.size == rows.size == columns.size == 0
