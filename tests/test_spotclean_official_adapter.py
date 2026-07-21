import csv

import h5py
import numpy as np
from scipy.sparse import csr_matrix

from evaluation.baselines.spotclean_official import (
    prepare_spotclean_spots,
    write_official_input,
)
from evaluation.scripts.run_spotclean_official import _make_spatial_tile_specs


def test_official_adapter_uses_cell_and_empty_bin_centroids_without_scaling(tmp_path):
    expression = csr_matrix(
        np.array([[2, 3, 5, 7], [0, 1, 2, 4]], dtype=np.float64)
    )
    coordinates = np.array([[0, 0], [2, 0], [10, 0], [12, 0]], dtype=np.float64)
    labels = np.array([101, 101, -1, -1])

    spots, centroids, tissue, barcodes, diagnostics = prepare_spotclean_spots(
        expression,
        coordinates,
        labels,
        np.array([101]),
        empty_bin_size=5,
        coordinate_scale=0.5,
    )

    np.testing.assert_array_equal(spots.toarray(), [[5, 12], [1, 6]])
    np.testing.assert_allclose(centroids, [[0.5, 0], [5.5, 0]])
    np.testing.assert_array_equal(tissue, [1, 0])
    assert barcodes.tolist() == ["cell_101", "background_0"]
    assert diagnostics["empty_exposure_normalized"] is False
    assert diagnostics["empty_bin_dnb_count_median"] == 2


def test_official_adapter_writes_csc_and_unit_slope_slide_metadata(tmp_path):
    spots = csr_matrix(np.array([[5, 12], [1, 6]], dtype=np.float64))
    centroids = np.array([[0.5, 1.5], [5.5, 6.5]])
    tissue = np.array([1, 0])
    barcodes = np.array(["cell_101", "background_0"])

    write_official_input(
        tmp_path,
        spots,
        centroids,
        tissue,
        barcodes,
        ["gene_a", "gene_b"],
        ["gene_a"],
    )

    with h5py.File(tmp_path / "counts_csc.h5", "r") as handle:
        assert handle["shape"][:].tolist() == [2, 2]
        np.testing.assert_array_equal(handle["data"][:], [5, 1, 12, 6])
        np.testing.assert_array_equal(handle["indices"][:], [0, 1, 0, 1])
        np.testing.assert_array_equal(handle["indptr"][:], [0, 2, 4])
    with open(tmp_path / "slide.csv", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0] == {
        "barcode": "cell_101",
        "tissue": "1",
        "row": "1.5",
        "col": "0.5",
        "imagerow": "1.5",
        "imagecol": "0.5",
    }


def test_spatial_tiles_assign_each_tissue_core_once_and_keep_background_context():
    coordinates = np.array(
        [
            [0.5, 0.5], [1.5, 0.5], [0.5, 1.5], [1.5, 1.5],
            [0.6, 0.6], [1.6, 0.6], [0.6, 1.6], [1.6, 1.6],
        ],
        dtype=np.float64,
    )
    tissue = np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=np.int8)
    config = {
        "tile_grid": (2, 2),
        "tile_halo": 0.0,
        "coordinate_scale": 1.0,
        "x_range": (0, 2),
        "y_range": (0, 2),
    }

    specs = _make_spatial_tile_specs(coordinates, tissue, config)

    assert [spec["tile_id"] for spec in specs] == ["r0c0", "r0c1", "r1c0", "r1c1"]
    np.testing.assert_array_equal(
        np.concatenate([spec["core_tissue_indices"] for spec in specs]),
        [0, 1, 2, 3],
    )
    for spec in specs:
        assert np.sum(tissue[spec["context_indices"]] == 1) == 1
        assert np.sum(tissue[spec["context_indices"]] == 0) == 1
