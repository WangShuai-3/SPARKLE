"""Tests for I/O utilities."""

import gzip
import numpy as np
import pytest
from pathlib import Path
from stambient.io_utils import load_visiumhd, load_stereoseq


pytest.importorskip("h5py")


# ── Visium HD fixtures ───────────────────────────────────────────────

def _make_minimal_visiumhd_h5(path):
    """Create a small synthetic Visium HD HDF5 file."""
    import h5py

    with h5py.File(path, "w") as f:
        # Tissue mask: 2x2 pixels
        mask = f.create_group("masks/filtered")
        mask.create_dataset("row", data=np.array([0, 0, 1, 1], dtype=np.int32))
        mask.create_dataset("col", data=np.array([0, 1, 0, 1], dtype=np.int32))

        # Features
        feat = f.create_group("features")
        feat.create_dataset("name", data=np.array([b"geneA", b"geneB"], dtype="S10"))

        # Feature slices: gene 0 = geneA, gene 1 = geneB
        fs = f.create_group("feature_slices")
        g0 = fs.create_group("0")
        g0.create_dataset("row", data=np.array([0, 1], dtype=np.int32))
        g0.create_dataset("col", data=np.array([0, 0], dtype=np.int32))
        g0.create_dataset("data", data=np.array([5.0, 3.0], dtype=np.float64))

        g1 = fs.create_group("1")
        g1.create_dataset("row", data=np.array([0, 1], dtype=np.int32))
        g1.create_dataset("col", data=np.array([1, 1], dtype=np.int32))
        g1.create_dataset("data", data=np.array([2.0, 4.0], dtype=np.float64))

        # Segmentation: pixel (0,0) and (1,1) -> cell 7, others empty
        seg = f.create_group("segmentations/cell_segmentation_mask")
        seg.create_dataset("row", data=np.array([0, 1], dtype=np.int32))
        seg.create_dataset("col", data=np.array([0, 1], dtype=np.int32))
        seg.create_dataset("data", data=np.array([7, 7], dtype=np.int64))


class TestLoadVisiumHD:
    def test_basic_load(self, tmp_path):
        h5_path = tmp_path / "test.h5"
        _make_minimal_visiumhd_h5(h5_path)

        data = load_visiumhd(h5_path, verbose=False)

        assert data["spot_expr"].shape == (2, 4)
        assert data["spot_coords"].shape == (4, 2)
        assert len(data["spot_labels"]) == 4
        assert list(data["gene_names"]) == ["geneA", "geneB"]
        assert list(data["cell_ids"]) == [7]

        # Labels: pixel 0 (0,0) -> cell, pixel 3 (1,1) -> cell, others empty
        assert data["spot_labels"][0] == 0
        assert data["spot_labels"][3] == 0
        assert data["spot_labels"][1] == -1
        assert data["spot_labels"][2] == -1

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_visiumhd(tmp_path / "does_not_exist.h5", verbose=False)


# ── Stereo-seq GEM fixtures ───────────────────────────────────────────

def _write_gem(path, content, gzip_compress=False):
    """Write a GEM file, optionally gzip-compressed."""
    if gzip_compress:
        with gzip.open(path, "wt", encoding="utf-8") as f:
            f.write(content)
    else:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)


def _minimal_gem_tsv():
    return (
        "# comment line\n"
        "geneID\tx\ty\tMIDCounts\tcell\n"
        "geneA\t0\t0\t5\t1\n"
        "geneA\t1\t0\t3\t1\n"
        "geneB\t0\t0\t2\t1\n"
        "geneB\t1\t1\t4\t2\n"
        "geneA\t0\t1\t1\t0\n"
    )


class TestLoadStereoSeq:
    def test_basic_tsv(self, tmp_path):
        gem_path = tmp_path / "test.tsv"
        _write_gem(gem_path, _minimal_gem_tsv())

        data = load_stereoseq(gem_path, verbose=False)

        assert data["spot_expr"].shape == (2, 4)
        assert data["spot_coords"].shape == (4, 2)
        assert len(data["spot_labels"]) == 4
        assert list(data["gene_names"]) == ["geneA", "geneB"]
        assert list(data["cell_ids"]) == [1, 2]

        # DNB order is first-encounter: (0,0), (1,0), (1,1), (0,1)
        labels = data["spot_labels"]
        assert labels[0] == 0  # cell 1
        assert labels[1] == 0  # cell 1
        assert labels[2] == 1  # cell 2
        assert labels[3] == -1  # empty

        # geneA total = 5 + 3 + 1 = 9; geneB total = 2 + 4 = 6
        totals = np.asarray(data["spot_expr"].sum(axis=1)).ravel()
        assert totals[0] == pytest.approx(9.0)
        assert totals[1] == pytest.approx(6.0)

    def test_gzipped(self, tmp_path):
        gem_path = tmp_path / "test.tsv.gz"
        _write_gem(gem_path, _minimal_gem_tsv(), gzip_compress=True)

        data = load_stereoseq(gem_path, verbose=False)
        assert data["spot_expr"].shape == (2, 4)
        assert list(data["gene_names"]) == ["geneA", "geneB"]

    def test_pitch_converts_raw_coordinates_to_micrometres(self, tmp_path):
        gem_path = tmp_path / "test.tsv"
        _write_gem(gem_path, _minimal_gem_tsv())

        data = load_stereoseq(gem_path, pitch_um=0.5, verbose=False)

        np.testing.assert_allclose(
            data["spot_coords"],
            [[0.0, 0.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5]],
        )

    @pytest.mark.parametrize("pitch_um", [0.0, -0.5, np.inf, np.nan])
    def test_invalid_pitch_is_rejected(self, tmp_path, pitch_um):
        gem_path = tmp_path / "test.tsv"
        _write_gem(gem_path, _minimal_gem_tsv())

        with pytest.raises(ValueError, match="positive finite"):
            load_stereoseq(gem_path, pitch_um=pitch_um, verbose=False)

    def test_csv(self, tmp_path):
        gem_path = tmp_path / "test.csv"
        content = (
            "# comment\n"
            "geneID,x,y,MIDCounts,cell\n"
            "geneA,0,0,5,1\n"
            "geneB,1,1,4,2\n"
        )
        _write_gem(gem_path, content)

        data = load_stereoseq(gem_path, verbose=False)
        assert data["spot_expr"].shape == (2, 2)
        assert list(data["gene_names"]) == ["geneA", "geneB"]

    def test_custom_columns(self, tmp_path):
        gem_path = tmp_path / "test.tsv"
        content = (
            "# comment\n"
            "gene\tx\ty\tumi_count\tcell_label\n"
            "geneA\t0\t0\t5\t1\n"
            "geneB\t1\t1\t4\t2\n"
        )
        _write_gem(gem_path, content)

        data = load_stereoseq(
            gem_path,
            gene_col="gene",
            count_col="umi_count",
            cell_label_col="cell_label",
            verbose=False,
        )
        assert data["spot_expr"].shape == (2, 2)
        assert list(data["gene_names"]) == ["geneA", "geneB"]
        assert list(data["cell_ids"]) == [1, 2]

    def test_custom_empty_label(self, tmp_path):
        gem_path = tmp_path / "test.tsv"
        content = (
            "geneID\tx\ty\tMIDCounts\tcell\n"
            "geneA\t0\t0\t5\t-1\n"
            "geneB\t1\t1\t4\t10\n"
        )
        _write_gem(gem_path, content)

        data = load_stereoseq(gem_path, empty_labels={-1}, verbose=False)
        assert data["spot_labels"][0] == -1
        assert data["spot_labels"][1] == 0

    def test_missing_column(self, tmp_path):
        gem_path = tmp_path / "test.tsv"
        content = "geneID\tx\ty\tMIDCounts\n"
        _write_gem(gem_path, content)

        with pytest.raises(ValueError, match="Missing required columns"):
            load_stereoseq(gem_path, verbose=False)

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_stereoseq(tmp_path / "does_not_exist.tsv", verbose=False)

    def test_no_header(self, tmp_path):
        gem_path = tmp_path / "test.tsv"
        _write_gem(gem_path, "geneA\t0\t0\t5\t1\n")

        # A single data line is interpreted as the header; the required
        # column names are then missing.
        with pytest.raises(ValueError, match="Missing required columns"):
            load_stereoseq(gem_path, verbose=False)
