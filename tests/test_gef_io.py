"""Tests for the native BGI GEF reader/writer (stambient.gef_io)."""

import numpy as np
import pytest
from scipy.sparse import csr_matrix

import h5py

from stambient.gef_io import (
    _assign_points_to_cells,
    _points_in_polygon,
    load_bgi_gef,
    read_bin_gef,
    read_cellbin_gef,
    save_cellbin_gef,
)

# Official SAW 8.1 demo file (mouse whole brain); ~737 MB, not committed.
# Download: https://demo.stomicsdb.tech/C04042E3_Mouse_Whole_Brain_Stereo-seq_FF_V1.3_HE/outs/C04042E3.cellbin.gef
_REAL_CELLBIN = (
    __file__.rsplit("/", 1)[0] + "/../data/bgi/C04042E3.cellbin.gef"
)
_REAL_RAW = (
    __file__.rsplit("/", 1)[0] + "/../data/bgi/C04042E3.raw.gef"
)


def _tiny_expression(n_genes=5, n_cells=6, seed=0):
    rng = np.random.default_rng(seed)
    mat = rng.poisson(3, size=(n_genes, n_cells)).astype(np.float64)
    mat[0, 0] = 0.0
    return csr_matrix(mat)


def _tiny_borders(centers_um, radius=3.0, n_pts=32):
    angles = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    borders = np.full((len(centers_um), n_pts, 2), np.nan)
    borders[:, :, 0] = centers_um[:, 0:1] + radius * np.cos(angles)[None, :]
    borders[:, :, 1] = centers_um[:, 1:2] + radius * np.sin(angles)[None, :]
    return borders


class TestSaveLoadRoundTrip:
    def test_round_trip_integer(self, tmp_path):
        expr = _tiny_expression()
        gene_names = np.array([f"G{i}" for i in range(expr.shape[0])])
        centers = np.column_stack([np.arange(6) * 10.0, np.arange(6) * 4.0])

        out = save_cellbin_gef(
            tmp_path / "s.cellbin.gef", expr, gene_names, centers, verbose=False
        )
        back = read_cellbin_gef(out, verbose=False)

        assert back["spot_expr"].shape == expr.shape
        assert np.allclose(back["spot_expr"].toarray(), expr.toarray())
        assert np.array_equal(back["gene_names"], gene_names)
        assert np.allclose(back["spot_coords"], centers)
        assert np.array_equal(back["cell_ids"], np.arange(6))
        assert back["format"] == "cellBin"
        assert not back.get("has_stored_float", False)

    def test_round_trip_float_sidecar(self, tmp_path):
        expr = _tiny_expression().multiply(0.87)
        gene_names = np.array([f"G{i}" for i in range(expr.shape[0])])
        centers = np.column_stack([np.arange(6) * 10.0, np.arange(6) * 4.0])
        out = save_cellbin_gef(
            tmp_path / "f.cellbin.gef", expr, gene_names, centers, verbose=False
        )
        back_float = read_cellbin_gef(out, verbose=False)
        back_int = read_cellbin_gef(
            out, prefer_stored_float=False, verbose=False
        )
        # Float path restores the exact values (float32 precision).
        assert np.allclose(back_float["spot_expr"].toarray(), expr.toarray(), atol=1e-5)
        # Integer path matches rounding.
        assert np.allclose(
            back_int["spot_expr"].toarray(), np.rint(expr.toarray()), atol=0
        )

    def test_negative_values_clipped(self, tmp_path):
        dense = np.array([[2.0, -3.0], [0.5, 1.0]])
        out = save_cellbin_gef(
            tmp_path / "c.cellbin.gef", dense, ["A", "B"],
            np.array([[0.0, 0.0], [10.0, 10.0]]), verbose=False,
        )
        back = read_cellbin_gef(out, verbose=False)
        values = back["spot_expr"].toarray()
        assert values.min() >= 0
        assert values[0, 0] == 2.0

    def test_over_uint16_raises(self, tmp_path):
        dense = np.array([[70000.0]])
        with pytest.raises(ValueError, match="uint16"):
            save_cellbin_gef(
                tmp_path / "big.cellbin.gef", dense, ["A"],
                np.array([[0.0, 0.0]]), verbose=False,
            )

    def test_file_layout_invariants(self, tmp_path):
        """The written file must satisfy the official format invariants."""
        expr = _tiny_expression(n_genes=4, n_cells=7, seed=3)
        gene_names = np.array([f"gene{i}" for i in range(4)])
        centers = np.column_stack([
            np.arange(7) * 30.0, (np.arange(7) % 3) * 25.0
        ])
        out = save_cellbin_gef(
            tmp_path / "inv.cellbin.gef", expr, gene_names, centers,
            cell_ids=np.array([10, 11, 12, 13, 14, 15, 16]), verbose=False,
        )
        with h5py.File(out, "r") as f:
            cell = f["cellBin/cell"][:]
            cellexp = f["cellBin/cellExp"][:]
            gene = f["cellBin/gene"][:]
            geneexp = f["cellBin/geneExp"][:]
            bs = f["cellBin/blockSize"][:]
            bi = f["cellBin/blockIndex"][:]

            # cellExp: cell-major, geneID ascending within each cell.
            assert int(cell["geneCount"].sum()) == len(cellexp)
            assert np.array_equal(
                cell["offset"],
                np.concatenate([[0], np.cumsum(cell["geneCount"])[:-1]]),
            )
            for i in range(len(cell) - 1):
                block = cellexp[
                    cell[i]["offset"]: cell[i]["offset"] + cell[i]["geneCount"]
                ]
                assert np.all(np.diff(block["geneID"]) > 0)

            # geneExp: gene-major, cellID ascending within each gene.
            for i in range(len(gene) - 1):
                block = geneexp[
                    gene[i]["offset"]: gene[i]["offset"] + gene[i]["cellCount"]
                ]
                assert np.all(np.diff(block["cellID"]) > 0)

            # blockIndex: cumulative row-major block occupancy.
            bx = (cell["x"] - 0) // bs[0]
            by = (cell["y"] - 0) // bs[1]
            counts = np.bincount(
                by * bs[2] + bx, minlength=int(bs[2]) * int(bs[3])
            )
            assert np.array_equal(
                bi, np.concatenate([[0], np.cumsum(counts)]).astype(np.uint32)
            )

            # Original cell ids are preserved, not overwritten by 0..n-1.
            assert set(cell["id"]) == {10, 11, 12, 13, 14, 15, 16}

            # File attributes required by readers (1-element arrays,
            # matching the official writer).
            assert f.attrs["bin_type"][0] == b"CellBin"
            assert int(f.attrs["version"][0]) == 4
            assert int(f.attrs["resolution"][0]) == 500

    def test_borders_and_areas_round_trip(self, tmp_path):
        centers = np.array([[10.0, 20.0], [50.0, 60.0], [90.0, 25.0]])
        borders = _tiny_borders(centers, radius=4.0)
        areas = np.array([100, 120, -1])  # last one unknown -> from polygon
        expr = _tiny_expression(n_genes=2, n_cells=3)
        out = save_cellbin_gef(
            tmp_path / "b.cellbin.gef", expr, ["A", "B"], centers,
            cell_areas=areas, cell_borders=borders, verbose=False,
        )
        back = read_cellbin_gef(out, verbose=False)
        # The file stores cells block-ordered; align via centroids.
        order = np.argsort(back["spot_coords"][:, 0])
        assert np.allclose(back["spot_coords"][order], centers)
        # Vertices are quantised to the raw grid (0.5 um at 500 nm).
        assert np.nanmax(np.abs(back["cell_borders"][order] - borders)) <= 0.26
        assert back["cell_areas"][order][0] == 100
        assert back["cell_areas"][order][2] == pytest.approx(50, abs=2)  # pi*4^2

    def test_custom_resolution(self, tmp_path):
        centers = np.array([[5.0, 5.0], [8.0, 9.0]])
        out = save_cellbin_gef(
            tmp_path / "r.cellbin.gef", np.array([[1.0, 2.0]]), ["A"], centers,
            resolution_nm=1000, verbose=False,  # 1 um per raw unit
        )
        back = read_cellbin_gef(out, verbose=False)
        assert back["resolution_nm"] == 1000
        assert np.allclose(back["spot_coords"], centers)
        with h5py.File(out, "r") as f:
            assert np.allclose(f["cellBin/cell"]["x"][:], [5, 8])


class TestLegacyLayout:
    def _write_legacy(self, path):
        cell_ids = np.array([3, 1, 2], dtype=np.int32)
        x = np.array([10, 40, 70], dtype=np.int32)
        y = np.array([20, 30, 40], dtype=np.int32)
        area = np.array([50, 60, 70], dtype=np.int32)
        # (cellId, geneid, count) triplets
        cid = np.array([3, 3, 1, 2, 2], dtype=np.int32)
        gid = np.array([0, 1, 1, 0, 1], dtype=np.int32)
        count = np.array([5, 7, 3, 2, 9], dtype=np.uint32)
        gene_name = np.array([b"GeneA", b"GeneB"])
        with h5py.File(path, "w") as f:
            cd = f.create_group("cellData")
            cd.create_dataset("cellId", data=cell_ids)
            cd.create_dataset("x", data=x)
            cd.create_dataset("y", data=y)
            cd.create_dataset("area", data=area)
            ce = f.create_group("cellExp")
            ce.create_dataset("cellId", data=cid)
            ce.create_dataset("geneid", data=gid)
            ce.create_dataset("count", data=count)
            ce.create_dataset("geneName", data=gene_name)
            ge = f.create_group("geneExp")
            ge.create_dataset(
                "geneName", data=np.array([b"GeneA", b"GeneB"])
            )

    def test_legacy_read(self, tmp_path):
        path = tmp_path / "legacy.cellbin.gef"
        self._write_legacy(path)
        data = read_cellbin_gef(path, verbose=False)
        assert data["format"] == "legacy"
        # cells sorted by original id: 1, 2, 3
        assert list(data["cell_ids"]) == [1, 2, 3]
        dense = data["spot_expr"].toarray()
        assert dense.shape == (2, 3)
        assert dense[1, 0] == 3  # gene 1, cell id 1
        assert dense[0, 2] == 5  # gene 0, cell id 3
        assert dense[1, 2] == 7
        assert list(data["gene_names"]) == ["GeneA", "GeneB"]
        assert np.allclose(data["spot_coords"][:, 0], [20, 35, 5])  # raw 40/70/10 x 0.5 um


class TestBinGef:
    def _write_bin_gef(self, path, resolution_nm=500):
        exp_dtype = np.dtype([("x", "<i4"), ("y", "<i4"), ("count", "<u2")])
        gene_dtype = np.dtype(
            [("geneID", "S64"), ("geneName", "S64"), ("offset", "<u4"), ("count", "<u4")]
        )
        # gene 0 at (10,10) c=2 and (11,12) c=1 ; gene 1 at (10,10) c=4
        expression = np.array(
            [(10, 10, 2), (11, 12, 1), (10, 10, 4)], dtype=exp_dtype
        )
        gene = np.array(
            [(b"E1", b"GeneA", 0, 2), (b"E2", b"GeneB", 2, 1)], dtype=gene_dtype
        )
        with h5py.File(path, "w") as f:
            f.attrs.create("version", 3, dtype=np.uint32)
            f.attrs.create("resolution", np.uint32(resolution_nm))
            grp = f.create_group("geneExp/bin1")
            ds = grp.create_dataset("expression", data=expression)
            ds.attrs.create("minX", 10)
            ds.attrs.create("minY", 10)
            ds.attrs.create("maxX", 11)
            ds.attrs.create("maxY", 12)
            grp.create_dataset("gene", data=gene)

    def test_bin_read(self, tmp_path):
        path = tmp_path / "t.gef"
        self._write_bin_gef(path)
        data = read_bin_gef(path, verbose=False)
        # two unique locations: (10,10) and (11,12)
        assert data["spot_expr"].shape == (2, 2)
        assert int((data["spot_labels"] == -1).all())
        coords = {tuple(c) for c in data["spot_coords"]}
        assert coords == {(5.0, 5.0), (5.5, 6.0)}  # x0.5 um
        dense = data["spot_expr"].toarray()
        col = np.flatnonzero(np.all(data["spot_coords"] == [5.0, 5.0], axis=1))[0]
        assert dense[0, col] == 2 and dense[1, col] == 4

    def test_region_crop(self, tmp_path):
        path = tmp_path / "t.gef"
        self._write_bin_gef(path)
        data = read_bin_gef(
            path, region_um=[5.0, 5.0, 5.0, 5.0], verbose=False
        )  # only (10,10) raw
        assert data["spot_expr"].shape == (2, 1)
        assert np.allclose(data["spot_coords"], [[5.0, 5.0]])

    def test_rejects_cellbin_file(self, tmp_path):
        out = save_cellbin_gef(
            tmp_path / "c.cellbin.gef", np.array([[1.0]]), ["A"],
            np.array([[0.0, 0.0]]), verbose=False,
        )
        with pytest.raises(ValueError, match="cellbin"):
            read_bin_gef(out, verbose=False)

    def test_missing_bin_raises(self, tmp_path):
        path = tmp_path / "t.gef"
        self._write_bin_gef(path)
        with pytest.raises(ValueError, match="not available"):
            read_bin_gef(path, bin_size=50, verbose=False)


class TestPointInPolygon:
    def test_square(self):
        vx = np.array([0, 10, 10, 0], dtype=float)
        vy = np.array([0, 0, 10, 10], dtype=float)
        px = np.array([5.0, 15.0, -1.0, 9.5, 0.5])
        py = np.array([5.0, 5.0, 5.0, 5.0, 0.5])
        res = _points_in_polygon(px, py, vx, vy)
        assert list(res) == [True, False, False, True, True]

    def test_nan_padding_ignored(self):
        vx = np.array([0, 10, 10, 0], dtype=float)
        vy = np.array([0, 0, 10, 10], dtype=float)
        centers = np.array([[5.0, 5.0]])
        borders = np.full((1, 8, 2), np.nan)
        borders[0, :4, 0] = vx
        borders[0, :4, 1] = vy
        labels = _assign_points_to_cells(
            np.array([[5.0, 5.0], [50.0, 50.0]]), centers, borders
        )
        assert list(labels) == [0, -1]

    def test_overlap_resolves_to_nearest(self):
        centers = np.array([[0.0, 0.0], [6.0, 0.0]])
        borders = np.full((2, 8, 2), np.nan)
        # two overlapping 10x10 squares centred on each cell
        for i, cx in enumerate([0.0, 6.0]):
            borders[i, :4, 0] = [cx - 5, cx + 5, cx + 5, cx - 5]
            borders[i, :4, 1] = [-5, -5, 5, 5]
        labels = _assign_points_to_cells(
            np.array([[1.0, 0.0], [5.0, 0.0]]), centers, borders
        )
        assert list(labels) == [0, 1]


class TestPairedLoading:
    def _make_pair(self, tmp_path):
        """One cell covering a 4x4 um square; DNBs inside and outside."""
        centers = np.array([[10.0, 10.0]])
        borders = np.full((1, 32, 2), np.nan)
        borders[0, :4, 0] = [8, 12, 12, 8]
        borders[0, :4, 1] = [8, 8, 12, 12]

        cellbin = tmp_path / "c.cellbin.gef"
        save_cellbin_gef(
            cellbin, csr_matrix(np.array([[3.0]])), ["GeneA"], centers,
            cell_areas=[16], cell_borders=borders, verbose=False,
        )

        # raw GEF: DNB grid at 1 um pitch around the cell
        exp_dtype = np.dtype([("x", "<i4"), ("y", "<i4"), ("count", "<u2")])
        gene_dtype = np.dtype(
            [("geneID", "S64"), ("geneName", "S64"), ("offset", "<u4"), ("count", "<u4")]
        )
        rows = []
        coords = [(9, 9), (10, 9), (11, 11), (10, 11), (13, 13), (20, 20)]
        counts = [2, 1, 4, 3, 5, 7]
        for (x, y), c in zip(coords, counts):
            rows.append((x, y, c))
        raw = tmp_path / "r.raw.gef"
        with h5py.File(raw, "w") as f:
            f.attrs.create("resolution", np.uint32(1000))  # 1 um units
            grp = f.create_group("geneExp/bin1")
            grp.create_dataset("expression", data=np.array(rows, dtype=exp_dtype))
            grp.create_dataset(
                "gene",
                data=np.array([(b"E1", b"GeneA", 0, len(rows))], dtype=gene_dtype),
            )
        return cellbin, raw, coords, counts

    def test_pair_assignment(self, tmp_path):
        cellbin, raw, coords, counts = self._make_pair(tmp_path)
        data = load_bgi_gef(cellbin, raw, verbose=False)

        labels = data["spot_labels"]
        coord_list = [tuple(c) for c in data["spot_coords"]]
        inside = {(9, 9), (10, 9), (11, 11), (10, 11)}
        for (x, y), lab in zip(coord_list, labels):
            expected = 0 if (x, y) in inside else -1
            assert lab == expected, (x, y, lab)

        assert data["n_mask_locations"] == 4
        assert data["n_out_of_mask_locations"] == 2
        total = data["spot_expr"].sum()
        assert total == sum(counts)

    def test_cellbin_only_returns_cell_level(self, tmp_path):
        cellbin, raw, _, _ = self._make_pair(tmp_path)
        data = load_bgi_gef(cellbin, verbose=False)
        assert data["spot_expr"].shape == (1, 1)
        assert int((data["spot_labels"] >= 0).all())

    def test_sparkle_fit_on_paired_data(self, tmp_path):
        from stambient import SPARKLE
        from stambient.io_utils import check_inputs

        cellbin, raw, _, _ = self._make_pair(tmp_path)
        data = load_bgi_gef(cellbin, raw, verbose=False)
        check_inputs(
            data["spot_expr"], data["spot_coords"], data["spot_labels"]
        )
        model = SPARKLE(bin_size=5.0, max_radius=50.0, verbose=False)
        corrected, diagnostics = model.fit_transform(
            data["spot_expr"], data["spot_coords"], data["spot_labels"]
        )
        assert corrected.shape == (1, 1)
        assert np.isfinite(corrected).all()


class TestRealFiles:
    """Regression against the official SAW 8.1 demo (downloaded manually)."""

    def test_read_official_cellbin(self):
        import os

        path = _REAL_CELLBIN
        if not os.path.exists(path):
            pytest.skip(f"official demo file not downloaded: {path}")
        data = read_cellbin_gef(path, verbose=False)
        assert data["spot_expr"].shape == (27570, 130588)
        assert data["spot_expr"].nnz == 46389781
        assert data["format"] == "cellBin"
        assert data["resolution_nm"] == 500

    def test_official_cellbin_matches_stored_expcounts(self):
        import os

        path = _REAL_CELLBIN
        if not os.path.exists(path):
            pytest.skip(f"official demo file not downloaded: {path}")
        data = read_cellbin_gef(path, verbose=False)
        with h5py.File(path, "r") as f:
            stored = f["cellBin/cell"]["expCount"][:].astype(np.int64)
        col_sums = np.asarray(data["spot_expr"].sum(axis=0)).ravel().astype(np.int64)
        assert np.array_equal(col_sums, stored)

    def test_official_pair_load_and_sparkle(self):
        import os

        if not (os.path.exists(_REAL_CELLBIN) and os.path.exists(_REAL_RAW)):
            pytest.skip("official demo pair not downloaded")
        region = [5000.0, 6000.0, 5000.0, 6000.0]  # 1 mm^2 window
        data = load_bgi_gef(
            _REAL_CELLBIN, _REAL_RAW, region_um=region, verbose=True
        )
        assert data["n_mask_locations"] > 1000
        assert data["n_out_of_mask_locations"] > 1000

        from stambient import SPARKLE

        model = SPARKLE(bin_size=25.0, max_radius=100.0, verbose=False)
        corrected, diagnostics = model.fit_transform(
            data["spot_expr"], data["spot_coords"], data["spot_labels"]
        )
        n_cells = len(np.unique(data["spot_labels"][data["spot_labels"] >= 0]))
        assert corrected.shape[1] == n_cells
        assert np.isfinite(corrected).all()
