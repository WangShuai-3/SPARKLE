"""Tests for the RYTools-style GEM + scGEM loader."""

import gzip
import numpy as np
import pytest

from stambient.io_utils import load_RYTools_data


@pytest.fixture
def rytools_files(tmp_path):
    """Create a small GEM + scGEM file pair."""
    gem_path = tmp_path / "sample.gem.gz"
    scgem_path = tmp_path / "sample.scgem.gz"

    # Full GEM: x, y, geneID, MIDCounts
    gem_lines = [
        "x\ty\tgeneID\tMIDCounts",
        "1\t1\tGeneA\t2",
        "1\t1\tGeneB\t1",    # same DNB, second gene
        "2\t2\tGeneA\t3",    # cell DNB
        "2\t2\tGeneC\t1",
        "3\t3\tGeneA\t1",    # background DNB (not in scGEM)
        "3\t3\tGeneB\t2",
        "4\t4\tGeneA\t5",    # cell DNB
    ]

    # scGEM: x, y, geneID, MIDCounts, cell
    # Only cell DNBs; no background rows
    scgem_lines = [
        "x\ty\tgeneID\tMIDCounts\tcell",
        "2\t2\tGeneA\t3\tCell_1",
        "2\t2\tGeneC\t1\tCell_1",
        "4\t4\tGeneA\t5\tCell_2",
    ]

    with gzip.open(gem_path, "wt") as f:
        f.write("\n".join(gem_lines) + "\n")
    with gzip.open(scgem_path, "wt") as f:
        f.write("\n".join(scgem_lines) + "\n")

    return str(gem_path), str(scgem_path)


class TestRYToolsLoader:
    def test_basic_load(self, rytools_files):
        gem_path, scgem_path = rytools_files
        data = load_RYTools_data(gem_path, scgem_path, verbose=False)

        # 3 genes: GeneA, GeneB, GeneC
        assert data["spot_expr"].shape[0] == 3
        # 4 unique DNB positions: (1,1), (2,2), (3,3), (4,4)
        assert data["spot_expr"].shape[1] == 4
        # 2 cells: Cell_1, Cell_2
        assert len(data["cell_ids"]) == 2

        # Labels: (1,1) and (3,3) are empty, (2,2) and (4,4) are cells
        labels = data["spot_labels"]
        assert np.sum(labels == -1) == 2
        assert np.sum(labels >= 0) == 2

        # Cell IDs should be sorted original IDs from scGEM
        assert list(data["cell_ids"]) == ["Cell_1", "Cell_2"] or list(data["cell_ids"]) == [1, 2]
        # Actually our test uses string labels "Cell_1"; they will be sorted as strings
        assert list(data["cell_ids"]) == ["Cell_1", "Cell_2"]

    def test_counts_match(self, rytools_files):
        gem_path, scgem_path = rytools_files
        data = load_RYTools_data(gem_path, scgem_path, verbose=False)

        expr = data["spot_expr"].toarray()
        gene_names = list(data["gene_names"])

        # (1,1): GeneA=2, GeneB=1
        d1 = np.where((data["spot_coords"][:, 0] == 1) & (data["spot_coords"][:, 1] == 1))[0][0]
        assert expr[gene_names.index("GeneA"), d1] == 2.0
        assert expr[gene_names.index("GeneB"), d1] == 1.0

        # (2,2): GeneA=3, GeneC=1
        d2 = np.where((data["spot_coords"][:, 0] == 2) & (data["spot_coords"][:, 1] == 2))[0][0]
        assert expr[gene_names.index("GeneA"), d2] == 3.0
        assert expr[gene_names.index("GeneC"), d2] == 1.0

        # (3,3): GeneA=1, GeneB=2 (background)
        d3 = np.where((data["spot_coords"][:, 0] == 3) & (data["spot_coords"][:, 1] == 3))[0][0]
        assert data["spot_labels"][d3] == -1
        assert expr[gene_names.index("GeneA"), d3] == 1.0
        assert expr[gene_names.index("GeneB"), d3] == 2.0

        # (4,4): GeneA=5
        d4 = np.where((data["spot_coords"][:, 0] == 4) & (data["spot_coords"][:, 1] == 4))[0][0]
        assert expr[gene_names.index("GeneA"), d4] == 5.0

    def test_empty_labels_filter(self, tmp_path):
        """scGEM may contain explicit background rows; empty_labels should skip them."""
        gem_path = tmp_path / "gem.gem.gz"
        scgem_path = tmp_path / "scgem.scgem.gz"

        gem_lines = ["x\ty\tgeneID\tMIDCounts", "1\t1\tGeneA\t1", "2\t2\tGeneA\t2"]
        scgem_lines = ["x\ty\tgeneID\tMIDCounts\tcell", "1\t1\tGeneA\t1\t0", "2\t2\tGeneA\t2\t1"]

        with gzip.open(gem_path, "wt") as f:
            f.write("\n".join(gem_lines) + "\n")
        with gzip.open(scgem_path, "wt") as f:
            f.write("\n".join(scgem_lines) + "\n")

        data = load_RYTools_data(str(gem_path), str(scgem_path), empty_labels={0}, verbose=False)
        assert len(data["cell_ids"]) == 1
        assert data["cell_ids"][0] == "1"
        assert np.sum(data["spot_labels"] >= 0) == 1
        assert np.sum(data["spot_labels"] == -1) == 1

    def test_pitch_um_scaling(self, rytools_files):
        gem_path, scgem_path = rytools_files
        data = load_RYTools_data(gem_path, scgem_path, pitch_um=0.5, verbose=False)
        assert np.allclose(data["spot_coords"], np.array([[1, 1], [2, 2], [3, 3], [4, 4]]) * 0.5)

    def test_csv_delimiter(self, tmp_path):
        """Test custom comma delimiter."""
        gem_path = tmp_path / "gem.csv.gz"
        scgem_path = tmp_path / "scgem.csv.gz"

        gem_lines = ["x,y,geneID,MIDCounts", "1,1,GeneA,1", "2,2,GeneA,2"]
        scgem_lines = ["x,y,geneID,MIDCounts,cell", "2,2,GeneA,2,Cell1"]

        with gzip.open(gem_path, "wt") as f:
            f.write("\n".join(gem_lines) + "\n")
        with gzip.open(scgem_path, "wt") as f:
            f.write("\n".join(scgem_lines) + "\n")

        data = load_RYTools_data(
            str(gem_path), str(scgem_path),
            gem_sep=",", scgem_sep=",", verbose=False
        )
        assert data["spot_expr"].shape == (1, 2)
        assert len(data["cell_ids"]) == 1

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_RYTools_data("nonexistent.gem.gz", "nonexistent.scgem.gz", verbose=False)

    def test_missing_columns_raises(self, tmp_path):
        gem_path = tmp_path / "bad.gem.gz"
        scgem_path = tmp_path / "bad.scgem.gz"
        with gzip.open(gem_path, "wt") as f:
            f.write("a\tb\tc\n")
        with gzip.open(scgem_path, "wt") as f:
            f.write("x\ty\tgeneID\tMIDCounts\tcell\n")

        with pytest.raises(ValueError, match="Missing required columns in GEM"):
            load_RYTools_data(str(gem_path), str(scgem_path), verbose=False)
