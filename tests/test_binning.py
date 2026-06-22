"""Tests for binning module."""

import numpy as np
import pytest
from stambient.binning import dnb_to_bins


class TestBinning:
    def test_basic_binning(self):
        """Test binning with a small grid of DNBs."""
        # Create a 10×10 DNB grid at 500nm pitch
        pitch = 0.5  # μm
        n = 10
        x = np.arange(n) * pitch
        y = np.arange(n) * pitch
        xx, yy = np.meshgrid(x, y)
        coords = np.column_stack([xx.ravel(), yy.ravel()])
        n_dnbs = coords.shape[0]  # 100

        # DNB expression: 2 genes, all zero
        expr = np.zeros((2, n_dnbs), dtype=np.float64)

        # All empty labels
        labels = np.full(n_dnbs, -1, dtype=np.int64)

        Y_cell, Y_empty, n_cell, n_empty, bin_coords, bin_assignment, classification, n_total = \
            dnb_to_bins(expr, coords, labels, bin_size=5)

        # With bin_size=5, each bin is 2.5μm × 2.5μm → 25 DNBs
        # Total 100 DNBs → 4 bins (2×2)
        assert len(n_cell) == 4
        assert np.all(n_cell == 0)
        assert np.all(n_empty == 25)
        assert np.all(classification == 0)  # all pure empty

    def test_cell_assignment(self):
        """Test binning with cell-labeled DNBs."""
        pitch = 0.5
        n = 10
        x = np.arange(n) * pitch
        y = np.arange(n) * pitch
        xx, yy = np.meshgrid(x, y)
        coords = np.column_stack([xx.ravel(), yy.ravel()])
        n_dnbs = coords.shape[0]

        expr = np.zeros((2, n_dnbs), dtype=np.float64)
        expr[0, :] = 1.0  # gene 0 has 1 UMI per DNB

        # Half cell 0, half cell 1
        labels = np.full(n_dnbs, -1, dtype=np.int64)
        labels[:50] = 0
        labels[50:] = 1

        Y_cell, Y_empty, n_cell, n_empty, bin_coords, bin_assignment, classification, n_total = \
            dnb_to_bins(expr, coords, labels, bin_size=5)

        # All bins have only cell DNBs → pure cell
        for i in range(len(n_cell)):
            if n_total[i] > 0:
                assert classification[i] == 1

        # Check cell assignment
        for bid, d in bin_assignment.items():
            total_assigned = sum(d.values())
            assert total_assigned == n_cell[bid]

    def test_mixed_bin(self):
        """Test bin classification with pure cell and pure empty bins."""
        pitch = 0.5
        n = 4
        x = np.arange(n) * pitch
        y = np.arange(n) * pitch
        xx, yy = np.meshgrid(x, y)
        coords = np.column_stack([xx.ravel(), yy.ravel()])
        n_dnbs = coords.shape[0]  # 16

        expr = np.zeros((2, n_dnbs), dtype=np.float64)

        # Cell 0: first 2 rows (8 DNBs), rest empty
        labels = np.full(n_dnbs, -1, dtype=np.int64)
        labels[:8] = 0

        _, _, _, _, _, _, classification, _ = \
            dnb_to_bins(expr, coords, labels, bin_size=2)

        # Bins sorted by (x_idx, y_idx): (0,0), (0,1), (1,0), (1,1)
        # (0,0) = cols0-1, rows0-1 = 4 DNBs, all cell 0 → pure cell
        # (0,1) = cols0-1, rows2-3 = 4 DNBs, all empty → pure empty
        # (1,0) = cols2-3, rows0-1 = 4 DNBs, all cell 0 → pure cell
        # (1,1) = cols2-3, rows2-3 = 4 DNBs, all empty → pure empty
        assert classification[0] == 1  # pure cell
        assert classification[1] == 0  # pure empty
        assert classification[2] == 1  # pure cell
        assert classification[3] == 0  # pure empty

    def test_mixed_bin_actual(self):
        """Test that truly mixed bins are classified as mixed."""
        pitch = 0.5
        n = 4
        x = np.arange(n) * pitch
        y = np.arange(n) * pitch
        xx, yy = np.meshgrid(x, y)
        coords = np.column_stack([xx.ravel(), yy.ravel()])
        n_dnbs = coords.shape[0]  # 16

        expr = np.zeros((2, n_dnbs), dtype=np.float64)

        # bin_size=2 → 2×2 bins, each 4 DNBs
        # Pattern: col 0=cell, col 1=empty, col 2=cell, col 3=empty
        # Each bin contains 2 cell + 2 empty → mixed
        labels = np.full(n_dnbs, -1, dtype=np.int64)
        for i in range(n_dnbs):
            col = i % 4
            if col == 0 or col == 2:
                labels[i] = 0

        _, _, _, _, _, _, classification, _ = \
            dnb_to_bins(expr, coords, labels, bin_size=2)

        # Each bin: 2 cell + 2 empty (50%) → mixed (2)
        for c in classification:
            assert c == 2

    def test_expression_aggregation(self):
        """Test that expression is correctly aggregated."""
        pitch = 0.5
        n = 4
        x = np.arange(n) * pitch
        y = np.arange(n) * pitch
        xx, yy = np.meshgrid(x, y)
        coords = np.column_stack([xx.ravel(), yy.ravel()])
        n_dnbs = coords.shape[0]  # 16

        expr = np.zeros((1, n_dnbs), dtype=np.float64)
        expr[0, :] = 2.0  # each DNB has 2 UMIs

        # First 8 = cell 0, last 8 = empty
        labels = np.full(n_dnbs, -1, dtype=np.int64)
        labels[:8] = 0

        Y_cell, Y_empty, n_cell, n_empty, _, _, _, _ = \
            dnb_to_bins(expr, coords, labels, bin_size=2)

        # bin_size=2 → each bin = 1μm × 1μm → 4 DNBs per bin
        # 4 bins total
        assert len(n_cell) == 4

        # First 2 bins: 4 DNBs each, all cell → Y_cell = 8
        # Last 2 bins: 4 DNBs each, all empty → Y_empty = 8
        total_cell_expr = Y_cell.sum()
        total_empty_expr = Y_empty.sum()
        assert total_cell_expr == 16.0  # 8 DNBs × 2
        assert total_empty_expr == 16.0
