"""Tests for estimation module."""

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from stambient.estimation import (
    select_high_expression_genes,
    estimate_alpha_per_gene,
)


class TestGeneSelection:
    def test_basic_selection(self):
        """Test high-expr gene selection."""
        # 5 genes, 3 bins: bins 0 and 2 have empty DNBs
        Y_empty = csr_matrix(np.array([
            [0, 0, 100],   # gene 0: high empty expr in bin 2
            [10, 0, 5],    # gene 1
            [0, 0, 0],     # gene 2: no empty expr
            [5, 0, 3],     # gene 3
            [0, 0, 1],     # gene 4
        ], dtype=np.float64))
        n_empty = np.array([10, 0, 20])

        genes = select_high_expression_genes(Y_empty, n_empty, n_high=3)
        # Gene 0: 100/30 = 3.33 per-DNB
        # Gene 1: 15/30 = 0.5
        # Gene 3: 8/30 = 0.267
        # Gene 4: 1/30 = 0.033
        # Gene 2: 0
        assert genes[0] == 0
        assert genes[1] == 1
        assert genes[2] == 3


class TestAlphaEstimation:
    def test_perfect_recovery(self):
        """Test that α is recovered under perfect conditions.

        Set up data where Y_empty = α_true * N_gb exactly (no noise).
        """
        n_bins = 4
        n_genes = 3

        # Coordinates: line of bins
        coords = np.array([
            [0.0, 0.0],
            [10.0, 0.0],
            [20.0, 0.0],
            [30.0, 0.0],
        ])

        # Bins 0, 3 = cell bins; bins 1, 2 = empty bins
        n_cell = np.array([10, 0, 0, 10], dtype=np.int64)
        n_empty = np.array([0, 20, 20, 0], dtype=np.int64)

        # Create expression with known α
        true_alpha = np.array([0.01, 0.02, 0.0])
        lam = 50.0

        # Bin 0 and 3 are cell bins expressing gene 0 at high level
        Y_cell_data = np.zeros((n_genes, n_bins), dtype=np.float64)
        Y_cell_data[0, 0] = 100.0
        Y_cell_data[0, 3] = 100.0
        Y_cell_data[1, 0] = 50.0
        Y_cell_data[1, 3] = 50.0
        Y_cell = csr_matrix(Y_cell_data)

        # Compute expected empty expression
        empty_bin_mask = np.array([False, True, True, False])

        # Source strengths
        source0 = Y_cell_data[0] / n_cell
        source0 = np.nan_to_num(source0, nan=0.0)
        source1 = Y_cell_data[1] / n_cell
        source1 = np.nan_to_num(source1, nan=0.0)

        # Manual neighbor sum
        # Bin 1 neighbors: bin 0 (10 μm), bin 3 (20 μm)
        d_10 = np.exp(-10.0 / lam)
        d_20 = np.exp(-20.0 / lam)
        N_1_0 = 20 * (d_10 * source0[0] + d_20 * source0[3])
        N_1_1 = 20 * (d_10 * source1[0] + d_20 * source1[3])

        # Bin 2 neighbors: bin 0 (20 μm), bin 3 (10 μm)
        N_2_0 = 20 * (d_20 * source0[0] + d_10 * source0[3])
        N_2_1 = 20 * (d_20 * source1[0] + d_10 * source1[3])

        # Set Y_empty exactly as α * N (no noise)
        Y_empty_data = np.zeros((n_genes, n_bins), dtype=np.float64)
        Y_empty_data[0, 1] = true_alpha[0] * N_1_0
        Y_empty_data[0, 2] = true_alpha[0] * N_2_0
        Y_empty_data[1, 1] = true_alpha[1] * N_1_1
        Y_empty_data[1, 2] = true_alpha[1] * N_2_1
        Y_empty = csr_matrix(Y_empty_data)

        alphas, r2, _ = estimate_alpha_per_gene(
            Y_empty, Y_cell, n_empty, n_cell, coords,
            np.array([0, 1, 2]), empty_bin_mask,
            lam=lam, max_radius=50.0, metric="exponential",
        )

        assert alphas[0] == pytest.approx(true_alpha[0], rel=1e-3)
        assert alphas[1] == pytest.approx(true_alpha[1], rel=1e-3)
        assert alphas[2] == pytest.approx(0.0, abs=1e-6)
        # R² should be near 1 for perfect fit (floating point may give ~1-1e-15)
        assert r2[0] == pytest.approx(1.0, rel=1e-3)
        assert r2[1] == pytest.approx(1.0, rel=1e-3)

    def test_no_empty_bins(self):
        """Test that α=0 when no empty bins exist."""
        n_bins = 4
        coords = np.array([
            [0.0, 0.0], [10.0, 0.0], [20.0, 0.0], [30.0, 0.0]
        ])
        n_cell = np.array([10, 10, 10, 10], dtype=np.int64)
        n_empty = np.zeros(4, dtype=np.int64)

        Y_cell = csr_matrix(np.ones((2, n_bins)))
        Y_empty = csr_matrix(np.zeros((2, n_bins)))

        empty_bin_mask = np.array([False, False, False, False])

        alphas, r2, _ = estimate_alpha_per_gene(
            Y_empty, Y_cell, n_empty, n_cell, coords,
            np.array([0, 1]), empty_bin_mask,
            lam=50.0, max_radius=100.0,
        )

        assert alphas[0] == 0.0
        assert alphas[1] == 0.0
