"""Tests for correction module."""

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from stambient.correction import correct_expression


class TestCorrection:
    def test_no_correction_for_low_r2(self):
        """Genes with R² below threshold should not be corrected."""
        n_bins = 4
        n_genes = 3

        coords = np.array([
            [0.0, 0.0], [10.0, 0.0], [20.0, 0.0], [30.0, 0.0]
        ])
        n_cell = np.array([10, 10, 10, 10], dtype=np.int64)
        n_empty = np.array([0, 0, 0, 0], dtype=np.int64)
        n_total = n_cell + n_empty

        Y_cell_data = np.ones((n_genes, n_bins), dtype=np.float64) * 10.0
        Y_cell = csr_matrix(Y_cell_data)
        Y_empty = csr_matrix(np.zeros((n_genes, n_bins)))

        # All bins belong to cell 0
        bin_cell_assignment = {i: {0: n_cell[i]} for i in range(n_bins)}

        gene_indices = np.array([0, 1, 2])
        alphas = np.array([0.01, 0.02, 0.03])
        r2_scores = np.array([0.001, 0.002, 0.003])  # all below threshold
        beta = np.ones(n_bins)

        corrected, amb_frac, corr_mag = correct_expression(
            Y_cell, Y_empty, n_cell, n_empty, n_total, coords,
            bin_cell_assignment, gene_indices, alphas, r2_scores,
            r2_threshold=0.05, lam=50.0, max_radius=100.0,
            beta=beta, metric="exponential",
        )

        # Only 1 cell, all genes should keep original expression
        assert corrected.shape == (n_genes, 1)
        # Each bin contributes 10, 4 bins per cell → sum = 40
        for g in range(n_genes):
            assert corrected[g, 0] == pytest.approx(40.0)

        assert np.all(amb_frac == 0.0)

    def test_correction_reduces_expression(self):
        """Test that correction actually reduces expression."""
        n_bins = 4
        n_genes = 2

        coords = np.array([
            [0.0, 0.0], [10.0, 0.0], [20.0, 0.0], [30.0, 0.0]
        ])
        n_cell = np.array([10, 10, 10, 10], dtype=np.int64)
        n_empty = np.array([0, 0, 0, 0], dtype=np.int64)
        n_total = n_cell + n_empty

        Y_cell_data = np.ones((n_genes, n_bins), dtype=np.float64) * 100.0
        Y_cell = csr_matrix(Y_cell_data)
        Y_empty = csr_matrix(np.zeros((n_genes, n_bins)))

        # Bins 0,1 → cell 0; bins 2,3 → cell 1
        bin_cell_assignment = {
            0: {0: 10},
            1: {0: 10},
            2: {1: 10},
            3: {1: 10},
        }

        gene_indices = np.array([0, 1])
        alphas = np.array([0.01, 0.01])
        r2_scores = np.array([0.5, 0.5])  # above threshold
        beta = np.ones(n_bins)

        corrected, amb_frac, corr_mag = correct_expression(
            Y_cell, Y_empty, n_cell, n_empty, n_total, coords,
            bin_cell_assignment, gene_indices, alphas, r2_scores,
            r2_threshold=0.05, lam=50.0, max_radius=100.0,
            beta=beta, metric="exponential",
        )

        # 2 cells
        assert corrected.shape == (2, 2)

        # Each cell gets 2 bins × 100 UMIs = 200 before correction
        # After correction, should be less (ambient from other cell)
        assert corrected[0, 0] < 200.0  # reduced
        assert corrected[0, 0] >= 0.0  # non-negative

    def test_non_negative_output(self):
        """Test that corrected expression is always non-negative."""
        n_bins = 3
        n_genes = 1

        coords = np.array([
            [0.0, 0.0], [5.0, 0.0], [10.0, 0.0]
        ])
        n_cell = np.array([10, 10, 10], dtype=np.int64)
        n_empty = np.array([0, 0, 0], dtype=np.int64)
        n_total = n_cell + n_empty

        Y_cell_data = np.array([[5.0, 5.0, 5.0]])
        Y_cell = csr_matrix(Y_cell_data)
        Y_empty = csr_matrix(np.zeros((1, 3)))

        bin_cell_assignment = {i: {i: 10} for i in range(3)}

        gene_indices = np.array([0])
        alphas = np.array([1.0])  # Very high α → strong correction
        r2_scores = np.array([0.8])
        beta = np.ones(3)

        corrected, _, _ = correct_expression(
            Y_cell, Y_empty, n_cell, n_empty, n_total, coords,
            bin_cell_assignment, gene_indices, alphas, r2_scores,
            r2_threshold=0.05, lam=5.0, max_radius=50.0,
            beta=beta, metric="exponential",
        )

        # Output should have no negative values
        assert corrected.shape == (1, 3)
        for c in range(3):
            assert corrected[0, c] >= 0.0
