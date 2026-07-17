import numpy as np
import pytest
from scipy.sparse import csr_matrix

from evaluation.baselines.spotclean import (
    run_spotclean,
    run_spotclean_sparse_cells,
    run_spotclean_with_background,
)


def _fixture():
    # Two DNBs per cell plus two empty DNBs.  Empty columns contain deliberately
    # large counts so the test can detect accidental background use.
    expr = csr_matrix(
        np.array(
            [
                [8, 6, 1, 2, 100, 200],
                [0, 1, 7, 9, 300, 400],
            ],
            dtype=np.float64,
        )
    )
    coords = np.array(
        [[0, 0], [0, 1], [10, 0], [10, 1], [5, 5], [20, 20]],
        dtype=np.float64,
    )
    labels = np.array([0, 0, 1, 1, -1, -1], dtype=np.int64)
    return expr, coords, labels


def test_cell_spotclean_ignores_empty_dnbs_and_conserves_gene_totals():
    expr, coords, labels = _fixture()
    corrected, diag = run_spotclean(
        expr,
        coords,
        labels,
        candidate_bandwidths=[5, 20],
        bleed_rate=0.1,
        distal_rate=0.1,
        maxit=10,
        tol=1e-8,
    )

    expected_cell_totals = np.array([17.0, 17.0])
    assert corrected.shape == (2, 2)
    assert np.all(corrected >= 0)
    assert np.allclose(corrected.sum(axis=1), expected_cell_totals)
    assert diag["background_used"] is False
    assert diag["empty_dnbs_used"] is False
    assert diag["max_gene_count_conservation_error"] < 1e-10

    changed = expr.toarray().copy()
    changed[:, labels < 0] *= 1000
    changed_corrected, _ = run_spotclean(
        csr_matrix(changed),
        coords,
        labels,
        candidate_bandwidths=[5, 20],
        bleed_rate=0.1,
        distal_rate=0.1,
        maxit=10,
        tol=1e-8,
    )
    assert np.allclose(changed_corrected, corrected)


def test_cell_spotclean_one_cell_is_unchanged():
    expr = csr_matrix(np.array([[2, 3, 99], [1, 0, 99]], dtype=np.float64))
    coords = np.array([[0, 0], [1, 0], [2, 0]], dtype=np.float64)
    labels = np.array([7, 7, -1], dtype=np.int64)

    corrected, diag = run_spotclean(expr, coords, labels, candidate_bandwidths=[10])
    assert np.array_equal(corrected, np.array([[5.0], [1.0]]))
    assert diag["n_cells"] == 1


def test_background_spotclean_uses_exposure_normalized_empty_bins():
    expr, coords, labels = _fixture()
    corrected, diag = run_spotclean_with_background(
        expr,
        coords,
        labels,
        empty_bin_size=5,
        candidate_bandwidths=[5, 20],
        maxit=20,
        tol=1e-8,
    )

    # Each empty bin contains one DNB while the median cell contains two, so
    # the background observations are scaled by two before concatenation.
    expected_totals = np.array([17.0, 17.0]) + 2.0 * np.array([300.0, 700.0])
    assert corrected.shape == (2, 2)
    assert np.all(corrected >= 0)
    assert np.allclose(corrected.sum(axis=1), expected_totals)
    assert diag["background_used"] is True
    assert diag["empty_dnbs_used"] is True
    assert diag["n_empty_bins"] == 2
    assert diag["empty_exposure_normalized"] is True
    assert diag["empty_exposure_target_dnb_area"] == 2.0
    assert diag["empty_exposure_scale_min"] == 2.0
    assert diag["empty_exposure_scale_max"] == 2.0
    assert 0.0 <= diag["bleeding_rate"] <= 1.0
    assert 0.1 <= diag["distal_rate"] <= 1.0
    assert diag["max_gene_count_conservation_error"] < 1e-8


def test_background_spotclean_rejects_missing_empty_dnbs():
    expr, coords, labels = _fixture()
    with pytest.raises(ValueError, match="requires empty DNBs"):
        run_spotclean_with_background(expr[:, :4], coords[:4], labels[:4])


def test_background_spotclean_can_use_literal_unscaled_concatenation():
    expr, coords, labels = _fixture()
    corrected, diag = run_spotclean_with_background(
        expr,
        coords,
        labels,
        empty_bin_size=5,
        normalize_empty_exposure=False,
        candidate_bandwidths=[5],
        maxit=10,
    )
    expected_totals = np.array([17.0, 17.0]) + np.array([300.0, 700.0])
    assert np.allclose(corrected.sum(axis=1), expected_totals)
    assert diag["empty_exposure_normalized"] is False
    assert diag["empty_exposure_scale_min"] == 1.0
    assert diag["empty_exposure_scale_max"] == 1.0


@pytest.mark.parametrize(
    "keyword,value",
    [("bleed_rate", 1.0), ("distal_rate", 1.1), ("tol", -1.0)],
)
def test_cell_spotclean_validates_parameters(keyword, value):
    expr, coords, labels = _fixture()
    with pytest.raises(ValueError):
        run_spotclean(expr, coords, labels, **{keyword: value})


def test_background_spotclean_validates_empty_bin_size():
    expr, coords, labels = _fixture()
    with pytest.raises(ValueError, match="empty_bin_size"):
        run_spotclean_with_background(expr, coords, labels, empty_bin_size=0)


def test_sparse_cell_spotclean_matches_dense_when_all_neighbors_are_used():
    expr, coords, labels = _fixture()
    dense, _ = run_spotclean(
        expr,
        coords,
        labels,
        candidate_bandwidths=[5],
        maxit=3,
        tol=1e-8,
    )
    cell_expr = np.array([[14, 3], [1, 16]], dtype=np.float64)
    cell_coords = np.array([[0, 0.5], [10, 0.5]], dtype=np.float64)
    streamed = np.empty_like(cell_expr)

    returned, diag = run_spotclean_sparse_cells(
        csr_matrix(cell_expr),
        cell_coords,
        candidate_bandwidths=[5],
        n_neighbors=1,
        n_selection_genes=2,
        gene_batch_size=1,
        maxit=3,
        tol=1e-8,
        n_jobs=2,
        batch_writer=lambda start, end, values: streamed.__setitem__(
            slice(start, end), values
        ),
    )

    assert returned is None
    assert np.allclose(streamed, dense)
    assert diag["background_used"] is False
    assert diag["n_neighbors"] == 1
    assert diag["max_gene_count_conservation_error"] < 1e-8
