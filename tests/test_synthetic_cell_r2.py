import numpy as np
import pytest

from evaluation.scripts.evaluate_synthetic_cell_r2 import cellwise_pearson_r2


def test_cellwise_pearson_r2_is_scale_and_offset_invariant():
    truth = np.array([[1.0, 2.0, 4.0], [2.0, 5.0, 3.0]])
    pred = truth * np.array([[3.0], [0.5]]) + np.array([[7.0], [2.0]])
    np.testing.assert_allclose(cellwise_pearson_r2(pred, truth), 1.0)


def test_cellwise_pearson_r2_matches_known_correlations():
    truth = np.array([[0.0, 1.0, 2.0], [0.0, 1.0, 0.0]])
    pred = np.array([[2.0, 1.0, 0.0], [1.0, 0.0, 1.0]])
    np.testing.assert_allclose(cellwise_pearson_r2(pred, truth), [1.0, 1.0])


def test_cellwise_pearson_r2_marks_constant_profiles_undefined():
    truth = np.array([[1.0, 1.0, 1.0], [0.0, 1.0, 2.0]])
    pred = np.array([[1.0, 2.0, 3.0], [5.0, 5.0, 5.0]])
    assert np.isnan(cellwise_pearson_r2(pred, truth)).all()


def test_cellwise_pearson_r2_validates_shape():
    with pytest.raises(ValueError, match="matching cells-by-genes"):
        cellwise_pearson_r2(np.ones((2, 3)), np.ones((3, 2)))
