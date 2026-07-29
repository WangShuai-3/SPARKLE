"""Tests for the SPARKLE facade class."""

import numpy as np
import pytest

from stambient import SPARKLE


def _small_spatial_data(seed=3):
    rng = np.random.RandomState(seed)
    x, y = np.meshgrid(np.arange(24, dtype=float), np.arange(24, dtype=float))
    coords = np.column_stack((x.ravel(), y.ravel()))
    labels = np.full(len(coords), -1, dtype=np.int64)
    for cell_id, center in enumerate([(6, 6), [18, 6], [6, 18], [18, 18]]):
        inside = np.linalg.norm(coords - center, axis=1) <= 4.5
        labels[inside] = cell_id
    expr = rng.poisson(0.3, size=(8, len(coords))).astype(np.float64)
    for cell_id in range(4):
        mask = labels == cell_id
        expr[cell_id, mask] += rng.poisson(3.0, size=mask.sum())
    return expr, coords, labels


def test_cell_based_false_raises():
    """The legacy bin-level pipeline has been removed."""
    with pytest.raises(ValueError, match="cell_based=False"):
        SPARKLE(cell_based=False)


def test_cell_based_default_accepted():
    model = SPARKLE()
    assert model.cell_based is True
    assert model.lambda_ is None
    assert model.diagnostics_ is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"bin_size": 0.0},
        {"bin_size": -1.0},
        {"bin_size": float("nan")},
        {"max_radius": 0.0},
        {"max_radius": -5.0},
        {"lambda_grid": [10.0, 0.0]},
        {"lambda_grid": [-1.0]},
    ],
)
def test_invalid_lengths_raise(kwargs):
    with pytest.raises(ValueError):
        SPARKLE(**kwargs)


def test_inverse_metric_skips_lambda_search(capsys):
    """The inverse kernel does not use λ: no search, λ reported as None."""
    expr, coords, labels = _small_spatial_data()
    model = SPARKLE(
        distance_metric="inverse",
        bin_size=4,
        max_radius=30,
        lambda_grid=[5, 10, 20],
        n_lambda_genes=4,
        verbose=True,
    )
    corrected, diag = model.fit_transform(expr, coords, labels)

    assert diag["lambda_estimated"] is None
    assert model.lambda_ is None
    assert "skipping the λ grid search" in capsys.readouterr().out

    # The grid contents must not influence the result.
    other = SPARKLE(
        distance_metric="inverse",
        bin_size=4,
        max_radius=30,
        lambda_grid=[99, 55, 11],
        n_lambda_genes=4,
        verbose=False,
    )
    corrected_other, diag_other = other.fit_transform(expr, coords, labels)
    np.testing.assert_array_equal(corrected, corrected_other)
    np.testing.assert_array_equal(diag["alphas"], diag_other["alphas"])


def test_exponential_metric_reports_estimated_lambda():
    """λ-dependent kernels still run the grid search and report a value."""
    expr, coords, labels = _small_spatial_data()
    model = SPARKLE(
        distance_metric="exponential",
        bin_size=4,
        max_radius=30,
        lambda_grid=[5, 10, 20],
        n_lambda_genes=4,
        verbose=False,
    )
    _, diag = model.fit_transform(expr, coords, labels)

    assert diag["lambda_estimated"] in {5.0, 10.0, 20.0}
    assert model.lambda_ == diag["lambda_estimated"]
