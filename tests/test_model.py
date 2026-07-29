"""Tests for the SPARKLE facade class."""

import pytest

from stambient import SPARKLE


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
