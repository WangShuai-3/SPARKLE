"""Ambient-source weighting helpers used by the SPARKLE pipeline.

Two independent weighting ideas appear in the model:

- ``expression_weight``: soft-sigmoid source weighting that suppresses
  non-expressing cells so they do not dilute the predicted ambient signal.
- ``self_confidence_weight``: per-cell penalty applied during correction so
  high-expressing cells are not over-subtracted by their own signal.

Both helpers are pure NumPy functions shared by the lambda search, alpha
estimation, and correction stages.
"""

import numpy as np


def expression_weight(source: np.ndarray) -> np.ndarray:
    """Expression-weight ambient source to suppress non-expressing cells.

    W(s) = s / (s + s_median) where s_median is the median of positive rates.
    This is a soft sigmoid: high-expressors weight → 1, low-expressors → 0.5
    or lower, preventing non-expressing cells from diluting the signal.

    For cell-type-specific genes (e.g., SST): non-expressing cells
    (rate ~0.005) get w ≈ 0.5 when median = 0.005; expressing cells
    (rate ~0.17) get w ≈ 0.97 → 2× relative weight boost.
    For housekeeping genes (uniform ~0.15): all cells get w ≈ 0.5.
    """
    positive = source[source > 0]
    if len(positive) == 0:
        return np.zeros_like(source)
    s_median = np.median(positive)
    if s_median <= 0:
        return source
    weight = source / (source + s_median)
    return source * weight  # s * s/(s+median) = s²/(s+median)


def self_confidence_weight(source_c: np.ndarray, mode: str = "1/(1+s/p90)") -> np.ndarray:
    """Per-cell penalty factor. Modes:
    - '1/(1+s/p90)': sigmoid, p90 ref (default)
    - '1/(1+s/p50)': sigmoid, median ref (stronger)
    - '1/(1+(s/p90)²)': quadratic sigmoid (sharp cutoff)
    - 'exp(-s/p90)': exponential decay
    - 'p50/s': linear ramp above median, 1 below

    Supports both 1D (single gene) and 2D (batch of genes, last axis = cells).
    """
    if source_c.ndim == 2:
        return np.array([self_confidence_weight(row, mode=mode) for row in source_c])

    positive = source_c[source_c > 0]
    if len(positive) == 0:
        return np.ones_like(source_c)
    p50 = np.percentile(positive, 50)
    p90 = np.percentile(positive, 90)
    if p50 <= 0 or p90 <= 0:
        return np.ones_like(source_c)

    if mode == "1/(1+s/p50)":
        return 1.0 / (1.0 + source_c / p50)
    elif mode == "1/(1+(s/p90)²)":
        return 1.0 / (1.0 + (source_c / p90) ** 2)
    elif mode == "exp(-s/p90)":
        return np.exp(-source_c / p90)
    elif mode == "p50/s":
        return np.where(source_c < p50, 1.0, p50 / source_c)
    else:  # default: 1/(1+s/p90)
        return 1.0 / (1.0 + source_c / p90)


# Backward-compatible private aliases (the pipeline historically exposed the
# underscore-prefixed names from stambient.cell_pipeline).
_expression_weight = expression_weight
_self_confidence_weight = self_confidence_weight
