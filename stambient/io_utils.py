"""I/O utilities for SPARKLE."""

import numpy as np
from scipy.sparse import csr_matrix, issparse
from typing import Optional, Tuple


def check_inputs(
    expression: np.ndarray,
    coordinates: np.ndarray,
    labels: np.ndarray,
) -> None:
    """Validate input dimensions and types.

    Args:
        expression: [genes × spots] expression matrix.
        coordinates: [spots × 2] coordinates.
        labels: [spots] cell labels.

    Raises:
        ValueError: If inputs are inconsistent.
    """
    n_spots = coordinates.shape[0]

    if expression.shape[1] != n_spots:
        raise ValueError(
            f"Expression has {expression.shape[1]} spots but coordinates have {n_spots}"
        )
    if len(labels) != n_spots:
        raise ValueError(
            f"Labels has {len(labels)} entries but coordinates have {n_spots}"
        )
    if coordinates.shape[1] != 2:
        raise ValueError(f"Coordinates must be [N × 2], got {coordinates.shape}")


def sparse_to_dense_if_needed(mat) -> np.ndarray:
    """Convert a matrix to dense numpy if sparse."""
    if issparse(mat):
        return mat.toarray()
    return np.asarray(mat)


def save_results(
    corrected_expr: csr_matrix,
    diagnostics: dict,
    output_prefix: str,
):
    """Save corrected expression and diagnostics to files.

    Args:
        corrected_expr: [genes × cells] corrected per-cell expression.
        diagnostics: Diagnostic dictionary.
        output_prefix: File path prefix.
    """
    import json

    # Save sparse matrix
    from scipy.sparse import save_npz
    save_npz(f"{output_prefix}_corrected.npz", corrected_expr)

    # Save diagnostics as JSON
    with open(f"{output_prefix}_diagnostics.json", "w") as f:
        json.dump(diagnostics, f, indent=2, default=str)
