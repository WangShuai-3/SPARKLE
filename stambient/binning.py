"""Empty-space binning: aggregate out-of-mask DNBs into square spatial bins.

Empty DNBs (label < 0) carry no cell identity but still capture ambient RNA.
They are binned into a coarse square grid so the number of "ambient probes"
stays tractable and each probe has a well-defined area (DNB count).
"""

import numpy as np
from typing import Tuple


def bin_empty_dnbs(
    dnb_coords: np.ndarray,
    dnb_labels: np.ndarray,
    bin_size: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Assign empty DNBs to square spatial bins in one pass.

    ``dnb_coords`` and ``bin_size`` must use the same spatial unit.  The
    public API requires micrometres, but keeping this helper unit-agnostic is
    useful for adapters that scale the resulting centroids afterwards.

    Returns bin centroids/areas plus two aligned arrays describing the sparse
    DNB-to-bin mapping.  Returning the mapping directly avoids the former
    per-bin boolean scan, whose complexity was O(n_empty * n_bins).
    """
    if not np.isfinite(bin_size) or bin_size <= 0:
        raise ValueError("bin_size must be a positive finite spatial length")

    empty_mask = dnb_labels < 0
    n_empty = empty_mask.sum()
    if n_empty == 0:
        empty = np.zeros(0, dtype=np.int64)
        return np.zeros((0, 2)), empty, empty, empty

    empty_orig_idx = np.flatnonzero(empty_mask)
    empty_coords = dnb_coords[empty_orig_idx]

    x_min, y_min = empty_coords.min(axis=0)
    x_max, y_max = empty_coords.max(axis=0)

    n_bins_x = max(1, int(np.ceil((x_max - x_min) / bin_size)))
    n_bins_y = max(1, int(np.ceil((y_max - y_min) / bin_size)))

    bin_x = np.floor((empty_coords[:, 0] - x_min) / bin_size).astype(np.int64)
    bin_y = np.floor((empty_coords[:, 1] - y_min) / bin_size).astype(np.int64)
    bin_x = np.clip(bin_x, 0, n_bins_x - 1)
    bin_y = np.clip(bin_y, 0, n_bins_y - 1)
    bin_ids = bin_x * np.int64(n_bins_y) + bin_y

    _, inverse = np.unique(bin_ids, return_inverse=True)
    inverse = inverse.astype(np.int64, copy=False)
    n_bins = int(inverse.max()) + 1

    bin_areas = np.bincount(inverse, minlength=n_bins).astype(
        np.int64, copy=False
    )
    bin_coords = np.column_stack(
        (
            np.bincount(inverse, weights=empty_coords[:, 0], minlength=n_bins),
            np.bincount(inverse, weights=empty_coords[:, 1], minlength=n_bins),
        )
    )
    bin_coords /= bin_areas[:, None]

    return bin_coords, bin_areas, empty_orig_idx, inverse


# Backward-compatible private alias used by the cell-pipeline tests and the
# SpotClean I/O adapter.
_bin_empty_dnbs = bin_empty_dnbs
