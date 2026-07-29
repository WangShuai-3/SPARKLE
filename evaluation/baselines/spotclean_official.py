"""I/O adapter for running the official SpotClean R package.

This module intentionally contains no reimplementation of the SpotClean model.
It only turns segmented DNB data into a gene-by-spot matrix, serializes that
matrix for R, invokes the standalone R wrapper, and reads its output.
"""

from __future__ import annotations

import csv
import subprocess
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
from scipy.sparse import csc_matrix, csr_matrix, hstack, issparse

from stambient.cell_pipeline import _bin_empty_dnbs


def _contiguous_labels(dnb_labels: np.ndarray, cell_ids: np.ndarray) -> np.ndarray:
    """Map non-negative DNB labels to the order of ``cell_ids``."""
    labels = np.asarray(dnb_labels)
    cell_ids = np.asarray(cell_ids)
    if labels.ndim != 1:
        raise ValueError("dnb_labels must be one-dimensional")
    if len(cell_ids) == 0:
        raise ValueError("SpotClean requires at least one tissue cell")

    covered = labels >= 0
    present = np.unique(labels[covered])
    contiguous = np.arange(len(cell_ids), dtype=present.dtype)
    if np.array_equal(present, contiguous) and labels[covered].max(initial=-1) < len(cell_ids):
        return labels.astype(np.int64, copy=False)

    lookup = {int(cell_id): index for index, cell_id in enumerate(cell_ids)}
    remapped = np.full(labels.shape, -1, dtype=np.int64)
    remapped[covered] = np.fromiter(
        (lookup.get(int(value), -1) for value in labels[covered]),
        dtype=np.int64,
        count=int(covered.sum()),
    )
    if np.any(remapped[covered] < 0):
        missing = np.unique(labels[covered][remapped[covered] < 0])
        raise ValueError(f"DNB labels are absent from cell_ids: {missing[:5]}")
    return remapped


def prepare_spotclean_spots(
    dnb_expr,
    dnb_coords: np.ndarray,
    dnb_labels: np.ndarray,
    cell_ids: np.ndarray,
    *,
    empty_bin_size: float,
    coordinate_scale: float = 1.0,
):
    """Aggregate cells and empty DNBs into official SpotClean input spots.

    Cells are tissue spots. Empty DNBs are aggregated on a fixed grid and are
    background spots. Both spot types use their arithmetic coordinate centroids.
    Empty-bin counts are literal sums; no exposure or area scaling is applied.
    """
    if not np.isfinite(empty_bin_size) or empty_bin_size <= 0:
        raise ValueError("empty_bin_size must be positive and finite")
    if not np.isfinite(coordinate_scale) or coordinate_scale <= 0:
        raise ValueError("coordinate_scale must be positive and finite")
    if not issparse(dnb_expr):
        dnb_expr = csr_matrix(dnb_expr)
    else:
        dnb_expr = dnb_expr.tocsr()
    coords = np.asarray(dnb_coords, dtype=np.float64)
    labels = _contiguous_labels(np.asarray(dnb_labels), np.asarray(cell_ids))
    if coords.shape != (dnb_expr.shape[1], 2):
        raise ValueError("dnb_coords must have shape (n_dnb, 2)")

    covered = labels >= 0
    empty = ~covered
    if not np.any(empty):
        raise ValueError("Official SpotClean background input requires empty DNBs")

    n_cells = len(cell_ids)
    covered_indices = np.flatnonzero(covered)
    cell_assignment = csr_matrix(
        (
            np.ones(len(covered_indices), dtype=np.float64),
            (covered_indices, labels[covered]),
        ),
        shape=(dnb_expr.shape[1], n_cells),
    )
    cell_expr = (dnb_expr @ cell_assignment).tocsr()
    cell_counts = np.bincount(labels[covered], minlength=n_cells).astype(np.float64)
    if np.any(cell_counts == 0):
        raise ValueError("At least one requested cell contains no DNBs")
    cell_x = np.bincount(
        labels[covered], weights=coords[covered, 0], minlength=n_cells
    )
    cell_y = np.bincount(
        labels[covered], weights=coords[covered, 1], minlength=n_cells
    )
    cell_centroids = np.column_stack((cell_x / cell_counts, cell_y / cell_counts))

    empty_coords, empty_areas, empty_dnb_indices, empty_bin_indices = _bin_empty_dnbs(
        coords, labels, empty_bin_size
    )
    if len(empty_coords) == 0:
        raise ValueError("Empty-DNB aggregation produced no background spots")
    empty_assignment = csr_matrix(
        (
            np.ones(len(empty_dnb_indices), dtype=np.float64),
            (empty_dnb_indices, empty_bin_indices),
        ),
        shape=(dnb_expr.shape[1], len(empty_coords)),
    )
    empty_expr = (dnb_expr @ empty_assignment).tocsr()

    spot_expr = hstack((cell_expr, empty_expr), format="csc")
    spot_coords = np.vstack((cell_centroids, empty_coords)) * coordinate_scale
    tissue = np.concatenate(
        (np.ones(n_cells, dtype=np.int8), np.zeros(len(empty_coords), dtype=np.int8))
    )
    barcodes = np.asarray(
        [f"cell_{value}" for value in cell_ids]
        + [f"background_{index}" for index in range(len(empty_coords))],
        dtype=object,
    )
    diagnostics = {
        "n_tissue_spots": int(n_cells),
        "n_background_spots": int(len(empty_coords)),
        "n_all_spots": int(n_cells + len(empty_coords)),
        "n_empty_dnbs": int(empty.sum()),
        "empty_bin_size_input_units": float(empty_bin_size),
        "coordinate_scale": float(coordinate_scale),
        "empty_exposure_normalized": False,
        "empty_bin_dnb_count_min": int(np.min(empty_areas)),
        "empty_bin_dnb_count_median": float(np.median(empty_areas)),
        "empty_bin_dnb_count_max": int(np.max(empty_areas)),
    }
    return spot_expr, spot_coords, tissue, barcodes, diagnostics


def write_official_input(
    input_dir: Path,
    spot_expr,
    spot_coords: np.ndarray,
    tissue: np.ndarray,
    barcodes: np.ndarray,
    gene_names: Iterable[str],
    gene_keep: Iterable[str] | None,
) -> None:
    """Write a CSC matrix and metadata consumable by the R wrapper."""
    input_dir = Path(input_dir)
    input_dir.mkdir(parents=True, exist_ok=True)
    matrix = csc_matrix(spot_expr, dtype=np.float64)
    genes = np.asarray([str(value) for value in gene_names], dtype=object)
    if matrix.shape != (len(genes), len(barcodes)):
        raise ValueError("Expression matrix, genes, and barcodes have inconsistent shapes")

    with h5py.File(input_dir / "counts_csc.h5", "w") as handle:
        handle.create_dataset("data", data=matrix.data, compression="gzip")
        handle.create_dataset("indices", data=matrix.indices.astype(np.int32), compression="gzip")
        handle.create_dataset("indptr", data=matrix.indptr.astype(np.int64), compression="gzip")
        handle.create_dataset("shape", data=np.asarray(matrix.shape, dtype=np.int64))

    with open(input_dir / "genes.tsv", "w", encoding="utf-8", newline="") as handle:
        handle.write("\n".join(genes))
        handle.write("\n")
    with open(input_dir / "gene_keep.tsv", "w", encoding="utf-8", newline="") as handle:
        values = [] if gene_keep is None else [str(value) for value in gene_keep]
        handle.write("\n".join(values))
        if values:
            handle.write("\n")
    with open(input_dir / "slide.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["barcode", "tissue", "row", "col", "imagerow", "imagecol"])
        for barcode, is_tissue, (x_coord, y_coord) in zip(
            barcodes, tissue, spot_coords, strict=True
        ):
            # row/col deliberately share the image-coordinate scale. Official
            # SpotClean then estimates an adjacent-spot distance of one, making
            # candidate_radius use these physical coordinate units.
            writer.writerow(
                [barcode, int(is_tissue), y_coord, x_coord, y_coord, x_coord]
            )


def run_official_r(
    input_dir: Path,
    output_dir: Path,
    *,
    r_script: Path,
    candidate_radius: Iterable[float],
    maxit: int = 30,
    tol: float = 1.0,
    rscript: str = "Rscript",
) -> subprocess.CompletedProcess:
    """Invoke the standalone R wrapper and fail on any official-package error."""
    radii = [float(value) for value in candidate_radius]
    if not radii or any(value <= 0 for value in radii):
        raise ValueError("candidate_radius must contain positive values")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        rscript,
        str(r_script),
        str(Path(input_dir)),
        str(output_dir),
        str(int(maxit)),
        str(float(tol)),
        ",".join(f"{value:g}" for value in radii),
    ]
    return subprocess.run(command, check=True, text=True)


def read_key_value_tsv(path: Path) -> dict[str, object]:
    """Read scalar diagnostics emitted by the R wrapper."""
    result: dict[str, object] = {}
    with open(path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            value: object = row["value"]
            try:
                value = float(value)
            except (TypeError, ValueError):
                pass
            result[row["key"]] = value
    return result
