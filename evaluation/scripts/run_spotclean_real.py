#!/usr/bin/env python3
"""Generate only cell-only SpotClean h5ad files for final real-data windows.

The script deliberately reuses the existing RAW h5ad as its output template and
does not run or overwrite any other correction method or metrics JSON.
"""

import argparse
import gzip
import json
import os
import shutil
import sys
import time
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
from scipy.sparse import csr_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.baselines.spotclean import run_spotclean_sparse_cells


DATASETS = {
    "axolotl": {
        "x_range": (10500, 12500),
        "y_range": (6000, 11100),
        "bandwidths": (10, 20, 30, 50, 70, 100, 150, 200),
    },
    "mousebrain": {
        "x_range": (12500, 20000),
        "y_range": (2000, 10000),
        "bandwidths": (10, 20, 30, 50, 70, 100, 150, 200, 300),
    },
    "ovarian": {
        "x_range": (1000, 1800),
        "y_range": (300, 1100),
        "bandwidths": (10, 20, 30, 50, 70, 100, 150, 200, 300),
    },
}


def _read_template(raw_path):
    """Read the exact prior RAW matrix and its ordered identifiers."""
    raw = ad.read_h5ad(raw_path, backed="r")
    try:
        cell_ids = raw.obs["cell_id"].to_numpy().copy()
        gene_names = raw.var_names.astype(str).to_numpy().copy()
    finally:
        raw.file.close()
    with h5py.File(raw_path, "r") as handle:
        if not isinstance(handle["X"], h5py.Dataset):
            raise TypeError("The existing RAW template must use a dense X dataset.")
        dense = handle["X"][:]
    expression = csr_matrix(dense.T)
    del dense
    return expression, cell_ids, gene_names


def _centroids_from_text(path, delimiter, cell_ids, x_range, y_range, columns):
    """Scan a coordinate table and count each labelled DNB coordinate once."""
    x0, x1 = x_range
    y0, y1 = y_range
    width = x1 - x0 + 1
    height = y1 - y0 + 1
    seen = np.zeros(width * height, dtype=bool)
    id_to_column = {int(cell_id): i for i, cell_id in enumerate(cell_ids)}
    counts = np.zeros(len(cell_ids), dtype=np.int64)
    x_sums = np.zeros(len(cell_ids), dtype=np.float64)
    y_sums = np.zeros(len(cell_ids), dtype=np.float64)
    with gzip.open(path, "rt") as handle:
        header = handle.readline().rstrip("\n").split(delimiter)
        x_idx = header.index(columns["x"])
        y_idx = header.index(columns["y"])
        cell_idx = header.index(columns["cell"])
        for line in handle:
            fields = line.rstrip("\n").split(delimiter)
            cell_id = int(fields[cell_idx])
            output_column = id_to_column.get(cell_id)
            if output_column is None:
                continue
            x = int(fields[x_idx])
            y = int(fields[y_idx])
            if not (x0 <= x <= x1 and y0 <= y <= y1):
                continue
            flat = (y - y0) * width + (x - x0)
            if seen[flat]:
                continue
            seen[flat] = True
            counts[output_column] += 1
            x_sums[output_column] += x
            y_sums[output_column] += y
    if np.any(counts == 0):
        missing = np.asarray(cell_ids)[counts == 0]
        raise ValueError(f"No labelled coordinates found for cells: {missing[:5]}")
    return np.column_stack((x_sums / counts, y_sums / counts))


def _load_centroids(name, cell_ids):
    config = DATASETS[name]
    data_root = PROJECT_ROOT / "evaluation" / "data"
    if name == "axolotl":
        return _centroids_from_text(
            data_root / "axolotl" / "Adult_scgem.csv.gz",
            ",",
            cell_ids,
            config["x_range"],
            config["y_range"],
            {"x": "x", "y": "y", "cell": "cell"},
        )
    if name == "mousebrain":
        return _centroids_from_text(
            data_root / "mousebrain" / "total_gene_T304_mouse_f001_2D_mouse1-20230119.txt.gz",
            "\t",
            cell_ids,
            config["x_range"],
            config["y_range"],
            {"x": "x", "y": "y", "cell": "cell_label"},
        )

    path = data_root / "ovarian" / "Visium_HD_Human_Ovarian_Cancer_FF_feature_slice.h5"
    with h5py.File(path, "r") as handle:
        segmentation = handle["segmentations/cell_segmentation_mask"]
        rows = segmentation["row"][:].astype(np.int64)
        cols = segmentation["col"][:].astype(np.int64)
        labels = segmentation["data"][:].astype(np.int64)
    x0, x1 = config["x_range"]
    y0, y1 = config["y_range"]
    keep = (cols >= x0) & (cols <= x1) & (rows >= y0) & (rows <= y1)
    rows, cols, labels = rows[keep], cols[keep], labels[keep]
    id_to_column = {int(cell_id): i for i, cell_id in enumerate(cell_ids)}
    output_columns = np.fromiter(
        (id_to_column.get(int(label), -1) for label in labels),
        dtype=np.int64,
        count=len(labels),
    )
    valid = output_columns >= 0
    output_columns = output_columns[valid]
    rows, cols = rows[valid], cols[valid]
    counts = np.bincount(output_columns, minlength=len(cell_ids)).astype(np.float64)
    if np.any(counts == 0):
        missing = np.asarray(cell_ids)[counts == 0]
        raise ValueError(f"No segmented pixels found for cells: {missing[:5]}")
    x_sums = np.bincount(
        output_columns, weights=cols * 2.0 + 1.0, minlength=len(cell_ids)
    )
    y_sums = np.bincount(
        output_columns, weights=rows * 2.0 + 1.0, minlength=len(cell_ids)
    )
    return np.column_stack((x_sums / counts, y_sums / counts))


def _replace_string_dataset(group, key, value):
    if key in group:
        del group[key]
    dataset = group.create_dataset(key, data=value, dtype=h5py.string_dtype("utf-8"))
    dataset.attrs["encoding-type"] = "string"
    dataset.attrs["encoding-version"] = "0.2.0"


def run_dataset(name, args):
    config = DATASETS[name]
    x0, x1 = config["x_range"]
    y0, y1 = config["y_range"]
    tag = f"{name}_x{x0}-{x1}_y{y0}-{y1}"
    reports = PROJECT_ROOT / "evaluation" / "reports" / "h5ad"
    raw_path = reports / f"{tag}_raw.h5ad"
    output_path = reports / f"{tag}_SpotClean.h5ad"
    temporary_path = reports / f".{tag}_SpotClean.tmp.h5ad"
    if not raw_path.exists():
        raise FileNotFoundError(f"Existing RAW h5ad template not found: {raw_path}")
    if output_path.exists() and not args.overwrite:
        print(f"[{name}] exists, skipping: {output_path}")
        return

    print(f"\n{'=' * 72}\n{name.upper()} CELL-ONLY SPOTCLEAN\n{'=' * 72}")
    print(f"window: x={config['x_range']}, y={config['y_range']}")
    started = time.time()
    expression, cell_ids, gene_names = _read_template(raw_path)
    centroids = _load_centroids(name, cell_ids)
    print(
        f"reused RAW: {expression.shape[0]:,} genes x "
        f"{expression.shape[1]:,} cells, {expression.nnz:,} nonzeros"
    )
    print(f"centroids: {len(centroids):,}; gene/cell order inherited exactly")

    if temporary_path.exists():
        temporary_path.unlink()
    shutil.copy2(raw_path, temporary_path)
    try:
        with h5py.File(temporary_path, "r+") as output:
            if not isinstance(output["X"], h5py.Dataset):
                raise TypeError("Streaming writer currently requires a dense h5ad X dataset.")

            def write_batch(start, end, corrected):
                output["X"][:, start:end] = corrected.T.astype(np.float32, copy=False)

            _, diagnostics = run_spotclean_sparse_cells(
                expression,
                centroids,
                candidate_bandwidths=config["bandwidths"],
                bleed_rate=args.bleed_rate,
                distal_rate=args.distal_rate,
                n_neighbors=args.n_neighbors,
                n_selection_genes=args.n_selection_genes,
                gene_batch_size=args.gene_batch_size,
                maxit=args.maxit,
                tol=args.tol,
                n_jobs=args.n_jobs,
                batch_writer=write_batch,
                verbose=True,
            )
            diagnostics["runtime_seconds"] = float(time.time() - started)
            diagnostics["dataset"] = tag
            diagnostics["x_range"] = list(config["x_range"])
            diagnostics["y_range"] = list(config["y_range"])
            _replace_string_dataset(output["uns"], "method", "SpotClean")
            _replace_string_dataset(
                output["uns"],
                "spotclean_diagnostics_json",
                json.dumps(diagnostics, sort_keys=True),
            )
            output.flush()
        os.replace(temporary_path, output_path)
    except Exception:
        if temporary_path.exists():
            temporary_path.unlink()
        raise

    check = ad.read_h5ad(output_path, backed="r")
    try:
        if check.shape != (expression.shape[1], expression.shape[0]):
            raise RuntimeError("Written SpotClean h5ad failed its shape check.")
        if check.uns.get("method") != "SpotClean":
            raise RuntimeError("Written SpotClean h5ad failed its method metadata check.")
    finally:
        check.file.close()
    print(
        f"saved: {output_path}\n"
        f"bandwidth={diagnostics['contamination_bandwidth']:g}, "
        f"runtime={diagnostics['runtime_seconds']:.1f}s, "
        f"conservation_error={diagnostics['max_gene_count_conservation_error']:.3e}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=tuple(DATASETS),
        default=list(DATASETS),
    )
    parser.add_argument("--bleed-rate", type=float, default=0.10)
    parser.add_argument("--distal-rate", type=float, default=0.10)
    parser.add_argument("--n-neighbors", type=int, default=32)
    parser.add_argument("--n-selection-genes", type=int, default=50)
    parser.add_argument("--gene-batch-size", type=int, default=64)
    parser.add_argument("--maxit", type=int, default=3)
    parser.add_argument("--tol", type=float, default=1.0)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    for name in args.datasets:
        run_dataset(name, args)


if __name__ == "__main__":
    main()
