#!/usr/bin/env python3
"""Prepare a compact Pelka CRC single-cell reference for RCTD and correlation.

The source h5ad contains 370,115 cells and is intentionally kept as the
authoritative raw reference.  Loading all cells into R/RCTD is unnecessary and
expensive, so this script reproducibly samples each ``ClusterMidway`` cell type,
extracts raw integer counts, and writes two reusable products:

1. a sparse genes-by-cells Matrix Market reference for RCTD;
2. a genes-by-cell-types log1p-CP10K pseudobulk CSV for the expression
   correlation analysis used by ``evaluate_mousebrain_h5ad.py``.

The same sampled cells and gene symbols feed both products, keeping RCTD labels
and correlation columns exactly aligned.

Example
-------
python evaluation/scripts/prepare_crc_scrna_reference.py \
    --max-cells-per-type 500
"""

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.io import mmwrite
from scipy.sparse import csr_matrix


def _collapse_duplicate_gene_symbols(counts, gene_symbols):
    """Sum duplicate feature rows while preserving first-seen symbol order.

    CELLxGENE stores Ensembl IDs as var names and human-readable gene symbols in
    ``feature_name``.  Spatial CRC input uses symbols.  R's ``make.names(...,
    unique=TRUE)`` would silently make duplicates incomparable, so duplicates are
    deliberately collapsed before either RCTD or pseudobulk construction.
    """
    symbols = pd.Series(gene_symbols, dtype="string").fillna("").astype(str)
    valid = symbols.str.len().to_numpy() > 0
    counts = counts[:, valid]
    symbols = symbols[valid].reset_index(drop=True)

    codes, unique_symbols = pd.factorize(symbols, sort=False)
    if len(unique_symbols) == len(symbols):
        return counts.tocsr(), np.asarray(unique_symbols, dtype=str), 0

    # Mapping has one non-zero per original feature.  Right multiplication sums
    # columns sharing the same symbol without ever densifying the cell matrix.
    mapping = csr_matrix(
        (np.ones(len(codes), dtype=np.float32),
         (np.arange(len(codes), dtype=np.int32), codes)),
        shape=(len(codes), len(unique_symbols)),
    )
    collapsed = (counts @ mapping).tocsr()
    return collapsed, np.asarray(unique_symbols, dtype=str), len(symbols) - len(unique_symbols)


def _select_balanced_cells(labels, min_cells, max_cells, seed):
    """Return sorted row indices sampled independently within each cell type."""
    labels = np.asarray(labels, dtype=str)
    rng = np.random.default_rng(seed)
    selected = []
    retained_counts = {}
    for cell_type in sorted(np.unique(labels)):
        candidates = np.flatnonzero(labels == cell_type)
        if len(candidates) < min_cells:
            continue
        if max_cells > 0 and len(candidates) > max_cells:
            candidates = rng.choice(candidates, size=max_cells, replace=False)
        selected.extend(candidates.tolist())
        retained_counts[cell_type] = int(len(candidates))
    return np.asarray(sorted(selected), dtype=np.int64), retained_counts


def main():
    project_root = Path(__file__).resolve().parents[2]
    ref_root = project_root / "evaluation" / "data" / "CRC" / "scrna_reference"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-h5ad",
        default=str(ref_root / "pelka_crc_all_cells.h5ad"),
        help="Pelka CRC CELLxGENE h5ad containing a raw counts slot.",
    )
    parser.add_argument("--label-column", default="ClusterMidway")
    parser.add_argument("--min-cells-per-type", type=int, default=25)
    parser.add_argument(
        "--max-cells-per-type", type=int, default=500,
        help="Balanced per-type cap; 0 keeps all cells (default: 500).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-sum", type=float, default=1e4)
    parser.add_argument(
        "--output-dir",
        default=str(ref_root / "prepared_cluster_midway"),
    )
    args = parser.parse_args()

    input_path = Path(args.input_h5ad)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    print(f"Opening backed reference: {input_path}")
    adata = ad.read_h5ad(input_path, backed="r")
    if adata.raw is None:
        raise ValueError("CRC reference h5ad has no raw slot with integer counts")
    if args.label_column not in adata.obs:
        raise KeyError(f"Missing obs label column: {args.label_column}")

    labels_all = adata.obs[args.label_column].astype(str).to_numpy()
    selected, retained_counts = _select_balanced_cells(
        labels_all,
        min_cells=args.min_cells_per_type,
        max_cells=args.max_cells_per_type,
        seed=args.seed,
    )
    if selected.size == 0:
        raise ValueError("No reference cells passed the cell-type filters")
    print(f"Selected {len(selected):,} cells across {len(retained_counts)} cell types")

    # Backed row slicing reads only the sampled cells rather than materializing
    # the 3.75-GB source object.  raw.X is cells x genes.
    counts = adata.raw[selected, :].X
    counts = csr_matrix(counts, dtype=np.float32)
    raw_var = adata.raw.var
    if "feature_name" in raw_var:
        gene_symbols = raw_var["feature_name"].astype(str).to_numpy()
    elif "feature_name" in adata.var:
        gene_symbols = adata.var["feature_name"].astype(str).to_numpy()
    else:
        gene_symbols = adata.raw.var_names.astype(str).to_numpy()
    counts, gene_symbols, n_collapsed = _collapse_duplicate_gene_symbols(
        counts, gene_symbols
    )
    print(f"Reference matrix: {counts.shape[0]:,} cells x {counts.shape[1]:,} genes; "
          f"collapsed {n_collapsed:,} duplicate symbols")

    selected_labels = np.asarray([label.replace("/", "-") for label in labels_all[selected]])
    selected_names = adata.obs_names[selected].astype(str).to_numpy()

    # RCTD reference: integer-like raw counts in genes x cells orientation.
    counts_path = output_dir / "rctd_counts.mtx"
    genes_path = output_dir / "rctd_genes.csv"
    cells_path = output_dir / "rctd_cells.csv"
    mmwrite(counts_path, counts.T.tocoo())
    pd.DataFrame({"gene_name": gene_symbols}).to_csv(genes_path, index=False)
    pd.DataFrame({
        "cell_name": selected_names,
        "cell_type": selected_labels,
    }).to_csv(cells_path, index=False)
    print(f"Wrote RCTD reference: {counts_path}")

    # Correlation reference: normalize every sampled cell to CP10K, log1p, then
    # average within each ClusterMidway type.  Sparse arithmetic avoids a large
    # cells-by-genes dense intermediate; only the final 20-by-genes matrix is dense.
    normalized = counts.astype(np.float64, copy=True)
    totals = np.asarray(normalized.sum(axis=1)).ravel()
    scales = np.divide(
        args.target_sum, totals,
        out=np.zeros_like(totals, dtype=np.float64), where=totals > 0,
    )
    normalized = normalized.multiply(scales[:, None]).tocsr()
    normalized.data = np.log1p(normalized.data)

    cell_types = sorted(np.unique(selected_labels))
    type_codes = pd.Categorical(selected_labels, categories=cell_types).codes
    membership = csr_matrix(
        (np.ones(len(type_codes)),
         (type_codes, np.arange(len(type_codes), dtype=np.int32))),
        shape=(len(cell_types), len(type_codes)),
    )
    type_sizes = np.bincount(type_codes, minlength=len(cell_types)).astype(np.float64)
    pseudobulk = membership @ normalized
    pseudobulk = pseudobulk.multiply((1.0 / type_sizes)[:, None]).toarray()
    pseudobulk_path = output_dir / "crc_cluster_midway_pseudobulk.csv"
    pd.DataFrame(pseudobulk.T, index=gene_symbols, columns=cell_types).to_csv(
        pseudobulk_path
    )
    print(f"Wrote correlation pseudobulk: {pseudobulk_path}")

    summary = {
        "input_h5ad": str(input_path),
        "label_column": args.label_column,
        "seed": args.seed,
        "min_cells_per_type": args.min_cells_per_type,
        "max_cells_per_type": args.max_cells_per_type,
        "n_cells": int(counts.shape[0]),
        "n_genes": int(counts.shape[1]),
        "n_duplicate_symbols_collapsed": int(n_collapsed),
        "cells_per_type": retained_counts,
        "target_sum": args.target_sum,
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"Wrote summary: {output_dir / 'summary.json'}")

    adata.file.close()


if __name__ == "__main__":
    main()
