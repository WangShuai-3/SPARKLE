#!/usr/bin/env python3
"""Build a cell-type pseudobulk reference from the Ovarian Cancer scFFPE scRNA-seq data.

Mirrors ``prepare_mousebrain_snrna_reference.R`` but operates directly on the 10x
Genomics feature-barcode HDF5 matrix (no Seurat / RDS required).

For each cell type in the FLEX annotation, we compute the mean of the
log1p-CPM-normalized expression across its cells, giving a genes x cell_type
matrix identical in spirit to Seurat's ``AverageExpression(slot = "data")``.

Usage:
    python evaluation/scripts/prepare_ovarian_scrna_reference.py \
        [--scrna-h5   evaluation/data/ovarian/17k_Ovarian_Cancer_scFFPE_count_filtered_feature_bc_matrix.h5] \
        [--annotation evaluation/data/ovarian/FLEX_Ovarian_Barcode_Cluster_Annotation.csv] \
        [--out        evaluation/data/ovarian/scrna_celltype_pseudobulk.csv] \
        [--min-cells  20]
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix


def load_10x_h5(h5_path):
    """Load a 10x feature-barcode HDF5 matrix as (genes x cells) CSC sparse."""
    with h5py.File(h5_path, "r") as f:
        m = f["matrix"]
        data = m["data"][:]
        indices = m["indices"][:]
        indptr = m["indptr"][:]
        shape = tuple(int(s) for s in m["shape"][:])  # [genes, cells]
        barcodes = np.array([b.decode() for b in m["barcodes"][:]])
        gene_names = np.array([g.decode() for g in m["features/name"][:]])
    mat = csc_matrix((data, indices, indptr), shape=shape)
    return mat, gene_names, barcodes


def main():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scrna-h5", type=str,
        default=str(root / "data" / "ovarian" /
                    "17k_Ovarian_Cancer_scFFPE_count_filtered_feature_bc_matrix.h5"))
    parser.add_argument(
        "--annotation", type=str,
        default=str(root / "data" / "ovarian" /
                    "FLEX_Ovarian_Barcode_Cluster_Annotation.csv"))
    parser.add_argument(
        "--out", type=str,
        default=str(root / "data" / "ovarian" / "scrna_celltype_pseudobulk.csv"))
    parser.add_argument("--barcode-col", type=str, default="Barcode")
    parser.add_argument("--annotation-col", type=str, default="Cell Annotation")
    parser.add_argument("--target-sum", type=float, default=1e4)
    parser.add_argument("--min-cells", type=int, default=20,
                        help="Drop cell types with fewer than this many cells.")
    args = parser.parse_args()

    print(f"Loading scRNA matrix: {args.scrna_h5}")
    mat, gene_names, barcodes = load_10x_h5(args.scrna_h5)  # genes x cells
    print(f"  {mat.shape[0]} genes x {mat.shape[1]} cells")

    print(f"Loading annotation: {args.annotation}")
    ann = pd.read_csv(args.annotation)
    ann = ann[[args.barcode_col, args.annotation_col]].dropna()
    bc_to_type = dict(zip(ann[args.barcode_col], ann[args.annotation_col]))

    # Keep only barcodes with an annotation, preserving matrix column order.
    keep_mask = np.array([bc in bc_to_type for bc in barcodes])
    kept_bc = barcodes[keep_mask]
    cell_types = np.array([bc_to_type[bc] for bc in kept_bc])
    print(f"  {keep_mask.sum()} / {len(barcodes)} cells matched to an annotation")

    # cells x genes for per-cell normalization
    X = mat[:, keep_mask].T.tocsr().astype(np.float64)  # cells x genes

    # log1p-CPM (normalize_total to target_sum, then log1p)
    counts_per_cell = np.asarray(X.sum(axis=1)).ravel()
    counts_per_cell[counts_per_cell == 0] = 1.0
    scale = args.target_sum / counts_per_cell
    X = X.multiply(scale[:, None]).tocsr()
    X.data = np.log1p(X.data)

    # Mean per cell type -> genes x cell_type
    # RCTD prohibits '/' in cell-type names (it replaces them with '-'); sanitize
    # here identically so downstream string matching against RCTD output holds.
    cell_types = np.array([t.replace("/", "-") for t in cell_types])
    unique_types = sorted(set(cell_types))
    pb = {}
    for t in unique_types:
        idx = np.where(cell_types == t)[0]
        if len(idx) < args.min_cells:
            print(f"  Skipping '{t}' ({len(idx)} cells < {args.min_cells})")
            continue
        mean_expr = np.asarray(X[idx].mean(axis=0)).ravel()
        pb[t] = mean_expr

    ref_df = pd.DataFrame(pb, index=gene_names)
    # Collapse duplicate gene symbols (mean) so the index is unique; downstream
    # correlation code indexes by gene name and requires a unique index.
    n_dup = int(ref_df.index.duplicated().sum())
    if n_dup > 0:
        print(f"  Collapsing {n_dup} duplicate gene symbols by mean")
        ref_df = ref_df.groupby(level=0).mean()
    # RCTD/eval-friendly: keep original type names (eval matches by exact string)
    print(f"Reference shape: {ref_df.shape[0]} genes x {ref_df.shape[1]} cell types")
    print(f"Cell types: {list(ref_df.columns)}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ref_df.to_csv(out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
