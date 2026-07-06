#!/usr/bin/env python3
"""Compare spatial FindMarker-like gene lists with snRNA reference markers.

For each cell group we define markers as the genes with the highest log2 fold
change relative to the mean of all other groups (pseudobulk-level DE).  We then
compute the overlap between the spatial-derived marker set (RAW or SPARKLE) and
the snRNA reference-derived marker set.

Usage:
    python evaluation/scripts/findmarker_overlap.py \
        --tag mousebrain_x12500-17500_y2000-5000 \
        --min-cells 20 \
        --n-markers 50
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc


def normalize_adata(adata):
    adata = adata.copy()
    if hasattr(adata.X, "toarray"):
        adata.X = adata.X.toarray()
    adata.X = np.maximum(np.asarray(adata.X, dtype=np.float64), 0.0)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    return adata


def compute_pseudobulk(adata, group_key="annotation", min_cells=20):
    groups = adata.obs[group_key].astype(str)
    valid = groups != "Unknown"
    adata = adata[valid].copy()
    groups = adata.obs[group_key]
    group_names = sorted(groups.unique())

    expr = adata.X
    if hasattr(expr, "toarray"):
        expr = expr.toarray()
    expr = np.asarray(expr)

    pb = {}
    kept_groups = []
    for g in group_names:
        mask = groups == g
        if mask.sum() < min_cells:
            continue
        pb[g] = expr[mask].mean(axis=0)
        kept_groups.append(g)
    pb_df = pd.DataFrame(pb, index=adata.var_names, columns=kept_groups)
    return pb_df


def find_markers(pb_df, n_markers=50, eps=1e-6):
    """Return dict group -> set of top logFC marker genes."""
    group_means = pb_df
    overall_mean_other = pd.DataFrame(
        {g: group_means.drop(columns=g).mean(axis=1) for g in group_means.columns},
        index=group_means.index,
    )
    logfc = np.log2((group_means + eps) / (overall_mean_other + eps))

    markers = {}
    for g in group_means.columns:
        top = logfc[g].sort_values(ascending=False).head(n_markers)
        markers[g] = set(top.index)
    return markers, logfc


def overlap_metrics(set_a, set_b):
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return {
        "n_a": len(set_a),
        "n_b": len(set_b),
        "intersection": inter,
        "union": union,
        "jaccard": inter / union if union > 0 else 0.0,
        "precision": inter / len(set_a) if len(set_a) > 0 else 0.0,
        "recall": inter / len(set_b) if len(set_b) > 0 else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, required=True)
    parser.add_argument("--input-dir", type=str, default="evaluation/reports/h5ad")
    parser.add_argument("--output-dir", type=str, default="evaluation/reports/mousebrain_eval")
    parser.add_argument("--snrna-ref", type=str,
                        default="evaluation/data/mousebrain/snrna_cell_group_pseudobulk.csv")
    parser.add_argument("--min-cells", type=int, default=20)
    parser.add_argument("--n-markers", type=int, default=50)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir) / "gene_level_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = normalize_adata(sc.read_h5ad(input_dir / f"{args.tag}_raw.h5ad"))
    sp = normalize_adata(sc.read_h5ad(input_dir / f"{args.tag}_SPARKLE.h5ad"))
    ref = pd.read_csv(args.snrna_ref, index_col=0)

    pb_raw = compute_pseudobulk(raw, min_cells=args.min_cells)
    pb_sp = compute_pseudobulk(sp, min_cells=args.min_cells)

    print(f"Groups retained (min_cells={args.min_cells}):")
    print(" ", pb_raw.columns.tolist())

    markers_raw, _ = find_markers(pb_raw, n_markers=args.n_markers)
    markers_sp, _ = find_markers(pb_sp, n_markers=args.n_markers)
    markers_ref, _ = find_markers(ref[pb_raw.columns], n_markers=args.n_markers)

    rows = []
    for g in pb_raw.columns:
        raw_metrics = overlap_metrics(markers_raw[g], markers_ref[g])
        sp_metrics = overlap_metrics(markers_sp[g], markers_ref[g])
        row = {"group": g}
        row.update({f"raw_{k}": v for k, v in raw_metrics.items()})
        row.update({f"sparkle_{k}": v for k, v in sp_metrics.items()})
        row["sparkle_minus_raw_jaccard"] = sp_metrics["jaccard"] - raw_metrics["jaccard"]
        row["sparkle_minus_raw_precision"] = sp_metrics["precision"] - raw_metrics["precision"]
        row["sparkle_minus_raw_recall"] = sp_metrics["recall"] - raw_metrics["recall"]
        rows.append(row)

    df = pd.DataFrame(rows)
    out_path = out_dir / f"{args.tag}_findmarker_overlap_min{args.min_cells}_top{args.n_markers}.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved per-group overlap: {out_path}")

    # Summary
    summary = pd.DataFrame({
        "metric": ["jaccard", "precision", "recall"],
        "raw_mean": [df["raw_jaccard"].mean(), df["raw_precision"].mean(), df["raw_recall"].mean()],
        "sparkle_mean": [df["sparkle_jaccard"].mean(), df["sparkle_precision"].mean(), df["sparkle_recall"].mean()],
        "sparkle_minus_raw": [
            df["sparkle_minus_raw_jaccard"].mean(),
            df["sparkle_minus_raw_precision"].mean(),
            df["sparkle_minus_raw_recall"].mean(),
        ],
    })
    print("\n=== Summary (mean across groups) ===")
    print(summary.to_markdown(index=False))

    summary_path = out_dir / f"{args.tag}_findmarker_overlap_summary_min{args.min_cells}_top{args.n_markers}.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved summary: {summary_path}")

    # Per-group table
    print("\n=== Per-group Jaccard / Precision / Recall ===")
    display = df[["group", "raw_jaccard", "sparkle_jaccard", "sparkle_minus_raw_jaccard",
                  "raw_precision", "sparkle_precision", "raw_recall", "sparkle_recall"]]
    print(display.to_markdown(index=False))


if __name__ == "__main__":
    main()
