#!/usr/bin/env python3
"""Compare spatial marker recovery with robust snRNA reference markers.

Reference markers are defined per cell group as genes that are both:
  - highly expressed in the target group (top N by expression)
  - specific to the target group (target_expr / mean_other > specificity_threshold)

We then ask: for each group, how many of these reference markers appear in the
spatial top M markers (by logFC or by specificity)?

This avoids the problem that raw logFC-based ranks are dominated by low-count
/novel genes (Gm..., Olfr..., Vmn2r...) that are not shared between spatial and
snRNA reference.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc


def normalize_adata(adata):
    adata = adata.copy()
    if hasattr(adata.X, "toarray"):
        adata.X = adata.toarray()
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
    return pd.DataFrame(pb, index=adata.var_names, columns=kept_groups)


def compute_logfc(pb_df, eps=1e-6):
    logfc = pd.DataFrame(index=pb_df.index)
    for g in pb_df.columns:
        other_mean = pb_df.drop(columns=g).mean(axis=1)
        logfc[g] = np.log2((pb_df[g] + eps) / (other_mean + eps))
    return logfc


def compute_specificity(pb_df):
    spec = pd.DataFrame(index=pb_df.index)
    for g in pb_df.columns:
        other_mean = pb_df.drop(columns=g).mean(axis=1)
        spec[g] = pb_df[g] / (other_mean + 1e-6)
    return spec


def find_reference_markers(ref_df, n_markers=50, specificity_threshold=2.0, min_expr=1.0):
    """Return dict group -> set of robust reference markers."""
    markers = {}
    for g in ref_df.columns:
        target_expr = ref_df[g]
        other_mean = ref_df.drop(columns=g).mean(axis=1)
        specificity = target_expr / (other_mean + 1e-6)
        candidates = ref_df[
            (target_expr >= min_expr) & (specificity >= specificity_threshold)
        ].copy()
        candidates["expr"] = candidates[g]
        candidates["spec"] = specificity[candidates.index]
        # rank by expr * specificity
        candidates["score"] = candidates["expr"] * np.log1p(candidates["spec"])
        top = candidates.sort_values("score", ascending=False).head(n_markers)
        markers[g] = set(top.index)
    return markers


def find_spatial_markers(pb_df, n_markers=50, by="logfc"):
    """Return dict group -> set of spatial markers."""
    markers = {}
    if by == "logfc":
        scores = compute_logfc(pb_df)
    else:
        scores = compute_specificity(pb_df)
    for g in pb_df.columns:
        markers[g] = set(scores[g].sort_values(ascending=False).head(n_markers).index)
    return markers


def overlap_metrics(set_a, set_b):
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return {
        "jaccard": inter / union if union > 0 else 0.0,
        "precision": inter / len(set_a) if len(set_a) > 0 else 0.0,
        "recall": inter / len(set_b) if len(set_b) > 0 else 0.0,
        "n_ref_markers": len(set_b),
        "n_spatial_markers": len(set_a),
        "n_shared": inter,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, required=True)
    parser.add_argument("--input-dir", type=str, default="evaluation/reports/h5ad")
    parser.add_argument("--output-dir", type=str, default="evaluation/reports/mousebrain_eval")
    parser.add_argument("--snrna-ref", type=str,
                        default="evaluation/data/mousebrain/snrna_cell_group_pseudobulk.csv")
    parser.add_argument("--min-cells", type=int, default=20)
    parser.add_argument("--n-ref-markers", type=int, default=50)
    parser.add_argument("--n-spatial-markers", type=int, default=100)
    parser.add_argument("--specificity-threshold", type=float, default=2.0)
    parser.add_argument("--min-expr", type=float, default=1.0)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir) / "gene_level_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = normalize_adata(sc.read_h5ad(input_dir / f"{args.tag}_raw.h5ad"))
    sp = normalize_adata(sc.read_h5ad(input_dir / f"{args.tag}_SPARKLE.h5ad"))
    ref = pd.read_csv(args.snrna_ref, index_col=0)

    pb_raw = compute_pseudobulk(raw, min_cells=args.min_cells)
    pb_sp = compute_pseudobulk(sp, min_cells=args.min_cells)
    ref_sub = ref[pb_raw.columns]

    print(f"Groups retained (min_cells={args.min_cells}): {pb_raw.columns.tolist()}")

    ref_markers = find_reference_markers(
        ref_sub,
        n_markers=args.n_ref_markers,
        specificity_threshold=args.specificity_threshold,
        min_expr=args.min_expr,
    )

    results = []
    for spatial_by in ["logfc", "specificity"]:
        raw_m = find_spatial_markers(pb_raw, n_markers=args.n_spatial_markers, by=spatial_by)
        sp_m = find_spatial_markers(pb_sp, n_markers=args.n_spatial_markers, by=spatial_by)

        for g in pb_raw.columns:
            raw_metrics = overlap_metrics(raw_m[g], ref_markers[g])
            sp_metrics = overlap_metrics(sp_m[g], ref_markers[g])
            results.append({
                "spatial_marker_def": spatial_by,
                "group": g,
                "n_ref_markers": raw_metrics["n_ref_markers"],
                "raw_n_spatial": raw_metrics["n_spatial_markers"],
                "sparkle_n_spatial": sp_metrics["n_spatial_markers"],
                "raw_n_shared": raw_metrics["n_shared"],
                "sparkle_n_shared": sp_metrics["n_shared"],
                "raw_jaccard": raw_metrics["jaccard"],
                "sparkle_jaccard": sp_metrics["jaccard"],
                "raw_precision": raw_metrics["precision"],
                "sparkle_precision": sp_metrics["precision"],
                "raw_recall": raw_metrics["recall"],
                "sparkle_recall": sp_metrics["recall"],
                "delta_jaccard": sp_metrics["jaccard"] - raw_metrics["jaccard"],
                "delta_precision": sp_metrics["precision"] - raw_metrics["precision"],
                "delta_recall": sp_metrics["recall"] - raw_metrics["recall"],
            })

    df = pd.DataFrame(results)
    out_path = out_dir / f"{args.tag}_findmarker_overlap_v3_min{args.min_cells}_ref{args.n_ref_markers}_sp{args.n_spatial_markers}.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    # Summary
    summary_rows = []
    for smd in df["spatial_marker_def"].unique():
        sub = df[df["spatial_marker_def"] == smd]
        summary_rows.append({
            "spatial_marker_def": smd,
            "raw_jaccard_mean": sub["raw_jaccard"].mean(),
            "sparkle_jaccard_mean": sub["sparkle_jaccard"].mean(),
            "delta_jaccard": sub["delta_jaccard"].mean(),
            "raw_precision_mean": sub["raw_precision"].mean(),
            "sparkle_precision_mean": sub["sparkle_precision"].mean(),
            "delta_precision": sub["delta_precision"].mean(),
            "raw_recall_mean": sub["raw_recall"].mean(),
            "sparkle_recall_mean": sub["sparkle_recall"].mean(),
            "delta_recall": sub["delta_recall"].mean(),
        })
    summary = pd.DataFrame(summary_rows)
    print("\n=== Summary across groups ===")
    print(summary.to_markdown(index=False))

    summary_path = out_dir / f"{args.tag}_findmarker_overlap_v3_summary_min{args.min_cells}_ref{args.n_ref_markers}_sp{args.n_spatial_markers}.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved summary: {summary_path}")

    # Per-group detail
    print("\n=== Per-group recall (reference markers recovered by spatial logFC topN) ===")
    sub = df[df["spatial_marker_def"] == "logfc"]
    print(sub[["group", "n_ref_markers", "raw_n_shared", "sparkle_n_shared",
               "raw_recall", "sparkle_recall", "delta_recall"]].to_markdown(index=False))


if __name__ == "__main__":
    main()
