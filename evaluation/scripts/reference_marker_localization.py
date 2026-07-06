#!/usr/bin/env python3
"""Assess how well snRNA reference markers are localized to the correct spatial cell group.

For each reference cell group, we take its marker genes and ask: in the spatial
pseudobulk, which cell group has the highest expression for each marker?  The
fraction of markers whose highest-expressing spatial group matches the reference
target group measures localization accuracy.

This is more robust than top-N overlap because it does not depend on choosing a
spatial marker threshold.
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


def find_reference_markers(ref_df, n_markers=50, specificity_threshold=2.0, min_expr=1.0):
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
        candidates["score"] = candidates["expr"] * np.log1p(candidates["spec"])
        top = candidates.sort_values("score", ascending=False).head(n_markers)
        markers[g] = set(top.index)
    return markers


def evaluate_localization(pb_df, ref_markers):
    """For each reference group, fraction of its markers localized correctly."""
    rows = []
    for g in pb_df.columns:
        if g not in ref_markers:
            continue
        markers = [m for m in ref_markers[g] if m in pb_df.index]
        if len(markers) == 0:
            continue
        correct = 0
        for m in markers:
            top_spatial_group = pb_df.loc[m].idxmax()
            if top_spatial_group == g:
                correct += 1
        rows.append({
            "group": g,
            "n_markers": len(markers),
            "n_correct": correct,
            "accuracy": correct / len(markers),
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, required=True)
    parser.add_argument("--input-dir", type=str, default="evaluation/reports/h5ad")
    parser.add_argument("--output-dir", type=str, default="evaluation/reports/mousebrain_eval")
    parser.add_argument("--snrna-ref", type=str,
                        default="evaluation/data/mousebrain/snrna_cell_group_pseudobulk.csv")
    parser.add_argument("--min-cells", type=int, default=20)
    parser.add_argument("--n-ref-markers", type=int, default=50)
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

    ref_markers = find_reference_markers(
        ref_sub,
        n_markers=args.n_ref_markers,
        specificity_threshold=args.specificity_threshold,
        min_expr=args.min_expr,
    )

    raw_loc = evaluate_localization(pb_raw, ref_markers)
    sp_loc = evaluate_localization(pb_sp, ref_markers)

    df = raw_loc.merge(sp_loc, on="group", suffixes=("_raw", "_sparkle"))
    df["accuracy_delta"] = df["accuracy_sparkle"] - df["accuracy_raw"]

    out_path = out_dir / f"{args.tag}_reference_marker_localization_min{args.min_cells}_ref{args.n_ref_markers}.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

    print("\n=== Reference marker localization accuracy ===")
    print(df.to_markdown(index=False))

    print(f"\nMean RAW accuracy: {df['accuracy_raw'].mean():.4f}")
    print(f"Mean SPARKLE accuracy: {df['accuracy_sparkle'].mean():.4f}")
    print(f"Mean delta: {df['accuracy_delta'].mean():.4f}")
    print(f"Groups improved: {(df['accuracy_delta'] > 0).sum()}")
    print(f"Groups worsened: {(df['accuracy_delta'] < 0).sum()}")


if __name__ == "__main__":
    main()
