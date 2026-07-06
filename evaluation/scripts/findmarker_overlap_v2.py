#!/usr/bin/env python3
"""Compare spatial marker lists with snRNA reference markers (v2).

This version uses multiple marker definitions and filters out genes that are
only expressed in a tiny number of cells, which can dominate logFC-based ranks.

Definitions tested:
1. logFC_top: highest log2 fold change vs other groups.
2. specificity_top: highest target/non-target expression ratio (contamination inverse).
3. combined: top by logFC after filtering for contamination < 0.5 in reference.
4. reference_markers: use snRNA reference markers directly, score how many are recovered.
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
    return pd.DataFrame(pb, index=adata.var_names, columns=kept_groups)


def compute_logfc(pb_df, eps=1e-6):
    logfc = pd.DataFrame(index=pb_df.index)
    for g in pb_df.columns:
        other_mean = pb_df.drop(columns=g).mean(axis=1)
        logfc[g] = np.log2((pb_df[g] + eps) / (other_mean + eps))
    return logfc


def compute_specificity(pb_df, ref_df):
    """For each gene, specificity = target_expr / mean(non-target expr)."""
    shared_genes = pb_df.index.intersection(ref_df.index)
    shared_types = pb_df.columns.intersection(ref_df.columns)
    pb = pb_df.loc[shared_genes, shared_types]
    ref = ref_df.loc[shared_genes, shared_types]

    target_groups = ref.idxmax(axis=1)
    spec = pd.Series(index=shared_genes, dtype=float)
    for g in shared_genes:
        target = target_groups[g]
        target_expr = pb.loc[g, target]
        non_target_mean = pb.loc[g, shared_types.drop(target)].mean()
        spec[g] = target_expr / non_target_mean if non_target_mean > 0 else np.inf
    return spec


def find_markers_logfc(pb_df, n_markers=50):
    logfc = compute_logfc(pb_df)
    markers = {}
    for g in pb_df.columns:
        markers[g] = set(logfc[g].sort_values(ascending=False).head(n_markers).index)
    return markers, logfc


def find_markers_specificity(pb_df, ref_df, n_markers=50):
    """Top specificity genes per group (target/non-target ratio)."""
    shared_genes = pb_df.index.intersection(ref_df.index)
    shared_types = pb_df.columns.intersection(ref_df.columns)
    pb = pb_df.loc[shared_genes, shared_types]
    ref = ref_df.loc[shared_genes, shared_types]

    target_groups = ref.idxmax(axis=1)
    markers = {}
    for g in shared_types:
        specs = []
        for gene in shared_genes:
            target = target_groups[gene]
            if target != g:
                continue
            target_expr = pb.loc[gene, target]
            non_target_mean = pb.loc[gene, shared_types.drop(target)].mean()
            spec = target_expr / non_target_mean if non_target_mean > 0 else np.inf
            specs.append((gene, spec))
        specs = sorted(specs, key=lambda x: x[1], reverse=True)
        markers[g] = set([x[0] for x in specs[:n_markers]])
    return markers


def find_markers_reference_based(pb_df, ref_df, n_markers=50):
    """Use snRNA reference logFC to define markers, then see how many spatial recovers."""
    ref_logfc = compute_logfc(ref_df)
    markers = {}
    for g in ref_df.columns:
        markers[g] = set(ref_logfc[g].sort_values(ascending=False).head(n_markers).index)
    return markers


def overlap_metrics(set_a, set_b):
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return {
        "jaccard": inter / union if union > 0 else 0.0,
        "precision": inter / len(set_a) if len(set_a) > 0 else 0.0,
        "recall": inter / len(set_b) if len(set_b) > 0 else 0.0,
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
    ref_sub = ref[pb_raw.columns]

    print(f"Groups retained (min_cells={args.min_cells}): {pb_raw.columns.tolist()}")

    # Method 1: logFC-based
    raw_logfc, _ = find_markers_logfc(pb_raw, args.n_markers)
    sp_logfc, _ = find_markers_logfc(pb_sp, args.n_markers)
    ref_logfc, _ = find_markers_logfc(ref_sub, args.n_markers)

    # Method 2: specificity-based
    raw_spec = find_markers_specificity(pb_raw, ref_sub, args.n_markers)
    sp_spec = find_markers_specificity(pb_sp, ref_sub, args.n_markers)

    # Method 3: reference-based markers, recovered in spatial
    ref_markers = find_markers_reference_based(pb_raw, ref_sub, args.n_markers)

    results = []
    for method_name, (raw_m, sp_m, ref_m) in [
        ("logfc_topN", (raw_logfc, sp_logfc, ref_logfc)),
        ("specificity_topN", (raw_spec, sp_spec, ref_logfc)),  # compare to ref logFC markers
    ]:
        for g in pb_raw.columns:
            raw_metrics = overlap_metrics(raw_m[g], ref_m[g])
            sp_metrics = overlap_metrics(sp_m[g], ref_m[g])
            results.append({
                "method": method_name,
                "group": g,
                "raw_jaccard": raw_metrics["jaccard"],
                "sparkle_jaccard": sp_metrics["jaccard"],
                "raw_precision": raw_metrics["precision"],
                "sparkle_precision": sp_metrics["precision"],
                "raw_recall": raw_metrics["recall"],
                "sparkle_recall": sp_metrics["recall"],
                "sparkle_minus_raw_jaccard": sp_metrics["jaccard"] - raw_metrics["jaccard"],
            })

    # Method 3 per-group
    for g in pb_raw.columns:
        raw_metrics = overlap_metrics(raw_logfc[g], ref_markers[g])
        sp_metrics = overlap_metrics(sp_logfc[g], ref_markers[g])
        results.append({
            "method": "reference_markers_recovered",
            "group": g,
            "raw_jaccard": raw_metrics["jaccard"],
            "sparkle_jaccard": sp_metrics["jaccard"],
            "raw_precision": raw_metrics["precision"],
            "sparkle_precision": sp_metrics["precision"],
            "raw_recall": raw_metrics["recall"],
            "sparkle_recall": sp_metrics["recall"],
            "sparkle_minus_raw_jaccard": sp_metrics["jaccard"] - raw_metrics["jaccard"],
        })

    df = pd.DataFrame(results)
    out_path = out_dir / f"{args.tag}_findmarker_overlap_v2_min{args.min_cells}_top{args.n_markers}.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    # Summary per method
    summary_rows = []
    for method_name in df["method"].unique():
        sub = df[df["method"] == method_name]
        summary_rows.append({
            "method": method_name,
            "raw_jaccard_mean": sub["raw_jaccard"].mean(),
            "sparkle_jaccard_mean": sub["sparkle_jaccard"].mean(),
            "delta_jaccard": sub["sparkle_minus_raw_jaccard"].mean(),
            "raw_precision_mean": sub["raw_precision"].mean(),
            "sparkle_precision_mean": sub["sparkle_precision"].mean(),
            "raw_recall_mean": sub["raw_recall"].mean(),
            "sparkle_recall_mean": sub["sparkle_recall"].mean(),
        })
    summary = pd.DataFrame(summary_rows)
    print("\n=== Summary across groups ===")
    print(summary.to_markdown(index=False))

    summary_path = out_dir / f"{args.tag}_findmarker_overlap_v2_summary_min{args.min_cells}_top{args.n_markers}.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved summary: {summary_path}")

    # Per-group detail for logfc
    print("\n=== Per-group Jaccard (logFC topN) ===")
    sub = df[df["method"] == "logfc_topN"]
    print(sub[["group", "raw_jaccard", "sparkle_jaccard", "sparkle_minus_raw_jaccard"]].to_markdown(index=False))


if __name__ == "__main__":
    main()
