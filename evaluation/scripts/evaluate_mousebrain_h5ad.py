#!/usr/bin/env python3
"""Evaluate corrected MouseBrain h5ad outputs.

Inputs:
    - Per-method cell-based h5ad files saved by final_comparison.py
      (e.g. mousebrain_x..._y..._RAW.h5ad, ..._SPARKLE.h5ad, ..._DecontX.h5ad)
    - Optional snRNA-seq reference pseudobulk (genes x cell_group)
      e.g. evaluation/data/mousebrain/snrna_cell_group_pseudobulk.csv

Outputs (under evaluation/reports/mousebrain_eval/):
    - {tag}_{method}_celltype_corr_heatmap.png : Pearson correlation heatmap
      between cell-group pseudobulk profiles.
    - {tag}_snrna_corr_barplot.png : mean Pearson/Spearman correlation of each
      method's cell-group profiles to the snRNA reference.
    - {tag}_snrna_corr_per_celltype.csv : per-cell-group correlation table.
    - {tag}_metrics.json : silhouette, snRNA mean correlation, etc.
    - {tag}_summary.csv : numeric summary table across methods.

Normalization:
    By default the script normalizes h5ad X to log1p-CPM.  If you already have
    SCTransform-standardized h5ad files, pass --use-sctransform.
"""

import argparse
import json
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import silhouette_score


def find_h5ad_files(input_dir, tag, methods):
    """Return dict method_name -> h5ad path for existing files."""
    files = {}
    for method in methods:
        path = Path(input_dir) / f"{tag}_{method}.h5ad"
        # final_comparison.py saves the RAW file as *_raw.h5ad (lowercase)
        if not path.exists() and method.lower() == "raw":
            path = Path(input_dir) / f"{tag}_raw.h5ad"
        if path.exists():
            files[method] = path
        else:
            warnings.warn(f"File not found: {path}")
    return files


def normalize_adata(adata, use_sctransform=False):
    """Return a normalized AnnData object.

    Corrected outputs may contain negative values; these are clipped to 0
    before normalization.
    """
    adata = adata.copy()
    if hasattr(adata.X, "toarray"):
        adata.X = adata.X.toarray()
    adata.X = np.maximum(np.asarray(adata.X, dtype=np.float64), 0.0)

    if use_sctransform:
        from pysctransform import SCTransform
        n_hvgs = min(2000, adata.n_vars)
        residuals = SCTransform(adata, var_features_n=n_hvgs)
        adata.X = residuals.values
    else:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    return adata


def compute_pseudobulk(adata, group_key="annotation", min_cells=3):
    """Compute mean normalized expression per cell group.

    Returns:
        DataFrame [genes x cell_group]
    """
    groups = adata.obs[group_key].astype(str)
    valid = groups != "Unknown"
    if valid.sum() == 0:
        raise ValueError("No cells with known annotations.")
    adata = adata[valid].copy()
    groups = adata.obs[group_key]
    group_names = sorted(groups.unique())

    expr = adata.X
    if hasattr(expr, "toarray"):
        expr = expr.toarray()
    expr = np.asarray(expr)

    pb = {}
    for g in group_names:
        mask = groups == g
        if mask.sum() < min_cells:
            continue
        pb[g] = expr[mask].mean(axis=0)
    pb_df = pd.DataFrame(pb, index=adata.var_names)
    return pb_df


def plot_celltype_correlation_heatmap(pb_df, title, out_path):
    """Pearson correlation heatmap between cell-group pseudobulk profiles."""
    corr = pb_df.corr(method="pearson")
    n = len(corr)
    fig, ax = plt.subplots(figsize=(max(6, n * 0.55), max(5, n * 0.5)))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="coolwarm",
        vmin=-1, vmax=1, center=0, square=True,
        linewidths=0.5, ax=ax,
    )
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return corr


def load_snrna_reference(ref_path):
    """Load snRNA reference pseudobulk as DataFrame [genes x cell_group]."""
    df = pd.read_csv(ref_path, index_col=0)
    return df


def compute_snrna_correlations(pb_df, snrna_ref, method="pearson"):
    """For each shared cell group, correlate spatial and snRNA profiles.

    Returns:
        dict cell_group -> correlation, and a pandas Series.
    """
    shared_genes = pb_df.index.intersection(snrna_ref.index)
    if len(shared_genes) < 10:
        raise ValueError(
            f"Only {len(shared_genes)} shared genes between spatial and snRNA."
        )
    pb = pb_df.loc[shared_genes]
    ref = snrna_ref.loc[shared_genes]
    shared_types = pb.columns.intersection(ref.columns)
    if len(shared_types) == 0:
        raise ValueError("No shared cell groups between spatial and snRNA.")

    corrs = {}
    for ct in shared_types:
        x = pb[ct].values
        y = ref[ct].values
        if np.std(x) < 1e-12 or np.std(y) < 1e-12:
            corrs[ct] = np.nan
            continue
        if method == "pearson":
            r, _ = pearsonr(x, y)
        else:
            r, _ = spearmanr(x, y)
        corrs[ct] = r
    return corrs, pd.Series(corrs, name=method)


def plot_snrna_comparison(summary, out_path):
    """Bar plot of mean spatial-vs-snRNA correlation per method."""
    methods = list(summary.keys())
    vals = [summary[m]["mean_snrna_corr"] for m in methods]
    colors = ["#999999" if m.lower() == "raw" else "#4c78a8" for m in methods]

    fig, ax = plt.subplots(figsize=(max(7, len(methods) * 0.8), 5))
    bars = ax.bar(methods, vals, color=colors)
    ax.set_ylabel("Mean correlation with snRNA reference")
    ax.set_title("Spatial cell-group profiles vs snRNA reference")
    y_min = min(0, np.nanmin(vals) - 0.05)
    y_max = np.nanmax(vals) + 0.05
    ax.set_ylim(y_min, y_max)
    ax.axhline(0, color="black", linewidth=0.5)
    for bar, v in zip(bars, vals):
        if not np.isnan(v):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                v + 0.01,
                f"{v:.3f}",
                ha="center", va="bottom", fontsize=9,
            )
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def compute_silhouette(adata, group_key="annotation", n_pcs=30):
    """Compute cell-group silhouette score on log-normalized PCA."""
    adata = adata.copy()
    n_pcs = min(n_pcs, adata.n_obs - 1, adata.n_vars - 1)
    if n_pcs < 2:
        return float("nan")
    sc.pp.pca(adata, n_comps=n_pcs, svd_solver="arpack")
    labels = adata.obs[group_key].astype(str).values
    valid = labels != "Unknown"
    if valid.sum() < 10 or len(set(labels[valid])) < 2:
        return float("nan")
    score = silhouette_score(adata.obsm["X_pca"][valid], labels[valid])
    return float(score)


def compute_mean_between_corr(pb_df):
    """Return mean off-diagonal (between cell-group) Pearson correlation."""
    corr = pb_df.corr(method="pearson").values
    if corr.shape[0] < 2:
        return float("nan")
    mask = ~np.eye(corr.shape[0], dtype=bool)
    return float(np.mean(corr[mask]))


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate corrected MouseBrain h5ad outputs."
    )
    parser.add_argument(
        "--tag", type=str, required=True,
        help="Filename tag, e.g. mousebrain_x13000-13500_y3000-3500",
    )
    parser.add_argument(
        "--input-dir", type=str,
        default="evaluation/reports/h5ad",
        help="Directory containing h5ad files (default: evaluation/reports/h5ad)",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default="evaluation/reports/mousebrain_eval",
        help="Output directory (default: evaluation/reports/mousebrain_eval)",
    )
    parser.add_argument(
        "--snrna-ref", type=str,
        default="evaluation/data/mousebrain/snrna_cell_group_pseudobulk.csv",
        help="Path to snRNA reference pseudobulk CSV (genes x cell_group). "
             "If omitted, snRNA comparison is skipped.",
    )
    parser.add_argument(
        "--methods", type=str,
        default="RAW,SPARKLE,SpatialSoupX,SoupX,DecontX",
        help="Comma-separated method names matching h5ad suffixes",
    )
    parser.add_argument(
        "--use-sctransform", action="store_true",
        help="Use pysctransform instead of log1p-CPM normalization",
    )
    parser.add_argument(
        "--corr-method", type=str, default="pearson",
        choices=["pearson", "spearman"],
        help="Correlation method for snRNA comparison",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    files = find_h5ad_files(input_dir, args.tag, methods)
    if not files:
        raise FileNotFoundError(f"No h5ad files found for tag {args.tag}")

    snrna_ref = None
    if args.snrna_ref:
        snrna_ref = load_snrna_reference(args.snrna_ref)
        print(f"Loaded snRNA reference: {snrna_ref.shape}")

    metrics = {
        "tag": args.tag,
        "methods": {},
        "snrna_comparison": {},
    }
    summary_rows = []
    per_method_corr = {}

    for method, path in files.items():
        print(f"\nProcessing {method} ...")
        adata = sc.read_h5ad(path)
        adata = normalize_adata(adata, use_sctransform=args.use_sctransform)

        # 1. cell-type correlation heatmap
        pb_df = compute_pseudobulk(adata, group_key="annotation", min_cells=3)
        corr_path = output_dir / f"{args.tag}_{method}_celltype_corr_heatmap.png"
        corr = plot_celltype_correlation_heatmap(
            pb_df,
            title=f"{method}: cell-group pseudobulk correlation",
            out_path=corr_path,
        )
        per_method_corr[method] = corr
        print(f"  Saved heatmap: {corr_path}")
        print(f"  Cell groups: {list(pb_df.columns)}")

        # Silhouette
        sil = compute_silhouette(adata, group_key="annotation")
        print(f"  Silhouette (cell_group): {sil:.4f}")

        mean_between = compute_mean_between_corr(pb_df)

        method_metrics = {
            "n_cells": int(adata.n_obs),
            "n_genes": int(adata.n_vars),
            "n_groups": int(pb_df.shape[1]),
            "silhouette": sil,
            "mean_between_corr": mean_between,
        }

        # 2. snRNA correlation comparison
        if snrna_ref is not None:
            try:
                corrs, series = compute_snrna_correlations(
                    pb_df, snrna_ref, method=args.corr_method
                )
                mean_corr = float(series.mean(skipna=True))
                method_metrics["mean_snrna_corr"] = mean_corr
                method_metrics["snrna_corr_per_group"] = {
                    k: (float(v) if not np.isnan(v) else None)
                    for k, v in corrs.items()
                }
                print(f"  Mean {args.corr_method} corr with snRNA: {mean_corr:.4f}")
            except Exception as e:
                warnings.warn(f"snRNA comparison failed for {method}: {e}")

        metrics["methods"][method] = method_metrics
        summary_rows.append({
            "method": method,
            "n_cells": method_metrics["n_cells"],
            "n_groups": method_metrics["n_groups"],
            "silhouette": method_metrics["silhouette"],
            "mean_between_corr": method_metrics["mean_between_corr"],
            "mean_snrna_corr": method_metrics.get("mean_snrna_corr", np.nan),
        })

    # snRNA bar plot
    if snrna_ref is not None:
        snrna_summary = {
            m: {"mean_snrna_corr": metrics["methods"][m].get("mean_snrna_corr", np.nan)}
            for m in metrics["methods"]
        }
        bar_path = output_dir / f"{args.tag}_snrna_corr_barplot.png"
        plot_snrna_comparison(snrna_summary, bar_path)
        print(f"\nSaved snRNA comparison bar plot: {bar_path}")

        # per-celltype CSV
        per_ct_rows = []
        for method in metrics["methods"]:
            per_sub = metrics["methods"][method].get("snrna_corr_per_group", {})
            for ct, val in per_sub.items():
                per_ct_rows.append({"method": method, "cell_group": ct, "corr": val})
        per_ct_df = pd.DataFrame(per_ct_rows)
        per_ct_path = output_dir / f"{args.tag}_snrna_corr_per_celltype.csv"
        per_ct_df.to_csv(per_ct_path, index=False)
        print(f"Saved per-celltype snRNA correlations: {per_ct_path}")

    # Save JSON and summary CSV
    json_path = output_dir / f"{args.tag}_metrics.json"
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else str(x))
    print(f"Saved metrics JSON: {json_path}")

    summary_df = pd.DataFrame(summary_rows)
    csv_path = output_dir / f"{args.tag}_summary.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"Saved summary CSV: {csv_path}")


if __name__ == "__main__":
    main()
