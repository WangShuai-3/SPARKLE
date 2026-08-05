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
    method_suffixes = {
        "RAW": "raw",
        "SpotClean": "SpotCleanOfficial",
    }
    files = {}
    for method in methods:
        suffix = method_suffixes.get(method, method)
        path = Path(input_dir) / f"{tag}_{suffix}.h5ad"
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
        # residuals is cells x HVGs DataFrame; build a new AnnData with HVGs only
        adata = sc.AnnData(
            X=residuals.values,
            obs=adata.obs.loc[residuals.index].copy(),
            var=pd.DataFrame(index=residuals.columns),
        )
    else:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    return adata


def compute_pseudobulk(adata, group_key="annotation", min_cells=3):
    """Compute mean normalized expression per cell group.

    Returns:
        DataFrame [genes x cell_group]
    """
    groups = (
        adata.obs[group_key].astype(object).fillna("Unknown").astype(str)
    )
    valid = ~groups.isin(["Unknown", "nan", "None", ""])
    if valid.sum() == 0:
        raise ValueError("No cells with known annotations.")
    adata = adata[valid].copy()
    groups = groups.loc[adata.obs_names]
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


def load_sparkle_corrected_genes(input_dir, tag, r2_threshold=None):
    """Load gene names marked sparkle_corrected from the SPARKLE h5ad.

    Args:
        r2_threshold: if provided (float), filter by sparkle_r2 >= r2_threshold
            instead of using the boolean sparkle_corrected column.

    Returns:
        list of gene names, or None if SPARKLE h5ad / column not found.
    """
    sp_path = Path(input_dir) / f"{tag}_SPARKLE.h5ad"
    if not sp_path.exists():
        return None
    adata = sc.read_h5ad(sp_path)
    if r2_threshold is not None:
        if "sparkle_r2" not in adata.var.columns:
            return None
        genes = adata.var_names[adata.var["sparkle_r2"].values >= r2_threshold].tolist()
    else:
        if "sparkle_corrected" not in adata.var.columns:
            return None
        genes = adata.var_names[adata.var["sparkle_corrected"].values].tolist()
    return genes


def get_hvg_gene_set(raw_path, n_hvgs, use_sctransform=False, sparkle_genes=None):
    """Return a gene set of top n_hvgs highly variable genes from RAW data.

    If sparkle_genes is provided, the result is intersected with it so that
    the final set only contains SPARKLE-corrected HVGs.  When use_sctransform
    is True, SCTransform itself selects HVGs, so this function just returns
    the sparkle gene set (or None) to avoid double selection.
    """
    if use_sctransform:
        return sparkle_genes

    adata = sc.read_h5ad(raw_path)
    adata = normalize_adata(adata, use_sctransform=False)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_hvgs, flavor="seurat")
    hvgs = adata.var_names[adata.var["highly_variable"].values].tolist()
    print(f"Selected {len(hvgs)} HVGs from RAW data")
    if sparkle_genes is not None:
        sparkle_set = set(sparkle_genes)
        hvgs = [g for g in hvgs if g in sparkle_set]
        print(f"After intersecting with SPARKLE-corrected genes: {len(hvgs)} genes")
    return hvgs


def compute_snrna_correlations(pb_df, snrna_ref, method="pearson", gene_mask=None):
    """For each shared cell group, correlate spatial and snRNA profiles.

    Args:
        gene_mask: optional list of gene names to restrict the comparison to
            (e.g. genes that SPARKLE actually corrected).

    Returns:
        dict cell_group -> correlation, and a pandas Series.
    """
    shared_genes = pb_df.index.intersection(snrna_ref.index)
    if gene_mask is not None:
        shared_genes = shared_genes.intersection(pd.Index(gene_mask))
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
    labels = (
        adata.obs[group_key].astype(object).fillna("Unknown").astype(str)
    )
    valid = ~labels.isin(["Unknown", "nan", "None", ""])
    adata = adata[valid].copy()
    labels = labels.loc[adata.obs_names].to_numpy()
    n_pcs = min(n_pcs, adata.n_obs - 1, adata.n_vars - 1)
    if n_pcs < 2:
        return float("nan")
    if len(labels) < 10 or len(set(labels)) < 2:
        return float("nan")
    sc.pp.pca(adata, n_comps=n_pcs, svd_solver="arpack")
    score = silhouette_score(adata.obsm["X_pca"], labels)
    return float(score)


def compute_mean_between_corr(pb_df):
    """Return mean off-diagonal (between cell-group) Pearson correlation."""
    corr = pb_df.corr(method="pearson").values
    if corr.shape[0] < 2:
        return float("nan")
    mask = ~np.eye(corr.shape[0], dtype=bool)
    return float(np.mean(corr[mask]))


def compute_contamination(pb_df, snrna_ref):
    """Per-gene contamination score based on the snRNA target group.

    For each gene, the target group is the cell group with the highest snRNA
    expression.  Contamination is defined as mean expression in non-target
    groups divided by expression in the target group.

    Returns a Series indexed by gene with the per-gene contamination score.
    """
    shared_genes = pb_df.index.intersection(snrna_ref.index)
    shared_types = pb_df.columns.intersection(snrna_ref.columns)
    pb = pb_df.loc[shared_genes, shared_types]
    ref = snrna_ref.loc[shared_genes, shared_types]

    target_groups = ref.idxmax(axis=1)
    target_idx = [pb.columns.get_loc(g) for g in target_groups]
    target_expr = pb.to_numpy()[np.arange(len(pb)), target_idx]

    # Mean expression in non-target groups for each gene
    row_means = pb.mean(axis=1).to_numpy()
    non_target_mean = (row_means * pb.shape[1] - target_expr) / (pb.shape[1] - 1)

    contamination = np.full_like(target_expr, np.nan, dtype=float)
    mask = target_expr > 0
    contamination[mask] = non_target_mean[mask] / target_expr[mask]
    return pd.Series(contamination, index=shared_genes, name="contamination")


def compute_contamination_summary(pb_df, snrna_ref, raw_pb_df=None):
    """Return contamination score summary for a pseudobulk profile.

    If raw_pb_df is provided, also returns reductions vs RAW on the full gene
    set and on genes that were highly contaminated in RAW (contamination > 1).
    """
    contam = compute_contamination(pb_df, snrna_ref)
    summary = {
        "mean_contamination": float(contam.mean(skipna=True)),
        "median_contamination": float(contam.median(skipna=True)),
        "n_contam_genes": int(contam.notna().sum()),
    }

    if raw_pb_df is not None:
        raw_contam = compute_contamination(raw_pb_df, snrna_ref)
        # Compare all methods on the same gene set: genes with valid RAW
        # contamination scores.  This avoids gene-set differences introduced by
        # methods that change which genes have non-zero target expression.
        shared = raw_contam.dropna().index.intersection(contam.index)
        reduction = raw_contam.loc[shared] - contam.loc[shared]
        summary["mean_contamination_reduction"] = float(reduction.mean(skipna=True))
        summary["median_contamination_reduction"] = float(reduction.median(skipna=True))

        # Focus on genes highly contaminated in RAW
        high_mask = raw_contam.loc[shared] > 1.0
        if high_mask.sum() > 0:
            summary["mean_contamination_reduction_high_raw"] = float(
                reduction.loc[high_mask].mean(skipna=True)
            )
            summary["n_high_contam_genes"] = int(high_mask.sum())
        else:
            summary["mean_contamination_reduction_high_raw"] = float("nan")
            summary["n_high_contam_genes"] = 0

    return summary


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
        default="RAW,SPARKLE,SoupX,DecontX,SpotClean",
        help="Comma-separated method names matching h5ad suffixes",
    )
    parser.add_argument(
        "--use-sctransform", action="store_true",
        help="Use pysctransform instead of log1p-CPM normalization",
    )
    parser.add_argument(
        "--n-hvgs", type=int, default=None,
        help="If set, restrict analysis to the top n_hvgs highly variable "
             "genes (computed from RAW data) intersected with "
             "SPARKLE-corrected genes when applicable. Ignored under "
             "--use-sctransform.",
    )
    parser.add_argument(
        "--corr-method", type=str, default="pearson",
        choices=["pearson", "spearman"],
        help="Correlation method for snRNA comparison. Pearson is kept as the "
             "default because SPARKLE's main mechanism is removing ambient-RNA "
             "contamination (i.e. reducing expression magnitude), which Pearson "
             "captures more directly than Spearman.",
    )
    parser.add_argument(
        "--use-sparkle-corrected-genes", action=argparse.BooleanOptionalAction, default=True,
        help="Restrict snRNA correlation to genes that SPARKLE actually corrected "
             "(sparkle_corrected == True in the SPARKLE h5ad var). "
             "Disable with --no-use-sparkle-corrected-genes to use all shared genes.",
    )
    parser.add_argument(
        "--r2-threshold", type=float, default=None,
        help="If set, filter SPARKLE-corrected genes by sparkle_r2 >= threshold "
             "instead of using the boolean sparkle_corrected column. "
        "Implies --use-sparkle-corrected-genes.",
    )
    parser.add_argument(
        "--incremental", action="store_true",
        help="Evaluate only --methods and merge them into an existing output "
             "directory, preserving metrics for methods evaluated previously.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if args.n_hvgs is not None:
        sub = f"method_level_hvg{args.n_hvgs}"
    elif args.use_sctransform:
        sub = "method_level_sct"
    else:
        sub = "method_level_new"
    if args.r2_threshold is not None:
        sub += f"_r2{args.r2_threshold}"
    output_dir = Path(args.output_dir) / sub
    output_dir.mkdir(parents=True, exist_ok=True)

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    files = find_h5ad_files(input_dir, args.tag, methods)
    if not files:
        raise FileNotFoundError(f"No h5ad files found for tag {args.tag}")

    snrna_ref = None
    if args.snrna_ref:
        snrna_ref = load_snrna_reference(args.snrna_ref)
        print(f"Loaded snRNA reference: {snrna_ref.shape}")

    sparkle_corrected_genes = None
    if args.r2_threshold is not None:
        sparkle_corrected_genes = load_sparkle_corrected_genes(
            input_dir, args.tag, r2_threshold=args.r2_threshold
        )
        if sparkle_corrected_genes is None:
            warnings.warn(
                f"--r2-threshold={args.r2_threshold} requested but could not find "
                f"{args.tag}_SPARKLE.h5ad or sparkle_r2 column. "
                "Falling back to all shared genes."
            )
        else:
            print(f"Restricting snRNA comparison to {len(sparkle_corrected_genes)} genes with sparkle_r2 >= {args.r2_threshold}")
    elif args.use_sparkle_corrected_genes:
        sparkle_corrected_genes = load_sparkle_corrected_genes(input_dir, args.tag)
        if sparkle_corrected_genes is None:
            warnings.warn(
                "--sparkle-corrected-only requested but could not find "
                f"{args.tag}_SPARKLE.h5ad or sparkle_corrected column. "
                "Falling back to all shared genes."
            )
        else:
            print(f"Restricting snRNA comparison to {len(sparkle_corrected_genes)} SPARKLE-corrected genes")
    else:
        sparkle_corrected_genes = None

    # Optionally restrict to a HVG subset (computed from RAW)
    hvg_gene_set = None
    if args.n_hvgs is not None and args.n_hvgs > 0:
        raw_path = files.get("RAW") or files.get("raw")
        if raw_path is None:
            warnings.warn("--n-hvgs requested but no RAW file found; ignoring.")
        else:
            hvg_gene_set = get_hvg_gene_set(
                raw_path, args.n_hvgs,
                use_sctransform=args.use_sctransform,
                sparkle_genes=sparkle_corrected_genes,
            )
            if hvg_gene_set is not None:
                print(f"Restricting analysis to {len(hvg_gene_set)} HVGs")

    json_path = output_dir / f"{args.tag}_metrics.json"
    csv_path = output_dir / f"{args.tag}_summary.csv"
    per_ct_path = output_dir / f"{args.tag}_snrna_corr_per_celltype.csv"

    if args.incremental and json_path.exists():
        with open(json_path) as f:
            metrics = json.load(f)
        metrics["tag"] = args.tag
        metrics.setdefault("methods", {})
        metrics.setdefault("snrna_comparison", {})
    else:
        metrics = {
            "tag": args.tag,
            "methods": {},
            "snrna_comparison": {},
        }

    if args.incremental and csv_path.exists():
        previous_summary = pd.read_csv(csv_path)
        previous_summary = previous_summary[
            ~previous_summary["method"].isin(files.keys())
        ]
        previous_summary = previous_summary.drop(
            columns=["contamination_reduction_vs_raw"], errors="ignore"
        )
        summary_rows = previous_summary.to_dict("records")
    else:
        summary_rows = []
    per_method_corr = {}
    raw_pb_df = None
    if args.incremental and snrna_ref is not None and not any(
        method.lower() == "raw" for method in files
    ):
        raw_path = find_h5ad_files(input_dir, args.tag, ["RAW"]).get("RAW")
        if raw_path is not None:
            print("Loading RAW pseudobulk only for paired contamination deltas ...")
            raw_adata = normalize_adata(
                sc.read_h5ad(raw_path), use_sctransform=args.use_sctransform
            )
            if hvg_gene_set is not None:
                shared_hvgs = raw_adata.var_names.intersection(hvg_gene_set)
                raw_adata = raw_adata[:, shared_hvgs].copy()
            raw_pb_df = compute_pseudobulk(
                raw_adata, group_key="annotation", min_cells=3
            )
            del raw_adata

    for method, path in files.items():
        print(f"\nProcessing {method} ...")
        adata = sc.read_h5ad(path)
        adata = normalize_adata(adata, use_sctransform=args.use_sctransform)
        if hvg_gene_set is not None:
            shared_hvgs = adata.var_names.intersection(hvg_gene_set)
            adata = adata[:, shared_hvgs].copy()
            print(f"  Using {adata.n_vars} HVGs for analysis")

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
        group_values = (
            adata.obs["annotation"]
            .astype(object)
            .fillna("Unknown")
            .astype(str)
        )
        retained_mask = ~group_values.isin(["Unknown", "nan", "None", ""])

        method_metrics = {
            "n_cells": int(retained_mask.sum()),
            "n_input_cells": int(adata.n_obs),
            "n_genes": int(adata.n_vars),
            "n_groups": int(pb_df.shape[1]),
            "silhouette": sil,
            "mean_between_corr": mean_between,
        }

        # 2. snRNA correlation comparison
        if snrna_ref is not None:
            try:
                gene_mask = hvg_gene_set if hvg_gene_set is not None else sparkle_corrected_genes
                corrs, series = compute_snrna_correlations(
                    pb_df, snrna_ref, method=args.corr_method,
                    gene_mask=gene_mask,
                )
                mean_corr = float(series.mean(skipna=True))
                method_metrics["n_snrna_genes"] = len(
                    pb_df.index.intersection(snrna_ref.index).intersection(
                        pd.Index(sparkle_corrected_genes)
                    )
                ) if sparkle_corrected_genes is not None else len(
                    pb_df.index.intersection(snrna_ref.index)
                )
                method_metrics["mean_snrna_corr"] = mean_corr
                method_metrics["snrna_corr_per_group"] = {
                    k: (float(v) if not np.isnan(v) else None)
                    for k, v in corrs.items()
                }
                scope = " (SPARKLE-corrected genes only)" if sparkle_corrected_genes is not None else ""
                print(f"  Mean {args.corr_method} corr with snRNA{scope}: {mean_corr:.4f} ({method_metrics['n_snrna_genes']} genes)")
            except Exception as e:
                warnings.warn(f"snRNA comparison failed for {method}: {e}")

            # 3. Contamination summary (uses full shared gene set)
            try:
                if method.lower() == "raw":
                    raw_pb_df = pb_df
                contam_summary = compute_contamination_summary(
                    pb_df, snrna_ref, raw_pb_df=raw_pb_df
                )
                method_metrics.update(contam_summary)
                print(
                    f"  Mean contamination: {contam_summary['mean_contamination']:.4f} "
                    f"(median {contam_summary['median_contamination']:.4f}, "
                    f"{contam_summary['n_contam_genes']} genes)"
                )
                if "mean_contamination_reduction_high_raw" in contam_summary:
                    print(
                        f"  Contamination reduction vs RAW (high-contam genes): "
                        f"{contam_summary['mean_contamination_reduction_high_raw']:.4f} "
                        f"({contam_summary.get('n_high_contam_genes', 0)} genes)"
                    )
            except Exception as e:
                warnings.warn(f"Contamination summary failed for {method}: {e}")

        metrics["methods"][method] = method_metrics
        summary_rows.append({
            "method": method,
            "n_cells": method_metrics["n_cells"],
            "n_input_cells": method_metrics["n_input_cells"],
            "n_groups": method_metrics["n_groups"],
            "silhouette": method_metrics["silhouette"],
            "mean_between_corr": method_metrics["mean_between_corr"],
            "mean_snrna_corr": method_metrics.get("mean_snrna_corr", np.nan),
            "mean_contamination": method_metrics.get("mean_contamination", np.nan),
            "median_contamination": method_metrics.get("median_contamination", np.nan),
            "mean_contamination_reduction": method_metrics.get("mean_contamination_reduction", np.nan),
            "mean_contamination_reduction_high_raw": method_metrics.get("mean_contamination_reduction_high_raw", np.nan),
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
        if args.incremental and per_ct_path.exists():
            old_per_ct = pd.read_csv(per_ct_path)
            old_per_ct = old_per_ct[~old_per_ct["method"].isin(files.keys())]
            key = ["method", "cell_group"]
            spreads = old_per_ct.groupby(key, dropna=False)["corr"].agg(
                lambda values: values.max() - values.min()
            )
            if (spreads.fillna(0) > 1e-12).any():
                raise ValueError(
                    "Existing per-celltype CSV has conflicting duplicate rows."
                )
            old_per_ct = old_per_ct.drop_duplicates(key, keep="first")
            per_ct_df = pd.concat([old_per_ct, per_ct_df], ignore_index=True)
        if per_ct_df.duplicated(["method", "cell_group"]).any():
            raise ValueError("Per-celltype output keys are not unique.")
        per_ct_df.to_csv(per_ct_path, index=False)
        print(f"Saved per-celltype snRNA correlations: {per_ct_path}")

    # Save JSON and summary CSV
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else str(x))
    print(f"Saved metrics JSON: {json_path}")

    summary_df = pd.DataFrame(summary_rows)

    # Contamination reduction vs RAW
    raw_method = next(
        (m for m in metrics["methods"] if m.lower() == "raw"), None
    )
    if snrna_ref is not None and raw_method is not None:
        raw_contam = metrics["methods"][raw_method].get(
            "mean_contamination", np.nan
        )
        if not np.isnan(raw_contam):
            reductions = []
            for row in summary_rows:
                m = row["method"]
                mc = metrics["methods"][m].get("mean_contamination", np.nan)
                reductions.append({
                    "method": m,
                    "mean_contamination": mc,
                    "contamination_reduction_vs_raw": raw_contam - mc,
                })
            reduc_df = pd.DataFrame(reductions)
            reduc_path = output_dir / f"{args.tag}_contamination_reduction.csv"
            reduc_df.to_csv(reduc_path, index=False)
            print(f"Saved contamination reduction CSV: {reduc_path}")
            summary_df = summary_df.merge(
                reduc_df[["method", "contamination_reduction_vs_raw"]],
                on="method", how="left"
            )
            print("\nContamination reduction vs RAW:")
            for _, r in reduc_df.iterrows():
                print(f"  {r['method']}: {r['contamination_reduction_vs_raw']:.4f}")

    summary_df.to_csv(csv_path, index=False)
    print(f"Saved summary CSV: {csv_path}")


if __name__ == "__main__":
    main()
