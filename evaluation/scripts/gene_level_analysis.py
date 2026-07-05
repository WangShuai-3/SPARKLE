#!/usr/bin/env python3
"""Gene-level analysis for MouseBrain evaluation.

Replicates and extends the analysis in
evaluation/reports/mousebrain_eval/gene_level_analysis/report.md.

For each gene we compute:
  - Pearson/Spearman correlation between the spatial cell-group pseudobulk
    profile and the snRNA reference profile.
  - Improvement = corr(SPARKLE) - corr(RAW).
  - Contamination score = mean expression in non-target groups / expression in
    target group, where target group is the group with highest snRNA expression.
  - Marker gene ranking within the target cell group.

Outputs are written to output_dir/gene_level_analysis/.
"""

import argparse
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from scipy.stats import pearsonr, spearmanr


KNOWN_MARKERS = {
    # Interneurons
    "Sst": "TE-N-GABA-SST",
    "Pvalb": "TE-N-GABA-PVALB",
    "Lamp5": "TE-N-GABA-LAMP5",
    "Vip": "TE-N-GABA-VIP",
    # Excitatory neurons
    "Cux2": "L2/3-IT-GLU",
    "Rorb": "L4/5-IT-GLU",
    "Bcl11b": "L5-PT-GLU",
    "Tbr1": "L6-N-GLU",
    # Glia / non-neuronal
    "Gfap": "ASC",
    "Plp1": "OL",
    "C1qa": "MGL",
    "Flt1": "VLMC",
}

LAYER_MARKERS = {
    "Cux2": "L2/3-IT-GLU",
    "Satb2": "L2/3-IT-GLU",
    "Rorb": "L4/5-IT-GLU",
    "Scnn1a": "L4/5-IT-GLU",
    "Fezf2": "L5-IT-GLU",
    "Bcl11b": "L5-PT-GLU",
    "Sema3e": "L5-PT-GLU",
    "Osr1": "L6-IT-GLU",
    "Tbr1": "L6-N-GLU",
}


def normalize_adata(adata, use_sctransform=False):
    """log1p-CPM or SCTransform normalization, clipping negatives to 0."""
    adata = adata.copy()
    if hasattr(adata.X, "toarray"):
        adata.X = adata.X.toarray()
    adata.X = np.maximum(np.asarray(adata.X, dtype=np.float64), 0.0)

    if use_sctransform:
        from pysctransform import SCTransform
        n_hvgs = min(2000, adata.n_vars)
        residuals = SCTransform(adata, var_features_n=n_hvgs)
        adata = sc.AnnData(
            X=residuals.values,
            obs=adata.obs.loc[residuals.index].copy(),
            var=pd.DataFrame(index=residuals.columns),
        )
    else:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    return adata


def compute_pseudobulk(adata, group_key="annotation", min_cells=1):
    """Mean normalized expression per cell group [genes x groups].

    Defaults to min_cells=1 so that rare cell groups (e.g. a single cell in
    this spatial window) are retained, matching the original gene-level report.
    """
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
    for g in group_names:
        mask = groups == g
        if mask.sum() < min_cells:
            continue
        pb[g] = expr[mask].mean(axis=0)
    return pd.DataFrame(pb, index=adata.var_names)


def corr_func(x, y, method="pearson"):
    """Return correlation or NaN for constant vectors."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan
    if method == "pearson":
        return pearsonr(x, y)[0]
    return spearmanr(x, y)[0]


def compute_per_gene_correlations(pb_df, ref_df, method="pearson"):
    """For each shared gene, correlate spatial and snRNA group profiles."""
    shared_genes = pb_df.index.intersection(ref_df.index)
    shared_types = pb_df.columns.intersection(ref_df.columns)
    corrs = {}
    for g in shared_genes:
        x = pb_df.loc[g, shared_types].values
        y = ref_df.loc[g, shared_types].values
        corrs[g] = corr_func(x, y, method=method)
    return pd.Series(corrs, name=f"{method}_corr")


def compute_contamination(pb_df, ref_df):
    """Contamination score per gene based on snRNA target group."""
    shared_genes = pb_df.index.intersection(ref_df.index)
    shared_types = pb_df.columns.intersection(ref_df.columns)
    records = []
    for g in shared_genes:
        ref_vals = ref_df.loc[g, shared_types]
        target = ref_vals.idxmax()
        target_expr = pb_df.loc[g, target]
        non_target = pb_df.loc[g, shared_types.drop(target)]
        contam = float(non_target.mean() / target_expr) if target_expr > 0 else np.nan
        records.append({
            "gene": g,
            "target_group": target,
            "contamination": contam,
            "mean_expr": pb_df.loc[g, shared_types].mean(),
            "max_expr_ref": float(ref_vals.max()),
            "specificity_ref": float(ref_vals.max() / ref_vals.sum()) if ref_vals.sum() > 0 else np.nan,
        })
    return pd.DataFrame(records).set_index("gene")


def compute_marker_ranking(pb_raw, pb_sp, ref_df, markers):
    """Rank of each marker within its target group by expression."""
    rows = []
    shared_types = pb_raw.columns.intersection(ref_df.columns)
    for gene, target in markers.items():
        if gene not in pb_raw.index or target not in shared_types:
            continue
        raw_ranks = pb_raw[target].rank(ascending=False)
        sp_ranks = pb_sp[target].rank(ascending=False)
        ref_ranks = ref_df[target].rank(ascending=False)
        rows.append({
            "gene": gene,
            "target_group": target,
            "raw_rank": int(raw_ranks.loc[gene]),
            "sparkle_rank": int(sp_ranks.loc[gene]),
            "ref_rank": int(ref_ranks.loc[gene]),
            "rank_improvement": int(raw_ranks.loc[gene]) - int(sp_ranks.loc[gene]),
        })
    return pd.DataFrame(rows)


def plot_scatter(df, x, y, xlabel, ylabel, out_path, title=None, alpha=0.3):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(df[x], df[y], alpha=alpha, s=5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.axhline(0, color="black", linewidth=0.5)
    sns.despine()
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_box(df, x, y, out_path, title=None):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df, x=x, y=y, ax=ax, showfliers=False)
    ax.axhline(0, color="black", linewidth=0.5)
    if title:
        ax.set_title(title)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Gene-level MouseBrain analysis.")
    parser.add_argument("--tag", type=str, required=True)
    parser.add_argument("--input-dir", type=str, default="evaluation/reports/h5ad")
    parser.add_argument("--output-dir", type=str, default="evaluation/reports/mousebrain_eval")
    parser.add_argument("--snrna-ref", type=str,
                        default="evaluation/data/mousebrain/snrna_cell_group_pseudobulk.csv")
    parser.add_argument("--corr-method", type=str, default="pearson",
                        choices=["pearson", "spearman"])
    parser.add_argument(
        "--min-cells", type=int, default=1,
        help="Minimum cells per group for pseudobulk (default 1 to retain rare groups)"
    )
    parser.add_argument("--use-sparkle-corrected-genes", action=argparse.BooleanOptionalAction,
                        default=False)
    parser.add_argument("--use-sctransform", action="store_true",
                        help="Use pysctransform instead of log1p-CPM normalization")
    parser.add_argument("--n-hvgs", type=int, default=None,
                        help="If set, restrict analysis to the top n_hvgs highly "
                             "variable genes computed from RAW data, intersected "
                             "with SPARKLE-corrected genes when applicable. "
                             "Ignored under --use-sctransform.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if args.n_hvgs is not None:
        out_dir = Path(args.output_dir) / f"gene_level_analysis_hvg{args.n_hvgs}"
    elif args.use_sctransform:
        out_dir = Path(args.output_dir) / "gene_level_analysis_sct"
    else:
        out_dir = Path(args.output_dir) / "gene_level_analysis_new"
    out_dir = out_dir / "gene_level_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_path = input_dir / f"{args.tag}_raw.h5ad"
    sp_path = input_dir / f"{args.tag}_SPARKLE.h5ad"

    print("Loading h5ad files ...")
    raw_adata = normalize_adata(sc.read_h5ad(raw_path), use_sctransform=args.use_sctransform)
    sp_adata = normalize_adata(sc.read_h5ad(sp_path), use_sctransform=args.use_sctransform)

    # Optionally restrict to top HVGs from RAW data
    if args.n_hvgs is not None and args.n_hvgs > 0 and not args.use_sctransform:
        sc.pp.highly_variable_genes(raw_adata, n_top_genes=args.n_hvgs, flavor="seurat")
        hvgs = raw_adata.var_names[raw_adata.var["highly_variable"].values].tolist()
        print(f"Selected {len(hvgs)} HVGs from RAW data")
        raw_adata = raw_adata[:, hvgs].copy()
        shared_hvgs = sp_adata.var_names.intersection(hvgs)
        sp_adata = sp_adata[:, shared_hvgs].copy()
        print(f"Using {raw_adata.n_vars} HVGs for analysis")

    ref_df = pd.read_csv(args.snrna_ref, index_col=0)

    # Restrict to SPARKLE-corrected genes if requested
    if args.use_sparkle_corrected_genes and "sparkle_corrected" in sp_adata.var.columns:
        corrected = sp_adata.var_names[sp_adata.var["sparkle_corrected"].values]
        print(f"Using {len(corrected)} SPARKLE-corrected genes")
    else:
        corrected = None

    print("Computing pseudobulk ...")
    pb_raw = compute_pseudobulk(raw_adata, min_cells=args.min_cells)
    pb_sp = compute_pseudobulk(sp_adata, min_cells=args.min_cells)

    shared_genes = pb_raw.index.intersection(pb_sp.index).intersection(ref_df.index)
    if corrected is not None:
        shared_genes = shared_genes.intersection(pd.Index(corrected))
    print(f"Shared genes for analysis: {len(shared_genes)}")

    pb_raw = pb_raw.loc[shared_genes]
    pb_sp = pb_sp.loc[shared_genes]
    ref_df = ref_df.loc[shared_genes]

    print(f"Computing per-gene {args.corr_method} correlations ...")
    raw_corr = compute_per_gene_correlations(pb_raw, ref_df, method=args.corr_method)
    sp_corr = compute_per_gene_correlations(pb_sp, ref_df, method=args.corr_method)

    print("Computing contamination scores ...")
    contam_raw = compute_contamination(pb_raw, ref_df)
    contam_sp = compute_contamination(pb_sp, ref_df)

    improvement = sp_corr - raw_corr
    log2fc = np.log2((pb_sp.mean(axis=1) + 1e-6) / (pb_raw.mean(axis=1) + 1e-6))

    gene_metrics = pd.DataFrame({
        "target_group": contam_raw["target_group"],
        "raw_corr": raw_corr,
        "sparkle_corr": sp_corr,
        "improvement": improvement,
        "mean_expr_raw": pb_raw.mean(axis=1),
        "mean_expr_sp": pb_sp.mean(axis=1),
        "mean_expr_ref": ref_df.mean(axis=1),
        "max_expr_ref": contam_raw["max_expr_ref"],
        "specificity_ref": contam_raw["specificity_ref"],
        "contamination_ref": contam_raw["contamination"],
        "contamination_raw": contam_raw["contamination"],
        "contamination_sp": contam_sp["contamination"],
        "contamination_reduction": contam_raw["contamination"] - contam_sp["contamination"],
        "log2fc_sparkle": log2fc,
    })
    gene_metrics.index.name = "gene"

    per_gene_path = out_dir / f"{args.tag}_{args.corr_method}_gene_metrics_detailed.csv"
    gene_metrics.to_csv(per_gene_path)
    print(f"Saved gene metrics: {per_gene_path}")

    # Per-group summary
    per_group = pd.DataFrame({
        "raw_corr": pb_raw.corrwith(ref_df, axis=0, method=args.corr_method),
        "sparkle_corr": pb_sp.corrwith(ref_df, axis=0, method=args.corr_method),
    })
    per_group["improvement"] = per_group["sparkle_corr"] - per_group["raw_corr"]
    per_group_path = out_dir / f"{args.tag}_{args.corr_method}_per_group_correlation.csv"
    per_group.to_csv(per_group_path)
    print(f"Saved per-group correlations: {per_group_path}")

    # Marker ranking
    marker_ranking = compute_marker_ranking(pb_raw, pb_sp, ref_df, KNOWN_MARKERS)
    marker_ranking_path = out_dir / f"{args.tag}_{args.corr_method}_marker_gene_ranking.csv"
    marker_ranking.to_csv(marker_ranking_path, index=False)
    print(f"Saved marker ranking: {marker_ranking_path}")

    # Known marker improvement
    known_rows = []
    for gene, cell_type in KNOWN_MARKERS.items():
        if gene not in gene_metrics.index:
            continue
        row = gene_metrics.loc[gene].copy()
        row["cell_type"] = cell_type
        known_rows.append(row)
    known_df = pd.DataFrame(known_rows)
    known_path = out_dir / f"{args.tag}_{args.corr_method}_known_marker_improvement.csv"
    known_df.to_csv(known_path)
    print(f"Saved known marker improvement: {known_path}")

    # Layer marker improvement
    layer_rows = []
    for gene, cell_type in LAYER_MARKERS.items():
        if gene not in gene_metrics.index:
            continue
        row = gene_metrics.loc[gene].copy()
        row["cell_type"] = cell_type
        layer_rows.append(row)
    layer_df = pd.DataFrame(layer_rows)
    layer_path = out_dir / f"{args.tag}_{args.corr_method}_layer_marker_improvement.csv"
    layer_df.to_csv(layer_path)
    print(f"Saved layer marker improvement: {layer_path}")

    # Top improved genes per target group
    top_genes = []
    for tg in gene_metrics["target_group"].unique():
        sub = gene_metrics[gene_metrics["target_group"] == tg].sort_values("improvement", ascending=False)
        top_genes.append(sub.head(5).reset_index())
    top_genes_df = pd.concat(top_genes, ignore_index=True)
    top_genes_path = out_dir / f"{args.tag}_{args.corr_method}_top_genes_per_target_group.csv"
    top_genes_df.to_csv(top_genes_path, index=False)
    print(f"Saved top genes per group: {top_genes_path}")

    # Plots
    print("Generating plots ...")
    plot_scatter(
        gene_metrics, "contamination_raw", "improvement",
        "RAW contamination", "SPARKLE improvement",
        out_dir / f"{args.tag}_{args.corr_method}_improvement_vs_contamination.png",
        title=f"Improvement vs contamination ({args.corr_method})",
    )
    plot_scatter(
        gene_metrics, "specificity_ref", "improvement",
        "snRNA specificity", "SPARKLE improvement",
        out_dir / f"{args.tag}_{args.corr_method}_improvement_vs_specificity.png",
        title=f"Improvement vs specificity ({args.corr_method})",
    )

    # Report
    report_path = out_dir / f"{args.tag}_{args.corr_method}_report.md"
    with open(report_path, "w") as f:
        f.write(f"# Gene-level analysis ({args.corr_method})\n\n")
        f.write(f"- **tag**: {args.tag}\n")
        f.write(f"- **correlation method**: {args.corr_method}\n")
        f.write(f"- **genes analyzed**: {len(gene_metrics)}\n")
        f.write(f"- **mean RAW corr**: {gene_metrics['raw_corr'].mean():.4f}\n")
        f.write(f"- **mean SPARKLE corr**: {gene_metrics['sparkle_corr'].mean():.4f}\n")
        f.write(f"- **mean improvement**: {gene_metrics['improvement'].mean():.4f}\n")
        f.write(f"- **mean contamination reduction**: {gene_metrics['contamination_reduction'].mean():.4f}\n\n")

        f.write("## Per-group correlation\n\n")
        f.write(per_group.to_markdown())
        f.write("\n\n")

        f.write("## Known markers\n\n")
        f.write(known_df[["cell_type", "raw_corr", "sparkle_corr", "improvement"]].to_markdown())
        f.write("\n\n")

        f.write("## Marker ranking\n\n")
        f.write(marker_ranking.to_markdown(index=False))
        f.write("\n\n")

        f.write("## Contamination vs improvement\n\n")
        f.write(f"Correlation: {gene_metrics['contamination_reduction'].corr(gene_metrics['improvement']):.3f}\n")
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
