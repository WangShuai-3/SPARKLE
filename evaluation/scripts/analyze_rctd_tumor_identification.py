#!/usr/bin/env python3
"""
Compare learned scRNA tumor markers between RCTD-predicted tumor and non-tumor cells.

For each correction method (RAW, SPARKLE, SpatialSoupX) and each learned marker,
compute the log fold-change and Wilcoxon p-value between:
  - RCTD-predicted tumor singlets
  - RCTD-predicted non-tumor singlets

These are compared against the scRNA reference logFC/p-value to judge whether a
method improves marker-based tumor/non-tumor discrimination.

Inputs:
  - evaluation/reports/rctd_visiumhd/rctd_doublet_results.csv
  - evaluation/reports/rctd_visiumhd/tumor_markers_from_scRNA.csv
  - evaluation/reports/h5ad_visiumhd/{raw,sparkle,spatial_soupx}.h5ad

Outputs under evaluation/reports/rctd_visiumhd/tumor_analysis/
"""

from pathlib import Path

import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse
from scipy.stats import ranksums
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "evaluation" / "reports"
H5AD_DIR = REPORTS_DIR / "h5ad_visiumhd"
RCTD_DIR = REPORTS_DIR / "rctd_visiumhd"
OUT_DIR = RCTD_DIR / "tumor_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

METHOD_FILES = {
    "RAW": "raw.h5ad",
    "SPARKLE": "sparkle.h5ad",
    "SpatialSoupX": "spatial_soupx.h5ad",
}

RESULTS_CSV = RCTD_DIR / "rctd_doublet_results.csv"
MARKER_CSV = RCTD_DIR / "tumor_markers_from_scRNA.csv"

N_MARKERS = 100  # use top-N scRNA-derived tumor markers


def load_results():
    df = pd.read_csv(RESULTS_CSV)
    df["cell_barcode"] = df["cell_barcode"].astype(str)
    return df


def load_h5ad(method):
    path = H5AD_DIR / METHOD_FILES[method]
    adata = ad.read_h5ad(path)
    adata.obs_names = adata.obs_names.astype(str)
    return adata


def attach_predictions(adata, rctd_df, method):
    sub = rctd_df[rctd_df["method"] == method].copy()
    sub = sub.drop_duplicates("cell_barcode")
    adata.obs["rctd_first_type"] = pd.Series(
        sub.set_index("cell_barcode")["first_type"], index=adata.obs_names
    ).values
    adata.obs["rctd_spot_class"] = pd.Series(
        sub.set_index("cell_barcode")["spot_class"], index=adata.obs_names
    ).values
    return adata


def load_scRNA_markers(adata):
    """Load scRNA markers and filter to those present in the h5ad."""
    df = pd.read_csv(MARKER_CSV)
    de = df[df["comparison"] == "Tumor_vs_NonTumor"].sort_values("logFC", ascending=False)
    top = de.head(N_MARKERS)
    present = top[top["gene"].isin(adata.var_names)].copy()
    present = present.rename(columns={"logFC": "scRNA_logFC", "pval_adj": "scRNA_pval_adj"})
    return present[["gene", "scRNA_logFC", "scRNA_pval_adj"]]


def safe_log_mean(values):
    """log2(mean(non-zero) + 1) style pseudocount mean."""
    return np.log2(np.mean(values) + 1.0)


def per_marker_de(adata, marker_df):
    """Compute logFC and Wilcoxon p-value per marker: pred_tumor vs pred_other singlet."""
    obs = adata.obs.copy()
    is_tumor = (obs["rctd_spot_class"] == "singlet") & (obs["rctd_first_type"] == "Tumor")
    is_other = (obs["rctd_spot_class"] == "singlet") & (obs["rctd_first_type"] != "Tumor") & obs["rctd_first_type"].notna()

    records = []
    for gene in marker_df["gene"]:
        if gene not in adata.var_names:
            continue
        vals = np.asarray(adata[:, gene].X).ravel()
        t_vals = vals[is_tumor.values]
        o_vals = vals[is_other.values]
        if len(t_vals) == 0 or len(o_vals) == 0:
            continue
        logfc = safe_log_mean(t_vals) - safe_log_mean(o_vals)
        try:
            stat, pval = ranksums(t_vals, o_vals)
        except ValueError:
            pval = np.nan
        records.append({
            "gene": gene,
            "n_tumor": int(is_tumor.sum()),
            "n_other": int(is_other.sum()),
            "visium_logFC": logfc,
            "visium_pval": pval,
        })
    return pd.DataFrame(records)


def make_plots(combined_df, out_dir):
    # 1. Visium logFC vs scRNA logFC
    fig, ax = plt.subplots(figsize=(8, 6))
    for method in combined_df["method"].unique():
        d = combined_df[combined_df["method"] == method]
        ax.scatter(d["scRNA_logFC"], d["visium_logFC"], label=method, alpha=0.7, s=40)
    lim_min = min(combined_df["scRNA_logFC"].min(), combined_df["visium_logFC"].min())
    lim_max = max(combined_df["scRNA_logFC"].max(), combined_df["visium_logFC"].max())
    ax.plot([lim_min, lim_max], [lim_min, lim_max], "k--", linewidth=1)
    ax.set_xlabel("scRNA logFC (Tumor / NonTumor)")
    ax.set_ylabel("Visium HD logFC (pred_tumor / pred_other)")
    ax.set_title("Marker logFC: scRNA reference vs Visium HD methods")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "marker_logfc_scRNA_vs_Visium.png", dpi=200)
    plt.close(fig)

    # 2. Top marker logFC comparison
    genes = combined_df["gene"].unique()
    n_genes = min(20, len(genes))
    # pick top by scRNA logFC
    top_genes = combined_df.drop_duplicates("gene").nlargest(n_genes, "scRNA_logFC")["gene"].tolist()
    sub = combined_df[combined_df["gene"].isin(top_genes)].copy()
    piv = sub.pivot_table(index="gene", columns="method", values="visium_logFC")
    piv = piv.loc[[g for g in top_genes if g in piv.index]]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(piv))
    width = 0.25
    for i, method in enumerate(piv.columns):
        ax.bar(x + i * width, piv[method], width, label=method)
    ax.set_xticks(x + width)
    ax.set_xticklabels(piv.index, rotation=45, ha="right")
    ax.set_ylabel("Visium HD logFC (pred_tumor / pred_other)")
    ax.set_title("Top scRNA tumor markers: Visium HD logFC by method")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "top_marker_logfc_by_method.png", dpi=200)
    plt.close(fig)


def main():
    print("Loading RCTD results ...")
    rctd_df = load_results()

    print("Loading scRNA-derived tumor markers ...")
    raw_adata = load_h5ad("RAW")
    marker_df = load_scRNA_markers(raw_adata)
    print(f"  {len(marker_df)} learned markers present in Visium HD")

    all_records = []
    for method in METHOD_FILES.keys():
        print(f"Processing {method} ...")
        adata = load_h5ad(method)
        adata = attach_predictions(adata, rctd_df, method)
        de = per_marker_de(adata, marker_df)
        de["method"] = method
        all_records.append(de)

    visium_df = pd.concat(all_records, ignore_index=True)
    combined = visium_df.merge(marker_df, on="gene", how="left")

    combined.to_csv(OUT_DIR / "marker_tumor_vs_nontumor_de.csv", index=False)

    print("Saving plots ...")
    make_plots(combined, OUT_DIR)

    # Summary: how many markers have positive logFC in Visium, and correlation with scRNA
    print("\n=== Marker discrimination summary ===")
    summary = []
    for method in METHOD_FILES.keys():
        d = combined[combined["method"] == method]
        n_pos = int((d["visium_logFC"] > 0).sum())
        n_sig = int((d["visium_pval"] < 0.05).sum())
        corr = d["scRNA_logFC"].corr(d["visium_logFC"])
        mean_abs_logfc = d["visium_logFC"].abs().mean()
        summary.append({
            "method": method,
            "n_markers": len(d),
            "n_positive_logFC": n_pos,
            "n_significant_p05": n_sig,
            "corr_with_scRNA_logFC": corr,
            "mean_abs_visium_logFC": mean_abs_logfc,
        })
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(OUT_DIR / "marker_discrimination_summary.csv", index=False)
    print(summary_df.to_string(index=False))

    print("\n=== Top 10 markers by scRNA logFC ===")
    top = combined.drop_duplicates("gene").nlargest(10, "scRNA_logFC")["gene"].tolist()
    print(combined[combined["gene"].isin(top)]
          [["gene", "method", "scRNA_logFC", "visium_logFC", "visium_pval"]]
          .sort_values(["gene", "method"])
          .to_string(index=False))

    print(f"\nOutputs written to {OUT_DIR}")


if __name__ == "__main__":
    main()
