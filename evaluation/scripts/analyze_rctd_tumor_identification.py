#!/usr/bin/env python3
"""
Analyze how well each correction method improves RCTD-based tumor-cell identification.

This script intentionally does NOT treat the h5ad `cell_type` column as ground truth.
Instead it uses:
  1. RCTD internal scores (min_score, singlet_score) by predicted class.
  2. The score margin between Tumor and Intestinal Epithelial calls.
  3. Cancer-marker expression in cells RCTD assigns to Tumor vs Intestinal Epithelial vs others.

Inputs:
  - evaluation/reports/rctd_visiumhd/rctd_doublet_results.csv
  - evaluation/reports/h5ad_visiumhd/{raw,sparkle,spatial_soupx}.h5ad

Outputs under evaluation/reports/rctd_visiumhd/tumor_analysis/
"""

from pathlib import Path

import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse
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


def load_learned_markers(adata, top_n=50):
    """Load scRNA-derived tumor markers and return those present in the h5ad."""
    df = pd.read_csv(MARKER_CSV)
    de = df[df["comparison"] == "Tumor_vs_NonTumor"].sort_values("logFC", ascending=False)
    tumor_markers = de.head(top_n)["gene"].tolist()
    nontumor_markers = de.tail(top_n)["gene"].tolist()
    present_tumor = [g for g in tumor_markers if g in adata.var_names]
    present_nontumor = [g for g in nontumor_markers if g in adata.var_names]
    return present_tumor, present_nontumor


def load_results():
    df = pd.read_csv(RESULTS_CSV)
    df["cell_barcode"] = df["cell_barcode"].astype(str)
    df["min_score"] = pd.to_numeric(df["min_score"], errors="coerce")
    df["singlet_score"] = pd.to_numeric(df["singlet_score"], errors="coerce")
    return df


def load_h5ad(method):
    path = H5AD_DIR / METHOD_FILES[method]
    adata = ad.read_h5ad(path)
    adata.obs_names = adata.obs_names.astype(str)
    adata.obs["cell_barcode"] = adata.obs_names
    return adata


def attach_predictions(adata, rctd_df, method):
    sub = rctd_df[rctd_df["method"] == method].copy()
    sub = sub.drop_duplicates("cell_barcode")
    adata.obs["rctd_first_type"] = pd.Series(
        sub.set_index("cell_barcode")["first_type"], index=adata.obs_names
    ).values
    adata.obs["rctd_second_type"] = pd.Series(
        sub.set_index("cell_barcode")["second_type"], index=adata.obs_names
    ).values
    adata.obs["rctd_spot_class"] = pd.Series(
        sub.set_index("cell_barcode")["spot_class"], index=adata.obs_names
    ).values
    adata.obs["min_score"] = pd.Series(
        sub.set_index("cell_barcode")["min_score"], index=adata.obs_names
    ).values
    adata.obs["singlet_score"] = pd.Series(
        sub.set_index("cell_barcode")["singlet_score"], index=adata.obs_names
    ).values
    return adata


def get_method_adata(method, rctd_df):
    adata = load_h5ad(method)
    adata = attach_predictions(adata, rctd_df, method)
    return adata


def classify_calls(obs):
    obs = obs.copy()
    first = obs["rctd_first_type"].astype(str)
    second = obs["rctd_second_type"].fillna("").astype(str)
    spot = obs["rctd_spot_class"].astype(str)

    obs["pred_tumor_singlet"] = (spot == "singlet") & (first == "Tumor")
    obs["pred_ie_singlet"] = (spot == "singlet") & (first == "Intestinal Epithelial")
    obs["pred_tumor_dominant"] = first == "Tumor"
    obs["pred_ie_dominant"] = first == "Intestinal Epithelial"
    obs["tumor_ie_doublet"] = (
        (spot != "singlet")
        & ((first == "Tumor") | (second == "Tumor"))
        & ((first == "Intestinal Epithelial") | (second == "Intestinal Epithelial"))
    )
    obs["has_rctd"] = obs["rctd_first_type"].notna()
    return obs


def sparse_mean(adata, genes, mask):
    if not np.any(mask):
        return np.full(len(genes), np.nan)
    present_genes = [g for g in genes if g in adata.var_names]
    idx = [adata.var_names.get_loc(g) for g in present_genes]
    sub = adata.X[mask, :][:, idx]
    if sparse.issparse(sub):
        return np.asarray(sub.mean(axis=0)).ravel()
    return sub.mean(axis=0)


def call_summary(obs, method, subset_label="all"):
    obs = classify_calls(obs)
    has = obs["has_rctd"]
    sub = obs.loc[has]
    spot = sub["rctd_spot_class"].astype(str)
    first = sub["rctd_first_type"].astype(str)
    second = sub["rctd_second_type"].fillna("").astype(str)

    n_singlet_tumor = int(((spot == "singlet") & (first == "Tumor")).sum())
    n_singlet_ie = int(((spot == "singlet") & (first == "Intestinal Epithelial")).sum())
    n_singlet_other = int(((spot == "singlet") & (~first.isin(["Tumor", "Intestinal Epithelial"]))).sum())
    n_doublet_tumor_ie = int(sub["tumor_ie_doublet"].sum())
    n_doublet_tumor_other = int(((spot != "singlet") & (first == "Tumor") & ~sub["tumor_ie_doublet"]).sum())
    n_doublet_ie_other = int(((spot != "singlet") & (first == "Intestinal Epithelial") & ~sub["tumor_ie_doublet"]).sum())
    n_reject = int((spot == "reject").sum())

    return pd.DataFrame([{
        "method": method,
        "subset": subset_label,
        "n_total": len(obs),
        "n_with_rctd": int(has.sum()),
        "n_singlet_tumor": n_singlet_tumor,
        "n_singlet_ie": n_singlet_ie,
        "n_singlet_other": n_singlet_other,
        "n_doublet_tumor_ie": n_doublet_tumor_ie,
        "n_doublet_tumor_other": n_doublet_tumor_other,
        "n_doublet_ie_other": n_doublet_ie_other,
        "n_reject": n_reject,
    }])


def score_by_class(obs, method, subset_label="all"):
    """Mean/min/max RCTD scores for Tumor/IE singlet calls and Tumor-vs-IE doublets."""
    obs = classify_calls(obs)
    has = obs["has_rctd"]
    sub = obs.loc[has].copy()
    records = []

    for label, mask in [
        ("tumor_singlet", sub["pred_tumor_singlet"]),
        ("ie_singlet", sub["pred_ie_singlet"]),
        ("other_singlet", (sub["rctd_spot_class"] == "singlet") & ~sub["pred_tumor_singlet"] & ~sub["pred_ie_singlet"]),
        ("tumor_ie_doublet", sub["tumor_ie_doublet"]),
    ]:
        grp = sub[mask]
        if grp.empty:
            continue
        records.append({
            "method": method,
            "subset": subset_label,
            "class_group": label,
            "n": len(grp),
            "mean_min_score": grp["min_score"].mean(),
            "median_min_score": grp["min_score"].median(),
            "std_min_score": grp["min_score"].std(),
            "mean_singlet_score": grp["singlet_score"].mean(),
            "median_singlet_score": grp["singlet_score"].median(),
            "std_singlet_score": grp["singlet_score"].std(),
            "mean_score_gap": (grp["singlet_score"] - grp["min_score"]).mean(),
        })
    return pd.DataFrame(records)


def tumor_ie_margin(obs, method, subset_label="all"):
    """
    For cells where RCTD's top two candidates are Tumor and IE (in either order),
    report the singlet-vs-min score gap. A larger gap means RCTD more strongly
    prefers one of the two classes over the other (and over the pure singlet model).
    """
    obs = classify_calls(obs)
    has = obs["has_rctd"]
    sub = obs.loc[has].copy()
    sub = sub[sub["tumor_ie_doublet"]]
    if sub.empty:
        return pd.DataFrame()
    sub["score_gap"] = sub["singlet_score"] - sub["min_score"]
    return pd.DataFrame([{
        "method": method,
        "subset": subset_label,
        "n_tumor_ie_doublet": len(sub),
        "mean_score_gap": sub["score_gap"].mean(),
        "median_score_gap": sub["score_gap"].median(),
        "std_score_gap": sub["score_gap"].std(),
        "mean_min_score": sub["min_score"].mean(),
        "mean_singlet_score": sub["singlet_score"].mean(),
    }])


def marker_expression_by_class(adata, method):
    """Mean marker expression in RCTD-predicted classes."""
    obs = classify_calls(adata.obs)
    available = [g for g in (TUMOR_MARKERS + STROMAL_MARKERS) if g in adata.var_names]
    records = []
    for label, mask in [
        ("pred_tumor_singlet", obs["pred_tumor_singlet"] & obs["has_rctd"]),
        ("pred_ie_singlet", obs["pred_ie_singlet"] & obs["has_rctd"]),
        ("pred_other_singlet", (obs["rctd_spot_class"] == "singlet") & ~obs["pred_tumor_singlet"] & ~obs["pred_ie_singlet"] & obs["has_rctd"]),
        ("pred_tumor_ie_doublet", obs["tumor_ie_doublet"] & obs["has_rctd"]),
    ]:
        vals = sparse_mean(adata, available, mask)
        records.append({
            "method": method,
            "group": label,
            "n_cells": int(mask.sum()),
            **{g: float(v) for g, v in zip(available, vals)},
        })
    return pd.DataFrame(records)


def marker_ratios(marker_df, methods):
    """Tumor-singlet / IE-singlet marker ratio per method."""
    records = []
    for method in methods:
        t = marker_df[(marker_df["method"] == method) & (marker_df["group"] == "pred_tumor_singlet")]
        ie = marker_df[(marker_df["method"] == method) & (marker_df["group"] == "pred_ie_singlet")]
        if t.empty or ie.empty:
            continue
        gene_cols = [c for c in marker_df.columns if c not in ["method", "group", "n_cells"]]
        for g in gene_cols:
            records.append({
                "method": method,
                "gene": g,
                "tumor_mean": float(t[g].values[0]),
                "ie_mean": float(ie[g].values[0]),
                "tumor_ie_ratio": float(t[g].values[0]) / float(ie[g].values[0]) if float(ie[g].values[0]) > 0 else np.nan,
            })
    return pd.DataFrame(records)


def contamination_signal_table(adata, method):
    """Signal (tumor marker mean) vs contamination (non-tumor marker mean) in pred tumor."""
    obs = classify_calls(adata.obs)
    tumor_mask = obs["pred_tumor_singlet"] & obs["has_rctd"]
    other_mask = (obs["rctd_spot_class"] == "singlet") & ~obs["pred_tumor_dominant"] & obs["has_rctd"]
    tumor_genes = [g for g in TUMOR_MARKERS if g in adata.var_names]
    nontumor_genes = [g for g in STROMAL_MARKERS if g in adata.var_names]

    tumor_mean_in_tumor = float(np.nanmean(sparse_mean(adata, tumor_genes, tumor_mask)))
    tumor_mean_in_other = float(np.nanmean(sparse_mean(adata, tumor_genes, other_mask)))
    nontumor_mean_in_tumor = float(np.nanmean(sparse_mean(adata, nontumor_genes, tumor_mask)))
    nontumor_mean_in_other = float(np.nanmean(sparse_mean(adata, nontumor_genes, other_mask)))

    return pd.DataFrame([{
        "method": method,
        "n_pred_tumor_singlet": int(tumor_mask.sum()),
        "tumor_marker_mean_in_tumor": tumor_mean_in_tumor,
        "tumor_marker_mean_in_other": tumor_mean_in_other,
        "tumor_marker_enrichment": tumor_mean_in_tumor / tumor_mean_in_other if tumor_mean_in_other > 0 else np.nan,
        "nontumor_marker_mean_in_tumor": nontumor_mean_in_tumor,
        "nontumor_marker_mean_in_other": nontumor_mean_in_other,
        "nontumor_marker_depletion": nontumor_mean_in_tumor / nontumor_mean_in_other if nontumor_mean_in_other > 0 else np.nan,
    }])


def make_plots(call_df, score_df, margin_df, marker_df, ratio_df, contam_df, out_dir):
    methods = [m for m in METHOD_FILES.keys()]

    # 1. RCTD score by predicted class
    if not score_df.empty:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        for ax, score_col, title in [(axes[0], "mean_min_score", "min_score"),
                                      (axes[1], "mean_singlet_score", "singlet_score")]:
            piv = score_df.pivot_table(index="class_group", columns="method", values=score_col)
            piv = piv.loc[[c for c in ["tumor_singlet", "ie_singlet", "other_singlet", "tumor_ie_doublet"] if c in piv.index]]
            piv.plot(kind="bar", ax=ax)
            ax.set_title(f"Mean {title} by RCTD class")
            ax.set_ylabel(score_col)
            ax.legend(title="Method")
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
        fig.tight_layout()
        fig.savefig(out_dir / "rctd_score_by_class.png", dpi=200)
        plt.close(fig)

    # 2. Tumor-vs-IE score gap
    if not margin_df.empty:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        sub = margin_df[margin_df["subset"] == "all"]
        ax.bar(sub["method"], sub["mean_score_gap"])
        ax.set_ylabel("Mean score gap (singlet - min)")
        ax.set_title("Tumor-vs-Intestinal-Epithelial doublet score gap")
        fig.tight_layout()
        fig.savefig(out_dir / "tumor_ie_score_gap.png", dpi=200)
        plt.close(fig)

    # 3. Marker expression heatmap by predicted class
    tumor_expr = marker_df[marker_df["group"] == "pred_tumor_singlet"].set_index("method")
    ie_expr = marker_df[marker_df["group"] == "pred_ie_singlet"].set_index("method")
    gene_cols = [c for c in tumor_expr.columns if c not in ["method", "group", "n_cells"]]
    if len(gene_cols) > 0 and not tumor_expr.empty and not ie_expr.empty:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        for ax, df, title in [(axes[0], tumor_expr, "Predicted Tumor singlets"),
                              (axes[1], ie_expr, "Predicted IE singlets")]:
            mat = df[gene_cols].values
            im = ax.imshow(mat, aspect="auto", cmap="YlOrRd")
            ax.set_xticks(np.arange(len(gene_cols)))
            ax.set_xticklabels(gene_cols, rotation=45, ha="right")
            ax.set_yticks(np.arange(len(df)))
            ax.set_yticklabels(df.index)
            ax.set_title(title)
            plt.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(out_dir / "marker_expression_by_class.png", dpi=200)
        plt.close(fig)

    # 4. Tumor/IE marker ratio
    if not ratio_df.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        for method in methods:
            d = ratio_df[ratio_df["method"] == method]
            d = d.set_index("gene")
            d = d.loc[[g for g in TUMOR_MARKERS if g in d.index]]
            ax.plot(d.index, d["tumor_ie_ratio"], marker="o", label=method)
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
        ax.set_ylabel("Tumor-singlet / IE-singlet mean expression")
        ax.set_title("Cancer-marker enrichment in predicted Tumor vs IE")
        ax.legend()
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
        fig.tight_layout()
        fig.savefig(out_dir / "tumor_ie_marker_ratio.png", dpi=200)
        plt.close(fig)

    # 5. Tumor vs non-tumor marker enrichment in predicted tumor singlets
    if not contam_df.empty:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        sub = contam_df[["method", "tumor_marker_enrichment", "nontumor_marker_depletion"]].set_index("method")
        sub.plot(kind="bar", ax=ax)
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
        ax.set_ylabel("Fold change vs predicted non-tumor singlets")
        ax.set_title("Learned marker enrichment/depletion in predicted Tumor singlets")
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=0)
        fig.tight_layout()
        fig.savefig(out_dir / "marker_enrichment_in_tumor_calls.png", dpi=200)
        plt.close(fig)


def main():
    global TUMOR_MARKERS, STROMAL_MARKERS
    print("Loading RCTD results ...")
    rctd_df = load_results()

    # Load scRNA-derived tumor markers, using RAW h5ad gene set as reference
    print("Loading scRNA-derived tumor markers ...")
    raw_adata = load_h5ad("RAW")
    TUMOR_MARKERS, STROMAL_MARKERS = load_learned_markers(raw_adata, top_n=50)
    print(f"  Learned tumor markers present: {len(TUMOR_MARKERS)}")
    print(f"  Learned non-tumor markers present: {len(STROMAL_MARKERS)}")

    call_records = []
    score_records = []
    margin_records = []
    marker_records = []
    contam_records = []

    raw_barcodes = None
    adatas = {}

    for method in METHOD_FILES.keys():
        print(f"Processing {method} ...")
        adata = get_method_adata(method, rctd_df)
        adatas[method] = adata
        if raw_barcodes is None:
            raw_barcodes = set(adata.obs.loc[adata.obs["rctd_first_type"].notna(), "cell_barcode"])

        call_records.append(call_summary(adata.obs, method, "all"))
        score_records.append(score_by_class(adata.obs, method, "all"))
        margin_records.append(tumor_ie_margin(adata.obs, method, "all"))
        marker_records.append(marker_expression_by_class(adata, method))
        contam_records.append(contamination_signal_table(adata, method))

    # Shared-cell analysis against RAW
    print("Computing shared-cell analysis ...")
    for method in METHOD_FILES.keys():
        adata = adatas[method]
        common = adata.obs["cell_barcode"].isin(raw_barcodes)
        adata_shared = adata[common].copy()
        call_records.append(call_summary(adata_shared.obs, method, "shared_with_RAW"))
        score_records.append(score_by_class(adata_shared.obs, method, "shared_with_RAW"))
        margin_records.append(tumor_ie_margin(adata_shared.obs, method, "shared_with_RAW"))
        marker_records.append(marker_expression_by_class(adata_shared, method + "_shared"))
        contam_records.append(contamination_signal_table(adata_shared, method + "_shared"))

    call_df = pd.concat(call_records, ignore_index=True)
    score_df = pd.concat(score_records, ignore_index=True)
    margin_df = pd.concat(margin_records, ignore_index=True)
    marker_df = pd.concat(marker_records, ignore_index=True)
    ratio_df = marker_ratios(marker_df, METHOD_FILES.keys())
    contam_df = pd.concat(contam_records, ignore_index=True)

    # Save tables
    call_df.to_csv(OUT_DIR / "tumor_call_summary.csv", index=False)
    score_df.to_csv(OUT_DIR / "rctd_score_by_class.csv", index=False)
    margin_df.to_csv(OUT_DIR / "tumor_ie_score_gap.csv", index=False)
    marker_df.to_csv(OUT_DIR / "marker_expression_by_class.csv", index=False)
    ratio_df.to_csv(OUT_DIR / "tumor_ie_marker_ratio.csv", index=False)
    contam_df.to_csv(OUT_DIR / "contamination_signal_ratio.csv", index=False)

    print("Saving plots ...")
    make_plots(call_df, score_df, margin_df, marker_df, ratio_df, contam_df, OUT_DIR)

    print("\n=== Tumor / IE call summary (all cells) ===")
    print(call_df[call_df["subset"] == "all"]
          [["method", "n_with_rctd", "n_singlet_tumor", "n_singlet_ie",
            "n_doublet_tumor_ie", "n_reject"]].to_string(index=False))

    print("\n=== RCTD score by predicted class (all cells) ===")
    print(score_df[score_df["subset"] == "all"]
          [["method", "class_group", "n", "mean_min_score", "mean_singlet_score", "mean_score_gap"]]
          .to_string(index=False))

    print("\n=== Tumor-vs-IE doublet score gap (all cells) ===")
    print(margin_df[margin_df["subset"] == "all"]
          [["method", "n_tumor_ie_doublet", "mean_score_gap", "median_score_gap"]]
          .to_string(index=False))

    print("\n=== Tumor/IE marker ratio in predicted singlets (selected markers) ===")
    sel = ratio_df[ratio_df["gene"].isin(["EPCAM", "KRT8", "KRT18", "CEACAM5", "CEACAM6", "CDH1"])]
    print(sel.pivot(index="gene", columns="method", values="tumor_ie_ratio").to_string())

    print("\n=== Marker enrichment in predicted tumor singlets (learned scRNA markers) ===")
    print(contam_df[contam_df["method"].isin(METHOD_FILES.keys())]
          [["method", "n_pred_tumor_singlet", "tumor_marker_enrichment",
            "nontumor_marker_depletion"]].to_string(index=False))

    print(f"\nOutputs written to {OUT_DIR}")


if __name__ == "__main__":
    main()
