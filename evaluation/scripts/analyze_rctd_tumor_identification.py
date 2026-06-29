#!/usr/bin/env python3
"""
Analyze how well each correction method improves RCTD-based tumor-cell identification.

Uses:
  - evaluation/reports/rctd_visiumhd/rctd_doublet_results.csv  (RCTD predictions)
  - evaluation/reports/h5ad_visiumhd/{raw,sparkle,spatial_soupx}.h5ad

Produces tables and plots under:
  - evaluation/reports/rctd_visiumhd/tumor_analysis/
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
from scipy import sparse
from sklearn.metrics import roc_auc_score
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

# Colon cancer / epithelial markers expected in the 2000-HVG set
TUMOR_MARKERS = ["EPCAM", "KRT8", "KRT18", "KRT19", "CEACAM5", "CEACAM6",
                 "CDH1", "LGALS4", "FABP1", "TFF3"]
STROMAL_MARKERS = ["VIM", "COL1A1", "ACTA2"]
ALL_MARKERS = TUMOR_MARKERS + STROMAL_MARKERS


def load_results():
    df = pd.read_csv(RESULTS_CSV)
    df["cell_barcode"] = df["cell_barcode"].astype(str)
    return df


def load_h5ad(method):
    path = H5AD_DIR / METHOD_FILES[method]
    adata = ad.read_h5ad(path)
    adata.obs_names = adata.obs_names.astype(str)
    adata.obs["cell_barcode"] = adata.obs_names
    adata.obs["true_cell_type"] = adata.obs["cell_type"].astype(str)
    return adata


def attach_predictions(adata, rctd_df, method):
    sub = rctd_df[rctd_df["method"] == method].copy()
    sub = sub.drop_duplicates("cell_barcode")
    pred = pd.Series(index=adata.obs_names, dtype=object)
    pred.loc[sub["cell_barcode"]] = sub.set_index("cell_barcode")["first_type"]
    adata.obs["rctd_first_type"] = pred.values
    adata.obs["rctd_spot_class"] = pd.Series(
        sub.set_index("cell_barcode")["spot_class"], index=adata.obs_names
    ).values
    for col in ["second_type", "min_score", "singlet_score"]:
        if col in sub.columns:
            adata.obs[col] = pd.Series(
                sub.set_index("cell_barcode")[col], index=adata.obs_names
            ).values
    return adata


def get_method_adata(method, rctd_df):
    adata = load_h5ad(method)
    adata = attach_predictions(adata, rctd_df, method)
    return adata


def classify_calls(obs):
    obs = obs.copy()
    first = obs["rctd_first_type"].astype(str)
    second = obs["second_type"].fillna("").astype(str)
    spot = obs["rctd_spot_class"].astype(str)

    obs["pred_tumor_singlet"] = (spot == "singlet") & (first == "Tumor")
    obs["pred_tumor_dominant"] = first == "Tumor"
    obs["pred_tumor_involved"] = (first == "Tumor") | (second == "Tumor")
    obs["pred_ie_singlet"] = (spot == "singlet") & (first == "Intestinal Epithelial")
    obs["true_tumor"] = obs["true_cell_type"] == "Tumor"
    obs["true_ie"] = obs["true_cell_type"] == "Intestinal Epithelial"
    obs["has_rctd"] = obs["rctd_first_type"].notna()
    return obs


def sparse_mean(adata, genes, mask):
    if not np.any(mask):
        return np.full(len(genes), np.nan)
    idx = [adata.var_names.get_loc(g) for g in genes if g in adata.var_names]
    present_genes = [g for g in genes if g in adata.var_names]
    sub = adata.X[mask, :][:, idx]
    if sparse.issparse(sub):
        return np.asarray(sub.mean(axis=0)).ravel()
    return sub.mean(axis=0)


def marker_expression_table(adata, method):
    """Mean marker expression in predicted tumor / non-tumor / true tumor groups."""
    obs = classify_calls(adata.obs)
    mask_pred_tumor = obs["pred_tumor_singlet"] & obs["has_rctd"]
    mask_pred_nontumor = (
        (obs["rctd_spot_class"] == "singlet")
        & (~obs["pred_tumor_dominant"])
        & obs["has_rctd"]
    )
    available = [g for g in ALL_MARKERS if g in adata.var_names]

    records = []
    for label, mask in [
        ("pred_tumor", mask_pred_tumor),
        ("pred_nontumor", mask_pred_nontumor),
        ("true_tumor", obs["true_tumor"]),
        ("true_nontumor", ~obs["true_tumor"]),
    ]:
        vals = sparse_mean(adata, available, mask)
        records.append({
            "method": method,
            "group": label,
            "n_cells": int(mask.sum()),
            **{g: float(v) for g, v in zip(available, vals)},
        })
    return pd.DataFrame(records)


def marker_auc_table(adata, method):
    """AUC of each marker for predicting true tumor status (per-method expression)."""
    obs = adata.obs.copy()
    obs["true_tumor"] = obs["true_cell_type"] == "Tumor"
    available = [g for g in ALL_MARKERS if g in adata.var_names]
    records = []
    for g in available:
        vals = np.asarray(adata[:, g].X).ravel()
        y = obs["true_tumor"].values
        try:
            auc = roc_auc_score(y, vals)
        except ValueError:
            auc = np.nan
        records.append({"method": method, "gene": g, "auc_true_tumor": auc})
    return pd.DataFrame(records)


def marker_rctd_auc_table(adata, method):
    """AUC of each marker for predicting RCTD tumor-dominant call."""
    obs = classify_calls(adata.obs)
    available = [g for g in ALL_MARKERS if g in adata.var_names]
    has = obs["has_rctd"].values
    y = obs.loc[obs["has_rctd"], "pred_tumor_dominant"].values
    records = []
    for g in available:
        vals = np.asarray(adata[:, g].X).ravel()
        vals_has = vals[has]
        try:
            auc = roc_auc_score(y, vals_has)
        except ValueError:
            auc = np.nan
        records.append({"method": method, "gene": g, "auc_pred_tumor": auc})
    return pd.DataFrame(records)


def tumor_detection_metrics(obs, method, subset_label="all"):
    obs = classify_calls(obs)
    has = obs["has_rctd"]
    true = obs.loc[has, "true_tumor"].values

    metrics = []
    for pred_col, name in [
        ("pred_tumor_singlet", "tumor_singlet"),
        ("pred_tumor_dominant", "tumor_dominant"),
        ("pred_tumor_involved", "tumor_involved"),
    ]:
        pred = obs.loc[has, pred_col].values
        tp = int((true & pred).sum())
        fp = int((~true & pred).sum())
        fn = int((true & ~pred).sum())
        tn = int((~true & ~pred).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else np.nan
        specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        metrics.append({
            "method": method,
            "subset": subset_label,
            "prediction_level": name,
            "n_with_rctd": int(has.sum()),
            "n_pred_tumor": int(pred.sum()),
            "n_true_tumor": int(true.sum()),
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "specificity": specificity,
        })
    return pd.DataFrame(metrics)


def per_class_metrics(obs, method, subset_label="all"):
    """Precision/recall per RCTD-predicted class vs. ground-truth class."""
    obs = classify_calls(obs)
    has = obs["has_rctd"]
    sub = obs.loc[has].copy()
    sub = sub[sub["rctd_spot_class"] == "singlet"]
    if sub.empty:
        return pd.DataFrame()

    pred = sub["rctd_first_type"].astype(str)
    true = sub["true_cell_type"].astype(str)
    classes = sorted(set(pred.unique()) | set(true.unique()))

    records = []
    for cls in classes:
        p_mask = pred == cls
        t_mask = true == cls
        tp = int((p_mask & t_mask).sum())
        fp = int((p_mask & ~t_mask).sum())
        fn = int((~p_mask & t_mask).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else np.nan
        records.append({
            "method": method,
            "subset": subset_label,
            "class": cls,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp, "fp": fp, "fn": fn,
        })
    return pd.DataFrame(records)


def confusion_matrix(obs, method, subset_label="all"):
    obs = classify_calls(obs)
    has = obs["has_rctd"]
    sub = obs.loc[has].copy()
    sub = sub[sub["rctd_spot_class"] == "singlet"]
    tab = pd.crosstab(sub["true_cell_type"], sub["rctd_first_type"])
    tab = tab.reset_index().melt(id_vars="true_cell_type", var_name="rctd_first_type", value_name="n_cells")
    tab["method"] = method
    tab["subset"] = subset_label
    return tab[["method", "subset", "true_cell_type", "rctd_first_type", "n_cells"]]


def confidence_by_correctness(obs, method, subset_label="all"):
    obs = classify_calls(obs)
    has = obs["has_rctd"]
    sub = obs.loc[has].copy()
    sub = sub[sub["pred_tumor_dominant"]].copy()
    if sub.empty:
        return pd.DataFrame()

    sub["correct"] = sub["true_tumor"]
    sub["min_score"] = pd.to_numeric(sub["min_score"], errors="coerce")
    sub["singlet_score"] = pd.to_numeric(sub["singlet_score"], errors="coerce")

    records = []
    for correct, label in [(True, "correct"), (False, "incorrect")]:
        grp = sub[sub["correct"] == correct]
        if grp.empty:
            continue
        records.append({
            "method": method,
            "subset": subset_label,
            "correctness": label,
            "n": len(grp),
            "mean_min_score": grp["min_score"].mean(),
            "mean_singlet_score": grp["singlet_score"].mean(),
        })
    return pd.DataFrame(records)


def confidence_true_tumor_singlet(obs, method, subset_label="all"):
    """Mean RCTD confidence among true tumor cells that RCTD calls as tumor singlet."""
    obs = classify_calls(obs)
    has = obs["has_rctd"]
    sub = obs.loc[has].copy()
    mask = sub["true_tumor"] & (sub["rctd_spot_class"] == "singlet") & (sub["rctd_first_type"] == "Tumor")
    sub = sub[mask].copy()
    if sub.empty:
        return pd.DataFrame()
    sub["min_score"] = pd.to_numeric(sub["min_score"], errors="coerce")
    sub["singlet_score"] = pd.to_numeric(sub["singlet_score"], errors="coerce")
    return pd.DataFrame([{
        "method": method,
        "subset": subset_label,
        "n_true_tumor_singlet": len(sub),
        "mean_min_score": sub["min_score"].mean(),
        "mean_singlet_score": sub["singlet_score"].mean(),
    }])


def contamination_signal_table(adata, method):
    """Signal (epithelial marker mean) vs contamination (stromal marker mean) in pred tumor."""
    obs = classify_calls(adata.obs)
    mask = obs["pred_tumor_singlet"] & obs["has_rctd"]
    epi_genes = [g for g in TUMOR_MARKERS if g in adata.var_names]
    stro_genes = [g for g in STROMAL_MARKERS if g in adata.var_names]
    epi_mean = float(np.nanmean(sparse_mean(adata, epi_genes, mask)))
    stro_mean = float(np.nanmean(sparse_mean(adata, stro_genes, mask)))
    return pd.DataFrame([{
        "method": method,
        "n_pred_tumor_singlet": int(mask.sum()),
        "epithelial_marker_mean": epi_mean,
        "stromal_marker_mean": stro_mean,
        "signal_to_contamination": epi_mean / stro_mean if stro_mean > 0 else np.nan,
    }])


def call_summary(obs, method, subset_label="all"):
    obs = classify_calls(obs)
    has = obs["has_rctd"]
    sub = obs.loc[has]
    spot = sub["rctd_spot_class"].astype(str)
    first = sub["rctd_first_type"].astype(str)
    second = sub["second_type"].fillna("").astype(str)

    n_singlet_tumor = int(((spot == "singlet") & (first == "Tumor")).sum())
    n_singlet_ie = int(((spot == "singlet") & (first == "Intestinal Epithelial")).sum())
    n_singlet_nontumor = int(((spot == "singlet") & (~first.isin(["Tumor", "Intestinal Epithelial"]))).sum())
    n_doublet_tumor_dominant = int(((spot != "singlet") & (first == "Tumor")).sum())
    n_doublet_tumor_secondary = int(((spot != "singlet") & (first != "Tumor") & (second == "Tumor")).sum())
    n_reject = int((spot == "reject").sum())

    return pd.DataFrame([{
        "method": method,
        "subset": subset_label,
        "n_total": len(obs),
        "n_with_rctd": int(has.sum()),
        "n_singlet_tumor": n_singlet_tumor,
        "n_singlet_ie": n_singlet_ie,
        "n_singlet_nontumor": n_singlet_nontumor,
        "n_doublet_tumor_dominant": n_doublet_tumor_dominant,
        "n_doublet_tumor_secondary": n_doublet_tumor_secondary,
        "n_reject": n_reject,
    }])


def make_plots(call_df, detection_df, marker_df, conf_df, perclass_df, auc_df, out_dir):
    # 1. F1 comparison
    f1_df = detection_df[detection_df["prediction_level"] == "tumor_dominant"].copy()
    if not f1_df.empty:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        metrics = ["precision", "recall", "f1", "specificity"]
        x = np.arange(len(f1_df))
        width = 0.2
        for i, m in enumerate(metrics):
            ax.bar(x + i * width, f1_df[m], width, label=m)
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(f1_df["method"])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Score")
        ax.set_title("Tumor detection performance (tumor-dominant RCTD call)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "tumor_detection_f1.png", dpi=200)
        plt.close(fig)

    # 2. Marker expression in predicted tumor vs non-tumor (tumor markers only)
    tumor_expr = marker_df[marker_df["group"] == "pred_tumor"].set_index("method")
    nontumor_expr = marker_df[marker_df["group"] == "pred_nontumor"].set_index("method")
    gene_cols = [c for c in tumor_expr.columns if c not in ["method", "group", "n_cells"]]
    tumor_cols = [c for c in gene_cols if c in TUMOR_MARKERS]
    if len(tumor_cols) > 0 and not tumor_expr.empty and not nontumor_expr.empty:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        for ax, df, title in [(axes[0], tumor_expr, "Predicted tumor singlets"),
                              (axes[1], nontumor_expr, "Predicted non-tumor singlets")]:
            mat = df[tumor_cols].values
            im = ax.imshow(mat, aspect="auto", cmap="YlOrRd")
            ax.set_xticks(np.arange(len(tumor_cols)))
            ax.set_xticklabels(tumor_cols, rotation=45, ha="right")
            ax.set_yticks(np.arange(len(df)))
            ax.set_yticklabels(df.index)
            ax.set_title(title)
            plt.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(out_dir / "marker_expression_heatmap.png", dpi=200)
        plt.close(fig)

    # 3. Confidence by correctness (grouped bar)
    if not conf_df.empty:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        piv = conf_df.pivot_table(index="method", columns="correctness", values="mean_min_score")
        piv = piv[[c for c in ["correct", "incorrect"] if c in piv.columns]]
        piv.plot(kind="bar", ax=ax)
        ax.set_ylabel("Mean RCTD min_score")
        ax.set_title("Confidence of tumor-dominant calls")
        ax.legend(title="Call correctness")
        fig.tight_layout()
        fig.savefig(out_dir / "confidence_by_correctness.png", dpi=200)
        plt.close(fig)

    # 4. Per-class F1 for Tumor and Intestinal Epithelial
    if not perclass_df.empty:
        sub = perclass_df[perclass_df["class"].isin(["Tumor", "Intestinal Epithelial"])]
        if not sub.empty:
            fig, ax = plt.subplots(figsize=(8, 5))
            for cls in ["Tumor", "Intestinal Epithelial"]:
                d = sub[sub["class"] == cls]
                if d.empty:
                    continue
                x = np.arange(len(d))
                offset = 0.2 * (1 if cls == "Tumor" else -1)
                ax.scatter(x + offset, d["f1"], label=cls, s=80)
                ax.plot(x + offset, d["f1"], alpha=0.5)
            ax.set_xticks(np.arange(len(sub["method"].unique())))
            ax.set_xticklabels(sub["method"].unique())
            ax.set_ylim(0, 1)
            ax.set_ylabel("F1 score")
            ax.set_title("Per-class F1: Tumor vs Intestinal Epithelial")
            ax.legend()
            fig.tight_layout()
            fig.savefig(out_dir / "per_class_f1.png", dpi=200)
            plt.close(fig)

    # 5. Marker AUC for predicting true tumor
    if not auc_df.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        for method in auc_df["method"].unique():
            d = auc_df[auc_df["method"] == method].set_index("gene")
            d = d.loc[[g for g in TUMOR_MARKERS if g in d.index]]
            ax.plot(d.index, d["auc_true_tumor"], marker="o", label=method)
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
        ax.set_ylabel("AUC (marker -> true tumor)")
        ax.set_title("Marker discriminative power for true tumor cells")
        ax.legend()
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
        fig.tight_layout()
        fig.savefig(out_dir / "marker_auc_true_tumor.png", dpi=200)
        plt.close(fig)


def main():
    print("Loading RCTD results ...")
    rctd_df = load_results()

    summary_records = []
    marker_records = []
    auc_records = []
    rctd_auc_records = []
    confidence_records = []
    confidence_true_tumor_records = []
    contam_records = []
    perclass_records = []
    confusion_records = []

    raw_barcodes = None
    adatas = {}

    for method in METHOD_FILES.keys():
        print(f"Processing {method} ...")
        adata = get_method_adata(method, rctd_df)
        adatas[method] = adata
        if raw_barcodes is None:
            raw_barcodes = set(adata.obs.loc[adata.obs["rctd_first_type"].notna(), "cell_barcode"])

        summary_records.append(call_summary(adata.obs, method, "all"))
        summary_records.append(tumor_detection_metrics(adata.obs, method, "all"))
        marker_records.append(marker_expression_table(adata, method))
        auc_records.append(marker_auc_table(adata, method))
        rctd_auc_records.append(marker_rctd_auc_table(adata, method))
        confidence_records.append(confidence_by_correctness(adata.obs, method, "all"))
        confidence_true_tumor_records.append(confidence_true_tumor_singlet(adata.obs, method, "all"))
        contam_records.append(contamination_signal_table(adata, method))
        perclass_records.append(per_class_metrics(adata.obs, method, "all"))
        confusion_records.append(confusion_matrix(adata.obs, method, "all"))

    # Shared-cell analysis against RAW
    print("Computing shared-cell analysis ...")
    for method in METHOD_FILES.keys():
        adata = adatas[method]
        common = adata.obs["cell_barcode"].isin(raw_barcodes)
        adata_shared = adata[common].copy()
        summary_records.append(call_summary(adata_shared.obs, method, "shared_with_RAW"))
        summary_records.append(tumor_detection_metrics(adata_shared.obs, method, "shared_with_RAW"))
        marker_records.append(marker_expression_table(adata_shared, method + "_shared"))
        auc_records.append(marker_auc_table(adata_shared, method + "_shared"))
        rctd_auc_records.append(marker_rctd_auc_table(adata_shared, method + "_shared"))
        confidence_records.append(confidence_by_correctness(adata_shared.obs, method, "shared_with_RAW"))
        confidence_true_tumor_records.append(confidence_true_tumor_singlet(adata_shared.obs, method, "shared_with_RAW"))
        contam_records.append(contamination_signal_table(adata_shared, method + "_shared"))
        perclass_records.append(per_class_metrics(adata_shared.obs, method, "shared_with_RAW"))
        confusion_records.append(confusion_matrix(adata_shared.obs, method, "shared_with_RAW"))

    call_summary_df = pd.concat([r for r in summary_records if "n_singlet_tumor" in r.columns],
                                ignore_index=True)
    detection_df = pd.concat([r for r in summary_records if "prediction_level" in r.columns],
                             ignore_index=True)
    marker_df = pd.concat(marker_records, ignore_index=True)
    auc_df = pd.concat(auc_records, ignore_index=True)
    rctd_auc_df = pd.concat(rctd_auc_records, ignore_index=True)
    conf_df = pd.concat(confidence_records, ignore_index=True)
    conf_true_tumor_df = pd.concat(confidence_true_tumor_records, ignore_index=True)
    contam_df = pd.concat(contam_records, ignore_index=True)
    perclass_df = pd.concat(perclass_records, ignore_index=True)
    confusion_df = pd.concat(confusion_records, ignore_index=True)

    # Save tables
    call_summary_df.to_csv(OUT_DIR / "tumor_call_summary.csv", index=False)
    detection_df.to_csv(OUT_DIR / "tumor_detection_metrics.csv", index=False)
    marker_df.to_csv(OUT_DIR / "marker_expression_by_prediction.csv", index=False)
    auc_df.to_csv(OUT_DIR / "marker_auc_true_tumor.csv", index=False)
    rctd_auc_df.to_csv(OUT_DIR / "marker_auc_pred_tumor.csv", index=False)
    conf_df.to_csv(OUT_DIR / "tumor_call_confidence.csv", index=False)
    conf_true_tumor_df.to_csv(OUT_DIR / "confidence_true_tumor_singlet.csv", index=False)
    contam_df.to_csv(OUT_DIR / "contamination_signal_ratio.csv", index=False)
    perclass_df.to_csv(OUT_DIR / "per_class_precision_recall.csv", index=False)
    confusion_df.to_csv(OUT_DIR / "confusion_matrix.csv", index=False)

    print("Saving plots ...")
    make_plots(call_summary_df, detection_df, marker_df, conf_df, perclass_df, auc_df, OUT_DIR)

    print("\n=== Tumor detection F1 (tumor-dominant) ===")
    print(detection_df[detection_df["prediction_level"] == "tumor_dominant"]
          [["method", "subset", "precision", "recall", "f1", "specificity"]]
          .to_string(index=False))

    print("\n=== Per-class F1: Tumor and Intestinal Epithelial ===")
    print(perclass_df[perclass_df["class"].isin(["Tumor", "Intestinal Epithelial"])]
          [["method", "subset", "class", "precision", "recall", "f1"]]
          .to_string(index=False))

    print("\n=== Marker AUC for true tumor (selected markers) ===")
    sel = auc_df[auc_df["gene"].isin(["EPCAM", "KRT8", "KRT18", "CEACAM5", "CEACAM6", "CDH1"])]
    print(sel[sel["method"].isin([m for m in METHOD_FILES.keys()])]
          .pivot(index="gene", columns="method", values="auc_true_tumor")
          .to_string())

    print("\n=== Contamination in predicted tumor singlets ===")
    print(contam_df[contam_df["method"].isin([m for m in METHOD_FILES.keys()])]
          [["method", "n_pred_tumor_singlet", "epithelial_marker_mean",
            "stromal_marker_mean", "signal_to_contamination"]]
          .to_string(index=False))

    print("\n=== Confidence of correctly called tumor singlets ===")
    print(conf_true_tumor_df[conf_true_tumor_df["method"].isin([m for m in METHOD_FILES.keys()])]
          .to_string(index=False))

    print(f"\nOutputs written to {OUT_DIR}")


if __name__ == "__main__":
    main()
