#!/usr/bin/env python3
"""Fig 5d 替代指标(原始计数版,与 axolotl SST 分析同口径): 源保留率 × ectopic 去除率。

与 analyze_ovarian_marker_retention_auroc.py 的唯一区别: 不做 CP10K 归一化,
直接用每个细胞的原始/修正计数求组均值 —— 消除重归一化伪影(保留率不再 >1)。

输出(Report_V2/Figure5_ovarian/marker_analysis/):
  marker_retention_removal_rawcounts_per_gene.csv / _summary.csv
  fig5d_retention_removal_panels_rawcounts.{png,pdf}

用法: cd evaluation && python scripts/analyze_ovarian_marker_retention_rawcounts.py
"""
import os

import anndata as ad
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL_DIR = os.path.dirname(HERE)
H5AD_DIR = os.path.join(EVAL_DIR, "reports", "h5ad_ovarian_annotated")
REF_CSV = os.path.join(EVAL_DIR, "data", "ovarian", "scrna_celltype_pseudobulk.csv")
OUT = os.path.join(EVAL_DIR, "Report_V2", "Figure5_ovarian", "marker_analysis")
TAG = "ovarian_x1000-1800_y300-1100"

METHODS = ["RAW", "SoupX", "DecontX", "SpotClean", "SPARKLE"]
SUFFIX = {"RAW": "raw", "SpotClean": "SpotCleanOfficial"}
METHOD_COLORS = {"RAW": "#80858c", "SoupX": "#bc8e36", "DecontX": "#457f78",
                 "SpotClean": "#8f6aa8", "SPARKLE": "#1c456e"}
EPITHELIAL_TUMOR_MARKERS = [
    "MUC16", "WFDC2", "PAX8", "MSLN", "MUC1", "FOLR1", "EPCAM",
    "KRT7", "KRT19", "WT1", "CD24", "CLDN3", "CLDN4", "CLDN6",
    "LSR", "TACSTD2", "EPHA2",
]
# 已排除血管类 PECAM1/VWF/CDH5(RAW 下已完美定位, AUROC≈1.0)
# 已排除 FAP(sparkle_r2=0.006,未通过 SPARKLE 门控,未被修正)
STROMAL_FIBROBLAST_MARKERS = ["COL1A1", "COL1A2", "DCN", "ACTA2", "VIM"]
MARKERS = EPITHELIAL_TUMOR_MARKERS + STROMAL_FIBROBLAST_MARKERS
SOURCE_FRAC = 0.25


def main():
    ref = pd.read_csv(REF_CSV, index_col=0)
    types = list(ref.columns)
    source_mask = {}
    for g in MARKERS:
        v = ref.loc[g, types].to_numpy()
        source_mask[g] = v >= SOURCE_FRAC * v.max()

    pb, ncells = {}, {}
    for m in METHODS:
        path = os.path.join(H5AD_DIR, f"{TAG}_{SUFFIX.get(m, m)}.h5ad")
        print(f"加载 {m} ...", flush=True)
        a = ad.read_h5ad(path)
        expr = a.X.toarray() if hasattr(a.X, "toarray") else np.asarray(a.X)
        expr = np.maximum(np.asarray(expr, dtype=np.float64), 0.0)  # 原始计数
        labels = a.obs["annotation"].astype(str).to_numpy()
        df, nn = {}, {}
        for t in types:
            mask = labels == t
            df[t] = expr[mask].mean(axis=0)
            nn[t] = int(mask.sum())
        pb[m] = pd.DataFrame(df, index=a.var_names)
        ncells[m] = nn
        del a, expr

    n_vec = np.array([ncells["RAW"][t] for t in types], dtype=float)
    rows = []
    for g in MARKERS:
        smask = source_mask[g]
        src_types = [t for t, s in zip(types, smask) if s]
        if not src_types or smask.all():
            continue
        panel = "tumor" if g in EPITHELIAL_TUMOR_MARKERS else "stromal"

        def pooled(m, use_src):
            idx = np.where(smask if use_src else ~smask)[0]
            v = pb[m].loc[g, [types[i] for i in idx]].to_numpy()
            w = n_vec[idx]
            return float((v * w).sum() / w.sum())

        raw_src, raw_ns = pooled("RAW", True), pooled("RAW", False)
        for m in METHODS:
            rows.append({
                "gene": g, "panel": panel, "method": m,
                "source_types": ";".join(src_types),
                "src_mean_rawcounts": pooled(m, True),
                "nonsrc_mean_rawcounts": pooled(m, False),
                "retention": pooled(m, True) / raw_src if raw_src > 0 else np.nan,
                "removal": 1 - pooled(m, False) / raw_ns if raw_ns > 0 else np.nan,
            })
    rr = pd.DataFrame(rows)
    rr.to_csv(os.path.join(OUT, "marker_retention_removal_rawcounts_per_gene.csv"),
              index=False)

    rows = []
    for panel in ("tumor", "stromal"):
        for m in METHODS:
            sub = rr[(rr.panel == panel) & (rr.method == m)]
            rows.append({"panel": panel, "method": m,
                         "mean_retention": sub["retention"].mean(),
                         "mean_removal": sub["removal"].mean(),
                         "median_retention": sub["retention"].median(),
                         "median_removal": sub["removal"].median()})
    summ = pd.DataFrame(rows)
    summ.to_csv(os.path.join(OUT, "marker_retention_removal_rawcounts_summary.csv"),
                index=False)
    for panel in ("tumor", "stromal"):
        print(f"\n=== {panel} markers(原始计数)===")
        print(summ[summ.panel == panel].drop(columns="panel")
              .set_index("method").reindex(METHODS).round(3).to_string())

    # 分面板散点
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2), sharex=True, sharey=True)
    for ax, panel, title in [(axes[0], "tumor", "Tumor markers (n=17)"),
                             (axes[1], "stromal", "Stromal markers (n=5)")]:
        for m in METHODS:
            sub = rr[(rr.method == m) & (rr.panel == panel)]
            ax.scatter(sub["removal"], sub["retention"], s=18,
                       color=METHOD_COLORS[m], alpha=0.3, zorder=2)
            ax.scatter(sub["removal"].mean(), sub["retention"].mean(), s=120,
                       color=METHOD_COLORS[m], edgecolor="black", lw=0.9,
                       zorder=3)
            ax.annotate(m, (sub["removal"].mean(), sub["retention"].mean()),
                        fontsize=7.5, xytext=(6, 6), textcoords="offset points",
                        color=METHOD_COLORS[m])
        ax.axhline(1.0, ls="--", lw=0.8, color="0.5")
        ax.axvline(0.0, lw=0.6, color="0.7")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Ectopic removal rate (non-source cells)", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlim(-0.15, 1.05)
        ax.set_ylim(-0.05, 1.3)
    axes[0].set_ylabel("Source signal retention", fontsize=9)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        path = os.path.join(OUT,
                            f"fig5d_retention_removal_panels_rawcounts.{ext}")
        fig.savefig(path, dpi=300)
        print("Saved:", path)


if __name__ == "__main__":
    main()
