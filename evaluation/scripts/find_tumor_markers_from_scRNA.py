#!/usr/bin/env python3
"""
Find tumor markers from the scRNA reference using Level1 annotations.

Outputs marker lists to:
  - evaluation/reports/rctd_visiumhd/tumor_markers_from_scRNA.csv

Compares:
  1. Tumor vs. all non-tumor Level1 cell types
  2. Tumor vs. Intestinal Epithelial specifically
"""

from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRNA_H5 = PROJECT_ROOT / "evaluation" / "data" / "visiumhd" / "scrna" / "HumanColonCancer_Flex_Multiplex_count_filtered_feature_bc_matrix.h5"
SCRNA_META = PROJECT_ROOT / "evaluation" / "data" / "visiumhd" / "scrna" / "SingleCell_MetaData.csv.gz"
OUT_DIR = PROJECT_ROOT / "evaluation" / "reports" / "rctd_visiumhd"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_MARKERS = 200


def main():
    print("Loading scRNA reference ...")
    adata = sc.read_10x_h5(str(SCRNA_H5))
    adata.var_names_make_unique()

    print("Loading metadata ...")
    meta = pd.read_csv(SCRNA_META, compression="gzip")
    meta = meta[meta["QCFilter"] == "Keep"]
    meta = meta.set_index("Barcode")

    # Subset to cells with metadata
    common = adata.obs_names.intersection(meta.index)
    adata = adata[common].copy()
    adata.obs["Level1"] = meta.loc[adata.obs_names, "Level1"].values

    print(f"After QC: {adata.n_obs} cells x {adata.n_vars} genes")
    print("Level1 counts:")
    print(adata.obs["Level1"].value_counts())

    # Basic normalization for DE (log1p on raw counts is sufficient for ranking)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # --- Comparison 1: Tumor vs. all non-tumor ---
    print("\nRunning DE: Tumor vs. non-tumor ...")
    adata.obs["group"] = adata.obs["Level1"].apply(lambda x: "Tumor" if x == "Tumor" else "NonTumor")
    # balance groups for speed
    tumor_cells = adata.obs_names[adata.obs["group"] == "Tumor"]
    nontumor_cells = adata.obs_names[adata.obs["group"] == "NonTumor"]
    np.random.seed(42)
    if len(nontumor_cells) > len(tumor_cells):
        nontumor_cells = np.random.choice(nontumor_cells, size=len(tumor_cells), replace=False)
    sub1 = adata[np.concatenate([tumor_cells, nontumor_cells])].copy()
    sc.tl.rank_genes_groups(sub1, groupby="group", groups=["Tumor"], reference="NonTumor",
                            method="wilcoxon", n_genes=N_MARKERS, pts=True)
    de1 = sc.get.rank_genes_groups_df(sub1, group="Tumor")
    de1["comparison"] = "Tumor_vs_NonTumor"

    # --- Comparison 2: Tumor vs. Intestinal Epithelial ---
    print("Running DE: Tumor vs. Intestinal Epithelial ...")
    mask = adata.obs["Level1"].isin(["Tumor", "Intestinal Epithelial"])
    sub2 = adata[mask].copy()
    tumor_cells2 = sub2.obs_names[sub2.obs["Level1"] == "Tumor"]
    ie_cells = sub2.obs_names[sub2.obs["Level1"] == "Intestinal Epithelial"]
    np.random.seed(42)
    if len(ie_cells) < len(tumor_cells2):
        tumor_cells2 = np.random.choice(tumor_cells2, size=len(ie_cells), replace=False)
    sub2 = sub2[np.concatenate([tumor_cells2, ie_cells])].copy()
    sc.tl.rank_genes_groups(sub2, groupby="Level1", groups=["Tumor"], reference="Intestinal Epithelial",
                            method="wilcoxon", n_genes=N_MARKERS, pts=True)
    de2 = sc.get.rank_genes_groups_df(sub2, group="Tumor")
    de2["comparison"] = "Tumor_vs_IntestinalEpithelial"

    # Combine and save
    de_all = pd.concat([de1, de2], ignore_index=True)
    print("DE columns:", de_all.columns.tolist())
    de_all = de_all.rename(columns={"names": "gene", "scores": "wilcoxon_score",
                                     "logfoldchanges": "logFC", "pvals": "pval",
                                     "pvals_adj": "pval_adj"})
    keep_cols = ["comparison", "gene", "logFC", "pval", "pval_adj"]
    keep_cols += [c for c in de_all.columns if c.startswith("pts")]
    de_all = de_all[keep_cols]
    de_all.to_csv(OUT_DIR / "tumor_markers_from_scRNA.csv", index=False)

    print("\n=== Top Tumor vs Non-Tumor markers ===")
    print(de1.head(20)[["names", "logfoldchanges", "pvals_adj"]].to_string(index=False))

    print("\n=== Top Tumor vs Intestinal Epithelial markers ===")
    print(de2.head(20)[["names", "logfoldchanges", "pvals_adj"]].to_string(index=False))

    print(f"\nSaved to {OUT_DIR / 'tumor_markers_from_scRNA.csv'}")


if __name__ == "__main__":
    main()
