#!/usr/bin/env python
"""Compare spatial cell-cell communication (CellChat via liana) against a
single-cell (scRNA) reference to quantify whether ambient RNA correction
removes confident FALSE-POSITIVE interactions.

Run with: /home/shuaiwang/21.STambiant/.venv_liana/bin/python
"""
import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
import h5py
import scanpy as sc
import liana as li

BASE = "/home/shuaiwang/21.STambiant/evaluation"
DATA = os.path.join(BASE, "data/ovarian")
OUT = os.path.join(BASE, "reports/ovarian_eval/cellchat")
H5 = os.path.join(DATA, "17k_Ovarian_Cancer_scFFPE_count_filtered_feature_bc_matrix.h5")
ANNOT = os.path.join(DATA, "FLEX_Ovarian_Barcode_Cluster_Annotation.csv")

os.makedirs(OUT, exist_ok=True)

PVAL_THR = 0.05
KEYCOLS = ["source", "target", "ligand", "receptor"]
TUMOR_TYPES = {
    "Tumor Cells", "Proliferative Tumor Cells", "VEGFA+ Tumor Cells",
    "MT-High, Jun+-Fos+ Tumor Cells", "Inflammatory Tumor Cells",
    "Malignant Cells Lining Cyst",
}

SPATIAL_METHODS = [
    "RAW", "SPARKLE", "SpatialSoupX", "SoupX", "DecontX", "SpotClean"
]


# --------------------------------------------------------------------------
# Step 1: build scRNA AnnData and run CellChat
# --------------------------------------------------------------------------
def build_scrna_adata():
    print("[Step 1] Loading 10x h5 (CSC genes x cells) ...")
    with h5py.File(H5, "r") as f:
        g = f["matrix"]
        data = g["data"][:]
        indices = g["indices"][:]
        indptr = g["indptr"][:]
        shape = g["shape"][:]  # (n_genes, n_cells)
        barcodes = [b.decode() for b in g["barcodes"][:]]
        names = [n.decode() for n in g["features"]["name"][:]]
    # CSC matrix genes x cells
    mat = sp.csc_matrix((data, indices, indptr), shape=(int(shape[0]), int(shape[1])))
    # transpose to cells x genes
    X = mat.T.tocsr().astype(np.float32)
    import anndata as ad
    adata = ad.AnnData(X=X)
    adata.obs_names = barcodes
    adata.var_names = names
    adata.var_names_make_unique()
    print(f"  scRNA raw AnnData: {adata.shape} (cells x genes)")

    print("[Step 1] Loading annotation ...")
    ann = pd.read_csv(ANNOT)
    ann = ann.dropna(subset=["Cell Annotation"])
    ann["Cell Annotation"] = ann["Cell Annotation"].str.replace("/", "-", regex=False)
    ann = ann.set_index("Barcode")["Cell Annotation"]

    keep = adata.obs_names.intersection(ann.index)
    adata = adata[keep].copy()
    adata.obs["annotation"] = ann.reindex(adata.obs_names).values
    adata = adata[~adata.obs["annotation"].isna()].copy()
    print(f"  scRNA after annotation filter: {adata.shape}")
    print("  annotation counts:\n", adata.obs["annotation"].value_counts())

    print("[Step 1] Preprocessing (normalize_total + log1p) ...")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    if not sp.isspmatrix_csr(adata.X):
        adata.X = sp.csr_matrix(adata.X)
    adata.X = adata.X.astype(np.float32)
    return adata


def run_scrna_cellchat(adata):
    print("[Step 1] Running liana CellChat on scRNA ...")
    li.mt.cellchat(adata, groupby="annotation", resource_name="cellchatdb",
                   expr_prop=0.1, use_raw=False, verbose=True)
    res = adata.uns["liana_res"].copy()
    return res


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def normalize_liana_df(df):
    """Return df with standard cols: source,target,ligand,receptor,lr_probs,pvals."""
    df = df.copy()
    rename = {}
    if "ligand_complex" in df.columns and "ligand" not in df.columns:
        rename["ligand_complex"] = "ligand"
    if "receptor_complex" in df.columns and "receptor" not in df.columns:
        rename["receptor_complex"] = "receptor"
    if "cellchat_pvals" in df.columns and "pvals" not in df.columns:
        rename["cellchat_pvals"] = "pvals"
    df = df.rename(columns=rename)
    return df[["source", "target", "ligand", "receptor", "lr_probs", "pvals"]]


def keyset(df, sig_only=False):
    d = df
    if sig_only:
        d = d[d["pvals"] <= PVAL_THR]
    return set(map(tuple, d[KEYCOLS].values))


def is_tumor_key(k):
    return k[0] in TUMOR_TYPES or k[1] in TUMOR_TYPES


# --------------------------------------------------------------------------
# Steps 3-6 confusion + FP removal
# --------------------------------------------------------------------------
def compute_confusion(spatial, scrna, scope="all"):
    """scope: 'all' or 'tumor'. Returns per-method dict rows and RAW FP/TP key sets."""
    def flt(keys):
        if scope == "tumor":
            return {k for k in keys if is_tumor_key(k)}
        return keys

    # testable keys = all keys present in scRNA result (passed expr_prop filter)
    scrna_all = flt(keyset(scrna, sig_only=False))
    scrna_sig = flt(keyset(scrna, sig_only=True))

    rows = []
    raw_fp = raw_tp = None
    method_sig = {}
    for m in SPATIAL_METHODS:
        df = spatial[m]
        sig = flt(keyset(df, sig_only=True))
        method_sig[m] = sig
        testable = sig & scrna_all                 # significant AND testable in scRNA
        not_testable = sig - scrna_all             # significant but LR not tested in scRNA
        tp = testable & scrna_sig
        fp = testable - scrna_sig                   # testable, sig in spatial, not sig in scRNA
        precision = len(tp) / (len(tp) + len(fp)) if (len(tp) + len(fp)) else np.nan
        recall = len(tp) / len(scrna_sig) if len(scrna_sig) else np.nan
        rows.append({
            "scope": scope, "method": m,
            "n_sig_total": len(sig),
            "n_sig_testable": len(testable),
            "n_not_testable_in_scrna": len(not_testable),
            "TP": len(tp), "FP": len(fp),
            "precision": precision, "recall": recall,
            "scrna_sig_testable": len(scrna_sig),
        })
        if m == "RAW":
            raw_fp = fp
            raw_tp = tp

    # Step 5: FP removal / TP retention anchored on RAW
    for row in rows:
        m = row["method"]
        sig = method_sig[m]
        fp_removed = len(raw_fp - sig)
        tp_retained = len(raw_tp & sig)
        row["RAW_FP"] = len(raw_fp)
        row["RAW_TP"] = len(raw_tp)
        row["fp_removed"] = fp_removed
        row["fp_removal_rate"] = fp_removed / len(raw_fp) if len(raw_fp) else np.nan
        row["tp_retained"] = tp_retained
        row["tp_retention_rate"] = tp_retained / len(raw_tp) if len(raw_tp) else np.nan
    return rows, raw_fp, raw_tp, scrna_sig


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main():
    scrna_path = os.path.join(OUT, "scRNA_liana_cellchat.csv")
    if os.path.exists(scrna_path):
        print(f"[Step 1] Found cached scRNA result: {scrna_path} (reusing)")
        scrna_raw = pd.read_csv(scrna_path)
    else:
        adata = build_scrna_adata()
        scrna_raw = run_scrna_cellchat(adata)
        scrna_raw.to_csv(scrna_path, index=False)
        print(f"  saved {scrna_path}")

    scrna = normalize_liana_df(scrna_raw)
    print(f"\nscRNA interactions: {len(scrna)} testable; "
          f"{(scrna['pvals'] <= PVAL_THR).sum()} significant")

    # Step 2: load spatial
    spatial = {}
    for m in SPATIAL_METHODS:
        p = os.path.join(OUT, f"{m}_liana_cellchat.csv")
        spatial[m] = normalize_liana_df(pd.read_csv(p))
        print(f"  loaded {m}: {len(spatial[m])} rows, "
              f"{(spatial[m]['pvals'] <= PVAL_THR).sum()} significant")

    # Steps 3-6
    rows_all, raw_fp_all, raw_tp_all, _ = compute_confusion(spatial, scrna, "all")
    rows_tum, raw_fp_tum, raw_tp_tum, _ = compute_confusion(spatial, scrna, "tumor")
    conf = pd.DataFrame(rows_all + rows_tum)
    conf_path = os.path.join(OUT, "cellchat_vs_scrna_confusion.csv")
    conf.to_csv(conf_path, index=False)

    # print tables
    show_cols = ["method", "n_sig_total", "n_sig_testable", "n_not_testable_in_scrna",
                 "TP", "FP", "precision", "recall", "fp_removed", "fp_removal_rate",
                 "tp_retained", "tp_retention_rate"]
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.float_format", lambda x: f"{x:.3f}")

    print("\n" + "=" * 90)
    print("CONFUSION vs scRNA TRUTH  --  ALL CELLS")
    print("=" * 90)
    print(pd.DataFrame(rows_all)[show_cols].to_string(index=False))

    print("\n" + "=" * 90)
    print("CONFUSION vs scRNA TRUTH  --  TUMOR-INVOLVING ONLY")
    print("=" * 90)
    print(pd.DataFrame(rows_tum)[show_cols].to_string(index=False))

    # Step 7: SPARKLE examples ---------------------------------------------
    scrna_key_pval = scrna.set_index(KEYCOLS)["pvals"].to_dict()
    raw_df = spatial["RAW"].set_index(KEYCOLS)
    sparkle_sig = keyset(spatial["SPARKLE"], sig_only=True)

    # RAW-FP (tumor) that SPARKLE removed, sorted by RAW lr_probs
    removed = sorted(raw_fp_tum - sparkle_sig)
    rec = []
    for k in removed:
        row = raw_df.loc[k]
        # in case of duplicate index rows take first
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        rec.append({
            "source": k[0], "target": k[1], "ligand": k[2], "receptor": k[3],
            "RAW_lr_prob": float(row["lr_probs"]), "RAW_pval": float(row["pvals"]),
            "scRNA_pval": scrna_key_pval.get(k, np.nan),
        })
    removed_df = pd.DataFrame(rec).sort_values("RAW_lr_prob", ascending=False)
    removed_df.to_csv(os.path.join(OUT, "sparkle_removed_false_positives.csv"), index=False)

    # RAW-TP that SPARKLE wrongly removed (use all-cells anchor)
    lost = sorted(raw_tp_all - sparkle_sig)
    lrec = []
    for k in lost:
        row = raw_df.loc[k]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        lrec.append({
            "source": k[0], "target": k[1], "ligand": k[2], "receptor": k[3],
            "RAW_lr_prob": float(row["lr_probs"]), "RAW_pval": float(row["pvals"]),
            "scRNA_pval": scrna_key_pval.get(k, np.nan),
        })
    lost_df = pd.DataFrame(lrec).sort_values("RAW_lr_prob", ascending=False) if lrec else pd.DataFrame(
        columns=["source", "target", "ligand", "receptor", "RAW_lr_prob", "RAW_pval", "scRNA_pval"])
    lost_df.to_csv(os.path.join(OUT, "sparkle_lost_true_positives.csv"), index=False)

    print("\n" + "=" * 90)
    print("SPARKLE FOCUS")
    print("=" * 90)
    sp_all = [r for r in rows_all if r["method"] == "SPARKLE"][0]
    sp_tum = [r for r in rows_tum if r["method"] == "SPARKLE"][0]
    print(f"[ALL]   RAW FP={sp_all['RAW_FP']}  ->  SPARKLE removed {sp_all['fp_removed']} "
          f"({sp_all['fp_removal_rate']:.1%})  | RAW TP={sp_all['RAW_TP']} retained "
          f"{sp_all['tp_retained']} ({sp_all['tp_retention_rate']:.1%})")
    print(f"[TUMOR] RAW FP={sp_tum['RAW_FP']}  ->  SPARKLE removed {sp_tum['fp_removed']} "
          f"({sp_tum['fp_removal_rate']:.1%})  | RAW TP={sp_tum['RAW_TP']} retained "
          f"{sp_tum['tp_retained']} ({sp_tum['tp_retention_rate']:.1%})")

    print("\nTop 15 tumor-involving RAW false positives that SPARKLE removed "
          "(spurious signals cleaned up):")
    print(removed_df.head(15).to_string(index=False))

    print(f"\nRAW true positives (genuine per scRNA) that SPARKLE wrongly removed "
          f"(cost side): {len(lost_df)}")
    if len(lost_df):
        print(lost_df.to_string(index=False))
    else:
        print("  (none)")

    print("\nSaved outputs:")
    for fn in ["scRNA_liana_cellchat.csv", "cellchat_vs_scrna_confusion.csv",
               "sparkle_removed_false_positives.csv", "sparkle_lost_true_positives.csv"]:
        print("  ", os.path.join(OUT, fn))


if __name__ == "__main__":
    main()
