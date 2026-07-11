#!/usr/bin/env python
"""Prove that SPARKLE's reduction of spatial CellChat communication is biologically
justified, using the COLLAGEN (COL1A2/COL1A1 -> SDC4) tumor->tumor interaction as
a worked example.

Logic
-----
Collagen genes (COL1A1/COL1A2) are canonical fibroblast products. Tumour cells do
NOT synthesise abundant collagen (confirmed in dissociated scRNA-seq). In the
high-resolution Visium HD RAW data, however, collagen mRNA from neighbouring
fibroblasts diffuses (ambient RNA) onto tumour cells, so CellChat infers a
spurious COLLAGEN tumour->tumour autocrine loop. SPARKLE removes this ambient
signal, driving tumour collagen expression back toward the scRNA truth while
leaving the true fibroblast source almost untouched -> the CellChat probability
drop is therefore correct, not a loss of real biology.

Run with the base python (anndata/scanpy not required):
    python evaluation/scripts/analyze_ovarian_cellchat_proof.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
TAG = "ovarian_x1000-1800_y300-1100"
H5AD = ROOT / "reports" / "h5ad_ovarian_annotated"
CC = ROOT / "reports" / "ovarian_eval" / "cellchat_spatial"
OUT = ROOT / "reports" / "ovarian_eval" / "cellchat_spatial"
OUT.mkdir(parents=True, exist_ok=True)


def norm_log1p_cpm(X):
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.maximum(np.asarray(X, dtype=np.float64), 0)
    t = X.sum(1, keepdims=True)
    t[t == 0] = 1
    return np.log1p(X / t * 1e4)


def main():
    raw = ad.read_h5ad(H5AD / f"{TAG}_raw.h5ad")
    sp = ad.read_h5ad(H5AD / f"{TAG}_SPARKLE.h5ad")
    ref = pd.read_csv(ROOT / "data" / "ovarian" / "scrna_celltype_pseudobulk.csv", index_col=0)

    ann = raw.obs["annotation"].values
    genes = list(raw.var_names)
    Xr, Xs = norm_log1p_cpm(raw.X), norm_log1p_cpm(sp.X)

    def stat(X, ct, g):
        gi = genes.index(g)
        m = ann == ct
        v = X[m, gi]
        return float(v.mean()), float((v > 0).mean() * 100), int(m.sum())

    cell_types = [
        "VEGFA+ Tumor Cells",
        "MT-High, Jun+-Fos+ Tumor Cells",
        "Tumor Associated Fibroblasts",
        "Stromal Associated Fibroblasts",
    ]
    example_genes = ["COL1A2", "COL1A1", "SDC4"]

    rows = []
    for g in example_genes:
        for ct in cell_types:
            sc = float(ref.loc[g, ct]) if (g in ref.index and ct in ref.columns) else np.nan
            rm, rp, n = stat(Xr, ct, g)
            sm, spp, _ = stat(Xs, ct, g)
            rows.append(dict(gene=g, cell_type=ct, n_cells=n,
                             scRNA_truth=sc,
                             RAW_mean=rm, RAW_pct=rp,
                             SPARKLE_mean=sm, SPARKLE_pct=spp))
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "proof_collagen_expression.csv", index=False)
    print(tab.to_string(index=False))

    # CellChat probabilities for the specific removed interactions
    rawnet = pd.read_csv(CC / "RAW_cellchat_spatial.csv")
    spnet = pd.read_csv(CC / "SPARKLE_cellchat_spatial.csv")
    key = ["source", "target", "ligand", "receptor"]
    for df in (rawnet, spnet):
        df["k"] = df[key].astype(str).agg("|".join, axis=1)
    ex = rawnet[(rawnet["pathway_name"] == "COLLAGEN")
                & (rawnet["source"].str.contains("Tumor"))
                & (rawnet["target"].str.contains("Tumor"))].copy()
    ex = ex.merge(spnet[["k", "prob"]], on="k", how="left", suffixes=("_RAW", "_SPARKLE"))
    ex["prob_SPARKLE"] = ex["prob_SPARKLE"].fillna(0)
    ex = ex[["source", "target", "ligand", "receptor", "prob_RAW", "prob_SPARKLE"]]
    ex.to_csv(OUT / "proof_collagen_cellchat_prob.csv", index=False)
    print("\n=== CellChat COLLAGEN tumor->tumor prob (RAW vs SPARKLE) ===")
    print(ex.to_string(index=False))

    # ---- Figure: COL1A2 expression truth vs RAW vs SPARKLE ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, g in zip(axes, ["COL1A2", "COL1A1"]):
        sub = tab[tab["gene"] == g]
        cts = ["VEGFA+ Tumor Cells", "MT-High, Jun+-Fos+ Tumor Cells", "Tumor Associated Fibroblasts"]
        short = ["VEGFA+\nTumor", "MT-High\nTumor", "TAF\n(fibroblast)"]
        x = np.arange(len(cts))
        w = 0.26
        sc = [sub[sub.cell_type == c]["scRNA_truth"].values[0] for c in cts]
        rw = [sub[sub.cell_type == c]["RAW_mean"].values[0] for c in cts]
        spv = [sub[sub.cell_type == c]["SPARKLE_mean"].values[0] for c in cts]
        ax.bar(x - w, sc, w, label="scRNA (truth)", color="#2ca02c")
        ax.bar(x, rw, w, label="RAW spatial", color="#d62728")
        ax.bar(x + w, spv, w, label="SPARKLE", color="#1f77b4")
        ax.set_xticks(x)
        ax.set_xticklabels(short)
        ax.set_ylabel("mean log1p-CPM")
        ax.set_title(f"{g}: collagen is a fibroblast gene\n(tumour signal in RAW = ambient)")
        ax.legend()
    fig.suptitle("Why SPARKLE's COLLAGEN->SDC4 tumour communication drop is justified", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "proof_collagen_expression.png", dpi=150)
    print(f"\nSaved figure: {OUT / 'proof_collagen_expression.png'}")


if __name__ == "__main__":
    main()
