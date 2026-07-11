#!/usr/bin/env python
"""Run CellChat (via liana) on ovarian Visium HD corrected outputs.

Compares whether cancer-cell communication improved after ambient RNA
correction across methods (RAW, SPARKLE, SpatialSoupX, SoupX, DecontX).

Hypothesis: ambient RNA contamination inflates spurious cell-cell
communication (autocrine source==target signals and ubiquitous background).
Good correction should reduce autocrine inflation, reduce the fraction of
nonspecific cell-pair connections, and retain genuine tumor-stromal
paracrine signaling.

IMPORTANT
---------
This script MUST be run with the liana virtual-env interpreter, which has
`liana` + `scanpy` installed (they are NOT in the base env):

    /home/shuaiwang/21.STambiant/.venv_liana/bin/python \
        evaluation/scripts/analyze_ovarian_cellchat.py
"""

import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad
import scanpy as sc
import liana as li

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
TAG = "ovarian_x1000-1800_y300-1100"
IN_DIR = "/home/shuaiwang/21.STambiant/evaluation/reports/h5ad_ovarian_annotated"
OUT_DIR = "/home/shuaiwang/21.STambiant/evaluation/reports/ovarian_eval/cellchat"

# method -> filename suffix (RAW uses lowercase _raw)
METHODS = {
    "RAW": "raw",
    "SPARKLE": "SPARKLE",
    "SpatialSoupX": "SpatialSoupX",
    "SoupX": "SoupX",
    "DecontX": "DecontX",
}

TUMOR_TYPES = {
    "Tumor Cells",
    "Proliferative Tumor Cells",
    "VEGFA+ Tumor Cells",
    "MT-High, Jun+-Fos+ Tumor Cells",
    "Inflammatory Tumor Cells",
    "Malignant Cells Lining Cyst",
}

PVAL_THRESH = 0.05


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def clip_negatives(adata):
    """Clip negative values in adata.X to 0 (in place)."""
    if sp.issparse(adata.X):
        adata.X.data = np.clip(adata.X.data, 0, None)
        adata.X.eliminate_zeros()
    else:
        np.clip(adata.X, 0, None, out=adata.X)


def preprocess(adata):
    clip_negatives(adata)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    # liana expects a sparse matrix (uses .A1 on row/col sums)
    if not sp.issparse(adata.X):
        adata.X = sp.csr_matrix(adata.X)


def pick_col(cols, candidates):
    """Return the first candidate present in cols, else None."""
    for c in candidates:
        if c in cols:
            return c
    return None


def category(ct):
    return "tumor" if ct in TUMOR_TYPES else "stromal"


# ----------------------------------------------------------------------------
# Main analysis per method
# ----------------------------------------------------------------------------
def run_method(method, suffix):
    path = os.path.join(IN_DIR, f"{TAG}_{suffix}.h5ad")
    print(f"\n{'='*70}\n[{method}] loading {path}")
    adata = ad.read_h5ad(path)

    # drop Unknown cells
    keep = adata.obs["annotation"].astype(str) != "Unknown"
    adata = adata[keep].copy()
    adata.obs["annotation"] = adata.obs["annotation"].astype(str).astype("category")
    print(f"[{method}] {adata.n_obs} cells x {adata.n_vars} genes after Unknown removal")

    preprocess(adata)

    # ---- run liana CellChat ----
    key = "cellchat_res"
    try:
        li.mt.cellchat(
            adata,
            groupby="annotation",
            resource_name="cellchatdb",
            expr_prop=0.1,
            verbose=True,
            key_added=key,
            use_raw=False,
        )
        res = adata.uns[key]
    except TypeError:
        # key_added not supported in this version
        li.mt.cellchat(
            adata,
            groupby="annotation",
            resource_name="cellchatdb",
            expr_prop=0.1,
            verbose=True,
            use_raw=False,
        )
        res = adata.uns["liana_res"]

    res = res.copy()
    print(f"[{method}] liana result columns: {res.columns.tolist()}")

    # robustly resolve column names
    src_c = pick_col(res.columns, ["source"])
    tgt_c = pick_col(res.columns, ["target"])
    lig_c = pick_col(res.columns, ["ligand_complex", "ligand"])
    rec_c = pick_col(res.columns, ["receptor_complex", "receptor"])
    mag_c = pick_col(res.columns, ["lr_probs", "lr_means", "magnitude", "expr_prod"])
    pval_c = pick_col(res.columns, ["cellchat_pvals", "lr_pvals", "specificity", "pvals"])

    print(f"[{method}] using magnitude='{mag_c}', pvalue='{pval_c}'")

    # normalised working frame
    df = pd.DataFrame({
        "source": res[src_c].astype(str),
        "target": res[tgt_c].astype(str),
        "ligand": res[lig_c].astype(str),
        "receptor": res[rec_c].astype(str),
        "lr_probs": pd.to_numeric(res[mag_c], errors="coerce"),
        "pvals": pd.to_numeric(res[pval_c], errors="coerce"),
    })

    # save full result per method
    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_csv(os.path.join(OUT_DIR, f"{method}_liana_cellchat.csv"), index=False)

    return adata, df


def summarise(method, adata, df):
    sig = df[df["pvals"] <= PVAL_THRESH].copy()

    n_sig = len(sig)
    total_strength = float(sig["lr_probs"].sum())

    # categories
    sig["src_cat"] = sig["source"].map(category)
    sig["tgt_cat"] = sig["target"].map(category)

    directional = {}
    for sc_cat in ("tumor", "stromal"):
        for tg_cat in ("tumor", "stromal"):
            sub = sig[(sig["src_cat"] == sc_cat) & (sig["tgt_cat"] == tg_cat)]
            directional[f"{sc_cat}->{tg_cat}"] = {
                "count": int(len(sub)),
                "strength": float(sub["lr_probs"].sum()),
            }

    # autocrine (self-communication) inflation
    auto = sig[sig["source"] == sig["target"]]
    autocrine_strength = float(auto["lr_probs"].sum())
    autocrine_fraction = (autocrine_strength / total_strength) if total_strength > 0 else 0.0

    # communication specificity: fraction of all source-target cell-type pairs
    # that have >=1 significant interaction
    all_types = sorted(set(adata.obs["annotation"].astype(str)))
    n_pairs_possible = len(all_types) ** 2  # ordered pairs incl. self
    connected_pairs = set(zip(sig["source"], sig["target"]))
    pair_connectivity_fraction = (
        len(connected_pairs) / n_pairs_possible if n_pairs_possible > 0 else 0.0
    )

    row = {
        "method": method,
        "n_sig_interactions": n_sig,
        "total_strength": total_strength,
        "autocrine_strength": autocrine_strength,
        "autocrine_fraction": autocrine_fraction,
        "pair_connectivity_fraction": pair_connectivity_fraction,
        "n_celltypes": len(all_types),
        "n_pairs_possible": n_pairs_possible,
        "n_pairs_connected": len(connected_pairs),
    }
    for k, v in directional.items():
        row[f"{k}_count"] = v["count"]
        row[f"{k}_strength"] = v["strength"]

    return row, directional


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    summary_rows = []
    directional_rows = []

    for method, suffix in METHODS.items():
        adata, df = run_method(method, suffix)
        row, directional = summarise(method, adata, df)
        summary_rows.append(row)
        for pair, vals in directional.items():
            directional_rows.append({
                "method": method,
                "direction": pair,
                "count": vals["count"],
                "strength": vals["strength"],
            })

    summary = pd.DataFrame(summary_rows).set_index("method")
    directional_df = pd.DataFrame(directional_rows)

    summary.to_csv(os.path.join(OUT_DIR, "cellchat_summary.csv"))
    directional_df.to_csv(
        os.path.join(OUT_DIR, "cellchat_tumor_directional.csv"), index=False
    )

    # ------------------------------------------------------------------
    # Comparison table (RAW baseline, % change)
    # ------------------------------------------------------------------
    print("\n\n" + "#" * 78)
    print("# CellChat comparison — RAW baseline vs corrected methods")
    print("#" * 78)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 50)

    cols_show = [
        "n_sig_interactions", "total_strength",
        "autocrine_strength", "autocrine_fraction",
        "pair_connectivity_fraction",
        "tumor->tumor_strength", "tumor->stromal_strength",
        "stromal->tumor_strength", "stromal->stromal_strength",
    ]
    print("\n--- Raw values ---")
    print(summary[cols_show].to_string(float_format=lambda x: f"{x:.4g}"))

    if "RAW" in summary.index:
        base = summary.loc["RAW"]
        print("\n--- % change vs RAW (negative = reduction) ---")
        pct = pd.DataFrame(index=summary.index)
        for c in cols_show:
            b = base[c]
            pct[c] = summary[c].apply(
                lambda v: ((v - b) / b * 100.0) if b != 0 else np.nan
            )
        print(pct.to_string(float_format=lambda x: f"{x:+.1f}%"))

        print("\n--- Interpretation highlights ---")
        for m in summary.index:
            if m == "RAW":
                continue
            auto_chg = pct.loc[m, "autocrine_strength"]
            frac_chg = pct.loc[m, "pair_connectivity_fraction"]
            t2s = pct.loc[m, "tumor->stromal_strength"]
            s2t = pct.loc[m, "stromal->tumor_strength"]
            print(
                f"[{m}] autocrine inflation {auto_chg:+.1f}% | "
                f"pair-connectivity {frac_chg:+.1f}% | "
                f"tumor->stromal {t2s:+.1f}% | stromal->tumor {s2t:+.1f}%"
            )

    print(f"\nOutputs written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
