#!/usr/bin/env python3
"""Gene-level snRNA-concordance analysis for MouseBrain (cell_group & cell_subclass).

For every gene in the analysis gene set, computes the Pearson correlation between
the per-cell-type spatial expression profile and the per-cell-type snRNA
reference profile, for RAW and SPARKLE, and reports
improvement = corr(SPARKLE) - corr(RAW).

The gene set is controlled by exactly one of:
  --sparkle-corrected-only : genes with sparkle_corrected=True in the SPARKLE h5ad
  --genes-file <csv>       : a fixed list from a CSV with a 'gene' column
                             (recommended: reuse the 1211-gene set from the 7/5
                             HVG3000 run to avoid scanpy-version-dependent HVG
                             recomputation)
If neither is given, all shared genes are used.

Ribosomal (Rp*, Rpl*, Rps*) and mitochondrial (Mt-*) genes are excluded from the
top-improvement table.

Usage:
    python evaluation/scripts/gene_level_analysis_mousebrain.py \
        --tag mousebrain_x12500-20000_y2000-10000 \
        --h5ad-dir evaluation/reports/h5ad_mousebrain_annotated_cellgroup \
        --snrna-ref evaluation/data/mousebrain/snrna_cell_group_pseudobulk.csv \
        --out-dir evaluation/reports/mousebrain_eval_cellgroup_v2/gene_level_hvg1211 \
        --genes-file evaluation/reports/mousebrain_eval/gene_level_analysis_hvg3000/gene_level_analysis/mousebrain_x12500-20000_y2000-10000_pearson_gene_metrics_detailed.csv
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import anndata as ad


def normalize_adata(adata):
    import scanpy as sc
    adata = adata.copy()
    if hasattr(adata.X, "toarray"):
        adata.X = adata.toarray()
    adata.X = np.maximum(np.asarray(adata.X, dtype=np.float64), 0.0)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    return adata


def compute_pseudobulk(adata, group_key="annotation", min_cells=3):
    groups = adata.obs[group_key].astype(object).fillna("Unknown").astype(str)
    valid = ~groups.isin(["Unknown", "nan", "None", ""])
    adata = adata[valid].copy()
    groups = groups.loc[adata.obs_names]
    group_names = sorted(groups.unique())
    expr = adata.X
    if hasattr(expr, "toarray"):
        expr = expr.toarray()
    expr = np.asarray(expr)
    pb = {}
    kept = []
    for g in group_names:
        mask = groups == g
        if mask.sum() < min_cells:
            continue
        pb[g] = expr[mask].mean(axis=0)
        kept.append(g)
    return pd.DataFrame(pb, index=adata.var_names, columns=kept)


def per_gene_improvement(pb_raw, pb_sp, ref):
    """Per-gene corr improvement SPARKLE vs RAW (both against snRNA ref)."""
    common = pb_raw.index.intersection(ref.index)
    shared_types = pb_raw.columns.intersection(ref.columns)
    common = common.intersection(pb_sp.index)
    xr = pb_raw.loc[common, shared_types]
    xs = pb_sp.loc[common, shared_types]
    y = ref.loc[common, shared_types]

    rows = []
    for g in common:
        def corr(mat):
            v = mat.loc[g].values
            t = y.loc[g].values
            if np.std(v) < 1e-12 or np.std(t) < 1e-12:
                return np.nan
            return float(np.corrcoef(v, t)[0, 1])
        cr = corr(xr)
        cs = corr(xs)
        rows.append({"gene": g, "raw_corr": cr, "sparkle_corr": cs,
                     "improvement": (cs - cr) if not (np.isnan(cr) or np.isnan(cs)) else np.nan})
    return pd.DataFrame(rows)


def is_housekeeping(gene):
    return bool(re.match(r"^(Rp[lsp]|Rps|Rpl|Mt-|MT-|mt-|RP[LS]|MT)", gene))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--h5ad-dir", required=True)
    parser.add_argument("--snrna-ref", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--min-cells", type=int, default=3)
    parser.add_argument("--sparkle-corrected-only", action="store_true",
                        help="Restrict to genes with sparkle_corrected=True in the SPARKLE h5ad var.")
    parser.add_argument("--hvg-from-corrected", type=int, default=0,
                        help="If >0, compute the top-N HVGs WITHIN the SPARKLE-corrected "
                        "gene subset (HVGs are taken only from corrected genes).")
    parser.add_argument("--genes-file", type=str, default=None,
                        help="Optional CSV with a 'gene' column listing the fixed gene set.")
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ref = pd.read_csv(args.snrna_ref, index_col=0)

    fixed_genes = None
    if args.genes_file:
        fixed_genes = pd.read_csv(args.genes_file)["gene"].tolist()
        print(f"  Using fixed gene set from {args.genes_file}: {len(fixed_genes)} genes")

    pb_raw = None
    corrected_genes = None
    for method, fname in [("RAW", f"{args.tag}_raw.h5ad"),
                          ("SPARKLE", f"{args.tag}_SPARKLE.h5ad")]:
        p = Path(args.h5ad_dir) / fname
        print(f"Reading {p} ...")
        adata = normalize_adata(ad.read_h5ad(p))
        if method == "SPARKLE" and (args.sparkle_corrected_only or args.hvg_from_corrected):
            if "sparkle_corrected" in adata.var.columns:
                corrected_genes = adata.var_names[adata.var["sparkle_corrected"].astype(bool)].tolist()
                print(f"  Found {len(corrected_genes)} SPARKLE-corrected genes")
            else:
                raise ValueError("SPARKLE h5ad has no sparkle_corrected column")
        pb = compute_pseudobulk(adata, min_cells=args.min_cells)
        if method == "RAW":
            pb_raw = pb
            raw_adata = adata
        else:
            pb_sp = pb

    if args.hvg_from_corrected and corrected_genes is not None:
        import scanpy as sc
        sub = raw_adata[:, raw_adata.var_names.isin(corrected_genes)].copy()
        n_hvg = min(args.hvg_from_corrected, sub.n_vars)
        sc.pp.highly_variable_genes(sub, n_top_genes=n_hvg, flavor="seurat")
        hvgs = sub.var_names[sub.var["highly_variable"].values].tolist()
        print(f"  Selected {len(hvgs)} HVGs from within {sub.n_vars} corrected genes")
        corrected_genes = hvgs

    if fixed_genes is not None:
        keep = pb_raw.index.intersection(fixed_genes)
        print(f"  Keeping {len(keep)} fixed genes present in pseudobulk")
        pb_raw = pb_raw.loc[keep]
        pb_sp = pb_sp.loc[pb_raw.index]
    elif corrected_genes is not None:
        keep = pb_raw.index.intersection(corrected_genes)
        print(f"  Keeping {len(keep)} corrected genes present in pseudobulk")
        pb_raw = pb_raw.loc[keep]
        pb_sp = pb_sp.loc[pb_raw.index]

    df = per_gene_improvement(pb_raw, pb_sp, ref)
    df["housekeeping"] = df["gene"].map(is_housekeeping)
    df_clean = df[~df["housekeeping"]].dropna(subset=["improvement"]).copy()

    df_clean.to_csv(out / f"{args.tag}_per_gene_improvement.csv", index=False)

    # Per-cell-type (group) improvement, same pseudobulk and gene set
    common_types = pb_raw.columns.intersection(ref.columns).intersection(pb_sp.columns)
    g_raw = pb_raw.loc[:, common_types].corrwith(ref.loc[:, common_types], axis=0)
    g_sp = pb_sp.loc[:, common_types].corrwith(ref.loc[:, common_types], axis=0)
    pg = pd.DataFrame({"raw_corr": g_raw, "sparkle_corr": g_sp})
    pg["improvement"] = pg["sparkle_corr"] - pg["raw_corr"]
    pg = pg.dropna(subset=["improvement"]).sort_values("improvement", ascending=False)
    pg.to_csv(out / f"{args.tag}_per_celltype_improvement.csv")
    print("\n=== Per-cell-type improvement (top 10) ===")
    print(pg.head(10).round(4).to_string())

    top10 = df_clean.nlargest(10, "improvement")
    print("\n=== Top-10 improved genes (excluding ribosomal/mitochondrial) ===")
    print(top10[["gene", "raw_corr", "sparkle_corr", "improvement"]].round(4).to_string(index=False))

    n_up = (df_clean["improvement"] > 0).sum()
    print(f"\nGenes tested: {len(df_clean)} (excl. {int(df['housekeeping'].sum())} housekeeping), "
          f"improved: {n_up} ({n_up/len(df_clean)*100:.1f}%)")

    top10.to_csv(out / f"{args.tag}_top10_gene_improvement.csv", index=False)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
