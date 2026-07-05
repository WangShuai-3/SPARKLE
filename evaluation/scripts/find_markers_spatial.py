#!/usr/bin/env python3
"""Find markers in spatial (RAW/SPARKLE) h5ad using Wilcoxon rank-sum test.

Uses scanpy.tl.rank_genes_groups with method='wilcoxon'.
Limited to SPARKLE-corrected genes and cell groups with >= min_cells.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc


def normalize_adata(adata, use_sctransform=False):
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


def load_sparkle_genes(path):
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, required=True)
    parser.add_argument("--method", type=str, default="RAW", choices=["RAW", "SPARKLE"])
    parser.add_argument("--input-dir", type=str, default="evaluation/reports/h5ad")
    parser.add_argument("--output-dir", type=str, default="evaluation/reports/mousebrain_eval")
    parser.add_argument("--sparkle-genes", type=str,
                        default="evaluation/reports/mousebrain_eval/sparkle_corrected_genes.txt")
    parser.add_argument("--min-cells", type=int, default=20)
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--use-sctransform", action="store_true",
                        help="Use pysctransform instead of log1p-CPM normalization")
    parser.add_argument("--n-hvgs", type=int, default=None,
                        help="If set, restrict analysis to the top n_hvgs highly "
                             "variable genes computed from RAW data. Ignored under "
                             "--use-sctransform.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir) / "gene_level_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    suffix = "raw" if args.method == "RAW" else "SPARKLE"
    adata = sc.read_h5ad(input_dir / f"{args.tag}_{suffix}.h5ad")
    adata = normalize_adata(adata, use_sctransform=args.use_sctransform)

    # Optionally restrict to top HVGs computed from RAW data
    if args.n_hvgs is not None and args.n_hvgs > 0 and not args.use_sctransform:
        raw_path = input_dir / f"{args.tag}_raw.h5ad"
        raw_adata = normalize_adata(sc.read_h5ad(raw_path), use_sctransform=False)
        sc.pp.highly_variable_genes(raw_adata, n_top_genes=args.n_hvgs, flavor="seurat")
        hvgs = raw_adata.var_names[raw_adata.var["highly_variable"].values].tolist()
        shared_hvgs = adata.var_names.intersection(hvgs)
        adata = adata[:, shared_hvgs].copy()
        print(f"Using {len(shared_hvgs)} HVGs for {args.method}")

    # Filter groups by min_cells
    groups = adata.obs["annotation"].astype(str)
    valid = groups != "Unknown"
    adata = adata[valid].copy()
    group_counts = adata.obs["annotation"].value_counts()
    keep_groups = group_counts[group_counts >= args.min_cells].index.tolist()
    adata = adata[adata.obs["annotation"].isin(keep_groups)].copy()

    # Restrict to sparkle corrected genes
    sparkle_genes = load_sparkle_genes(args.sparkle_genes)
    shared_genes = adata.var_names.intersection(sparkle_genes)
    adata = adata[:, shared_genes].copy()
    print(f"{args.method}: {adata.n_obs} cells, {len(keep_groups)} groups, {len(shared_genes)} genes")

    # Rank genes
    sc.tl.rank_genes_groups(
        adata,
        groupby="annotation",
        method="wilcoxon",
        n_genes=args.top_n,
        use_raw=False,
        pts=True,
    )

    # Extract results
    result = adata.uns["rank_genes_groups"]
    groups = result["names"].dtype.names
    rows = []
    for g in groups:
        for i in range(args.top_n):
            rows.append({
                "cluster": g,
                "gene": result["names"][g][i],
                "logFC": result["logfoldchanges"][g][i],
                "pval": result["pvals"][g][i],
                "pval_adj": result["pvals_adj"][g][i],
                "pct_in": result["pts"].loc[result["names"][g][i], g] if result["names"][g][i] in result["pts"].index else np.nan,
            })

    df = pd.DataFrame(rows)
    if args.use_sctransform:
        suffix = "_sct"
    elif args.n_hvgs is not None:
        suffix = f"_hvg{args.n_hvgs}"
    else:
        suffix = ""
    out_path = out_dir / f"{args.tag}_{args.method}_wilcox_markers_min{args.min_cells}_sparkle_genes{suffix}.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
