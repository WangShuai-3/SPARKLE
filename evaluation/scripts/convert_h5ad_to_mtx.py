#!/usr/bin/env python3
"""Convert Visium HD h5ad files to sparse MTX + obs/var CSV for RCTD in R."""

import sys
import argparse
from pathlib import Path
import anndata as ad
from scipy.io import mmwrite
from scipy.sparse import csr_matrix
import pandas as pd


def convert(h5ad_path, out_dir):
    h5ad_path = Path(h5ad_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = h5ad_path.stem

    print(f"[convert] Reading {h5ad_path} ...")
    adata = ad.read_h5ad(h5ad_path)

    # X is cells x genes in anndata; store as genes x cells for RCTD
    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = csr_matrix(X.T)

    mtx_path = out_dir / f"{prefix}_counts.mtx"
    print(f"[convert] Writing {mtx_path} ({X.shape[0]} genes x {X.shape[1]} cells) ...")
    mmwrite(mtx_path, X)

    obs = adata.obs[["cell_id", "cell_type"]].copy()
    obs["cell_name"] = adata.obs_names
    obs_path = out_dir / f"{prefix}_obs.csv"
    print(f"[convert] Writing {obs_path} ...")
    obs.to_csv(obs_path, index=False)

    var = pd.DataFrame({"gene_name": adata.var_names})
    var_path = out_dir / f"{prefix}_var.csv"
    print(f"[convert] Writing {var_path} ...")
    var.to_csv(var_path, index=False)

    print(f"[convert] Done: {prefix}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("h5ad", help="Input h5ad file")
    parser.add_argument("out_dir", help="Output directory")
    args = parser.parse_args()
    convert(args.h5ad, args.out_dir)
