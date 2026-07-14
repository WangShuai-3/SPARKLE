#!/usr/bin/env python3
"""Quick inject of RCTD scores into mousebrain h5ad (match on cell_name)."""

import sys
from pathlib import Path
import anndata
import pandas as pd

TAG = "mousebrain_x12500-20000_y2000-10000"
H5AD_DIR = Path("evaluation/reports/h5ad")
FIRST_DIR = Path("evaluation/reports/rctd_mousebrain/first_type")
OUT_DIR = Path("evaluation/reports/h5ad_mousebrain_annotated")
OUT_DIR.mkdir(parents=True, exist_ok=True)

METH_MAP = {
    "raw": "RAW", "sparkle": "SPARKLE", 
    "soupx": "SOUPX", "SoupX": "SOUPX",
    "decontx": "DECONTX", "DecontX": "DECONTX",
}

prefix = f"{TAG}_"
h5ads = sorted(p for p in H5AD_DIR.iterdir() if p.name.startswith(prefix) and p.suffix==".h5ad")
print(f"Found {len(h5ads)} h5ad files")

for hp in h5ads:
    stem = hp.stem[len(prefix):]
    method = METH_MAP.get(stem.lower(), stem)
    csv = FIRST_DIR / f"{method}_first_type.csv"
    if not csv.is_file():
        print(f"  [{method}] SKIP: {csv} not found")
        continue

    print(f"  [{method}] injecting...")
    adata = anndata.read_h5ad(hp)
    rctd = pd.read_csv(csv)

    if "cell_name" in rctd.columns:
        # Match on cell_name (mousebrain) - h5ad obs_names match cell_name
        rctd_idx = rctd.set_index("cell_name")
        common = adata.obs_names.intersection(rctd_idx.index)
        print(f"    common cells: {len(common)} / {adata.n_obs}")

        # Annotation
        adata.obs["annotation"] = rctd_idx.loc[common, "first_type"]

        # Numeric scores
        for rcol, ocol in [("singlet_score","rctd_singlet_score"),
                           ("min_score","rctd_min_score"),
                           ("score_delta","rctd_score_delta")]:
            if rcol in rctd_idx.columns:
                mapped = rctd_idx.loc[common, rcol]
                adata.obs[ocol] = pd.to_numeric(mapped, errors="coerce").astype(float)
    else:
        print(f"    WARNING: no cell_name column, skipping")
        continue

    out = OUT_DIR / hp.name
    adata.write_h5ad(out)
    print(f"    Written: {out}")

print("Done.")
