#!/usr/bin/env python3
"""Regenerate cell_group first_type CSVs and inject into mousebrain h5ad.

RCTD Cell_group results use underscore-mangled type names (CA1_N_GLU) because R
make.names replaced the hyphens in the original snRNA reference (CA1-N-GLU).
The snRNA reference pseudobulk uses hyphens, so for correct per-cell-type
matching we restore hyphens when injecting into h5ad.obs['annotation'].

Steps:
  1. Read rctd_Cell_group_doublet_results.csv (5 methods, incl. SpotClean).
  2. Write {Method}_first_type.csv (cell_group level) into
     evaluation/reports/rctd_mousebrain/first_type_cellgroup/.
  3. Re-inject into h5ad_mousebrain_annotated/*.h5ad, converting '_' -> '-',
     writing to h5ad_mousebrain_annotated_cellgroup/.
"""

from pathlib import Path
import anndata
import pandas as pd

TAG = "mousebrain_x12500-20000_y2000-10000"
ROOT = Path(__file__).resolve().parent.parent.parent
RCTD_DIR = ROOT / "evaluation" / "reports" / "rctd_mousebrain"
FIRST_OUT = RCTD_DIR / "first_type_cellgroup"
H5AD_IN = ROOT / "evaluation" / "reports" / "h5ad_mousebrain_annotated"
H5AD_OUT = ROOT / "evaluation" / "reports" / "h5ad_mousebrain_annotated_cellgroup"
FIRST_OUT.mkdir(parents=True, exist_ok=True)
H5AD_OUT.mkdir(parents=True, exist_ok=True)

# ---- 1. Read Cell_group RCTD results and write per-method first_type CSVs ----
rctd = pd.read_csv(RCTD_DIR / "rctd_Cell_group_doublet_results.csv")
print(f"Cell_group RCTD methods: {sorted(rctd.method.unique())}")

# cell_name -> cell_id mapping (all methods share the same cells)
obs = pd.read_csv(RCTD_DIR / "mtx" / f"{TAG}_raw_obs.csv")
name2id = dict(zip(obs["cell_name"], obs["cell_id"]))

for method, sub in rctd.groupby("method"):
    out = pd.DataFrame({
        "cell_name": sub["cell_barcode"],
        "cell_id": sub["cell_barcode"].map(name2id),
        "first_type": sub["first_type"],
        "second_type": sub["second_type"],
        "singlet_score": sub["singlet_score"],
        "min_score": sub["min_score"],
        "score_delta": sub["singlet_score"] - sub["min_score"],
        "spot_class": sub["spot_class"],
    })
    out.to_csv(FIRST_OUT / f"{method}_first_type.csv", index=False)
    print(f"  Wrote {method}_first_type.csv ({len(out)} cells)")

# ---- 2. Inject into h5ad, restoring hyphens ----
print(f"\nInjecting cell_group annotation into h5ad -> {H5AD_OUT}")
for hp in sorted(H5AD_IN.glob(f"{TAG}_*.h5ad")):
    stem = hp.stem[len(TAG) + 1:]  # strip {TAG}_ and .h5ad
    method = {
        "raw": "RAW", "SPARKLE": "SPARKLE", "SoupX": "SoupX",
        "DecontX": "DecontX", "SpotCleanOfficial": "SpotClean",
    }.get(stem, stem)
    csv = FIRST_OUT / f"{method}_first_type.csv"
    if not csv.is_file():
        print(f"  [{method}] SKIP: {csv} not found")
        continue

    print(f"  [{method}] injecting...")
    adata = anndata.read_h5ad(hp)
    rctd_cs = pd.read_csv(csv).set_index("cell_name")
    common = adata.obs_names.intersection(rctd_cs.index)
    print(f"    common cells: {len(common)} / {adata.n_obs}")

    ann = rctd_cs.loc[common, "first_type"].astype(str).str.replace("_", "-", regex=False)
    adata.obs["annotation"] = ann
    for rcol, ocol in [("singlet_score", "rctd_singlet_score"),
                       ("min_score", "rctd_min_score"),
                       ("score_delta", "rctd_score_delta")]:
        adata.obs[ocol] = pd.to_numeric(rctd_cs.loc[common, rcol], errors="coerce").astype(float)

    out_path = H5AD_OUT / hp.name
    adata.write_h5ad(out_path)
    print(f"    Written: {out_path}")

print("Done.")
