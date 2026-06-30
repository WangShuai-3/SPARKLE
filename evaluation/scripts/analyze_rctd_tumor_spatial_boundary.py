#!/usr/bin/env python3
"""
Compute local entropy of RCTD class labels around each cell.

Local entropy measures how mixed RCTD predictions are within a fixed spatial
neighborhood. Lower entropy means the tumor (and other) spatial domains are
called more consistently by RCTD.

Inputs:
  - evaluation/reports/rctd_visiumhd/rctd_doublet_results.csv
  - evaluation/reports/h5ad_visiumhd/{raw,sparkle,spatial_soupx}.h5ad
  - evaluation/data/visiumhd/Visium_HD_6p5mm_Human_Colon_Cancer_feature_slice.h5

Outputs under evaluation/reports/rctd_visiumhd/spatial_boundary/
"""

from pathlib import Path

import numpy as np
import pandas as pd
import anndata as ad
import h5py
from sklearn.neighbors import KDTree
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "evaluation" / "reports"
DATA_DIR = PROJECT_ROOT / "evaluation" / "data" / "visiumhd"
H5AD_DIR = REPORTS_DIR / "h5ad_visiumhd"
RCTD_DIR = REPORTS_DIR / "rctd_visiumhd"
OUT_DIR = RCTD_DIR / "spatial_boundary"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_CSV = RCTD_DIR / "rctd_doublet_results.csv"


METHOD_NAME_MAP = {
    "raw": "RAW",
    "sparkle": "SPARKLE",
    "spatial_soupx": "SpatialSoupX",
}


def method_name_from_stem(stem: str) -> str:
    if stem in METHOD_NAME_MAP:
        return METHOD_NAME_MAP[stem]
    parts = stem.split("_")
    return "".join(part.capitalize() for part in parts)


def discover_methods():
    methods = {}
    for path in sorted(H5AD_DIR.glob("*.h5ad")):
        methods[method_name_from_stem(path.stem)] = path.name
    return methods
FEATURE_SLICE_H5 = DATA_DIR / "Visium_HD_6p5mm_Human_Colon_Cancer_feature_slice.h5"

RADII = [50, 100, 200]


def load_results():
    df = pd.read_csv(RESULTS_CSV)
    df["cell_barcode"] = df["cell_barcode"].astype(str)
    return df


def compute_cell_coordinates(feature_slice_h5, cell_ids_needed):
    """Return DataFrame with mean (x,y) per cell_id, matching R script scaling."""
    print("  Loading segmentation mask ...")
    with h5py.File(feature_slice_h5, "r") as h5:
        seg = h5["segmentations/cell_segmentation_mask"]
        rows = seg["row"][:]
        cols = seg["col"][:]
        data = seg["data"][:]

    dt = pd.DataFrame({"cell_id": data, "x": cols * 2, "y": rows * 2})
    needed = set(cell_ids_needed)
    dt = dt[dt["cell_id"].isin(needed)]
    coords = dt.groupby("cell_id")[["x", "y"]].mean().reset_index()
    return coords


def load_h5ad(method, method_files):
    path = H5AD_DIR / method_files[method]
    adata = ad.read_h5ad(path)
    adata.obs_names = adata.obs_names.astype(str)
    return adata


def attach_predictions(adata, rctd_df, method):
    sub = rctd_df[rctd_df["method"] == method].copy()
    sub = sub.drop_duplicates("cell_barcode")
    adata.obs["rctd_first_type"] = pd.Series(
        sub.set_index("cell_barcode")["first_type"], index=adata.obs_names
    ).values
    adata.obs["rctd_spot_class"] = pd.Series(
        sub.set_index("cell_barcode")["spot_class"], index=adata.obs_names
    ).values
    return adata


def local_entropy(labels, neighbors):
    """Shannon entropy of RCTD first_type labels within each neighborhood."""
    out = np.full(len(labels), np.nan)
    for i, nb in enumerate(neighbors):
        nb = nb[nb != i]
        if len(nb) == 0:
            continue
        vals, counts = np.unique(labels[nb], return_counts=True)
        p = counts / counts.sum()
        out[i] = -np.sum(p * np.log2(p + 1e-12))
    return out


def analyze_method(method, rctd_df, coords_all, raw_barcodes=None, method_files=None):
    print(f"\nProcessing {method} ...")
    adata = load_h5ad(method, method_files)
    adata = attach_predictions(adata, rctd_df, method)

    obs = adata.obs.copy()
    # preserve cell_barcode (Cell_N) index across merge
    obs = obs.reset_index().merge(coords_all, on="cell_id", how="left").set_index("index")
    valid = obs[["x", "y"]].notna().all(axis=1)
    obs = obs[valid].copy()

    if raw_barcodes is not None:
        keep = obs.index.isin(raw_barcodes)
        obs = obs[keep].copy()
        subset_label = "shared_with_RAW"
    else:
        subset_label = "all"

    # restrict to cells with RCTD predictions for meaningful entropy
    obs = obs[obs["rctd_first_type"].notna()].copy()
    coords = obs[["x", "y"]].values
    n = len(obs)
    print(f"  {subset_label}: {n} cells with coordinates and RCTD predictions")
    if n == 0:
        return None

    first_type = obs["rctd_first_type"].astype(str).values
    tree = KDTree(coords, leaf_size=40)

    records = []
    for radius in RADII:
        print(f"  radius={radius} ...")
        neighbors = tree.query_radius(coords, r=radius)
        neighbors = [nb[nb != i] for i, nb in enumerate(neighbors)]
        ent = local_entropy(first_type, neighbors)
        records.append({
            "method": method,
            "subset": subset_label,
            "radius": radius,
            "n_cells": n,
            "mean_local_entropy": np.nanmean(ent),
            "median_local_entropy": np.nanmedian(ent),
        })

    return pd.DataFrame(records)


def make_plots(entropy_df, method_files, out_dir):
    if entropy_df.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for method in method_files.keys():
        d = entropy_df[(entropy_df["method"] == method) & (entropy_df["subset"] == "all")]
        ax.plot(d["radius"], d["mean_local_entropy"], marker="o", label=method)
    ax.set_xlabel("Radius")
    ax.set_ylabel("Mean local entropy")
    ax.set_title("RCTD class-label disorder within local neighborhoods")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "local_entropy.png", dpi=200)
    plt.close(fig)


def main():
    print("Loading RCTD results ...")
    rctd_df = load_results()

    method_files = discover_methods()
    print(f"Discovered methods: {list(method_files.keys())}")

    print("Loading h5ad cell ids for coordinate computation ...")
    all_cell_ids = set()
    for method in method_files.keys():
        adata = load_h5ad(method, method_files)
        all_cell_ids.update(adata.obs["cell_id"].astype(int).tolist())
    print(f"  {len(all_cell_ids)} unique cell ids to look up")

    print("Computing cell coordinates from segmentation mask ...")
    coords_all = compute_cell_coordinates(FEATURE_SLICE_H5, all_cell_ids)
    print(f"  coordinates for {len(coords_all)} cells")

    raw_adata = load_h5ad("RAW", method_files)
    raw_adata = attach_predictions(raw_adata, rctd_df, "RAW")
    raw_barcodes = set(raw_adata.obs.loc[raw_adata.obs["rctd_first_type"].notna()].index)

    records = []
    for method in method_files.keys():
        res = analyze_method(method, rctd_df, coords_all, raw_barcodes=None, method_files=method_files)
        if res is not None:
            records.append(res)
        res_shared = analyze_method(method, rctd_df, coords_all, raw_barcodes=raw_barcodes, method_files=method_files)
        if res_shared is not None:
            records.append(res_shared)

    entropy_df = pd.concat(records, ignore_index=True)
    entropy_df.to_csv(OUT_DIR / "local_entropy_metrics.csv", index=False)

    print("Saving plot ...")
    make_plots(entropy_df, method_files, OUT_DIR)

    print("\n=== Local entropy (lower = purer neighborhoods) ===")
    print(entropy_df[entropy_df["subset"] == "all"]
          [["method", "radius", "n_cells", "mean_local_entropy"]]
          .to_string(index=False))

    print(f"\nOutputs written to {OUT_DIR}")


if __name__ == "__main__":
    main()
