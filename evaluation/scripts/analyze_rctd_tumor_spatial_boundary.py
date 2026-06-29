#!/usr/bin/env python3
"""
Assess whether correction methods make the tumor spatial boundary clearer.

Uses:
  - evaluation/reports/rctd_visiumhd/rctd_doublet_results.csv
  - evaluation/reports/h5ad_visiumhd/{raw,sparkle,spatial_soupx}.h5ad
  - evaluation/data/visiumhd/Visium_HD_6p5mm_Human_Colon_Cancer_feature_slice.h5

Computes spatial metrics per method:
  - global spatial autocorrelation (Moran's I) of the tumor indicator
  - local tumor fraction and local RCTD class entropy within a radius
  - distance-to-boundary profiles of RCTD scores and marker expression

Outputs under evaluation/reports/rctd_visiumhd/spatial_boundary/
"""

from pathlib import Path

import numpy as np
import pandas as pd
import anndata as ad
import h5py
from scipy import sparse
from scipy.spatial import cKDTree
from sklearn.neighbors import KDTree as SklearnKDTree
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

METHOD_FILES = {
    "RAW": "raw.h5ad",
    "SPARKLE": "sparkle.h5ad",
    "SpatialSoupX": "spatial_soupx.h5ad",
}
RESULTS_CSV = RCTD_DIR / "rctd_doublet_results.csv"
FEATURE_SLICE_H5 = DATA_DIR / "Visium_HD_6p5mm_Human_Colon_Cancer_feature_slice.h5"

# Radius in the same coordinate units as feature_slice (cols/rows * 2).
# A small radius captures nearest neighbours; a larger radius captures broader domains.
RADII = [50, 100, 200]

MARKER_CSV = RCTD_DIR / "tumor_markers_from_scRNA.csv"
# Placeholder; populated in main() from scRNA-derived markers
MARKERS = []


def load_learned_markers(adata, top_n=30):
    """Load scRNA-derived tumor markers and return those present in the h5ad."""
    df = pd.read_csv(MARKER_CSV)
    de = df[df["comparison"] == "Tumor_vs_NonTumor"].sort_values("logFC", ascending=False)
    tumor_markers = de.head(top_n)["gene"].tolist()
    nontumor_markers = de.tail(top_n)["gene"].tolist()
    present_tumor = [g for g in tumor_markers if g in adata.var_names]
    present_nontumor = [g for g in nontumor_markers if g in adata.var_names]
    return present_tumor + present_nontumor


def load_results():
    df = pd.read_csv(RESULTS_CSV)
    df["cell_barcode"] = df["cell_barcode"].astype(str)
    df["min_score"] = pd.to_numeric(df["min_score"], errors="coerce")
    df["singlet_score"] = pd.to_numeric(df["singlet_score"], errors="coerce")
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
    # Restrict to needed cell_ids to save memory
    needed = set(cell_ids_needed)
    dt = dt[dt["cell_id"].isin(needed)]
    coords = dt.groupby("cell_id")[["x", "y"]].mean().reset_index()
    return coords


def load_h5ad(method):
    path = H5AD_DIR / METHOD_FILES[method]
    adata = ad.read_h5ad(path)
    adata.obs_names = adata.obs_names.astype(str)
    adata.obs["cell_barcode"] = adata.obs_names
    return adata


def attach_predictions(adata, rctd_df, method):
    sub = rctd_df[rctd_df["method"] == method].copy()
    sub = sub.drop_duplicates("cell_barcode")
    adata.obs["rctd_first_type"] = pd.Series(
        sub.set_index("cell_barcode")["first_type"], index=adata.obs_names
    ).values
    adata.obs["rctd_second_type"] = pd.Series(
        sub.set_index("cell_barcode")["second_type"], index=adata.obs_names
    ).values
    adata.obs["rctd_spot_class"] = pd.Series(
        sub.set_index("cell_barcode")["spot_class"], index=adata.obs_names
    ).values
    adata.obs["min_score"] = pd.Series(
        sub.set_index("cell_barcode")["min_score"], index=adata.obs_names
    ).values
    adata.obs["singlet_score"] = pd.Series(
        sub.set_index("cell_barcode")["singlet_score"], index=adata.obs_names
    ).values
    return adata


def morans_i(x, neighbors):
    """Moran's I using pre-computed neighbor lists (list of arrays of neighbor indices)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    z = x - x.mean()
    denom = np.sum(z ** 2)
    if denom == 0:
        return np.nan

    num = 0.0
    w_sum = 0
    for i, nb in enumerate(neighbors):
        if len(nb) == 0:
            continue
        # exclude self if present
        nb = nb[nb != i]
        if len(nb) == 0:
            continue
        w_sum += len(nb)
        num += np.sum(z[i] * z[nb])

    if w_sum == 0:
        return np.nan
    return (n / w_sum) * (num / denom)


def local_tumor_fraction(tumor_indicator, neighbors):
    """For each cell, fraction of neighbors called tumor-dominant."""
    out = np.full(len(tumor_indicator), np.nan)
    for i, nb in enumerate(neighbors):
        nb = nb[nb != i]
        if len(nb) == 0:
            continue
        out[i] = tumor_indicator[nb].mean()
    return out


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


def sparse_mean(adata, genes, mask):
    if not np.any(mask):
        return np.full(len(genes), np.nan)
    present_genes = [g for g in genes if g in adata.var_names]
    idx = [adata.var_names.get_loc(g) for g in present_genes]
    sub = adata.X[mask, :][:, idx]
    if sparse.issparse(sub):
        return np.asarray(sub.mean(axis=0)).ravel()
    return sub.mean(axis=0)


def distance_to_boundary(coords, tumor_indicator):
    """
    For each cell, compute minimal Euclidean distance to the nearest cell with
    opposite tumor label. Positive = inside tumor region, negative = outside.
    Uses cKDTree for speed.
    """
    tumor_idx = np.where(tumor_indicator)[0]
    nontumor_idx = np.where(~tumor_indicator)[0]
    if len(tumor_idx) == 0 or len(nontumor_idx) == 0:
        return np.full(len(tumor_indicator), np.nan)

    tree_tumor = cKDTree(coords[tumor_idx])
    tree_nontumor = cKDTree(coords[nontumor_idx])

    dist_nontumor_to_tumor, _ = tree_tumor.query(coords[nontumor_idx], k=1)
    dist_tumor_to_nontumor, _ = tree_nontumor.query(coords[tumor_idx], k=1)

    d = np.full(len(tumor_indicator), np.nan)
    d[nontumor_idx] = -dist_nontumor_to_tumor
    d[tumor_idx] = dist_tumor_to_nontumor
    return d


def analyze_method(method, rctd_df, coords_all, raw_barcodes=None):
    print(f"\nProcessing {method} ...")
    adata = load_h5ad(method)
    adata = attach_predictions(adata, rctd_df, method)

    # merge coordinates
    obs = adata.obs.copy()
    obs = obs.merge(coords_all, on="cell_id", how="left")
    valid = obs[["x", "y"]].notna().all(axis=1)
    obs = obs[valid].copy()
    adata = adata[valid].copy()

    if raw_barcodes is not None:
        keep = obs["cell_barcode"].isin(raw_barcodes)
        obs = obs[keep].copy()
        adata = adata[keep].copy()
        subset_label = "shared_with_RAW"
    else:
        subset_label = "all"

    coords = obs[["x", "y"]].values
    n = len(obs)
    print(f"  {subset_label}: {n} cells with coordinates")
    if n == 0:
        return None

    first_type = obs["rctd_first_type"].astype(str).values
    tumor_indicator = (first_type == "Tumor").astype(int)

    # Build KDTree
    tree = SklearnKDTree(coords, leaf_size=40)

    records = []
    for radius in RADII:
        print(f"  radius={radius} ...")
        neighbors = tree.query_radius(coords, r=radius)
        # remove self from neighbor list for Moran's I / local stats
        neighbors = [nb[nb != i] for i, nb in enumerate(neighbors)]

        moran = morans_i(tumor_indicator, neighbors)
        local_tumor_frac = local_tumor_fraction(tumor_indicator, neighbors)
        local_ent = local_entropy(first_type, neighbors)

        records.append({
            "method": method,
            "subset": subset_label,
            "radius": radius,
            "n_cells": n,
            "morans_i_tumor": moran,
            "mean_local_tumor_fraction": np.nanmean(local_tumor_frac),
            "mean_local_entropy": np.nanmean(local_ent),
            "std_local_entropy": np.nanstd(local_ent),
            "frac_tumor_calls": tumor_indicator.mean(),
        })

    spatial_df = pd.DataFrame(records)

    # Distance-to-boundary analysis
    print("  distance-to-boundary profiles ...")
    dist = distance_to_boundary(coords, tumor_indicator.astype(bool))
    obs["dist_to_boundary"] = dist

    # Bin distances
    bins = np.linspace(-500, 500, 21)
    obs["dist_bin"] = pd.cut(obs["dist_to_boundary"], bins=bins)

    boundary_records = []
    for b, grp in obs.groupby("dist_bin", observed=False):
        if len(grp) < 10:
            continue
        rec = {
            "method": method,
            "subset": subset_label,
            "dist_bin_left": b.left,
            "dist_bin_right": b.right,
            "dist_bin_center": (b.left + b.right) / 2,
            "n_cells": len(grp),
            "frac_tumor": grp["rctd_first_type"].eq("Tumor").mean(),
            "mean_min_score": grp["min_score"].mean(),
            "mean_singlet_score": grp["singlet_score"].mean(),
        }
        # marker means inside this bin
        mask = obs.index.isin(grp.index)
        vals = sparse_mean(adata, MARKERS, mask)
        for g, v in zip([g for g in MARKERS if g in adata.var_names], vals):
            rec[g] = v
        boundary_records.append(rec)
    boundary_df = pd.DataFrame(boundary_records)

    return spatial_df, boundary_df


def make_plots(spatial_df, boundary_df, out_dir):
    if spatial_df.empty:
        return

    # 1. Moran's I across radii
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for method in spatial_df["method"].unique():
        d = spatial_df[(spatial_df["method"] == method) & (spatial_df["subset"] == "all")]
        ax.plot(d["radius"], d["morans_i_tumor"], marker="o", label=method)
    ax.set_xlabel("Radius")
    ax.set_ylabel("Moran's I (tumor indicator)")
    ax.set_title("Spatial autocorrelation of tumor calls")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "morans_i_tumor.png", dpi=200)
    plt.close(fig)

    # 2. Local entropy across radii
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for method in spatial_df["method"].unique():
        d = spatial_df[(spatial_df["method"] == method) & (spatial_df["subset"] == "all")]
        ax.plot(d["radius"], d["mean_local_entropy"], marker="o", label=method)
    ax.set_xlabel("Radius")
    ax.set_ylabel("Mean local entropy")
    ax.set_title("RCTD class label disorder within local neighborhoods")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "local_entropy.png", dpi=200)
    plt.close(fig)

    # 3. Tumor fraction across boundary
    if not boundary_df.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        for method in boundary_df["method"].unique():
            d = boundary_df[(boundary_df["method"] == method) & (boundary_df["subset"] == "all")]
            ax.plot(d["dist_bin_center"], d["frac_tumor"], marker="o", label=method)
        ax.axvline(0, color="gray", linestyle="--", linewidth=1)
        ax.set_xlabel("Distance to tumor boundary (negative = outside)")
        ax.set_ylabel("Fraction of cells called Tumor")
        ax.set_title("Tumor call fraction across spatial boundary")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "boundary_tumor_fraction.png", dpi=200)
        plt.close(fig)

    # 4. Marker profile across boundary (EPCAM and VIM as examples)
    if not boundary_df.empty and "EPCAM" in boundary_df.columns and "VIM" in boundary_df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        for ax, gene in [(axes[0], "EPCAM"), (axes[1], "VIM")]:
            for method in boundary_df["method"].unique():
                d = boundary_df[(boundary_df["method"] == method) & (boundary_df["subset"] == "all")]
                ax.plot(d["dist_bin_center"], d[gene], marker="o", label=method)
            ax.axvline(0, color="gray", linestyle="--", linewidth=1)
            ax.set_xlabel("Distance to tumor boundary")
            ax.set_ylabel(f"Mean {gene} expression")
            ax.set_title(f"{gene} across tumor boundary")
            ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "boundary_marker_profiles.png", dpi=200)
        plt.close(fig)


def main():
    global MARKERS
    print("Loading RCTD results ...")
    rctd_df = load_results()

    print("Loading scRNA-derived tumor markers ...")
    raw_adata = load_h5ad("RAW")
    MARKERS = load_learned_markers(raw_adata, top_n=30)
    print(f"  Learned markers present in Visium HD: {len(MARKERS)}")

    print("Loading h5ad cell ids for coordinate computation ...")
    all_cell_ids = set()
    for method in METHOD_FILES.keys():
        adata = load_h5ad(method)
        all_cell_ids.update(adata.obs["cell_id"].astype(int).tolist())
    print(f"  {len(all_cell_ids)} unique cell ids to look up")

    print("Computing cell coordinates from segmentation mask ...")
    coords_all = compute_cell_coordinates(FEATURE_SLICE_H5, all_cell_ids)
    print(f"  coordinates for {len(coords_all)} cells")

    raw_barcodes = None
    # first pass: get raw barcodes for shared analysis
    raw_adata = load_h5ad("RAW")
    raw_adata = attach_predictions(raw_adata, rctd_df, "RAW")
    raw_barcodes = set(raw_adata.obs.loc[raw_adata.obs["rctd_first_type"].notna(), "cell_barcode"])

    spatial_records = []
    boundary_records = []

    for method in METHOD_FILES.keys():
        res = analyze_method(method, rctd_df, coords_all, raw_barcodes=None)
        if res is not None:
            spatial_records.append(res[0])
            boundary_records.append(res[1])

        res_shared = analyze_method(method, rctd_df, coords_all, raw_barcodes=raw_barcodes)
        if res_shared is not None:
            spatial_records.append(res_shared[0])
            boundary_records.append(res_shared[1])

    spatial_df = pd.concat(spatial_records, ignore_index=True)
    boundary_df = pd.concat(boundary_records, ignore_index=True)

    spatial_df.to_csv(OUT_DIR / "spatial_autocorrelation_metrics.csv", index=False)
    boundary_df.to_csv(OUT_DIR / "boundary_profiles.csv", index=False)

    print("Saving plots ...")
    make_plots(spatial_df, boundary_df, OUT_DIR)

    print("\n=== Spatial autocorrelation (Moran's I, tumor indicator) ===")
    print(spatial_df[spatial_df["subset"] == "all"]
          [["method", "radius", "morans_i_tumor", "mean_local_entropy", "frac_tumor_calls"]]
          .to_string(index=False))

    print("\n=== Boundary sharpness: tumor fraction jump at boundary ===")
    bnd = boundary_df[boundary_df["subset"] == "all"].copy()
    summary = []
    for method in bnd["method"].unique():
        d = bnd[bnd["method"] == method].set_index("dist_bin_center")
        outside = d.loc[-25, "frac_tumor"] if -25 in d.index else np.nan
        inside = d.loc[25, "frac_tumor"] if 25 in d.index else np.nan
        summary.append({
            "method": method,
            "frac_tumor_outside_-25": outside,
            "frac_tumor_inside_+25": inside,
            "tumor_fraction_jump": inside - outside if pd.notna(inside) and pd.notna(outside) else np.nan,
        })
    print(pd.DataFrame(summary).to_string(index=False))

    print("\n=== Boundary marker contrast (inside +25 / outside -25) ===")
    contrast_records = []
    for method in bnd["method"].unique():
        d = bnd[bnd["method"] == method].set_index("dist_bin_center")
        if -25 not in d.index or 25 not in d.index:
            continue
        outside = d.loc[-25]
        inside = d.loc[25]
        for gene in [g for g in MARKERS if g in d.columns]:
            ratio = inside[gene] / outside[gene] if outside[gene] > 0 else np.nan
            contrast_records.append({
                "method": method,
                "gene": gene,
                "inside": inside[gene],
                "outside": outside[gene],
                "inside_outside_ratio": ratio,
            })
    contrast_df = pd.DataFrame(contrast_records)
    if not contrast_df.empty:
        print(contrast_df.pivot(index="gene", columns="method", values="inside_outside_ratio").to_string())

    print(f"\nOutputs written to {OUT_DIR}")


if __name__ == "__main__":
    main()
