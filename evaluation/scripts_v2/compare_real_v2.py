#!/usr/bin/env python3
"""SPARKLE 2.x real-data comparison: RAW vs SPARKLEv1 vs SPARKLEv2(NP).

Pure Python (no R/RCTD rerun). Uses the identical metrics and references as
the v1 evaluation:

  MouseBrain (tag mousebrain_x12500-20000_y2000-10000):
    - cell_group pseudobulk vs snRNA reference Pearson/Spearman correlation
      (evaluation/data/mousebrain/snrna_cell_group_pseudobulk.csv)
    - contamination score + reduction vs RAW
    - annotation silhouette on PCA
    Annotations come from the transfer labels baked into the h5ads.

  Ovarian (tag ovarian_x1000-1800_y300-1100):
    - same metrics against evaluation/data/ovarian/scrna_celltype_pseudobulk.csv
    - cell grouping reuses the v1 RCTD RAW first_type assignments
      (evaluation/reports/rctd_ovarian/first_type/RAW_first_type.csv), the
      same shared grouping the v1 manuscript used for every method.

  Axolotl (tag axolotl_x10500-12500_y6000-11100):
    - SST (AMEX60DD003175) mean expression in sstIN / neighbor / other cells
      and the sstIN:neighbor / neighbor:other contrasts.

Outputs under evaluation/reports/v2/real/.
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import anndata as ad

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.scripts.evaluate_mousebrain_h5ad import (
    compute_contamination_summary,
    compute_pseudobulk,
    compute_silhouette,
    compute_snrna_correlations,
    load_snrna_reference,
    normalize_adata,
)
from evaluation.scripts.visualize_axolotl_sst_boxplots import (
    SST_GENE,
    extract_sst_expression,
    load_cell_masks,
)

METHODS = ["RAW", "SPARKLEv1", "SPARKLEv2", "SPARKLEv2NP",
           "SPARKLEv2R1", "SPARKLEv2R2",
           "SPARKLEv2P1", "SPARKLEv2P2", "SPARKLEv2P2D",
           "SPARKLEv2E1", "SPARKLEv2E2"]
METHOD_FILES = {m: ("raw" if m == "RAW" else m) for m in METHODS}
METHOD_COLORS = {"RAW": "#7f8c8d", "SPARKLEv1": "#e74c3c",
                 "SPARKLEv2": "#2980b9", "SPARKLEv2NP": "#27ae60",
                 "SPARKLEv2R1": "#8e44ad", "SPARKLEv2R2": "#d35400",
                 "SPARKLEv2P1": "#16a085", "SPARKLEv2P2": "#c0392b",
                 "SPARKLEv2P2D": "#2c3e50", "SPARKLEv2E1": "#e67e22",
                 "SPARKLEv2E2": "#1abc9c"}

TAGS = {
    "mousebrain": "mousebrain_x12500-20000_y2000-10000",
    "ovarian": "ovarian_x1000-1800_y300-1100",
    "axolotl": "axolotl_x10500-12500_y6000-11100",
}
REFS = {
    "mousebrain": PROJECT_ROOT / "evaluation" / "data" / "mousebrain"
    / "snrna_cell_group_pseudobulk.csv",
    "ovarian": PROJECT_ROOT / "evaluation" / "data" / "ovarian"
    / "scrna_celltype_pseudobulk.csv",
}
OVARIAN_FIRST_TYPE = (
    PROJECT_ROOT / "evaluation" / "reports" / "rctd_ovarian"
    / "first_type" / "RAW_first_type.csv"
)


def _v2_reports_root():
    return PROJECT_ROOT / "evaluation" / "reports" / "v2"


def _load_method_adata(dataset, method):
    tag = TAGS[dataset]
    path = _v2_reports_root() / "h5ad" / f"{tag}_{METHOD_FILES[method]}.h5ad"
    if not path.exists():
        return None
    adata = ad.read_h5ad(path)
    if dataset == "ovarian":
        existing = (
            adata.obs["annotation"].astype(str)
            if "annotation" in adata.obs.columns
            else pd.Series(dtype=str)
        )
        has_labels = len(existing) > 0 and not existing.isin(
            ["Unknown", "nan", "None", ""]
        ).all()
        if not has_labels:
            # Shared v1 grouping: RCTD RAW first_type (same as the manuscript).
            ft = pd.read_csv(OVARIAN_FIRST_TYPE)
            mapping = dict(
                zip(ft["cell_id"].astype(int), ft["first_type"].astype(str))
            )
            adata.obs["annotation"] = [
                mapping.get(int(cid), "Unknown") for cid in adata.obs["cell_id"]
            ]
    return adata


def evaluate_reference_concordance(dataset, out_dir):
    """Pseudobulk-vs-reference metrics shared by mousebrain and ovarian."""
    ref = load_snrna_reference(REFS[dataset])
    rows = []
    per_type_frames = []
    raw_pb = None
    for method in METHODS:
        adata = _load_method_adata(dataset, method)
        if adata is None:
            print(f"  [skip] {dataset} {method}: h5ad not found")
            continue
        norm = normalize_adata(adata)
        pb = compute_pseudobulk(norm, group_key="annotation", min_cells=3)
        if method == "RAW":
            raw_pb = pb
        pearson, pearson_s = compute_snrna_correlations(pb, ref, method="pearson")
        _, spearman_s = compute_snrna_correlations(pb, ref, method="spearman")
        contam = compute_contamination_summary(pb, ref, raw_pb_df=raw_pb)
        sil = compute_silhouette(norm, group_key="annotation")
        row = {
            "dataset": dataset,
            "method": method,
            "mean_snrna_pearson": float(np.nanmean(pearson_s)),
            "mean_snrna_spearman": float(np.nanmean(spearman_s)),
            "median_snrna_pearson": float(np.nanmedian(pearson_s)),
            "silhouette": sil,
            **contam,
        }
        rows.append(row)
        per_type = pearson_s.rename("snrna_pearson").to_frame()
        per_type["snrna_spearman"] = spearman_s
        per_type["method"] = method
        per_type["cell_group"] = per_type.index
        per_type_frames.append(per_type.reset_index(drop=True))
        print(f"  {method:<11} pearson={row['mean_snrna_pearson']:.4f} "
              f"spearman={row['mean_snrna_spearman']:.4f} "
              f"contam={row['mean_contamination']:.4f}")

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / f"{dataset}_v2_summary.csv", index=False)
    if per_type_frames:
        pd.concat(per_type_frames).to_csv(
            out_dir / f"{dataset}_v2_per_cellgroup.csv", index=False
        )

    # Bar plot of the headline metrics.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, col, label in zip(
        axes,
        ["mean_snrna_pearson", "mean_snrna_spearman", "mean_contamination"],
        ["Mean Pearson vs snRNA/scRNA ref", "Mean Spearman vs ref",
         "Mean contamination score"],
    ):
        sub = summary.set_index("method").reindex(METHODS).dropna(subset=[col])
        ax.bar(sub.index, sub[col],
               color=[METHOD_COLORS[m] for m in sub.index])
        ax.set_ylabel(label)
        ax.tick_params(axis="x", rotation=30)
        for i, v in enumerate(sub[col]):
            ax.text(i, v, f"{v:.3f}", ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=8)
    fig.suptitle(f"{dataset}: SPARKLE 2.x latent-X ablation")
    fig.tight_layout()
    fig.savefig(out_dir / f"{dataset}_v2_summary.png", dpi=150)
    plt.close(fig)
    return summary


def evaluate_axolotl(out_dir, neighbor_radius=50.0):
    tag = TAGS["axolotl"]
    cache = out_dir / f"{tag}_masks_r{int(neighbor_radius)}.pkl"
    sstin_mask, neighbor_mask, other_mask, cid_to_idx = load_cell_masks(
        tag, neighbor_radius=neighbor_radius, cache_path=cache
    )
    rows = []
    for method in METHODS:
        path = _v2_reports_root() / "h5ad" / f"{tag}_{METHOD_FILES[method]}.h5ad"
        if not path.exists():
            print(f"  [skip] axolotl {method}: h5ad not found")
            continue
        vals, cids = extract_sst_expression(path)
        idx = np.array([cid_to_idx[int(c)] for c in cids])
        aligned = np.zeros(len(cid_to_idx))
        aligned[idx] = vals
        sst_in = aligned[sstin_mask]
        nbr = aligned[neighbor_mask]
        oth = aligned[other_mask]
        row = {
            "method": method,
            "sst_sstIN_mean": float(sst_in.mean()),
            "sst_neighbor_mean": float(nbr.mean()),
            "sst_other_mean": float(oth.mean()),
            "sstIN_over_neighbor": float(sst_in.mean() / max(nbr.mean(), 1e-12)),
            "neighbor_over_other": float(nbr.mean() / max(oth.mean(), 1e-12)),
        }
        rows.append(row)
        print(f"  {method:<11} sstIN={row['sst_sstIN_mean']:7.2f} "
              f"nbr={row['sst_neighbor_mean']:6.2f} oth={row['sst_other_mean']:5.2f} "
              f"S/N={row['sstIN_over_neighbor']:.2f}x N/O={row['neighbor_over_other']:.2f}x")

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "axolotl_v2_sst_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(summary))
    width = 0.27
    for i, (col, label) in enumerate([
        ("sst_sstIN_mean", "sstIN (source)"),
        ("sst_neighbor_mean", f"Neighbor (≤{neighbor_radius:.0f}µm)"),
        ("sst_other_mean", "Other"),
    ]):
        ax.bar(x + (i - 1) * width, summary[col], width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(summary["method"])
    ax.set_ylabel(f"Mean {SST_GENE} (SST) counts per cell")
    ax.set_title("Axolotl SST: source preservation vs neighbor leakage removal")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "axolotl_v2_sst.png", dpi=150)
    plt.close(fig)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", type=str, nargs="*",
                        default=["mousebrain", "ovarian", "axolotl"])
    parser.add_argument("--neighbor-radius", type=float, default=50.0)
    args = parser.parse_args()

    out_dir = _v2_reports_root() / "real"
    out_dir.mkdir(parents=True, exist_ok=True)

    for dataset in args.datasets:
        print(f"\n=== {dataset} ===")
        if dataset == "axolotl":
            evaluate_axolotl(out_dir, neighbor_radius=args.neighbor_radius)
        else:
            evaluate_reference_concordance(dataset, out_dir)
    print(f"\nOutputs written to {out_dir}")


if __name__ == "__main__":
    main()
