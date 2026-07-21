#!/usr/bin/env python3
"""Boxplot of SST expression across sstIN / neighbor / other cells for Axolotl.

Reads the corrected h5ad files produced by final_comparison.py and uses the
same sstIN / neighbor / other definitions as the Axolotl benchmark.
"""

import argparse
import pickle
import re
import sys
from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Import Axolotl loading helpers from final_comparison.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.scripts.final_comparison import compute_neighbor_stats, load_axolotl_data_windowed

SST_GENE = "AMEX60DD003175"
METHOD_ORDER = ["RAW", "SPARKLE", "SoupX", "DecontX", "SpotClean"]
CATEGORY_ORDER = ["sstIN", "Neighbor", "Other"]
METHOD_SUFFIX = {
    "RAW": "raw",
    "SPARKLE": "SPARKLE",
    "SoupX": "SoupX",
    "DecontX": "DecontX",
    "SpotClean": "SpotCleanOfficial",
}
METHOD_COLORS = {
    "RAW": "#7f8c8d",
    "SPARKLE": "#e74c3c",
    "SoupX": "#3498db",
    "DecontX": "#2ecc71",
    "SpotClean": "#9b59b6",
}


def parse_tag(tag):
    """Parse x_range/y_range from tag like 'axolotl_x10500-12500_y6000-11100'."""
    m = re.search(r"_x(\d+)-(\d+)_y(\d+)-(\d+)", tag)
    if not m:
        raise ValueError(f"Cannot parse ranges from tag '{tag}'")
    return (int(m.group(1)), int(m.group(2))), (int(m.group(3)), int(m.group(4)))


def load_cell_masks(tag, neighbor_radius=50.0, cache_path=None):
    """Return sstIN/neighbor/other masks and a cell_id -> mask-index map.

    Caches masks to disk so that visualization can be re-run without reloading
    the large Axolotl GEM file.
    """
    if cache_path is not None and Path(cache_path).exists():
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        print(f"Loaded cached masks from {cache_path}")
        return (
            cached["sstin_mask"],
            cached["neighbor_mask"],
            cached["other_mask"],
            cached["cid_to_idx"],
        )

    x_range, y_range = parse_tag(tag)
    data = load_axolotl_data_windowed(x_range=x_range, y_range=y_range)

    cell_ids = np.array(data["cell_ids"])
    sstin_set = data["sstin_set"]
    dnb_coords = data["dnb_coords"]
    dnb_labels = data["dnb_labels"]
    n_cells = len(cell_ids)

    sstin_mask = np.array([cell_ids[i] in sstin_set for i in range(n_cells)])

    # Compute cell centroids from DNB coordinates
    centroids = np.zeros((n_cells, 2))
    for c in range(n_cells):
        m = dnb_labels == cell_ids[c]
        if m.sum():
            centroids[c] = dnb_coords[m].mean(axis=0)

    neighbor_mask, other_mask = compute_neighbor_stats(centroids, sstin_mask, radius=neighbor_radius)

    cid_to_idx = {int(cid): i for i, cid in enumerate(cell_ids)}

    if cache_path is not None:
        cache = {
            "sstin_mask": sstin_mask,
            "neighbor_mask": neighbor_mask,
            "other_mask": other_mask,
            "cid_to_idx": cid_to_idx,
        }
        with open(cache_path, "wb") as f:
            pickle.dump(cache, f)
        print(f"Saved masks cache to {cache_path}")

    return sstin_mask, neighbor_mask, other_mask, cid_to_idx


def extract_sst_expression(h5ad_path, sst_gene=SST_GENE):
    """Load SST expression vector and cell IDs from an h5ad file."""
    adata = ad.read_h5ad(h5ad_path)
    x = adata[:, sst_gene].X
    if hasattr(x, "toarray"):
        x = x.toarray().ravel()
    else:
        x = np.asarray(x).ravel()
    return x, adata.obs["cell_id"].values


def build_dataframe(tag, input_dir, cache_path=None):
    """Build a long-form DataFrame: method, category, sst_expression."""
    sstin_mask, neighbor_mask, other_mask, cid_to_idx = load_cell_masks(tag, cache_path=cache_path)

    rows = []
    for method in METHOD_ORDER:
        suffix = METHOD_SUFFIX[method]
        path = Path(input_dir) / f"{tag}_{suffix}.h5ad"
        if not path.exists():
            raise FileNotFoundError(f"Missing h5ad for {method}: {path}")

        vals, cids = extract_sst_expression(path)
        categories = []
        for cid in cids:
            idx = cid_to_idx[int(cid)]
            if sstin_mask[idx]:
                categories.append("sstIN")
            elif neighbor_mask[idx]:
                categories.append("Neighbor")
            else:
                categories.append("Other")

        rows.extend({
            "method": method,
            "category": cat,
            "sst_expression": float(v),
        } for v, cat in zip(vals, categories))

    return pd.DataFrame(rows)


def plot_boxplots(df, output_path, figsize=(14, 5)):
    """Three subplots (sstIN / Neighbor / Other) with one box per method."""
    categories = CATEGORY_ORDER
    fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=False)

    for ax, cat in zip(axes, categories):
        sub = df[df["category"] == cat]
        sns.boxplot(
            data=sub,
            x="method",
            y="sst_log1p",
            hue="method",
            order=METHOD_ORDER,
            hue_order=METHOD_ORDER,
            palette=METHOD_COLORS,
            ax=ax,
            legend=False,
            showfliers=False,  # cleaner; outliers still visible via whiskers
        )
        ax.set_title(f"{cat} cells", fontsize=12)
        ax.set_xlabel("")
        ax.set_ylabel("log1p(SST expression)" if cat == "sstIN" else "")
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    fig.suptitle("log1p(SST expression) by cell category (Axolotl)", fontsize=14)
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved boxplots: {output_path}")


def plot_grouped_boxplot(df, output_path, figsize=(10, 6)):
    """Alternative: single figure with categories on x-axis and method as hue."""
    fig, ax = plt.subplots(figsize=figsize)
    sns.boxplot(
        data=df,
        x="category",
        y="sst_log1p",
        hue="method",
        order=CATEGORY_ORDER,
        hue_order=METHOD_ORDER,
        palette=METHOD_COLORS,
        ax=ax,
        showfliers=False,
    )
    ax.set_title("log1p(SST expression) by cell category and method (Axolotl)", fontsize=13)
    ax.set_xlabel("Cell category", fontsize=12)
    ax.set_ylabel("log1p(SST expression)", fontsize=12)
    ax.legend(title="Method", loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved grouped boxplot: {output_path}")


def plot_method_grouped_boxplot(df, output_path, figsize=(10, 6)):
    """Boxplot grouped by method: x=method, hue=category."""
    fig, ax = plt.subplots(figsize=figsize)
    sns.boxplot(
        data=df,
        x="method",
        y="sst_log1p",
        hue="category",
        order=METHOD_ORDER,
        hue_order=CATEGORY_ORDER,
        palette={"sstIN": "#e74c3c", "Neighbor": "#f39c12", "Other": "#3498db"},
        ax=ax,
        showfliers=False,
    )
    ax.set_title("log1p(SST expression) grouped by method (Axolotl)", fontsize=13)
    ax.set_xlabel("Method", fontsize=12)
    ax.set_ylabel("log1p(SST expression)", fontsize=12)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    ax.legend(title="Cell category", loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved method-grouped boxplot: {output_path}")


def plot_violins(df, output_path, figsize=(14, 5)):
    """Three violin subplots (sstIN / Neighbor / Other) with one violin per method."""
    categories = CATEGORY_ORDER
    fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=False)

    for ax, cat in zip(axes, categories):
        sub = df[df["category"] == cat]
        sns.violinplot(
            data=sub,
            x="method",
            y="sst_log1p",
            hue="method",
            order=METHOD_ORDER,
            hue_order=METHOD_ORDER,
            palette=METHOD_COLORS,
            ax=ax,
            legend=False,
            inner="box",
            cut=0,
        )
        ax.set_title(f"{cat} cells", fontsize=12)
        ax.set_xlabel("")
        ax.set_ylabel("log1p(SST expression)" if cat == "sstIN" else "")
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    fig.suptitle("log1p(SST expression) distribution by cell category (Axolotl)", fontsize=14)
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved violin plots: {output_path}")


def plot_method_grouped_violin(df, output_path, figsize=(10, 6)):
    """Violin plot grouped by method: x=method, hue=category."""
    fig, ax = plt.subplots(figsize=figsize)
    sns.violinplot(
        data=df,
        x="method",
        y="sst_log1p",
        hue="category",
        order=METHOD_ORDER,
        hue_order=CATEGORY_ORDER,
        palette={"sstIN": "#e74c3c", "Neighbor": "#f39c12", "Other": "#3498db"},
        ax=ax,
        inner="box",
        cut=0,
    )
    ax.set_title("log1p(SST expression) distribution grouped by method (Axolotl)", fontsize=13)
    ax.set_xlabel("Method", fontsize=12)
    ax.set_ylabel("log1p(SST expression)", fontsize=12)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    ax.legend(title="Cell category", loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved method-grouped violin plot: {output_path}")


def plot_logfc(df, output_path, figsize=(10, 5)):
    """Bar plot of log2 fold changes: sstIN/Neighbor and Neighbor/Other per method."""
    means = df.groupby(["method", "category"])["sst_expression"].mean().unstack()
    means = means.reindex(METHOD_ORDER)

    # Add tiny pseudo-count to avoid log(0)
    eps = 1e-6
    logfc_sn = np.log2((means["sstIN"] + eps) / (means["Neighbor"] + eps))
    logfc_no = np.log2((means["Neighbor"] + eps) / (means["Other"] + eps))

    x = np.arange(len(METHOD_ORDER))
    width = 0.35

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x - width / 2, logfc_sn.values, width, label="sstIN / Neighbor", color="#e74c3c")
    ax.bar(x + width / 2, logfc_no.values, width, label="Neighbor / Other", color="#3498db")

    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(METHOD_ORDER, rotation=30, ha="right")
    ax.set_ylabel("log2 FC", fontsize=12)
    ax.set_title("SST signal-to-noise ratios (Axolotl)\nhigher = better separation", fontsize=13)
    ax.legend(title="Comparison", loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved logFC plot: {output_path}")


def build_summary(df):
    """Return one comparison row per method with SST expression and separation."""
    grouped = df.groupby(["method", "category"])["sst_expression"]
    means = grouped.mean().unstack().reindex(METHOD_ORDER)
    medians = grouped.median().unstack().reindex(METHOD_ORDER)
    counts = grouped.count().unstack().reindex(METHOD_ORDER)
    eps = 1e-6
    summary = pd.DataFrame({
        "method": METHOD_ORDER,
        "sstin_mean": means["sstIN"].to_numpy(),
        "neighbor_mean": means["Neighbor"].to_numpy(),
        "other_mean": means["Other"].to_numpy(),
        "sstin_median": medians["sstIN"].to_numpy(),
        "neighbor_median": medians["Neighbor"].to_numpy(),
        "other_median": medians["Other"].to_numpy(),
        "n_sstin": counts["sstIN"].to_numpy(dtype=int),
        "n_neighbor": counts["Neighbor"].to_numpy(dtype=int),
        "n_other": counts["Other"].to_numpy(dtype=int),
        "log2fc_sstin_vs_neighbor": np.log2(
            (means["sstIN"].to_numpy() + eps) / (means["Neighbor"].to_numpy() + eps)
        ),
        "log2fc_neighbor_vs_other": np.log2(
            (means["Neighbor"].to_numpy() + eps) / (means["Other"].to_numpy() + eps)
        ),
    })
    return summary


def print_summary(summary):
    """Print the compact method-level SST comparison."""
    print("\nSST expression summary:")
    print(summary.round(3).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(
        description="Boxplot of SST expression in sstIN / neighbor / other cells for Axolotl."
    )
    parser.add_argument(
        "--tag",
        type=str,
        required=True,
        help="Axolotl window tag, e.g. axolotl_x10500-12500_y6000-11100",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="evaluation/reports/h5ad",
        help="Directory containing the corrected h5ad files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation/reports/axolotl_figures",
        help="Output directory for figures",
    )
    parser.add_argument(
        "--neighbor-radius",
        type=float,
        default=50.0,
        help="Radius (µm) for defining SST-neighbor cells (default: 50)",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cache_path = out_dir / f"{args.tag}_cell_masks_cache.pkl"
    df = build_dataframe(args.tag, args.input_dir, cache_path=cache_path)
    df["sst_log1p"] = np.log1p(df["sst_expression"])
    summary = build_summary(df)
    print_summary(summary)
    summary_path = out_dir / f"{args.tag}_sst_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved SST summary: {summary_path}")

    plot_boxplots(df, out_dir / f"{args.tag}_sst_boxplots.png")
    plot_grouped_boxplot(df, out_dir / f"{args.tag}_sst_grouped_boxplot.png")
    plot_method_grouped_boxplot(df, out_dir / f"{args.tag}_sst_method_grouped_boxplot.png")
    plot_violins(df, out_dir / f"{args.tag}_sst_violins.png")
    plot_method_grouped_violin(df, out_dir / f"{args.tag}_sst_method_grouped_violin.png")
    plot_logfc(df, out_dir / f"{args.tag}_sst_logfc.png")


if __name__ == "__main__":
    main()
