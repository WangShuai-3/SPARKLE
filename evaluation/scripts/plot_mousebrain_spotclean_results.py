#!/usr/bin/env python3
"""Plot MouseBrain SpotClean RCTD and snRNA-reference comparisons."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD_ORDER = ["RAW", "SPARKLE", "SpotClean", "SOUPX", "SPATIALSOUPX", "DECONTX"]
METHOD_LABELS = {
    "RAW": "RAW",
    "SPARKLE": "SPARKLE",
    "SpotClean": "SpotClean",
    "SOUPX": "SoupX",
    "SPATIALSOUPX": "SpatialSoupX",
    "DECONTX": "DecontX",
}
METHOD_COLORS = {
    "RAW": "#9e9e9e",
    "SPARKLE": "#f28e2b",
    "SpotClean": "#4e79a7",
    "SOUPX": "#59a14f",
    "SPATIALSOUPX": "#b07aa1",
    "DECONTX": "#e15759",
}


def save_figure(fig, base_path: Path) -> None:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_rctd(summary_path: Path, output_dir: Path) -> None:
    summary = pd.read_csv(summary_path).set_index("method")
    methods = [method for method in METHOD_ORDER if method in summary.index]
    summary = summary.loc[methods]
    labels = [METHOD_LABELS[method] for method in methods]
    colors = [METHOD_COLORS[method] for method in methods]

    panels = [
        ("n_cells", "Retained cells", "Cells retained by RCTD"),
        ("pct_singlet", "Singlet (%)", "Singlet classification rate"),
        ("mean_max_weight", "Mean max weight", "Top cell-type weight"),
        ("mean_entropy", "Mean entropy", "Cell-type weight entropy"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    x = np.arange(len(methods))
    for ax, (column, ylabel, title) in zip(axes.flat, panels):
        values = summary[column].to_numpy(float)
        bars = ax.bar(x, values, color=colors, width=0.72)
        ax.set_xticks(x, labels, rotation=28, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#dddddd", linewidth=0.7, alpha=0.7)
        ax.set_axisbelow(True)
        for bar, value in zip(bars, values):
            label = f"{value:,.0f}" if column == "n_cells" else f"{value:.3f}"
            ax.annotate(
                label,
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    fig.suptitle(
        "MouseBrain RCTD comparison (Cell_group reference)",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.005,
        "Rates and confidence metrics are calculated over cells retained by RCTD; retained n differs by method.",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    save_figure(fig, output_dir / "rctd_Cell_group_metrics_comparison")


def plot_snrna_delta(per_group_path: Path, output_dir: Path, tag: str) -> None:
    per_group = pd.read_csv(per_group_path)
    key = ["method", "cell_group"]
    spreads = per_group.groupby(key, dropna=False)["corr"].agg(
        lambda values: values.max() - values.min()
    )
    if (spreads.fillna(0) > 1e-12).any():
        raise ValueError("Conflicting duplicate method/cell_group rows detected.")
    per_group = per_group.drop_duplicates(key, keep="first")
    raw = per_group[per_group["method"].str.lower() == "raw"][
        ["cell_group", "corr"]
    ].rename(columns={"corr": "raw_corr"})
    spotclean = per_group[per_group["method"] == "SpotClean"][
        ["cell_group", "corr"]
    ].rename(columns={"corr": "spotclean_corr"})
    delta = raw.merge(spotclean, on="cell_group", validate="one_to_one")
    delta["delta"] = delta["spotclean_corr"] - delta["raw_corr"]
    delta = delta.sort_values("delta").reset_index(drop=True)

    delta_path = output_dir / f"{tag}_SpotClean_vs_RAW_snrna_cellgroup_delta.csv"
    delta.to_csv(delta_path, index=False)

    height = max(7, 0.24 * len(delta))
    fig, ax = plt.subplots(figsize=(9, height))
    y = np.arange(len(delta))
    colors = np.where(delta["delta"] >= 0, "#2a9d8f", "#d55e00")
    ax.barh(y, delta["delta"], color=colors, height=0.72)
    ax.set_yticks(y, delta["cell_group"], fontsize=8)
    ax.tick_params(axis="y", length=0)
    ax.axvline(0, color="#333333", linewidth=0.9)
    ax.set_xlabel("Pearson correlation change (SpotClean − RAW)")
    ax.set_title(
        "SpotClean changes snRNA-reference agreement by cell group",
        loc="left",
        fontweight="bold",
    )
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", color="#dddddd", linewidth=0.7, alpha=0.7)
    ax.set_axisbelow(True)
    mean_delta = delta["delta"].mean()
    improved = int((delta["delta"] > 0).sum())
    ax.text(
        0.99,
        0.01,
        f"Mean Δ = {mean_delta:+.3f}; improved groups = {improved}/{len(delta)}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color="#555555",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 2},
    )
    fig.tight_layout()
    save_figure(fig, output_dir / f"{tag}_SpotClean_vs_RAW_snrna_cellgroup_delta")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tag", default="mousebrain_x12500-20000_y2000-10000"
    )
    parser.add_argument(
        "--rctd-dir", default="evaluation/reports/rctd_mousebrain"
    )
    parser.add_argument(
        "--eval-dir", default="evaluation/reports/mousebrain_eval/method_level_new"
    )
    parser.add_argument(
        "--output-dir", default="evaluation/reports/mousebrain_eval/visualization"
    )
    args = parser.parse_args()

    rctd_dir = Path(args.rctd_dir)
    eval_dir = Path(args.eval_dir)
    output_dir = Path(args.output_dir)
    plot_rctd(rctd_dir / "rctd_Cell_group_summary_metrics.csv", output_dir)
    plot_snrna_delta(
        eval_dir / f"{args.tag}_snrna_corr_per_celltype.csv",
        output_dir,
        args.tag,
    )


if __name__ == "__main__":
    main()
