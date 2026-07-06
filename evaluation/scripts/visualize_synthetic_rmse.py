#!/usr/bin/env python3
"""Visualize synthetic S1-S10 RMSE across correction methods.

Reads metrics JSONs written by the synthetic benchmark and produces a bar
chart per scenario.  RAW (ground truth) is shown as a dashed line; other
methods are bars.  SpatialSoupX is excluded.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

# Make evaluation.synthetic.scenarios importable from evaluation/scripts/
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.synthetic.scenarios import SCENARIOS

METHOD_ORDER = ["SPARKLE", "SoupX", "DecontX"]
METHOD_COLORS = {
    "SPARKLE": "#e74c3c",
    "SoupX": "#3498db",
    "DecontX": "#2ecc71",
    "RAW": "#7f8c8d",
}


def load_metrics(metrics_dir):
    """Return (method_df, raw_df) for S1-S10 metrics."""
    rows, raw_rows = [], []
    pattern = re.compile(r"synthetic_(S\d+)_metrics\.json")
    for path in sorted(Path(metrics_dir).glob("synthetic_S*_metrics.json")):
        m = pattern.search(path.name)
        if not m:
            continue
        sid = m.group(1)
        with open(path) as f:
            data = json.load(f)

        raw_rows.append({"scenario": sid, "method": "RAW", "rmse": data["raw"]["rmse"]})
        for method, vals in data["methods"].items():
            if method == "SpatialSoupX":
                continue
            rows.append({"scenario": sid, "method": method, "rmse": vals["rmse"]})

    return pd.DataFrame(rows), pd.DataFrame(raw_rows)


def scenario_label(sid):
    """Return 'S1: Sparse multi-type (40% empty)' style label."""
    name = SCENARIOS.get(sid, {}).get("name", sid)
    return f"{sid}: {name}"


def _legend_handles():
    """Return handles/labels for the method + RAW legend."""
    handles = [Patch(facecolor=METHOD_COLORS[m], edgecolor="black") for m in METHOD_ORDER]
    handles.append(Line2D([0], [0], color=METHOD_COLORS["RAW"], linestyle="--", linewidth=2))
    labels = METHOD_ORDER + ["RAW (GT)"]
    return handles, labels


def _grid_shape(n):
    """Return (rows, cols) for up to n subplots, fixed 5 columns."""
    cols = 5
    rows = (n + cols - 1) // cols
    return rows, cols


def plot_rmse_subplots(method_df, raw_df, output_path, figsize=(20, 8)):
    """One subplot per scenario showing method RMSE bars and RAW dashed line."""
    scenarios = sorted(method_df["scenario"].unique(), key=lambda s: int(s[1:]))
    rows, cols = _grid_shape(len(scenarios))
    fig, axes = plt.subplots(rows, cols, figsize=figsize, sharey=False)
    axes = np.atleast_1d(axes).flatten()

    raw_map = raw_df.set_index("scenario")["rmse"]
    for idx, sid in enumerate(scenarios):
        ax = axes[idx]
        for i, method in enumerate(METHOD_ORDER):
            val = method_df[(method_df["scenario"] == sid) & (method_df["method"] == method)]["rmse"]
            if val.empty:
                continue
            ax.bar(
                i,
                val.values[0],
                color=METHOD_COLORS[method],
                edgecolor="black",
                linewidth=0.5,
            )
        ax.axhline(raw_map[sid], color=METHOD_COLORS["RAW"], linestyle="--", linewidth=2)
        ax.set_title(scenario_label(sid), fontsize=10)
        ax.set_xticks(range(len(METHOD_ORDER)))
        ax.set_xticklabels(METHOD_ORDER, rotation=30, ha="right", fontsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.set_axisbelow(True)

    for idx in range(len(scenarios), len(axes)):
        axes[idx].set_visible(False)

    handles, labels = _legend_handles()
    fig.legend(handles, labels, loc="upper right", ncol=4, title="Method", fontsize=10)
    fig.suptitle("RMSE per synthetic scenario (S1–S10)", fontsize=14)
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved RMSE subplots: {output_path}")


def plot_relative_rmse_subplots(method_df, raw_df, output_path, figsize=(20, 8)):
    """One subplot per scenario showing relative RMSE (method / RAW) bars."""
    scenarios = sorted(method_df["scenario"].unique(), key=lambda s: int(s[1:]))
    rows, cols = _grid_shape(len(scenarios))
    fig, axes = plt.subplots(rows, cols, figsize=figsize, sharey=False)
    axes = np.atleast_1d(axes).flatten()

    raw_only = raw_df[["scenario", "rmse"]].rename(columns={"rmse": "raw_rmse"})
    merged = method_df.merge(raw_only, on="scenario")
    merged["relative_rmse"] = merged["rmse"] / merged["raw_rmse"]

    for idx, sid in enumerate(scenarios):
        ax = axes[idx]
        sub = merged[merged["scenario"] == sid]
        for i, method in enumerate(METHOD_ORDER):
            val = sub[sub["method"] == method]["relative_rmse"]
            if val.empty:
                continue
            ax.bar(
                i,
                val.values[0],
                color=METHOD_COLORS[method],
                edgecolor="black",
                linewidth=0.5,
            )
        ax.axhline(1.0, color=METHOD_COLORS["RAW"], linestyle="--", linewidth=2)
        ax.set_title(scenario_label(sid), fontsize=10)
        ax.set_xticks(range(len(METHOD_ORDER)))
        ax.set_xticklabels(METHOD_ORDER, rotation=30, ha="right", fontsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.set_axisbelow(True)

    for idx in range(len(scenarios), len(axes)):
        axes[idx].set_visible(False)

    handles, labels = _legend_handles()
    fig.legend(handles, labels, loc="upper right", ncol=4, title="Method", fontsize=10)
    fig.suptitle("Relative RMSE per synthetic scenario (S1–S10)\n<1 means better than RAW", fontsize=14)
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved relative RMSE subplots: {output_path}")


def plot_rmse_bars(method_df, raw_df, output_path, figsize=(14, 6)):
    """Grouped bar chart of RMSE per scenario with RAW as dashed line."""
    scenarios = sorted(method_df["scenario"].unique(), key=lambda s: int(s[1:]))
    method_df = method_df[method_df["scenario"].isin(scenarios)].copy()
    raw_df = raw_df[raw_df["scenario"].isin(scenarios)].copy()

    labels = [scenario_label(s) for s in scenarios]
    x = np.arange(len(scenarios))
    n_methods = len(METHOD_ORDER)
    width = 0.75 / n_methods

    fig, ax = plt.subplots(figsize=figsize)

    for i, method in enumerate(METHOD_ORDER):
        subset = method_df[method_df["method"] == method].set_index("scenario")["rmse"].reindex(scenarios)
        offset = (i - (n_methods - 1) / 2) * width
        ax.bar(
            x + offset,
            subset.values,
            width,
            label=method,
            color=METHOD_COLORS.get(method),
            edgecolor="black",
            linewidth=0.5,
        )

    raw_vals = raw_df.set_index("scenario").loc[scenarios, "rmse"].values
    ax.plot(
        x,
        raw_vals,
        color=METHOD_COLORS["RAW"],
        linestyle="--",
        marker="o",
        linewidth=2,
        markersize=7,
        label="RAW (GT)",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("RMSE", fontsize=12)
    ax.set_title("RMSE across synthetic scenarios (S1–S10)\nRAW = ground truth (dashed line)", fontsize=13)
    ax.legend(title="Method", loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved bar chart: {output_path}")


def plot_relative_rmse(method_df, raw_df, output_path, figsize=(14, 6)):
    """Optional: relative RMSE vs RAW (RAW = 1.0) grouped bar chart."""
    scenarios = sorted(method_df["scenario"].unique(), key=lambda s: int(s[1:]))
    raw_only = raw_df[["scenario", "rmse"]].rename(columns={"rmse": "raw_rmse"})
    merged = method_df.merge(raw_only, on="scenario")
    merged["relative_rmse"] = merged["rmse"] / merged["raw_rmse"]

    labels = [scenario_label(s) for s in scenarios]
    x = np.arange(len(scenarios))
    n_methods = len(METHOD_ORDER)
    width = 0.75 / n_methods

    fig, ax = plt.subplots(figsize=figsize)
    for i, method in enumerate(METHOD_ORDER):
        subset = merged[merged["method"] == method].set_index("scenario")["relative_rmse"].reindex(scenarios)
        offset = (i - (n_methods - 1) / 2) * width
        ax.bar(
            x + offset,
            subset.values,
            width,
            label=method,
            color=METHOD_COLORS.get(method),
            edgecolor="black",
            linewidth=0.5,
        )

    ax.axhline(1.0, color=METHOD_COLORS["RAW"], linestyle="--", linewidth=2, label="RAW (GT)")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Relative RMSE (method / RAW)", fontsize=12)
    ax.set_title("Relative RMSE across synthetic scenarios (S1–S10)\n<1 means better than RAW", fontsize=13)
    ax.legend(title="Method", loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved relative RMSE chart: {output_path}")


def print_summary(method_df, raw_df):
    """Print a concise RMSE table to stdout."""
    raw_only = raw_df[["scenario", "rmse"]].rename(columns={"rmse": "raw_rmse"})
    merged = method_df.merge(raw_only, on="scenario")
    merged["reduction_pct"] = (merged["raw_rmse"] - merged["rmse"]) / merged["raw_rmse"] * 100
    pivot = merged.pivot(index="scenario", columns="method", values=["rmse", "reduction_pct"])
    pivot = pivot.loc[sorted(pivot.index, key=lambda s: int(s[1:]))]
    print("\nRMSE summary (excluding SpatialSoupX):")
    print(pivot["rmse"].round(3).to_string())
    print("\nReduction vs RAW (%):")
    print(pivot["reduction_pct"].round(2).to_string())


def main():
    parser = argparse.ArgumentParser(
        description="Visualize synthetic S1-S10 RMSE across correction methods."
    )
    parser.add_argument(
        "--metrics-dir",
        type=str,
        default="evaluation/reports/metrics",
        help="Directory containing synthetic_S*_metrics.json files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation/reports/synthetic_figures",
        help="Output directory for figures",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    method_df, raw_df = load_metrics(args.metrics_dir)
    if method_df.empty:
        raise FileNotFoundError(f"No synthetic_S*_metrics.json found in {args.metrics_dir}")

    print_summary(method_df, raw_df)

    plot_rmse_bars(
        method_df,
        raw_df,
        out_dir / "synthetic_rmse_comparison.png",
    )
    plot_relative_rmse(
        method_df,
        raw_df,
        out_dir / "synthetic_relative_rmse_comparison.png",
    )
    plot_rmse_subplots(
        method_df,
        raw_df,
        out_dir / "synthetic_rmse_subplots.png",
    )
    plot_relative_rmse_subplots(
        method_df,
        raw_df,
        out_dir / "synthetic_relative_rmse_subplots.png",
    )


if __name__ == "__main__":
    main()
