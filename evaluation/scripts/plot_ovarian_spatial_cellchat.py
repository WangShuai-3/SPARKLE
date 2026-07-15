#!/usr/bin/env python3
"""Render corrected ovarian Spatial CellChat v2 Figures 05 and 05b.

Chart contract
--------------
Question: how do aggregate and interaction-level communication probabilities
change after ambient-RNA correction?
Takeaway: SPARKLE lowers aggregate strength while retaining most interactions;
the repaired DecontX analysis is reduced but non-zero.
Surface: standalone Matplotlib PNG/PDF/SVG exports.
Grain: one aggregate estimate per method/metric in Figure 05, and the shared
union of source-target-ligand-receptor keys in Figure 05b.
Palette: fixed method identity colors plus neutral connectors/keylines.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "evaluation" / "reports" / "ovarian_eval" / "cellchat_spatial"
DEFAULT_OUTPUT = PROJECT_ROOT / "evaluation" / "reports" / "ovarian_eval" / "figures"
METHOD_ORDER = ("RAW", "SoupX", "DecontX", "SPARKLE")
METHOD_COLORS = {
    "RAW": "#80858c",
    "SoupX": "#bc8e36",
    "DecontX": "#457f78",
    "SPARKLE": "#1c456e",
}
INTERACTION_KEYS = ["source", "target", "ligand", "receptor"]


def _read_required(path: Path, columns: Iterable[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Required CellChat result is absent: {path}")
    frame = pd.read_csv(path)
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{path.name} is missing columns: {missing}")
    return frame


def prepare_spatial_cellchat_data(input_dir: Path) -> dict[str, pd.DataFrame]:
    """Load validated saved results and construct matched plotting units."""
    summary = _read_required(
        input_dir / "cellchat_spatial_summary.csv",
        ["Method", "total_prob", "autocrine_prob", "tumor_involving_prob"],
    ).rename(columns={"Method": "method"})
    summary = summary.loc[summary["method"].isin(METHOD_ORDER)].copy()
    if set(summary["method"]) != set(METHOD_ORDER):
        raise ValueError("Spatial summary does not contain all four plotting methods")

    indexed = summary.set_index("method")
    raw = indexed.loc["RAW"]
    sparkle = indexed.loc["SPARKLE"]
    metric_columns = (
        ("Total probability", "total_prob"),
        ("Autocrine probability", "autocrine_prob"),
        ("Tumour-involving probability", "tumor_involving_prob"),
    )
    effects = pd.DataFrame(
        [
            {
                "metric": label,
                "RAW": float(raw[column]),
                "SPARKLE": float(sparkle[column]),
                "reduction_pct": 100.0
                * (1.0 - float(sparkle[column]) / float(raw[column])),
            }
            for label, column in metric_columns
        ]
    )

    observed_frames = []
    for method in METHOD_ORDER:
        frame = _read_required(
            input_dir / f"{method}_cellchat_spatial.csv",
            INTERACTION_KEYS + ["prob", "pval"],
        ).copy()
        if frame.empty:
            raise ValueError(
                f"{method} interaction table is empty; the result is non-estimable"
            )
        if frame.duplicated(INTERACTION_KEYS).any():
            raise ValueError(f"{method} interaction keys are not unique")
        if not np.isfinite(frame["prob"].to_numpy(float)).all():
            raise ValueError(f"{method} interaction probabilities contain NaN/Inf")
        frame.insert(0, "method", method)
        observed_frames.append(frame)

    observed = pd.concat(observed_frames, ignore_index=True)
    interaction_union = observed[INTERACTION_KEYS].drop_duplicates().reset_index(drop=True)
    matched_frames = []
    for method in METHOD_ORDER:
        values = observed.loc[
            observed["method"] == method, INTERACTION_KEYS + ["prob"]
        ]
        matched = interaction_union.merge(
            values, on=INTERACTION_KEYS, how="left", validate="one_to_one"
        )
        matched["prob"] = matched["prob"].fillna(0.0).astype(float)
        matched.insert(0, "method", method)
        matched_frames.append(matched)
    matched = pd.concat(matched_frames, ignore_index=True)

    interaction_summary = (
        matched.groupby("method", sort=False)["prob"]
        .agg(n="size", mean="mean", std="std")
        .reset_index()
    )
    interaction_summary["sem"] = (
        interaction_summary["std"] / np.sqrt(interaction_summary["n"])
    )
    return {
        "summary": summary,
        "effects": effects,
        "interaction_union": matched,
        "interaction_summary": interaction_summary,
    }


def _clean_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", width=0.8)


def figure_05(data: dict[str, pd.DataFrame]) -> plt.Figure:
    """Aggregate RAW-to-SPARKLE effects with all-method context."""
    fig = plt.figure(figsize=(7.2, 3.15))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.1, 1.65])
    left = fig.add_subplot(grid[0, 0])
    right_grid = grid[0, 1].subgridspec(1, 3, wspace=0.42)
    right_axes = [fig.add_subplot(right_grid[0, i]) for i in range(3)]

    effects = data["effects"]
    y = np.arange(len(effects))
    for yi, row in effects.iterrows():
        left.plot(
            [row["SPARKLE"], row["RAW"]], [yi, yi],
            color="#B9BEC3", linewidth=1.2, zorder=1,
        )
        left.scatter(row["RAW"], yi, s=26, color=METHOD_COLORS["RAW"],
                     edgecolor="white", linewidth=0.45, zorder=3)
        left.scatter(row["SPARKLE"], yi, s=30, color=METHOD_COLORS["SPARKLE"],
                     edgecolor="white", linewidth=0.45, zorder=3)
        left.text(
            row["RAW"] + 0.035, yi, f"−{row['reduction_pct']:.1f}%",
            va="center", fontsize=5.6, color=METHOD_COLORS["SPARKLE"],
        )
    left.set_yticks(y, [value.replace(" probability", "") for value in effects["metric"]])
    left.invert_yaxis()
    left.set_xlim(0, max(1.18, float(effects["RAW"].max()) * 1.15))
    left.set_xlabel("Aggregate spatial CellChat probability")
    left.set_title("RAW → SPARKLE", loc="left", pad=5)
    _clean_axis(left)
    left.text(-0.22, 1.04, "a", transform=left.transAxes, fontweight="bold")

    summary = data["summary"].set_index("method").loc[list(METHOD_ORDER)]
    for idx, (axis, column, title) in enumerate(zip(
        right_axes,
        ("total_prob", "autocrine_prob", "tumor_involving_prob"),
        ("Total", "Autocrine", "Tumour-involving"),
    )):
        values = summary[column].to_numpy(float)
        x = np.arange(len(METHOD_ORDER))
        axis.bar(x, values, color=[METHOD_COLORS[m] for m in METHOD_ORDER],
                 width=0.72, edgecolor="none")
        axis.set_xticks(x, METHOD_ORDER, rotation=55, ha="right")
        axis.tick_params(axis="x", labelsize=4.7, pad=1)
        axis.set_title(title, fontsize=6.3, pad=4)
        axis.set_ylim(0, float(values.max()) * 1.10)
        if idx == 0:
            axis.set_ylabel("Probability")
            axis.text(-0.27, 1.04, "b", transform=axis.transAxes, fontweight="bold")
        _clean_axis(axis)
    fig.text(0.58, 0.785, "All-method context (independent aggregate estimates)",
             ha="center", fontsize=6.4, color="#4D5257")
    fig.suptitle("Spatial CellChat v2 communication probability", x=0.06,
                 ha="left", fontsize=8.4, fontweight="bold")
    fig.subplots_adjust(left=0.13, right=0.99, bottom=0.30, top=0.74, wspace=0.50)
    return fig


def figure_05b(data: dict[str, pd.DataFrame]) -> plt.Figure:
    """Matched interaction-level distributions with dynamic sample-size text."""
    matched = data["interaction_union"]
    summary = data["interaction_summary"].set_index("method").loc[list(METHOD_ORDER)]
    values = [
        matched.loc[matched["method"] == method, "prob"].to_numpy(float) * 1e4
        for method in METHOD_ORDER
    ]
    n_interactions = len(values[0])
    if len({len(method_values) for method_values in values}) != 1:
        raise ValueError("Methods do not have the same matched interaction count")

    positions = np.arange(len(METHOD_ORDER), dtype=float) * 0.62
    fig, axis = plt.subplots(figsize=(3.85, 3.10))
    boxes = axis.boxplot(
        values, positions=positions, widths=0.34, patch_artist=True,
        showfliers=False,
        medianprops={"color": "#24272A", "linewidth": 0.8},
        whiskerprops={"color": "#555A5F", "linewidth": 0.6},
        capprops={"color": "#555A5F", "linewidth": 0.6},
        boxprops={"linewidth": 0.7, "edgecolor": "#555A5F"},
    )
    for patch, method in zip(boxes["boxes"], METHOD_ORDER):
        patch.set_facecolor(METHOD_COLORS[method])
        patch.set_alpha(0.55)
    for position, method in zip(positions, METHOD_ORDER):
        axis.errorbar(
            position,
            float(summary.loc[method, "mean"] * 1e4),
            yerr=float(summary.loc[method, "sem"] * 1e4),
            fmt="D", ms=3.2, mfc=METHOD_COLORS[method], mec="white", mew=0.4,
            ecolor="#303438", elinewidth=0.7, capsize=1.8, zorder=4,
        )
    axis.set_xticks(positions, METHOD_ORDER, rotation=28, ha="right")
    axis.set_xlim(positions[0] - 0.30, positions[-1] + 0.30)
    quartiles = np.concatenate([
        np.quantile(method_values, [0.25, 0.75]) for method_values in values
    ])
    axis.set_ylim(0, max(4.0, float(np.max(quartiles)) * 2.8))
    axis.set_ylabel("Spatial CellChat v2 probability (×10^4)")
    axis.set_title("Average cell–cell communication strength", loc="left",
                   pad=22, fontweight="bold")
    axis.text(
        0, 1.035,
        f"{n_interactions:,} matched interactions; absent = 0; outliers not drawn",
        transform=axis.transAxes, fontsize=5.3, color="#666B70",
    )
    _clean_axis(axis)
    fig.subplots_adjust(left=0.18, right=0.97, bottom=0.22, top=0.76)
    return fig


def save_figure(figure: plt.Figure, stem: Path, formats: Iterable[str], dpi: int) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    for extension in formats:
        figure.savefig(stem.with_suffix(f".{extension}"), dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--formats", nargs="+", default=["png", "pdf", "svg"])
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()

    prepared = prepare_spatial_cellchat_data(args.input_dir)
    save_figure(figure_05(prepared), args.output_dir / "fig05_spatial_cellchat_v2",
                args.formats, args.dpi)
    save_figure(figure_05b(prepared), args.output_dir / "fig05b_cellchatv2_average_strength",
                args.formats, args.dpi)
    n = int(prepared["interaction_summary"]["n"].iloc[0])
    print(f"Rendered corrected ovarian Spatial CellChat figures with {n:,} matched interactions")


if __name__ == "__main__":
    main()
