#!/usr/bin/env python3
"""Plot Figure 4C using cells retained by all five MouseBrain methods."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD_ORDER = ["RAW", "SoupX", "DecontX", "SpotClean", "SPARKLE"]
METHOD_COLORS = {
    "RAW": "#80858c",
    "SoupX": "#bc8e36",
    "DecontX": "#457f78",
    "SpotClean": "#8f6aa8",
    "SPARKLE": "#1c456e",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path(
            "evaluation/reports/rctd_mousebrain/"
            "rctd_Cell_group_all_methods_shared_metrics.csv"
        ),
    )
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=Path(
            "evaluation/reports/rctd_mousebrain/"
            "figure4c_rctd_singlet_shared"
        ),
    )
    return parser.parse_args()


def load_metrics(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"method", "n_cells", "n_singlet", "pct_singlet"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
    if frame["method"].duplicated().any():
        raise ValueError("Expected one metrics row per method")
    observed = set(frame["method"])
    if observed != set(METHOD_ORDER):
        raise ValueError(
            f"Expected methods {METHOD_ORDER}, observed {sorted(observed)}"
        )
    frame = frame.set_index("method").loc[METHOD_ORDER].reset_index()
    if frame["n_cells"].nunique() != 1:
        raise ValueError("Figure 4C requires the same denominator for every method")
    denominator = int(frame["n_cells"].iloc[0])
    if denominator < 1000:
        raise ValueError(f"Shared-cell denominator looks wrong: {denominator:,}")
    expected_pct = frame["n_singlet"] / frame["n_cells"] * 100
    if not np.allclose(frame["pct_singlet"], expected_pct, atol=1e-10):
        raise ValueError("pct_singlet does not match n_singlet / n_cells")
    return frame


def plot(frame: pd.DataFrame, output_stem: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    x = np.arange(len(frame))
    values = frame["pct_singlet"].to_numpy(float)

    fig, ax = plt.subplots(figsize=(3.0, 2.65))
    ax.bar(
        x,
        values,
        width=0.64,
        color=[METHOD_COLORS[m] for m in frame["method"]],
        edgecolor="#20252b",
        linewidth=0.45,
        zorder=3,
    )
    for position, value in zip(x, values):
        ax.text(
            position,
            value + 1.0,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=6.8,
            color="#20252b",
        )

    ax.set_ylim(0, 50)
    ax.set_ylabel("Cells classified as singlet (%)")
    ax.set_xticks(x, frame["method"], rotation=35, ha="right")
    ax.set_title("Cell_group; shared cells, n = 9,190", loc="left", fontsize=7)
    ax.text(
        -0.16,
        1.12,
        "C",
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
    )
    ax.grid(axis="y", color="#d9dde1", linewidth=0.5, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=2.5, width=0.6)
    fig.subplots_adjust(left=0.25, right=0.98, bottom=0.25, top=0.86)

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    frame = load_metrics(args.metrics)
    plot(frame, args.output_stem)
    print(f"Saved {args.output_stem.with_suffix('.pdf')}")
    print(f"Saved {args.output_stem.with_suffix('.png')}")


if __name__ == "__main__":
    main()
