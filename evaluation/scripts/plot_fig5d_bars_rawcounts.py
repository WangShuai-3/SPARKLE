#!/usr/bin/env python3
"""Fig 5d 候选图(柱状版): 原始计数口径的五方法三指标 + 总文库保留。

五组指标 × 五方法:
  Tumor ectopic removal / Tumor source retention /
  Stromal ectopic removal / Stromal source retention / Library size retention

输出: Report_V2/Figure5_ovarian/marker_analysis/fig5d_bars_rawcounts.{png,pdf}
用法: cd evaluation && python scripts/plot_fig5d_bars_rawcounts.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "Report_V2", "Figure5_ovarian",
                   "marker_analysis")

METHODS = ["RAW", "SoupX", "DecontX", "SpotClean", "SPARKLE"]
MC = {"RAW": "#80858c", "SoupX": "#bc8e36", "DecontX": "#457f78",
      "SpotClean": "#8f6aa8", "SPARKLE": "#1c456e"}

# 原始计数均值(marker_retention_removal_rawcounts_summary.csv)+ 文库保留比
DATA = {
    "Tumor markers (n=17)\nectopic removal":
        [0.000, 0.962, 0.234, 0.154, 0.662],
    "Tumor markers (n=17)\nsource retention":
        [1.000, 0.119, 0.892, 1.133, 0.397],
    "Stromal markers (n=5)\nectopic removal":
        [0.000, 0.768, 0.138, -0.064, 0.482],
    "Stromal markers (n=5)\nsource retention":
        [1.000, 0.581, 0.963, 1.256, 0.654],
    "Total library\nsize retention":
        [1.000, 0.081, 0.889, 1.125, 0.545],
}


def main():
    groups = list(DATA.keys())
    n_g, n_m = len(groups), len(METHODS)
    w = 0.8 / n_m
    x = np.arange(n_g)

    fig, ax = plt.subplots(figsize=(8.6, 3.8))
    for j, m in enumerate(METHODS):
        vals = [DATA[g][j] for g in groups]
        xs = x + (j - (n_m - 1) / 2) * w
        ax.bar(xs, vals, width=w * 0.92, color=MC[m], label=m, zorder=3)
        for xi, v in zip(xs, vals):
            ax.text(xi, v + (0.03 if v >= 0 else -0.06), f"{v:.2f}",
                    ha="center", va="bottom" if v >= 0 else "top",
                    fontsize=6.3, color="0.25")

    ax.axhline(0.0, lw=0.8, color="0.3", zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=8.5)
    ax.set_ylabel("Rate (vs RAW)", fontsize=9)
    ax.set_ylim(-0.22, 1.38)
    ax.legend(fontsize=8, frameon=False, ncol=5, loc="upper center",
              bbox_to_anchor=(0.5, 1.14))
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        path = os.path.join(OUT, f"fig5d_bars_rawcounts.{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print("Saved:", path)


if __name__ == "__main__":
    main()
