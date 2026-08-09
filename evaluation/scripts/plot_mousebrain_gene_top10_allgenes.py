#!/usr/bin/env python3
"""Figure: Top-10 gene-level SPARKLE improvements from the ALL-GENES data.

Selection (as requested):
  1. read the full per_gene_improvement.csv
  2. keep genes with raw_corr > 0
  3. keep genes with sparkle_corr > raw_corr (improvement > 0)
  4. take the Top-10 by improvement
  5. display sorted by final sparkle_corr (descending)
  6. horizontal stacked bars: grey = RAW correlation, dark blue = SPARKLE improvement

Usage:
    python evaluation/scripts/plot_mousebrain_gene_top10_allgenes.py \
        --input evaluation/reports/mousebrain_eval_cellgroup_v2/gene_level/mousebrain_x12500-20000_y2000-10000_per_gene_improvement.csv \
        --out  evaluation/reports/mousebrain_eval_cellgroup_v2/gene_level/mousebrain_x12500-20000_y2000-10000_gene_top10_allgenes.png \
        --title "Top-10 genes (all genes, cell_group)"
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="per_gene_improvement.csv (all genes)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--title", default="Top-10 genes by SPARKLE improvement (all genes)")
    parser.add_argument("--top-n", type=int, default=10)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    print(f"Read {len(df)} genes")

    # 2. raw_corr > 0
    sub = df[df["raw_corr"] > 0].copy()
    print(f"raw_corr>0: {len(sub)}")

    # 3. sparkle_corr > raw_corr (positive improvement)
    sub = sub[sub["improvement"] > 0].copy()
    print(f"improvement>0: {len(sub)}")

    # 4. top-N by improvement
    sub = sub.sort_values("improvement", ascending=False).head(args.top_n).copy()

    # 5. display sorted by sparkle_corr descending (barh shows bottom->top, so ascending)
    sub = sub.sort_values("sparkle_corr", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    y_pos = np.arange(len(sub))

    # 6. stacked horizontal bars: grey = RAW, dark blue = improvement
    ax.barh(y_pos, sub["raw_corr"], color="#9aa5b1", label="RAW correlation")
    ax.barh(y_pos, sub["improvement"], left=sub["raw_corr"],
            color="#1f4e79", label="SPARKLE improvement")

    labels = [f"{g}" for g in sub["gene"]]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Pearson correlation with snRNA reference")
    ax.set_title(args.title)
    ax.legend(loc="lower right")
    ax.axvline(0, color="gray", linewidth=0.5)

    for i, (raw, sp, imp) in enumerate(zip(sub["raw_corr"], sub["sparkle_corr"], sub["improvement"])):
        ax.text(sp + 0.01, i, f"+{imp:.3f}", va="center", fontsize=8)

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")
    print(sub[["gene", "raw_corr", "sparkle_corr", "improvement"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
