#!/usr/bin/env python3
"""Top-10 marker genes improved by SPARKLE (corrected-HVG data, cell_group).

Selection:
  1. read the corrected-HVG per_gene_improvement.csv
  2. compute per-gene marker properties from the snRNA reference
     (target type = highest snRNA expression; specificity = max / sum)
  3. keep genes that are reference-derived markers:
       specificity >= SPEC_THR  AND  max_expr_ref >= EXPR_THR
  4. keep marker genes with raw_corr > 0 and improvement > 0
  5. top-10 by improvement, displayed sorted by final sparkle_corr
  6. horizontal stacked bars: grey = RAW correlation, dark blue = SPARKLE improvement
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
    parser.add_argument("--input", required=True, help="corrected-HVG per_gene_improvement.csv")
    parser.add_argument("--snrna-ref", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--title", default="Top-10 marker genes by SPARKLE improvement (corrected-HVG)")
    parser.add_argument("--spec-thr", type=float, default=0.15,
                        help="Min snRNA specificity (max/sum) to call a marker (default 0.15)")
    parser.add_argument("--expr-thr", type=float, default=1.0,
                        help="Min snRNA max expression to call a marker (default 1.0)")
    parser.add_argument("--top-n", type=int, default=10)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    ref = pd.read_csv(args.snrna_ref, index_col=0)

    # 2. marker properties from snRNA reference
    rows = []
    for g in df["gene"]:
        if g not in ref.index:
            continue
        v = ref.loc[g]
        rows.append({"gene": g, "target_type": v.idxmax(),
                     "max_expr_ref": float(v.max()),
                     "specificity": float(v.max() / v.sum())})
    spec_df = pd.DataFrame(rows)
    df = df.merge(spec_df, on="gene", how="inner")

    # 3. marker filter
    mk = df[(df["specificity"] >= args.spec_thr) &
            (df["max_expr_ref"] >= args.expr_thr)].copy()
    print(f"Marker genes (spec>={args.spec_thr}, expr>={args.expr_thr}): {len(mk)}")

    # 4. raw_corr>0 & improvement>0
    mk = mk[(mk["raw_corr"] > 0) & (mk["improvement"] > 0)].copy()
    print(f"  with raw_corr>0 & imp>0: {len(mk)}")

    # 5. top-N by improvement, display by sparkle_corr descending
    top = mk.sort_values("improvement", ascending=False).head(args.top_n).copy()
    top = top.sort_values("sparkle_corr", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 6.5))
    y_pos = np.arange(len(top))
    ax.barh(y_pos, top["raw_corr"], color="#9aa5b1", label="RAW correlation")
    ax.barh(y_pos, top["improvement"], left=top["raw_corr"],
            color="#1f4e79", label="SPARKLE improvement")
    labels = [f"{g} ({t})" for g, t in zip(top["gene"], top["target_type"])]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Pearson correlation with snRNA reference")
    ax.set_title(args.title)
    ax.legend(loc="lower right")
    ax.axvline(0, color="gray", linewidth=0.5)
    for i, (raw, sp, imp) in enumerate(zip(top["raw_corr"], top["sparkle_corr"], top["improvement"])):
        ax.text(sp + 0.01, i, f"+{imp:.3f}", va="center", fontsize=8)
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")
    print(top[["gene", "target_type", "raw_corr", "sparkle_corr", "improvement"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
