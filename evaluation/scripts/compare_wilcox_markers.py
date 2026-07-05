#!/usr/bin/env python3
"""Compare spatial Wilcoxon markers (RAW vs SPARKLE) with snRNA reference markers.

Outputs summary tables for different min_cells and top-N thresholds.
"""

import argparse
from pathlib import Path

import pandas as pd


def load_markers(path):
    df = pd.read_csv(path)
    # standardize names
    rename = {}
    if "p_val_adj" in df.columns:
        rename["p_val_adj"] = "padj"
    if "pval_adj" in df.columns:
        rename["pval_adj"] = "padj"
    if "avg_log2FC" in df.columns:
        rename["avg_log2FC"] = "logFC"
    return df.rename(columns=rename)


def top_n_markers(df, n):
    return (
        df.groupby("cluster")
        .apply(lambda x: set(x.sort_values("padj").head(n)["gene"]), include_groups=False)
        .to_dict()
    )


def sig_markers(df, p_cut=0.05, lfc_cut=0):
    sig = df[(df["padj"] < p_cut) & (df["logFC"] > lfc_cut)]
    return sig.groupby("cluster").apply(lambda x: set(x["gene"]), include_groups=False).to_dict()


def compare(snrna_sig, raw_top, sp_top):
    shared = sorted(set(snrna_sig.keys()) & set(raw_top.keys()) & set(sp_top.keys()))
    rows = []
    for g in shared:
        a = snrna_sig[g]
        b_raw = raw_top[g]
        b_sp = sp_top[g]
        inter_raw = len(a & b_raw)
        inter_sp = len(a & b_sp)
        rows.append(
            {
                "group": g,
                "snrna_n": len(a),
                "raw_n": len(b_raw),
                "sp_n": len(b_sp),
                "raw_shared": inter_raw,
                "sp_shared": inter_sp,
                "raw_precision": inter_raw / len(b_raw) if b_raw else 0,
                "sp_precision": inter_sp / len(b_sp) if b_sp else 0,
                "raw_recall": inter_raw / len(a) if a else 0,
                "sp_recall": inter_sp / len(a) if a else 0,
                "raw_jaccard": inter_raw / len(a | b_raw) if (a | b_raw) else 0,
                "sp_jaccard": inter_sp / len(a | b_sp) if (a | b_sp) else 0,
            }
        )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="evaluation/reports/mousebrain_eval")
    parser.add_argument("--snrna-file", type=str, default="snrna_wilcox_markers_sparkle_genes.csv",
                        help="Filename of snRNA marker CSV inside output-dir/gene_level_analysis")
    parser.add_argument("--suffix", type=str, default="",
                        help="Suffix appended to spatial marker filenames (e.g. '_sct')")
    parser.add_argument("--min-cells-list", type=int, nargs="+", default=[10, 20])
    parser.add_argument("--top-n-list", type=int, nargs="+", default=[20, 50, 100])
    args = parser.parse_args()

    out_dir = Path(args.output_dir) / "gene_level_analysis"
    snrna = load_markers(out_dir / args.snrna_file)
    snrna_sig = sig_markers(snrna)

    summary_rows = []
    detail_list = []

    for mc in args.min_cells_list:
        raw = load_markers(
            out_dir / f"{args.tag}_RAW_wilcox_markers_min{mc}_sparkle_genes{args.suffix}.csv"
        )
        sp = load_markers(
            out_dir / f"{args.tag}_SPARKLE_wilcox_markers_min{mc}_sparkle_genes{args.suffix}.csv"
        )

        for n in args.top_n_list:
            raw_top = top_n_markers(raw, n)
            sp_top = top_n_markers(sp, n)
            df = compare(snrna_sig, raw_top, sp_top)

            if len(df) == 0:
                continue

            summary_rows.append(
                {
                    "min_cells": mc,
                    "top_n": n,
                    "n_groups": len(df),
                    "raw_recall_mean": df["raw_recall"].mean(),
                    "sp_recall_mean": df["sp_recall"].mean(),
                    "delta_recall": (df["sp_recall"] - df["raw_recall"]).mean(),
                    "raw_precision_mean": df["raw_precision"].mean(),
                    "sp_precision_mean": df["sp_precision"].mean(),
                    "delta_precision": (df["sp_precision"] - df["raw_precision"]).mean(),
                    "raw_jaccard_mean": df["raw_jaccard"].mean(),
                    "sp_jaccard_mean": df["sp_jaccard"].mean(),
                    "delta_jaccard": (df["sp_jaccard"] - df["raw_jaccard"]).mean(),
                    "groups_improved": (df["sp_recall"] > df["raw_recall"]).sum(),
                    "groups_worsened": (df["sp_recall"] < df["raw_recall"]).sum(),
                }
            )

            # Save detail for the first valid (min_cells, top_n) combination
            if "detail" not in dir():
                detail = df.copy()

    summary = pd.DataFrame(summary_rows)
    summary_path = out_dir / f"{args.tag}_wilcox_marker_overlap{args.suffix}_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved summary: {summary_path}")
    print("\n=== Summary ===")
    print(summary.to_markdown(index=False))

    if "detail" in dir():
        detail_path = out_dir / f"{args.tag}_wilcox_marker_overlap{args.suffix}_detail.csv"
        detail.to_csv(detail_path, index=False)
        print("\n=== Detail (first valid min_cells/top_n) ===")
        print(detail.to_markdown(index=False))


if __name__ == "__main__":
    main()
