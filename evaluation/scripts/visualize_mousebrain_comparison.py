#!/usr/bin/env python3
"""Visualize MouseBrain method comparison across normalization scenarios.

Produces four figures:
    1. HVG-3000 cell-type pseudobulk correlation heatmaps with a unified
       colorbar.
    2. Per-cell-type snRNA correlation distributions across scenarios
       (boxplots).
    3. Top-10 per-cell-type SPARKLE improvements (horizontal stacked bar).
    4. Top-10 gene-level SPARKLE improvements (horizontal stacked bar).
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns


SCENARIOS = {
    "log1p-CPM": {
        "method_dir": "method_level_new",
        "gene_dir": "gene_level_analysis_new/gene_level_analysis",
    },
    "HVG-3000": {
        "method_dir": "method_level_hvg3000",
        "gene_dir": "gene_level_analysis_hvg3000/gene_level_analysis",
    },
    "SCTransform": {
        "method_dir": "method_level_sct",
        "gene_dir": "gene_level_analysis_sct/gene_level_analysis",
    },
}

METHODS = ["raw", "SPARKLE", "SoupX", "DecontX", "SpotClean"]


def normalize_adata(adata):
    adata = adata.copy()
    if hasattr(adata.X, "toarray"):
        adata.X = adata.X.toarray()
    adata.X = np.maximum(np.asarray(adata.X, dtype=np.float64), 0.0)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    return adata


def compute_pseudobulk(adata, group_key="annotation", min_cells=3):
    groups = (
        adata.obs[group_key].astype(object).fillna("Unknown").astype(str)
    )
    valid = ~groups.isin(["Unknown", "nan", "None", ""])
    adata = adata[valid].copy()
    groups = groups.loc[adata.obs_names]
    group_names = sorted(groups.unique())
    expr = adata.X
    if hasattr(expr, "toarray"):
        expr = expr.toarray()
    expr = np.asarray(expr)
    pb = {}
    for g in group_names:
        mask = groups == g
        if mask.sum() < min_cells:
            continue
        pb[g] = expr[mask].mean(axis=0)
    return pd.DataFrame(pb, index=adata.var_names)


def get_hvg_gene_set(raw_path, n_hvgs):
    adata = sc.read_h5ad(raw_path)
    adata = normalize_adata(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_hvgs, flavor="seurat")
    return adata.var_names[adata.var["highly_variable"].values].tolist()


def load_method_corr(tag, h5ad_dir, method, hvg_genes=None):
    suffix_map = {"raw": "raw", "SpotClean": "SpotCleanOfficial"}
    suffix = suffix_map.get(method, method)
    path = Path(h5ad_dir) / f"{tag}_{suffix}.h5ad"
    adata = sc.read_h5ad(path)
    adata = normalize_adata(adata)
    if hvg_genes is not None:
        shared = adata.var_names.intersection(hvg_genes)
        adata = adata[:, shared].copy()
    pb = compute_pseudobulk(adata, min_cells=3)
    return pb.corr(method="pearson")


def plot_hvg_between_corr_heatmaps(tag, h5ad_dir, output_path, n_hvgs=3000):
    """Figure 1: cell-type pseudobulk correlation heatmaps (HVG-3000)."""
    raw_path = Path(h5ad_dir) / f"{tag}_raw.h5ad"
    hvg_genes = get_hvg_gene_set(raw_path, n_hvgs)

    corr_mats = {}
    for method in METHODS:
        try:
            corr_mats[method] = load_method_corr(tag, h5ad_dir, method, hvg_genes)
        except Exception as e:
            print(f"Skipping {method} for HVG heatmap: {e}")

    if not corr_mats:
        return

    # unified color scale
    all_vals = np.concatenate([c.values.ravel() for c in corr_mats.values()])
    vmin, vmax = np.nanmin(all_vals), np.nanmax(all_vals)

    n = len(corr_mats)
    cols = 2
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4.5))
    axes = np.atleast_1d(axes).flatten()

    for ax, (method, corr) in zip(axes, corr_mats.items()):
        sns.heatmap(
            corr,
            ax=ax,
            vmin=vmin,
            vmax=vmax,
            cmap="RdBu_r",
            center=0,
            square=True,
            linewidths=0.5,
            cbar=False,
            xticklabels=True,
            yticklabels=True,
        )
        ax.set_title(method, fontsize=12)
        ax.tick_params(axis="x", rotation=90, labelsize=6)
        ax.tick_params(axis="y", rotation=0, labelsize=6)

    # hide unused axes
    for ax in axes[len(corr_mats):]:
        ax.axis("off")

    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax, label="Pearson correlation")

    fig.suptitle(f"HVG-{n_hvgs} cell-type pseudobulk correlation", fontsize=14, y=1.02)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_snrna_corr_boxplots(tag, reports_dir, output_path):
    """Figure 2: per-cell-type snRNA correlation distributions across scenarios."""
    frames = []
    for scenario, dirs in SCENARIOS.items():
        csv_path = Path(reports_dir) / dirs["method_dir"] / f"{tag}_snrna_corr_per_celltype.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        df = df[df["method"].isin(METHODS)].copy()
        df["scenario"] = scenario
        frames.append(df)

    if not frames:
        return
    df = pd.concat(frames, ignore_index=True)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
    scenario_order = ["log1p-CPM", "HVG-3000", "SCTransform"]
    for ax, scenario in zip(axes, scenario_order):
        sub = df[df["scenario"] == scenario]
        if sub.empty:
            ax.axis("off")
            continue
        sns.boxplot(
            data=sub,
            x="method",
            y="corr",
            order=METHODS,
            hue="method",
            palette="Set2",
            legend=False,
            ax=ax,
        )
        ax.set_title(scenario, fontsize=12)
        ax.set_xlabel("")
        ax.set_ylabel("snRNA Pearson correlation" if scenario == "log1p-CPM" else "")
        ax.tick_params(axis="x", rotation=30)
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)

    fig.suptitle("Per-cell-type snRNA correlation by method and scenario", fontsize=14)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_celltype_improvement(tag, reports_dir, output_path, scenario="HVG-3000", top_n=10):
    """Figure 3: top-10 per-cell-type SPARKLE improvements (horizontal stacked bar)."""
    csv_path = Path(reports_dir) / SCENARIOS[scenario]["method_dir"] / f"{tag}_snrna_corr_per_celltype.csv"
    df = pd.read_csv(csv_path)
    raw_df = df[df["method"] == "raw"][["cell_group", "corr"]].rename(columns={"corr": "raw_corr"})
    sp_df = df[df["method"] == "SPARKLE"][["cell_group", "corr"]].rename(columns={"corr": "sparkle_corr"})
    merged = raw_df.merge(sp_df, on="cell_group")
    merged["improvement"] = merged["sparkle_corr"] - merged["raw_corr"]
    merged = merged.sort_values("sparkle_corr", ascending=True).tail(top_n)

    fig, ax = plt.subplots(figsize=(8, 6))
    y_pos = np.arange(len(merged))

    # stacked bar: raw_corr as base, improvement as the added segment
    ax.barh(y_pos, merged["raw_corr"], color="#4c78a8", label="raw_corr")
    ax.barh(
        y_pos,
        merged["improvement"],
        left=merged["raw_corr"],
        color="#f58518",
        label="improvement",
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(merged["cell_group"])
    ax.set_xlabel("Pearson correlation")
    ax.set_title(f"Top-{top_n} cell groups by SPARKLE corr ({scenario})")
    ax.legend(loc="lower right")
    ax.axvline(0, color="gray", linewidth=0.5)

    # annotate values
    for i, (raw, sp, imp) in enumerate(zip(merged["raw_corr"], merged["sparkle_corr"], merged["improvement"])):
        ax.text(sp + 0.01, i, f"+{imp:.3f}", va="center", fontsize=8)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_gene_improvement(tag, reports_dir, output_path, scenario="HVG-3000", top_n=10):
    """Figure 4: top-10 gene-level SPARKLE improvements (horizontal stacked bar).

    Genes are selected by largest positive improvement (with positive raw_corr),
    then sorted by sparkle_corr for display.
    """
    csv_path = (
        Path(reports_dir) / SCENARIOS[scenario]["gene_dir"] /
        f"{tag}_pearson_gene_metrics_detailed.csv"
    )
    df = pd.read_csv(csv_path)
    # focus on genes with positive raw_corr and positive improvement
    sub = df[(df["raw_corr"] > 0) & (df["improvement"] > 0)].copy()
    sub = sub.sort_values("improvement", ascending=False).head(top_n)
    # display sorted by sparkle_corr (descending)
    sub = sub.sort_values("sparkle_corr", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    y_pos = np.arange(len(sub))

    ax.barh(y_pos, sub["raw_corr"], color="#4c78a8", label="raw_corr")
    ax.barh(
        y_pos,
        sub["improvement"],
        left=sub["raw_corr"],
        color="#f58518",
        label="improvement",
    )

    labels = [f"{g} ({tg})" for g, tg in zip(sub["gene"], sub["target_group"])]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Pearson correlation")
    ax.set_title(f"Top-{top_n} genes by SPARKLE improvement ({scenario})")
    ax.legend(loc="lower right")
    ax.axvline(0, color="gray", linewidth=0.5)

    for i, (raw, sp, imp) in enumerate(zip(sub["raw_corr"], sub["sparkle_corr"], sub["improvement"])):
        ax.text(sp + 0.01, i, f"+{imp:.3f}", va="center", fontsize=8)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, required=True)
    parser.add_argument("--h5ad-dir", type=str, default="evaluation/reports/h5ad")
    parser.add_argument("--reports-dir", type=str, default="evaluation/reports/mousebrain_eval")
    parser.add_argument("--output-dir", type=str, default="evaluation/reports/mousebrain_eval/visualization")
    parser.add_argument("--n-hvgs", type=int, default=3000)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_hvg_between_corr_heatmaps(
        args.tag, args.h5ad_dir,
        out_dir / f"{args.tag}_HVG{args.n_hvgs}_between_corr_heatmaps.png",
        n_hvgs=args.n_hvgs,
    )
    plot_snrna_corr_boxplots(
        args.tag, args.reports_dir,
        out_dir / f"{args.tag}_snrna_corr_boxplots.png",
    )
    plot_celltype_improvement(
        args.tag, args.reports_dir,
        out_dir / f"{args.tag}_top10_celltype_improvement.png",
    )
    plot_gene_improvement(
        args.tag, args.reports_dir,
        out_dir / f"{args.tag}_top10_gene_improvement.png",
    )


if __name__ == "__main__":
    main()
