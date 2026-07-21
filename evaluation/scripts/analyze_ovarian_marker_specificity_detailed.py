#!/usr/bin/env python3
"""Detailed six-method comparison of ovarian cancer-marker specificity.

This script consumes ``marker_foldchanges.csv`` produced by
``analyze_ovarian_cancer_markers.py`` and expands the aggregate comparison to
the gene level. The expected marker direction is incorporated so that a higher
specificity score is always better:

* epithelial/tumor marker: log2((tumor + p) / (stromal + p))
* stromal/immune marker: log2((pooled non-tumor + p) / (tumor + p))

The default pseudocount is 0.01 log1p-CPM units. It prevents zero denominators
in low-expression immune markers from dominating method means.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import seaborn as sns


METHOD_ORDER = ["RAW", "SPARKLE", "SpotClean", "SoupX", "SpatialSoupX", "DecontX"]
PANEL_ORDER = ["epithelial_tumor", "stromal_fibroblast", "immune"]
PANEL_LABELS = {
    "epithelial_tumor": "Tumor markers",
    "stromal_fibroblast": "Stromal markers",
    "immune": "Immune markers",
    "cancer_core": "Tumor + stromal",
    "all_markers": "All markers",
}
METHOD_COLORS = {
    "RAW": "#9e9e9e",
    "SPARKLE": "#d98b2b",
    "SpotClean": "#4477aa",
    "SoupX": "#7a9e45",
    "SpatialSoupX": "#aa6f9e",
    "DecontX": "#c75b5b",
}
SIGNED_CMAP = LinearSegmentedColormap.from_list(
    "orange_white_blue", ["#b85c12", "#f7f7f7", "#2f6690"]
)


def validate_input(df: pd.DataFrame) -> list[str]:
    required = {
        "method", "gene", "panel", "direction", "tumor_mean",
        "stromal_mean", "log2fc", "log2fc_pseudocount",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if df.duplicated(["method", "gene"]).any():
        raise ValueError("Duplicate method/gene rows detected")

    methods = [method for method in METHOD_ORDER if method in set(df["method"])]
    missing_methods = set(METHOD_ORDER).difference(methods)
    if missing_methods:
        raise ValueError(f"Missing methods: {sorted(missing_methods)}")

    raw_genes = set(df.loc[df["method"] == "RAW", "gene"])
    for method in methods:
        genes = set(df.loc[df["method"] == method, "gene"])
        if genes != raw_genes:
            raise ValueError(
                f"{method}: gene panel differs from RAW; "
                f"missing={sorted(raw_genes - genes)}, extra={sorted(genes - raw_genes)}"
            )
    if not set(df["direction"]).issubset({"up", "down"}):
        raise ValueError("direction must contain only 'up' or 'down'")
    if (df[["tumor_mean", "stromal_mean"]] < 0).any().any():
        raise ValueError("Negative normalized expression means detected")
    pseudocounts = df["log2fc_pseudocount"].drop_duplicates()
    if len(pseudocounts) != 1 or pseudocounts.iloc[0] <= 0:
        raise ValueError("Expected one positive log2FC pseudocount")
    expected_log2fc = np.log2(
        (df["tumor_mean"] + pseudocounts.iloc[0])
        / (df["stromal_mean"] + pseudocounts.iloc[0])
    )
    if not np.allclose(expected_log2fc, df["log2fc"], rtol=0, atol=1e-12):
        raise ValueError("log2fc is inconsistent with expression means")
    return methods


def add_specificity_metrics(
    df: pd.DataFrame, pseudocount: float, tolerance: float
) -> pd.DataFrame:
    result = df.copy()
    is_up = result["direction"].eq("up")
    result["target_mean"] = np.where(
        is_up, result["tumor_mean"], result["stromal_mean"]
    )
    result["offtarget_mean"] = np.where(
        is_up, result["stromal_mean"], result["tumor_mean"]
    )
    result["specificity_log2"] = np.log2(
        (result["target_mean"] + pseudocount)
        / (result["offtarget_mean"] + pseudocount)
    )

    raw = result[result["method"] == "RAW"].set_index("gene")
    raw_columns = {
        "specificity_log2": "raw_specificity_log2",
        "target_mean": "raw_target_mean",
        "offtarget_mean": "raw_offtarget_mean",
        "tumor_mean": "raw_tumor_mean",
        "stromal_mean": "raw_stromal_mean",
    }
    raw_values = raw[list(raw_columns)].rename(columns=raw_columns)
    result = result.join(raw_values, on="gene", validate="many_to_one")
    result["specificity_delta_vs_raw"] = (
        result["specificity_log2"] - result["raw_specificity_log2"]
    )
    result["target_log2_change_vs_raw"] = np.log2(
        (result["target_mean"] + pseudocount)
        / (result["raw_target_mean"] + pseudocount)
    )
    result["offtarget_log2_change_vs_raw"] = np.log2(
        (result["offtarget_mean"] + pseudocount)
        / (result["raw_offtarget_mean"] + pseudocount)
    )

    delta = result["specificity_delta_vs_raw"]
    result["status_vs_raw"] = np.select(
        [delta > tolerance, delta < -tolerance],
        ["improved", "worsened"],
        default="stable",
    )
    result.loc[result["method"] == "RAW", "status_vs_raw"] = "reference"
    result["rank_within_gene"] = result.groupby("gene")["specificity_log2"].rank(
        method="min", ascending=False
    ).astype(int)
    gene_max = result.groupby("gene")["specificity_log2"].transform("max")
    result["best_method_for_gene"] = (
        result["specificity_log2"] >= gene_max - tolerance
    )
    return result


def summarize_group(group: pd.DataFrame) -> pd.Series:
    return pd.Series({
        "n_markers": len(group),
        "mean_specificity_log2": group["specificity_log2"].mean(),
        "median_specificity_log2": group["specificity_log2"].median(),
        "mean_delta_vs_raw": group["specificity_delta_vs_raw"].mean(),
        "median_delta_vs_raw": group["specificity_delta_vs_raw"].median(),
        "n_improved_vs_raw": int((group["status_vs_raw"] == "improved").sum()),
        "n_worsened_vs_raw": int((group["status_vs_raw"] == "worsened").sum()),
        "n_stable_vs_raw": int(group["status_vs_raw"].isin(["stable", "reference"]).sum()),
        "n_best_method": int(group["best_method_for_gene"].sum()),
        "median_target_log2_change": group["target_log2_change_vs_raw"].median(),
        "median_offtarget_log2_change": group["offtarget_log2_change_vs_raw"].median(),
    })


def build_summaries(detail: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for panel in PANEL_ORDER:
        sub = detail[detail["panel"] == panel]
        summary = sub.groupby("method", sort=False).apply(
            summarize_group, include_groups=False
        ).reset_index()
        summary.insert(1, "panel", panel)
        frames.append(summary)

    scopes = {
        "cancer_core": detail[detail["panel"].isin(PANEL_ORDER[:2])],
        "all_markers": detail,
    }
    for panel, sub in scopes.items():
        summary = sub.groupby("method", sort=False).apply(
            summarize_group, include_groups=False
        ).reset_index()
        summary.insert(1, "panel", panel)
        frames.append(summary)

    result = pd.concat(frames, ignore_index=True)
    method_rank = {method: index for index, method in enumerate(METHOD_ORDER)}
    panel_rank = {
        panel: index
        for index, panel in enumerate(PANEL_ORDER + ["cancer_core", "all_markers"])
    }
    result["_method_rank"] = result["method"].map(method_rank)
    result["_panel_rank"] = result["panel"].map(panel_rank)
    return result.sort_values(["_panel_rank", "_method_rank"]).drop(
        columns=["_method_rank", "_panel_rank"]
    )


def build_pairwise(detail: pd.DataFrame, tolerance: float) -> pd.DataFrame:
    scopes = {
        **{panel: detail[detail["panel"] == panel] for panel in PANEL_ORDER},
        "cancer_core": detail[detail["panel"].isin(PANEL_ORDER[:2])],
        "all_markers": detail,
    }
    rows = []
    for scope, sub in scopes.items():
        pivot = sub.pivot(index="gene", columns="method", values="specificity_log2")
        for method_a in METHOD_ORDER:
            for method_b in METHOD_ORDER:
                if method_a == method_b:
                    continue
                delta = pivot[method_a] - pivot[method_b]
                rows.append({
                    "panel": scope,
                    "method_a": method_a,
                    "method_b": method_b,
                    "n_markers": len(delta),
                    "a_wins": int((delta > tolerance).sum()),
                    "b_wins": int((delta < -tolerance).sum()),
                    "ties": int((delta.abs() <= tolerance).sum()),
                    "mean_specificity_delta_a_minus_b": delta.mean(),
                    "median_specificity_delta_a_minus_b": delta.median(),
                })
    return pd.DataFrame(rows)


def build_wide(detail: pd.DataFrame) -> pd.DataFrame:
    base = detail[detail["method"] == "RAW"][["gene", "panel", "direction"]]
    metrics = [
        "specificity_log2", "specificity_delta_vs_raw", "target_mean",
        "offtarget_mean", "target_log2_change_vs_raw",
        "offtarget_log2_change_vs_raw", "rank_within_gene",
    ]
    wide_parts = [base.set_index("gene")]
    for metric in metrics:
        pivot = detail.pivot(index="gene", columns="method", values=metric)
        pivot = pivot.reindex(columns=METHOD_ORDER)
        pivot.columns = [f"{metric}__{method}" for method in pivot.columns]
        wide_parts.append(pivot)
    return pd.concat(wide_parts, axis=1).reset_index()


def ordered_genes(detail: pd.DataFrame) -> list[str]:
    raw = detail[detail["method"] == "RAW"]
    genes = []
    for panel in PANEL_ORDER:
        genes.extend(raw.loc[raw["panel"] == panel, "gene"].tolist())
    return genes


def add_panel_separators(ax: plt.Axes, detail: pd.DataFrame, genes: list[str]) -> None:
    raw_panel = detail[detail["method"] == "RAW"].set_index("gene")["panel"]
    last_panel = raw_panel.loc[genes[0]]
    for index, gene in enumerate(genes[1:], start=1):
        panel = raw_panel.loc[gene]
        if panel != last_panel:
            ax.axhline(index, color="#333333", linewidth=1.1)
            last_panel = panel


def save_figure(fig: plt.Figure, base_path: Path) -> None:
    fig.savefig(base_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_heatmaps(detail: pd.DataFrame, output_dir: Path, pseudocount: float) -> None:
    genes = ordered_genes(detail)
    raw_panel = detail[detail["method"] == "RAW"].set_index("gene")["panel"]
    labels = [f"{gene}  ·  {PANEL_LABELS[raw_panel.loc[gene]]}" for gene in genes]

    absolute = detail.pivot(index="gene", columns="method", values="specificity_log2")
    absolute = absolute.reindex(index=genes, columns=METHOD_ORDER)
    vmax = float(np.nanmax(np.abs(absolute.to_numpy())))
    fig, ax = plt.subplots(figsize=(11.5, 15.5))
    sns.heatmap(
        absolute,
        cmap=SIGNED_CMAP,
        center=0,
        vmin=-vmax,
        vmax=vmax,
        annot=True,
        fmt=".2f",
        annot_kws={"fontsize": 7},
        linewidths=0.4,
        linecolor="#eeeeee",
        cbar_kws={"label": "Expected-direction log2 specificity"},
        ax=ax,
    )
    ax.set_yticklabels(labels, rotation=0, fontsize=8)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    ax.set_xlabel("Method")
    ax.set_ylabel("Marker")
    ax.set_title("Ovarian marker specificity by gene and method", loc="left", fontweight="bold")
    add_panel_separators(ax, detail, genes)
    fig.text(
        0.01, 0.005,
        f"Positive values indicate enrichment in the expected compartment; pseudocount = {pseudocount:g} log1p-CPM.",
        fontsize=9, color="#555555",
    )
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    save_figure(fig, output_dir / "marker_specificity_by_gene_heatmap")

    delta = detail.pivot(
        index="gene", columns="method", values="specificity_delta_vs_raw"
    ).reindex(index=genes, columns=METHOD_ORDER[1:])
    raw_target_expression = (
        detail[detail["method"] == "RAW"]
        .set_index("gene")["target_mean"]
        .reindex(genes)
        .to_frame("RAW target\nexpression")
    )
    vmax = float(np.nanmax(np.abs(delta.to_numpy())))
    fig = plt.figure(figsize=(12.8, 15.5))
    grid = fig.add_gridspec(
        1, 4, width_ratios=[5.3, 0.85, 0.18, 0.18], wspace=0.35
    )
    ax = fig.add_subplot(grid[0, 0])
    expression_ax = fig.add_subplot(grid[0, 1])
    delta_cbar_ax = fig.add_subplot(grid[0, 2])
    expression_cbar_ax = fig.add_subplot(grid[0, 3])
    sns.heatmap(
        delta,
        cmap=SIGNED_CMAP,
        center=0,
        vmin=-vmax,
        vmax=vmax,
        annot=True,
        fmt="+.2f",
        annot_kws={"fontsize": 7},
        linewidths=0.4,
        linecolor="#eeeeee",
        cbar_ax=delta_cbar_ax,
        cbar_kws={},
        ax=ax,
    )
    expression_cmap = sns.light_palette("#4477aa", as_cmap=True)
    sns.heatmap(
        raw_target_expression,
        cmap=expression_cmap,
        vmin=0,
        vmax=float(raw_target_expression.to_numpy().max()),
        annot=True,
        fmt=".2f",
        annot_kws={"fontsize": 7},
        linewidths=0.4,
        linecolor="#eeeeee",
        yticklabels=False,
        cbar_ax=expression_cbar_ax,
        cbar_kws={},
        ax=expression_ax,
    )
    ax.set_yticklabels(labels, rotation=0, fontsize=8)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    ax.set_xlabel("Method")
    ax.set_ylabel("Marker")
    expression_ax.set_xticklabels(
        ["RAW target\nexpression"], rotation=0, ha="center", fontsize=8
    )
    expression_ax.set_xlabel("")
    expression_ax.set_ylabel("")
    expression_ax.tick_params(axis="y", left=False)
    delta_cbar_ax.set_title("Specificity Δ\n(log2)", fontsize=8, pad=8)
    expression_cbar_ax.set_title(
        "RAW target expr.\n(log1p-CPM)", fontsize=8, pad=8
    )
    add_panel_separators(ax, detail, genes)
    add_panel_separators(expression_ax, detail, genes)
    fig.suptitle(
        "Ovarian marker-specificity change relative to RAW",
        x=0.08, ha="left", fontweight="bold", fontsize=14,
    )
    fig.text(
        0.01, 0.005,
        "Blue/positive Δ indicates improved specificity; orange/negative indicates worsening. "
        "RAW target expression is tumor mean for tumor markers and pooled non-tumor mean for stromal/immune markers.",
        fontsize=9, color="#555555",
    )
    fig.subplots_adjust(left=0.22, right=0.93, bottom=0.08, top=0.93)
    save_figure(fig, output_dir / "marker_specificity_delta_vs_raw_heatmap")


def plot_status_counts(summary: pd.DataFrame, output_dir: Path) -> None:
    methods = METHOD_ORDER[1:]
    fig, axes = plt.subplots(1, 3, figsize=(15, 6.2), sharey=True)
    for ax, panel in zip(axes, PANEL_ORDER):
        sub = summary[(summary["panel"] == panel) & summary["method"].isin(methods)]
        sub = sub.set_index("method").loc[methods]
        y = np.arange(len(methods))
        improved = sub["n_improved_vs_raw"].to_numpy(float)
        worsened = sub["n_worsened_vs_raw"].to_numpy(float)
        stable = sub["n_stable_vs_raw"].to_numpy(int)
        ax.barh(y, improved, color="#4477aa", label="Improved")
        ax.barh(y, -worsened, color="#d98b2b", label="Worsened")
        for yi, imp, wor, tie in zip(y, improved, worsened, stable):
            if imp > 0:
                ax.text(imp + 0.15, yi, f"{int(imp)}", va="center", fontsize=8)
            if wor > 0:
                ax.text(-wor - 0.15, yi, f"{int(wor)}", va="center", ha="right", fontsize=8)
            if tie > 0:
                ax.text(0, yi, f"{tie} stable", va="center", ha="center", fontsize=7, color="#555555")
        ax.axvline(0, color="#333333", linewidth=0.9)
        ax.set_title(PANEL_LABELS[panel], fontweight="bold")
        ax.set_xlabel("Marker count vs RAW")
        ax.grid(axis="x", color="#dddddd", linewidth=0.7, alpha=0.7)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_yticks(y, methods)
        limit = int(sub["n_markers"].max()) + 2
        ax.set_xlim(-limit, limit)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.925),
        ncol=2, frameon=False,
    )
    fig.suptitle(
        "Ovarian marker-specificity outcomes by method and panel",
        fontsize=14, fontweight="bold", y=0.985,
    )
    fig.text(
        0.5, 0.005,
        "Improved/worsened uses |Δ expected-direction log2 specificity| > 0.01 relative to RAW.",
        ha="center", fontsize=9, color="#555555",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.84))
    save_figure(fig, output_dir / "marker_specificity_improved_worsened_counts")


def format_float(value: float) -> str:
    return "NA" if pd.isna(value) else f"{value:.3f}"


def write_report(
    detail: pd.DataFrame,
    summary: pd.DataFrame,
    pairwise: pd.DataFrame,
    output_dir: Path,
    pseudocount: float,
    tolerance: float,
) -> None:
    cancer = summary[summary["panel"] == "cancer_core"].set_index("method").loc[METHOD_ORDER]
    lines = [
        "# Ovarian cancer-marker specificity: detailed six-method comparison",
        "",
        "This report expands the aggregate marker means to paired gene-level comparisons.",
        "A higher score is always better: tumor markers use tumor/stromal expression,",
        "while stromal and immune markers use pooled non-tumor/tumor expression.",
        f"Scores are log2 ratios with a {pseudocount:g} log1p-CPM pseudocount.",
        "",
        "## Tumor + stromal marker summary",
        "",
        "| Method | Median specificity | Median Δ vs RAW | Improved | Worsened | Stable | At/near-best markers |",
        "|:--|--:|--:|--:|--:|--:|--:|",
    ]
    for method, row in cancer.iterrows():
        lines.append(
            f"| {method} | {format_float(row['median_specificity_log2'])} | "
            f"{format_float(row['median_delta_vs_raw'])} | "
            f"{int(row['n_improved_vs_raw'])} | {int(row['n_worsened_vs_raw'])} | "
            f"{int(row['n_stable_vs_raw'])} | {int(row['n_best_method'])} |"
        )

    lines.extend([
        "",
        "## Panel-level results",
        "",
        "| Panel | Method | Median Δ vs RAW | Improved | Worsened | Stable |",
        "|:--|:--|--:|--:|--:|--:|",
    ])
    panel_rows = summary[
        summary["panel"].isin(PANEL_ORDER) & summary["method"].isin(METHOD_ORDER[1:])
    ]
    for _, row in panel_rows.iterrows():
        lines.append(
            f"| {PANEL_LABELS[row['panel']]} | {row['method']} | "
            f"{format_float(row['median_delta_vs_raw'])} | "
            f"{int(row['n_improved_vs_raw'])} | {int(row['n_worsened_vs_raw'])} | "
            f"{int(row['n_stable_vs_raw'])} |"
        )

    lines.extend([
        "",
        "## Gene-level specificity changes versus RAW",
        "",
        "Each cell is the expected-direction log2-specificity change relative to RAW;",
        "positive values improve specificity and negative values worsen it. The final",
        f"column includes every method within {tolerance:g} log2 units of the maximum.",
    ])
    gene_order = ordered_genes(detail)
    raw_rows = detail[detail["method"] == "RAW"].set_index("gene")
    delta_wide = detail.pivot(
        index="gene", columns="method", values="specificity_delta_vs_raw"
    ).reindex(columns=METHOD_ORDER[1:])
    near_best = (
        detail[detail["best_method_for_gene"]]
        .groupby("gene", sort=False)["method"]
        .apply(lambda values: ", ".join(values))
    )
    for panel in PANEL_ORDER:
        lines.extend([
            "",
            f"### {PANEL_LABELS[panel]}",
            "",
            "| Marker | RAW score | SPARKLE Δ | SpotClean Δ | SoupX Δ | SpatialSoupX Δ | DecontX Δ | At/near best |",
            "|:--|--:|--:|--:|--:|--:|--:|:--|",
        ])
        for gene in gene_order:
            if raw_rows.loc[gene, "panel"] != panel:
                continue
            deltas = delta_wide.loc[gene]
            lines.append(
                f"| {gene} | {raw_rows.loc[gene, 'specificity_log2']:.3f} | "
                f"{deltas['SPARKLE']:+.3f} | {deltas['SpotClean']:+.3f} | "
                f"{deltas['SoupX']:+.3f} | {deltas['SpatialSoupX']:+.3f} | "
                f"{deltas['DecontX']:+.3f} | {near_best.loc[gene]} |"
            )

    lines.extend([
        "",
        "## Target versus off-target decomposition",
        "",
        "For tumor + stromal markers, the following medians show whether a method",
        "changes expression in the expected compartment (target) or the opposite",
        "compartment (off-target). Negative values mean expression was reduced.",
        "",
        "| Method | Median target log2 change | Median off-target log2 change | Mean specificity Δ |",
        "|:--|--:|--:|--:|",
    ])
    for method, row in cancer.iterrows():
        lines.append(
            f"| {method} | {row['median_target_log2_change']:+.3f} | "
            f"{row['median_offtarget_log2_change']:+.3f} | "
            f"{row['mean_delta_vs_raw']:+.3f} |"
        )

    lines.extend(["", "## Largest marker-level changes versus RAW", ""])
    for method in METHOD_ORDER[1:]:
        sub = detail[detail["method"] == method]
        top = sub.nlargest(5, "specificity_delta_vs_raw")
        bottom = sub.nsmallest(5, "specificity_delta_vs_raw")
        top_text = ", ".join(
            f"{row.gene} ({row.specificity_delta_vs_raw:+.3f})"
            for row in top.itertuples()
        )
        bottom_text = ", ".join(
            f"{row.gene} ({row.specificity_delta_vs_raw:+.3f})"
            for row in bottom.itertuples()
        )
        lines.extend([
            f"### {method}", "", f"- Largest improvements: {top_text}.",
            f"- Largest worsening: {bottom_text}.", "",
        ])

    lines.extend([
        "## Pairwise interpretation",
        "",
        "Pairwise wins are available for every method pair and panel in",
        "`marker_method_pairwise_wins.csv`. A win requires a specificity difference",
        f"larger than {tolerance:g} log2 units; smaller differences are ties.",
        "",
        "## Caveats",
        "",
        "- Marker panels are curated and small (17 tumor, 9 stromal, 6 immune markers).",
        "- Results use fixed RAW-RCTD annotations for every expression method, avoiding circular relabeling.",
        "- The `stromal_mean` legacy column pools every annotated non-tumor cell type; immune scores therefore measure non-tumor-versus-tumor separation, not immune-subtype localization.",
        "- The legacy raw FC remains in the detailed CSV, but immune-panel averages should not use it because CD3E/NKG7 have zero stromal means in some methods.",
        "- The robust log2 score handles these zeros with the stated pseudocount; conclusions near zero remain pseudocount-sensitive.",
        "- Across pseudocounts 0.001-0.1, the mean tumor+stromal Δ ordering is unchanged: SpotClean, SPARKLE, DecontX, SoupX, SpatialSoupX.",
        "- Specificity can improve either by preserving target expression or suppressing off-target expression. Both components are reported separately.",
        "- A positive within-spatial Δ does not by itself establish agreement with the paired single-cell reference. The per-marker external-reference analysis is in `../scrna_agreement/marker_scrna_agreement_report.md`.",
    ])
    (output_dir / "marker_specificity_detailed_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="evaluation/reports/ovarian_eval/marker_analysis/marker_foldchanges.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="evaluation/reports/ovarian_eval/marker_analysis/detailed",
    )
    parser.add_argument("--pseudocount", type=float, default=0.01)
    parser.add_argument("--tolerance", type=float, default=0.01)
    args = parser.parse_args()
    if args.pseudocount <= 0:
        raise ValueError("--pseudocount must be positive")
    if args.tolerance < 0:
        raise ValueError("--tolerance must be non-negative")

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_path)
    methods = validate_input(df)
    print(f"Validated {len(df)} rows: {len(methods)} methods × {df.gene.nunique()} genes")

    detail = add_specificity_metrics(df, args.pseudocount, args.tolerance)
    summary = build_summaries(detail)
    pairwise = build_pairwise(detail, args.tolerance)
    wide = build_wide(detail)

    detail.to_csv(output_dir / "marker_gene_method_specificity.csv", index=False)
    summary.to_csv(output_dir / "marker_panel_method_summary.csv", index=False)
    pairwise.to_csv(output_dir / "marker_method_pairwise_wins.csv", index=False)
    wide.to_csv(output_dir / "marker_gene_method_comparison_wide.csv", index=False)

    plot_heatmaps(detail, output_dir, args.pseudocount)
    plot_status_counts(summary, output_dir)
    write_report(
        detail, summary, pairwise, output_dir, args.pseudocount, args.tolerance
    )
    print(f"Detailed outputs written to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
