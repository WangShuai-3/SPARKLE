#!/usr/bin/env python3
"""Compare COL1A2-SDC4 tumor signaling across ovarian correction methods.

The figure links three levels of evidence on the same 16,198 CellChat cells:

1. COL1A2 expression by cell type (mean log1p-CP10K and positive fraction).
2. The original route-specific comparison: VEGFA+ tumor autocrine versus
   TAF-to-VEGFA+ tumor paracrine signaling.
3. The same two COL1A2-SDC4 routes in the non-spatial scRNA CellChat reference.

SpatialSoupX is deliberately excluded from this focused comparison.
"""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[2]
TAG = "ovarian_x1000-1800_y300-1100"
ANNOTATED_DIR = ROOT / "evaluation" / "reports" / "h5ad_ovarian_annotated"
CELLCHAT_DIR = ROOT / "evaluation" / "reports" / "ovarian_eval" / "cellchat_spatial"
SCRNA_CELLCHAT = (
    ROOT / "evaluation" / "reports" / "ovarian_eval" / "cellchat"
    / "scRNA_liana_cellchat.csv"
)
OUTPUT_DIR = CELLCHAT_DIR / "col1a2_sdc4_method_comparison"

METHOD_SUFFIXES = {
    "RAW": "raw",
    "SPARKLE": "SPARKLE",
    "SpotClean": "SpotCleanOfficial",
    "SoupX": "SoupX",
    "DecontX": "DecontX",
}
METHODS = list(METHOD_SUFFIXES)

VEGFA_TUMOR = "VEGFA+ Tumor Cells"
MT_HIGH_TUMOR = "MT-High, Jun+-Fos+ Tumor Cells"
TAF = "Tumor Associated Fibroblasts"
DISPLAY_GROUPS = [VEGFA_TUMOR, MT_HIGH_TUMOR, TAF]
GROUP_LABELS = {
    VEGFA_TUMOR: "VEGFA+ tumor",
    MT_HIGH_TUMOR: "MT-High tumor",
    TAF: "TAF (fibroblast)",
}

TUMOR_TYPES = {
    "Tumor Cells",
    "Proliferative Tumor Cells",
    VEGFA_TUMOR,
    MT_HIGH_TUMOR,
    "Inflammatory Tumor Cells",
    "Malignant Cells Lining Cyst",
}

FALSE_COLOR = "#4C78A8"
TRUE_COLOR = "#D39C2C"
INK = "#252525"
GRID = "#D9D9D9"


def load_expression_summary(qc: pd.DataFrame) -> pd.DataFrame:
    """Compute COL1A2 summaries from the exact cross-method CellChat cells."""
    included = qc.loc[qc["included_in_cellchat"].astype(bool)].copy()
    included.index = included["cell_id"].astype(int)
    included_ids = set(included.index)
    rows: list[dict] = []

    for method, suffix in METHOD_SUFFIXES.items():
        path = ANNOTATED_DIR / f"{TAG}_{suffix}.h5ad"
        adata = ad.read_h5ad(path, backed="r")
        try:
            keep = adata.obs["cell_id"].astype(int).isin(included_ids).to_numpy()
            obs = adata.obs.loc[keep]
            gene_idx = adata.var_names.get_loc("COL1A2")
            counts = np.asarray(adata.X[:, gene_idx]).reshape(-1)[keep]
            counts = np.maximum(counts.astype(float), 0.0)
            library_sizes = (
                obs["cell_id"]
                .astype(int)
                .map(included[f"{method}_library_size"])
                .to_numpy(float)
            )
            if np.any(~np.isfinite(library_sizes)) or np.any(library_sizes <= 0):
                raise ValueError(f"Invalid library sizes for {method}")
            expression = np.log1p(counts / library_sizes * 1e4)
            labels = obs["annotation"].astype(str).to_numpy()

            for group in DISPLAY_GROUPS:
                mask = labels == group
                rows.append(
                    {
                        "method": method,
                        "cell_type": group,
                        "n_cells": int(mask.sum()),
                        "mean_log1p_cp10k": float(expression[mask].mean()),
                        "pct_positive": float((counts[mask] > 0).mean() * 100),
                    }
                )
        finally:
            adata.file.close()

    # The existing proof table contains the paired scFFPE reference means;
    # LIANA's ligand_props supplies the corresponding positive fraction.
    proof = pd.read_csv(CELLCHAT_DIR / "proof_collagen_expression.csv")
    scrna_cellchat = pd.read_csv(SCRNA_CELLCHAT)
    scrna_col1a2_props = (
        scrna_cellchat.loc[
            scrna_cellchat["ligand_complex"].eq("COL1A2"),
            ["source", "ligand_props"],
        ]
        .drop_duplicates("source")
        .set_index("source")["ligand_props"]
    )
    truth = proof.loc[
        proof["gene"].eq("COL1A2") & proof["cell_type"].isin(DISPLAY_GROUPS),
        ["cell_type", "n_cells", "scRNA_truth"],
    ].drop_duplicates("cell_type")
    for row in truth.itertuples(index=False):
        rows.append(
            {
                "method": "scRNA reference",
                "cell_type": row.cell_type,
                "n_cells": np.nan,
                "mean_log1p_cp10k": float(row.scRNA_truth),
                "pct_positive": float(scrna_col1a2_props.loc[row.cell_type] * 100),
            }
        )
    return pd.DataFrame(rows)


def load_cellchat_summary() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load route-specific and tumor-aggregate COL1A2-SDC4 probabilities."""
    route_rows: list[dict] = []
    aggregate_rows: list[dict] = []

    route_definitions = [
        (VEGFA_TUMOR, VEGFA_TUMOR, "VEGFA+ tumor self", "false autocrine"),
        (TAF, VEGFA_TUMOR, "TAF to VEGFA+ tumor", "true paracrine"),
    ]

    for method in METHODS:
        path = CELLCHAT_DIR / f"{method}_cellchat_spatial_all_tested.csv"
        data = pd.read_csv(path)
        lr = data.loc[
            data["ligand"].eq("COL1A2") & data["receptor"].eq("SDC4")
        ].copy()

        for source, target, route, route_class in route_definitions:
            selected = lr.loc[lr["source"].eq(source) & lr["target"].eq(target)]
            route_rows.append(
                {
                    "method": method,
                    "source": source,
                    "target": target,
                    "route": route,
                    "route_class": route_class,
                    "probability": float(selected["prob"].sum()),
                    "detected_significant": bool(
                        len(selected) and selected["pval"].iloc[0] < 0.05
                    ),
                    "p_value": (
                        float(selected["pval"].iloc[0]) if len(selected) else np.nan
                    ),
                }
            )

        # Sum all tested probabilities, not only significant exported edges.
        same_type_self = lr.loc[
            lr["source"].eq(lr["target"]) & lr["source"].isin(TUMOR_TYPES), "prob"
        ].sum()
        taf_to_tumor = lr.loc[
            lr["source"].eq(TAF) & lr["target"].isin(TUMOR_TYPES), "prob"
        ].sum()
        aggregate_rows.extend(
            [
                {
                    "method": method,
                    "aggregate_route": "Same-type tumor self",
                    "route_class": "false autocrine",
                    "probability": float(same_type_self),
                    "score_scale": "spatial CellChat probability",
                },
                {
                    "method": method,
                    "aggregate_route": "TAF to all tumor types",
                    "route_class": "true paracrine",
                    "probability": float(taf_to_tumor),
                    "score_scale": "spatial CellChat probability",
                },
            ]
        )

    # Append the non-spatial single-cell CellChat reference.  Its LIANA
    # lr_probs are on a different absolute scale, so Panel C compares only the
    # within-dataset autocrine/paracrine ratio, never raw score magnitudes.
    scrna = pd.read_csv(SCRNA_CELLCHAT)
    scrna_lr = scrna.loc[
        scrna["ligand_complex"].eq("COL1A2")
        & scrna["receptor_complex"].eq("SDC4")
    ].copy()
    for source, target, route, route_class in route_definitions:
        selected = scrna_lr.loc[
            scrna_lr["source"].eq(source) & scrna_lr["target"].eq(target)
        ]
        route_rows.append(
            {
                "method": "scRNA reference",
                "source": source,
                "target": target,
                "route": route,
                "route_class": route_class,
                "probability": float(selected["lr_probs"].sum()),
                "detected_significant": bool(
                    len(selected) and selected["cellchat_pvals"].iloc[0] < 0.05
                ),
                "p_value": (
                    float(selected["cellchat_pvals"].iloc[0])
                    if len(selected)
                    else np.nan
                ),
            }
        )
    scrna_self = scrna_lr.loc[
        scrna_lr["source"].eq(scrna_lr["target"])
        & scrna_lr["source"].isin(TUMOR_TYPES),
        "lr_probs",
    ].sum()
    scrna_taf = scrna_lr.loc[
        scrna_lr["source"].eq(TAF) & scrna_lr["target"].isin(TUMOR_TYPES),
        "lr_probs",
    ].sum()
    aggregate_rows.extend(
        [
            {
                "method": "scRNA reference",
                "aggregate_route": "Same-type tumor self",
                "route_class": "false autocrine",
                "probability": float(scrna_self),
                "score_scale": "LIANA CellChat lr_probs",
            },
            {
                "method": "scRNA reference",
                "aggregate_route": "TAF to all tumor types",
                "route_class": "true paracrine",
                "probability": float(scrna_taf),
                "score_scale": "LIANA CellChat lr_probs",
            },
        ]
    )

    route_df = pd.DataFrame(route_rows)
    aggregate_df = pd.DataFrame(aggregate_rows)
    raw_probability = (
        aggregate_df.loc[aggregate_df["method"].eq("RAW")]
        .set_index("aggregate_route")["probability"]
    )
    aggregate_df["retained_vs_raw_pct"] = aggregate_df.apply(
        lambda row: (
            100 * row["probability"] / raw_probability.loc[row["aggregate_route"]]
            if row["method"] != "scRNA reference"
            else np.nan
        ),
        axis=1,
    )
    ratio_by_method = (
        aggregate_df.pivot(
            index="method", columns="route_class", values="probability"
        )
        .assign(
            autocrine_paracrine_ratio=lambda frame: (
                frame["false autocrine"] / frame["true paracrine"]
            )
        )["autocrine_paracrine_ratio"]
    )
    aggregate_df["autocrine_paracrine_ratio"] = aggregate_df["method"].map(
        ratio_by_method
    )
    return route_df, aggregate_df


def plot_figure(
    expression_df: pd.DataFrame,
    route_df: pd.DataFrame,
    aggregate_df: pd.DataFrame,
) -> plt.Figure:
    sns.set_theme(style="whitegrid", context="talk")
    fig = plt.figure(figsize=(17, 12), constrained_layout=False)
    grid = fig.add_gridspec(
        2, 2, height_ratios=[0.9, 1.25], width_ratios=[1.15, 1],
        left=0.07, right=0.97, top=0.80, bottom=0.11, hspace=0.50, wspace=0.30,
    )
    ax_a = fig.add_subplot(grid[0, :])
    ax_b = fig.add_subplot(grid[1, 0])
    ax_c = fig.add_subplot(grid[1, 1])

    # Panel A: dot plot. Color encodes mean expression and size encodes the
    # positive fraction, using the same encoding for spatial and scRNA data.
    expression_order = ["scRNA reference", *METHODS]
    mean_matrix = (
        expression_df.pivot(index="cell_type", columns="method", values="mean_log1p_cp10k")
        .reindex(index=DISPLAY_GROUPS, columns=expression_order)
    )
    pct_matrix = (
        expression_df.pivot(index="cell_type", columns="method", values="pct_positive")
        .reindex(index=DISPLAY_GROUPS, columns=expression_order)
    )
    x_idx, y_idx = np.meshgrid(
        np.arange(len(expression_order)), np.arange(len(DISPLAY_GROUPS))
    )
    means = mean_matrix.to_numpy().ravel()
    percentages = pct_matrix.to_numpy().ravel()
    size_values = 45 + percentages * 5.3
    dot = ax_a.scatter(
        x_idx.ravel(),
        y_idx.ravel(),
        s=size_values,
        c=means,
        cmap=sns.light_palette(FALSE_COLOR, as_cmap=True),
        vmin=0,
        vmax=max(5.2, float(np.nanmax(means))),
        edgecolor=INK,
        linewidth=0.8,
    )
    ax_a.axvspan(-0.5, 0.5, color="#F2F2F2", zorder=-2)
    ax_a.axvline(0.5, color="#8C8C8C", linestyle="--", linewidth=1)
    colorbar = fig.colorbar(dot, ax=ax_a, fraction=0.025, pad=0.025)
    colorbar.set_label("Mean COL1A2 log1p-CP10K")
    size_handles = [
        ax_a.scatter(
            [], [], s=45 + pct * 5.3, facecolor="white", edgecolor=INK,
            linewidth=0.8, label=f"{pct}%"
        )
        for pct in [25, 50, 75, 100]
    ]
    ax_a.legend(
        handles=size_handles,
        title="Positive cells",
        frameon=False,
        loc="center right",
        bbox_to_anchor=(0.985, 0.5),
        fontsize=10,
        title_fontsize=10,
    )
    ax_a.set_title(
        "A  |  COL1A2 expression and detection rate",
        loc="left",
        weight="bold",
        fontsize=18,
        y=1.14,
    )
    ax_a.set_xlabel("")
    ax_a.set_ylabel("")
    ax_a.set_xticks(
        np.arange(len(expression_order)),
        ["scRNA\nreference", "RAW", "SPARKLE", "SpotClean", "SoupX", "DecontX"],
    )
    ax_a.set_yticks(
        np.arange(len(DISPLAY_GROUPS)),
        [GROUP_LABELS[g] for g in DISPLAY_GROUPS],
    )
    ax_a.set_xlim(-0.5, len(expression_order) + 0.65)
    ax_a.set_ylim(len(DISPLAY_GROUPS) - 0.5, -0.5)
    ax_a.grid(color="#E6E6E6", linewidth=0.8)
    ax_a.set_axisbelow(True)
    ax_a.text(
        0,
        1.03,
        "Dot color shows mean expression; dot size shows the percentage of COL1A2-positive cells.",
        transform=ax_a.transAxes,
        fontsize=11,
        color="#555555",
    )

    # Panel B: direct comparison of the original two routes.
    x = np.arange(len(METHODS))
    width = 0.36
    route_specs = [
        ("VEGFA+ tumor self", "False: VEGFA+ tumor self", FALSE_COLOR, -width / 2),
        ("TAF to VEGFA+ tumor", "True: TAF → VEGFA+ tumor", TRUE_COLOR, width / 2),
    ]
    for route, label, color, offset in route_specs:
        selected = route_df.loc[route_df["route"].eq(route)].set_index("method").reindex(METHODS)
        values = selected["probability"].to_numpy() * 1e3
        bars = ax_b.bar(
            x + offset,
            values,
            width,
            label=label,
            color=color,
            edgecolor=INK,
            linewidth=0.8,
        )
        for bar, value in zip(bars, values):
            ax_b.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.07,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
                color=INK,
            )
    ax_b.set_title("B  |  Original COL1A2–SDC4 routes", loc="left", weight="bold")
    ax_b.set_ylabel("CellChat probability (×10⁻³)")
    ax_b.set_xlabel("")
    ax_b.set_xticks(x, METHODS, rotation=22, ha="right")
    ax_b.set_ylim(0, 4.35)
    ax_b.legend(frameon=False, fontsize=11, loc="upper right")
    ax_b.grid(axis="x", visible=False)

    # Panel C: the same two routes in the scRNA reference, using all lr_probs
    # without p-value filtering or significance styling. It mirrors Panel B,
    # but keeps the LIANA CellChat score on its own axis because the absolute
    # scale is not comparable with spatial CellChat v2 probabilities.
    scrna_routes = (
        route_df.loc[route_df["method"].eq("scRNA reference")]
        .set_index("route")
        .reindex([spec[0] for spec in route_specs])
    )
    scrna_values = scrna_routes["probability"].to_numpy()
    scrna_x = np.arange(len(route_specs))
    bars = ax_c.bar(
        scrna_x,
        scrna_values,
        color=[FALSE_COLOR, TRUE_COLOR],
        edgecolor=INK,
        linewidth=0.8,
        width=0.62,
    )
    for bar, value in zip(bars, scrna_values):
        ax_c.text(
            bar.get_x() + bar.get_width() / 2,
            value + max(scrna_values.max() * 0.035, 0.002),
            f"{value:.3f}",
            va="bottom",
            ha="center",
            fontsize=11,
        )
    ax_c.set_title(
        "C  |  scRNA reference COL1A2–SDC4 routes",
        loc="left",
        weight="bold",
    )
    ax_c.set_ylabel("LIANA CellChat lr_probs")
    ax_c.set_xlabel("")
    ax_c.set_xticks(
        scrna_x,
        ["VEGFA+ tumor\nself", "TAF → VEGFA+\ntumor"],
    )
    ax_c.set_ylim(0, max(float(scrna_values.max()) * 1.22, 0.14))
    ax_c.grid(axis="x", visible=False)

    fig.suptitle(
        "Cross-method comparison of COL1A2–SDC4 tumor signaling",
        fontsize=23,
        weight="bold",
        color=INK,
        y=0.975,
    )
    fig.text(
        0.5,
        0.935,
        "Same 16,198 spatial cells · shared RAW-RCTD annotations · CellChat range 250 µm · matched scRNA routes in C · SpatialSoupX excluded",
        ha="center",
        fontsize=12,
        color="#555555",
    )
    fig.text(
        0.07,
        0.035,
        "B and C show all tested scores without significance filtering or styling. "
        "B shows spatial CellChat v2 probabilities; C shows matching scRNA routes using LIANA CellChat lr_probs. "
        "The separate axes reflect that spatial probabilities and scRNA scores are not directly comparable in magnitude.",
        fontsize=10,
        color="#555555",
    )
    return fig


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    qc = pd.read_csv(CELLCHAT_DIR / "cellchat_spatial_cell_qc.csv")
    expression_df = load_expression_summary(qc)
    route_df, aggregate_df = load_cellchat_summary()

    expression_df.to_csv(OUTPUT_DIR / "col1a2_expression_by_method.csv", index=False)
    route_df.to_csv(OUTPUT_DIR / "col1a2_sdc4_routes_by_method.csv", index=False)
    aggregate_df.to_csv(
        OUTPUT_DIR / "col1a2_sdc4_tumor_aggregate_by_method.csv", index=False
    )

    fig = plot_figure(expression_df, route_df, aggregate_df)
    fig.savefig(
        OUTPUT_DIR / "col1a2_sdc4_method_comparison.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    fig.savefig(
        OUTPUT_DIR / "col1a2_sdc4_method_comparison.pdf",
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)
    print(f"Saved comparison outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
