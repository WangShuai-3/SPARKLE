#!/usr/bin/env python3
"""Compare ovarian cancer-marker profiles with the paired scFFPE reference.

The analysis complements within-spatial tumor/non-tumor specificity with two
single-cell-reference checks:

1. Per-marker Pearson and Spearman correlation across matched cell types.
2. Agreement of a type-balanced expected-direction specificity score with the
   corresponding score in the single-cell reference.

Both all observed cell types and a robust subset with at least 20 spatial cells
are reported.  The h5ad matrices are read in chunks so only the requested
marker columns are retained in memory.
"""

import argparse
from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import sparse
from scipy.stats import pearsonr, spearmanr


TAG = "ovarian_x1000-1800_y300-1100"
METHOD_ORDER = ["RAW", "SPARKLE", "SpotClean", "SoupX", "SpatialSoupX", "DecontX"]
METHOD_SUFFIXES = {"RAW": "raw", "SpotClean": "SpotCleanOfficial"}
PANEL_ORDER = ["epithelial_tumor", "stromal_fibroblast", "immune"]
PANEL_LABELS = {
    "epithelial_tumor": "Tumor markers",
    "stromal_fibroblast": "Stromal markers",
    "immune": "Immune markers",
    "cancer_core": "Tumor + stromal",
    "all_markers": "All markers",
}
TUMOR_TYPES = {
    "Tumor Cells",
    "Proliferative Tumor Cells",
    "VEGFA+ Tumor Cells",
    "MT-High, Jun+-Fos+ Tumor Cells",
    "Inflammatory Tumor Cells",
    "Malignant Cells Lining Cyst",
}
STRUCTURAL_STROMAL_TYPES = {
    "Tumor Associated Fibroblasts",
    "Stromal Associated Fibroblasts",
    "Endothelial Cells",
    "Pericytes",
    "Smooth Muscle Cells",
}
PANEL_COLORS = {
    "epithelial_tumor": "#4477aa",
    "stromal_fibroblast": "#d98b2b",
    "immune": "#8b6f9e",
}
SIGNED_CMAP = LinearSegmentedColormap.from_list(
    "orange_white_blue", ["#b85c12", "#f7f7f7", "#2f6690"]
)


def safe_correlation(x: np.ndarray, y: np.ndarray, method: str) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan
    if method == "pearson":
        return float(pearsonr(x, y).statistic)
    return float(spearmanr(x, y).statistic)


def extract_marker_expression(
    adata: ad.AnnData, marker_indices: np.ndarray, chunk_size: int
) -> np.ndarray:
    """Return cells × markers log1p-CPM while reading full rows in chunks."""
    result = np.empty((adata.n_obs, len(marker_indices)), dtype=np.float64)
    for start in range(0, adata.n_obs, chunk_size):
        stop = min(start + chunk_size, adata.n_obs)
        block = adata.X[start:stop]
        if sparse.issparse(block):
            block = block.toarray()
        block = np.asarray(block, dtype=np.float64)
        np.maximum(block, 0.0, out=block)
        totals = block.sum(axis=1)
        scale = np.divide(
            1e4, totals, out=np.zeros_like(totals), where=totals > 0
        )
        result[start:stop] = np.log1p(block[:, marker_indices] * scale[:, None])
    return result


def method_profiles(
    input_dir: Path,
    tag: str,
    markers: list[str],
    reference_types: list[str],
    chunk_size: int,
) -> tuple[dict[str, pd.DataFrame], pd.Series]:
    profiles = {}
    reference_obs_names = None
    reference_var_names = None
    reference_annotations = None
    group_counts = None

    for method in METHOD_ORDER:
        suffix = METHOD_SUFFIXES.get(method, method)
        path = input_dir / f"{tag}_{suffix}.h5ad"
        if not path.exists():
            raise FileNotFoundError(path)
        print(f"Loading marker profiles: {method} ({path.name})")
        adata = ad.read_h5ad(path, backed="r")
        try:
            annotations = adata.obs["annotation"].astype(str)
            if reference_obs_names is None:
                reference_obs_names = adata.obs_names.copy()
                reference_var_names = adata.var_names.copy()
                reference_annotations = annotations.to_numpy()
                group_counts = annotations.value_counts().reindex(reference_types).fillna(0).astype(int)
            else:
                if not adata.obs_names.equals(reference_obs_names):
                    raise ValueError(f"{method}: cell order differs from RAW")
                if not adata.var_names.equals(reference_var_names):
                    raise ValueError(f"{method}: gene order differs from RAW")
                if not np.array_equal(annotations.to_numpy(), reference_annotations):
                    raise ValueError(f"{method}: annotations differ from RAW")

            marker_indices = adata.var_names.get_indexer(markers)
            if (marker_indices < 0).any():
                missing = np.asarray(markers)[marker_indices < 0].tolist()
                raise ValueError(f"{method}: missing markers {missing}")
            expression = extract_marker_expression(adata, marker_indices, chunk_size)
            profile = {}
            labels = annotations.to_numpy()
            for cell_type in reference_types:
                mask = labels == cell_type
                if not mask.any():
                    raise ValueError(f"{method}: no spatial cells for {cell_type}")
                profile[cell_type] = expression[mask].mean(axis=0)
            profiles[method] = pd.DataFrame(profile, index=markers)
        finally:
            adata.file.close()
    return profiles, group_counts


def build_profile_table(
    profiles: dict[str, pd.DataFrame],
    reference: pd.DataFrame,
    metadata: pd.DataFrame,
    group_counts: pd.Series,
    min_cells: int,
) -> pd.DataFrame:
    rows = []
    meta = metadata.set_index("gene")
    for method, profile in profiles.items():
        for gene in metadata["gene"]:
            for cell_type in reference.columns:
                rows.append({
                    "method": method,
                    "gene": gene,
                    "panel": meta.loc[gene, "panel"],
                    "direction": meta.loc[gene, "direction"],
                    "cell_type": cell_type,
                    "n_spatial_cells": int(group_counts[cell_type]),
                    "robust_min_cells": bool(group_counts[cell_type] >= min_cells),
                    "spatial_mean_log1pcpm": float(profile.loc[gene, cell_type]),
                    "scrna_mean_log1pcpm": float(reference.loc[gene, cell_type]),
                })
    return pd.DataFrame(rows)


def build_scrna_marker_logfc(
    reference: pd.DataFrame,
    metadata: pd.DataFrame,
    annotation_path: Path,
    pseudocount: float,
) -> tuple[pd.DataFrame, pd.Series]:
    """Compute scFFPE tumor/non-tumor log2FC for the preset core markers."""
    annotation = pd.read_csv(annotation_path)
    required = {"Barcode", "Cell Annotation"}
    missing = required.difference(annotation.columns)
    if missing:
        raise ValueError(f"scRNA annotation missing columns: {sorted(missing)}")
    annotation = annotation.dropna(subset=["Barcode", "Cell Annotation"]).copy()
    annotation["cell_type"] = annotation["Cell Annotation"].str.replace(
        "/", "-", regex=False
    )
    cell_counts = annotation["cell_type"].value_counts().reindex(
        reference.columns
    )
    if cell_counts.isna().any() or (cell_counts <= 0).any():
        raise ValueError("Every scRNA reference cell type must have a positive cell count")
    cell_counts = cell_counts.astype(int)

    tumor_types = [cell_type for cell_type in reference.columns if cell_type in TUMOR_TYPES]
    non_tumor_types = [cell_type for cell_type in reference.columns if cell_type not in TUMOR_TYPES]
    structural_stromal_types = [
        cell_type for cell_type in reference.columns
        if cell_type in STRUCTURAL_STROMAL_TYPES
    ]
    if len(tumor_types) != 6 or len(non_tumor_types) != 10 or len(structural_stromal_types) != 5:
        raise ValueError("Unexpected tumor/non-tumor/structural-stromal type coverage")

    def weighted_mean(gene: str, cell_types: list[str]) -> float:
        return float(np.average(
            reference.loc[gene, cell_types].to_numpy(float),
            weights=cell_counts.loc[cell_types].to_numpy(float),
        ))

    rows = []
    core = metadata[metadata["panel"].isin(PANEL_ORDER[:2])]
    for row in core.itertuples(index=False):
        tumor_weighted = weighted_mean(row.gene, tumor_types)
        non_tumor_weighted = weighted_mean(row.gene, non_tumor_types)
        structural_stromal_weighted = weighted_mean(
            row.gene, structural_stromal_types
        )
        tumor_balanced = float(reference.loc[row.gene, tumor_types].mean())
        non_tumor_balanced = float(reference.loc[row.gene, non_tumor_types].mean())

        logfc_weighted = float(np.log2(
            (tumor_weighted + pseudocount)
            / (non_tumor_weighted + pseudocount)
        ))
        logfc_balanced = float(np.log2(
            (tumor_balanced + pseudocount)
            / (non_tumor_balanced + pseudocount)
        ))
        logfc_structural = float(np.log2(
            (tumor_weighted + pseudocount)
            / (structural_stromal_weighted + pseudocount)
        ))
        expected_positive = row.direction == "up"
        rows.append({
            "gene": row.gene,
            "panel": row.panel,
            "expected_direction": "tumor_up" if expected_positive else "stromal_up",
            "scrna_tumor_mean_log1pcpm_weighted": tumor_weighted,
            "scrna_non_tumor_mean_log1pcpm_weighted": non_tumor_weighted,
            "scrna_log2fc_tumor_vs_non_tumor_weighted": logfc_weighted,
            "scrna_tumor_mean_log1pcpm_type_balanced": tumor_balanced,
            "scrna_non_tumor_mean_log1pcpm_type_balanced": non_tumor_balanced,
            "scrna_log2fc_tumor_vs_non_tumor_type_balanced": logfc_balanced,
            "scrna_structural_stromal_mean_log1pcpm_weighted": structural_stromal_weighted,
            "scrna_log2fc_tumor_vs_structural_stroma_weighted": logfc_structural,
            "supported_cell_weighted": bool(logfc_weighted > 0) == expected_positive,
            "supported_type_balanced": bool(logfc_balanced > 0) == expected_positive,
            "supported_structural_stroma": bool(logfc_structural > 0) == expected_positive,
            "low_absolute_expression_weighted": max(
                tumor_weighted, non_tumor_weighted
            ) < 0.1,
        })
    return pd.DataFrame(rows), cell_counts


def build_gene_correlations(
    profiles: dict[str, pd.DataFrame],
    reference: pd.DataFrame,
    metadata: pd.DataFrame,
    group_counts: pd.Series,
    min_cells: int,
    tolerance: float,
) -> pd.DataFrame:
    scopes = {
        "all_types": list(reference.columns),
        f"robust_min{min_cells}": [
            cell_type for cell_type in reference.columns
            if group_counts[cell_type] >= min_cells
        ],
    }
    meta = metadata.set_index("gene")
    rows = []
    for scope, cell_types in scopes.items():
        for method, profile in profiles.items():
            for gene in metadata["gene"]:
                spatial_values = profile.loc[gene, cell_types].to_numpy(float)
                reference_values = reference.loc[gene, cell_types].to_numpy(float)
                rows.append({
                    "scope": scope,
                    "method": method,
                    "gene": gene,
                    "panel": meta.loc[gene, "panel"],
                    "n_cell_types": len(cell_types),
                    "pearson": safe_correlation(spatial_values, reference_values, "pearson"),
                    "spearman": safe_correlation(spatial_values, reference_values, "spearman"),
                    "reference_std": float(np.std(reference_values, ddof=1)),
                    "spatial_std": float(np.std(spatial_values, ddof=1)),
                    "reference_dynamic_range": float(reference_values.max() - reference_values.min()),
                })
    result = pd.DataFrame(rows)
    raw = result[result["method"] == "RAW"].set_index(["scope", "gene"])
    for metric in ["pearson", "spearman"]:
        result[f"raw_{metric}"] = [
            raw.loc[(scope, gene), metric]
            for scope, gene in zip(result["scope"], result["gene"])
        ]
        result[f"{metric}_delta_vs_raw"] = result[metric] - result[f"raw_{metric}"]
    delta = result["pearson_delta_vs_raw"]
    result["pearson_status_vs_raw"] = np.select(
        [delta > tolerance, delta < -tolerance],
        ["improved", "worsened"], default="stable",
    )
    result.loc[result["method"] == "RAW", "pearson_status_vs_raw"] = "reference"
    result["low_reference_dynamic_range"] = result["reference_std"] < 0.05
    return result


def panel_subsets(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        **{panel: frame[frame["panel"] == panel] for panel in PANEL_ORDER},
        "cancer_core": frame[frame["panel"].isin(PANEL_ORDER[:2])],
        "all_markers": frame,
    }


def fisher_mean(values: pd.Series) -> float:
    values = values.dropna().clip(-0.999999, 0.999999)
    if values.empty:
        return np.nan
    return float(np.tanh(np.arctanh(values).mean()))


def summarize_correlations(correlations: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scope in correlations["scope"].unique():
        scoped = correlations[correlations["scope"] == scope]
        for panel, panel_frame in panel_subsets(scoped).items():
            for method in METHOD_ORDER:
                group = panel_frame[panel_frame["method"] == method]
                rows.append({
                    "scope": scope,
                    "panel": panel,
                    "method": method,
                    "n_markers": len(group),
                    "n_valid_pearson": int(group["pearson"].notna().sum()),
                    "median_pearson": group["pearson"].median(),
                    "fisher_mean_pearson": fisher_mean(group["pearson"]),
                    "median_spearman": group["spearman"].median(),
                    "median_pearson_delta_vs_raw": group["pearson_delta_vs_raw"].median(),
                    "n_pearson_improved_vs_raw": int((group["pearson_status_vs_raw"] == "improved").sum()),
                    "n_pearson_worsened_vs_raw": int((group["pearson_status_vs_raw"] == "worsened").sum()),
                    "n_pearson_stable_vs_raw": int(group["pearson_status_vs_raw"].isin(["stable", "reference"]).sum()),
                })
    return pd.DataFrame(rows)


def build_specificity_agreement(
    profiles: dict[str, pd.DataFrame],
    reference: pd.DataFrame,
    metadata: pd.DataFrame,
    group_counts: pd.Series,
    min_cells: int,
    pseudocount: float,
    tolerance: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_tumor_types = [cell_type for cell_type in reference.columns if cell_type in TUMOR_TYPES]
    all_non_tumor_types = [cell_type for cell_type in reference.columns if cell_type not in TUMOR_TYPES]
    if len(all_tumor_types) != 6 or len(all_non_tumor_types) != 10:
        raise ValueError(
            f"Expected 6 tumor and 10 non-tumor types; got {len(all_tumor_types)} and {len(all_non_tumor_types)}"
        )
    meta = metadata.set_index("gene")
    scopes = {
        "all_types": list(reference.columns),
        f"robust_min{min_cells}": [
            cell_type for cell_type in reference.columns
            if group_counts[cell_type] >= min_cells
        ],
    }

    def score(
        frame: pd.DataFrame,
        gene: str,
        tumor_types: list[str],
        non_tumor_types: list[str],
    ) -> tuple[float, float, float]:
        tumor = float(frame.loc[gene, tumor_types].mean())
        non_tumor = float(frame.loc[gene, non_tumor_types].mean())
        if meta.loc[gene, "direction"] == "up":
            target, off_target = tumor, non_tumor
        else:
            target, off_target = non_tumor, tumor
        value = float(np.log2((target + pseudocount) / (off_target + pseudocount)))
        return value, target, off_target

    rows = []
    for scope, cell_types in scopes.items():
        tumor_types = [cell_type for cell_type in cell_types if cell_type in TUMOR_TYPES]
        non_tumor_types = [cell_type for cell_type in cell_types if cell_type not in TUMOR_TYPES]
        if len(tumor_types) < 2 or len(non_tumor_types) < 2:
            raise ValueError(f"{scope}: insufficient tumor/non-tumor cell types")
        reference_scores = {
            gene: score(reference, gene, tumor_types, non_tumor_types)
            for gene in metadata["gene"]
        }
        for method, profile in profiles.items():
            for gene in metadata["gene"]:
                spatial_score, target, off_target = score(
                    profile, gene, tumor_types, non_tumor_types
                )
                reference_score, reference_target, reference_off_target = reference_scores[gene]
                rows.append({
                    "scope": scope,
                    "n_tumor_types": len(tumor_types),
                    "n_non_tumor_types": len(non_tumor_types),
                    "method": method,
                    "gene": gene,
                    "panel": meta.loc[gene, "panel"],
                    "direction": meta.loc[gene, "direction"],
                    "spatial_specificity_log2": spatial_score,
                    "scrna_specificity_log2": reference_score,
                    "specificity_signed_error": spatial_score - reference_score,
                    "specificity_abs_error": abs(spatial_score - reference_score),
                    "spatial_target_mean": target,
                    "spatial_offtarget_mean": off_target,
                    "scrna_target_mean": reference_target,
                    "scrna_offtarget_mean": reference_off_target,
                })
    detail = pd.DataFrame(rows)
    raw_error = detail[detail["method"] == "RAW"].set_index(
        ["scope", "gene"]
    )["specificity_abs_error"]
    detail["raw_specificity_abs_error"] = [
        raw_error.loc[(scope, gene)]
        for scope, gene in zip(detail["scope"], detail["gene"])
    ]
    detail["specificity_error_reduction_vs_raw"] = (
        detail["raw_specificity_abs_error"] - detail["specificity_abs_error"]
    )
    reduction = detail["specificity_error_reduction_vs_raw"]
    detail["error_status_vs_raw"] = np.select(
        [reduction > tolerance, reduction < -tolerance],
        ["improved", "worsened"], default="stable",
    )
    detail.loc[detail["method"] == "RAW", "error_status_vs_raw"] = "reference"

    summary_rows = []
    for scope in scopes:
        scoped = detail[detail["scope"] == scope]
        for panel, panel_frame in panel_subsets(scoped).items():
            for method in METHOD_ORDER:
                group = panel_frame[panel_frame["method"] == method]
                summary_rows.append({
                    "scope": scope,
                    "panel": panel,
                    "method": method,
                    "n_markers": len(group),
                    "pearson_specificity_vs_scrna": safe_correlation(
                        group["spatial_specificity_log2"], group["scrna_specificity_log2"], "pearson"
                    ),
                    "spearman_specificity_vs_scrna": safe_correlation(
                        group["spatial_specificity_log2"], group["scrna_specificity_log2"], "spearman"
                    ),
                    "mean_absolute_error": group["specificity_abs_error"].mean(),
                    "median_absolute_error": group["specificity_abs_error"].median(),
                    "root_mean_squared_error": float(np.sqrt(np.mean(group["specificity_signed_error"] ** 2))),
                    "n_expected_direction_positive": int((group["spatial_specificity_log2"] > 0).sum()),
                    "expected_direction_fraction": float((group["spatial_specificity_log2"] > 0).mean()),
                    "mean_error_reduction_vs_raw": group["specificity_error_reduction_vs_raw"].mean(),
                    "median_error_reduction_vs_raw": group["specificity_error_reduction_vs_raw"].median(),
                    "n_error_improved_vs_raw": int((group["error_status_vs_raw"] == "improved").sum()),
                    "n_error_worsened_vs_raw": int((group["error_status_vs_raw"] == "worsened").sum()),
                    "n_error_stable_vs_raw": int(group["error_status_vs_raw"].isin(["stable", "reference"]).sum()),
                })
    return detail, pd.DataFrame(summary_rows)


def build_internal_reference_concordance(
    internal_path: Path,
    specificity: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Relate the original within-spatial Δ to scRNA error reduction."""
    internal = pd.read_csv(internal_path)
    required = {"method", "gene", "panel", "specificity_delta_vs_raw", "status_vs_raw"}
    missing = required.difference(internal.columns)
    if missing:
        raise ValueError(f"Internal specificity table missing: {sorted(missing)}")
    detail = internal[list(required)].merge(
        specificity[[
            "scope", "method", "gene", "specificity_error_reduction_vs_raw", "error_status_vs_raw"
        ]],
        on=["method", "gene"], how="inner", validate="one_to_many",
    )
    if len(detail) != len(specificity):
        raise ValueError(
            f"Internal/reference concordance join returned {len(detail)} rows; expected {len(specificity)}"
        )
    detail["interpretation"] = np.select(
        [
            (detail["status_vs_raw"] == "improved") & (detail["error_status_vs_raw"] == "improved"),
            (detail["status_vs_raw"] == "improved") & (detail["error_status_vs_raw"] == "worsened"),
            (detail["status_vs_raw"] == "worsened") & (detail["error_status_vs_raw"] == "improved"),
        ],
        [
            "internal_improvement_reference_supported",
            "internal_improvement_reference_contradicted",
            "internal_worsening_but_reference_improved",
        ],
        default="other_or_stable",
    )

    rows = []
    for scope in detail["scope"].unique():
        scoped = detail[detail["scope"] == scope]
        for panel, panel_frame in panel_subsets(scoped).items():
            for method in METHOD_ORDER[1:]:
                group = panel_frame[panel_frame["method"] == method]
                internal_improved = group["status_vs_raw"] == "improved"
                n_internal_improved = int(internal_improved.sum())
                n_supported = int((internal_improved & (group["error_status_vs_raw"] == "improved")).sum())
                n_contradicted = int((internal_improved & (group["error_status_vs_raw"] == "worsened")).sum())
                n_stable = int((internal_improved & (group["error_status_vs_raw"] == "stable")).sum())
                rows.append({
                    "scope": scope,
                    "panel": panel,
                    "method": method,
                    "n_markers": len(group),
                    "n_internal_improved": n_internal_improved,
                    "n_internal_improved_reference_supported": n_supported,
                    "n_internal_improved_reference_stable": n_stable,
                    "n_internal_improved_reference_contradicted": n_contradicted,
                    "reference_support_rate_among_internal_improvements": (
                        n_supported / n_internal_improved if n_internal_improved else np.nan
                    ),
                    "pearson_internal_delta_vs_reference_error_reduction": safe_correlation(
                        group["specificity_delta_vs_raw"],
                        group["specificity_error_reduction_vs_raw"],
                        "pearson",
                    ),
                })
    return detail, pd.DataFrame(rows)


def ordered_genes(metadata: pd.DataFrame) -> list[str]:
    genes = []
    for panel in PANEL_ORDER:
        genes.extend(metadata.loc[metadata["panel"] == panel, "gene"].tolist())
    return genes


def add_panel_separators(ax: plt.Axes, metadata: pd.DataFrame, genes: list[str]) -> None:
    panels = metadata.set_index("gene")["panel"]
    previous = panels.loc[genes[0]]
    for index, gene in enumerate(genes[1:], start=1):
        if panels.loc[gene] != previous:
            ax.axhline(index, color="#333333", linewidth=1.1)
            previous = panels.loc[gene]


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_delta_heatmap(
    detail: pd.DataFrame,
    metadata: pd.DataFrame,
    metric: str,
    title: str,
    colorbar_label: str,
    note: str,
    output_path: Path,
    color_limit: float | None = None,
) -> None:
    genes = ordered_genes(metadata)
    panels = metadata.set_index("gene")["panel"]
    labels = [f"{gene}  ·  {PANEL_LABELS[panels.loc[gene]]}" for gene in genes]
    matrix = detail.pivot(index="gene", columns="method", values=metric).reindex(
        index=genes, columns=METHOD_ORDER[1:]
    )
    vmax = (
        float(color_limit) if color_limit is not None
        else float(np.nanmax(np.abs(matrix.to_numpy())))
    )
    fig, ax = plt.subplots(figsize=(10.8, 15.5))
    sns.heatmap(
        matrix, cmap=SIGNED_CMAP, center=0, vmin=-vmax, vmax=vmax,
        annot=True, fmt="+.2f", annot_kws={"fontsize": 7},
        linewidths=0.4, linecolor="#eeeeee",
        cbar_kws={"label": colorbar_label}, ax=ax,
    )
    ax.set_yticklabels(labels, rotation=0, fontsize=8)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    ax.set_xlabel("Method")
    ax.set_ylabel("Marker")
    ax.set_title(title, loc="left", fontweight="bold")
    add_panel_separators(ax, metadata, genes)
    fig.text(0.01, 0.005, note, fontsize=9, color="#555555")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    save_figure(fig, output_path)


def plot_specificity_scatter(
    detail: pd.DataFrame, output_dir: Path
) -> None:
    all_values = pd.concat([
        detail["scrna_specificity_log2"], detail["spatial_specificity_log2"]
    ])
    low, high = float(all_values.min()), float(all_values.max())
    padding = (high - low) * 0.06
    low -= padding
    high += padding
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 9.5), sharex=True, sharey=True)
    for ax, method in zip(axes.flat, METHOD_ORDER):
        sub = detail[detail["method"] == method]
        for panel in PANEL_ORDER:
            panel_frame = sub[sub["panel"] == panel]
            ax.scatter(
                panel_frame["scrna_specificity_log2"],
                panel_frame["spatial_specificity_log2"],
                s=34, alpha=0.82, color=PANEL_COLORS[panel],
                edgecolors="white", linewidths=0.4,
                label=PANEL_LABELS[panel],
            )
        ax.plot([low, high], [low, high], color="#333333", linewidth=1, linestyle="--")
        worst = sub.nlargest(2, "specificity_abs_error")
        for annotation_index, row in enumerate(worst.itertuples()):
            ax.annotate(
                row.gene,
                (row.scrna_specificity_log2, row.spatial_specificity_log2),
                xytext=(4, 5 if annotation_index == 0 else -10),
                textcoords="offset points", fontsize=7,
            )
        corr = safe_correlation(
            sub["spatial_specificity_log2"], sub["scrna_specificity_log2"], "pearson"
        )
        mae = sub["specificity_abs_error"].mean()
        ax.set_title(f"{method}  ·  r={corr:.2f}, MAE={mae:.2f}", fontweight="bold")
        ax.set_xlim(low, high)
        ax.set_ylim(low, high)
        ax.grid(color="#e5e5e5", linewidth=0.7)
        ax.set_axisbelow(True)
    for ax in axes[-1]:
        ax.set_xlabel("scFFPE specificity (log2)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Spatial specificity (log2)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.945))
    fig.suptitle(
        "Ovarian marker specificity: spatial methods versus scFFPE reference",
        fontsize=15, fontweight="bold", y=0.99,
    )
    fig.text(
        0.5, 0.01,
        "Each point is one marker; dashed line is exact agreement. Profiles are balanced equally across 6 tumor and 10 non-tumor cell types.",
        ha="center", fontsize=9, color="#555555",
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.89))
    save_figure(fig, output_dir / "marker_specificity_vs_scrna_scatter")


def plot_internal_reference_concordance(
    detail: pd.DataFrame, output_dir: Path
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 10.0), sharex=True, sharey=True)
    methods = METHOD_ORDER[1:]
    for ax, method in zip(axes.flat, methods):
        sub = detail[detail["method"] == method]
        for panel in PANEL_ORDER:
            panel_frame = sub[sub["panel"] == panel]
            ax.scatter(
                panel_frame["specificity_delta_vs_raw"],
                panel_frame["specificity_error_reduction_vs_raw"],
                s=34, alpha=0.82, color=PANEL_COLORS[panel],
                edgecolors="white", linewidths=0.4,
                label=PANEL_LABELS[panel],
            )
        ax.axhline(0, color="#555555", linewidth=0.9)
        ax.axvline(0, color="#555555", linewidth=0.9)
        disagreement = sub.assign(
            disagreement=np.abs(
                sub["specificity_delta_vs_raw"]
                - sub["specificity_error_reduction_vs_raw"]
            )
        ).nlargest(3, "disagreement")
        for row in disagreement.itertuples():
            ax.annotate(
                row.gene,
                (row.specificity_delta_vs_raw, row.specificity_error_reduction_vs_raw),
                xytext=(4, 4), textcoords="offset points", fontsize=7,
            )
        corr = safe_correlation(
            sub["specificity_delta_vs_raw"],
            sub["specificity_error_reduction_vs_raw"],
            "pearson",
        )
        ax.set_title(f"{method}  ·  r={corr:.2f}", fontweight="bold")
        ax.grid(color="#e5e5e5", linewidth=0.7)
        ax.set_axisbelow(True)
    axes.flat[-1].axis("off")
    for ax in axes[1, :2]:
        ax.set_xlabel("Within-spatial specificity Δ vs RAW")
    axes[1, 2].set_xlabel("")
    for ax in axes[:, 0]:
        ax.set_ylabel("scFFPE absolute-error reduction vs RAW")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.92))
    fig.suptitle(
        "Within-spatial marker improvement versus scFFPE-reference support",
        fontsize=15, fontweight="bold", y=0.985,
    )
    fig.text(
        0.5, 0.01,
        "Top-right: both metrics improve. Bottom-right: spatial specificity improves but moves farther from the single-cell reference.",
        ha="center", fontsize=9, color="#555555",
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.85))
    save_figure(fig, output_dir / "marker_internal_vs_scrna_concordance")


def write_report(
    correlations: pd.DataFrame,
    correlation_summary: pd.DataFrame,
    specificity: pd.DataFrame,
    specificity_summary: pd.DataFrame,
    concordance: pd.DataFrame,
    concordance_summary: pd.DataFrame,
    scrna_logfc: pd.DataFrame,
    scrna_cell_counts: pd.Series,
    metadata: pd.DataFrame,
    group_counts: pd.Series,
    output_dir: Path,
    min_cells: int,
    tolerance: float,
) -> None:
    corr_all = correlation_summary[
        (correlation_summary["scope"] == "all_types")
        & (correlation_summary["panel"] == "all_markers")
    ].set_index("method").loc[METHOD_ORDER]
    corr_robust = correlation_summary[
        (correlation_summary["scope"] == f"robust_min{min_cells}")
        & (correlation_summary["panel"] == "all_markers")
    ].set_index("method").loc[METHOD_ORDER]
    spec_all = specificity_summary[
        (specificity_summary["scope"] == "all_types")
        & (specificity_summary["panel"] == "cancer_core")
    ].set_index("method").loc[METHOD_ORDER]
    spec_robust = specificity_summary[
        (specificity_summary["scope"] == f"robust_min{min_cells}")
        & (specificity_summary["panel"] == "cancer_core")
    ].set_index("method").loc[METHOD_ORDER]
    specificity_all = specificity[specificity["scope"] == "all_types"]
    concordance_all = concordance[concordance["scope"] == "all_types"]

    lines = [
        "# Ovarian cancer markers 与 scFFPE 单细胞参考的一致性",
        "",
        "该分析把空间内部的 tumor/non-tumor specificity 扩展为与配对 scFFPE 参考的直接比较。",
        "逐 marker 相关性是在匹配的细胞类型之间计算；同时比较 type-balanced specificity 的绝对误差，",
        "以避免只看相关性而忽略表达幅度偏差。",
        "",
        "## 预设肿瘤/基质 marker 是否得到单细胞支持",
        "",
        "主 log2FC 使用单细胞数加权的 mean log1p-CPM，并计算",
        "`log2((tumor + 0.01)/(non-tumor + 0.01))`。正值表示肿瘤富集，负值表示非肿瘤富集。",
        "这里的 non-tumor 与原空间脚本口径相同，包含全部 10 个非肿瘤类型；另给出只使用5个结构性基质类型的敏感性结果。",
        "",
    ]
    for panel in PANEL_ORDER[:2]:
        panel_frame = scrna_logfc[scrna_logfc["panel"] == panel]
        n_supported = int(panel_frame["supported_cell_weighted"].sum())
        expected = "positive" if panel == "epithelial_tumor" else "negative"
        lines.append(
            f"- **{PANEL_LABELS[panel]}**：{n_supported}/{len(panel_frame)} 的 cell-weighted log2FC 为预期的 {expected}；type-balanced 和 structural-stroma 口径方向完全一致。"
        )
    lines.extend([
        "",
        f"scFFPE 细胞数：tumor={int(scrna_cell_counts.loc[list(TUMOR_TYPES)].sum()):,}，"
        f"non-tumor={int(scrna_cell_counts.drop(list(TUMOR_TYPES)).sum()):,}，"
        f"structural stroma={int(scrna_cell_counts.loc[list(STRUCTURAL_STROMAL_TYPES)].sum()):,}。",
        "",
        "| Marker | Panel | Tumor mean | Non-tumor mean | log2FC tumor/non-tumor | Type-balanced log2FC | log2FC tumor/structural-stroma | Supported |",
        "|:--|:--|--:|--:|--:|--:|--:|:--:|",
    ])
    for row in scrna_logfc.itertuples(index=False):
        lines.append(
            f"| {row.gene} | {PANEL_LABELS[row.panel]} | "
            f"{row.scrna_tumor_mean_log1pcpm_weighted:.4f} | "
            f"{row.scrna_non_tumor_mean_log1pcpm_weighted:.4f} | "
            f"{row.scrna_log2fc_tumor_vs_non_tumor_weighted:+.3f} | "
            f"{row.scrna_log2fc_tumor_vs_non_tumor_type_balanced:+.3f} | "
            f"{row.scrna_log2fc_tumor_vs_structural_stroma_weighted:+.3f} | "
            f"{'yes' if row.supported_cell_weighted else 'no'} |"
        )
    lines.extend([
        "",
        "CLDN6 的方向得到支持，但肿瘤/非肿瘤均值仅 0.0255/0.0034；其 log2FC 幅度对 0.01 伪计数较敏感。",
        "FAP 在肿瘤中也很低（0.0154），但非肿瘤和结构性基质均值分别为 0.2730/0.4189，负方向清晰。",
        "",
        "## 六方法总体结果（32 markers）",
        "",
        "Cell-type correlation columns use all 32 markers; specificity columns use the 26-marker tumor+stromal core because the robust scope excludes the T & NK target of several immune markers.",
        "",
        "| Method | Median Pearson (16 types) | Median Δ vs RAW | Median Pearson (robust types) | Core specificity r | Core MAE (16 types) | Core MAE (robust) | Expected direction | Core error improved/worsened/stable |",
        "|:--|--:|--:|--:|--:|--:|--:|--:|:--|",
    ])
    for method in METHOD_ORDER:
        c = corr_all.loc[method]
        cr = corr_robust.loc[method]
        s = spec_all.loc[method]
        sr = spec_robust.loc[method]
        lines.append(
            f"| {method} | {c['median_pearson']:.3f} | "
            f"{c['median_pearson_delta_vs_raw']:+.3f} | {cr['median_pearson']:.3f} | "
            f"{s['pearson_specificity_vs_scrna']:.3f} | {s['mean_absolute_error']:.3f} | "
            f"{sr['mean_absolute_error']:.3f} | "
            f"{int(s['n_expected_direction_positive'])}/26 | "
            f"{int(s['n_error_improved_vs_raw'])}/{int(s['n_error_worsened_vs_raw'])}/{int(s['n_error_stable_vs_raw'])} |"
        )

    lines.extend([
        "",
        "## 分 panel 结果",
        "",
        "下表给出 16 类型逐 marker Pearson 的中位 Δ，以及 specificity 误差改善的基因数；两者越大越好。",
        "",
        "| Panel | Method | Median Pearson Δ | Corr improved/worsened/stable | Mean specificity error reduction | Error improved/worsened/stable |",
        "|:--|:--|--:|:--|--:|:--|",
    ])
    for panel in PANEL_ORDER:
        corr_panel = correlation_summary[
            (correlation_summary["scope"] == "all_types")
            & (correlation_summary["panel"] == panel)
        ].set_index("method")
        spec_panel = specificity_summary[
            (specificity_summary["scope"] == "all_types")
            & (specificity_summary["panel"] == panel)
        ].set_index("method")
        for method in METHOD_ORDER[1:]:
            c = corr_panel.loc[method]
            s = spec_panel.loc[method]
            lines.append(
                f"| {PANEL_LABELS[panel]} | {method} | {c['median_pearson_delta_vs_raw']:+.3f} | "
                f"{int(c['n_pearson_improved_vs_raw'])}/{int(c['n_pearson_worsened_vs_raw'])}/{int(c['n_pearson_stable_vs_raw'])} | "
                f"{s['mean_error_reduction_vs_raw']:+.3f} | "
                f"{int(s['n_error_improved_vs_raw'])}/{int(s['n_error_worsened_vs_raw'])}/{int(s['n_error_stable_vs_raw'])} |"
            )

    core_concordance_all = concordance_summary[
        (concordance_summary["scope"] == "all_types")
        & (concordance_summary["panel"] == "cancer_core")
    ].set_index("method").loc[METHOD_ORDER[1:]]
    core_concordance_robust = concordance_summary[
        (concordance_summary["scope"] == f"robust_min{min_cells}")
        & (concordance_summary["panel"] == "cancer_core")
    ].set_index("method").loc[METHOD_ORDER[1:]]
    lines.extend([
        "",
        "## 原空间 specificity 改善是否得到单细胞参考支持",
        "",
        "这里只考察原分析中被判为改善的肿瘤+基质 marker，并检查其到 scFFPE specificity 的绝对误差是否同时下降。",
        "`supported` 表示更接近参考，`contradicted` 表示反而更远；单细胞参考是外部一致性证据而非绝对真值。",
        "",
        "| Method | Internal improved | Supported/stable/contradicted (16 types) | Support rate | Supported/stable/contradicted (robust) | Support rate |",
        "|:--|--:|:--|--:|:--|--:|",
    ])
    for method, row in core_concordance_all.iterrows():
        robust_row = core_concordance_robust.loc[method]
        lines.append(
            f"| {method} | {int(row['n_internal_improved'])} | "
            f"{int(row['n_internal_improved_reference_supported'])}/"
            f"{int(row['n_internal_improved_reference_stable'])}/"
            f"{int(row['n_internal_improved_reference_contradicted'])} | "
            f"{row['reference_support_rate_among_internal_improvements']:.1%} | "
            f"{int(robust_row['n_internal_improved_reference_supported'])}/"
            f"{int(robust_row['n_internal_improved_reference_stable'])}/"
            f"{int(robust_row['n_internal_improved_reference_contradicted'])} | "
            f"{robust_row['reference_support_rate_among_internal_improvements']:.1%} |"
        )

    lines.extend([
        "",
        "对结论影响最大的矛盾 marker（空间内部 Δ>0，但相对 scFFPE 误差增加）包括：",
        "",
    ])
    for method in METHOD_ORDER[1:]:
        conflict = concordance_all[
            (concordance_all["method"] == method)
            & (concordance_all["panel"].isin(PANEL_ORDER[:2]))
            & (concordance_all["interpretation"] == "internal_improvement_reference_contradicted")
        ].sort_values("specificity_error_reduction_vs_raw")
        text = ", ".join(
            f"{row.gene} ({row.specificity_error_reduction_vs_raw:+.3f})"
            for row in conflict.itertuples()
        ) or "none"
        lines.append(f"- **{method}**：{text}。")

    lines.extend([
        "",
        "## 每种方法变化最大的 marker",
        "",
    ])
    all_corr = correlations[correlations["scope"] == "all_types"]
    for method in METHOD_ORDER[1:]:
        corr_method = all_corr[all_corr["method"] == method]
        spec_method = specificity_all[specificity_all["method"] == method]
        corr_up = ", ".join(
            f"{row.gene} ({row.pearson_delta_vs_raw:+.3f})"
            for row in corr_method.nlargest(5, "pearson_delta_vs_raw").itertuples()
        )
        corr_down = ", ".join(
            f"{row.gene} ({row.pearson_delta_vs_raw:+.3f})"
            for row in corr_method.nsmallest(5, "pearson_delta_vs_raw").itertuples()
        )
        error_up = ", ".join(
            f"{row.gene} ({row.specificity_error_reduction_vs_raw:+.3f})"
            for row in spec_method.nlargest(5, "specificity_error_reduction_vs_raw").itertuples()
        )
        error_down = ", ".join(
            f"{row.gene} ({row.specificity_error_reduction_vs_raw:+.3f})"
            for row in spec_method.nsmallest(5, "specificity_error_reduction_vs_raw").itertuples()
        )
        lines.extend([
            f"### {method}",
            "",
            f"- Cell-type correlation 最大改善：{corr_up}。",
            f"- Cell-type correlation 最大下降：{corr_down}。",
            f"- Specificity 误差最大改善：{error_up}。",
            f"- Specificity 误差最大恶化：{error_down}。",
            "",
        ])

    lines.extend([
        "## 完整逐 marker 对比",
        "",
        f"正值表示相对 RAW 改善；变化阈值为 |Δ|>{tolerance:g}。",
        "",
        "| Marker | Panel | SPARKLE corr Δ | SpotClean corr Δ | SoupX corr Δ | SpatialSoupX corr Δ | DecontX corr Δ | SPARKLE error reduction | SpotClean error reduction | DecontX error reduction |",
        "|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|",
    ])
    corr_wide = all_corr.pivot(index="gene", columns="method", values="pearson_delta_vs_raw")
    error_wide = specificity_all.pivot(index="gene", columns="method", values="specificity_error_reduction_vs_raw")
    meta = metadata.set_index("gene")
    for gene in ordered_genes(metadata):
        lines.append(
            f"| {gene} | {PANEL_LABELS[meta.loc[gene, 'panel']]} | "
            f"{corr_wide.loc[gene, 'SPARKLE']:+.3f} | {corr_wide.loc[gene, 'SpotClean']:+.3f} | "
            f"{corr_wide.loc[gene, 'SoupX']:+.3f} | {corr_wide.loc[gene, 'SpatialSoupX']:+.3f} | "
            f"{corr_wide.loc[gene, 'DecontX']:+.3f} | {error_wide.loc[gene, 'SPARKLE']:+.3f} | "
            f"{error_wide.loc[gene, 'SpotClean']:+.3f} | {error_wide.loc[gene, 'DecontX']:+.3f} |"
        )

    low_types = group_counts[group_counts < min_cells]
    low_type_text = ", ".join(f"{name} (n={count})" for name, count in low_types.items())
    lines.extend([
        "",
        "## 解释限制",
        "",
        f"- 空间窗口低样本类型：{low_type_text}；因此同时报告仅保留 n≥{min_cells} 类型的稳健结果。",
        "- 低样本类型包含若干基质/免疫 marker 的主要来源；全 16 类型保留生物学覆盖但噪声较大，robust 范围降低噪声但会遗漏这些来源。两者应作为敏感性边界共同解读。",
        "- CD3E、CD8A、NKG7 的单细胞最高表达类型为 T & NK Cells，但空间仅 2 个该类型细胞；其 16-type 相关性只能作为探索性证据。",
        "- Robust 范围移除了 T & NK Cells，导致 CD3E/CD8A/NKG7 缺少真实靶类型；因此总体 robust specificity 汇总只使用 26 个肿瘤+基质 marker，不用于免疫 panel 排名。",
        "- CLDN6 在单细胞 16 类型间动态范围很低，相关系数容易受微小数值变化影响。",
        "- 单细胞与空间来自不同实验平台；相关性和误差衡量的是参考一致性，不代表单细胞数据是绝对真值。",
        "- Type-balanced specificity 对每个细胞类型等权，避免两套数据不同细胞组成导致的混杂；它与原始按细胞汇总的 specificity 数值不可直接互换。",
    ])
    (output_dir / "marker_scrna_agreement_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default=TAG)
    parser.add_argument(
        "--input-dir", default="evaluation/reports/h5ad_ovarian_annotated"
    )
    parser.add_argument(
        "--marker-csv",
        default="evaluation/reports/ovarian_eval/marker_analysis/marker_foldchanges.csv",
    )
    parser.add_argument(
        "--scrna-reference",
        default="evaluation/data/ovarian/scrna_celltype_pseudobulk.csv",
    )
    parser.add_argument(
        "--scrna-annotation",
        default="evaluation/data/ovarian/FLEX_Ovarian_Barcode_Cluster_Annotation.csv",
    )
    parser.add_argument(
        "--internal-specificity-detail",
        default="evaluation/reports/ovarian_eval/marker_analysis/detailed/marker_gene_method_specificity.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="evaluation/reports/ovarian_eval/marker_analysis/scrna_agreement",
    )
    parser.add_argument("--min-cells", type=int, default=20)
    parser.add_argument("--pseudocount", type=float, default=0.01)
    parser.add_argument("--tolerance", type=float, default=0.01)
    parser.add_argument("--chunk-size", type=int, default=512)
    args = parser.parse_args()
    if args.min_cells < 1 or args.pseudocount <= 0 or args.tolerance < 0:
        raise ValueError("Invalid min-cells, pseudocount, or tolerance")

    marker_table = pd.read_csv(args.marker_csv)
    metadata = marker_table[marker_table["method"] == "RAW"][[
        "gene", "panel", "direction"
    ]].copy()
    if metadata["gene"].duplicated().any() or len(metadata) != 32:
        raise ValueError("Expected 32 unique RAW marker definitions")
    if set(metadata["panel"]) != set(PANEL_ORDER):
        raise ValueError("Unexpected marker panels")

    reference = pd.read_csv(args.scrna_reference, index_col=0)
    missing = set(metadata["gene"]).difference(reference.index)
    if missing:
        raise ValueError(f"Markers missing from scRNA reference: {sorted(missing)}")
    reference = reference.loc[metadata["gene"], :].astype(float)
    if reference.index.duplicated().any() or reference.columns.duplicated().any():
        raise ValueError("Duplicate scRNA gene or cell-type labels")
    if not np.isfinite(reference.to_numpy()).all() or (reference < 0).any().any():
        raise ValueError("Invalid scRNA reference values")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    profiles, group_counts = method_profiles(
        Path(args.input_dir), args.tag, metadata["gene"].tolist(),
        reference.columns.tolist(), args.chunk_size,
    )
    print(f"Matched {len(reference.columns)} cell types; robust n>={args.min_cells}: {(group_counts >= args.min_cells).sum()}")
    scrna_logfc, scrna_cell_counts = build_scrna_marker_logfc(
        reference, metadata, Path(args.scrna_annotation), args.pseudocount
    )

    profile_table = build_profile_table(
        profiles, reference, metadata, group_counts, args.min_cells
    )
    correlations = build_gene_correlations(
        profiles, reference, metadata, group_counts, args.min_cells, args.tolerance
    )
    correlation_summary = summarize_correlations(correlations)
    specificity, specificity_summary = build_specificity_agreement(
        profiles, reference, metadata, group_counts, args.min_cells,
        args.pseudocount, args.tolerance
    )
    concordance, concordance_summary = build_internal_reference_concordance(
        Path(args.internal_specificity_detail), specificity
    )

    profile_table.to_csv(output_dir / "marker_celltype_profiles_vs_scrna.csv", index=False)
    correlations.to_csv(output_dir / "marker_celltype_correlations_vs_scrna.csv", index=False)
    correlation_summary.to_csv(output_dir / "marker_celltype_correlation_summary.csv", index=False)
    specificity.to_csv(output_dir / "marker_specificity_agreement_vs_scrna.csv", index=False)
    specificity_summary.to_csv(output_dir / "marker_specificity_agreement_summary.csv", index=False)
    concordance.to_csv(output_dir / "marker_internal_specificity_vs_scrna_concordance.csv", index=False)
    concordance_summary.to_csv(output_dir / "marker_internal_specificity_vs_scrna_concordance_summary.csv", index=False)
    group_counts.rename("n_spatial_cells").to_csv(output_dir / "spatial_celltype_counts.csv")
    scrna_logfc.to_csv(
        output_dir / "preset_marker_scrna_tumor_stromal_log2fc.csv", index=False
    )
    scrna_cell_counts.rename("n_scrna_cells").to_csv(
        output_dir / "scrna_celltype_counts.csv"
    )

    all_corr = correlations[correlations["scope"] == "all_types"]
    all_specificity = specificity[specificity["scope"] == "all_types"]
    all_concordance = concordance[concordance["scope"] == "all_types"]
    plot_delta_heatmap(
        all_corr, metadata, "pearson_delta_vs_raw",
        "Per-marker cell-type correlation change versus RAW",
        "Pearson correlation change vs RAW",
        "Blue/positive means closer to scFFPE across 16 types. Color is capped at ±0.5 for visibility; exact annotations are not clipped.",
        output_dir / "marker_celltype_correlation_delta_heatmap",
        color_limit=0.5,
    )
    plot_delta_heatmap(
        all_specificity, metadata, "specificity_error_reduction_vs_raw",
        "Per-marker scFFPE specificity-error reduction versus RAW",
        "Absolute-error reduction vs RAW",
        "Blue/positive means type-balanced tumor/non-tumor specificity moved closer to the scFFPE reference.",
        output_dir / "marker_specificity_error_reduction_heatmap",
    )
    plot_specificity_scatter(all_specificity, output_dir)
    plot_internal_reference_concordance(all_concordance, output_dir)
    write_report(
        correlations, correlation_summary, specificity, specificity_summary,
        concordance, concordance_summary, scrna_logfc, scrna_cell_counts,
        metadata, group_counts, output_dir, args.min_cells, args.tolerance,
    )
    print(f"Outputs written to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
