#!/usr/bin/env python3
"""Evaluate ovarian tumor/stroma contrast using genes discovered in scRNA-seq.

The gene panel is selected only from the paired scFFPE single-cell count matrix:

1. Compare the six tumor cell types with five structural stromal cell types.
2. Normalize each cell to CP10K and use Welch's t test on log1p-CP10K.
3. Keep independently selected tumor-up and stroma-up genes after expression,
   log2FC, and Benjamini-Hochberg FDR filters.
4. Measure tumor/stroma log2FC in RAW and five correction methods, orient each
   gene by its scRNA direction, and report contrast change relative to RAW.

Positive ``contrast_delta_vs_raw`` means that the method strengthened the
scRNA-supported direction.  The spatial data never participate in gene
selection, which avoids circularly choosing genes that favor one method.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import anndata as ad
import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import sparse
from scipy.stats import t as student_t
from scipy.stats import wilcoxon


TAG = "ovarian_x1000-1800_y300-1100"
METHOD_ORDER = ["RAW", "SPARKLE", "SpotClean", "SoupX", "SpatialSoupX", "DecontX"]
METHOD_SUFFIXES = {"RAW": "raw", "SpotClean": "SpotCleanOfficial"}
TUMOR_TYPES = {
    "Tumor Cells",
    "Proliferative Tumor Cells",
    "VEGFA+ Tumor Cells",
    "MT-High, Jun+-Fos+ Tumor Cells",
    "Inflammatory Tumor Cells",
    "Malignant Cells Lining Cyst",
}
STROMAL_TYPES = {
    "Tumor Associated Fibroblasts",
    "Stromal Associated Fibroblasts",
    "Endothelial Cells",
    "Pericytes",
    "Smooth Muscle Cells",
}
DIRECTION_ORDER = ["tumor_up", "stroma_up"]
DIRECTION_LABELS = {"tumor_up": "Tumor-up", "stroma_up": "Stroma-up"}
METHOD_COLORS = {
    "SPARKLE": "#4477AA",
    "SpotClean": "#EE6677",
    "SoupX": "#228833",
    "SpatialSoupX": "#CCBB44",
    "DecontX": "#AA3377",
}


def bh_adjust(pvalues: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjustment, preserving NaNs."""
    pvalues = np.asarray(pvalues, dtype=float)
    adjusted = np.full_like(pvalues, np.nan)
    finite = np.isfinite(pvalues)
    p = pvalues[finite]
    if len(p) == 0:
        return adjusted
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    restored = np.empty_like(q)
    restored[order] = np.clip(q, 0.0, 1.0)
    adjusted[finite] = restored
    return adjusted


def load_10x_h5(path: Path) -> tuple[sparse.csc_matrix, np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as handle:
        matrix = handle["matrix"]
        data = matrix["data"][:]
        indices = matrix["indices"][:]
        indptr = matrix["indptr"][:]
        shape = tuple(int(value) for value in matrix["shape"][:])
        barcodes = np.array([value.decode() for value in matrix["barcodes"][:]])
        genes = np.array([value.decode() for value in matrix["features/name"][:]])
    return sparse.csc_matrix((data, indices, indptr), shape=shape), genes, barcodes


def collapse_duplicate_gene_counts(
    matrix: sparse.csr_matrix, genes: np.ndarray
) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Sum count columns sharing the same gene symbol."""
    unique_genes, inverse = np.unique(genes.astype(str), return_inverse=True)
    if len(unique_genes) == len(genes):
        return matrix, unique_genes
    mapping = sparse.csr_matrix(
        (np.ones(len(genes)), (np.arange(len(genes)), inverse)),
        shape=(len(genes), len(unique_genes)),
    )
    return (matrix @ mapping).tocsr(), unique_genes


def sparse_log_moments(
    matrix: sparse.csr_matrix, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    block = matrix[mask]
    n = block.shape[0]
    if n < 2:
        raise ValueError("Each scRNA comparison group must contain at least two cells")
    mean = np.asarray(block.sum(axis=0)).ravel() / n
    sum_squares = np.asarray(block.power(2).sum(axis=0)).ravel()
    variance = np.maximum((sum_squares - n * mean**2) / (n - 1), 0.0)
    return mean, variance


def is_technical_gene(gene: str) -> bool:
    return bool(re.match(r"^(MT-|RPL\d|RPS\d)", gene.upper()))


def discover_scrna_de_genes(
    scrna_h5: Path,
    annotation_path: Path,
    spatial_genes: pd.Index,
    target_sum: float,
    pseudocount: float,
    min_detection: float,
    min_target_mean: float,
    min_abs_log2fc: float,
    max_fdr: float,
    top_n: int,
    exclude_technical: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    print(f"Loading scRNA counts: {scrna_h5}")
    count_matrix, genes, barcodes = load_10x_h5(scrna_h5)
    annotation = pd.read_csv(annotation_path)
    required = {"Barcode", "Cell Annotation"}
    missing = required.difference(annotation.columns)
    if missing:
        raise ValueError(f"scRNA annotation is missing columns: {sorted(missing)}")
    annotation = annotation.dropna(subset=list(required)).copy()
    if annotation["Barcode"].duplicated().any():
        conflicts = annotation.groupby("Barcode")["Cell Annotation"].nunique()
        if (conflicts > 1).any():
            raise ValueError("Some scRNA barcodes have conflicting annotations")
        annotation = annotation.drop_duplicates("Barcode")
    barcode_to_type = dict(zip(annotation["Barcode"], annotation["Cell Annotation"]))
    labels = np.array([
        str(barcode_to_type.get(barcode, "")).replace("/", "-")
        for barcode in barcodes
    ])
    tumor_mask_all = np.isin(labels, list(TUMOR_TYPES))
    stroma_mask_all = np.isin(labels, list(STROMAL_TYPES))
    keep_cells = tumor_mask_all | stroma_mask_all
    if not keep_cells.any():
        raise ValueError("No annotated tumor or structural stromal scRNA cells found")
    labels = labels[keep_cells]
    cells_by_genes = count_matrix[:, keep_cells].T.tocsr().astype(np.float64)
    cells_by_genes, genes = collapse_duplicate_gene_counts(cells_by_genes, genes)
    tumor_mask = np.isin(labels, list(TUMOR_TYPES))
    stroma_mask = np.isin(labels, list(STROMAL_TYPES))

    totals = np.asarray(cells_by_genes.sum(axis=1)).ravel()
    scale = np.divide(
        target_sum, totals, out=np.zeros_like(totals), where=totals > 0
    )
    cp10k = cells_by_genes.multiply(scale[:, None]).tocsr()
    log1p_cp10k = cp10k.copy()
    log1p_cp10k.data = np.log1p(log1p_cp10k.data)

    n_tumor = int(tumor_mask.sum())
    n_stroma = int(stroma_mask.sum())
    tumor_mean_cp10k = np.asarray(cp10k[tumor_mask].mean(axis=0)).ravel()
    stroma_mean_cp10k = np.asarray(cp10k[stroma_mask].mean(axis=0)).ravel()
    tumor_detection = np.asarray(
        cells_by_genes[tumor_mask].getnnz(axis=0), dtype=float
    ) / n_tumor
    stroma_detection = np.asarray(
        cells_by_genes[stroma_mask].getnnz(axis=0), dtype=float
    ) / n_stroma
    tumor_log_mean, tumor_log_var = sparse_log_moments(log1p_cp10k, tumor_mask)
    stroma_log_mean, stroma_log_var = sparse_log_moments(log1p_cp10k, stroma_mask)

    standard_error_squared = tumor_log_var / n_tumor + stroma_log_var / n_stroma
    standard_error = np.sqrt(standard_error_squared)
    statistic = np.divide(
        tumor_log_mean - stroma_log_mean,
        standard_error,
        out=np.zeros_like(standard_error),
        where=standard_error > 0,
    )
    numerator = standard_error_squared**2
    denominator = (
        (tumor_log_var / n_tumor) ** 2 / (n_tumor - 1)
        + (stroma_log_var / n_stroma) ** 2 / (n_stroma - 1)
    )
    degrees_freedom = np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.inf),
        where=denominator > 0,
    )
    pvalue = 2.0 * student_t.sf(np.abs(statistic), degrees_freedom)
    log2fc = np.log2(
        (tumor_mean_cp10k + pseudocount)
        / (stroma_mean_cp10k + pseudocount)
    )

    table = pd.DataFrame({
        "gene": genes,
        "scrna_tumor_mean_cp10k": tumor_mean_cp10k,
        "scrna_stroma_mean_cp10k": stroma_mean_cp10k,
        "scrna_tumor_detection_fraction": tumor_detection,
        "scrna_stroma_detection_fraction": stroma_detection,
        "scrna_mean_log1p_cp10k_tumor": tumor_log_mean,
        "scrna_mean_log1p_cp10k_stroma": stroma_log_mean,
        "scrna_log2fc_tumor_vs_stroma": log2fc,
        "welch_t": statistic,
        "welch_pvalue": pvalue,
    })
    table["present_in_spatial"] = table["gene"].isin(spatial_genes)
    table["technical_gene"] = table["gene"].map(is_technical_gene)
    enriched_detection = np.where(
        table["scrna_log2fc_tumor_vs_stroma"] >= 0,
        table["scrna_tumor_detection_fraction"],
        table["scrna_stroma_detection_fraction"],
    )
    enriched_mean = np.where(
        table["scrna_log2fc_tumor_vs_stroma"] >= 0,
        table["scrna_tumor_mean_cp10k"],
        table["scrna_stroma_mean_cp10k"],
    )
    testable = (
        table["present_in_spatial"]
        & (enriched_detection >= min_detection)
        & (enriched_mean >= min_target_mean)
    )
    if exclude_technical:
        testable &= ~table["technical_gene"]
    table["tested"] = testable
    table["welch_fdr"] = np.nan
    table.loc[testable, "welch_fdr"] = bh_adjust(
        table.loc[testable, "welch_pvalue"].to_numpy()
    )
    table["direction"] = np.where(
        table["scrna_log2fc_tumor_vs_stroma"] >= 0, "tumor_up", "stroma_up"
    )
    table["passes_de_filters"] = (
        table["tested"]
        & (table["welch_fdr"] <= max_fdr)
        & (table["scrna_log2fc_tumor_vs_stroma"].abs() >= min_abs_log2fc)
    )
    candidates = table[table["passes_de_filters"]].copy()
    selected_parts = []
    for direction in DIRECTION_ORDER:
        group = candidates[candidates["direction"] == direction].copy()
        group = group.sort_values(
            ["scrna_log2fc_tumor_vs_stroma", "welch_fdr", "gene"],
            ascending=[direction == "stroma_up", True, True],
        ).head(top_n)
        group["direction_rank"] = np.arange(1, len(group) + 1)
        selected_parts.append(group)
    selected = pd.concat(selected_parts, ignore_index=True)
    selected["direction_sign"] = np.where(selected["direction"] == "tumor_up", 1, -1)
    selected["scrna_oriented_contrast"] = (
        selected["direction_sign"] * selected["scrna_log2fc_tumor_vs_stroma"]
    )
    selected["selected"] = True
    table = table.merge(
        selected[["gene", "direction_rank", "selected"]],
        on="gene", how="left", validate="one_to_one",
    )
    table["selected"] = table["selected"].eq(True)
    counts = {
        "n_scRNA_tumor_cells": n_tumor,
        "n_scRNA_stroma_cells": n_stroma,
        "n_tested_genes": int(testable.sum()),
        "n_de_candidates": len(candidates),
        "n_selected_tumor_up": int((selected["direction"] == "tumor_up").sum()),
        "n_selected_stroma_up": int((selected["direction"] == "stroma_up").sum()),
    }
    print(
        "scRNA DE: "
        f"{n_tumor} tumor cells, {n_stroma} stromal cells; "
        f"selected {counts['n_selected_tumor_up']} tumor-up and "
        f"{counts['n_selected_stroma_up']} stroma-up genes"
    )
    return table, selected, counts


def extract_spatial_group_means(
    path: Path,
    genes: list[str],
    chunk_size: int,
    target_sum: float,
    min_type_cells: int,
) -> tuple[dict[str, np.ndarray], pd.Series, pd.Index, np.ndarray]:
    adata = ad.read_h5ad(path, backed="r")
    try:
        if "annotation" not in adata.obs:
            raise ValueError(f"{path.name}: missing obs['annotation']")
        annotations = adata.obs["annotation"].astype(str).to_numpy()
        tumor_mask = np.isin(annotations, list(TUMOR_TYPES))
        stroma_mask = np.isin(annotations, list(STROMAL_TYPES))
        if not tumor_mask.any() or not stroma_mask.any():
            raise ValueError(f"{path.name}: tumor or stromal spatial cells are absent")
        gene_indices = adata.var_names.get_indexer(genes)
        if (gene_indices < 0).any():
            missing = np.asarray(genes)[gene_indices < 0].tolist()
            raise ValueError(f"{path.name}: missing selected genes {missing[:10]}")

        relevant_types = sorted(TUMOR_TYPES | STROMAL_TYPES)
        type_counts = pd.Series(annotations).value_counts().reindex(
            relevant_types, fill_value=0
        ).astype(int)
        sums = {cell_type: np.zeros(len(genes), dtype=np.float64) for cell_type in relevant_types}
        for start in range(0, adata.n_obs, chunk_size):
            stop = min(start + chunk_size, adata.n_obs)
            block = adata.X[start:stop]
            if sparse.issparse(block):
                block = block.toarray()
            block = np.asarray(block, dtype=np.float64)
            np.maximum(block, 0.0, out=block)
            totals = block.sum(axis=1)
            scale = np.divide(
                target_sum, totals, out=np.zeros_like(totals), where=totals > 0
            )
            selected = block[:, gene_indices] * scale[:, None]
            chunk_annotations = annotations[start:stop]
            for cell_type in relevant_types:
                mask = chunk_annotations == cell_type
                if mask.any():
                    sums[cell_type] += selected[mask].sum(axis=0)

        type_means = {
            cell_type: sums[cell_type] / type_counts[cell_type]
            for cell_type in relevant_types if type_counts[cell_type] > 0
        }
        tumor_sum = sum(sums[cell_type] for cell_type in TUMOR_TYPES)
        stroma_sum = sum(sums[cell_type] for cell_type in STROMAL_TYPES)
        tumor_count = int(type_counts.reindex(list(TUMOR_TYPES)).sum())
        stroma_count = int(type_counts.reindex(list(STROMAL_TYPES)).sum())
        robust_tumor = [
            cell_type for cell_type in TUMOR_TYPES
            if type_counts[cell_type] >= min_type_cells
        ]
        robust_stroma = [
            cell_type for cell_type in STROMAL_TYPES
            if type_counts[cell_type] >= min_type_cells
        ]
        if not robust_tumor or not robust_stroma:
            raise ValueError(
                f"{path.name}: no robust tumor/stromal types at n>={min_type_cells}"
            )
        results = {
            "cell_weighted_tumor": tumor_sum / tumor_count,
            "cell_weighted_stroma": stroma_sum / stroma_count,
            "type_balanced_tumor": np.mean(
                [type_means[cell_type] for cell_type in robust_tumor], axis=0
            ),
            "type_balanced_stroma": np.mean(
                [type_means[cell_type] for cell_type in robust_stroma], axis=0
            ),
        }
        results["n_cell_weighted_tumor"] = np.array([tumor_count])
        results["n_cell_weighted_stroma"] = np.array([stroma_count])
        results["n_type_balanced_tumor"] = np.array([len(robust_tumor)])
        results["n_type_balanced_stroma"] = np.array([len(robust_stroma)])
        return results, type_counts, adata.obs_names.copy(), annotations.copy()
    finally:
        adata.file.close()


def build_spatial_contrast_table(
    input_dir: Path,
    tag: str,
    selected: pd.DataFrame,
    chunk_size: int,
    target_sum: float,
    pseudocount: float,
    min_type_cells: int,
    tolerance: float,
) -> tuple[pd.DataFrame, pd.Series]:
    genes = selected["gene"].tolist()
    rows = []
    reference_obs_names = None
    reference_annotations = None
    reference_type_counts = None
    selected_index = selected.set_index("gene")
    for method in METHOD_ORDER:
        suffix = METHOD_SUFFIXES.get(method, method)
        path = input_dir / f"{tag}_{suffix}.h5ad"
        if not path.exists():
            raise FileNotFoundError(path)
        print(f"Reading spatial contrast: {method} ({path.name})")
        means, type_counts, obs_names, annotations = extract_spatial_group_means(
            path, genes, chunk_size, target_sum, min_type_cells
        )
        if reference_obs_names is None:
            reference_obs_names = obs_names
            reference_annotations = annotations
            reference_type_counts = type_counts
        else:
            if not obs_names.equals(reference_obs_names):
                raise ValueError(f"{method}: spatial cell order differs from RAW")
            if not np.array_equal(annotations, reference_annotations):
                raise ValueError(f"{method}: spatial annotations differ from RAW")
            if not type_counts.equals(reference_type_counts):
                raise ValueError(f"{method}: spatial annotation counts differ from RAW")

        for scope in ["cell_weighted", "type_balanced"]:
            tumor_mean = means[f"{scope}_tumor"]
            stroma_mean = means[f"{scope}_stroma"]
            log2fc = np.log2((tumor_mean + pseudocount) / (stroma_mean + pseudocount))
            for index, gene in enumerate(genes):
                meta = selected_index.loc[gene]
                rows.append({
                    "scope": scope,
                    "method": method,
                    "gene": gene,
                    "direction": meta["direction"],
                    "direction_rank": int(meta["direction_rank"]),
                    "direction_sign": int(meta["direction_sign"]),
                    "scrna_log2fc_tumor_vs_stroma": float(
                        meta["scrna_log2fc_tumor_vs_stroma"]
                    ),
                    "spatial_tumor_mean_cp10k": float(tumor_mean[index]),
                    "spatial_stroma_mean_cp10k": float(stroma_mean[index]),
                    "spatial_log2fc_tumor_vs_stroma": float(log2fc[index]),
                    "oriented_contrast": float(meta["direction_sign"] * log2fc[index]),
                })
    detail = pd.DataFrame(rows)
    raw = detail[detail["method"] == "RAW"].set_index(["scope", "gene"])
    detail["raw_spatial_log2fc"] = [
        raw.loc[(scope, gene), "spatial_log2fc_tumor_vs_stroma"]
        for scope, gene in zip(detail["scope"], detail["gene"])
    ]
    detail["raw_oriented_contrast"] = [
        raw.loc[(scope, gene), "oriented_contrast"]
        for scope, gene in zip(detail["scope"], detail["gene"])
    ]
    detail["contrast_delta_vs_raw"] = (
        detail["oriented_contrast"] - detail["raw_oriented_contrast"]
    )
    detail["contrast_status_vs_raw"] = np.select(
        [
            detail["contrast_delta_vs_raw"] > tolerance,
            detail["contrast_delta_vs_raw"] < -tolerance,
        ],
        ["enhanced", "weakened"], default="stable",
    )
    detail.loc[detail["method"] == "RAW", "contrast_status_vs_raw"] = "reference"
    detail["direction_supported"] = detail["oriented_contrast"] > 0
    detail["scrna_absolute_error"] = (
        detail["spatial_log2fc_tumor_vs_stroma"]
        - detail["scrna_log2fc_tumor_vs_stroma"]
    ).abs()
    raw_error = detail[detail["method"] == "RAW"].set_index(["scope", "gene"])[
        "scrna_absolute_error"
    ]
    detail["raw_scrna_absolute_error"] = [
        raw_error.loc[(scope, gene)]
        for scope, gene in zip(detail["scope"], detail["gene"])
    ]
    detail["scrna_error_reduction_vs_raw"] = (
        detail["raw_scrna_absolute_error"] - detail["scrna_absolute_error"]
    )
    return detail, reference_type_counts


def safe_wilcoxon(values: pd.Series) -> tuple[float, float]:
    values = values[np.isfinite(values)].to_numpy(float)
    values = values[np.abs(values) > 1e-12]
    if len(values) == 0:
        return np.nan, np.nan
    result = wilcoxon(values, alternative="two-sided")
    return float(result.statistic), float(result.pvalue)


def summarize_contrasts(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scope in ["cell_weighted", "type_balanced"]:
        scoped = detail[detail["scope"] == scope]
        for direction in ["all", *DIRECTION_ORDER]:
            subset = scoped if direction == "all" else scoped[scoped["direction"] == direction]
            for method in METHOD_ORDER:
                group = subset[subset["method"] == method]
                statistic, pvalue = safe_wilcoxon(group["contrast_delta_vs_raw"])
                rows.append({
                    "scope": scope,
                    "direction": direction,
                    "method": method,
                    "n_genes": len(group),
                    "mean_oriented_contrast": group["oriented_contrast"].mean(),
                    "median_oriented_contrast": group["oriented_contrast"].median(),
                    "mean_contrast_delta_vs_raw": group["contrast_delta_vs_raw"].mean(),
                    "median_contrast_delta_vs_raw": group["contrast_delta_vs_raw"].median(),
                    "n_enhanced": int((group["contrast_status_vs_raw"] == "enhanced").sum()),
                    "n_weakened": int((group["contrast_status_vs_raw"] == "weakened").sum()),
                    "n_stable_or_reference": int(group["contrast_status_vs_raw"].isin(["stable", "reference"]).sum()),
                    "direction_supported_fraction": group["direction_supported"].mean(),
                    "median_scrna_error_reduction_vs_raw": group["scrna_error_reduction_vs_raw"].median(),
                    "fraction_closer_to_scrna": (group["scrna_error_reduction_vs_raw"] > 0).mean(),
                    "wilcoxon_statistic_delta_vs_zero": statistic,
                    "wilcoxon_pvalue_delta_vs_zero": pvalue,
                })
    return pd.DataFrame(rows)


def save_figure(fig: plt.Figure, path_stem: Path) -> None:
    fig.savefig(path_stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(path_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_contrast_delta(detail: pd.DataFrame, output_dir: Path) -> None:
    data = detail[
        (detail["scope"] == "cell_weighted") & (detail["method"] != "RAW")
    ].copy()
    data["direction_label"] = data["direction"].map(DIRECTION_LABELS)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw={"width_ratios": [1.7, 1]})
    sns.boxplot(
        data=data, x="method", y="contrast_delta_vs_raw", hue="direction_label",
        order=METHOD_ORDER[1:], hue_order=["Tumor-up", "Stroma-up"],
        palette={"Tumor-up": "#4477AA", "Stroma-up": "#CC6677"},
        showfliers=False, ax=axes[0],
    )
    axes[0].axhline(0, color="black", linewidth=0.8, linestyle="--")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Oriented tumor–stroma contrast change vs RAW (log2FC)")
    axes[0].tick_params(axis="x", rotation=22)
    axes[0].legend(title="scRNA DE direction", frameon=False)

    fractions = pd.crosstab(
        data["method"], data["contrast_status_vs_raw"], normalize="index"
    ).reindex(
        index=METHOD_ORDER[1:], columns=["enhanced", "stable", "weakened"],
        fill_value=0.0,
    )
    bottom = np.zeros(len(METHOD_ORDER) - 1)
    colors = {"enhanced": "#4477AA", "stable": "#BBBBBB", "weakened": "#CC6677"}
    for status in ["enhanced", "stable", "weakened"]:
        values = fractions[status]
        axes[1].bar(METHOD_ORDER[1:], values, bottom=bottom, color=colors[status], label=status.capitalize())
        bottom += values.to_numpy()
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Fraction of scRNA-selected genes")
    axes[1].set_xlabel("")
    axes[1].tick_params(axis="x", rotation=22)
    axes[1].legend(frameon=False)
    fig.suptitle("Ovarian scRNA-derived DE genes: spatial contrast change relative to RAW", fontsize=14)
    fig.text(
        0.5, -0.02,
        "Positive delta strengthens the scRNA-supported tumor/stroma direction; cell-weighted spatial result.",
        ha="center", fontsize=9,
    )
    fig.tight_layout()
    save_figure(fig, output_dir / "scrna_de_contrast_change_vs_raw")


def plot_gene_heatmap(
    detail: pd.DataFrame, selected: pd.DataFrame, output_dir: Path, n_per_direction: int
) -> None:
    top = (
        selected.sort_values(["direction", "direction_rank"])
        .groupby("direction", sort=False).head(n_per_direction)
    )
    genes = []
    for direction in DIRECTION_ORDER:
        genes.extend(top[top["direction"] == direction].sort_values("direction_rank")["gene"])
    data = detail[
        (detail["scope"] == "cell_weighted")
        & (detail["method"] != "RAW")
        & detail["gene"].isin(genes)
    ]
    matrix = data.pivot(index="gene", columns="method", values="contrast_delta_vs_raw").reindex(
        index=genes, columns=METHOD_ORDER[1:]
    )
    finite = np.abs(matrix.to_numpy()[np.isfinite(matrix.to_numpy())])
    limit = max(float(np.quantile(finite, 0.95)), 0.25) if len(finite) else 1.0
    fig, ax = plt.subplots(figsize=(8.2, max(8.0, 0.28 * len(genes))))
    sns.heatmap(
        matrix, cmap="vlag", center=0, vmin=-limit, vmax=limit,
        linewidths=0.2, linecolor="white", cbar_kws={"label": "Contrast delta vs RAW"}, ax=ax,
    )
    ax.axhline((top["direction"] == "tumor_up").sum(), color="black", linewidth=1.2)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(
        f"Top {n_per_direction} scRNA DE genes per direction: spatial contrast change"
    )
    fig.tight_layout()
    save_figure(fig, output_dir / "scrna_de_gene_contrast_delta_heatmap")


def format_number(value: float, digits: int = 3, signed: bool = False) -> str:
    if not np.isfinite(value):
        return "NA"
    return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"


def format_pvalue(value: float) -> str:
    if not np.isfinite(value):
        return "NA"
    return f"{value:.2e}"


def write_report(
    selected: pd.DataFrame,
    detail: pd.DataFrame,
    summary: pd.DataFrame,
    sc_counts: dict[str, int],
    spatial_type_counts: pd.Series,
    args: argparse.Namespace,
    output_dir: Path,
) -> None:
    primary = summary[
        (summary["scope"] == "cell_weighted") & (summary["direction"] == "all")
    ].set_index("method")
    sensitivity = summary[
        (summary["scope"] == "type_balanced") & (summary["direction"] == "all")
    ].set_index("method")
    lines = [
        "# scRNA-derived tumor/stroma DE genes：空间对比度变化",
        "",
        "## 设计",
        "",
        "基因仅由配对 scFFPE 单细胞数据筛选；空间 RAW 或任一校正方法均不参与选基因。",
        "单细胞比较为 6 类肿瘤细胞对 5 类结构性基质细胞，免疫和其他正常上皮类型不进入比较。",
        "空间评价沿用相同类型集合，并固定使用同一套 RCTD annotation。",
        "",
        f"- scRNA 细胞：tumor={sc_counts['n_scRNA_tumor_cells']:,}，structural stroma={sc_counts['n_scRNA_stroma_cells']:,}。",
        f"- 筛选：target detection ≥ {args.min_detection:g}，target mean CP10K ≥ {args.min_target_mean:g}，|log2FC| ≥ {args.min_abs_log2fc:g}，BH FDR ≤ {args.max_fdr:g}。",
        f"- 最终 panel：tumor-up={sc_counts['n_selected_tumor_up']}，stroma-up={sc_counts['n_selected_stroma_up']}；每个方向最多 {args.top_n_per_direction} 个。",
        f"- 默认排除线粒体和核糖体基因：{'是' if not args.include_technical_genes else '否'}。",
        f"- 对比度：按 scRNA 方向定向的 spatial tumor/stroma log2FC；Δ>{args.tolerance:g} 为增强，Δ<-{args.tolerance:g} 为减弱。",
        "",
        "## 主结果：按空间细胞数加权",
        "",
        "| Method | Median contrast | Median Δ vs RAW | Enhanced / Stable / Weakened | Direction supported | Median scRNA-error reduction | Closer to scRNA | Wilcoxon P |",
        "|:--|--:|--:|:--|--:|--:|--:|--:|",
    ]
    for method in METHOD_ORDER:
        row = primary.loc[method]
        lines.append(
            f"| {method} | {format_number(row.median_oriented_contrast)} | "
            f"{format_number(row.median_contrast_delta_vs_raw, signed=True)} | "
            f"{int(row.n_enhanced)} / {int(row.n_stable_or_reference)} / {int(row.n_weakened)} | "
            f"{row.direction_supported_fraction:.1%} | "
            f"{format_number(row.median_scrna_error_reduction_vs_raw, signed=True)} | "
            f"{row.fraction_closer_to_scrna:.1%} | "
            f"{format_pvalue(row.wilcoxon_pvalue_delta_vs_zero)} |"
        )
    lines.extend([
        "",
        "正的 Δ 只表示增强了 scRNA 支持的方向；不必然表示更接近 scRNA 的效应量。后两列单独衡量参考误差。",
        "",
        "## 主结果分方向",
        "",
        "| Direction | Method | Median Δ vs RAW | Enhanced / Stable / Weakened | Median scRNA-error reduction | Closer to scRNA |",
        "|:--|:--|--:|:--|--:|--:|",
    ])
    direction_primary = summary[
        (summary["scope"] == "cell_weighted")
        & summary["direction"].isin(DIRECTION_ORDER)
        & (summary["method"] != "RAW")
    ]
    for direction in DIRECTION_ORDER:
        for method in METHOD_ORDER[1:]:
            row = direction_primary[
                (direction_primary["direction"] == direction)
                & (direction_primary["method"] == method)
            ].iloc[0]
            lines.append(
                f"| {DIRECTION_LABELS[direction]} | {method} | "
                f"{format_number(row.median_contrast_delta_vs_raw, signed=True)} | "
                f"{int(row.n_enhanced)} / {int(row.n_stable_or_reference)} / {int(row.n_weakened)} | "
                f"{format_number(row.median_scrna_error_reduction_vs_raw, signed=True)} | "
                f"{row.fraction_closer_to_scrna:.1%} |"
            )
    lines.extend([
        "",
        "分方向结果用于识别总体中位数掩盖的 trade-off；两组各 100 个基因，因此总体结果没有 panel 数量不平衡。",
        "",
        f"## 类型等权敏感性（空间每类 n≥{args.min_spatial_type_cells}）",
        "",
        "| Method | Median Δ vs RAW | Enhanced / Stable / Weakened | Direction supported | Median scRNA-error reduction |",
        "|:--|--:|:--|--:|--:|",
    ])
    for method in METHOD_ORDER:
        row = sensitivity.loc[method]
        lines.append(
            f"| {method} | {format_number(row.median_contrast_delta_vs_raw, signed=True)} | "
            f"{int(row.n_enhanced)} / {int(row.n_stable_or_reference)} / {int(row.n_weakened)} | "
            f"{row.direction_supported_fraction:.1%} | "
            f"{format_number(row.median_scrna_error_reduction_vs_raw, signed=True)} |"
        )

    lines.extend(["", "## 各方法变化最大的基因", ""])
    primary_detail = detail[
        (detail["scope"] == "cell_weighted") & (detail["method"] != "RAW")
    ]
    for method in METHOD_ORDER[1:]:
        group = primary_detail[primary_detail["method"] == method]
        enhanced = ", ".join(
            f"{row.gene} ({row.contrast_delta_vs_raw:+.2f})"
            for row in group.nlargest(8, "contrast_delta_vs_raw").itertuples()
        )
        weakened = ", ".join(
            f"{row.gene} ({row.contrast_delta_vs_raw:+.2f})"
            for row in group.nsmallest(8, "contrast_delta_vs_raw").itertuples()
        )
        lines.extend([
            f"### {method}", "",
            f"- 增强最多：{enhanced}。",
            f"- 减弱最多：{weakened}。", "",
        ])

    low_types = spatial_type_counts[spatial_type_counts < args.min_spatial_type_cells]
    low_text = ", ".join(f"{name} (n={count})" for name, count in low_types.items())
    lines.extend([
        "## 解释限制",
        "",
        "- 数据来自一个配对样本；单细胞 Welch 检验把细胞作为观察单位，FDR 适合 marker 筛选，但不能替代跨患者差异表达推断。",
        "- 主结果按细胞数加权，反映该空间窗口的实际组成；类型等权结果用于检查大类细胞数量主导。",
        f"- 类型等权中排除的低样本空间类型：{low_text}。",
        "- 增强对比度与恢复真实表达不是同义词；过度校正也可能放大对比度，因此同时报告相对 scRNA log2FC 的误差变化。",
        "- scRNA 与空间平台不同，效应量接近程度属于外部一致性证据，不把 scRNA 当作无误差真值。",
        "",
        "## 输出文件",
        "",
        "- `scrna_all_gene_de_statistics.csv`：所有单细胞基因的筛选统计。",
        "- `scrna_selected_de_genes.csv`：进入空间评价的独立基因 panel。",
        "- `spatial_scrna_de_gene_contrasts.csv`：逐基因、逐方法、两种口径的完整对比。",
        "- `spatial_scrna_de_contrast_summary.csv`：方法汇总。",
    ])
    (output_dir / "scrna_de_contrast_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default=TAG)
    parser.add_argument(
        "--input-dir", default="evaluation/reports/h5ad_ovarian_annotated"
    )
    parser.add_argument(
        "--scrna-h5",
        default="evaluation/data/ovarian/17k_Ovarian_Cancer_scFFPE_count_filtered_feature_bc_matrix.h5",
    )
    parser.add_argument(
        "--scrna-annotation",
        default="evaluation/data/ovarian/FLEX_Ovarian_Barcode_Cluster_Annotation.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="evaluation/reports/ovarian_eval/marker_analysis/scrna_de_contrast",
    )
    parser.add_argument("--top-n-per-direction", type=int, default=100)
    parser.add_argument("--min-detection", type=float, default=0.10)
    parser.add_argument("--min-target-mean", type=float, default=0.5)
    parser.add_argument("--min-abs-log2fc", type=float, default=1.0)
    parser.add_argument("--max-fdr", type=float, default=0.05)
    parser.add_argument("--target-sum", type=float, default=1e4)
    parser.add_argument("--pseudocount", type=float, default=0.1)
    parser.add_argument("--tolerance", type=float, default=0.1)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--min-spatial-type-cells", type=int, default=20)
    parser.add_argument("--heatmap-top-n", type=int, default=20)
    parser.add_argument("--include-technical-genes", action="store_true")
    args = parser.parse_args()

    if not 0 <= args.min_detection <= 1:
        raise ValueError("--min-detection must be between 0 and 1")
    if not 0 < args.max_fdr <= 1:
        raise ValueError("--max-fdr must be in (0, 1]")
    if min(
        args.top_n_per_direction, args.target_sum, args.pseudocount,
        args.chunk_size, args.min_spatial_type_cells, args.heatmap_top_n,
    ) <= 0:
        raise ValueError("Positive numeric arguments must be greater than zero")
    if min(args.min_target_mean, args.min_abs_log2fc, args.tolerance) < 0:
        raise ValueError("Expression, log2FC, and tolerance thresholds cannot be negative")

    input_dir = Path(args.input_dir)
    raw_path = input_dir / f"{args.tag}_{METHOD_SUFFIXES['RAW']}.h5ad"
    raw = ad.read_h5ad(raw_path, backed="r")
    try:
        spatial_genes = raw.var_names.copy()
    finally:
        raw.file.close()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_de, selected, sc_counts = discover_scrna_de_genes(
        Path(args.scrna_h5), Path(args.scrna_annotation), spatial_genes,
        args.target_sum, args.pseudocount, args.min_detection,
        args.min_target_mean, args.min_abs_log2fc, args.max_fdr,
        args.top_n_per_direction, not args.include_technical_genes,
    )
    if selected.empty or set(selected["direction"]) != set(DIRECTION_ORDER):
        raise ValueError("DE filters must select at least one gene in each direction")
    all_de.to_csv(output_dir / "scrna_all_gene_de_statistics.csv", index=False)
    selected.to_csv(output_dir / "scrna_selected_de_genes.csv", index=False)

    detail, spatial_type_counts = build_spatial_contrast_table(
        input_dir, args.tag, selected, args.chunk_size, args.target_sum,
        args.pseudocount, args.min_spatial_type_cells, args.tolerance,
    )
    summary = summarize_contrasts(detail)
    detail.to_csv(output_dir / "spatial_scrna_de_gene_contrasts.csv", index=False)
    summary.to_csv(output_dir / "spatial_scrna_de_contrast_summary.csv", index=False)
    spatial_type_counts.rename("n_spatial_cells").to_csv(
        output_dir / "spatial_tumor_stroma_type_counts.csv"
    )
    plot_contrast_delta(detail, output_dir)
    plot_gene_heatmap(detail, selected, output_dir, args.heatmap_top_n)
    write_report(
        selected, detail, summary, sc_counts, spatial_type_counts,
        args, output_dir,
    )
    print(f"Outputs written to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
