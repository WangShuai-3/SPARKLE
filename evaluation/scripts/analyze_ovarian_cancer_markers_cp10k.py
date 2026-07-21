#!/usr/bin/env python3
"""Ovarian marker contrast sensitivity analysis on unlogged expression.

Unlike ``analyze_ovarian_cancer_markers.py``, this script does not apply
``log1p``.  It supports either per-cell CP10K or completely unnormalized
matrix values, averages expression across tumor and pooled non-tumor cells,
and then calculates a pseudocount-stabilized log2FC per marker.

Outputs are isolated under ``marker_analysis/cp10k`` or
``marker_analysis/unnormalized`` so the primary log1p-CP10K analysis is not
overwritten.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from analyze_ovarian_cancer_markers import (
    EPITHELIAL_TUMOR_MARKERS,
    GENE_DIRECTION,
    IMMUNE_MARKERS,
    METHODS,
    METHOD_SUFFIXES,
    STROMAL_FIBROBLAST_MARKERS,
    STROMAL_TYPES,
    TAG,
    TUMOR_TYPES,
    plot_marker_log2fc,
)


TARGET_SUM = 1e4
PSEUDOCOUNTS = {"cp10k": 0.1, "unnormalized": 0.01}


def extract_unlogged_means(
    path: Path,
    genes: list[str],
    chunk_size: int,
    expression_scale: str,
) -> tuple[np.ndarray, np.ndarray, pd.Index, pd.Index, np.ndarray, dict[str, float]]:
    """Return cell-weighted tumor and pooled non-tumor expression means."""
    adata = ad.read_h5ad(path, backed="r")
    try:
        if "annotation" not in adata.obs:
            raise ValueError(f"{path.name}: missing obs['annotation']")
        annotations = adata.obs["annotation"].astype(str).to_numpy()
        tumor_mask = np.isin(annotations, list(TUMOR_TYPES))
        non_tumor_mask = np.isin(annotations, list(STROMAL_TYPES))
        if not tumor_mask.any() or not non_tumor_mask.any():
            raise ValueError(f"{path.name}: tumor or non-tumor group is empty")
        gene_indices = adata.var_names.get_indexer(genes)
        if (gene_indices < 0).any():
            missing = np.asarray(genes)[gene_indices < 0].tolist()
            raise ValueError(f"{path.name}: missing markers {missing}")

        tumor_sum = np.zeros(len(genes), dtype=np.float64)
        non_tumor_sum = np.zeros(len(genes), dtype=np.float64)
        library_sizes = np.empty(adata.n_obs, dtype=np.float64)
        for start in range(0, adata.n_obs, chunk_size):
            stop = min(start + chunk_size, adata.n_obs)
            block = adata.X[start:stop]
            if sparse.issparse(block):
                block = block.toarray()
            block = np.asarray(block, dtype=np.float64)
            np.maximum(block, 0.0, out=block)
            library_size = block.sum(axis=1)
            library_sizes[start:stop] = library_size
            marker_values = block[:, gene_indices]
            if expression_scale == "cp10k":
                scale = np.divide(
                    TARGET_SUM,
                    library_size,
                    out=np.zeros_like(library_size),
                    where=library_size > 0,
                )
                marker_values = marker_values * scale[:, None]
            chunk_tumor = tumor_mask[start:stop]
            chunk_non_tumor = non_tumor_mask[start:stop]
            tumor_sum += marker_values[chunk_tumor].sum(axis=0)
            non_tumor_sum += marker_values[chunk_non_tumor].sum(axis=0)
        return (
            tumor_sum / tumor_mask.sum(),
            non_tumor_sum / non_tumor_mask.sum(),
            adata.obs_names.copy(),
            adata.var_names.copy(),
            annotations.copy(),
            {
                "n_tumor_cells": int(tumor_mask.sum()),
                "n_non_tumor_cells": int(non_tumor_mask.sum()),
                "tumor_mean_library_size": float(library_sizes[tumor_mask].mean()),
                "tumor_median_library_size": float(np.median(library_sizes[tumor_mask])),
                "non_tumor_mean_library_size": float(library_sizes[non_tumor_mask].mean()),
                "non_tumor_median_library_size": float(np.median(library_sizes[non_tumor_mask])),
            },
        )
    finally:
        adata.file.close()


def build_outputs(
    input_dir: Path,
    tag: str,
    chunk_size: int,
    expression_scale: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pseudocount = PSEUDOCOUNTS[expression_scale]
    all_genes = (
        EPITHELIAL_TUMOR_MARKERS
        + STROMAL_FIBROBLAST_MARKERS
        + IMMUNE_MARKERS
    )
    reference_obs_names = None
    reference_var_names = None
    reference_annotations = None
    rows = []
    library_rows = []
    for method in METHODS:
        suffix = METHOD_SUFFIXES.get(method, method)
        path = input_dir / f"{tag}_{suffix}.h5ad"
        if not path.exists():
            raise FileNotFoundError(path)
        print(
            f"Reading {expression_scale} marker means: {method} ({path.name})"
        )
        (
            tumor_mean, non_tumor_mean, obs_names, var_names, annotations,
            library_stats,
        ) = (
            extract_unlogged_means(
                path, all_genes, chunk_size, expression_scale
            )
        )
        library_rows.append({"method": method, **library_stats})
        if reference_obs_names is None:
            reference_obs_names = obs_names
            reference_var_names = var_names
            reference_annotations = annotations
        else:
            if not obs_names.equals(reference_obs_names):
                raise ValueError(f"{method}: spatial cell order differs from RAW")
            if not var_names.equals(reference_var_names):
                raise ValueError(f"{method}: spatial gene order differs from RAW")
            if not np.array_equal(annotations, reference_annotations):
                raise ValueError(f"{method}: annotations differ from RAW")

        log2fc = np.log2(
            (tumor_mean + pseudocount) / (non_tumor_mean + pseudocount)
        )
        for index, gene in enumerate(all_genes):
            panel = (
                "epithelial_tumor"
                if gene in EPITHELIAL_TUMOR_MARKERS
                else "stromal_fibroblast"
                if gene in STROMAL_FIBROBLAST_MARKERS
                else "immune"
            )
            rows.append({
                "method": method,
                "gene": gene,
                "panel": panel,
                "direction": GENE_DIRECTION[gene],
                "tumor_mean": float(tumor_mean[index]),
                "stromal_mean": float(non_tumor_mean[index]),
                "log2fc": float(log2fc[index]),
                "log2fc_pseudocount": pseudocount,
                "expression_scale": expression_scale,
            })

    detail = pd.DataFrame(rows)
    summary_rows = []
    for method in METHODS:
        group = detail[detail["method"] == method]
        tumor_log2fc = group.loc[
            group["panel"] == "epithelial_tumor", "log2fc"
        ].mean()
        stromal_log2fc = group.loc[
            group["panel"] == "stromal_fibroblast", "log2fc"
        ].mean()
        summary_rows.append({
            "method": method,
            "tumor_marker_mean_log2fc": tumor_log2fc,
            "stromal_marker_mean_log2fc": stromal_log2fc,
            "tumor_stromal_contrast": tumor_log2fc - stromal_log2fc,
            "log2fc_pseudocount": pseudocount,
            "expression_scale": expression_scale,
        })
    summary = pd.DataFrame(summary_rows)

    raw = detail[detail["method"] == "RAW"].set_index("gene")["log2fc"]
    delta = detail[detail["method"] != "RAW"][
        ["method", "gene", "panel", "direction", "log2fc"]
    ].copy()
    delta["log2fc_raw"] = delta["gene"].map(raw)
    delta = delta.rename(columns={"log2fc": "log2fc_method"})
    delta["log2fc_delta"] = delta["log2fc_method"] - delta["log2fc_raw"]
    delta["improved"] = np.where(
        delta["direction"] == "up",
        delta["log2fc_delta"] > 0,
        delta["log2fc_delta"] < 0,
    )
    library_summary = pd.DataFrame(library_rows)
    library_summary["mean_library_ratio_tumor_vs_non_tumor"] = (
        library_summary["tumor_mean_library_size"]
        / library_summary["non_tumor_mean_library_size"]
    )
    library_summary["median_library_ratio_tumor_vs_non_tumor"] = (
        library_summary["tumor_median_library_size"]
        / library_summary["non_tumor_median_library_size"]
    )
    return detail, summary, delta, library_summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", default="evaluation/reports/h5ad_ovarian_annotated"
    )
    parser.add_argument("--tag", default=TAG)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument(
        "--expression-scale",
        choices=sorted(PSEUDOCOUNTS),
        default="cp10k",
        help="Use unlogged CP10K or the unnormalized h5ad matrix values.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
    )
    args = parser.parse_args()
    if args.chunk_size < 1:
        raise ValueError("--chunk-size must be positive")

    output_dir = Path(args.output_dir) if args.output_dir else Path(
        "evaluation/reports/ovarian_eval/marker_analysis"
    ) / args.expression_scale
    output_dir.mkdir(parents=True, exist_ok=True)
    detail, summary, delta, library_summary = build_outputs(
        Path(args.input_dir), args.tag, args.chunk_size, args.expression_scale
    )
    suffix = args.expression_scale
    detail.to_csv(output_dir / f"marker_log2foldchanges_{suffix}.csv", index=False)
    summary.to_csv(output_dir / f"marker_summary_{suffix}.csv", index=False)
    delta.to_csv(
        output_dir / f"marker_log2foldchange_delta_{suffix}.csv", index=False
    )
    library_summary.to_csv(
        output_dir / f"library_size_summary_{suffix}.csv", index=False
    )
    log1p_summary_path = output_dir.parent / "marker_summary.csv"
    sensitivity_frames = []
    if log1p_summary_path.exists():
        log1p_summary = pd.read_csv(log1p_summary_path)
        log1p_summary["expression_scale"] = "log1p_cp10k"
        sensitivity_frames.append(log1p_summary)
    cp10k_summary_path = output_dir.parent / "cp10k" / "marker_summary_cp10k.csv"
    if cp10k_summary_path.exists() and args.expression_scale != "cp10k":
        sensitivity_frames.append(pd.read_csv(cp10k_summary_path))
    sensitivity_frames.append(summary)
    sensitivity = pd.concat(sensitivity_frames, ignore_index=True)
    sensitivity = sensitivity[[
        "expression_scale", "method", "tumor_marker_mean_log2fc",
        "stromal_marker_mean_log2fc", "tumor_stromal_contrast",
        "log2fc_pseudocount",
    ]].drop_duplicates(["expression_scale", "method"], keep="last")
    sensitivity["contrast_rank_within_scale"] = sensitivity.groupby(
        "expression_scale"
    )["tumor_stromal_contrast"].rank(
        ascending=False, method="min"
    ).astype(int)
    sensitivity.to_csv(
        output_dir / "normalization_sensitivity_long.csv", index=False
    )
    expression_label = (
        "CP10K (no log1p)"
        if args.expression_scale == "cp10k"
        else "unnormalized matrix values"
    )
    plot_marker_log2fc(detail, output_dir, expression_label=expression_label)
    print(summary.sort_values("tumor_stromal_contrast", ascending=False).to_string(index=False))
    print(f"Outputs written to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
