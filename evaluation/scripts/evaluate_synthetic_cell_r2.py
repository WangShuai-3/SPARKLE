#!/usr/bin/env python3
"""Compare synthetic correction outputs using cell-level expression-profile R².

For each cell, Pearson correlation is calculated across genes between the
method output and the known uncontaminated expression, then squared. This is
scale-invariant and complements the absolute-error RMSE benchmark. Existing
h5ad files are reused; correction methods and the simulator are not rerun.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.scripts.final_comparison import load_synthetic_scenario_data


METHOD_FILES = {
    "RAW": "raw",
    "SPARKLE": "SPARKLE",
    "SoupX": "SoupX",
    "DecontX": "DecontX",
    "SpotClean": "SpotCleanOfficial",
}
METHOD_COLORS = {
    "RAW": "#7f8c8d",
    "SPARKLE": "#e74c3c",
    "SoupX": "#3498db",
    "DecontX": "#2ecc71",
    "SpotClean": "#9b59b6",
}


def cellwise_pearson_r2(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Return squared Pearson correlation across genes for every cell.

    Both arrays must be cells-by-genes. Cells with a constant prediction or
    truth profile have undefined correlation and are returned as NaN.
    """
    pred = np.asarray(pred, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    if pred.shape != truth.shape or pred.ndim != 2:
        raise ValueError(
            f"pred and truth must be matching cells-by-genes matrices; "
            f"got {pred.shape} and {truth.shape}"
        )
    if not np.isfinite(pred).all() or not np.isfinite(truth).all():
        raise ValueError("pred and truth must contain only finite values")

    pred_centered = pred - pred.mean(axis=1, keepdims=True)
    truth_centered = truth - truth.mean(axis=1, keepdims=True)
    numerator = np.einsum("ij,ij->i", pred_centered, truth_centered)
    pred_ss = np.einsum("ij,ij->i", pred_centered, pred_centered)
    truth_ss = np.einsum("ij,ij->i", truth_centered, truth_centered)
    denominator = pred_ss * truth_ss

    r2 = np.full(pred.shape[0], np.nan, dtype=np.float64)
    valid = denominator > 0
    r2[valid] = numerator[valid] ** 2 / denominator[valid]
    # Guard against floating-point excursions just above one.
    r2[valid] = np.clip(r2[valid], 0.0, 1.0)
    return r2


def _aligned_h5ad_matrix(
    path: Path, expected_cell_ids: np.ndarray, expected_genes: np.ndarray
) -> np.ndarray:
    result = ad.read_h5ad(path)
    if "cell_id" not in result.obs:
        raise ValueError(f"{path} lacks obs['cell_id']")
    observed_cells = result.obs["cell_id"].to_numpy()
    observed_genes = result.var_names.astype(str).to_numpy()
    if len(set(observed_cells.tolist())) != len(observed_cells):
        raise ValueError(f"{path} has duplicate cell IDs")
    if len(set(observed_genes.tolist())) != len(observed_genes):
        raise ValueError(f"{path} has duplicate gene names")

    cell_lookup = {value: index for index, value in enumerate(observed_cells)}
    gene_lookup = {value: index for index, value in enumerate(observed_genes)}
    try:
        cell_order = np.asarray([cell_lookup[value] for value in expected_cell_ids])
        gene_order = np.asarray([gene_lookup[value] for value in expected_genes])
    except KeyError as error:
        raise ValueError(f"{path} does not match synthetic truth: missing {error}") from error

    matrix = result.X
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    matrix = np.asarray(matrix, dtype=np.float64)
    return matrix[np.ix_(cell_order, gene_order)]


def _summarize(values: np.ndarray) -> dict:
    valid = values[np.isfinite(values)]
    if not len(valid):
        return {
            "cell_pearson_r2_mean": None,
            "cell_pearson_r2_median": None,
            "cell_pearson_r2_q25": None,
            "cell_pearson_r2_q75": None,
            "cell_pearson_r2_n_valid": 0,
        }
    return {
        "cell_pearson_r2_mean": float(np.mean(valid)),
        "cell_pearson_r2_median": float(np.median(valid)),
        "cell_pearson_r2_q25": float(np.quantile(valid, 0.25)),
        "cell_pearson_r2_q75": float(np.quantile(valid, 0.75)),
        "cell_pearson_r2_n_valid": int(len(valid)),
    }


def _update_metrics(metrics_path: Path, method: str, summary: dict) -> None:
    if not metrics_path.exists():
        return
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if method == "RAW":
        entry = metrics.setdefault("raw", {})
    else:
        entry = metrics.setdefault("methods", {}).setdefault(method, {})
    entry.update(summary)
    entry["cell_r2_definition"] = "squared_Pearson_across_genes_per_cell"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def evaluate(h5ad_dir: Path, metrics_dir: Path, scenarios: list[str]):
    summary_rows = []
    per_cell_rows = []
    for scenario in scenarios:
        data = load_synthetic_scenario_data(scenario, seed=42)
        truth = np.asarray(data["true_expr"], dtype=np.float64).T
        cell_ids = np.asarray(data["cell_ids"])
        genes = np.asarray(data["gene_names"], dtype=str)
        tag = f"synthetic_{scenario}"

        for method, suffix in METHOD_FILES.items():
            path = h5ad_dir / f"{tag}_{suffix}.h5ad"
            if not path.exists():
                print(f"WARNING: missing {path}; skipping {method}", flush=True)
                continue
            pred = _aligned_h5ad_matrix(path, cell_ids, genes)
            r2 = cellwise_pearson_r2(pred, truth)
            stats = _summarize(r2)
            summary_rows.append(
                {
                    "scenario": scenario,
                    "method": method,
                    **stats,
                    "n_cells": int(len(cell_ids)),
                    "source_h5ad": str(path),
                }
            )
            per_cell_rows.extend(
                {
                    "scenario": scenario,
                    "cell_id": cell_id,
                    "method": method,
                    "cell_pearson_r2": value,
                }
                for cell_id, value in zip(cell_ids, r2)
            )
            _update_metrics(metrics_dir / f"{tag}_metrics.json", method, stats)
            print(
                f"{scenario:>3} {method:<12} mean={stats['cell_pearson_r2_mean']:.4f} "
                f"median={stats['cell_pearson_r2_median']:.4f}",
                flush=True,
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(per_cell_rows)


def plot_mean_comparison(summary: pd.DataFrame, output_path: Path) -> None:
    scenarios = sorted(summary["scenario"].unique(), key=lambda value: int(value[1:]))
    methods = [method for method in METHOD_FILES if method in set(summary["method"])]
    x = np.arange(len(scenarios))
    width = 0.84 / len(methods)
    fig, ax = plt.subplots(figsize=(15, 6.5))
    for index, method in enumerate(methods):
        values = (
            summary[summary["method"] == method]
            .set_index("scenario")["cell_pearson_r2_mean"]
            .reindex(scenarios)
        )
        offset = (index - (len(methods) - 1) / 2) * width
        ax.bar(
            x + offset,
            values,
            width,
            label=method,
            color=METHOD_COLORS[method],
            edgecolor="black",
            linewidth=0.4,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean cell-level Pearson R² across genes")
    ax.set_title(
        "Cell-level expression-profile R² on synthetic scenarios\n"
        "Pearson r² across 500 genes per cell; bars show the mean across cells"
    )
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    ax.legend(
        ncol=len(methods),
        title="Method",
        loc="lower center",
        bbox_to_anchor=(0.5, 1.11),
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_cell_distributions(per_cell: pd.DataFrame, output_path: Path) -> None:
    scenarios = sorted(per_cell["scenario"].unique(), key=lambda value: int(value[1:]))
    methods = [method for method in METHOD_FILES if method in set(per_cell["method"])]
    fig, axes = plt.subplots(2, 5, figsize=(22, 9), sharey=True)
    axes = axes.ravel()
    for ax, scenario in zip(axes, scenarios):
        subset = per_cell[per_cell["scenario"] == scenario]
        values = [
            subset.loc[subset["method"] == method, "cell_pearson_r2"].dropna().to_numpy()
            for method in methods
        ]
        boxes = ax.boxplot(values, tick_labels=methods, patch_artist=True, showfliers=False)
        for patch, method in zip(boxes["boxes"], methods):
            patch.set_facecolor(METHOD_COLORS[method])
            patch.set_alpha(0.8)
        n_cells = subset["cell_id"].nunique()
        ax.set_title(f"{scenario} (n={n_cells})")
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=40, labelsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
    for ax in axes[len(scenarios) :]:
        ax.set_visible(False)
    fig.supylabel("Cell-level Pearson R² across genes")
    fig.suptitle(
        "Per-cell R² distributions on synthetic scenarios\n"
        "Pearson r² computed across 500 genes within each cell",
        fontsize=15,
    )
    fig.tight_layout(rect=(0.015, 0, 1, 0.95))
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5ad-dir", type=Path, default=Path("evaluation/reports/h5ad"))
    parser.add_argument(
        "--metrics-dir", type=Path, default=Path("evaluation/reports/metrics")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("evaluation/reports/synthetic_figures")
    )
    parser.add_argument("--scenarios", nargs="+", default=[f"S{i}" for i in range(1, 11)])
    parser.add_argument(
        "--append",
        action="store_true",
        help="Merge the selected scenarios into existing summary/per-cell CSV files",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary, per_cell = evaluate(args.h5ad_dir, args.metrics_dir, args.scenarios)
    if summary.empty:
        raise FileNotFoundError("No matching synthetic h5ad results were found")
    summary_path = args.output_dir / "synthetic_cell_r2_summary.csv"
    per_cell_path = args.output_dir / "synthetic_cell_r2_per_cell.csv"
    if args.append and summary_path.exists() and per_cell_path.exists():
        replaced = set(summary["scenario"])
        old_summary = pd.read_csv(summary_path)
        old_per_cell = pd.read_csv(per_cell_path)
        summary = pd.concat(
            [old_summary[~old_summary["scenario"].isin(replaced)], summary],
            ignore_index=True,
        )
        per_cell = pd.concat(
            [old_per_cell[~old_per_cell["scenario"].isin(replaced)], per_cell],
            ignore_index=True,
        )
    scenario_rank = {f"S{i}": i for i in range(1, 11)}
    method_rank = {method: index for index, method in enumerate(METHOD_FILES)}
    summary = summary.sort_values(
        ["scenario", "method"],
        key=lambda column: column.map(scenario_rank if column.name == "scenario" else method_rank),
    )
    per_cell = per_cell.sort_values(
        ["scenario", "cell_id", "method"],
        key=lambda column: column.map(
            scenario_rank
            if column.name == "scenario"
            else method_rank
            if column.name == "method"
            else None
        )
        if column.name != "cell_id"
        else column,
    )
    summary.to_csv(summary_path, index=False)
    per_cell.to_csv(per_cell_path, index=False)
    plot_mean_comparison(summary, args.output_dir / "synthetic_cell_r2_comparison.png")
    plot_cell_distributions(per_cell, args.output_dir / "synthetic_cell_r2_subplots.png")

    pivot = summary.pivot(
        index="scenario", columns="method", values="cell_pearson_r2_mean"
    ).reindex([f"S{i}" for i in range(1, 11)])
    print("\nMean cell-level Pearson R²:")
    print(pivot.round(4).to_string())
    print(f"\nSaved {summary_path}")
    print(f"Saved {per_cell_path}")


if __name__ == "__main__":
    main()
