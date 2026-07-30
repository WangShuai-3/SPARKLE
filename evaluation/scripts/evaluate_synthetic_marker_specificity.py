#!/usr/bin/env python3
"""Marker-specificity log2FC benchmark for synthetic scenarios.

For every marker gene (ground-truth assignment from the generator) compute
the log2 fold change between its owning cell type and all other cells, on
RAW and on each corrected matrix.  The metric is scale-invariant and tests
the correction goal directly: true marker signal retained in owning cells,
leaked marker removed elsewhere.  Existing h5ad files are reused; correction
methods and the simulator are not rerun.
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
EPS = 1.0


def marker_log2fc(matrix: np.ndarray, cell_types: np.ndarray,
                  marker_types: np.ndarray) -> np.ndarray:
    """Per-marker log2 fold change (owning type vs all other cells).

    ``matrix`` is cells-by-genes.  Returns one value per marker gene, in
    ascending gene-index order.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    marker_genes = np.flatnonzero(marker_types >= 0)
    fc = np.full(len(marker_genes), np.nan, dtype=np.float64)
    for t in np.unique(marker_types[marker_genes]):
        genes = np.flatnonzero(marker_types == t)
        in_mask = cell_types == t
        if in_mask.sum() == 0 or (~in_mask).sum() == 0:
            continue
        mu_in = matrix[in_mask][:, genes].mean(axis=0)
        mu_out = matrix[~in_mask][:, genes].mean(axis=0)
        pos = np.searchsorted(marker_genes, genes)
        fc[pos] = np.log2((mu_in + EPS) / (mu_out + EPS))
    return fc


def _aligned_h5ad_matrix(path: Path, cell_ids: np.ndarray,
                         gene_names: np.ndarray) -> np.ndarray:
    """Read an h5ad and return a cells-by-genes matrix aligned to the
    regenerated cell_ids/gene_names order."""
    result = ad.read_h5ad(path)
    X = result.X.toarray() if hasattr(result.X, "toarray") else np.asarray(result.X)
    obs_ids = result.obs["cell_id"].to_numpy()
    cell_order = np.array(
        [int(np.flatnonzero(obs_ids == cid)[0]) for cid in cell_ids]
    )
    var_names = result.var_names.astype(str).to_numpy()
    gene_lookup = {g: i for i, g in enumerate(var_names)}
    gene_order = np.array([gene_lookup[g] for g in gene_names])
    return np.asarray(X, dtype=np.float64)[cell_order][:, gene_order]


def evaluate(h5ad_dir: Path, metrics_dir: Path, scenarios: list[str]):
    """Compute per-marker log2FC for RAW and every available method."""
    per_marker_rows, summary_rows = [], []
    for sid in scenarios:
        tag = f"synthetic_{sid}"
        data = load_synthetic_scenario_data(sid)
        cell_ids = data["cell_ids"]
        gene_names = data["gene_names"]
        cell_types = data["cell_types"]
        marker_types = data["marker_types"]
        n_markers = int((marker_types >= 0).sum())
        if n_markers == 0:
            print(f"{sid}: no marker genes, skipping")
            continue

        fc_by_method = {}
        for method, suffix in METHOD_FILES.items():
            path = h5ad_dir / f"{tag}_{suffix}.h5ad"
            if not path.exists():
                print(f"  {sid} {method}: {path.name} missing, recorded as NaN")
                continue
            matrix = _aligned_h5ad_matrix(path, cell_ids, gene_names)
            fc_by_method[method] = marker_log2fc(matrix, cell_types, marker_types)

        marker_genes = np.flatnonzero(marker_types >= 0)
        raw_fc = fc_by_method.get("RAW")
        metrics_path = metrics_dir / f"{tag}_metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8")) \
            if metrics_path.exists() else {"dataset": tag, "raw": {}, "methods": {}}

        if raw_fc is not None:
            metrics.setdefault("raw", {})["marker_log2fc_mean"] = float(np.nanmean(raw_fc))
        for method, fc in fc_by_method.items():
            if method == "RAW":
                continue
            entry = metrics.setdefault("methods", {}).setdefault(method, {})
            entry["marker_log2fc_mean"] = float(np.nanmean(fc))
            if raw_fc is not None:
                entry["marker_log2fc_delta_vs_raw"] = float(np.nanmean(fc - raw_fc))

        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        for i, g in enumerate(marker_genes):
            row = {"scenario": sid, "gene": gene_names[g],
                   "owning_type": int(marker_types[g])}
            for method in METHOD_FILES:
                fc = fc_by_method.get(method)
                row[method] = float(fc[i]) if fc is not None else float("nan")
            per_marker_rows.append(row)

        for method in METHOD_FILES:
            fc = fc_by_method.get(method)
            mean_fc = float(np.nanmean(fc)) if fc is not None else float("nan")
            delta = (float(np.nanmean(fc - raw_fc))
                     if fc is not None and raw_fc is not None and method != "RAW"
                     else float("nan"))
            summary_rows.append({
                "scenario": sid, "method": method,
                "n_markers": n_markers,
                "marker_log2fc_mean": mean_fc,
                "marker_log2fc_delta_vs_raw": delta,
            })
        print(f"{sid}: markers={n_markers}, " + ", ".join(
            f"{m}={np.nanmean(f):.2f}" for m, f in fc_by_method.items()))

    return pd.DataFrame(per_marker_rows), pd.DataFrame(summary_rows)


def _plot_grouped(summary: pd.DataFrame, value_col: str, title: str,
                  ylabel: str, output_path: Path, figsize=(14, 6)):
    methods = [m for m in METHOD_FILES if m in summary["method"].unique()]
    scenarios = sorted(summary["scenario"].unique(), key=lambda s: int(s[1:]))
    x = np.arange(len(scenarios))
    width = 0.75 / len(methods)
    fig, ax = plt.subplots(figsize=figsize)
    for i, method in enumerate(methods):
        vals = (summary[summary["method"] == method]
                .set_index("scenario")[value_col].reindex(scenarios))
        offset = (i - (len(methods) - 1) / 2) * width
        ax.bar(x + offset, vals.values, width, label=method,
               color=METHOD_COLORS[method], edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(title="Method", loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    if (summary[value_col] < 0).any() or "delta" in value_col:
        ax.axhline(0.0, color="black", linewidth=0.8)
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Marker-specificity log2FC benchmark on synthetic h5ad outputs."
    )
    parser.add_argument("--h5ad-dir", type=str, default="evaluation/reports/h5ad")
    parser.add_argument("--metrics-dir", type=str, default="evaluation/reports/metrics")
    parser.add_argument("--output-dir", type=str,
                        default="evaluation/reports/synthetic_figures")
    parser.add_argument("--scenarios", nargs="*", default=None,
                        help="Scenario IDs (default: all with metrics JSONs)")
    args = parser.parse_args()

    h5ad_dir = Path(args.h5ad_dir)
    metrics_dir = Path(args.metrics_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.scenarios:
        scenarios = args.scenarios
    else:
        scenarios = sorted(
            (p.stem.split("_")[1] for p in metrics_dir.glob("synthetic_S*_metrics.json")),
            key=lambda s: int(s[1:]),
        )

    per_marker, summary = evaluate(h5ad_dir, metrics_dir, scenarios)

    per_marker_path = out_dir / "synthetic_marker_log2fc_per_marker.csv"
    summary_path = out_dir / "synthetic_marker_log2fc_summary.csv"
    per_marker.to_csv(per_marker_path, index=False)
    summary.to_csv(summary_path, index=False)
    print(f"Saved {per_marker_path}\nSaved {summary_path}")

    _plot_grouped(
        summary, "marker_log2fc_mean",
        "Marker specificity (log2FC: owning type vs other cells)\n"
        "higher = marker expression better confined to its owning cell type",
        "Mean marker log2FC",
        out_dir / "synthetic_marker_log2fc_comparison.png",
    )
    _plot_grouped(
        summary[summary["method"] != "RAW"], "marker_log2fc_delta_vs_raw",
        "Marker-specificity gain over RAW (Δ mean marker log2FC)\n"
        ">0 means correction improved marker confinement",
        "Δ mean marker log2FC vs RAW",
        out_dir / "synthetic_marker_log2fc_delta.png",
    )


if __name__ == "__main__":
    main()
