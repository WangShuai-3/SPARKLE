#!/usr/bin/env python3
"""Marker-detection accuracy benchmark for synthetic scenarios.

Leakage spreads marker expression into cells of other types, so standard
marker-detection pipelines lose accuracy on contaminated data; a good
correction should restore it.  For every cell type and every method we
score each gene by its Wilcoxon rank-sum effect size (AUC of the gene
separating the type from the rest) and measure, against the ground-truth
marker assignment from the generator:

  * detection AUROC — do true markers of the type outrank non-markers?
    (threshold-free)
  * precision@k — fraction of true markers among the top-k ranked genes,
    with k = number of true markers of the type

Existing h5ad files are reused; correction methods and the simulator are
not rerun.  Scenarios where a method has no h5ad are recorded as NaN.
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
from scipy.stats import mannwhitneyu, rankdata

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


def gene_type_auc(matrix: np.ndarray, cell_types: np.ndarray) -> np.ndarray:
    """Per-type, per-gene Wilcoxon effect size.

    Returns [n_types × n_genes]: AUC of each gene's expression separating
    the type's cells from all other cells (1.0 = perfectly specific).
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    types = np.unique(cell_types)
    scores = np.full((len(types), matrix.shape[1]), 0.5, dtype=np.float64)
    for ti, t in enumerate(types):
        in_mask = cell_types == t
        x_in, x_out = matrix[in_mask], matrix[~in_mask]
        n_in, n_out = x_in.shape[0], x_out.shape[0]
        if n_in == 0 or n_out == 0:
            continue
        for g in range(matrix.shape[1]):
            u, _ = mannwhitneyu(x_in[:, g], x_out[:, g], alternative="greater")
            scores[ti, g] = u / (n_in * n_out)
    return scores


def detection_metrics(scores: np.ndarray, marker_types: np.ndarray):
    """Per-type detection AUROC and precision@k from gene scores.

    ``scores`` is [n_types × n_genes] (rows follow the ascending type order
    of ``np.unique(cell_types)``).  Returns (auroc_per_type, precision_per_type)
    with NaN for types that have no markers.
    """
    types = np.unique(marker_types[marker_types >= 0])
    auroc = np.full(scores.shape[0], np.nan)
    precision = np.full(scores.shape[0], np.nan)
    for t in types:
        positives = marker_types == t
        k = int(positives.sum())
        if k == 0:
            continue
        s = scores[int(t)]
        labels = positives.astype(np.int64)
        # Detection AUROC: P(score(pos) > score(neg)) + 0.5 P(tie)
        ranks = rankdata(s)
        n_pos = labels.sum()
        n_neg = len(labels) - n_pos
        auroc[int(t)] = (ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
        # precision@k: true markers among the k highest-scoring genes
        top_k = np.argsort(s)[::-1][:k]
        precision[int(t)] = labels[top_k].mean()
    return auroc, precision


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
    per_type_rows, summary_rows = [], []
    for sid in scenarios:
        tag = f"synthetic_{sid}"
        data = load_synthetic_scenario_data(sid)
        cell_ids = data["cell_ids"]
        gene_names = data["gene_names"]
        cell_types = data["cell_types"]
        marker_types = data["marker_types"]
        if (marker_types >= 0).sum() == 0:
            print(f"{sid}: no marker genes, skipping")
            continue

        metrics_path = metrics_dir / f"{tag}_metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8")) \
            if metrics_path.exists() else {"dataset": tag, "raw": {}, "methods": {}}

        for method, suffix in METHOD_FILES.items():
            path = h5ad_dir / f"{tag}_{suffix}.h5ad"
            if not path.exists():
                print(f"  {sid} {method}: {path.name} missing, recorded as NaN")
                continue
            matrix = _aligned_h5ad_matrix(path, cell_ids, gene_names)
            scores = gene_type_auc(matrix, cell_types)
            auroc, precision = detection_metrics(scores, marker_types)

            entry = (metrics.setdefault("raw", {}) if method == "RAW"
                     else metrics.setdefault("methods", {}).setdefault(method, {}))
            entry["marker_detection_auroc_mean"] = float(np.nanmean(auroc))
            entry["marker_detection_precision_at_k"] = float(np.nanmean(precision))

            n_types = scores.shape[0]
            for t in range(n_types):
                if np.isnan(auroc[t]):
                    continue
                per_type_rows.append({
                    "scenario": sid, "method": method, "cell_type": t,
                    "n_cells_type": int((cell_types == t).sum()),
                    "n_markers_type": int((marker_types == t).sum()),
                    "detection_auroc": float(auroc[t]),
                    "precision_at_k": float(precision[t]),
                })
            summary_rows.append({
                "scenario": sid, "method": method,
                "detection_auroc_mean": float(np.nanmean(auroc)),
                "precision_at_k_mean": float(np.nanmean(precision)),
            })

        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        done = [r["method"] for r in summary_rows if r["scenario"] == sid]
        print(f"{sid}: evaluated {len(done)} methods")

    return pd.DataFrame(per_type_rows), pd.DataFrame(summary_rows)


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
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Marker-detection accuracy benchmark on synthetic h5ad outputs."
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

    per_type, summary = evaluate(h5ad_dir, metrics_dir, scenarios)

    per_type_path = out_dir / "synthetic_marker_detection_per_type.csv"
    summary_path = out_dir / "synthetic_marker_detection_summary.csv"
    per_type.to_csv(per_type_path, index=False)
    summary.to_csv(summary_path, index=False)
    print(f"Saved {per_type_path}\nSaved {summary_path}")

    _plot_grouped(
        summary, "detection_auroc_mean",
        "Marker-detection AUROC (Wilcoxon effect-size ranking)\n"
        "higher = true markers better recovered by standard detection",
        "Mean detection AUROC",
        out_dir / "synthetic_marker_detection_auroc.png",
    )
    _plot_grouped(
        summary, "precision_at_k_mean",
        "Marker-detection precision@k (k = true markers per type)\n"
        "higher = top-ranked genes are true markers",
        "Mean precision@k",
        out_dir / "synthetic_marker_detection_precision.png",
    )


if __name__ == "__main__":
    main()
