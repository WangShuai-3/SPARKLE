#!/usr/bin/env python3
"""Cell-type clustering accuracy benchmark for synthetic scenarios.

Leakage blurs cell-type structure; a good correction restores it.  For RAW
and every corrected matrix we cluster cells (log1p(CP10K) -> TruncatedSVD
-> KMeans with k = n_true_types) and compare clusters with the ground-truth
cell types via the Adjusted Rand Index (ARI).  The metric is library-size
invariant.  Existing h5ad files are reused; correction methods and the
simulator are not rerun.  Scenarios where a method has no h5ad are recorded
as NaN.
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
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import adjusted_rand_score

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


def log1p_cp10k(matrix: np.ndarray) -> np.ndarray:
    """Library-size normalize each cell to 10k counts and log1p-transform."""
    matrix = np.asarray(matrix, dtype=np.float64)
    lib = matrix.sum(axis=1, keepdims=True)
    return np.log1p(matrix / np.maximum(lib, 1.0) * 1e4)


def clustering_ari(matrix: np.ndarray, cell_types: np.ndarray,
                   seed: int = 42) -> float:
    """ARI of KMeans clusters (k = n_true_types) vs ground-truth types."""
    X = log1p_cp10k(matrix)
    n_types = len(np.unique(cell_types))
    n_pc = min(30, X.shape[0] - 1, X.shape[1])
    # TruncatedSVD and a single k-means++ init: full PCA and n_init=10 are
    # an order of magnitude slower on this OpenBLAS build for no measurable
    # gain on PC-space clusters.
    pcs = TruncatedSVD(n_components=n_pc, random_state=seed).fit_transform(X)
    labels = KMeans(
        n_clusters=n_types, n_init="auto", random_state=seed
    ).fit_predict(pcs)
    return adjusted_rand_score(cell_types, labels)


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


def evaluate(h5ad_dir: Path, metrics_dir: Path, scenarios: list[str],
             summary_path: Path | None = None):
    summary_rows = []
    for sid in scenarios:
        tag = f"synthetic_{sid}"
        data = load_synthetic_scenario_data(sid)
        cell_ids = data["cell_ids"]
        gene_names = data["gene_names"]
        cell_types = data["cell_types"]

        metrics_path = metrics_dir / f"{tag}_metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8")) \
            if metrics_path.exists() else {"dataset": tag, "raw": {}, "methods": {}}

        for method, suffix in METHOD_FILES.items():
            path = h5ad_dir / f"{tag}_{suffix}.h5ad"
            if not path.exists():
                print(f"  {sid} {method}: {path.name} missing, recorded as NaN")
                continue
            matrix = _aligned_h5ad_matrix(path, cell_ids, gene_names)
            ari = clustering_ari(matrix, cell_types)

            entry = (metrics.setdefault("raw", {}) if method == "RAW"
                     else metrics.setdefault("methods", {}).setdefault(method, {}))
            entry["clustering_ari"] = float(ari)
            summary_rows.append({
                "scenario": sid, "method": method, "clustering_ari": float(ari),
            })

        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"{sid}: done", flush=True)
        # Save partial results after every scenario so a long run can be
        # interrupted without losing progress.
        if summary_path is not None:
            pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    return pd.DataFrame(summary_rows)


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
        description="Cell-type clustering accuracy benchmark on synthetic h5ad outputs."
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

    summary_path = out_dir / "synthetic_clustering_summary.csv"
    summary = evaluate(h5ad_dir, metrics_dir, scenarios,
                       summary_path=summary_path)
    # Merge with any existing summary so single-scenario reruns do not
    # wipe the rows of scenarios not evaluated this time.
    if summary_path.exists():
        prior = pd.read_csv(summary_path)
        prior = prior[~prior["scenario"].isin(summary["scenario"].unique())]
        summary = pd.concat([prior, summary], ignore_index=True)
        summary.to_csv(summary_path, index=False)
    print(f"Saved {summary_path}")

    _plot_grouped(
        summary, "clustering_ari",
        "Cell-type clustering accuracy (ARI, KMeans on log1p(CP10K) PCs)\n"
        "higher = correction better preserves cell-type structure",
        "Adjusted Rand Index",
        out_dir / "synthetic_clustering_ari.png",
    )


if __name__ == "__main__":
    main()
