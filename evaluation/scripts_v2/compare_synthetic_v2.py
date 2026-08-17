#!/usr/bin/env python3
"""SPARKLE 2.x synthetic comparison: RAW vs SPARKLEv1 vs SPARKLEv2(NP).

Reads the h5ad outputs of ``run_correction_v2.py`` for every scenario and
computes against the regenerated ground truth (seed 42, identical to v1):

  v1 metrics (identical definitions to evaluation/scripts/):
    - RMSE on log1p(CP10K)
    - cell-wise Pearson R^2 (mean over cells)
    - KMeans ARI on SVD PCs of log1p(CP10K)

  Phase-1 metrics (roadmap section 1.6):
    - marker source preservation: sum of corrected marker expression in the
      owning cell type / sum of ground truth, averaged over marker genes
      (1.0 = perfect; <1 means true source signal was subtracted away)
    - overcorrection rate (OCR): sum((true - corrected)+) / sum(true)
    - marker non-source residual: mean |corrected - true| of marker genes in
      non-owning cells (leakage removal; lower is better)
    - latent convergence: iterations / final relative change (from metrics
      JSON written by run_correction_v2.py)

Outputs:
    evaluation/reports/v2/synthetic/synthetic_v2_summary.csv
    evaluation/reports/v2/synthetic/synthetic_v2_*.png
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.scripts.evaluate_synthetic_cell_r2 import cellwise_pearson_r2
from evaluation.scripts.evaluate_synthetic_clustering import (
    _aligned_h5ad_matrix,
    clustering_ari,
)
from evaluation.scripts.final_comparison import (
    _rmse_log1p_cp10k,
    load_synthetic_scenario_data,
)
from evaluation.synthetic import SCENARIOS

METHODS = ["RAW", "SPARKLEv1", "SPARKLEv2", "SPARKLEv2NP",
           "SPARKLEv2R1", "SPARKLEv2R2"]
METHOD_FILES = {m: ("raw" if m == "RAW" else m) for m in METHODS}
METHOD_COLORS = {"RAW": "#7f8c8d", "SPARKLEv1": "#e74c3c",
                 "SPARKLEv2": "#2980b9", "SPARKLEv2NP": "#27ae60",
                 "SPARKLEv2R1": "#8e44ad", "SPARKLEv2R2": "#d35400"}


def _v2_reports_root():
    return PROJECT_ROOT / "evaluation" / "reports" / "v2"


def marker_metrics(corrected_gc: np.ndarray, true_gc: np.ndarray,
                   cell_types: np.ndarray, marker_types: np.ndarray):
    """Source preservation, non-source residual on marker genes; and OCR."""
    marker_mask = marker_types >= 0
    if not marker_mask.any():
        return np.nan, np.nan, np.nan
    preservations = []
    nonsource_resid = []
    for g in np.flatnonzero(marker_mask):
        owning = cell_types == marker_types[g]
        true_src = true_gc[g, owning].sum()
        if true_src > 0:
            preservations.append(corrected_gc[g, owning].sum() / true_src)
        nonsource_resid.append(
            np.abs(corrected_gc[g, ~owning] - true_gc[g, ~owning]).mean()
        )
    ocr = float(
        np.maximum(true_gc - corrected_gc, 0.0).sum()
        / max(true_gc.sum(), 1e-12)
    )
    return (
        float(np.mean(preservations)) if preservations else np.nan,
        float(np.mean(nonsource_resid)) if nonsource_resid else np.nan,
        ocr,
    )


def evaluate_scenario(scenario_id: str, seed: int = 42):
    tag = f"synthetic_{scenario_id}"
    h5ad_dir = _v2_reports_root() / "h5ad"
    data = load_synthetic_scenario_data(scenario_id, seed=seed)
    true_gc = np.asarray(data["true_expr"], dtype=np.float64)
    cell_types = np.asarray(data["cell_types"])
    marker_types = np.asarray(data["marker_types"])
    rows = []
    for method, suffix in METHOD_FILES.items():
        path = h5ad_dir / f"{tag}_{suffix}.h5ad"
        if not path.exists():
            print(f"  [skip] {tag} {method}: {path} not found")
            continue
        cells_by_genes = _aligned_h5ad_matrix(
            path, data["cell_ids"], data["gene_names"]
        )
        corrected_gc = cells_by_genes.T
        preservation, nonsource_resid, ocr = marker_metrics(
            corrected_gc, true_gc, cell_types, marker_types
        )
        rows.append({
            "scenario": scenario_id,
            "method": method,
            "rmse_log1p_cp10k": _rmse_log1p_cp10k(corrected_gc, true_gc),
            "cellwise_r2_mean": float(
                np.nanmean(cellwise_pearson_r2(corrected_gc.T, true_gc.T))
            ),
            "clustering_ari": float(clustering_ari(cells_by_genes, cell_types)),
            "marker_source_preservation": preservation,
            "marker_nonsource_resid": nonsource_resid,
            "overcorrection_rate": ocr,
        })
    # Latent convergence diagnostics from the runner's metrics JSON.
    metrics_path = _v2_reports_root() / "metrics" / f"{tag}_metrics_v2.json"
    if metrics_path.exists():
        stored = json.loads(metrics_path.read_text(encoding="utf-8"))
        for row in rows:
            entry = stored.get("methods", {}).get(row["method"], {})
            lat = entry.get("latent") or {}
            row["latent_converged"] = lat.get("converged")
            row["latent_n_iterations"] = lat.get("n_iterations_max")
            row["latent_final_rel_change"] = lat.get("final_rel_change_max")
            row["runtime_sec"] = entry.get("runtime_sec")
    return rows


def plot_summary(df: pd.DataFrame, out_dir: Path):
    metrics = [
        ("rmse_log1p_cp10k", "RMSE log1p(CP10K)", False),
        ("cellwise_r2_mean", "Cell-wise Pearson R²", True),
        ("clustering_ari", "Clustering ARI", True),
        ("marker_source_preservation", "Marker source preservation", True),
        ("marker_nonsource_resid", "Marker non-source residual", False),
        ("overcorrection_rate", "Overcorrection rate (OCR)", False),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(17, 8))
    for ax, (col, title, higher_better) in zip(axes.ravel(), metrics):
        for method in METHODS:
            sub = df[df["method"] == method].set_index("scenario")
            sub = sub.reindex(sorted(df["scenario"].unique()))
            ax.plot(sub.index, sub[col], marker="o", label=method,
                    color=METHOD_COLORS[method])
        ax.set_title(f"{title}\n({'higher is better' if higher_better else 'lower is better'})",
                     fontsize=10)
        ax.tick_params(axis="x", rotation=45)
        ax.grid(alpha=0.3)
    axes.ravel()[0].legend(fontsize=8)
    fig.suptitle("SPARKLE 2.x latent-X ablation — synthetic S1–S10", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "synthetic_v2_metrics.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", type=str, nargs="*", default=None,
                        help="Scenario ids (default: all available in "
                             "reports/v2/h5ad)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = _v2_reports_root() / "synthetic"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.scenarios:
        scenario_ids = args.scenarios
    else:
        scenario_ids = sorted(SCENARIOS.keys())

    rows = []
    for sid in scenario_ids:
        print(f"[{sid}]")
        rows.extend(evaluate_scenario(sid, seed=args.seed))

    df = pd.DataFrame(rows)
    out_csv = out_dir / "synthetic_v2_summary.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved {out_csv}")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(df.round(4).to_string(index=False))
    plot_summary(df, out_dir)
    print(f"Saved {out_dir / 'synthetic_v2_metrics.png'}")


if __name__ == "__main__":
    main()
