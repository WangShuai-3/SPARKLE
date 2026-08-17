#!/usr/bin/env python3
"""SPARKLE 2.x Phase-2 mismatch comparison (M1/M2 diffuse scenarios).

Reads the h5ad outputs of ``run_correction_v2.py --dataset mismatch`` and
evaluates every available method against the regenerated ground truth:

  correction metrics (same definitions as compare_synthetic_v2):
    - RMSE on log1p(CP10K), cell-wise Pearson R^2, clustering ARI
    - marker source preservation / non-source residual / OCR

  count-model recovery (Phase 2 specific):
    - beta recovery: fitted diffuse rate vs the injected per-DNB rate d_g
      (from adata.var['sparkle_beta_poisson']; ratio median/IQR over the
      pipeline-selected genes)
    - local-leakage strength: mean fitted rho (or alpha for OLS methods)
    - M2 false-local-leakage: M2 has zero local leakage by construction, so
      any R2-pass correction strength there is a false positive.

Outputs:
    evaluation/reports/v2/mismatch/mismatch_v2_summary.csv
"""

import argparse
import json
import sys
from pathlib import Path

import anndata as ad
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
from evaluation.scripts.final_comparison import _rmse_log1p_cp10k
from evaluation.scripts_v2.compare_synthetic_v2 import METHODS, marker_metrics
from evaluation.scripts_v2.mismatch import (
    MISMATCH_SCENARIOS,
    load_mismatch_scenario_data,
)


def _v2_reports_root():
    return PROJECT_ROOT / "evaluation" / "reports" / "v2"


def evaluate_scenario(mid: str, seed: int = 42):
    tag = f"mismatch_{mid}"
    h5ad_dir = _v2_reports_root() / "h5ad"
    data = load_mismatch_scenario_data(mid, seed=seed)
    true_gc = np.asarray(data["true_expr"], dtype=np.float64)
    cell_types = np.asarray(data["cell_types"])
    marker_types = np.asarray(data["marker_types"])
    true_beta = np.asarray(data["true_beta"], dtype=np.float64)

    rows = []
    for method in METHODS:
        suffix = "raw" if method == "RAW" else method
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
        adata = ad.read_h5ad(path)
        beta_fit = np.asarray(
            adata.var.get("sparkle_beta_poisson", np.full(len(true_gc), np.nan)),
            dtype=np.float64,
        )
        rho_fit = np.asarray(
            adata.var.get("sparkle_rho_poisson", np.full(len(true_gc), np.nan)),
            dtype=np.float64,
        )
        fit_mask = ~np.isnan(beta_fit)
        with np.errstate(divide="ignore", invalid="ignore"):
            beta_ratio = beta_fit[fit_mask] / np.maximum(true_beta[fit_mask], 1e-12)
        beta_ratio = beta_ratio[true_beta[fit_mask] > 0]
        rows.append({
            "scenario": mid,
            "method": method,
            "rmse_log1p_cp10k": _rmse_log1p_cp10k(corrected_gc, true_gc),
            "cellwise_r2_mean": float(
                np.nanmean(cellwise_pearson_r2(corrected_gc.T, true_gc.T))
            ),
            "clustering_ari": float(clustering_ari(cells_by_genes, cell_types)),
            "marker_source_preservation": preservation,
            "marker_nonsource_resid": nonsource_resid,
            "overcorrection_rate": ocr,
            "beta_ratio_median": (
                float(np.median(beta_ratio)) if len(beta_ratio) else np.nan
            ),
            "beta_ratio_iqr": (
                float(np.percentile(beta_ratio, 75)
                      - np.percentile(beta_ratio, 25))
                if len(beta_ratio) else np.nan
            ),
            "rho_mean": (
                float(np.nanmean(rho_fit)) if (~np.isnan(rho_fit)).any()
                else np.nan
            ),
        })

    # alpha/R2 gate statistics from the runner's metrics JSON.
    metrics_path = _v2_reports_root() / "metrics" / f"{tag}_metrics_v2.json"
    if metrics_path.exists():
        stored = json.loads(metrics_path.read_text(encoding="utf-8"))
        for row in rows:
            entry = stored.get("methods", {}).get(row["method"], {})
            row["alpha_mean"] = entry.get("alpha_mean")
            row["n_genes_corrected"] = entry.get("n_genes_corrected")
            row["lambda_estimated"] = entry.get("lambda_estimated")
            poi = entry.get("poisson") or {}
            row["delta_loglik_mean"] = poi.get("delta_loglik_mean")
            row["runtime_sec"] = entry.get("runtime_sec")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", type=str, nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = _v2_reports_root() / "mismatch"
    out_dir.mkdir(parents=True, exist_ok=True)

    scenario_ids = args.scenarios or sorted(MISMATCH_SCENARIOS.keys())
    rows = []
    for mid in scenario_ids:
        print(f"[{mid}]")
        rows.extend(evaluate_scenario(mid, seed=args.seed))

    df = pd.DataFrame(rows)
    out_csv = out_dir / "mismatch_v2_summary.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved {out_csv}")
    with pd.option_context("display.width", 240, "display.max_columns", 30):
        print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
