#!/usr/bin/env python3
"""Dropout control experiment: dropout_rate=0 vs the default 0.95.

Motivation: the 95% UMI dropout introduced for SoupX compatibility removes
weak long-range leaked counts, which biases SPARKLE's lambda estimate one
grid step downward.  For every scenario this script regenerates the data
with dropout disabled, runs all four methods (SPARKLE, SoupX, DecontX and
the official SpotClean), and records SPARKLE's estimated lambda plus
cell-level Pearson R2 and clustering ARI for every method.  The dropout=0.95
side is read from the production metrics JSONs.  All control outputs go to
evaluation/reports/dropout_control/; production artifacts are untouched.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stambient import SPARKLE
from evaluation.scripts.final_comparison import (
    load_synthetic_scenario_data,
    run_decontx_method,
    compute_cell_expr,
)
from evaluation.synthetic.scenarios import SCENARIOS
from evaluation.baselines.soupx import run_soupx
from evaluation.scripts.evaluate_synthetic_cell_r2 import cellwise_pearson_r2
from evaluation.scripts.evaluate_synthetic_clustering import clustering_ari
import evaluation.scripts.run_spotclean_official as spotclean_official_run

CONTROL_DIR = PROJECT_ROOT / "evaluation" / "reports" / "dropout_control"
METHODS = ["RAW", "SPARKLE", "SoupX", "DecontX", "SpotClean"]

# SPARKLE lambda estimated on the dropout=0.95 production run
# (self_confidence_penalty=True, logged in the rerun output).
LAMBDA_095 = {
    "S1": 30, "S2": 30, "S3": 30, "S4": 20, "S5": 300,
    "S6": 30, "S7": 30, "S8": 30, "S9": 30, "S10": 30,
}


def run_sparkle_synthetic(data):
    """SPARKLE with the synthetic recipe (penalty enabled)."""
    model = SPARKLE(
        bin_size=25,
        distance_metric="exponential",
        max_radius=300.0,
        n_high_genes=min(80, data["dnb_expr"].shape[0]),
        n_lambda_genes=min(50, data["dnb_expr"].shape[0]),
        r2_threshold=0.01,
        lambda_grid=[10, 20, 30, 50, 70, 100, 150, 200, 300, 500],
        cell_based=True,
        self_confidence_penalty=True,
        verbose=False,
    )
    corrected, diag = model.fit_transform_from_dnb(
        data["dnb_expr"], data["dnb_coords"], data["dnb_labels"]
    )
    return corrected, float(diag["lambda_estimated"])


def evaluate_matrix(corrected_gxc, data):
    """Cell-level Pearson R2 and clustering ARI of a genes-by-cells matrix."""
    cells = np.asarray(corrected_gxc, dtype=np.float64).T
    truth = np.asarray(data["true_expr"], dtype=np.float64).T
    r2 = cellwise_pearson_r2(cells, truth)
    return float(np.nanmean(r2)), float(clustering_ari(cells, data["cell_types"]))


def run_control_scenario(sid: str) -> list[dict]:
    """Run every method on one scenario regenerated with dropout_rate=0."""
    rows = []
    t_step = time.time()

    def _log(msg):
        print(f"  [{time.time() - t_step:6.1f}s] {sid} {msg}", flush=True)

    data = load_synthetic_scenario_data(sid)
    _log("generated")

    # RAW baseline
    raw_cell = compute_cell_expr(
        data["dnb_expr"], data["dnb_labels"], len(data["cell_ids"])
    )
    r2, ari = evaluate_matrix(raw_cell, data)
    rows.append({"scenario": sid, "dropout": 0.0, "method": "RAW",
                 "r2_mean": r2, "ari": ari})
    _log(f"RAW r2={r2:.3f} ari={ari:.3f}")

    # SPARKLE (also records lambda)
    t0 = time.time()
    corrected, lam = run_sparkle_synthetic(data)
    r2, ari = evaluate_matrix(corrected, data)
    rows.append({"scenario": sid, "dropout": 0.0, "method": "SPARKLE",
                 "lambda_est": lam, "r2_mean": r2, "ari": ari})
    _log(f"SPARKLE lambda={lam} r2={r2:.3f} ari={ari:.3f}")

    # SoupX (may fail without dropout; recorded as NaN)
    try:
        corrected, rho = run_soupx(
            data["dnb_expr"], data["dnb_labels"], tfidf_min=0.2, verbose=False
        )
        r2, ari = evaluate_matrix(corrected, data)
        rows.append({"scenario": sid, "dropout": 0.0, "method": "SoupX",
                     "r2_mean": r2, "ari": ari, "rho": rho})
        _log(f"SoupX rho={rho:.3f} r2={r2:.3f} ari={ari:.3f}")
    except Exception as exc:
        rows.append({"scenario": sid, "dropout": 0.0, "method": "SoupX",
                     "error": str(exc)})
        _log(f"SoupX FAILED: {str(exc)[:60]}")

    # DecontX
    corrected, diag = run_decontx_method(
        {"dnb_expr": data["dnb_expr"], "dnb_labels": data["dnb_labels"],
         "gene_names": list(data["gene_names"]), "cell_ids": data["cell_ids"]},
        verbose=False,
    )
    if corrected is not None:
        r2, ari = evaluate_matrix(corrected, data)
        rows.append({"scenario": sid, "dropout": 0.0, "method": "DecontX",
                     "r2_mean": r2, "ari": ari,
                     "contamination": diag.get("contamination")})
        _log(f"DecontX r2={r2:.3f} ari={ari:.3f}")
    else:
        rows.append({"scenario": sid, "dropout": 0.0, "method": "DecontX",
                     "error": str(diag.get("error"))})
        _log("DecontX FAILED")

    # Official SpotClean (outputs redirected to the control directory)
    args = SimpleNamespace(
        maxit=30, tol=1.0,
        rscript=spotclean_official_run.DEFAULT_RSCRIPT, overwrite=True,
    )
    try:
        spotclean_official_run.run_synthetic(sid, args)
        import anndata as ad
        result = ad.read_h5ad(CONTROL_DIR / "h5ad" / f"synthetic_{sid}_SpotCleanOfficial.h5ad")
        X = result.X.toarray() if hasattr(result.X, "toarray") else np.asarray(result.X)
        truth = np.asarray(data["true_expr"], dtype=np.float64).T
        r2 = float(np.nanmean(cellwise_pearson_r2(
            np.asarray(X, dtype=np.float64), truth)))
        ari = float(clustering_ari(np.asarray(X, dtype=np.float64),
                                   data["cell_types"]))
        rows.append({"scenario": sid, "dropout": 0.0, "method": "SpotClean",
                     "r2_mean": r2, "ari": ari})
        _log(f"SpotClean r2={r2:.3f} ari={ari:.3f}")
    except Exception as exc:
        rows.append({"scenario": sid, "dropout": 0.0, "method": "SpotClean",
                     "error": str(exc)})
        _log(f"SpotClean FAILED: {str(exc)[:60]}")

    print(f"  {sid}: control done ({time.time() - t_step:.0f}s)", flush=True)
    return rows


def load_production_rows(metrics_dir: Path, scenarios: list[str]) -> list[dict]:
    """Read the dropout=0.95 side from the production metrics JSONs."""
    rows = []
    for sid in scenarios:
        path = metrics_dir / f"synthetic_{sid}_metrics.json"
        metrics = json.loads(path.read_text(encoding="utf-8"))
        truth_lambda = float(SCENARIOS[sid]["ambient_lambda"])
        for method in METHODS:
            entry = metrics["raw"] if method == "RAW" else metrics["methods"].get(method)
            row = {"scenario": sid, "dropout": 0.95, "method": method,
                   "true_lambda": truth_lambda}
            if method == "SPARKLE":
                row["lambda_est"] = float(LAMBDA_095[sid])
            if entry is None:
                row["error"] = "no entry"
            else:
                row["r2_mean"] = entry.get("cell_pearson_r2_mean")
                row["ari"] = entry.get("clustering_ari")
            rows.append(row)
    return rows


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Dropout control experiment.")
    parser.add_argument("--scenarios", nargs="*", default=None,
                        help="Scenario IDs (default: all)")
    args_cli = parser.parse_args()

    if args_cli.scenarios:
        scenarios = args_cli.scenarios
    else:
        scenarios = sorted(SCENARIOS.keys(), key=lambda s: int(s[1:]))
    CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    spotclean_official_run.REPORTS = CONTROL_DIR

    # Force dropout off for the control run; restore afterwards.
    saved = {sid: SCENARIOS[sid].get("dropout_rate", 0.0) for sid in scenarios}
    for sid in scenarios:
        SCENARIOS[sid]["dropout_rate"] = 0.0

    control_rows = []
    out_csv = CONTROL_DIR / "dropout_control_summary.csv"
    try:
        for sid in scenarios:
            control_rows.extend(run_control_scenario(sid))
            # Save partial results after every scenario (interruption-safe).
            pd.DataFrame(control_rows).to_csv(out_csv, index=False)
    finally:
        for sid in scenarios:
            SCENARIOS[sid]["dropout_rate"] = saved[sid]

    prod_rows = load_production_rows(
        PROJECT_ROOT / "evaluation" / "reports" / "metrics", scenarios
    )
    df = pd.DataFrame(control_rows + prod_rows)
    df["true_lambda"] = df["scenario"].map(
        {sid: float(SCENARIOS[sid]["ambient_lambda"]) for sid in scenarios}
    )
    df.to_csv(out_csv, index=False)
    print(f"Saved {out_csv}")

    # Side-by-side comparison table.
    print("\n=== SPARKLE lambda: truth vs dropout=0 / 0.95 ===")
    lam = df[df["method"] == "SPARKLE"].pivot(
        index="scenario", columns="dropout", values="lambda_est")
    lam["true"] = lam.index.map(
        {sid: SCENARIOS[sid]["ambient_lambda"] for sid in scenarios})
    print(lam.to_string())

    for metric in ("r2_mean", "ari"):
        print(f"\n=== {metric}: dropout=0 / 0.95 ===")
        piv = df.pivot_table(index=["scenario", "method"], columns="dropout",
                             values=metric)
        print(piv.round(3).to_string())


if __name__ == "__main__":
    main()
