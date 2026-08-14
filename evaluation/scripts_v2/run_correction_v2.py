#!/usr/bin/env python3
"""SPARKLE 2.x correction runner: RAW / SPARKLE 1.x (A0) / SPARKLE 2.x (A1, A2).

Runs the exact v1 manuscript datasets/windows/parameters, varying only the
inference ablation:

    RAW        - aggregated, uncorrected per-cell expression
    SPARKLEv1  - inference_mode="legacy", self_confidence_penalty=True   (A0)
    SPARKLEv2  - inference_mode="latent", self_confidence_penalty=True   (A1)
    SPARKLEv2NP- inference_mode="latent", self_confidence_penalty=False  (A2)

Outputs (isolated from the v1 reports):
    evaluation/reports/v2/h5ad/{tag}_{method}.h5ad
    evaluation/reports/v2/metrics/{tag}_metrics_v2.json

Run from the repository root, e.g.:
    conda run -n scvi python evaluation/scripts_v2/run_correction_v2.py \
        --dataset synthetic --all-scenarios
    conda run -n scvi python evaluation/scripts_v2/run_correction_v2.py \
        --dataset mousebrain --use-gpu --gpu-dtype float64
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stambient import SPARKLE
from evaluation.scripts.final_comparison import (
    DEFAULT_EMPTY_BIN_SIZE_UM,
    STEREOSEQ_PITCH_UM,
    _build_sparkle_var_data,
    compute_cell_expr,
    load_axolotl_data_windowed,
    load_mousebrain_data,
    load_ovarian_data,
    load_synthetic_scenario_data,
    save_metrics_json,
    save_result_h5ad,
    subsample_data,
)
from evaluation.synthetic import SCENARIOS

METHODS = ["raw", "SPARKLEv1", "SPARKLEv2", "SPARKLEv2NP"]
LAMBDA_GRID = [10, 20, 30, 50, 70, 100, 150, 200, 300, 500]


def _v2_reports_root():
    root = PROJECT_ROOT / "evaluation" / "reports" / "v2"
    (root / "h5ad").mkdir(parents=True, exist_ok=True)
    (root / "metrics").mkdir(parents=True, exist_ok=True)
    return root


# Fixed v1 manuscript settings per dataset.
DATASET_CONFIG = {
    "synthetic": dict(x_range=None, y_range=None, n_genes=200, n_high_genes=80,
                      n_lambda_genes=50, coord_scale=1.0, max_radius=300.0,
                      tag="synthetic"),
    "axolotl": dict(x_range=(10500, 12500), y_range=(6000, 11100),
                    n_genes=200, n_high_genes=200, n_lambda_genes=100,
                    coord_scale=STEREOSEQ_PITCH_UM,
                    max_radius=200.0, tag="axolotl_x10500-12500_y6000-11100"),
    "mousebrain": dict(x_range=(12500, 20000), y_range=(2000, 10000),
                       n_genes=30000, n_high_genes=30000,
                       n_lambda_genes=100, coord_scale=STEREOSEQ_PITCH_UM,
                       max_radius=300.0,
                       tag="mousebrain_x12500-20000_y2000-10000"),
    "ovarian": dict(x_range=(1000, 1800), y_range=(300, 1100),
                    n_genes=30000, n_high_genes=30000, n_lambda_genes=100,
                    coord_scale=1.0,
                    max_radius=300.0, tag="ovarian_x1000-1800_y300-1100"),
}


def _run_sparkle_v2(sub, cfg, inference_mode, penalty, use_gpu, gpu_dtype,
                    r2_threshold=0.01, verbose=False):
    """Mirror run_sparkle_method with the 2.x ablation knobs exposed."""
    dnb_expr = sub["dnb_expr"]
    dnb_coords_um = np.asarray(sub["dnb_coords"], dtype=np.float64) * cfg["coord_scale"]
    dnb_labels = sub["dnb_labels"]
    n_high = min(cfg["n_high_genes"], dnb_expr.shape[0])
    model = SPARKLE(
        bin_size=DEFAULT_EMPTY_BIN_SIZE_UM,
        distance_metric="exponential",
        max_radius=cfg["max_radius"],
        n_high_genes=n_high,
        n_lambda_genes=min(cfg["n_lambda_genes"], dnb_expr.shape[0]),
        r2_threshold=r2_threshold,
        lambda_grid=LAMBDA_GRID,
        cell_based=True,
        self_confidence_penalty=penalty,
        verbose=verbose,
        use_gpu=use_gpu,
        gpu_dtype=gpu_dtype,
        inference_mode=inference_mode,
    )
    t0 = time.time()
    corrected, diag = model.fit_transform_from_dnb(dnb_expr, dnb_coords_um, dnb_labels)
    runtime = time.time() - t0
    if hasattr(corrected, "toarray"):
        corrected = corrected.toarray()
    var_data = _build_sparkle_var_data(dnb_expr.shape[0], diag, r2_threshold)
    return corrected, {"runtime": runtime, "var_data": var_data, **diag}


def run_dataset(dataset, sub, cell_ids, ann_map, tag, methods, use_gpu,
                gpu_dtype, save_h5ad, seed=42, verbose=False):
    cfg = DATASET_CONFIG[dataset]
    reports = _v2_reports_root()
    metrics_path = reports / "metrics" / f"{tag}_metrics_v2.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics.setdefault("methods", {})
    else:
        metrics = {"dataset": tag, "seed": seed, "raw": {}, "methods": {}}

    raw_cell = compute_cell_expr(sub["dnb_expr"], sub["dnb_labels"], len(cell_ids))
    metrics["raw"].update({
        "library_total": float(raw_cell.sum()),
        "n_cells": int(raw_cell.shape[1]),
        "n_genes": int(raw_cell.shape[0]),
    })

    if "raw" in methods:
        save_result_h5ad(raw_cell, sub["gene_names"], cell_ids, ann_map,
                         reports / "h5ad" / f"{tag}_raw.h5ad", "RAW",
                         save_h5ad=save_h5ad)

    specs = {
        "SPARKLEv1": dict(inference_mode="legacy", penalty=True),
        "SPARKLEv2": dict(inference_mode="latent", penalty=True),
        "SPARKLEv2NP": dict(inference_mode="latent", penalty=False),
    }
    for name, spec in specs.items():
        if name not in methods:
            continue
        print(f"\n[{tag}] {name} ({spec['inference_mode']}, "
              f"penalty={spec['penalty']})...")
        corrected, diag = _run_sparkle_v2(
            sub, cfg, spec["inference_mode"], spec["penalty"],
            use_gpu, gpu_dtype, verbose=verbose,
        )
        save_result_h5ad(corrected, sub["gene_names"], cell_ids, ann_map,
                         reports / "h5ad" / f"{tag}_{name}.h5ad", name,
                         save_h5ad=save_h5ad, var_data=diag["var_data"])
        entry = {
            "inference_mode": spec["inference_mode"],
            "self_confidence_penalty": spec["penalty"],
            "runtime_sec": diag["runtime"],
            "lambda_estimated": diag["lambda_estimated"],
            "n_genes_corrected": diag["n_genes_corrected"],
            "compute_backend": diag["compute_backend"],
            "library_total": float(corrected.sum()),
        }
        if diag.get("latent") is not None:
            lat = diag["latent"]
            entry["latent"] = {
                "eta": lat["eta"],
                "max_iter": lat["max_iter"],
                "tol": lat["tol"],
                "converged": lat["converged"],
                "n_iterations_max": max(lat["n_iterations"], default=0),
                "final_rel_change_max": max(lat["final_rel_change"], default=float("nan")),
            }
        metrics["methods"][name] = entry
        print(f"  ✓ {name} done in {diag['runtime']:.1f}s "
              f"(λ={diag['lambda_estimated']}, backend={diag['compute_backend']})")

    save_metrics_json(metrics, metrics_path)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True,
                        choices=["synthetic", "axolotl", "mousebrain", "ovarian"])
    parser.add_argument("--scenario", type=str, default=None,
                        help="Synthetic scenario id (e.g. S6)")
    parser.add_argument("--all-scenarios", action="store_true")
    parser.add_argument("--methods", type=str, default=",".join(METHODS),
                        help=f"Comma list from: {','.join(METHODS)}")
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--gpu-dtype", type=str, default="float64",
                        choices=["float64", "mixed", "float32"])
    parser.add_argument("--save-h5ad", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    unknown = sorted(set(methods) - set(METHODS))
    if unknown:
        parser.error(f"unknown method(s): {', '.join(unknown)}")

    if args.dataset == "synthetic":
        scenarios = sorted(SCENARIOS.keys()) if args.all_scenarios else [
            args.scenario or "S1"
        ]
        for sid in scenarios:
            data = load_synthetic_scenario_data(sid, seed=args.seed)
            run_dataset("synthetic", data, data["cell_ids"], None,
                        f"synthetic_{sid}", methods, args.use_gpu,
                        args.gpu_dtype, args.save_h5ad, seed=args.seed,
                        verbose=args.verbose)
        return

    cfg = DATASET_CONFIG[args.dataset]
    if args.dataset == "axolotl":
        data = load_axolotl_data_windowed(cfg["x_range"], cfg["y_range"])
        sub = data  # v1 runs the full gene set for axolotl
    elif args.dataset == "mousebrain":
        data = load_mousebrain_data(x_range=cfg["x_range"], y_range=cfg["y_range"],
                                    annotation_level="cell_group")
        sub = subsample_data(data, cfg["n_genes"], cut_genes=False)
    else:
        data = load_ovarian_data(x_range=cfg["x_range"], y_range=cfg["y_range"],
                                 n_genes=cfg["n_genes"])
        sub = subsample_data(data, cfg["n_genes"], cut_genes=False)

    run_dataset(args.dataset, sub, data["cell_ids"], data.get("ann_map"),
                cfg["tag"], methods, args.use_gpu, args.gpu_dtype,
                args.save_h5ad, verbose=args.verbose)


if __name__ == "__main__":
    main()
