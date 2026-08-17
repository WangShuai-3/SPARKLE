#!/usr/bin/env python3
"""Held-out spatial-block deviance: OLS vs Poisson count models (Phase 2).

For each dataset the background-bin model is fit on a spatially held-out
checkerboard of empty bins (4 folds, block edge = 2x bin size) and scored by
Poisson negative log-likelihood on the held-out bins:

    OLS  - mu = A * alpha_OLS * S        (v1 observation model)
    P1   - mu = A * rho * S              (Poisson, local-only; B1)
    P2   - mu = A * (beta + rho * S)     (Poisson, diffuse+local; B2)
    NULL - mu = A * beta0                (rate-only reference)

lambda, W_empty and the gene panel are fixed across folds (estimated once on
all bins) so the comparison isolates the background observation model.

Outputs:
    evaluation/reports/v2/count_model/count_model_deviance.csv

Run from the repository root, e.g.:
    python evaluation/scripts_v2/compare_count_model_v2.py \
        --datasets S2,M1,M2
    conda run -n scvi python evaluation/scripts_v2/compare_count_model_v2.py \
        --datasets mousebrain --use-gpu --gpu-dtype float64
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.scripts.final_comparison import (
    DEFAULT_EMPTY_BIN_SIZE_UM,
    load_axolotl_data_windowed,
    load_mousebrain_data,
    load_ovarian_data,
    load_synthetic_scenario_data,
    subsample_data,
)
from evaluation.scripts_v2.mismatch import load_mismatch_scenario_data
from evaluation.scripts_v2.run_correction_v2 import DATASET_CONFIG, LAMBDA_GRID
from stambient.aggregation import (
    aggregate_empty_bin_expression,
    extract_cells,
    select_high_expression_genes,
)
from stambient.alpha_estimation import estimate_alphas
from stambient.binning import bin_empty_dnbs
from stambient.count_model import _poisson_nll, estimate_leakage_poisson
from stambient.gpu import resolve_gpu
from stambient.graphs import build_empty_to_cell_graph
from stambient.lambda_search import estimate_lambda

SYNTHETIC_IDS = ("S2", "M1", "M2")
REAL_IDS = ("axolotl", "mousebrain", "ovarian")


def _load_dataset(dataset_id, seed=42):
    """Return (dnb_expr, dnb_coords_um, dnb_labels, n_high, n_lambda)."""
    if dataset_id in SYNTHETIC_IDS:
        if dataset_id.startswith("M"):
            data = load_mismatch_scenario_data(dataset_id, seed=seed)
        else:
            data = load_synthetic_scenario_data(dataset_id, seed=seed)
        cfg = DATASET_CONFIG["synthetic"]
        coords = np.asarray(data["dnb_coords"], dtype=np.float64)
        return data["dnb_expr"], coords, data["dnb_labels"], cfg

    cfg = DATASET_CONFIG[dataset_id]
    if dataset_id == "axolotl":
        sub = load_axolotl_data_windowed(cfg["x_range"], cfg["y_range"])
    elif dataset_id == "mousebrain":
        data = load_mousebrain_data(x_range=cfg["x_range"],
                                    y_range=cfg["y_range"],
                                    annotation_level="cell_group")
        sub = subsample_data(data, cfg["n_genes"], cut_genes=False)
    else:
        data = load_ovarian_data(x_range=cfg["x_range"], y_range=cfg["y_range"],
                                 n_genes=cfg["n_genes"])
        sub = subsample_data(data, cfg["n_genes"], cut_genes=False)
    coords = np.asarray(sub["dnb_coords"], dtype=np.float64) * cfg["coord_scale"]
    return sub["dnb_expr"], coords, sub["dnb_labels"], cfg


def _spatial_blocks(bin_coords, block_size):
    """2x2 checkerboard fold ids in {0,1,2,3} for each empty bin."""
    xy = (bin_coords - bin_coords.min(axis=0)) / block_size
    bx = np.floor(xy[:, 0]).astype(np.int64) % 2
    by = np.floor(xy[:, 1]).astype(np.int64) % 2
    return bx + 2 * by


def _heldout_nll(y, mu):
    """-loglik(y; mu) with the constant log(y!) term dropped."""
    return float(_poisson_nll(y, mu).sum())


def evaluate_dataset(dataset_id, gpu, storage_dtype, reduction_dtype,
                     seed=42, verbose=True):
    dnb_expr, coords_um, labels, cfg = _load_dataset(dataset_id, seed=seed)
    dnb_csc = dnb_expr.tocsc()
    n_genes = dnb_expr.shape[0]

    _, cell_areas, cell_centroids, cell_expr = extract_cells(
        dnb_csc, coords_um, labels
    )
    bin_coords, bin_areas, bin_dnb_idx, bin_bin_idx = bin_empty_dnbs(
        coords_um, labels, DEFAULT_EMPTY_BIN_SIZE_UM
    )
    bin_expr = aggregate_empty_bin_expression(
        dnb_csc, bin_dnb_idx, bin_bin_idx, len(bin_areas), n_genes
    )
    n_high = min(cfg["n_high_genes"], n_genes)
    gene_indices = select_high_expression_genes(bin_expr, bin_areas, n_high)
    e2c, _, _ = build_empty_to_cell_graph(
        bin_coords, cell_centroids, cfg["max_radius"], None, None
    )
    best_lam, _, _, W_empty = estimate_lambda(
        lambda_grid=LAMBDA_GRID, distance_metric="exponential",
        gene_indices=gene_indices,
        n_lambda_genes=min(cfg["n_lambda_genes"], n_genes),
        cell_expr=cell_expr, cell_areas=cell_areas,
        empty_bin_expr=bin_expr, empty_bin_areas=bin_areas,
        empty_to_cell_distances=e2c, empty_distance_gpu=None,
        use_expr_weight=False, gpu=gpu, storage_dtype=storage_dtype,
        reduction_dtype=reduction_dtype, verbose=False,
    )
    if verbose:
        print(f"  lambda={best_lam}, {len(gene_indices)} genes, "
              f"{len(bin_areas)} empty bins")

    W_empty = W_empty.tocsr()
    sources = np.divide(
        cell_expr[gene_indices], cell_areas[None, :],
        out=np.zeros((len(gene_indices), cell_expr.shape[1])),
        where=cell_areas[None, :] > 0,
    )
    S_all = W_empty.dot(sources.T)            # [n_bins x n_genes_use]
    y_all = bin_expr[gene_indices].T          # [n_bins x n_genes_use]
    blocks = _spatial_blocks(bin_coords, 2.0 * DEFAULT_EMPTY_BIN_SIZE_UM)

    nll = {"OLS": 0.0, "P1": 0.0, "P2": 0.0, "NULL": 0.0}
    for fold in range(4):
        tr = blocks != fold
        va = blocks == fold
        bin_expr_tr = bin_expr[:, tr]
        areas_tr = bin_areas[tr]
        W_tr = W_empty[tr]

        alphas_ols, _, _ = estimate_alphas(
            gene_indices=gene_indices, cell_expr=cell_expr,
            cell_areas=cell_areas, empty_bin_expr=bin_expr_tr,
            empty_bin_areas=areas_tr, W_empty=W_tr, use_expr_weight=False,
            gpu=gpu, storage_dtype=storage_dtype,
            reduction_dtype=reduction_dtype, gpu_gene_batch_size=None,
        )
        betas_p1, rhos_p1, _, _ = estimate_leakage_poisson(
            gene_indices=gene_indices, cell_expr=cell_expr,
            cell_areas=cell_areas, empty_bin_expr=bin_expr_tr,
            empty_bin_areas=areas_tr, W_empty=W_tr, use_expr_weight=False,
            fit_diffuse=False, gpu=gpu, storage_dtype=storage_dtype,
            reduction_dtype=reduction_dtype, gpu_gene_batch_size=None,
        )
        betas_p2, rhos_p2, _, _ = estimate_leakage_poisson(
            gene_indices=gene_indices, cell_expr=cell_expr,
            cell_areas=cell_areas, empty_bin_expr=bin_expr_tr,
            empty_bin_areas=areas_tr, W_empty=W_tr, use_expr_weight=False,
            fit_diffuse=True, gpu=gpu, storage_dtype=storage_dtype,
            reduction_dtype=reduction_dtype, gpu_gene_batch_size=None,
        )

        y_va = y_all[va]
        A_va = bin_areas[va][:, None]
        S_va = S_all[va]
        nll["OLS"] += _heldout_nll(y_va, A_va * alphas_ols[None, :] * S_va)
        nll["P1"] += _heldout_nll(y_va, A_va * rhos_p1[None, :] * S_va)
        nll["P2"] += _heldout_nll(
            y_va, A_va * (betas_p2[None, :] + rhos_p2[None, :] * S_va)
        )
        beta0 = bin_expr_tr[gene_indices].sum(axis=1) / max(areas_tr.sum(), 1e-12)
        nll["NULL"] += _heldout_nll(y_va, A_va * beta0[None, :])

    row = {
        "dataset": dataset_id,
        "lambda": float(best_lam),
        "n_genes": len(gene_indices),
        "n_bins": len(bin_areas),
    }
    for name in ("NULL", "OLS", "P1", "P2"):
        row[f"nll_{name}_per_gene"] = nll[name] / len(gene_indices)
    row["dnll_OLS_vs_null"] = (nll["OLS"] - nll["NULL"]) / len(gene_indices)
    row["dnll_P1_vs_OLS"] = (nll["P1"] - nll["OLS"]) / len(gene_indices)
    row["dnll_P2_vs_P1"] = (nll["P2"] - nll["P1"]) / len(gene_indices)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", type=str,
                        default=",".join(SYNTHETIC_IDS),
                        help=f"Comma list from: {','.join(SYNTHETIC_IDS + REAL_IDS)}")
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--gpu-dtype", type=str, default="float64",
                        choices=["float64", "float32"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    gpu, reason = resolve_gpu(args.use_gpu)
    if args.use_gpu and gpu is None:
        print(f"[warn] GPU requested but unavailable: {reason}; using CPU")
    storage_dtype = reduction_dtype = None
    if gpu is not None:
        storage_dtype = reduction_dtype = (
            gpu.torch.float64 if args.gpu_dtype == "float64"
            else gpu.torch.float32
        )

    rows = []
    for dataset_id in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        print(f"[{dataset_id}]")
        rows.append(evaluate_dataset(
            dataset_id, gpu, storage_dtype, reduction_dtype, seed=args.seed,
        ))

    out_dir = PROJECT_ROOT / "evaluation" / "reports" / "v2" / "count_model"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    out_csv = out_dir / "count_model_deviance.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved {out_csv}")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
