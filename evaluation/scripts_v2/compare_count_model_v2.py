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
from stambient.binning import bin_empty_dnbs
from stambient.count_model import (
    _fit_poisson_batch_cpu,
    _fit_poisson_batch_gpu,
    _poisson_nll,
)
from stambient.gpu import (
    choose_gpu_gene_batch_size,
    resolve_gpu,
    sparse_distance_graph_to_gpu_csr,
    sparse_mm,
    to_cpu,
    to_gpu,
    weighted_gpu_csr,
)
from stambient.graphs import build_empty_to_cell_graph
from stambient.lambda_search import estimate_lambda
from stambient.spatial import distance_graph_to_weights

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


def _ols_alphas(y, S, A):
    """Area-weighted one-parameter OLS (identical to estimate_alphas)."""
    w = A.astype(np.float64)[:, None]
    N = w * S
    denom = (w * N**2).sum(axis=0)
    alpha = np.divide(
        (w * y * N).sum(axis=0),
        denom,
        out=np.zeros(y.shape[1], dtype=np.float64),
        where=denom > 0,
    )
    return np.maximum(alpha, 0.0)


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
    e2c, _, e2c_gpu = build_empty_to_cell_graph(
        bin_coords, cell_centroids, cfg["max_radius"], gpu, storage_dtype
    )
    # Lambda is searched on the requested device (GPU for speed on the real
    # datasets); the fold fits below rebuild fold-specific W matrices from
    # the winning lambda, on CPU or GPU as available.
    best_lam, _, _, _ = estimate_lambda(
        lambda_grid=LAMBDA_GRID, distance_metric="exponential",
        gene_indices=gene_indices,
        n_lambda_genes=min(cfg["n_lambda_genes"], n_genes),
        cell_expr=cell_expr, cell_areas=cell_areas,
        empty_bin_expr=bin_expr, empty_bin_areas=bin_areas,
        empty_to_cell_distances=e2c, empty_distance_gpu=e2c_gpu,
        use_expr_weight=False, gpu=gpu, storage_dtype=storage_dtype,
        reduction_dtype=reduction_dtype, verbose=False,
    )
    if verbose:
        print(f"  lambda={best_lam}, {len(gene_indices)} genes, "
              f"{len(bin_areas)} empty bins")

    # The sparse W must be row-sliceable across folds, which GPU sparse CSR
    # is not; keep W on CPU and use it for S_all below.  On GPU, the fold
    # fits use a GPU-rebuilt fold W inside _fit_models below.
    W_empty = distance_graph_to_weights(e2c, best_lam, "exponential").tocsr()
    sources = np.divide(
        cell_expr[gene_indices], cell_areas[None, :],
        out=np.zeros((len(gene_indices), cell_expr.shape[1])),
        where=cell_areas[None, :] > 0,
    )
    S_all = W_empty.dot(sources.T)            # [n_bins x n_genes_use]
    y_all = bin_expr[gene_indices].T          # [n_bins x n_genes_use]
    blocks = _spatial_blocks(bin_coords, 2.0 * DEFAULT_EMPTY_BIN_SIZE_UM)

    def _fit_models(bin_expr_tr, areas_tr, e2c_tr):
        """Fit OLS / P1 / P2 on training bins; CPU or GPU batched."""
        if gpu is None:
            W_tr = distance_graph_to_weights(
                e2c_tr, best_lam, "exponential"
            ).tocsr()
            S_tr = W_tr.dot(sources.T)
            y_tr = bin_expr_tr[gene_indices].T
            A_tr = areas_tr
            alphas_ols = _ols_alphas(y_tr, S_tr, A_tr)
            _, rhos_p1, ll1, _ = _fit_poisson_batch_cpu(
                y_tr, A_tr, S_tr, False, 25, 1e-6
            )
            betas_p2, rhos_p2, ll2, _ = _fit_poisson_batch_cpu(
                y_tr, A_tr, S_tr, True, 25, 1e-6
            )
            # Same nested-model safeguard as estimate_leakage_poisson.
            better = ll1 > ll2
            betas_p2 = np.where(better, 0.0, betas_p2)
            rhos_p2 = np.where(better, rhos_p1, rhos_p2)
            return alphas_ols, rhos_p1, betas_p2, rhos_p2

        e2c_tr_gpu = sparse_distance_graph_to_gpu_csr(
            e2c_tr, gpu, dtype=storage_dtype
        )
        W_tr_gpu = weighted_gpu_csr(e2c_tr_gpu, best_lam, "exponential", gpu)
        A_tr_gpu = to_gpu(
            areas_tr[:, None].astype(np.float64), gpu, dtype=reduction_dtype
        )
        batch = choose_gpu_gene_batch_size(
            gpu, n_rows=e2c_tr.shape[0], n_cells=sources.shape[1],
            dtype=storage_dtype, max_batch_size=len(gene_indices),
        )
        alphas_ols = np.zeros(len(gene_indices))
        rhos_p1 = np.zeros(len(gene_indices))
        betas_p2 = np.zeros(len(gene_indices))
        rhos_p2 = np.zeros(len(gene_indices))
        for start in range(0, len(gene_indices), batch):
            stop = min(start + batch, len(gene_indices))
            y_tr_gpu = to_gpu(
                bin_expr_tr[gene_indices[start:stop]].T, gpu,
                dtype=reduction_dtype,
            )
            # Batch the sparse_mm as well: the full bins x genes product
            # does not fit on a 12GB card.
            Sb = sparse_mm(
                W_tr_gpu,
                to_gpu(sources[start:stop].T, gpu, dtype=storage_dtype),
                gpu,
            ).to(dtype=reduction_dtype)
            AS = A_tr_gpu * Sb
            denom = (A_tr_gpu * AS**2).sum(dim=0)
            alphas_ols[start:stop] = to_cpu(
                gpu.torch.where(
                    denom > 0,
                    ((A_tr_gpu * AS * y_tr_gpu).sum(dim=0) / denom).clamp_min(0.0),
                    gpu.torch.zeros_like(denom),
                ), gpu,
            )
            _, r1, ll1, _ = _fit_poisson_batch_gpu(
                y_tr_gpu, A_tr_gpu, Sb, gpu, False, 25, 1e-6
            )
            b2, r2, ll2, _ = _fit_poisson_batch_gpu(
                y_tr_gpu, A_tr_gpu, Sb, gpu, True, 25, 1e-6
            )
            # Same nested-model safeguard as estimate_leakage_poisson.
            better = ll1 > ll2
            b2 = gpu.torch.where(better, gpu.torch.zeros_like(b2), b2)
            r2 = gpu.torch.where(better, r1, r2)
            rhos_p1[start:stop] = to_cpu(r1, gpu)
            betas_p2[start:stop] = to_cpu(b2, gpu)
            rhos_p2[start:stop] = to_cpu(r2, gpu)
        return alphas_ols, rhos_p1, betas_p2, rhos_p2

    nll = {"OLS": 0.0, "P1": 0.0, "P2": 0.0, "NULL": 0.0}
    for fold in range(4):
        tr = blocks != fold
        va = blocks == fold
        bin_expr_tr = bin_expr[:, tr]
        areas_tr = bin_areas[tr]
        alphas_ols, rhos_p1, betas_p2, rhos_p2 = _fit_models(
            bin_expr_tr, areas_tr, e2c[tr]
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
    if out_csv.exists():
        old = pd.read_csv(out_csv)
        old = old[~old["dataset"].isin(df["dataset"])]
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(out_csv, index=False)
    print(f"\nSaved {out_csv}")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
