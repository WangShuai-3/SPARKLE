"""Probe: does refitting alpha against the converged latent X close the gap?

v2.0-alpha1 keeps alpha calibrated on the Y-based predictor. Hypothesis:
latent mode under-subtracts because alpha_Y was fit against the larger
Y-source. One alternating round (X -> refit alpha/R2 -> X) should recover.
"""
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, "/home/shuaiwang/21.STambiant")
from evaluation.scripts.final_comparison import (
    load_synthetic_scenario_data, _rmse_log1p_cp10k,
)
from evaluation.scripts.evaluate_synthetic_cell_r2 import cellwise_pearson_r2
from stambient.aggregation import (
    extract_cells, aggregate_empty_bin_expression, select_high_expression_genes,
)
from stambient.binning import bin_empty_dnbs
from stambient.graphs import build_cell_to_cell_graph, build_empty_to_cell_graph
from stambient.lambda_search import estimate_lambda
from stambient.alpha_estimation import estimate_alphas
from stambient.latent_correction import correct_cells_latent
from scipy.sparse import csr_matrix

LAMBDA_GRID = [10, 20, 30, 50, 70, 100, 150, 200, 300, 500]

for sid in ["S1", "S2", "S3", "S6"]:
    data = load_synthetic_scenario_data(sid, seed=42)
    dnb_expr, coords, labels = data["dnb_expr"], data["dnb_coords"], data["dnb_labels"]
    true_gc = np.asarray(data["true_expr"], dtype=np.float64)
    dnb_csc = dnb_expr.tocsc()
    n_genes = dnb_expr.shape[0]

    unique_cells, cell_areas, cell_centroids, cell_expr = extract_cells(dnb_csc, coords, labels)
    bin_coords, bin_areas, bin_dnb_idx, bin_bin_idx = bin_empty_dnbs(coords, labels, 25.0)
    bin_expr = aggregate_empty_bin_expression(dnb_csc, bin_dnb_idx, bin_bin_idx, len(bin_areas), n_genes)
    gene_indices = select_high_expression_genes(bin_expr, bin_areas, 80)
    e2c, _, _ = build_empty_to_cell_graph(bin_coords, cell_centroids, 300.0, None, None)
    best_lam, _, _, W_empty = estimate_lambda(
        lambda_grid=LAMBDA_GRID, distance_metric="exponential", gene_indices=gene_indices,
        n_lambda_genes=50, cell_expr=cell_expr, cell_areas=cell_areas,
        empty_bin_expr=bin_expr, empty_bin_areas=bin_areas,
        empty_to_cell_distances=e2c, empty_distance_gpu=None,
        use_expr_weight=False, gpu=None, storage_dtype=None, reduction_dtype=None, verbose=False)
    c2c, _, _ = build_cell_to_cell_graph(cell_centroids, 300.0, None, None)

    def run_latent(cell_expr_src, n_refits):
        alphas, r2, _ = estimate_alphas(
            gene_indices=gene_indices, cell_expr=cell_expr_src, cell_areas=cell_areas,
            empty_bin_expr=bin_expr, empty_bin_areas=bin_areas, W_empty=W_empty,
            use_expr_weight=False, gpu=None, storage_dtype=None, reduction_dtype=None,
            gpu_gene_batch_size=None)
        X = None
        for it in range(n_refits + 1):
            X, info = correct_cells_latent(
                gene_indices=gene_indices, r2_scores=r2, r2_threshold=0.01,
                n_genes=n_genes, n_cells=len(unique_cells), cell_expr=cell_expr,
                cell_areas=cell_areas, alphas=alphas, cell_distance_graph=c2c,
                lam_weights=best_lam, distance_metric="exponential",
                use_expr_weight=False, self_confidence_penalty=True,
                penalty_mode="1/(1+(s/p90)²)", gpu=None, storage_dtype=None,
                reduction_dtype=None, effective_gpu_gene_batch_size=None)
            if it < n_refits:
                alphas, r2, _ = estimate_alphas(
                    gene_indices=gene_indices, cell_expr=X, cell_areas=cell_areas,
                    empty_bin_expr=bin_expr, empty_bin_areas=bin_areas, W_empty=W_empty,
                    use_expr_weight=False, gpu=None, storage_dtype=None,
                    reduction_dtype=None, gpu_gene_batch_size=None)
        return X, alphas

    X0, a0 = run_latent(cell_expr, 0)   # alpha1 (= committed latent mode)
    X1, a1 = run_latent(cell_expr, 1)   # one alpha refit round
    X2, a2 = run_latent(cell_expr, 2)   # two rounds
    for name, X, a in [("alpha1 (current)", X0, a0), ("+1 alpha refit", X1, a1), ("+2 alpha refit", X2, a2)]:
        rmse = _rmse_log1p_cp10k(X, true_gc)
        r2m = float(np.nanmean(cellwise_pearson_r2(X.T, true_gc.T)))
        print(f"{sid} {name:<18} rmse={rmse:.4f} r2={r2m:.4f} alpha_mean={a.mean():.4f}")
