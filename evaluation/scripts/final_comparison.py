#!/usr/bin/env python3
"""Final comparison: cell-based SPARKLE and manuscript baselines.

Supports the synthetic scenarios and the Axolotl, MouseBrain, and Ovarian
windows used in the final manuscript.
"""

import sys, os, time, argparse, gzip, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from scipy.sparse import csr_matrix, issparse, lil_matrix
from scipy.stats import ttest_ind
import anndata as ad
from stambient import SPARKLE
try:
    from evaluation.baselines.soupx import run_soupx
except ImportError as exc:
    _SOUPX_IMPORT_ERROR = exc

    def run_soupx(*args, **kwargs):
        raise ImportError(
            "SoupX is unavailable. Install soupx-python to run that baseline."
        ) from _SOUPX_IMPORT_ERROR
from evaluation.synthetic import generate_synthetic_data, SCENARIOS
from scipy.spatial import cKDTree

STEREOSEQ_PITCH_UM = 0.5
DEFAULT_EMPTY_BIN_SIZE_UM = 25.0
SYNTHETIC_GRID_SIZE_DNB = 500

# Optional line-by-line memory profiler; falls back to no-op if not installed.
try:
    from memory_profiler import profile
except ImportError:
    def profile(func):
        return func


def compute_neighbor_stats(cell_centroids, sstin_mask, radius=50.0):
    """Find non-sstIN neighbors within radius of any sstIN cell.

    Originally from evaluation/scripts/test_axolotl.py; inlined here after
    that file was removed as unused standalone data script.
    """
    sstin_indices = np.where(sstin_mask)[0]
    non_sstin_indices = np.where(~sstin_mask)[0]

    if len(sstin_indices) == 0:
        return np.zeros(len(sstin_mask), dtype=bool), np.zeros(len(sstin_mask), dtype=bool)

    tree = cKDTree(cell_centroids[non_sstin_indices])
    neighbor_set = set()
    for si in sstin_indices:
        nbrs = tree.query_ball_point(cell_centroids[si], radius)
        for n in nbrs:
            neighbor_set.add(non_sstin_indices[n])

    neighbor_mask = np.zeros(len(sstin_mask), dtype=bool)
    if neighbor_set:
        neighbor_mask[list(neighbor_set)] = True
    other_mask = ~sstin_mask & ~neighbor_mask
    return neighbor_mask, other_mask


def load_synthetic_scenario_data(scenario_id="S1", seed=42):
    """Load a synthetic benchmark scenario (S1–S10).

    Data is always generated on the fly using evaluation.synthetic.generator.
    This avoids NumPy version compatibility issues with old cached NPZ files.

    Returns a dict compatible with the comparison pipeline:
        dnb_expr, dnb_coords, dnb_labels, gene_names, cell_ids,
        true_expr, gene_is_high, params, true_alpha, true_lambda, metadata.
    """
    from evaluation.synthetic.scenarios import get_scenario

    scenario = get_scenario(scenario_id)
    print(f"[Synthetic] Scenario {scenario_id}: {scenario['name']}")

    print(f"  Generating synthetic data on the fly...")
    data = generate_synthetic_data(
        n_cells=scenario.get("n_cells", 200),
        grid_width=SYNTHETIC_GRID_SIZE_DNB,
        grid_height=SYNTHETIC_GRID_SIZE_DNB,
        dnb_pitch=0.5,
        cell_radius=5.0,
        cell_radius_cv=0.2,
        n_genes=500,
        n_high_genes=80,
        ambient_lambda=scenario["ambient_lambda"],
        ambient_alpha=scenario["ambient_alpha"],
        empty_fraction=scenario["empty_fraction"],
        n_cell_types=scenario.get("n_cell_types", 1),
        marker_fraction=scenario.get("marker_fraction", 0.0),
        cluster_strength=scenario.get("cluster_strength", 0.5),
        dropout_rate=scenario.get("dropout_rate", 0.0),
        seed=seed,
    )

    n_genes = data["dnb_expr"].shape[0]
    n_cells = data["true_expr"].shape[1]
    gene_names = np.array([f"gene_{i}" for i in range(n_genes)])
    cell_ids = np.arange(n_cells, dtype=np.int64)

    print(f"  Ground-truth λ={scenario['ambient_lambda']}µm, "
          f"α(mean)={data['true_alpha'].mean():.4f}, empty_fraction={scenario['empty_fraction']}")

    return {
        "dnb_expr": csr_matrix(data["dnb_expr"].astype(np.float64)),
        "dnb_coords": data["dnb_coords"],
        "dnb_labels": data["dnb_labels"],
        "gene_names": gene_names,
        "cell_ids": cell_ids,
        "true_expr": data["true_expr"],
        "scenario_id": scenario_id,
        "gene_is_high": data["gene_is_high"],
        "true_alpha": data["true_alpha"],
        "true_lambda": data["true_lambda"],
        "params": data["params"],
    }


def _rmse(pred, true):
    """Root-mean-square error over all elements."""
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def run_synthetic_comparison(
    data,
    n_genes=500,
    methods=None,
    lambda_grid=None,
    r2_threshold=None,
    save_h5ad=True,
    max_radius=None,
    use_gpu=False,
):
    """Run methods on a synthetic scenario and report RMSE reduction vs raw."""
    if methods is None:
        methods = ["sparkle", "soupx", "decontx"]
    methods = [m.lower().strip() for m in methods]

    dnb_expr = data["dnb_expr"]
    dnb_coords = data["dnb_coords"]
    dnb_labels = data["dnb_labels"]
    true_expr = data["true_expr"]
    n_cells = true_expr.shape[1]

    # Per-cell raw expression
    raw_cell = compute_cell_expr(dnb_expr, dnb_labels, n_cells)
    rmse_raw = _rmse(raw_cell, true_expr)

    print(f"\n{'='*60}")
    print("SYNTHETIC SCENARIO EVALUATION")
    print(f"{'='*60}")
    print(f"  Raw RMSE: {rmse_raw:.4f}")

    results = {}
    if lambda_grid is None:
        lambda_grid_sp = [10, 20, 30, 50, 70, 100, 150, 200, 300, 500]
    else:
        lambda_grid_sp = lambda_grid

    if r2_threshold is None:
        r2_threshold_sp = 0.01
    else:
        r2_threshold_sp = r2_threshold

    if max_radius is None:
        max_radius_sp = 300.0
    else:
        max_radius_sp = max_radius

    # 1. SPARKLE
    if "sparkle" in methods:
        print(f"\n[SPARKLE]")
        t0 = time.time()
        model = SPARKLE(
            bin_size=25,
            distance_metric="exponential",
            max_radius=max_radius_sp,
            n_high_genes=min(80, dnb_expr.shape[0]),
            n_lambda_genes=min(50, dnb_expr.shape[0]),
            r2_threshold=r2_threshold_sp,
            lambda_grid=lambda_grid_sp,
            cell_based=True,
            self_confidence_penalty=False,
            verbose=False,
            use_gpu=use_gpu,
        )
        sp_corr, diag = model.fit_transform_from_dnb(dnb_expr, dnb_coords, dnb_labels)
        sp_t = time.time() - t0
        if hasattr(sp_corr, "toarray"):
            sp_corr = sp_corr.toarray()
        rmse_sp = _rmse(sp_corr, true_expr)
        reduc_sp = (rmse_raw - rmse_sp) / rmse_raw * 100.0
        print(f"  RMSE={rmse_sp:.4f}, reduction={reduc_sp:.1f}%, "
              f"λ={model.lambda_:.0f}µm, time={sp_t:.1f}s")
        sp_var_data = _build_sparkle_var_data(dnb_expr.shape[0], diag, r2_threshold_sp)
        results["SPARKLE"] = {"rmse": rmse_sp, "reduction": reduc_sp, "runtime": sp_t,
                               "diag": {"var_data": sp_var_data}}

    # 2. SoupX
    if "soupx" in methods:
        print(f"\n[SoupX]")
        t0 = time.time()
        try:
            # Synthetic scenarios use 3-5 balanced cell types, so the best
            # achievable tf-idf is log(n_types) ≈ 1.1; the default tfidfMin=1.0
            # leaves no headroom once any background expression is present.
            # tfidf_min=0.2 is the documented protocol choice for synthetic.
            sx_corr, sx_rho = run_soupx(
                dnb_expr, dnb_labels, tfidf_min=0.2, verbose=False
            )
        except Exception as e:
            # No heuristic substitution: record the failure explicitly as NaN.
            sx_t = time.time() - t0
            print(f"  SoupX FAILED ({e}); recording NaN metrics")
            results["SoupX"] = {
                "rmse": float("nan"),
                "reduction": float("nan"),
                "runtime": sx_t,
                "error": str(e),
            }
        else:
            sx_t = time.time() - t0
            rmse_sx = _rmse(sx_corr, true_expr)
            reduc_sx = (rmse_raw - rmse_sx) / rmse_raw * 100.0
            print(f"  RMSE={rmse_sx:.4f}, reduction={reduc_sx:.1f}%, "
                  f"ρ={sx_rho:.4f}, time={sx_t:.1f}s")
            results["SoupX"] = {"rmse": rmse_sx, "reduction": reduc_sx, "runtime": sx_t}

    # 3. DecontX
    if "decontx" in methods:
        print(f"\n[DecontX]")
        t0 = time.time()
        dx_corr, dx_diag = run_decontx_method({
            "dnb_expr": dnb_expr,
            "dnb_labels": dnb_labels,
            "gene_names": list(data["gene_names"]),
            "cell_ids": data["cell_ids"],
        }, verbose=False)
        dx_t = time.time() - t0
        if dx_corr is not None:
            rmse_dx = _rmse(dx_corr, true_expr)
            reduc_dx = (rmse_raw - rmse_dx) / rmse_raw * 100.0
            print(f"  RMSE={rmse_dx:.4f}, reduction={reduc_dx:.1f}%, "
                  f"contamination={dx_diag.get('contamination', 'N/A'):.3f}, time={dx_t:.1f}s")
            results["DecontX"] = {"rmse": rmse_dx, "reduction": reduc_dx, "runtime": dx_t}

    # Save cell-based h5ad and metrics
    print(f"\n  {'='*60}")
    print(f"  Saving results")
    print(f"  {'='*60}")
    reports_root = _reports_root()
    tag = f"synthetic_{data['scenario_id']}"
    save_result_h5ad(raw_cell, data['gene_names'], data['cell_ids'], None,
                     reports_root / "h5ad" / f"{tag}_raw.h5ad", "RAW",
                     save_h5ad=save_h5ad)
    metrics = {
        "dataset": tag,
        "scenario": data.get("scenario_name", ""),
        "n_cells": int(raw_cell.shape[1]),
        "n_genes": int(raw_cell.shape[0]),
        "rmse_raw": rmse_raw,
        "raw": {"rmse": rmse_raw},
        "methods": {},
    }
    for method_name, r in results.items():
        # Try to recover corrected matrix saved in result dict
        corrected = None
        if method_name == "SPARKLE" and 'sp_corr' in locals():
            corrected = sp_corr
        elif method_name == "SoupX" and 'sx_corr' in locals():
            corrected = sx_corr
        elif method_name == "DecontX" and 'dx_corr' in locals():
            corrected = dx_corr
        if corrected is not None:
            var_data = r.get('diag', {}).get('var_data') if method_name == 'SPARKLE' else None
            save_result_h5ad(corrected, data['gene_names'], data['cell_ids'], None,
                             reports_root / "h5ad" / f"{tag}_{method_name}.h5ad",
                             method_name,
                             save_h5ad=save_h5ad,
                             var_data=var_data)
        metrics["methods"][method_name] = {
            "rmse": r['rmse'],
            "reduction_pct": r['reduction'],
            "runtime": r['runtime'],
        }
    save_metrics_json(metrics, reports_root / "metrics" / f"{tag}_metrics.json")

    # Summary table
    print(f"\n{'='*60}")
    print("SUMMARY (RMSE reduction vs raw)")
    print(f"{'='*60}")
    print(f"  {'Method':<16} {'RMSE':>10} {'Reduction':>12} {'Runtime':>10}")
    print(f"  {'-'*16} {'-'*10} {'-'*12} {'-'*10}")
    print(f"  {'RAW':<16} {rmse_raw:>10.4f} {'—':>12} {'—':>10}")
    for name, r in results.items():
        print(f"  {name:<16} {r['rmse']:>10.4f} {r['reduction']:>11.1f}% {r['runtime']:>9.1f}s")

    return {"rmse_raw": rmse_raw, "results": results}


def run_all_synthetic_scenarios(
    methods=None,
    lambda_grid=None,
    r2_threshold=None,
    save_h5ad=True,
    max_radius=None,
    use_gpu=False,
):
    """Run all S1–S10 scenarios and print a consolidated benchmark table."""
    from evaluation.synthetic.scenarios import list_scenarios

    scenario_ids = list_scenarios()
    all_results = {}
    method_names = []

    print("\n" + "=" * 80)
    print("SYNTHETIC BENCHMARK: ALL 10 SCENARIOS")
    print("=" * 80)

    for sid in scenario_ids:
        data = load_synthetic_scenario_data(sid, seed=42)
        summary = run_synthetic_comparison(
            data,
            n_genes=500,
            methods=methods,
            lambda_grid=lambda_grid,
            r2_threshold=r2_threshold,
            save_h5ad=save_h5ad,
            max_radius=max_radius,
            use_gpu=use_gpu,
        )
        all_results[sid] = summary
        if not method_names:
            method_names = list(summary["results"].keys())

    # Consolidated table
    print("\n" + "=" * 80)
    print("CONSOLIDATED RMSE REDUCTION (% vs raw)")
    print("=" * 80)
    header = f"  {'Scenario':<10}"
    for name in method_names:
        header += f" {name:>14}"
    print(header)
    print("  " + "-" * (10 + 15 * len(method_names)))

    for sid in scenario_ids:
        row = f"  {sid:<10}"
        summary = all_results[sid]
        for name in method_names:
            r = summary["results"].get(name, {})
            val = r.get("reduction", float('nan'))
            row += f" {val:>13.1f}%"
        print(row)

    # Average row
    row = f"  {'Average':<10}"
    for name in method_names:
        vals = [all_results[sid]["results"][name]["reduction"]
                for sid in scenario_ids if name in all_results[sid]["results"]]
        avg = np.mean(vals) if vals else float('nan')
        row += f" {avg:>13.1f}%"
    print(row)


def load_scgem_label_map_filtered(scgem_path, x_range=None, y_range=None):
    """Build (x,y) -> cell_id mapping from scgem, with optional spatial filter."""
    print("  Building cell label map from scgem...")
    label_map = {}
    with gzip.open(scgem_path, 'rt') as f:
        f.readline()  # header
        for i, line in enumerate(f):
            parts = line.strip().split(',')
            x, y, cell = int(parts[0]), int(parts[1]), int(parts[4])
            if x_range and not (x_range[0] <= x <= x_range[1]):
                continue
            if y_range and not (y_range[0] <= y <= y_range[1]):
                continue
            label_map[(x, y)] = cell
            if (i + 1) % 5000000 == 0:
                print(f"    Mapped {i+1:,} rows...")
    print(f"    Total: {len(label_map):,} unique DNB positions with cell labels")
    return label_map


def load_gem_chunked_filtered(filepath, x_range=None, y_range=None, nrows_per_chunk=500000):
    """Generator: yield chunks of gem TSV, filtered by spatial window."""
    import pandas as pd
    with gzip.open(filepath, 'rt') as f:
        header = f.readline().strip()
        colnames = header.split('\t')
        chunk_rows = []
        for line in f:
            parts = line.strip().split('\t')
            x, y = int(parts[0]), int(parts[1])
            if x_range and not (x_range[0] <= x <= x_range[1]):
                continue
            if y_range and not (y_range[0] <= y <= y_range[1]):
                continue
            chunk_rows.append(parts)
            if len(chunk_rows) >= nrows_per_chunk:
                yield pd.DataFrame(chunk_rows, columns=colnames)
                chunk_rows = []
        if chunk_rows:
            yield pd.DataFrame(chunk_rows, columns=colnames)


def build_dnb_matrix_from_gem_filtered(gem_path, label_map, x_range=None, y_range=None):
    """Build genes x DNBs sparse matrix from gem, filtering by spatial window.

    Returns:
        dnb_expr: [n_genes x n_dnbs] CSR sparse float64 matrix.
        dnb_coords: [n_dnbs x 2] float64.
        dnb_labels: [n_dnbs] int64, original cell ID, -1 for empty.
        gene_names: list of gene names.
        cell_ids: sorted list of original cell IDs.
    """
    print("Pass 1: scanning gene & DNB vocabulary from gem...")
    gene_set = set()
    dnb_set = set()

    for chunk in load_gem_chunked_filtered(gem_path, x_range, y_range):
        gene_set.update(chunk['geneID'].unique())
        dnb_set.update(zip(chunk['x'].astype(int), chunk['y'].astype(int)))
        if len(gene_set) % 2000 == 0:
            print(f"  Genes: {len(gene_set)}, DNBs: {len(dnb_set)}")

    genes = sorted(gene_set)
    dnb_list = sorted(dnb_set)
    gene_to_idx = {g: i for i, g in enumerate(genes)}
    dnb_to_idx = {d: i for i, d in enumerate(dnb_list)}

    n_genes = len(genes)
    n_dnbs = len(dnb_list)

    all_cells = sorted(set(label_map.values()))
    cell_to_idx = {c: i for i, c in enumerate(all_cells)}
    n_cells = len(all_cells)

    n_matched = sum(1 for d in dnb_list if d in label_map)
    n_empty = n_dnbs - n_matched
    print(f"  Final: {n_genes} genes, {n_dnbs} DNBs "
          f"({n_matched} cell, {n_empty} empty), {n_cells} cells")

    dnb_coords = np.array(dnb_list, dtype=np.float64)
    dnb_labels = np.full(n_dnbs, -1, dtype=np.int64)
    for i, d in enumerate(dnb_list):
        if d in label_map:
            dnb_labels[i] = label_map[d]

    print("Pass 2: building sparse matrix from gem...")
    dnb_expr = lil_matrix((n_genes, n_dnbs), dtype=np.float64)
    total_rows = 0

    for chunk in load_gem_chunked_filtered(gem_path, x_range, y_range):
        x_arr = chunk['x'].astype(int).values
        y_arr = chunk['y'].astype(int).values
        count_arr = chunk['MIDCounts'].astype(float).values
        gene_arr = chunk['geneID'].values

        for i in range(len(chunk)):
            g_idx = gene_to_idx[gene_arr[i]]
            d_idx = dnb_to_idx[(x_arr[i], y_arr[i])]
            dnb_expr[g_idx, d_idx] += count_arr[i]

        total_rows += len(chunk)
        if total_rows % 5000000 == 0:
            print(f"  Processed {total_rows:,} rows...")

    print(f"  Total rows: {total_rows:,}, nonzeros: {dnb_expr.nnz:,}")
    return dnb_expr.tocsr(), dnb_coords, dnb_labels, genes, all_cells


def load_axolotl_data_windowed(x_range=None, y_range=None):
    """Load Axolotl data with optional spatial window (filtered during load)."""
    data_dir = Path(__file__).resolve().parent.parent.parent / "evaluation" / "data" / "axolotl"

    print("[1/3] Loading annotations...")
    adata = ad.read_h5ad(data_dir / "Adult.h5ad")
    ann_map = dict(zip(adata.obs['cell_id'].values, adata.obs['Annotation'].values))
    sstin_set = set(adata.obs.loc[adata.obs['Annotation'] == 'sstIN', 'cell_id'].values)

    print("[2/3] Building DNB matrix...")
    t0 = time.time()
    label_map = load_scgem_label_map_filtered(
        str(data_dir / "Adult_scgem.csv.gz"), x_range, y_range
    )
    dnb_expr, dnb_coords, dnb_labels, gene_names, cell_ids = build_dnb_matrix_from_gem_filtered(
        str(data_dir / "Adult.gem.gz"), label_map, x_range, y_range
    )
    print(f"  Done in {time.time()-t0:.0f}s ({dnb_expr.shape[0]} genes, {dnb_expr.shape[1]} DNBs, {len(cell_ids)} cells)")

    return {
        'dnb_expr': dnb_expr, 'dnb_coords': dnb_coords, 'dnb_labels': dnb_labels,
        'gene_names': gene_names, 'cell_ids': cell_ids,
        'ann_map': ann_map, 'sstin_set': sstin_set,
        'x_range': x_range, 'y_range': y_range,
    }


def run_axolotl_comparison(data, n_genes=200, methods=None, lambda_grid=None, r2_threshold=None, save_h5ad=True, max_radius=None, use_gpu=False):
    """Run multi-method comparison for Axolotl."""
    if methods is None:
        methods = ['sparkle', 'soupx', 'decontx']
    methods = [m.lower().strip() for m in methods]

    dnb_expr = data['dnb_expr']
    dnb_coords = data['dnb_coords']
    dnb_labels = data['dnb_labels']
    dnb_coords_um = dnb_coords * STEREOSEQ_PITCH_UM
    gene_names = data['gene_names']
    cell_ids = np.array(data['cell_ids'])
    ann_map = data['ann_map']
    sstin_set = data['sstin_set']
    sst_gene = "AMEX60DD003175"

    n_cells = len(cell_ids)
    sstin_mask = np.array([cell_ids[i] in sstin_set for i in range(n_cells)])

    # Cell centroids & neighbors
    cc = np.zeros((n_cells, 2))
    for c in range(n_cells):
        m = dnb_labels == cell_ids[c]
        if m.sum():
            cc[c] = dnb_coords_um[m].mean(axis=0)
    neighbor_mask, other_mask = compute_neighbor_stats(cc, sstin_mask)

    # Select genes: top n_genes by total expression, ensure SST is included
    sst_idx_all = gene_names.index(sst_gene)
    gene_totals_arr = np.asarray(dnb_expr.sum(axis=1)).ravel()
    top_n = list(np.argsort(gene_totals_arr)[::-1][:n_genes])
    if sst_idx_all not in top_n:
        top_n.append(sst_idx_all)
    sst_loc_n = top_n.index(sst_idx_all)
    # For scIB: keep all genes (like MOSTA), SPARKLE corrects only top-N
    # Build sub with top_n for SPARKLE, but full data for evaluation

    # Map original cell IDs to 0-based indices for aggregation
    label_to_idx = {cid: i for i, cid in enumerate(cell_ids)}
    labels_0based = np.array([label_to_idx.get(l, -1) for l in dnb_labels])

    # Raw SST
    valid = labels_0based >= 0
    C = csr_matrix((np.ones(valid.sum()), (np.where(valid)[0], labels_0based[valid])),
                   shape=(dnb_expr.shape[1], n_cells))
    raw_cell = (dnb_expr @ C).toarray()
    sst_raw = raw_cell[sst_idx_all]

    def sst_summary(vals, label):
        si = vals[sstin_mask].mean()
        sn = vals[neighbor_mask].mean()
        so = vals[other_mask].mean()
        extra = f"s/N={si/sn:5.2f}x  N/O={sn/so:5.2f}x" if so > 0 else ""
        print(f"  {label:<30} sstIN={si:7.1f}  Nbr={sn:6.1f}  Oth={so:5.1f}  {extra}")

    print(f"\n{'='*60}")
    print("RAW DATA")
    print(f"{'='*60}")
    sst_summary(sst_raw, "Raw SST")

    results = {}
    n_method_steps = sum(1 for m in methods if m in ('sparkle', 'soupx', 'decontx'))
    step = 0

    # 1. Cell SPARKLE
    if 'sparkle' in methods:
        step += 1
        print(f"\n[{step}/{n_method_steps}] Cell SPARKLE (cell_based + penalty)...")
        t0 = time.time()
        sub_all = dnb_expr  # use all genes
        n_high = min(n_genes, sub_all.shape[0])
        if lambda_grid is None:
            lambda_grid_sp = [10, 20, 30, 50, 70, 100, 150, 200]
        else:
            lambda_grid_sp = lambda_grid
        if r2_threshold is None:
            r2_threshold_sp = 0.01
        else:
            r2_threshold_sp = r2_threshold
        if max_radius is None:
            max_radius_sp = 200.0
        else:
            max_radius_sp = max_radius
        model = SPARKLE(
            bin_size=DEFAULT_EMPTY_BIN_SIZE_UM, max_radius=max_radius_sp,
            n_high_genes=n_high, n_lambda_genes=min(100, sub_all.shape[0]),
            r2_threshold=r2_threshold_sp, lambda_grid=lambda_grid_sp,
            cell_based=True, verbose=True,
            use_gpu=use_gpu,
        )
        sp_corr, sp_diag = model.fit_transform_from_dnb(
            sub_all, dnb_coords_um, dnb_labels
        )
        sp_t = time.time() - t0
        if hasattr(sp_corr, 'toarray'):
            sp_corr = sp_corr.toarray()
        sst_summary(sp_corr[sst_idx_all], f"Cell SPARKLE ({sp_t:.0f}s)")
        results['SPARKLE'] = sp_corr[sst_idx_all]
        results['sp_full'] = sp_corr
        results['sp_var_data'] = _build_sparkle_var_data(sub_all.shape[0], sp_diag, r2_threshold_sp)

    # 2. SoupX
    soupx_error = None
    if 'soupx' in methods:
        step += 1
        print(f"\n[{step}/{n_method_steps}] SoupX (top {n_genes} genes)...")
        t0 = time.time()
        try:
            sx_corr, sx_rho = run_soupx(dnb_expr[top_n, :], labels_0based, verbose=False)
        except Exception as e:
            # No heuristic substitution: record the failure explicitly as NaN.
            sx_t = time.time() - t0
            soupx_error = str(e)
            print(f"  SoupX FAILED ({e}); recording NaN values")
            results['SoupX'] = np.full(n_cells, np.nan)
        else:
            sx_t = time.time() - t0
            # labels_0based maps DNB→0-based cell index; SoupX returns corrected for these cells
            sx_aligned = np.full(n_cells, np.nan)
            for c in range(n_cells):
                if c < sx_corr.shape[1]:
                    sx_aligned[c] = sx_corr[sst_loc_n, c]
            valid_sx = ~np.isnan(sx_aligned)
            if valid_sx.sum() > 0:
                si = np.nanmean(sx_aligned[sstin_mask])
                sn = np.nanmean(sx_aligned[neighbor_mask])
                so = np.nanmean(sx_aligned[other_mask])
                extra = f"s/N={si/sn:5.2f}x  N/O={sn/so:5.2f}x" if so > 0 else ""
                print(f"  SoupX ({sx_t:.0f}s, rho={sx_rho:.4f})  sstIN={si:7.1f}  Nbr={sn:6.1f}  Oth={so:5.1f}  {extra}")
                results['SoupX'] = sx_aligned
                results['sx_full'] = sx_corr

    # 3. DecontX
    if 'decontx' in methods:
        step += 1
        print(f"\n[{step}/{n_method_steps}] DecontX...")
        t0 = time.time()
        dx_corr, dx_diag = run_decontx_method({
            "dnb_expr": dnb_expr,
            "dnb_labels": dnb_labels,
            "gene_names": list(gene_names),
            "cell_ids": list(cell_ids),
        }, verbose=True)
        dx_t = time.time() - t0
        if dx_corr is not None:
            dx_sst = dx_corr[sst_idx_all]
            sst_summary(dx_sst, f"DecontX ({dx_t:.0f}s, contam={dx_diag.get('contamination', 0):.3f})")
            results['DecontX'] = dx_sst
            results['dx_full'] = dx_corr

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Method':<18} {'sstIN':>7} {'Nbr':>7} {'Oth':>7} {'s/N':>7} {'N/O':>7} {'Retain':>7} {'Remove':>7}")
    print(f"  {'-'*18} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
    r_si = sst_raw[sstin_mask].mean()
    r_sn = sst_raw[neighbor_mask].mean()
    r_so = sst_raw[other_mask].mean()
    r_no = r_sn / r_so if r_so > 0 else float('nan')
    print(f"  {'Raw':<18} {r_si:7.1f} {r_sn:7.1f} {r_so:7.1f} {r_si/r_sn:7.2f}x {r_no:7.2f}x {'-':>7} {'-':>7}")

    for name, vals in results.items():
        if name.endswith('_full') or name == 'sp_var_data':
            continue
        si = vals[sstin_mask].mean()
        sn = vals[neighbor_mask].mean()
        so = vals[other_mask].mean()
        no_ratio = sn / so if so > 0 else float('nan')
        print(f"  {name:<18} {si:7.1f} {sn:7.1f} {so:7.1f} {si/sn:7.2f}x {no_ratio:7.2f}x "
              f"{si/r_si*100:6.1f}% {(1-sn/r_sn)*100:6.1f}%")

    # Save cell-based h5ad and metrics
    print(f"\n  {'='*60}")
    print(f"  Saving results")
    print(f"  {'='*60}")
    reports_root = _reports_root()
    x_range = data.get('x_range')
    y_range = data.get('y_range')
    if x_range is not None and y_range is not None:
        tag = f"axolotl_x{x_range[0]}-{x_range[1]}_y{y_range[0]}-{y_range[1]}"
    else:
        tag = "axolotl_full"

    sub_gene_names = [gene_names[i] for i in top_n]
    save_result_h5ad(raw_cell, gene_names, cell_ids, ann_map,
                     reports_root / "h5ad" / f"{tag}_raw.h5ad", "RAW",
                     save_h5ad=save_h5ad)
    metrics = {
        "dataset": tag,
        "n_cells": int(raw_cell.shape[1]),
        "n_genes": int(raw_cell.shape[0]),
        "raw": {
            "sstIN": float(sst_raw[sstin_mask].mean()) if sstin_mask.any() else None,
            "sstNbr": float(sst_raw[neighbor_mask].mean()) if neighbor_mask.any() else None,
            "sstOth": float(sst_raw[other_mask].mean()) if other_mask.any() else None,
        },
        "methods": {},
    }
    full_map = {'SPARKLE': 'sp_full', 'SoupX': 'sx_full', 'DecontX': 'dx_full'}
    for method_name, full_key in full_map.items():
        full = results.get(full_key)
        if full is None:
            continue
        if method_name in ('SPARKLE', 'DecontX'):
            gnames_save = gene_names
            sst_vals = full[sst_idx_all]
        else:
            gnames_save = sub_gene_names
            sst_vals = full[sst_loc_n]
        var_data = results.get('sp_var_data') if method_name == 'SPARKLE' else None
        save_result_h5ad(full, gnames_save, cell_ids, ann_map,
                         reports_root / "h5ad" / f"{tag}_{method_name}.h5ad",
                         method_name,
                         save_h5ad=save_h5ad,
                         var_data=var_data)
        metrics["methods"][method_name] = {
            "sstIN": float(sst_vals[sstin_mask].mean()) if sstin_mask.any() else None,
            "sstNbr": float(sst_vals[neighbor_mask].mean()) if neighbor_mask.any() else None,
            "sstOth": float(sst_vals[other_mask].mean()) if other_mask.any() else None,
        }
        if method_name == "SPARKLE":
            metrics["methods"][method_name].update({
                "runtime": float(sp_t),
                "lambda": float(sp_diag["lambda_estimated"]),
                "bin_size_um": float(sp_diag["bin_size"]),
                "n_empty_bins": int(sp_diag["n_empty_bins"]),
                "empty_bin_dnb_count_max": int(
                    sp_diag["empty_bin_dnb_count_max"]
                ),
            })
    if soupx_error is not None:
        # SoupX failed: no corrected matrix, but the failure must be visible
        # in the metrics as NaN values instead of silent omission.
        metrics["methods"]["SoupX"] = {
            "sstIN": None,
            "sstNbr": None,
            "sstOth": None,
            "error": soupx_error,
        }
    save_metrics_json(metrics, reports_root / "metrics" / f"{tag}_metrics.json")


def run_mosta_comparison(data, sub, n_genes=200, methods=None, n_high_genes=None, lambda_grid=None, r2_threshold=None, save_h5ad=True, max_radius=None, use_gpu=False):
    """Run 3-way comparison for MOSTA with cortical layer evaluation."""
    if methods is None:
        methods = ['sparkle', 'soupx', 'decontx']
    methods = [m.lower().strip() for m in methods]
    if n_high_genes is None:
        n_high_genes = min(n_genes, 500)

    print(f"\n{'='*60}")
    print("RUNNING METHODS")
    print(f"{'='*60}")

    results = {}

    if 'sparkle' in methods:
        corrected, diag = run_sparkle_method(
            sub,
            coordinate_scale_to_um=STEREOSEQ_PITCH_UM,
            n_high_genes=n_high_genes,
            lambda_grid=lambda_grid,
            r2_threshold=r2_threshold,
            max_radius=max_radius,
            use_gpu=use_gpu,
        )
        if corrected is not None:
            results['SPARKLE'] = {'corrected': corrected, 'diag': diag}

    if 'soupx' in methods:
        corrected, diag = run_soupx_method(sub)
        # Record even on failure (corrected=None) so the failure surfaces as
        # NaN/error in summaries and metrics instead of vanishing silently.
        results['SoupX'] = {'corrected': corrected, 'diag': diag}

    if 'decontx' in methods:
        corrected, diag = run_decontx_method(sub)
        if corrected is not None:
            results['DecontX'] = {'corrected': corrected, 'diag': diag}

    # Evaluation
    n_cells = len(data.get('cell_ids', []))
    raw = compute_cell_expr(sub['dnb_expr'], sub['dnb_labels'], n_cells)
    eval_ret = evaluate_mosta(results, data, sub, sub['gene_names'], raw=raw)
    if eval_ret is None or eval_ret[0] is None:
        raw_summary = {}
    else:
        raw, raw_summary = eval_ret

    # Save cell-based h5ad and metrics
    print(f"\n  {'='*60}")
    print(f"  Saving results")
    print(f"  {'='*60}")
    reports_root = _reports_root()
    x_range = data.get('x_range')
    y_range = data.get('y_range')
    if x_range is not None and y_range is not None:
        tag = f"mosta_x{x_range[0]}-{x_range[1]}_y{y_range[0]}-{y_range[1]}"
    else:
        tag = "mosta_full"
    ann_map = data.get('ann_map', {})
    cell_ids = data.get('cell_ids', np.arange(raw.shape[1]))

    save_result_h5ad(raw, sub['gene_names'], cell_ids, ann_map,
                     reports_root / "h5ad" / f"{tag}_raw.h5ad", "RAW",
                     save_h5ad=save_h5ad)
    metrics = {
        "dataset": tag,
        "n_cells": int(raw.shape[1]),
        "n_genes": int(raw.shape[0]),
        "raw": {},
        "methods": {},
    }
    for method_name, r in results.items():
        corrected = r.get('corrected')
        if corrected is not None:
            var_data = r.get('diag', {}).get('var_data') if method_name == 'SPARKLE' else None
            save_result_h5ad(corrected, sub['gene_names'], cell_ids, ann_map,
                             reports_root / "h5ad" / f"{tag}_{method_name}.h5ad",
                             method_name,
                             save_h5ad=save_h5ad,
                             var_data=var_data)
            metrics["methods"][method_name] = {
                "runtime": r['diag'].get('runtime', 0),
                "de_genes": r.get('mosta_de_genes'),
                "layer_spearman": r.get('mosta_layer_spearman'),
            }
        else:
            # Method failed (e.g. SoupX): keep the failure visible as NaN.
            metrics["methods"][method_name] = {
                "runtime": r.get('diag', {}).get('runtime', 0),
                "de_genes": None,
                "layer_spearman": None,
                "error": r.get('diag', {}).get('error'),
            }
    save_metrics_json(metrics, reports_root / "metrics" / f"{tag}_metrics.json")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Method':<16} {'Runtime':>8} {'DE':>8} {'Sprmn':>8}")
    print(f"  {'─'*16} {'─'*8} {'─'*8} {'─'*8}")
    # RAW row
    raw_de = raw_summary.get('mosta_de_genes', raw_summary.get('de_genes', '─'))
    raw_s = raw_summary.get('mosta_layer_spearman', raw_summary.get('layer_spearman', '─'))
    raw_s_str = f"{raw_s:.4f}" if isinstance(raw_s, float) and not np.isnan(raw_s) else str(raw_s)
    print(f"  {'RAW':<16} {'─':>8} {str(raw_de):>8} {raw_s_str:>8}")
    for method_name, r in results.items():
        d = r['diag']
        runtime = d.get('runtime', 0)
        de = r.get('mosta_de_genes', 'N/A')
        s = r.get('mosta_layer_spearman', r.get('mosta_between_corr', 'N/A'))
        s_str = f"{s:.4f}" if isinstance(s, float) and not np.isnan(s) else str(s)
        print(f"  {method_name:<16} {runtime:7.1f}s  {str(de):>8} {s_str:>8}")


def run_mousebrain_comparison(data, sub, n_genes=200, methods=None, n_high_genes=None, lambda_grid=None, r2_threshold=None, save_h5ad=True, max_radius=None, use_gpu=False):
    """Run comparison for MouseBrain (T304), using cell_group annotations."""
    if methods is None:
        methods = ['sparkle', 'soupx', 'decontx']
    methods = [m.lower().strip() for m in methods]
    if n_high_genes is None:
        n_high_genes = min(n_genes, 500)

    print(f"\n{'='*60}")
    print("RUNNING METHODS")
    print(f"{'='*60}")

    results = {}

    if 'sparkle' in methods:
        corrected, diag = run_sparkle_method(
            sub,
            coordinate_scale_to_um=STEREOSEQ_PITCH_UM,
            n_high_genes=n_high_genes,
            lambda_grid=lambda_grid,
            r2_threshold=r2_threshold,
            max_radius=max_radius,
            use_gpu=use_gpu,
        )
        if corrected is not None:
            results['SPARKLE'] = {'corrected': corrected, 'diag': diag}

    if 'soupx' in methods:
        corrected, diag = run_soupx_method(sub)
        # Record even on failure (corrected=None) so the failure surfaces as
        # NaN/error in summaries and metrics instead of vanishing silently.
        results['SoupX'] = {'corrected': corrected, 'diag': diag}

    if 'decontx' in methods:
        corrected, diag = run_decontx_method(sub)
        if corrected is not None:
            results['DecontX'] = {'corrected': corrected, 'diag': diag}

    # Evaluation
    n_cells = len(data.get('cell_ids', []))
    raw = compute_cell_expr(sub['dnb_expr'], sub['dnb_labels'], n_cells)
    eval_ret = evaluate_mousebrain(results, data, sub, sub['gene_names'], raw=raw)
    if eval_ret is None or eval_ret[0] is None:
        raw_summary = {}
    else:
        raw, raw_summary = eval_ret

    # Save cell-based h5ad and metrics
    print(f"\n  {'='*60}")
    print(f"  Saving results")
    print(f"  {'='*60}")
    reports_root = _reports_root()
    x_range = data.get('x_range')
    y_range = data.get('y_range')
    if x_range is not None and y_range is not None:
        tag = f"mousebrain_x{x_range[0]}-{x_range[1]}_y{y_range[0]}-{y_range[1]}"
    else:
        tag = "mousebrain_full"
    ann_map = data.get('ann_map', {})
    cell_ids = data.get('cell_ids', np.arange(raw.shape[1]))

    save_result_h5ad(raw, sub['gene_names'], cell_ids, ann_map,
                     reports_root / "h5ad" / f"{tag}_raw.h5ad", "RAW",
                     save_h5ad=save_h5ad)
    metrics = {
        "dataset": tag,
        "n_cells": int(raw.shape[1]),
        "n_genes": int(raw.shape[0]),
        "raw": {},
        "methods": {},
    }
    for method_name, r in results.items():
        corrected = r.get('corrected')
        if corrected is not None:
            var_data = r.get('diag', {}).get('var_data') if method_name == 'SPARKLE' else None
            save_result_h5ad(corrected, sub['gene_names'], cell_ids, ann_map,
                             reports_root / "h5ad" / f"{tag}_{method_name}.h5ad",
                             method_name,
                             save_h5ad=save_h5ad,
                             var_data=var_data)
            metrics["methods"][method_name] = {
                "runtime": r['diag'].get('runtime', 0),
                "de_genes": r.get('mousebrain_de_genes'),
                "class_spearman": r.get('mousebrain_class_spearman'),
            }
            if method_name == "SPARKLE":
                metrics["methods"][method_name].update({
                    "lambda": float(r["diag"]["lambda_estimated"]),
                    "bin_size_um": float(r["diag"]["bin_size"]),
                    "n_empty_bins": int(r["diag"]["n_empty_bins"]),
                    "empty_bin_dnb_count_max": int(
                        r["diag"]["empty_bin_dnb_count_max"]
                    ),
                })
        else:
            # Method failed (e.g. SoupX): keep the failure visible as NaN.
            metrics["methods"][method_name] = {
                "runtime": r.get('diag', {}).get('runtime', 0),
                "de_genes": None,
                "class_spearman": None,
                "error": r.get('diag', {}).get('error'),
            }
    save_metrics_json(metrics, reports_root / "metrics" / f"{tag}_metrics.json")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Method':<16} {'Runtime':>8} {'DE':>8} {'Sprmn':>8}")
    print(f"  {'─'*16} {'─'*8} {'─'*8} {'─'*8}")
    raw_de = raw_summary.get('mousebrain_de_genes', raw_summary.get('de_genes', '─'))
    raw_s = raw_summary.get('mousebrain_class_spearman', raw_summary.get('class_spearman', '─'))
    raw_s_str = f"{raw_s:.4f}" if isinstance(raw_s, float) and not np.isnan(raw_s) else str(raw_s)
    print(f"  {'RAW':<16} {'─':>8} {str(raw_de):>8} {raw_s_str:>8}")
    for method_name, r in results.items():
        d = r['diag']
        runtime = d.get('runtime', 0)
        de = r.get('mousebrain_de_genes', 'N/A')
        s = r.get('mousebrain_class_spearman', 'N/A')
        s_str = f"{s:.4f}" if isinstance(s, float) and not np.isnan(s) else str(s)
        print(f"  {method_name:<16} {runtime:7.1f}s  {str(de):>8} {s_str:>8}")


def load_mosta_data(x_range=None, y_range=None):
    """加载 MOSTA 成年鼠脑 Stereo-seq 数据。

    数据来源（三个文件）：
      Mouse_brain_Adult_GEM_bin1.tsv.gz    — 80M 行全量 GEM（含空 DNB，无 label 列）
      Mouse_brain_Adult_GEM_CellBin.tsv.gz — 57M 行细胞 mask GEM（含 label 列）
      Mouse_brain_cell_bin.h5ad            — 细胞注释（皮层层次 EX L2/3, L4, L5/6, L6 等）

    处理流程（两遍扫描）：
      Pass 1 — CellBin → label_map:
        遍历 CellBin.tsv.gz，提取每行的 (x, y) → cell_label，
        构建字典 label_map[(x, y)] = cell_label。
        这一步只保留 label 列有用的坐标映射，舍弃基因表达信息。

      Pass 2 — bin1 GEM → dnb_records:
        遍历 bin1.tsv.gz（全量 GEM），对每行 (gene, x, y, count)：
          - 如果 (x, y) 在 label_map 中 → cell_label = label_map[(x, y)]
          - 否则 → cell_label = -1（这是一个空 DNB，落在所有细胞 mask 之外）
        这个逻辑和 Axolotl 完全一致：DNB 在细胞 mask 内则分配 cell label，
        否则标记为空的 ambient probe。

    为什么需要两个文件？
      CellBin 只有被分配到某个细胞的 DNB（57M 行），缺少细胞间隙的空 DNB。
      bin1 包含所有 DNB（80M 行），但没有 label 列。
      两者结合才能获得 Axolotl 那样的"cell DNB + 空 DNB"完整数据。

    Args:
        x_range: (x_min, x_max) 空间窗口过滤，默认 None（全脑）。
        y_range: (y_min, y_max) 空间窗口过滤，默认 None（全脑）。

    Returns:
        dict，字段见本文件开头统一说明。
    """
    import anndata as ad

    data_dir = Path(__file__).resolve().parent.parent.parent / "evaluation" / "data" / "mosta"
    bin1_path = data_dir / "Mouse_brain_Adult_GEM_bin1.tsv.gz"
    cellbin_path = data_dir / "Mouse_brain_Adult_GEM_CellBin.tsv.gz"
    h5ad_path = data_dir / "Mouse_brain_cell_bin.h5ad"

    if not bin1_path.exists():
        raise FileNotFoundError(f"{bin1_path} not found. Download the full GEM file first.")

    # ── 加载细胞注释 ─────────────────────────────────────────────
    print("Loading MOSTA annotations...")
    adata = ad.read_h5ad(h5ad_path)
    ann_map = {}
    for i, idx in enumerate(adata.obs.index):
        cell_num = int(idx.split('_')[1])    # h5ad index 格式: Cell_1, Cell_16, ...
        ann_map[cell_num] = adata.obs['annotation'].values[i]

    # ═══════════════════════════════════════════════════════════════
    # Pass 1: 从 CellBin 构建 (x,y) → cell_label 映射表
    # ═══════════════════════════════════════════════════════════════
    # CellBin 格式: geneID\tx\ty\tUMICount\tlabel
    # 我们只需要 (x, y) → label 的映射，基因信息在此阶段忽略。
    print("Building cell label map from CellBin...")
    label_map = {}  # key: (x, y) tuple of ints, value: cell_label (int)
    with gzip.open(cellbin_path, 'rt') as f:
        next(f)  # 跳过表头
        for i, line in enumerate(f):
            parts = line.strip().split('\t')
            x, y = int(parts[1]), int(parts[2])
            label = int(float(parts[4]))
            # 空间窗口过滤（如果指定）
            if x_range and not (x_range[0] <= x <= x_range[1]):
                continue
            if y_range and not (y_range[0] <= y <= y_range[1]):
                continue
            label_map[(x, y)] = label
            if i % 10000000 == 0 and i > 0:
                print(f"  {i//1000000}M rows, {len(label_map)} labeled DNBs...")
    print(f"  Label map: {len(label_map)} DNBs with cell assignments")

    # ═══════════════════════════════════════════════════════════════
    # Pass 2: 加载 bin1 全量 GEM，匹配 label_map
    # ═══════════════════════════════════════════════════════════════
    # bin1 格式: geneID\tx\ty\tMIDCounts
    # MIDCounts 是 bin1 的分子计数（可能 >1），这里暂存为 1 个 UMI（稀疏矩阵存 1）
    print("Loading bin1 GEM (all DNBs)...")
    gene_to_idx = {}          # gene_name → gene_index (0..n_genes-1)
    dnb_records = []          # list of (gene_idx, x, y, cell_label_or_minus1)

    with gzip.open(bin1_path, 'rt') as f:
        next(f)  # 跳过表头
        for i, line in enumerate(f):
            parts = line.strip().split('\t')
            gene, x, y = parts[0], int(parts[1]), int(parts[2])
            # MIDCounts = int(parts[3])  # bin1 的分子计数，暂不使用

            # 空间窗口过滤
            if x_range and not (x_range[0] <= x <= x_range[1]):
                continue
            if y_range and not (y_range[0] <= y <= y_range[1]):
                continue

            if gene not in gene_to_idx:
                gene_to_idx[gene] = len(gene_to_idx)

            # 核心逻辑：DNB 是否在细胞 mask 内？
            #   label_map 中有 (x,y) → 该 DNB 属于某个细胞
            #   label_map 中无 (x,y) → 该 DNB 是空 DNB（label = -1）
            cell_label = label_map.get((x, y), -1)
            dnb_records.append((gene_to_idx[gene], x, y, cell_label))

            if i % 10000000 == 0 and i > 0:
                n_empty = sum(1 for r in dnb_records if r[3] < 0)
                print(f"  {i//1000000}M rows, {len(gene_to_idx)} genes, "
                      f"{len(dnb_records)} DNBs ({n_empty} empty)...")

    n_genes = len(gene_to_idx)
    n_dnbs = len(dnb_records)
    n_empty = sum(1 for r in dnb_records if r[3] < 0)
    print(f"  Total: {n_genes} genes, {n_dnbs} DNBs ({n_empty} empty, {n_dnbs - n_empty} cell)")

    # ── 构建稀疏矩阵 [genes × DNBs] ──────────────────────────
    # 每个 DNB 对应 1 个 UMI，用 CSR 格式存储
    gene_indices = np.array([r[0] for r in dnb_records], dtype=np.int32)
    dnb_indices = np.arange(n_dnbs, dtype=np.int32)
    dnb_expr = csr_matrix((np.ones(n_dnbs, dtype=np.float64),
                           (gene_indices, dnb_indices)), shape=(n_genes, n_dnbs))
    dnb_coords = np.array([[r[1], r[2]] for r in dnb_records], dtype=np.float64)

    # ── 构建连续 cell label ─────────────────────────────────
    # 原始 label 是不连续的大整数（如 27298），映射到 0..n_cells-1
    cell_labels_in_data = sorted(set(r[3] for r in dnb_records if r[3] >= 0))
    label_to_idx = {lbl: i for i, lbl in enumerate(cell_labels_in_data)}
    dnb_labels = np.array([label_to_idx.get(r[3], -1) for r in dnb_records], dtype=np.int32)

    gene_names = [None] * n_genes
    for g, idx in gene_to_idx.items():
        gene_names[idx] = g
    gene_names = np.array(gene_names)
    cell_ids = np.array(cell_labels_in_data)

    return {
        'dnb_expr': dnb_expr, 'dnb_coords': dnb_coords, 'dnb_labels': dnb_labels,
        'gene_names': gene_names, 'cell_ids': cell_ids, 'ann_map': ann_map, 'adata': adata,
        'x_range': x_range, 'y_range': y_range,
    }


def load_mousebrain_data(x_range=None, y_range=None, annotation_level='cell_group'):
    """加载新加入的 Mouse Brain Stereo-seq (T304) 数据。

    文件：
      total_gene_T304_mouse_f001_2D_mouse1-20230119.txt.gz — DNB 级 GEM
      stereoseq.celltypeTransfer.2mice.all.tsv.gz          — cell_id → cluster/subclass/class 映射
      mouseBrain.snRNAseq.308ClustersAnnotation.20230607.tsv — cluster 注释（可选参考）

    处理流程：
      1. 从 celltypeTransfer 中过滤 section_id == 'T304'，构建 cell_id → annotation 的 ann_map
         annotation_level 可以是 cell_class / cell_subclass / cell_group。
         cell_group 从 cell_cluster 列取最后一个下划线前的 prefix，并把下划线替换为 dash。
      2. 流式读取 GEM，按 x_range/y_range 过滤
      3. cell_label == 0 或不在 ann_map 中的 DNB 标记为 -1（empty/背景）
      4. 对同一个 (gene, DNB) 的多个记录按 umi_count 求和
      5. 构建 [genes × DNBs] CSR 稀疏矩阵，并把原始 cell_label 映射为 0-based 索引
    """
    from collections import defaultdict

    data_dir = Path(__file__).resolve().parent.parent / "data" / "mousebrain"
    gem_path = data_dir / "total_gene_T304_mouse_f001_2D_mouse1-20230119.txt.gz"
    transfer_path = data_dir / "stereoseq.celltypeTransfer.2mice.all.tsv.gz"

    if not gem_path.exists():
        raise FileNotFoundError(f"{gem_path} not found.")
    if not transfer_path.exists():
        raise FileNotFoundError(f"{transfer_path} not found.")

    # ── 加载 T304 的细胞注释 ─────────────────────────────────────
    print(f"Loading MouseBrain cell-type transfer (section T304, level={annotation_level})...")
    ann_map = {}
    valid_cell_ids = set()
    with gzip.open(transfer_path, 'rt') as f:
        header = f.readline().strip().split('\t')
        try:
            idx_id = header.index('cell_id')
            idx_section = header.index('section_id')
            if annotation_level == 'cell_group':
                idx_cluster = header.index('cell_cluster')
            else:
                idx_ann = header.index(annotation_level)
        except ValueError:
            raise ValueError(f"Transfer file missing required columns; header={header}")
        n = 0
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if parts[idx_section] != 'T304':
                continue
            cid = int(parts[idx_id])
            if annotation_level == 'cell_group':
                # cell_cluster looks like "CA1_N_GLU_78"; cell_group is everything before the last underscore.
                # Replace underscores with dashes to match snRNA reference naming.
                cluster = parts[idx_cluster]
                prefix = cluster.rsplit('_', 1)[0] if '_' in cluster else cluster
                ann = prefix.replace('_', '-')
            else:
                ann = parts[idx_ann]
            ann_map[cid] = ann
            valid_cell_ids.add(cid)
            n += 1
    print(f"  {n} cells with annotations in section T304")
    if n > 0:
        unique_anns = sorted(set(ann_map.values()))
        print(f"  {len(unique_anns)} unique {annotation_level} annotations: {unique_anns}")

    # ── 流式读取 GEM 并构建稀疏矩阵 ─────────────────────────────
    print("Loading MouseBrain GEM (section T304)...")
    gene_to_idx = {}
    dnb_to_idx = {}
    dnb_coords_list = []
    dnb_orig_label = {}
    counts = defaultdict(float)

    n_rows = 0
    with gzip.open(gem_path, 'rt') as f:
        f.readline()  # header: gene x y umi_count cell_label gene_area rx ry
        for line in f:
            parts = line.rstrip('\n').split('\t')
            gene = parts[0]
            x = int(parts[1])
            y = int(parts[2])
            if x_range and not (x_range[0] <= x <= x_range[1]):
                continue
            if y_range and not (y_range[0] <= y <= y_range[1]):
                continue

            umi_count = float(parts[3])
            cell_label = int(parts[4])

            if gene not in gene_to_idx:
                gene_to_idx[gene] = len(gene_to_idx)
            gidx = gene_to_idx[gene]

            d = (x, y)
            didx = dnb_to_idx.get(d)
            if didx is None:
                didx = len(dnb_to_idx)
                dnb_to_idx[d] = didx
                dnb_coords_list.append((x, y))
                if cell_label != 0 and cell_label in valid_cell_ids:
                    dnb_orig_label[didx] = cell_label
                else:
                    dnb_orig_label[didx] = -1
            else:
                if dnb_orig_label[didx] == -1 and cell_label != 0 and cell_label in valid_cell_ids:
                    dnb_orig_label[didx] = cell_label

            counts[(gidx, didx)] += umi_count
            n_rows += 1
            if n_rows % 5000000 == 0:
                print(f"  {n_rows/1e6:.1f}M rows, {len(gene_to_idx)} genes, {len(dnb_to_idx)} DNBs...")

    n_genes = len(gene_to_idx)
    n_dnbs = len(dnb_to_idx)
    if n_dnbs == 0:
        raise ValueError("No DNBs found in the specified spatial window.")

    gene_names = [None] * n_genes
    for g, idx in gene_to_idx.items():
        gene_names[idx] = g
    gene_names = np.array(gene_names)
    dnb_coords = np.array(dnb_coords_list, dtype=np.float64)

    # Remap original cell labels to 0..n_cells-1
    orig_labels = sorted({lbl for lbl in dnb_orig_label.values() if lbl >= 0})
    label_to_idx = {lbl: i for i, lbl in enumerate(orig_labels)}
    dnb_labels = np.array([label_to_idx.get(dnb_orig_label[i], -1) for i in range(n_dnbs)], dtype=np.int32)
    cell_ids = np.array(orig_labels)

    n_counts = len(counts)
    rows = np.fromiter((k[0] for k in counts.keys()), dtype=np.int32, count=n_counts)
    cols = np.fromiter((k[1] for k in counts.keys()), dtype=np.int32, count=n_counts)
    data = np.fromiter(counts.values(), dtype=np.float64, count=n_counts)
    dnb_expr = csr_matrix((data, (rows, cols)), shape=(n_genes, n_dnbs))

    n_empty = int((dnb_labels < 0).sum())
    print(f"  Loaded: {n_genes} genes, {n_dnbs} DNBs ({n_dnbs - n_empty} cell + {n_empty} empty), {len(cell_ids)} cells")
    return {
        'dnb_expr': dnb_expr, 'dnb_coords': dnb_coords, 'dnb_labels': dnb_labels,
        'gene_names': gene_names, 'cell_ids': cell_ids, 'ann_map': ann_map,
        'x_range': x_range, 'y_range': y_range,
    }


def subsample_data(data, n_genes, cut_genes=True):
    """对数据进行统一的基因子采样（DNB 全保留）。

    Args:
        data: dict，来自 load_* 函数的返回值。
        n_genes: 保留的 top 基因数（按总表达量排序）。
        cut_genes: 是否裁切基因矩阵。False 时返回全部基因（仅计算 top-N 排序）。

    Returns:
        dict，包含子采样后的 dnb_expr, dnb_coords, dnb_labels 等字段。
    """
    dnb_expr = data['dnb_expr']
    dnb_coords = data['dnb_coords']
    dnb_labels = data['dnb_labels']

    n_genes_total, n_dnbs_total = dnb_expr.shape

    # ── 选择 top N 基因（按总表达量降序）───────────────────
    gene_totals = np.asarray(dnb_expr.sum(axis=1)).ravel()
    top_genes = np.argsort(gene_totals)[-min(n_genes, n_genes_total):]
    top_genes = np.sort(top_genes)  # 保持原始顺序

    if cut_genes:
        sub_expr = dnb_expr[top_genes, :]
        sub_gene_names = np.array(data['gene_names'])[top_genes]
    else:
        # Keep all genes, SPARKLE will correct only top-N
        sub_expr = dnb_expr
        sub_gene_names = data['gene_names']

    n_empty_sub = int((dnb_labels < 0).sum())
    n_cell_sub = int((dnb_labels >= 0).sum())

    if n_empty_sub == 0:
        print(f"  WARNING: 0 empty DNBs in data. SPARKLE requires empty regions as probes.")
        print(f"           Use a larger spatial window or check data preprocessing.")

    if cut_genes:
        print(f"  Subsampled: {len(top_genes)}/{n_genes_total} genes, "
              f"{n_cell_sub} cell + {n_empty_sub} empty DNBs (total: {n_dnbs_total})")
    else:
        print(f"  Gene selection: top {len(top_genes)}/{n_genes_total} for SPARKLE correction, "
              f"all {n_genes_total} genes kept for evaluation")

    result = {
        'dnb_expr': sub_expr, 'dnb_coords': dnb_coords, 'dnb_labels': dnb_labels,
        'gene_names': sub_gene_names, 'cell_ids': data.get('cell_ids'),
    }
    # 保留原始数据中的可选字段（地面真值、注释等）
    for k in ['ann_map', 'adata', 'true_expr', 'gene_is_high', 'true_alpha',
              'true_lambda', 'params']:
        if k in data:
            result[k] = data[k]

    return result


def _build_sparkle_var_data(n_total_genes: int, diag: dict, r2_threshold: float) -> dict:
    """Build per-gene R² annotation arrays for h5ad adata.var.

    Returns a dict with:
      - sparkle_selected: bool, selected as high-expression gene.
      - sparkle_corrected: bool, selected AND passed R² threshold.
      - sparkle_r2: float, R² score (NaN for genes not selected).
    """
    gene_indices = diag.get("gene_indices")
    r2_scores = diag.get("r2_scores")
    r2_thresh = diag.get("r2_threshold", r2_threshold)
    sparkle_r2 = np.full(n_total_genes, np.nan, dtype=np.float64)
    sparkle_selected = np.zeros(n_total_genes, dtype=bool)
    sparkle_corrected = np.zeros(n_total_genes, dtype=bool)
    if gene_indices is not None and r2_scores is not None:
        gene_indices = np.asarray(gene_indices, dtype=int)
        r2_scores = np.asarray(r2_scores)
        sparkle_r2[gene_indices] = r2_scores
        sparkle_selected[gene_indices] = True
        sparkle_corrected[gene_indices] = r2_scores >= r2_thresh
    return {
        "sparkle_selected": sparkle_selected,
        "sparkle_corrected": sparkle_corrected,
        "sparkle_r2": sparkle_r2,
    }


@profile
def run_sparkle_method(
    sub,
    verbose=True,
    bin_size_um=DEFAULT_EMPTY_BIN_SIZE_UM,
    coordinate_scale_to_um=1.0,
    n_high_genes=500,
    lambda_grid=None,
    r2_threshold=None,
    max_radius=None,
    use_gpu=False,
    gpu_dtype="float64",
    gpu_gene_batch_size=None,
):
    """运行 SPARKLE（本方法）。

    核心流程：
      1. 检查空 DNB 是否存在 → 无则报错退出（不降级到 bin-level）
      2. 提取细胞 DNB → cell areas
      3. 空 DNB 分箱 → empty bins
      4. λ 网格搜索 → 最优空间扩散距离
      5. 基因特异性 α 估计（加权 OLS，R² 阈值过滤）
      6. cell-level ambient 校正（含 self-confidence penalty）

    参数：
      bin_size_um=25.0     — 空 bin 边长（µm）
      coordinate_scale_to_um
                           — 输入坐标乘以该值后转换为 µm；已是 µm 时用 1
      max_radius=300       — 空间邻域搜索半径（µm）
      r2_threshold=0.01    — α 估计的 R² 阈值
      lambda_grid          — λ 候选值列表（µm）
      cell_based=True      — 使用 cell-based pipeline

    输出 diagnostics 包含：lambda, n_genes_corrected, runtime 等。
    """
    if lambda_grid is None:
        lambda_grid = [10, 20, 30, 50, 70, 100, 150, 200, 300]
    if r2_threshold is None:
        r2_threshold = 0.01
    if max_radius is None:
        max_radius = 300.0
    if not np.isfinite(coordinate_scale_to_um) or coordinate_scale_to_um <= 0:
        raise ValueError("coordinate_scale_to_um must be positive and finite")
    if not np.isfinite(bin_size_um) or bin_size_um <= 0:
        raise ValueError("bin_size_um must be positive and finite")
    print(
        f"\n  SPARKLE (cell_based + penalty, bin={bin_size_um:g}µm, "
        f"coordinate_scale={coordinate_scale_to_um:g}µm/unit, "
        f"n_high={n_high_genes})..."
    )
    dnb_expr = sub['dnb_expr']
    dnb_coords_um = np.asarray(sub['dnb_coords'], dtype=np.float64)
    if coordinate_scale_to_um != 1.0:
        dnb_coords_um = dnb_coords_um * coordinate_scale_to_um
    dnb_labels = sub['dnb_labels']

    # 空 DNB 检查：SPARKLE 必须有空 DNB 作为 ambient probe
    n_empty = int((dnb_labels < 0).sum())
    if n_empty == 0:
        print(f"    ERROR: No empty DNBs available. SPARKLE requires empty regions as probes.")
        return None, {'error': 'no_empty_dnbs', 'runtime': 0}

    model = SPARKLE(
        bin_size=bin_size_um, distance_metric="exponential", max_radius=max_radius,
        n_high_genes=min(n_high_genes, dnb_expr.shape[0]),
        n_lambda_genes=min(100, dnb_expr.shape[0]),
        r2_threshold=r2_threshold,
        lambda_grid=lambda_grid,
        cell_based=True, verbose=verbose,
        use_gpu=use_gpu,
        gpu_dtype=gpu_dtype,
        gpu_gene_batch_size=gpu_gene_batch_size,
    )
    t0 = time.time()
    corrected, diag = model.fit_transform_from_dnb(
        dnb_expr, dnb_coords_um, dnb_labels
    )
    elapsed = time.time() - t0
    if hasattr(corrected, 'toarray'):
        corrected = corrected.toarray()
    print(f"    Done in {elapsed:.1f}s, λ={model.lambda_:.0f}μm, "
          f"{diag.get('n_genes_corrected','?')} genes corrected")
    var_data = _build_sparkle_var_data(dnb_expr.shape[0], diag, r2_threshold)
    return corrected, {'lambda': float(model.lambda_), 'runtime': elapsed, 'var_data': var_data, **diag}


def run_soupx_method(sub, verbose=True):
    """运行原始 SoupX（全局 ρ，无空间信息）。

    处理方式：
      1. DNBs 聚合为 per-cell 表达（toc）
      2. 空 DNBs 作为 "空液滴" 估计背景谱
      3. KMeans 聚类 → marker gene 检测 → autoEstCont 估计 ρ
      4. autoEstCont/adjustCounts 失败时 run_soupx 直接抛错（无启发式回退）；
         本函数捕获后返回 (None, {'error': ...})，由调用方在 results/metrics
         中显式记为 NaN

    输出 diagnostics 包含 ρ 和 runtime。
    """
    print(f"\n  SoupX...")
    dnb_expr = sub['dnb_expr']
    dnb_labels = sub['dnb_labels']

    t0 = time.time()
    try:
        corrected, rho = run_soupx(dnb_expr, dnb_labels, verbose=verbose)
        elapsed = time.time() - t0
        print(f"    Done in {elapsed:.1f}s, ρ={rho:.4f}")
        return corrected, {'rho': float(rho), 'runtime': elapsed}
    except Exception as e:
        print(f"    SoupX failed: {e}")
        return None, {'error': str(e), 'runtime': time.time() - t0}


def run_decontx_method(sub, verbose=True):
    """运行 DecontX（cell-level EM, gene-specific α, no spatial info）。

    先聚合 DNB → cell 表达，再调用 decontx-python。
    """
    import scanpy as sc
    from decontx import decontx as run_dx
    print(f"\n  DecontX...")
    dnb_expr = sub['dnb_expr']
    dnb_labels = sub['dnb_labels']
    gene_names = list(sub['gene_names'])
    cell_ids = np.array(sub['cell_ids'])
    n_cells = len(cell_ids)

    # Ensure dnb_labels are 0-based cell indices. Axolotl loader returns
    # original cell IDs, while MOSTA/VisiumHD/synthetic already use 0-based.
    if dnb_labels.max() >= n_cells:
        label_to_idx = {int(cid): i for i, cid in enumerate(cell_ids)}
        dnb_labels = np.array([label_to_idx.get(int(l), -1) for l in dnb_labels],
                              dtype=np.int32)

    raw_cell = compute_cell_expr(dnb_expr, dnb_labels, n_cells)

    adata = ad.AnnData(X=raw_cell.T, dtype=np.float64)
    adata.var_names = gene_names
    adata.obs_names = [f"Cell_{cid}" for cid in cell_ids]
    raw_counts = adata.X.copy()
    if verbose:
        print("    Preprocessing (normalize, log1p, HVGs, scale, PCA, neighbors, leiden)...")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    n_hvgs = min(200, adata.n_vars)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_hvgs, flavor='seurat')
    sc.pp.scale(adata, max_value=10)
    n_pcs = min(20, adata.n_obs - 1, adata.n_vars - 1)
    sc.tl.pca(adata, n_comps=n_pcs, svd_solver='arpack')
    n_nbrs = min(15, adata.n_obs - 1)
    sc.pp.neighbors(adata, n_neighbors=n_nbrs)
    sc.tl.leiden(adata, resolution=0.5)
    if verbose:
        print(f"    Clusters: {adata.obs['leiden'].nunique()}")
    adata.X = raw_counts
    t0 = time.time()
    run_dx(adata, cluster_key='leiden', max_iter=200, seed=12345, verbose=False)
    elapsed = time.time() - t0
    corrected = adata.layers['decontX_counts']
    if hasattr(corrected, 'toarray'):
        corrected = corrected.toarray()
    corrected = corrected.T  # [genes × cells]
    contamination = float(adata.obs['decontX_contamination'].values.mean())
    library_qc = summarize_cell_libraries(corrected, cell_ids)
    if library_qc["n_zero_library_cells"]:
        zero_ids = library_qc["zero_library_cell_ids"]
        preview = ", ".join(str(cid) for cid in zero_ids[:10])
        if len(zero_ids) > 10:
            preview += ", ..."
        print(
            "    WARNING: DecontX produced "
            f"{library_qc['n_zero_library_cells']} zero-library cells "
            f"after integer rounding (cell IDs: {preview})"
        )
    print(f"    Done in {elapsed:.1f}s, mean contamination={contamination:.3f}")
    return corrected, {
        'runtime': elapsed,
        'contamination': contamination,
        'library_qc': library_qc,
    }


def compute_cell_expr(dnb_expr, dnb_labels, n_cells):
    """将 DNB 级别表达聚合为细胞级别表达（按细胞求和）。
    使用稀疏矩阵乘法，避免 toarray() 导致 OOM。"""
    from scipy.sparse import csr_matrix
    n_dnbs = dnb_expr.shape[1]
    cell_dnb_idx = np.where(dnb_labels >= 0)[0]
    cell_indices = dnb_labels[cell_dnb_idx]
    C = csr_matrix(
        (np.ones(len(cell_dnb_idx), dtype=np.float64),
         (cell_dnb_idx, cell_indices)),
        shape=(n_dnbs, n_cells)
    )
    raw = (dnb_expr @ C).toarray() if hasattr(dnb_expr @ C, 'toarray') else np.asarray(dnb_expr @ C)
    return raw


def _cell_library_sizes(expr, clip_negative=True, row_chunk_size=512):
    """Return per-cell library sizes for a genes-by-cells matrix.

    Downstream ovarian R workflows clip negative corrected values before using
    them as counts.  ``clip_negative=True`` mirrors that behavior without
    densifying sparse matrices or copying an entire dense ovarian matrix.
    Non-finite values are rejected because a library-size summary containing
    NaN/Inf is not meaningful and would poison downstream normalization.
    """
    if len(expr.shape) != 2:
        raise ValueError(f"expr must be two-dimensional, got shape {expr.shape}")

    n_cells = expr.shape[1]
    totals = np.zeros(n_cells, dtype=np.float64)
    n_negative = 0

    if issparse(expr):
        values = expr.data
        n_nonfinite = int(np.count_nonzero(~np.isfinite(values)))
        if n_nonfinite:
            raise ValueError(f"expression matrix contains {n_nonfinite} non-finite values")
        n_negative = int(np.count_nonzero(values < 0))
        matrix = expr
        if clip_negative and n_negative:
            matrix = expr.copy()
            matrix.data[matrix.data < 0] = 0
            matrix.eliminate_zeros()
        totals = np.asarray(matrix.sum(axis=0), dtype=np.float64).ravel()
    else:
        array = np.asarray(expr)
        for start in range(0, array.shape[0], row_chunk_size):
            block = np.asarray(array[start:start + row_chunk_size], dtype=np.float64)
            n_nonfinite = int(np.count_nonzero(~np.isfinite(block)))
            if n_nonfinite:
                raise ValueError(
                    "expression matrix contains non-finite values "
                    f"(at least {n_nonfinite} in rows {start}:"
                    f"{min(start + row_chunk_size, array.shape[0])})"
                )
            negative = block < 0
            n_negative += int(np.count_nonzero(negative))
            if clip_negative and np.any(negative):
                block = block.copy()
                block[negative] = 0
            totals += block.sum(axis=0)

    if not np.all(np.isfinite(totals)):
        raise ValueError("computed cell library sizes contain non-finite values")
    return totals, n_negative


def summarize_cell_libraries(expr, cell_ids=None, clip_negative=True):
    """Build a compact, JSON-safe QC summary for cell library sizes."""
    totals, n_negative = _cell_library_sizes(expr, clip_negative=clip_negative)
    if cell_ids is None:
        cell_ids = np.arange(len(totals))
    cell_ids = np.asarray(cell_ids)
    if len(cell_ids) != len(totals):
        raise ValueError(
            f"cell_ids has length {len(cell_ids)}, expected {len(totals)}"
        )

    zero_mask = totals <= 0
    zero_ids = cell_ids[zero_mask]
    return {
        "n_cells": int(len(totals)),
        "n_zero_library_cells": int(zero_mask.sum()),
        "zero_library_cell_ids": zero_ids.tolist(),
        "n_negative_values_clipped_for_qc": int(n_negative if clip_negative else 0),
        "total_counts": float(totals.sum()),
        "min_cell_counts": float(totals.min()) if len(totals) else None,
        "median_cell_counts": float(np.median(totals)) if len(totals) else None,
        "max_cell_counts": float(totals.max()) if len(totals) else None,
    }


def _reports_root():
    """Return and create evaluation/reports sub-directories."""
    root = Path(__file__).resolve().parent.parent / "reports"
    (root / "h5ad").mkdir(parents=True, exist_ok=True)
    (root / "metrics").mkdir(parents=True, exist_ok=True)
    return root


def _convert_for_json(obj):
    """Recursively convert numpy scalars/arrays to Python JSON types."""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _convert_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_convert_for_json(v) for v in obj]
    return obj


def save_result_h5ad(expr, gene_names, cell_ids, ann_map, out_path,
                     method_name="", save_h5ad=True, var_data=None,
                     cell_coords=None):
    """Save a [genes x cells] expression matrix as cell-based h5ad.

    Args:
        expr: [genes x cells] dense or sparse matrix.
        gene_names: list/array of gene names.
        cell_ids: list/array of cell IDs. Only the first expr.shape[1] IDs
            are used, allowing corrected outputs that dropped trailing cells.
        ann_map: dict cell_id -> annotation, may be None.
        out_path: Path to write.
        method_name: optional method tag stored in .uns.
        save_h5ad: if False, skip writing h5ad file.
        var_data: optional dict of per-gene annotations (e.g. R² scores)
            to add to adata.var.
        cell_coords: optional [cells x 2] array of physical x/y centroids.  CRC
            stores these in obs so RCTD can reuse the exact cropped geometry.
    """
    if not save_h5ad:
        print(f"    Skipping h5ad save: {out_path}")
        return
    n_cells_expr = expr.shape[1]
    cell_ids_use = np.asarray(cell_ids)[:n_cells_expr]
    X = expr.T
    if hasattr(X, "toarray"):
        X = X.astype(np.float32)
    else:
        X = np.asarray(X, dtype=np.float32)

    adata = ad.AnnData(X=X)
    adata.var_names = [str(g) for g in gene_names]
    if var_data is not None:
        for key, vals in var_data.items():
            adata.var[key] = vals
    adata.obs_names = [f"Cell_{cid}" for cid in cell_ids_use]
    adata.obs["cell_id"] = cell_ids_use
    if ann_map is not None:
        # Some datasets use integer labels while CRC uses anonymized string IDs.
        # Try the original key first and only attempt an integer fallback when it
        # is meaningful; unconditional int(cid) would fail on CRC barcodes.
        annotations = []
        for cid in cell_ids_use:
            value = ann_map.get(cid)
            if value is None:
                value = ann_map.get(str(cid))
            if value is None:
                try:
                    value = ann_map.get(int(cid))
                except (TypeError, ValueError):
                    pass
            annotations.append(str(value if value is not None else "Unknown"))
        adata.obs["annotation"] = annotations
    if cell_coords is not None:
        coords_use = np.asarray(cell_coords, dtype=np.float64)[:n_cells_expr]
        if coords_use.shape != (n_cells_expr, 2):
            raise ValueError(
                f"cell_coords must have shape {(n_cells_expr, 2)}, got {coords_use.shape}"
            )
        adata.obs["x"] = coords_use[:, 0]
        adata.obs["y"] = coords_use[:, 1]
        adata.obsm["spatial"] = coords_use
    if method_name:
        adata.uns["method"] = str(method_name)
    adata.write_h5ad(out_path)
    print(f"    Saved h5ad: {out_path}")


def save_metrics_json(metrics, out_path):
    """Save a metrics dict as JSON."""
    import json
    with open(out_path, "w") as f:
        json.dump(_convert_for_json(metrics), f, indent=2)
    print(f"    Saved metrics: {out_path}")


def evaluate_mosta(results, data, sub, gene_names, raw=None):
    """对每个方法计算层间 DE 基因数和异类细胞间 Spearman 相关。

    指标说明：
      DE genes:    层间差异表达基因数（t-test p<0.05，top 200 var genes）
      Spearman:    不同皮层层次细胞间的平均 Spearman rank 相关（对过度校正更鲁棒）

    使用皮层层次标注：EX L2/3, EX L4, EX L5/6, EX L6
    """
    from scipy.stats import ttest_ind, spearmanr

    layer_annotations = ['EX L2/3', 'EX L4', 'EX L5/6', 'EX L6']
    ann_map = data.get('ann_map', {})
    cell_ids = data.get('cell_ids', np.array([]))

    cell_anns = np.array([ann_map.get(cid, 'Unknown') for cid in cell_ids])
    layer_mask = np.array([a in layer_annotations for a in cell_anns])

    if layer_mask.sum() < 10:
        print(f"  Too few cortical layer cells ({layer_mask.sum()}), skipping MOSTA evaluation")
        return None, None

    n_cells = len(cell_ids)
    if raw is None:
        raw = compute_cell_expr(sub['dnb_expr'], sub['dnb_labels'], n_cells)

    # Helper: layer-aggregated Spearman (mean per layer, then pairwise)
    def compute_layer_spearman(expr):
        """Aggregate cells by layer (mean), compute pairwise Spearman between layers."""
        unique_layers = sorted(set(layer_anns_sub))
        n_layers = len(unique_layers)
        if n_layers < 2:
            return float('nan')
        # Aggregate: mean expression per layer
        layer_means = np.zeros((n_layers, expr.shape[0]))
        for li, layer in enumerate(unique_layers):
            mask = layer_anns_sub == layer
            layer_means[li] = expr[:, mask].mean(axis=1)
        # Pairwise Spearman between layer mean vectors
        corrs = []
        for i in range(n_layers):
            for j in range(i + 1, n_layers):
                try:
                    s, _ = spearmanr(layer_means[i], layer_means[j])
                    corrs.append(s if not np.isnan(s) else 0)
                except:
                    pass
        return float(np.mean(corrs)) if corrs else float('nan')

    # Compute DE genes
    def compute_de_genes(expr):
        top_var_genes = np.argsort(np.var(expr, axis=1))[-200:]
        de_total = 0
        unique_layers = sorted(set(layer_anns_sub))
        for i in range(len(unique_layers)):
            for j in range(i + 1, len(unique_layers)):
                mi = layer_anns_sub == unique_layers[i]
                mj = layer_anns_sub == unique_layers[j]
                if mi.sum() < 3 or mj.sum() < 3:
                    continue
                for g in top_var_genes:
                    try:
                        _, p = ttest_ind(expr[g, mi], expr[g, mj])
                        if p < 0.05:
                            de_total += 1
                    except:
                        pass
        return de_total

    layer_anns_sub = cell_anns[layer_mask]

    # RAW evaluation
    raw_layer = raw[:, layer_mask]
    raw_de = compute_de_genes(raw_layer)
    raw_s = compute_layer_spearman(raw_layer)

    print(f"\n  {'='*60}")
    print(f"  MOSTA Cortical Layer Evaluation ({layer_mask.sum()} layer cells, {sub['dnb_expr'].shape[0]} genes)")
    print(f"  {'='*60}")
    print(f"  {'Method':<16} {'DE genes':>10} {'Spearman':>10}")
    print(f"  {'─'*16} {'─'*10} {'─'*10}")
    raw_s_str = f"{raw_s:.4f}" if not np.isnan(raw_s) else "nan"
    print(f"  {'RAW':<16} {raw_de:>10} {raw_s_str:>10}")

    for method_name, r in results.items():
        corrected = r.get('corrected')
        if corrected is None:
            continue

        layer_expr = corrected[:, layer_mask]
        de_total = compute_de_genes(layer_expr)
        s_mean = compute_layer_spearman(layer_expr)

        r['mosta_de_genes'] = de_total
        r['mosta_layer_spearman'] = s_mean
        r['mosta_between_corr'] = s_mean  # keep compat

        s_str = f"{s_mean:.4f}" if not np.isnan(s_mean) else "nan"
        print(f"  {method_name:<16} {de_total:>10} {s_str:>10}")

    raw_summary = {
        "de_genes": raw_de,
        "layer_spearman": raw_s,
    }
    return raw, raw_summary


def evaluate_mousebrain(results, data, sub, gene_names, raw=None):
    """MouseBrain 评估：以 snRNA-seq transfer 的 cell_subclass 为基准，

    计算类间 DE 基因数和类间 Spearman。
    """
    from scipy.stats import ttest_ind, spearmanr

    ann_map = data.get('ann_map', {})
    cell_ids = data.get('cell_ids', np.array([]))

    cell_anns = np.array([ann_map.get(int(cid), 'Unknown') for cid in cell_ids])
    known_mask = cell_anns != 'Unknown'
    if known_mask.sum() < 10:
        print(f"  Too few annotated cells ({known_mask.sum()}), skipping MouseBrain evaluation")
        return None, None

    n_cells = len(cell_ids)
    if raw is None:
        raw = compute_cell_expr(sub['dnb_expr'], sub['dnb_labels'], n_cells)

    class_anns = cell_anns[known_mask]
    unique_classes = sorted(set(class_anns))

    def compute_class_spearman(expr):
        """Aggregate cells by class, compute mean pairwise Spearman."""
        if len(unique_classes) < 2:
            return float('nan')
        class_means = np.zeros((len(unique_classes), expr.shape[0]))
        for li, cls in enumerate(unique_classes):
            mask = class_anns == cls
            class_means[li] = expr[:, mask].mean(axis=1)
        corrs = []
        for i in range(len(unique_classes)):
            for j in range(i + 1, len(unique_classes)):
                try:
                    s, _ = spearmanr(class_means[i], class_means[j])
                    corrs.append(s if not np.isnan(s) else 0)
                except Exception:
                    pass
        return float(np.mean(corrs)) if corrs else float('nan')

    def compute_de_genes(expr):
        top_var_genes = np.argsort(np.var(expr, axis=1))[-200:]
        de_total = 0
        for i in range(len(unique_classes)):
            for j in range(i + 1, len(unique_classes)):
                mi = class_anns == unique_classes[i]
                mj = class_anns == unique_classes[j]
                if mi.sum() < 3 or mj.sum() < 3:
                    continue
                for g in top_var_genes:
                    try:
                        _, p = ttest_ind(expr[g, mi], expr[g, mj])
                        if p < 0.05:
                            de_total += 1
                    except Exception:
                        pass
        return de_total

    raw_known = raw[:, known_mask]
    raw_de = compute_de_genes(raw_known)
    raw_s = compute_class_spearman(raw_known)

    print(f"\n  {'='*60}")
    print(f"  MouseBrain Cell-class Evaluation ({known_mask.sum()} annotated cells, {len(unique_classes)} classes, {sub['dnb_expr'].shape[0]} genes)")
    print(f"  {'='*60}")
    print(f"  {'Method':<16} {'DE genes':>10} {'Spearman':>10}")
    print(f"  {'─'*16} {'─'*10} {'─'*10}")
    raw_s_str = f"{raw_s:.4f}" if not np.isnan(raw_s) else "nan"
    print(f"  {'RAW':<16} {raw_de:>10} {raw_s_str:>10}")

    for method_name, r in results.items():
        corrected = r.get('corrected')
        if corrected is None:
            continue
        class_expr = corrected[:, known_mask]
        de_total = compute_de_genes(class_expr)
        s_mean = compute_class_spearman(class_expr)
        r['mousebrain_de_genes'] = de_total
        r['mousebrain_class_spearman'] = s_mean
        s_str = f"{s_mean:.4f}" if not np.isnan(s_mean) else "nan"
        print(f"  {method_name:<16} {de_total:>10} {s_str:>10}")

    raw_summary = {
        "de_genes": raw_de,
        "class_spearman": raw_s,
    }
    return raw, raw_summary


def load_visiumhd_data(x_range=None, y_range=None, n_genes=None, verbose=True,
                       h5_path=None, dataset_name="Visium HD"):
    """Load Visium HD human colon cancer data at 2µm pixel resolution.

    Keeps native 2µm pixels as DNBs (matching Stereo-seq's 500nm DNB concept).
    Coordinates are returned in µm, so SPARKLE receives a 25-µm bin side
    directly, matching the 25-µm empty bins used for Stereo-seq.

    Optimized: dict→array pixel lookup, COO accumulation, vectorized per-gene ops.

    Args:
        x_range: (x_min, x_max) spatial window in 2µm pixel column units.
        y_range: (y_min, y_max) spatial window in 2µm pixel row units.
        n_genes: if set, only load top-N genes by total expression (two-pass).

    Returns:
        dict with dnb_expr, dnb_coords, dnb_labels, gene_names, cell_ids, ann_map.
    """
    import h5py

    if h5_path is None:
        data_dir = Path(__file__).resolve().parent.parent / "data" / "visiumhd"
        h5_path = data_dir / "Visium_HD_6p5mm_Human_Colon_Cancer_feature_slice.h5"
    else:
        h5_path = Path(h5_path)

    if not h5_path.exists():
        raise FileNotFoundError(f"{h5_path} not found.")

    load_all = False  # will be set in pass 1
    f = h5py.File(h5_path, 'r')

    if "segmentations" not in f:
        f.close()
        raise ValueError(
            f"{h5_path.name} has no 'segmentations' group; SPARKLE needs a "
            f"cell_segmentation_mask. This feature_slice.h5 lacks embedded "
            f"segmentation (segmented outputs not available for this sample).")

    # ── Step 1: Load 2µm pixel mask, build array-based lookup ──────
    t0 = time.time()
    if verbose:
        print("Loading 2µm tissue mask...")
    filt = f['masks/filtered']
    pixel_rows = filt['row'][:]  # int32
    pixel_cols = filt['col'][:]  # int32
    n_pixels_total = len(pixel_rows)

    max_row = int(pixel_rows.max())
    max_col = int(pixel_cols.max())
    if verbose:
        print(f"  Grid: {max_row+1} x {max_col+1}, {n_pixels_total} tissue pixels")

    # Build numpy array: pixel_to_idx[row, col] = cropped index or -1
    pixel_to_idx = np.full((max_row + 1, max_col + 1), -1, dtype=np.int32)
    dnb_coords_list = []  # collect coords only for kept pixels

    if x_range is None and y_range is None:
        # Fast path: no cropping, all pixels kept
        kept_indices = list(range(n_pixels_total))
        for i in range(n_pixels_total):
            r, c = int(pixel_rows[i]), int(pixel_cols[i])
            pixel_to_idx[r, c] = i
            dnb_coords_list.append((c * 2.0 + 1.0, r * 2.0 + 1.0))
    else:
        kept_indices = []
        for i in range(n_pixels_total):
            r, c = int(pixel_rows[i]), int(pixel_cols[i])
            if x_range and not (x_range[0] <= c <= x_range[1]):
                continue
            if y_range and not (y_range[0] <= r <= y_range[1]):
                continue
            new_idx = len(kept_indices)
            pixel_to_idx[r, c] = new_idx
            kept_indices.append(i)
            dnb_coords_list.append((c * 2.0 + 1.0, r * 2.0 + 1.0))

    n_pixels = len(kept_indices)
    dnb_coords = np.array(dnb_coords_list, dtype=np.float64)
    del dnb_coords_list

    if verbose:
        print(f"  {n_pixels} kept pixels, pixel_to_idx array: {pixel_to_idx.nbytes/1e6:.1f}MB")
        if n_pixels == 0:
            print("  ERROR: No pixels in range.")
            f.close()
            return None

    # ── Step 2: Build gene vocabulary ──────────────────────────────
    if verbose:
        print("Building gene vocabulary...")
    fs = f['feature_slices']
    all_gene_indices = sorted(int(k) for k in fs.keys())
    n_all_genes = len(all_gene_indices)
    all_gene_names = [n.decode() for n in f['features/name'][:]]

    # ── Step 3: Two-pass gene selection + sparse matrix build ─────
    if not load_all:
        # Pass 1: scan all genes, compute per-gene totals (no matrix stored)
        if verbose:
            print(f"Pass 1: scanning {n_all_genes} genes for top {n_genes}...")
        gene_totals_all = np.zeros(n_all_genes, dtype=np.float64)
        for gi, gene_id in enumerate(all_gene_indices):
            g = fs[str(gene_id)]
            rows = g['row'][:].astype(np.int32)
            cols = g['col'][:].astype(np.int32)
            data = g['data'][:].astype(np.float64)
            pix_indices = pixel_to_idx[rows, cols]
            mask = pix_indices >= 0
            if mask.any():
                gene_totals_all[gi] = data[mask].sum()
            if verbose and (gi + 1) % 5000 == 0:
                print(f"  Scanned {gi+1}/{n_all_genes}")

        # Select top N genes
        top_order = np.argsort(gene_totals_all)[::-1][:n_genes]
        top_set = set(top_order)
        gene_indices = [all_gene_indices[i] for i in top_order]
        gene_totals = gene_totals_all[top_order]
        del gene_totals_all
        n_genes_use = len(gene_indices)
        selected_gene_ids = set(gene_indices)
    else:
        # Load all genes (no two-pass needed)
        gene_indices = all_gene_indices
        n_genes_use = n_all_genes
        selected_gene_ids = set(all_gene_indices)
        if n_genes is not None:
            print(f"  Loading all {n_all_genes} genes (n_genes >= total)")

    gene_names_arr = np.array([all_gene_names[g] for g in gene_indices])

    if verbose:
        if load_all:
            print(f"  Loading all {n_all_genes} genes")
        else:
            print(f"  Loading {n_genes_use} genes (from {n_all_genes} total)")

    # Pass 2: count nonzeros only for selected genes, preallocate, fill
    if verbose:
        print("Counting nonzeros for selected genes...")
    gene_to_new = {g: i for i, g in enumerate(gene_indices)}
    gene_nnz = np.zeros(n_genes_use, dtype=np.int32)
    for gi, gene_id in enumerate(all_gene_indices):
        if n_genes is not None and n_genes < n_all_genes and gene_id not in selected_gene_ids:
            continue
        g = fs[str(gene_id)]
        rows = g['row'][:].astype(np.int32)
        cols = g['col'][:].astype(np.int32)
        pix_indices = pixel_to_idx[rows, cols]
        new_gi = gene_to_new[gene_id]
        gene_nnz[new_gi] = int((pix_indices >= 0).sum())

    total_nnz = int(gene_nnz.sum())
    if verbose:
        print(f"  Total nonzeros in window: {total_nnz:,}")

    # Preallocate COO arrays
    coo_rows_arr = np.empty(total_nnz, dtype=np.int32)
    coo_cols_arr = np.empty(total_nnz, dtype=np.int32)
    coo_data_arr = np.empty(total_nnz, dtype=np.float64)
    if not load_all:
        gene_totals = np.zeros(n_genes_use, dtype=np.float64)
    offset = 0

    if verbose:
        print(f"Building {n_genes_use} x {n_pixels} sparse matrix (COO)...")
    for gene_id_orig in all_gene_indices:
        if n_genes is not None and n_genes < n_all_genes and gene_id_orig not in selected_gene_ids:
            continue
        gi = gene_to_new[gene_id_orig]
        g = fs[str(gene_id_orig)]
        rows = g['row'][:].astype(np.int32)
        cols = g['col'][:].astype(np.int32)
        data = g['data'][:].astype(np.float64)

        pix_indices = pixel_to_idx[rows, cols]
        mask = pix_indices >= 0
        n_add = mask.sum()

        if n_add > 0:
            end = offset + n_add
            coo_rows_arr[offset:end] = gi
            coo_cols_arr[offset:end] = pix_indices[mask]
            coo_data_arr[offset:end] = data[mask]
            if not load_all:
                gene_totals[gi] = data[mask].sum()
            offset = end

        if verbose and (gi + 1) % 5000 == 0:
            print(f"  Gene {gi+1}/{n_genes_use}, processed: {offset:,}/{total_nnz:,}")

    dnb_expr = csr_matrix((coo_data_arr, (coo_rows_arr, coo_cols_arr)),
                          shape=(n_genes_use, n_pixels))
    del coo_rows_arr, coo_cols_arr, coo_data_arr

    if verbose:
        print(f"  CSR matrix: {dnb_expr.shape}, {dnb_expr.nnz:,} nonzeros")
        print(f"  Done in {time.time()-t0:.0f}s")

    # ── Step 4: Assign cell labels via array-based lookup ────────
    if verbose:
        print("Assigning cell labels from segmentation mask...")
    t1 = time.time()
    seg = f['segmentations/cell_segmentation_mask']
    seg_rows = seg['row'][:].astype(np.int32)
    seg_cols = seg['col'][:].astype(np.int32)
    seg_data = seg['data'][:].astype(np.int64)

    # Vectorized: pixel_to_idx lookup for all segmentation entries
    pix_indices = pixel_to_idx[seg_rows, seg_cols]
    mask = pix_indices >= 0

    kept_pix = pix_indices[mask]
    kept_cell_ids = seg_data[mask]

    # Unique cell IDs
    cell_ids_list = sorted(set(int(x) for x in np.unique(kept_cell_ids)))
    cell_to_idx = {cid: i for i, cid in enumerate(cell_ids_list)}
    n_cells = len(cell_ids_list)

    # Build dnb_labels via vectorized mapping
    dnb_labels = np.full(n_pixels, -1, dtype=np.int32)
    cell_indices_mapped = np.array([cell_to_idx[int(cid)] for cid in kept_cell_ids], dtype=np.int32)
    dnb_labels[kept_pix.astype(np.int32)] = cell_indices_mapped

    n_cell_pixels = int((dnb_labels >= 0).sum())
    n_empty_pixels = int((dnb_labels < 0).sum())
    if verbose:
        print(f"  {n_cells} cells, {n_cell_pixels} cell pixels, {n_empty_pixels} empty pixels")
        print(f"  Done in {time.time()-t1:.0f}s")

    f.close()

    # RCTD annotations are no longer loaded; leave ann_map empty.
    cell_ids_arr = np.array(cell_ids_list)
    ann_map_out = {}

    if verbose:
        print(f"\n{'='*60}")
        print(f"{dataset_name} data loaded: {n_genes} genes, {n_pixels} 2µm pixels, {n_cells} cells")
        print(f"{'='*60}")

    return {
        'dnb_expr': dnb_expr,
        'dnb_coords': dnb_coords,
        'dnb_labels': dnb_labels,
        'gene_names': gene_names_arr,
        'cell_ids': cell_ids_arr,
        'ann_map': ann_map_out,
        'adata': None,
        'x_range': x_range, 'y_range': y_range,
    }


def load_ovarian_data(x_range=None, y_range=None, n_genes=None, verbose=True):
    """Load Visium HD Human Ovarian Cancer (FF) data at 2µm pixel resolution.

    Same feature_slice.h5 layout as the colon cancer 6.5mm sample (embedded
    ``segmentations/cell_segmentation_mask``), so it reuses ``load_visiumhd_data``.
    """
    data_dir = Path(__file__).resolve().parent.parent / "data" / "ovarian"
    h5_path = data_dir / "Visium_HD_Human_Ovarian_Cancer_FF_feature_slice.h5"
    return load_visiumhd_data(
        x_range=x_range, y_range=y_range, n_genes=n_genes, verbose=verbose,
        h5_path=h5_path, dataset_name="Visium HD Ovarian Cancer")


def _load_crc_grid_coords(bin_ids_path):
    """Recover the regular 2-µm grid coordinates encoded in CRC bin IDs.

    ``spot_coords_um.npy`` contains registered physical coordinates.  Registration
    applies a small rotation, which is correct for distance calculations but makes
    nearly every floating-point x/y value unique.  The generic binning code infers
    spot pitch from repeated axis coordinates, so feeding it the registered values
    would incorrectly infer a near-zero pitch and create one bin per spot.

    The IDs have the form ``s_002um_<grid_x>_<grid_y>-1``.  Multiplying both grid
    indices by 2 preserves all Euclidean distances under the rigid registration
    while restoring an exact rectilinear grid for robust 25-µm aggregation.
    """
    coords = []
    with open(bin_ids_path, "r", encoding="utf-8") as handle:
        next(handle)  # bin_id header
        for line_number, line in enumerate(handle, start=2):
            token = line.strip()
            parts = token.split("_")
            if len(parts) != 4 or parts[0] != "s" or parts[1] != "002um":
                raise ValueError(
                    f"Unexpected CRC bin ID at {bin_ids_path}:{line_number}: {token!r}"
                )
            try:
                grid_x = int(parts[2])
                grid_y = int(parts[3].split("-")[0])
            except ValueError as exc:
                raise ValueError(
                    f"Malformed CRC grid indices at {bin_ids_path}:{line_number}: {token!r}"
                ) from exc
            coords.append((grid_x * 2.0, grid_y * 2.0))
    return np.asarray(coords, dtype=np.float64)


def _cell_centroids(coords, labels, n_cells):
    """Return mean x/y coordinates for contiguous non-negative cell labels."""
    covered = labels >= 0
    counts = np.bincount(labels[covered], minlength=n_cells).astype(np.float64)
    if np.any(counts == 0):
        raise ValueError("Cell label remapping produced a cell without covered spots")
    x_sum = np.bincount(labels[covered], weights=coords[covered, 0], minlength=n_cells)
    y_sum = np.bincount(labels[covered], weights=coords[covered, 1], minlength=n_cells)
    return np.column_stack((x_sum / counts, y_sum / counts))


def load_crc_data(segmentation="proseg", x_range=None, y_range=None, verbose=True):
    """Load the SPARKLE-ready CRC slice for one segmentation condition.

    Proseg and StarDist share the same raw expression, registered spot
    coordinates, bin IDs, and genes.  Only ``spot_labels.npy`` and the associated
    ``cell_ids.npy`` differ.  Keeping ``segmentation`` explicit therefore makes
    the two conditions directly comparable without duplicating data preparation.

    Spatial windows are interpreted in the registered physical coordinate system
    reported by ``spot_coords_um.npy``.  Labels are remapped after cropping so
    every downstream method receives contiguous 0..N-1 indices; original cell
    IDs are retained in the output h5ad files for RCTD annotation round-tripping.
    """
    from scipy.sparse import load_npz

    segmentation = segmentation.lower()
    if segmentation not in {"proseg", "stardist"}:
        raise ValueError("CRC segmentation must be 'proseg' or 'stardist'")

    data_root = Path(__file__).resolve().parent.parent / "data" / "CRC" / "07.sparkle_ready"
    seg_dir = data_root / segmentation
    required = [
        seg_dir / "spot_expr_genes_by_bins.npz",
        seg_dir / "spot_coords_um.npy",
        seg_dir / "spot_labels.npy",
        seg_dir / "cell_ids.npy",
        seg_dir / "gene_names.tsv",
        seg_dir / "bin_ids.tsv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing CRC input file(s): " + ", ".join(missing))

    physical_coords_all = np.load(seg_dir / "spot_coords_um.npy", mmap_mode="r")
    labels_all = np.load(seg_dir / "spot_labels.npy", mmap_mode="r")
    cell_ids_all = np.load(seg_dir / "cell_ids.npy", allow_pickle=False)

    if verbose:
        xy_min = np.asarray(physical_coords_all).min(axis=0)
        xy_max = np.asarray(physical_coords_all).max(axis=0)
        print(f"CRC registered spot range (µm): "
              f"x=[{xy_min[0]:.2f}, {xy_max[0]:.2f}], "
              f"y=[{xy_min[1]:.2f}, {xy_max[1]:.2f}]")

    keep = np.ones(len(labels_all), dtype=bool)
    if x_range is not None:
        keep &= ((physical_coords_all[:, 0] >= x_range[0]) &
                 (physical_coords_all[:, 0] <= x_range[1]))
    if y_range is not None:
        keep &= ((physical_coords_all[:, 1] >= y_range[0]) &
                 (physical_coords_all[:, 1] <= y_range[1]))
    kept_indices = np.flatnonzero(keep)
    if kept_indices.size == 0:
        raise ValueError(
            f"CRC window x={x_range}, y={y_range} contains no registered spots"
        )

    # Crop the shared sparse expression only after the inexpensive coordinate
    # mask is known.  CSR column slicing is cheap at this dataset size and keeps
    # memory proportional to the selected window.
    dnb_expr = load_npz(seg_dir / "spot_expr_genes_by_bins.npz")[:, kept_indices].tocsr()
    physical_coords = np.asarray(physical_coords_all[kept_indices], dtype=np.float64)
    grid_coords_all = _load_crc_grid_coords(seg_dir / "bin_ids.tsv")
    dnb_coords = grid_coords_all[kept_indices]
    original_labels = np.asarray(labels_all[kept_indices], dtype=np.int32)

    # A cropped window usually contains only a subset of the full-slice cells.
    # Vectorized remapping prevents large original labels from being interpreted
    # as column indices by cell-level aggregation.
    present_labels = np.unique(original_labels[original_labels >= 0])
    if present_labels.size == 0:
        raise ValueError(
            f"CRC {segmentation} window x={x_range}, y={y_range} has no segmented cells"
        )
    if present_labels[-1] >= len(cell_ids_all):
        raise ValueError("CRC spot label exceeds the available cell_ids mapping")
    label_lookup = np.full(int(present_labels[-1]) + 1, -1, dtype=np.int32)
    label_lookup[present_labels] = np.arange(len(present_labels), dtype=np.int32)
    dnb_labels = np.full_like(original_labels, -1)
    covered = original_labels >= 0
    dnb_labels[covered] = label_lookup[original_labels[covered]]
    cell_ids = np.asarray(cell_ids_all)[present_labels]
    cell_coords = _cell_centroids(physical_coords, dnb_labels, len(cell_ids))

    with open(seg_dir / "gene_names.tsv", "r", encoding="utf-8") as handle:
        next(handle)  # gene_name header
        gene_names = np.asarray([line.rstrip("\n") for line in handle])
    if dnb_expr.shape[0] != len(gene_names):
        raise ValueError("CRC expression rows and gene_names.tsv are inconsistent")

    if verbose:
        n_cell_spots = int((dnb_labels >= 0).sum())
        n_empty_spots = int((dnb_labels < 0).sum())
        print(f"CRC/{segmentation}: {len(gene_names):,} genes, "
              f"{len(dnb_labels):,} spots ({n_cell_spots:,} cell + "
              f"{n_empty_spots:,} empty), {len(cell_ids):,} cells")

    return {
        "dnb_expr": dnb_expr,
        "dnb_coords": dnb_coords,
        "dnb_labels": dnb_labels,
        "gene_names": gene_names,
        "cell_ids": cell_ids,
        "cell_coords": cell_coords,
        "physical_spot_coords": physical_coords,
        "ann_map": {},
        "adata": None,
        "x_range": x_range,
        "y_range": y_range,
        "segmentation": segmentation,
    }


def run_visiumhd_comparison(data, sub, n_genes=200, methods=None, n_high_genes=None, lambda_grid=None, r2_threshold=None, save_h5ad=True, max_radius=None, dataset_tag="visiumhd", use_gpu=False):
    """Run the shared comparison pipeline for Visium HD-like 2-µm spots."""
    if methods is None:
        methods = ['sparkle', 'soupx', 'decontx']
    methods = [m.lower().strip() for m in methods]
    if n_high_genes is None:
        n_high_genes = min(n_genes, 500)

    print(f"\n{'='*60}")
    print("RUNNING METHODS")
    print(f"{'='*60}")

    results = {}

    if 'sparkle' in methods:
        corrected, diag = run_sparkle_method(
            sub,
            coordinate_scale_to_um=1.0,
            n_high_genes=n_high_genes,
            lambda_grid=lambda_grid,
            r2_threshold=r2_threshold,
            max_radius=max_radius,
            use_gpu=use_gpu,
        )
        if corrected is not None:
            results['SPARKLE'] = {'corrected': corrected, 'diag': diag}

    if 'soupx' in methods:
        corrected, diag = run_soupx_method(sub)
        # Record even on failure (corrected=None) so the failure surfaces as
        # NaN/error in summaries and metrics instead of vanishing silently.
        results['SoupX'] = {'corrected': corrected, 'diag': diag}

    if 'decontx' in methods:
        corrected, diag = run_decontx_method(sub)
        if corrected is not None:
            results['DecontX'] = {'corrected': corrected, 'diag': diag}

    # Save cell-based h5ad and metrics
    print(f"\n  {'='*60}")
    print(f"  Saving results")
    print(f"  {'='*60}")
    reports_root = _reports_root()
    x_range = data.get('x_range')
    y_range = data.get('y_range')
    if x_range is not None and y_range is not None:
        tag = f"{dataset_tag}_x{x_range[0]}-{x_range[1]}_y{y_range[0]}-{y_range[1]}"
    else:
        tag = f"{dataset_tag}_full"

    cell_ids = data.get('cell_ids', np.array([]))
    n_cells = len(cell_ids)
    raw_cell = compute_cell_expr(sub['dnb_expr'], sub['dnb_labels'], n_cells)
    ann_map = data.get('ann_map', {})
    cell_coords = data.get('cell_coords')
    save_result_h5ad(raw_cell, sub['gene_names'], cell_ids, ann_map,
                     reports_root / "h5ad" / f"{tag}_raw.h5ad", "RAW",
                     save_h5ad=save_h5ad, cell_coords=cell_coords)
    raw_library_qc = summarize_cell_libraries(raw_cell, cell_ids)
    metrics = {
        "dataset": tag,
        "n_cells": int(raw_cell.shape[1]),
        "n_genes": int(raw_cell.shape[0]),
        "raw": {"library_qc": raw_library_qc},
        "methods": {},
    }
    for method_name, r in results.items():
        corrected = r.get('corrected')
        if corrected is not None:
            var_data = r.get('diag', {}).get('var_data') if method_name == 'SPARKLE' else None
            save_result_h5ad(corrected, sub['gene_names'], cell_ids, ann_map,
                             reports_root / "h5ad" / f"{tag}_{method_name}.h5ad",
                             method_name,
                             save_h5ad=save_h5ad,
                             var_data=var_data,
                             cell_coords=cell_coords)
            library_qc = r.get('diag', {}).get('library_qc')
            if library_qc is None:
                library_qc = summarize_cell_libraries(corrected, cell_ids)
            if library_qc["n_zero_library_cells"]:
                print(
                    f"    WARNING: {method_name} has "
                    f"{library_qc['n_zero_library_cells']} zero-library cells; "
                    "downstream cross-method analyses must exclude their union."
                )
            metrics["methods"][method_name] = {
                "runtime": r['diag'].get('runtime', 0),
                "library_qc": library_qc,
            }
        else:
            # Method failed (e.g. SoupX): keep the failure visible as NaN.
            metrics["methods"][method_name] = {
                "runtime": r.get('diag', {}).get('runtime', 0),
                "library_qc": None,
                "error": r.get('diag', {}).get('error'),
            }
    save_metrics_json(metrics, reports_root / "metrics" / f"{tag}_metrics.json")

    # ── Summary ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Method':<16} {'Runtime':>8}")
    print(f"  {'─'*16} {'─'*8}")

    print(f"  {'RAW':<16} {'':>8}")

    for method_name, r in results.items():
        d = r['diag']
        runtime = d.get('runtime', 0)
        print(f"  {method_name:<16} {runtime:7.1f}s")


def main():
    parser = argparse.ArgumentParser(
        description="Final comparison: Cell SPARKLE and ambient-RNA baselines")
    parser.add_argument("--dataset", type=str, default="axolotl",
                        choices=["axolotl", "mousebrain", "ovarian", "synthetic"],
                        help="Dataset (default: axolotl)")
    parser.add_argument("--scenario", type=str, default="S1",
                        choices=sorted(SCENARIOS.keys()),
                        help="Synthetic scenario ID (default: S1)")
    parser.add_argument("--all-scenarios", action="store_true",
                        help="Run all S1–S10 synthetic scenarios and print summary")
    parser.add_argument("--x-range", type=int, nargs=2, default=None,
                        help="X range (two integers)")
    parser.add_argument("--y-range", type=int, nargs=2, default=None,
                        help="Y range (two integers)")
    parser.add_argument("--n-genes", type=int, default=200,
                        help="Number of top genes (default: 200)")
    parser.add_argument("--n-high-genes", type=int, default=None,
                        help="Number of top genes (default: 200)")
    parser.add_argument("--lambda-grid", type=int, nargs="+",
                        default=[10, 20, 30, 50, 70, 100, 150, 200, 300],
                        help="Lambda candidates in um for SPARKLE (default: 10 20 30 50 70 100 150 200 300)")
    parser.add_argument("--r2-threshold", type=float, default=0.01,
                        help="Minimum weighted R^2 for SPARKLE gene correction (default: 0.01)")
    parser.add_argument("--max-radius", type=float, default=None,
                        help="Spatial neighborhood radius in um (default: 200 for axolotl, 300 for others)")
    parser.add_argument("--use-gpu", action="store_true",
                        help="Use CUDA for SPARKLE when available; otherwise fall back to CPU")
    parser.add_argument("--save-h5ad", action=argparse.BooleanOptionalAction, default=True,
                        help="Save corrected h5ad files (default: True)")
    parser.add_argument("--cut-genes", action=argparse.BooleanOptionalAction, default=False,
                        help="Cut gene matrix to top N genes in subsample_data (default: False)")
    parser.add_argument("--methods", type=str,
                        default=None,
                        help="Comma-separated methods to run")
    parser.add_argument("--annotation-level", type=str, default="cell_group",
                        choices=["cell_class", "cell_subclass", "cell_group"],
                        help="MouseBrain annotation level to use (default: cell_group)")
    args = parser.parse_args()

    x_range = tuple(args.x_range) if args.x_range else None
    y_range = tuple(args.y_range) if args.y_range else None
    if args.methods is None:
        default_methods = "sparkle,soupx,decontx"
        methods = default_methods.split(",")
    else:
        methods = [m.strip().lower() for m in args.methods.split(',')]
    allowed_methods = {"sparkle", "soupx", "decontx"}
    unknown_methods = sorted(set(methods) - allowed_methods)
    if unknown_methods:
        parser.error(
            "unsupported method(s): "
            + ", ".join(unknown_methods)
            + "; choose from sparkle,soupx,decontx"
        )
    lambda_grid = args.lambda_grid
    r2_threshold = args.r2_threshold
    save_h5ad = args.save_h5ad
    if args.max_radius is None:
        # Match help text defaults: 200 for axolotl, 300 for others
        max_radius = 200.0 if args.dataset == "axolotl" else 300.0
    else:
        max_radius = args.max_radius
    cut_genes = args.cut_genes

    print("=" * 60)
    print(f"FINAL COMPARISON: {args.dataset.upper()}")
    print(f"  Genes: {args.n_genes}")
    print(f"  Methods: {', '.join(methods)}")
    if x_range:
        print(f"  X range: {x_range}")
    if y_range:
        print(f"  Y range: {y_range}")
    print("=" * 60)

    if args.dataset == "axolotl":
        data = load_axolotl_data_windowed(x_range, y_range)
        run_axolotl_comparison(data, args.n_genes, methods, lambda_grid=lambda_grid, r2_threshold=r2_threshold, save_h5ad=save_h5ad, max_radius=max_radius, use_gpu=args.use_gpu)
    elif args.dataset == "mousebrain":
        data = load_mousebrain_data(x_range=x_range, y_range=y_range, annotation_level=args.annotation_level)
        sub = subsample_data(data, args.n_genes, cut_genes=cut_genes)
        run_mousebrain_comparison(data, sub, args.n_genes, methods, n_high_genes=args.n_high_genes, lambda_grid=lambda_grid, r2_threshold=r2_threshold, save_h5ad=save_h5ad, max_radius=max_radius, use_gpu=args.use_gpu)
    elif args.dataset == "ovarian":
        data = load_ovarian_data(x_range=x_range, y_range=y_range, n_genes=args.n_genes)
        if data is None:
            sys.exit(1)
        sub = subsample_data(data, args.n_genes, cut_genes=cut_genes)
        run_visiumhd_comparison(data, sub, args.n_genes, methods, n_high_genes=args.n_high_genes, lambda_grid=lambda_grid, r2_threshold=r2_threshold, save_h5ad=save_h5ad, max_radius=max_radius, dataset_tag="ovarian", use_gpu=args.use_gpu)
    else:  # synthetic
        if args.all_scenarios:
            run_all_synthetic_scenarios(
                methods,
                lambda_grid=lambda_grid,
                r2_threshold=r2_threshold,
                save_h5ad=save_h5ad,
                max_radius=max_radius,
                use_gpu=args.use_gpu,
            )
        else:
            data = load_synthetic_scenario_data(args.scenario, seed=42)
            run_synthetic_comparison(
                data,
                args.n_genes,
                methods,
                lambda_grid=lambda_grid,
                r2_threshold=r2_threshold,
                save_h5ad=save_h5ad,
                max_radius=max_radius,
                use_gpu=args.use_gpu,
            )


if __name__ == "__main__":
    main()
