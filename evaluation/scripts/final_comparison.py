#!/usr/bin/env python3
"""Final comparison: Cell SPARKLE vs Spatial SoupX vs SoupX.

Supports Axolotl (sstIN evaluation) and MOSTA (cortical layer evaluation).
Supports spatial window via --x-range and --y-range.

Note: CellBender is excluded because its VAE fails on spatial DNB data
(empty bins have too few counts / zero division in prior estimation).
"""

import sys, os, time, argparse, gzip, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from scipy.sparse import csr_matrix, lil_matrix
from scipy.stats import ttest_ind
import anndata as ad
from stambient import SPARKLE
from evaluation.baselines.spatial_soupx import run_spatial_soupx
from evaluation.baselines.soupx import run_soupx
from evaluation.scripts.test_axolotl import compute_neighbor_stats
from evaluation.synthetic import generate_synthetic_data, SCENARIOS
import scanpy as sc


def load_synthetic_scenario_data(scenario_id="S1", seed=42):
    """Load a synthetic benchmark scenario (S1–S10).

    First tries to load a cached NPZ file at evaluation/data/S{scenario_id}.npz
    (the original benchmark data). If not present, falls back to the generator.

    Returns a dict compatible with the comparison pipeline:
        dnb_expr, dnb_coords, dnb_labels, gene_names, cell_ids,
        true_expr, gene_is_high, params, true_alpha, true_lambda, metadata.
    """
    from evaluation.synthetic.scenarios import get_scenario

    scenario = get_scenario(scenario_id)
    print(f"[Synthetic] Scenario {scenario_id}: {scenario['name']}")

    project_root = Path(__file__).resolve().parent.parent.parent
    npz_path = project_root / "evaluation" / "data" / f"{scenario_id}.npz"

    if npz_path.exists():
        print(f"  Loading cached data: {npz_path}")
        npz = np.load(npz_path, allow_pickle=True)
        dnb_expr = npz["dnb_expr"]
        dnb_coords = npz["dnb_coords"]
        dnb_labels = npz["dnb_labels"]
        true_expr = npz["true_expr"]
        gene_is_high = npz["gene_is_high"]
        true_alpha = npz["true_alpha"]
        true_lambda = float(npz["true_lambda"])
        metadata = npz["metadata"].item()

        n_genes, n_dnbs = dnb_expr.shape
        n_cells = true_expr.shape[1]
        gene_names = np.array([f"gene_{i}" for i in range(n_genes)])
        cell_ids = np.arange(n_cells, dtype=np.int64)
        n_empty = int((dnb_labels < 0).sum())

        print(f"  {n_genes} genes, {n_cells} cells, {n_dnbs} DNBs "
              f"({n_empty} empty, {n_dnbs - n_empty} cell)")
        print(f"  Ground-truth λ={true_lambda}µm, "
              f"α(mean)={true_alpha.mean():.4f} from cached NPZ")

        return {
            "dnb_expr": csr_matrix(dnb_expr.astype(np.float64)),
            "dnb_coords": dnb_coords,
            "dnb_labels": dnb_labels,
            "gene_names": gene_names,
            "cell_ids": cell_ids,
            "true_expr": true_expr,
            "gene_is_high": gene_is_high,
            "true_alpha": true_alpha,
            "true_lambda": true_lambda,
            "metadata": metadata,
            "params": {"source": "npz_cache", "scenario": scenario},
        }

    # Fallback: generate on the fly
    print(f"  NPZ cache not found, generating synthetic data on the fly...")
    data = generate_synthetic_data(
        n_cells=200,
        grid_width=200,
        grid_height=200,
        dnb_pitch=0.5,
        cell_radius=5.0,
        n_genes=500,
        n_high_genes=80,
        ambient_lambda=scenario["ambient_lambda"],
        ambient_alpha=scenario["ambient_alpha"],
        empty_fraction=scenario["empty_fraction"],
        n_cell_types=scenario.get("n_cell_types", 1),
        marker_fraction=scenario.get("marker_fraction", 0.0),
        seed=seed,
    )

    n_genes = data["dnb_expr"].shape[0]
    n_cells = data["true_expr"].shape[1]
    gene_names = np.array([f"gene_{i}" for i in range(n_genes)])
    cell_ids = np.arange(n_cells, dtype=np.int64)

    n_empty = int((data["dnb_labels"] < 0).sum())
    print(f"  {n_genes} genes, {n_cells} cells, {len(data['dnb_labels'])} DNBs "
          f"({n_empty} empty, {len(data['dnb_labels']) - n_empty} cell)")
    print(f"  Ground-truth λ={scenario['ambient_lambda']}µm, "
          f"α={scenario['ambient_alpha']}, empty_fraction={scenario['empty_fraction']}")

    return {
        "dnb_expr": csr_matrix(data["dnb_expr"].astype(np.float64)),
        "dnb_coords": data["dnb_coords"],
        "dnb_labels": data["dnb_labels"],
        "gene_names": gene_names,
        "cell_ids": cell_ids,
        "true_expr": data["true_expr"],
        "gene_is_high": data["gene_is_high"],
        "params": data["params"],
    }


def _rmse(pred, true):
    """Root-mean-square error over all elements."""
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def run_synthetic_comparison(data, n_genes=500, methods=None):
    """Run methods on a synthetic scenario and report RMSE reduction vs raw."""
    if methods is None:
        methods = ["sparkle", "spatial_soupx", "soupx", "decontx"]
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

    # 1. SPARKLE
    if "sparkle" in methods:
        print(f"\n[SPARKLE]")
        t0 = time.time()
        model = SPARKLE(
            bin_size=25,
            distance_metric="exponential",
            max_radius=300.0,
            n_high_genes=min(80, dnb_expr.shape[0]),
            n_lambda_genes=min(50, dnb_expr.shape[0]),
            r2_threshold=0.01,
            lambda_grid=[10, 20, 30, 50, 70, 100, 150, 200, 300],
            use_local_density=False,
            cell_based=True,
            self_confidence_penalty=False,
            verbose=False,
        )
        sp_corr, diag = model.fit_transform_from_dnb(dnb_expr, dnb_coords, dnb_labels)
        sp_t = time.time() - t0
        if hasattr(sp_corr, "toarray"):
            sp_corr = sp_corr.toarray()
        rmse_sp = _rmse(sp_corr, true_expr)
        reduc_sp = (rmse_raw - rmse_sp) / rmse_raw * 100.0
        print(f"  RMSE={rmse_sp:.4f}, reduction={reduc_sp:.1f}%, "
              f"λ={model.lambda_:.0f}µm, time={sp_t:.1f}s")
        results["SPARKLE"] = {"rmse": rmse_sp, "reduction": reduc_sp, "runtime": sp_t}

    # 2. Spatial SoupX
    if "spatial_soupx" in methods:
        print(f"\n[Spatial SoupX]")
        t0 = time.time()
        ss_corr, ss_rho, ss_lam = run_spatial_soupx(
            dnb_expr, dnb_coords, dnb_labels,
            bin_size=25, max_radius=300.0,
            lambda_grid=[10, 20, 30, 50, 70, 100, 150, 200, 300],
            verbose=False,
        )
        ss_t = time.time() - t0
        rmse_ss = _rmse(ss_corr, true_expr)
        reduc_ss = (rmse_raw - rmse_ss) / rmse_raw * 100.0
        print(f"  RMSE={rmse_ss:.4f}, reduction={reduc_ss:.1f}%, "
              f"ρ={ss_rho:.4f}, λ={ss_lam:.0f}µm, time={ss_t:.1f}s")
        results["SpatialSoupX"] = {"rmse": rmse_ss, "reduction": reduc_ss, "runtime": ss_t}

    # 3. SoupX
    if "soupx" in methods:
        print(f"\n[SoupX]")
        t0 = time.time()
        try:
            sx_corr, sx_rho = run_soupx(dnb_expr, dnb_labels, verbose=False)
        except Exception as e:
            # Robust fallback for on-the-fly generated data where adjustCounts
            # can produce pathological negatives.
            print(f"  run_soupx failed ({e}), using simple global subtraction fallback")
            empty_mask = np.asarray(dnb_labels < 0).ravel()
            cell_mask = np.asarray(dnb_labels >= 0).ravel()
            dnb_dense = np.asarray(dnb_expr.todense() if hasattr(dnb_expr, "todense") else dnb_expr.toarray())
            total_per_gene = dnb_dense.sum(axis=1)
            n_top = max(5, dnb_dense.shape[0] // 5)
            top_genes = np.argsort(total_per_gene)[-n_top:]
            mean_cell = dnb_dense[top_genes][:, cell_mask].mean(axis=1)
            mean_empty = dnb_dense[top_genes][:, empty_mask].mean(axis=1)
            valid = mean_cell > 0.01
            ratios = mean_empty[valid] / (mean_cell[valid] + mean_empty[valid])
            sx_rho = float(np.median(ratios))
            sx_rho = max(0.001, min(sx_rho, 0.8))
            raw_cell_soupx = compute_cell_expr(dnb_expr, dnb_labels, n_cells)
            sx_corr = np.maximum(raw_cell_soupx * (1.0 - sx_rho), 0.0)
        sx_t = time.time() - t0
        rmse_sx = _rmse(sx_corr, true_expr)
        reduc_sx = (rmse_raw - rmse_sx) / rmse_raw * 100.0
        print(f"  RMSE={rmse_sx:.4f}, reduction={reduc_sx:.1f}%, "
              f"ρ={sx_rho:.4f}, time={sx_t:.1f}s")
        results["SoupX"] = {"rmse": rmse_sx, "reduction": reduc_sx, "runtime": sx_t}

    # 4. DecontX
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


def run_all_synthetic_scenarios(methods=None):
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
        summary = run_synthetic_comparison(data, n_genes=500, methods=methods)
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
    }


def run_axolotl_comparison(data, n_genes=200, methods=None):
    """Run 3-way comparison for Axolotl."""
    if methods is None:
        methods = ['sparkle', 'spatial_soupx', 'soupx']
    methods = [m.lower().strip() for m in methods]

    dnb_expr = data['dnb_expr']
    dnb_coords = data['dnb_coords']
    dnb_labels = data['dnb_labels']
    gene_names = data['gene_names']
    cell_ids = np.array(data['cell_ids'])
    sstin_set = data['sstin_set']
    sst_gene = "AMEX60DD003175"

    n_cells = len(cell_ids)
    sstin_mask = np.array([cell_ids[i] in sstin_set for i in range(n_cells)])

    # Cell centroids & neighbors
    cc = np.zeros((n_cells, 2))
    for c in range(n_cells):
        m = dnb_labels == cell_ids[c]
        if m.sum():
            cc[c] = dnb_coords[m].mean(axis=0)
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

    # 1. Cell SPARKLE
    if 'sparkle' in methods:
        print(f"\n[3/3] Cell SPARKLE (cell_based + penalty)...")
        t0 = time.time()
        sub_all = dnb_expr  # use all genes
        n_high = min(n_genes, sub_all.shape[0])
        model = SPARKLE(
            bin_size=50, max_radius=200, n_high_genes=n_high, n_lambda_genes=min(100, sub_all.shape[0]),
            r2_threshold=0.01, lambda_grid=[10, 20, 30, 50, 70, 100, 150, 200],
            use_local_density=False, cell_based=True, verbose=True,
        )
        sp_corr, _ = model.fit_transform_from_dnb(sub_all, dnb_coords, dnb_labels)
        sp_t = time.time() - t0
        if hasattr(sp_corr, 'toarray'):
            sp_corr = sp_corr.toarray()
        sst_summary(sp_corr[sst_idx_all], f"Cell SPARKLE ({sp_t:.0f}s)")
        results['SPARKLE'] = sp_corr[sst_idx_all]
        results['sp_full'] = sp_corr

    # 2. Spatial SoupX
    if 'spatial_soupx' in methods:
        print(f"\n[4/3] Spatial SoupX...")
        t0 = time.time()
        sub_n_d = dnb_expr[top_n, :]
        ss_corr, ss_rho, ss_lam = run_spatial_soupx(
            sub_n_d, dnb_coords, dnb_labels, bin_size=50, max_radius=200,
            lambda_grid=[10, 20, 30, 50, 70, 100, 150, 200]
        )
        ss_t = time.time() - t0
        sst_summary(ss_corr[sst_loc_n], f"Spatial SoupX ({ss_t:.0f}s, lambda={ss_lam:.0f}, rho={ss_rho:.4f})")
        results['SpatialSoupX'] = ss_corr[sst_loc_n]
        results['ss_full'] = ss_corr

    # 3. SoupX
    if 'soupx' in methods:
        print(f"\n[5/3] SoupX (top {n_genes} genes)...")
        t0 = time.time()
        sx_corr, sx_rho = run_soupx(dnb_expr[top_n, :], labels_0based, verbose=False)
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

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Method':<18} {'sstIN':>7} {'Nbr':>7} {'s/N':>7} {'Retain':>7} {'Remove':>7}")
    print(f"  {'-'*18} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
    r_si = sst_raw[sstin_mask].mean()
    r_sn = sst_raw[neighbor_mask].mean()
    print(f"  {'Raw':<18} {r_si:7.1f} {r_sn:7.1f} {r_si/r_sn:7.2f}x {'-':>7} {'-':>7}")

    for name, vals in results.items():
        if name.endswith('_full'):
            continue
        si = vals[sstin_mask].mean()
        sn = vals[neighbor_mask].mean()
        print(f"  {name:<18} {si:7.1f} {sn:7.1f} {si/sn:7.2f}x "
              f"{si/r_si*100:6.1f}% {(1-sn/r_sn)*100:6.1f}%")

    # ── scIB evaluation ──────────────────────────────────────
    ann_map = data.get('ann_map', {})
    cell_anns = np.array([ann_map.get(cid, 'Unknown') for cid in cell_ids])
    n_annotated = int((cell_anns != 'Unknown').sum())
    if n_annotated >= 10:
        print(f"\n{'='*60}")
        print(f"scIB EVALUATION ({n_annotated} annotated cells)")
        print(f"{'='*60}")
        _compute_scib_metrics(raw_cell, cell_anns, "RAW")
        # SPARKLE: already cell-level from fit_transform_from_dnb
        if 'sparkle' in methods and 'sp_full' in results:
            _compute_scib_metrics(results['sp_full'], cell_anns, "SPARKLE")
        # SpatialSoupX: cell-level from run_spatial_soupx
        if 'spatial_soupx' in methods and 'ss_full' in results:
            _compute_scib_metrics(results['ss_full'], cell_anns, "SpatialSoupX")
        # SoupX: needs alignment
        if 'soupx' in methods and 'sx_full' in results:
            sx_cell = results['sx_full']
            _compute_scib_metrics(sx_cell, cell_anns, "SoupX")


def run_mosta_comparison(data, sub, n_genes=200, methods=None, n_high_genes=None):
    """Run 3-way comparison for MOSTA with cortical layer evaluation."""
    if methods is None:
        methods = ['sparkle', 'spatial_soupx', 'soupx']
    methods = [m.lower().strip() for m in methods]
    if n_high_genes is None:
        n_high_genes = min(n_genes, 500)

    print(f"\n{'='*60}")
    print("RUNNING METHODS")
    print(f"{'='*60}")

    results = {}

    if 'sparkle' in methods:
        corrected, diag = run_sparkle_method(sub, n_high_genes=n_high_genes)
        if corrected is not None:
            results['SPARKLE'] = {'corrected': corrected, 'diag': diag}

    if 'spatial_soupx' in methods:
        corrected, diag = run_spatial_soupx_method(sub)
        results['SpatialSoupX'] = {'corrected': corrected, 'diag': diag}

    if 'soupx' in methods:
        corrected, diag = run_soupx_method(sub)
        if corrected is not None:
            results['SoupX'] = {'corrected': corrected, 'diag': diag}

    if 'decontx' in methods:
        corrected, diag = run_decontx_method(sub)
        if corrected is not None:
            results['DecontX'] = {'corrected': corrected, 'diag': diag}

    # Evaluation
    evaluate_mosta(results, data, sub, sub['gene_names'])

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Method':<16} {'Runtime':>8} {'DE':>8} {'Sprmn':>8} {'ASW↑':>6} {'cLISI↓':>6} {'Sil↑':>6} {'PCA5↑':>6} {'ClustCoef↑':>8}")
    print(f"  {'─'*16} {'─'*8} {'─'*8} {'─'*8} {'─'*6} {'─'*6} {'─'*6} {'─'*6} {'─'*8}")
    # RAW row (metrics from evaluate_mosta scIB section are on raw_cell_expr)
    print(f"  {'RAW':<16} {'─':>8} {'─':>8} {'─':>8} {'─':>6} {'─':>6} {'─':>6} {'─':>6} {'─':>8}")
    for method_name, r in results.items():
        d = r['diag']
        runtime = d.get('runtime', 0)
        de = r.get('mosta_de_genes', 'N/A')
        s = r.get('mosta_layer_spearman', r.get('mosta_between_corr', 'N/A'))
        asw = r.get('mosta_asw')
        clisi = r.get('mosta_clisi')
        sil = r.get('mosta_silhouette')
        pca5 = r.get('mosta_pca_var_top5')
        clust = r.get('mosta_avg_clust_coef')
        s_str = f"{s:.4f}" if isinstance(s, float) and not np.isnan(s) else str(s)
        asw_str = f"{asw:.4f}" if asw is not None else "N/A"
        clisi_str = f"{clisi:.4f}" if clisi is not None else "N/A"
        sil_str = f"{sil:.4f}" if sil is not None else "N/A"
        pca5_str = f"{pca5:.4f}" if pca5 is not None else "N/A"
        clust_str = f"{clust:.4f}" if clust is not None else "N/A"
        print(f"  {method_name:<16} {runtime:7.1f}s  {str(de):>8} {s_str:>8} {asw_str:>6} {clisi_str:>6} {sil_str:>6} {pca5_str:>6} {clust_str:>8}")

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


def run_sparkle_method(sub, verbose=True, bin_size_um=25.0, spot_pitch_um=0.5, n_high_genes=500):
    """运行 SPARKLE（本方法）。

    核心流程：
      1. 检查空 DNB 是否存在 → 无则报错退出（不降级到 bin-level）
      2. 提取细胞 DNB → cell areas
      3. 空 DNB 分箱 → empty bins
      4. λ 网格搜索 → 最优空间扩散距离
      5. 基因特异性 α 估计（加权 OLS，R² 阈值过滤）
      6. cell-level ambient 校正（含 self-confidence penalty）

    参数：
      bin_size_um=25.0     — 空 bin 目标尺寸（µm）
      spot_pitch_um=0.5     — DNB 间距（Stereo-seq: 0.5µm, Visium HD: 2µm）
      max_radius=300       — 空间邻域搜索半径（µm）
      r2_threshold=0.01    — α 估计的 R² 阈值
      lambda_grid          — λ 候选值列表（µm）
      cell_based=True      — 使用 cell-based pipeline

    输出 diagnostics 包含：lambda, n_genes_corrected, runtime 等。
    """
    bin_size = max(1, int(bin_size_um / spot_pitch_um + 0.5))  # rounds 25/2=12.5→13
    print(f"\n  SPARKLE (cell_based + penalty, bin={bin_size} DNBs ≈ {bin_size*spot_pitch_um:.0f}µm, n_high={n_high_genes})...")
    dnb_expr = sub['dnb_expr']
    dnb_coords = sub['dnb_coords']
    dnb_labels = sub['dnb_labels']

    # 空 DNB 检查：SPARKLE 必须有空 DNB 作为 ambient probe
    n_empty = int((dnb_labels < 0).sum())
    if n_empty == 0:
        print(f"    ERROR: No empty DNBs available. SPARKLE requires empty regions as probes.")
        return None, {'error': 'no_empty_dnbs', 'runtime': 0}

    model = SPARKLE(
        bin_size=bin_size, distance_metric="exponential", max_radius=300,
        n_high_genes=min(n_high_genes, dnb_expr.shape[0]),
        n_lambda_genes=min(100, dnb_expr.shape[0]),
        r2_threshold=0.01,
        lambda_grid=[10, 20, 30, 50, 70, 100, 150, 200, 300],
        use_local_density=False, cell_based=True, verbose=verbose,
    )
    t0 = time.time()
    corrected, diag = model.fit_transform_from_dnb(dnb_expr, dnb_coords, dnb_labels)
    elapsed = time.time() - t0
    if hasattr(corrected, 'toarray'):
        corrected = corrected.toarray()
    print(f"    Done in {elapsed:.1f}s, λ={model.lambda_:.0f}μm, "
          f"{diag.get('n_genes_corrected','?')} genes corrected")
    return corrected, {'lambda': float(model.lambda_), 'runtime': elapsed, **diag}


def run_spatial_soupx_method(sub, verbose=True, bin_size_um=25.0, spot_pitch_um=0.5):
    """运行 Spatial SoupX（SoupX + 空间核）。

    与原始 SoupX 的区别：
      - 使用 SPARKLE 的空间核函数 w(d) = exp(-d/λ) 计算距离衰减权重
      - 估计一个全局 ρ（而非 SPARKLE 的 per-gene α）
      - λ 通过网格搜索选择（最小化空 bin RSS）

    参数与 SPARKLE 保持一致的 bin_size_um、max_radius、lambda_grid。
    """
    bin_size = max(1, int(bin_size_um / spot_pitch_um + 0.5))  # rounds 25/2=12.5→13
    print(f"\n  Spatial SoupX (bin={bin_size} DNBs ≈ {bin_size*spot_pitch_um:.0f}µm)...")
    dnb_expr = sub['dnb_expr']
    dnb_coords = sub['dnb_coords']
    dnb_labels = sub['dnb_labels']

    t0 = time.time()
    corrected, rho, lam = run_spatial_soupx(
        dnb_expr, dnb_coords, dnb_labels,
        bin_size=bin_size, max_radius=300,
        lambda_grid=[10, 20, 30, 50, 70, 100, 150, 200, 300],
        verbose=verbose,
    )
    elapsed = time.time() - t0
    print(f"    Done in {elapsed:.1f}s, λ={lam:.0f}μm, ρ={rho:.4f}")
    return corrected, {'lambda': float(lam), 'rho': float(rho), 'runtime': elapsed}


def run_soupx_method(sub, verbose=True):
    """运行原始 SoupX（全局 ρ，无空间信息）。

    处理方式：
      1. DNBs 聚合为 per-cell 表达（toc）
      2. 空 DNBs 作为 "空液滴" 估计背景谱
      3. KMeans 聚类 → marker gene 检测 → autoEstCont 估计 ρ
      4. 如果 autoEstCont 失败（常见于合成数据和无 marker 的数据），
         回退到简单的 empty/cell 比值估计 ρ

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
    from decontx import decontx as run_dx
    print(f"\n  DecontX...")
    dnb_expr = sub['dnb_expr']
    dnb_labels = sub['dnb_labels']
    gene_names = list(sub['gene_names'])
    cell_ids = np.array(sub['cell_ids'])
    n_cells = len(cell_ids)

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
    print(f"    Done in {elapsed:.1f}s, mean contamination={contamination:.3f}")
    return corrected, {'runtime': elapsed, 'contamination': contamination}


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


def evaluate_mosta(results, data, sub, gene_names):
    """对每个方法计算层间 DE 基因数、异类细胞间相关性、scIB 指标。

    指标说明：
      DE genes:    层间差异表达基因数（t-test p<0.05，top 200 var genes）
      Pearson:     不同皮层层次细胞间的平均 Pearson 相关系数（log1p）
      Spearman:    同上，使用 Spearman rank 相关（对过度校正更鲁棒）
      ASW:         Cell-type silhouette width（越高越好）
      cLISI:       Cell-type local inverse Simpson's index（越低越好）

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
        return

    n_cells = len(cell_ids)
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

    # ── scIB evaluation (ASW + cLISI) ──────────────────────────
    print(f"\n  {'='*60}")
    print(f"  scIB Evaluation (Cell-type ASW ↑, cLISI ↓)")
    print(f"  {'='*60}")

    raw_metrics = _compute_scib_metrics(raw, cell_anns, "RAW")
    for method_name, r in results.items():
        corrected = r.get('corrected')
        if corrected is None:
            continue
        metrics = _compute_scib_metrics(corrected, cell_anns, method_name)
        r['mosta_asw'] = metrics.get('asw')
        r['mosta_clisi'] = metrics.get('clisi')
        r['mosta_silhouette'] = metrics.get('silhouette')
        r['mosta_pca_var_top5'] = metrics.get('pca_var_top5')
        r['mosta_avg_clust_coef'] = metrics.get('avg_clust_coef')



def load_visiumhd_data(x_range=None, y_range=None, n_genes=None, verbose=True):
    """Load Visium HD human colon cancer data at 2µm pixel resolution.

    Keeps native 2µm pixels as DNBs (matching Stereo-seq's 500nm DNB concept).
    SPARKLE bin_size=13 will produce ~26µm empty bins, comparable to
    Stereo-seq's bin_size=50 at 500nm = 25µm.

    Optimized: dict→array pixel lookup, COO accumulation, vectorized per-gene ops.

    Args:
        x_range: (x_min, x_max) spatial window in 2µm pixel column units.
        y_range: (y_min, y_max) spatial window in 2µm pixel row units.
        n_genes: if set, only load top-N genes by total expression (two-pass).

    Returns:
        dict with dnb_expr, dnb_coords, dnb_labels, gene_names, cell_ids, ann_map.
    """
    import h5py, csv

    data_dir = Path(__file__).resolve().parent.parent / "data" / "visiumhd"
    h5_path = data_dir / "Visium_HD_6p5mm_Human_Colon_Cancer_feature_slice.h5"
    rctd_csv = data_dir / "rctd_first_type.csv"

    if not h5_path.exists():
        raise FileNotFoundError(f"{h5_path} not found.")
    if not rctd_csv.exists():
        raise FileNotFoundError(f"{rctd_csv} not found. Run R conversion first.")

    load_all = False  # will be set in pass 1
    f = h5py.File(h5_path, 'r')

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
            dnb_coords_list.append((c * 8 + 4, r * 8 + 4))
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
            dnb_coords_list.append((c * 8 + 4, r * 8 + 4))

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

    # ── Step 5: Load RCTD annotations ─────────────────────────────
    if verbose:
        print("Loading RCTD cell type annotations...")
    ann_map = {}
    with open(rctd_csv, 'r') as cf:
        reader = csv.reader(cf)
        header = next(reader)
        for row in reader:
            barcode, first_type = row[0], row[1]
            try:
                cell_id = int(barcode.split('_')[1].split('-')[0])
                ann_map[cell_id] = first_type
            except (IndexError, ValueError):
                continue

    cell_anns = np.array([ann_map.get(cid, 'Unknown') for cid in cell_ids_list], dtype=object)
    n_annotated = int((cell_anns != 'Unknown').sum())
    if verbose:
        print(f"  {n_annotated}/{n_cells} cells have RCTD annotations")
        unique_types = sorted(set(cell_anns[cell_anns != 'Unknown']))
        print(f"  Cell types: {unique_types}")

    cell_ids_arr = np.array(cell_ids_list)
    ann_map_out = {cid: ann_map.get(cid, 'Unknown') for cid in cell_ids_list}

    if verbose:
        print(f"\n{'='*60}")
        print(f"Visium HD data loaded: {n_genes} genes, {n_pixels} 2µm pixels, {n_cells} cells")
        print(f"{'='*60}")

    return {
        'dnb_expr': dnb_expr,
        'dnb_coords': dnb_coords,
        'dnb_labels': dnb_labels,
        'gene_names': gene_names_arr,
        'cell_ids': cell_ids_arr,
        'ann_map': ann_map_out,
        'adata': None,
    }

def _compute_scib_metrics(cell_expr, cell_anns, method_name, n_top_genes=2000):
    """Compute Cell-type ASW and cLISI via scib-metrics for a given expression matrix.

    Args:
        cell_expr: [genes × cells] array
        cell_anns: [cells] cell type annotations
        method_name: label for printing
        n_top_genes: number of HVGs to use

    Returns:
        dict with 'asw', 'clisi', 'silhouette' scores (or None on failure)
    """
    import scanpy as sc

    # Filter to cells with known annotations
    valid = cell_anns != 'Unknown'
    if valid.sum() < 10:
        print(f"  {method_name}: Too few annotated cells ({valid.sum()}), skipping scIB")
        return {'asw': None, 'clisi': None, 'silhouette': None}

    expr = cell_expr[:, valid]
    anns = cell_anns[valid]

    try:
        adata = sc.AnnData(X=expr.T, dtype=np.float64)
        adata.obs['cell_type'] = list(anns)
        adata.obs_names = [f"Cell_{i}" for i in range(adata.n_obs)]
        adata.var_names = [f"Gene_{i}" for i in range(adata.n_vars)]

        # Preprocessing: normalize, log1p, HVGs, PCA, neighbors
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        try:
            sc.pp.highly_variable_genes(adata, n_top_genes=min(n_top_genes, adata.n_vars),
                                         flavor='seurat_v3')
        except Exception:
            sc.pp.highly_variable_genes(adata, n_top_genes=min(n_top_genes, adata.n_vars),
                                         flavor='seurat')
        adata = adata[:, adata.var.highly_variable].copy()
        sc.pp.scale(adata, max_value=10)
        n_pcs = min(50, adata.n_obs - 1, adata.n_vars - 1)
        sc.tl.pca(adata, n_comps=n_pcs, svd_solver='arpack')

        # Compute scIB metrics via scib_metrics package
        from scib_metrics import silhouette_label, clisi_knn
        from scib_metrics.nearest_neighbors import NeighborsResults as NR
        from sklearn.neighbors import NearestNeighbors

        labels_arr = adata.obs['cell_type'].values.astype(str)
        X_pca = adata.obsm['X_pca']

        # Cell-type ASW (silhouette width by cell type)
        asw = silhouette_label(X_pca, labels_arr)

        # cLISI: build nearest neighbors
        n_nbrs_knn = min(15, X_pca.shape[0])
        nn = NearestNeighbors(n_neighbors=n_nbrs_knn)
        nn.fit(X_pca)
        distances, indices = nn.kneighbors(X_pca)
        nbrs = NR(distances=distances, indices=indices)
        clisi = clisi_knn(nbrs, labels_arr)

        # Raw Silhouette Score (sklearn)
        from sklearn.metrics import silhouette_score
        sil = silhouette_score(X_pca, labels_arr)

        # PCA explained variance (top 5 PCs)
        var_ratio = adata.uns['pca']['variance_ratio']
        pca_var_top5 = float(np.sum(var_ratio[:5]))

        # Average Clustering Coefficient from KNN graph
        n_nodes = indices.shape[0]
        # Build set of neighbor sets for fast lookup
        neighbor_sets = [set(indices[i]) for i in range(n_nodes)]
        clust_coeffs = np.zeros(n_nodes)
        for i in range(n_nodes):
            nbrs_i = neighbor_sets[i]
            k = len(nbrs_i)
            if k < 2:
                continue
            edges = 0
            for j in nbrs_i:
                if j == i:
                    continue
                edges += len(nbrs_i & neighbor_sets[j])
            edges //= 2  # each edge counted twice
            clust_coeffs[i] = (2.0 * edges) / (k * (k - 1))
        avg_clust = float(np.mean(clust_coeffs[clust_coeffs > 0])) if (clust_coeffs > 0).any() else 0.0

        print(f"  {method_name}: ASW={asw:.4f}, cLISI={clisi:.4f}, Silhouette={sil:.4f}, "
              f"PCA5={pca_var_top5:.4f}, ClustCoef={avg_clust:.4f}")
        return {'asw': float(asw), 'clisi': float(clisi), 'silhouette': float(sil),
                'pca_var_top5': pca_var_top5, 'avg_clust_coef': avg_clust}
    except Exception as e:
        print(f"  {method_name}: scIB failed: {e}")
        return {'asw': None, 'clisi': None, 'silhouette': None,
                'pca_var_top5': None, 'avg_clust_coef': None}


def run_visiumhd_comparison(data, sub, n_genes=200, methods=None, n_high_genes=None):
    """Run comparison for Visium HD with scIB metrics evaluation.

    Evaluates: Cell-type ASW (higher=better separation) and
    cLISI (lower=better integration within cell types).
    """
    if methods is None:
        methods = ['sparkle', 'spatial_soupx', 'soupx']
    methods = [m.lower().strip() for m in methods]
    if n_high_genes is None:
        n_high_genes = min(n_genes, 500)

    print(f"\n{'='*60}")
    print("RUNNING METHODS")
    print(f"{'='*60}")

    results = {}

    if 'sparkle' in methods:
        corrected, diag = run_sparkle_method(sub, spot_pitch_um=2.0, n_high_genes=n_high_genes)
        if corrected is not None:
            results['SPARKLE'] = {'corrected': corrected, 'diag': diag}

    if 'spatial_soupx' in methods:
        corrected, diag = run_spatial_soupx_method(sub, spot_pitch_um=2.0)
        results['SpatialSoupX'] = {'corrected': corrected, 'diag': diag}

    if 'soupx' in methods:
        corrected, diag = run_soupx_method(sub)
        if corrected is not None:
            results['SoupX'] = {'corrected': corrected, 'diag': diag}

    if 'decontx' in methods:
        corrected, diag = run_decontx_method(sub)
        if corrected is not None:
            results['DecontX'] = {'corrected': corrected, 'diag': diag}

    # Evaluation: scIB metrics
    print(f"\n{'='*60}")
    print("scIB EVALUATION (Cell-type ASW ↑, cLISI ↓)")
    print(f"{'='*60}")

    ann_map = data.get('ann_map', {})
    cell_ids = data.get('cell_ids', np.array([]))
    cell_anns = np.array([ann_map.get(cid, 'Unknown') for cid in cell_ids])
    n_annotated = int((cell_anns != 'Unknown').sum())
    print(f"  {n_annotated}/{len(cell_ids)} cells have cell type annotations")

    if n_annotated < 10:
        print("  Too few annotated cells, skipping scIB evaluation.")
        return

    if len(cell_ids) == 0:
        print("  No cells in data (empty spatial window?), skipping.")
        return

    # Compute raw cell expression
    n_cells = len(cell_ids)
    raw_cell = compute_cell_expr(sub['dnb_expr'], sub['dnb_labels'], n_cells)
    raw_metrics = _compute_scib_metrics(raw_cell, cell_anns, "RAW")

    for method_name, r in results.items():
        corrected = r.get('corrected')
        if corrected is None:
            continue
        # Align: SPARKLE may drop cells with no bins after subsampling
        n_cells_corr = corrected.shape[1]
        if n_cells_corr != len(cell_anns):
            print(f"  Warning: aligning cell annotations ({n_cells_corr} vs {len(cell_anns)})")
            cell_anns_use = cell_anns[:n_cells_corr]
        else:
            cell_anns_use = cell_anns
        metrics = _compute_scib_metrics(corrected, cell_anns_use, method_name)
        r['scib_asw'] = metrics.get('asw')
        r['scib_clisi'] = metrics.get('clisi')
        r['scib_silhouette'] = metrics.get('silhouette')
        r['scib_pca_var_top5'] = metrics.get('pca_var_top5')
        r['scib_avg_clust_coef'] = metrics.get('avg_clust_coef')

    # ── Summary ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Method':<16} {'Runtime':>8} {'ASW↑':>8} {'cLISI↓':>8} {'Sil↑':>8} {'PCA5↑':>8} {'ClustCoef↑':>10}")
    print(f"  {'─'*16} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*10}")

    raw_asw_str = f"{raw_metrics['asw']:.4f}" if raw_metrics['asw'] is not None else "N/A"
    raw_clisi_str = f"{raw_metrics['clisi']:.4f}" if raw_metrics['clisi'] is not None else "N/A"
    raw_sil_str = f"{raw_metrics['silhouette']:.4f}" if raw_metrics.get('silhouette') is not None else "N/A"
    raw_pca5_str = f"{raw_metrics['pca_var_top5']:.4f}" if raw_metrics.get('pca_var_top5') is not None else "N/A"
    raw_clust_str = f"{raw_metrics['avg_clust_coef']:.4f}" if raw_metrics.get('avg_clust_coef') is not None else "N/A"
    print(f"  {'RAW':<16} {'':>8} {raw_asw_str:>8} {raw_clisi_str:>8} {raw_sil_str:>8} {raw_pca5_str:>8} {raw_clust_str:>10}")

    for method_name, r in results.items():
        d = r['diag']
        runtime = d.get('runtime', 0)
        asw = r.get('scib_asw')
        clisi = r.get('scib_clisi')
        sil = r.get('scib_silhouette')
        pca5 = r.get('scib_pca_var_top5')
        clust = r.get('scib_avg_clust_coef')
        asw_str = f"{asw:.4f}" if asw is not None else "N/A"
        clisi_str = f"{clisi:.4f}" if clisi is not None else "N/A"
        sil_str = f"{sil:.4f}" if sil is not None else "N/A"
        pca5_str = f"{pca5:.4f}" if pca5 is not None else "N/A"
        clust_str = f"{clust:.4f}" if clust is not None else "N/A"
        print(f"  {method_name:<16} {runtime:7.1f}s {asw_str:>8} {clisi_str:>8} {sil_str:>8} {pca5_str:>8} {clust_str:>10}")


def main():
    parser = argparse.ArgumentParser(
        description="Final comparison: Cell SPARKLE vs Spatial SoupX vs SoupX")
    parser.add_argument("--dataset", type=str, default="axolotl",
                        choices=["axolotl", "mosta", "visiumhd", "synthetic"],
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
    parser.add_argument("--methods", type=str,
                        default="sparkle,spatial_soupx,soupx,decontx",
                        help="Comma-separated methods to run")
    args = parser.parse_args()

    x_range = tuple(args.x_range) if args.x_range else None
    y_range = tuple(args.y_range) if args.y_range else None
    methods = [m.strip().lower() for m in args.methods.split(',')]

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
        run_axolotl_comparison(data, args.n_genes, methods)
    elif args.dataset == "mosta":
        data = load_mosta_data(x_range=x_range, y_range=y_range)
        sub = subsample_data(data, args.n_genes, cut_genes=False)
        run_mosta_comparison(data, sub, args.n_genes, methods, n_high_genes=args.n_high_genes)
    elif args.dataset == "visiumhd":
        data = load_visiumhd_data(x_range=x_range, y_range=y_range, n_genes=args.n_genes)
        if data is None:
            sys.exit(1)
        sub = subsample_data(data, args.n_genes)
        run_visiumhd_comparison(data, sub, args.n_genes, methods, n_high_genes=args.n_high_genes)
    else:  # synthetic
        if args.all_scenarios:
            run_all_synthetic_scenarios(methods)
        else:
            data = load_synthetic_scenario_data(args.scenario, seed=42)
            run_synthetic_comparison(data, args.n_genes, methods)


if __name__ == "__main__":
    main()
