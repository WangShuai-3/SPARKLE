#!/usr/bin/env python3
"""Real-data test: SPARKLE on axolotl telencephalon Stereo-seq data.

Uses gem.gz (all DNBs) + scgem (cell labels) to build proper DNB matrix
with real empty DNBs from inter-cellular space.

Usage:
    python evaluation/scripts/test_axolotl.py
"""

import os
import sys
import time
import gzip
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from scipy.sparse import lil_matrix, csr_matrix
from scipy.spatial import cKDTree
import anndata as ad

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from stambient import SPARKLE


def load_scgem_label_map(scgem_path):
    """Build (x,y) → cell_id mapping from scgem (one pass)."""
    print("  Building cell label map from scgem...")
    label_map = {}
    with gzip.open(scgem_path, 'rt') as f:
        f.readline()  # header
        for i, line in enumerate(f):
            parts = line.strip().split(',')
            x, y, cell = int(parts[0]), int(parts[1]), int(parts[4])
            label_map[(x, y)] = cell
            if (i + 1) % 5000000 == 0:
                print(f"    Mapped {i+1:,} rows...")
    print(f"    Total: {len(label_map):,} unique DNB positions with cell labels")
    return label_map


def load_gem_chunked(filepath, nrows_per_chunk=500000):
    """Generator: yield chunks of gem TSV."""
    with gzip.open(filepath, 'rt') as f:
        header = f.readline().strip()
        colnames = header.split('\t')
        chunk_rows = []
        for line in f:
            chunk_rows.append(line.strip().split('\t'))
            if len(chunk_rows) >= nrows_per_chunk:
                yield pd.DataFrame(chunk_rows, columns=colnames)
                chunk_rows = []
        if chunk_rows:
            yield pd.DataFrame(chunk_rows, columns=colnames)


def build_dnb_matrix_from_gem(gem_path, label_map):
    """Build genes×DNBs sparse matrix from gem with scgem labels.

    Returns:
        dnb_expr: [n_genes × n_dnbs] CSR sparse float64 matrix.
        dnb_coords: [n_dnbs × 2] float64.
        dnb_labels: [n_dnbs] int64, cell ID (0-based), -1 for empty.
        gene_names: list of gene names.
        cell_ids: list of original cell IDs.
    """
    print("Pass 1: scanning gene & DNB vocabulary from gem...")
    gene_set = set()
    dnb_set = set()

    for chunk in load_gem_chunked(gem_path):
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

    # Collect all unique cell IDs from label_map
    all_cells = sorted(set(label_map.values()))
    cell_to_idx = {c: i for i, c in enumerate(all_cells)}
    n_cells = len(all_cells)

    n_matched = sum(1 for d in dnb_list if d in label_map)
    n_empty = n_dnbs - n_matched
    print(f"  Final: {n_genes} genes, {n_dnbs} DNBs "
          f"({n_matched} cell, {n_empty} empty), {n_cells} cells")

    # DNB coordinates and labels
    dnb_coords = np.array(dnb_list, dtype=np.float64)
    dnb_labels = np.full(n_dnbs, -1, dtype=np.int64)
    for i, d in enumerate(dnb_list):
        if d in label_map:
            dnb_labels[i] = cell_to_idx[label_map[d]]

    # Build sparse matrix
    print("Pass 2: building sparse matrix from gem...")
    dnb_expr = lil_matrix((n_genes, n_dnbs), dtype=np.float64)
    total_rows = 0

    for chunk in load_gem_chunked(gem_path):
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


def compute_neighbor_stats(cell_centroids, sstin_mask, radius=50.0):
    """Find non-sstIN neighbors within radius of any sstIN cell."""
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


def main():
    project_root = Path(__file__).resolve().parent.parent.parent
    data_dir = project_root / "evaluation" / "data" / "axolotl"
    report_dir = project_root / "evaluation" / "reports" / "axolotl"
    os.makedirs(report_dir, exist_ok=True)

    sst_gene = "AMEX60DD003175"
    n_top_genes = 50

    # ── 1. Load h5ad annotations ─────────────────────────────────
    print("Loading H5ad annotations...")
    adata = ad.read_h5ad(data_dir / "Adult.h5ad")
    ann_map = dict(zip(adata.obs['cell_id'].values, adata.obs['Annotation'].values))
    sstin_cell_orig = set(adata.obs.loc[
        adata.obs['Annotation'] == 'sstIN', 'cell_id'
    ].values)
    print(f"  sstIN cells in h5ad: {len(sstin_cell_orig)}")

    # ── 2. Build cell label map from scgem ───────────────────────
    label_map = load_scgem_label_map(str(data_dir / "Adult_scgem.csv.gz"))

    # ── 3. Build DNB matrix from gem ─────────────────────────────
    print("\nBuilding DNB matrix from gem...")
    t0 = time.time()
    dnb_expr, dnb_coords, dnb_labels, gene_names, cell_ids = build_dnb_matrix_from_gem(
        str(data_dir / "Adult.gem.gz"), label_map
    )
    print(f"  Built in {time.time() - t0:.1f}s")
    print(f"  dnb_expr: {dnb_expr.shape}")
    n_empty = int((dnb_labels < 0).sum())
    n_cell_dnb = int((dnb_labels >= 0).sum())
    print(f"  Empty DNBs: {n_empty} ({100*n_empty/len(dnb_labels):.1f}%)")
    print(f"  Cell DNBs:  {n_cell_dnb} ({100*n_cell_dnb/len(dnb_labels):.1f}%)")

    # ── 4. Map cells to annotations ──────────────────────────────
    print("\nMapping cell annotations...")
    n_cells = len(cell_ids)
    annotations = np.array([
        ann_map.get(cell_ids[i], "unknown") for i in range(n_cells)
    ])
    sstin_mask = np.array([
        cell_ids[i] in sstin_cell_orig for i in range(n_cells)
    ])
    print(f"  Total cells: {n_cells}, sstIN: {sstin_mask.sum()}")

    # ── 5. Select genes ──────────────────────────────────────────
    print(f"\nSelecting top {n_top_genes} genes + {sst_gene}...")
    total_per_gene = np.array(dnb_expr.sum(axis=1)).ravel()
    top_indices = np.argsort(total_per_gene)[::-1]

    gene_indices = []
    for idx in top_indices:
        if len(gene_indices) >= n_top_genes:
            break
        gene_indices.append(idx)

    try:
        sst_idx = gene_names.index(sst_gene)
    except ValueError:
        print(f"  ERROR: {sst_gene} not found!")
        return
    if sst_idx not in gene_indices:
        gene_indices.append(sst_idx)

    print(f"  Selected {len(gene_indices)} genes")
    print(f"  SST gene index: {sst_idx}, rank: {np.where(top_indices == sst_idx)[0][0] + 1}")

    # ── 6. Cell centroids & classification ───────────────────────
    print("\nComputing cell centroids...")
    cell_centroids = np.zeros((n_cells, 2))
    dnb_labels_arr = np.asarray(dnb_labels).ravel()
    for c in range(n_cells):
        mask = dnb_labels_arr == c
        if mask.sum() > 0:
            cell_centroids[c] = dnb_coords[mask].mean(axis=0)

    print("Classifying cells (sstIN / neighbor / other)...")
    neighbor_mask, other_mask = compute_neighbor_stats(
        cell_centroids, sstin_mask, radius=50.0
    )
    print(f"  sstIN: {sstin_mask.sum()}, Neighbors: {neighbor_mask.sum()}, "
          f"Other: {other_mask.sum()}")

    # ── 7. Raw per-cell expression ───────────────────────────────
    print("\nBuilding raw per-cell expression...")
    valid_mask = dnb_labels_arr >= 0
    dnb_idx = np.where(valid_mask)[0]
    cell_idx = dnb_labels_arr[valid_mask]
    C = csr_matrix(
        (np.ones(len(dnb_idx)), (dnb_idx, cell_idx)),
        shape=(dnb_expr.shape[1], n_cells)
    )
    raw_cell_expr = (dnb_expr @ C).toarray()
    print(f"  raw_cell_expr: {raw_cell_expr.shape}")

    # Per-DNB rate
    cell_dnb_counts = np.bincount(dnb_labels_arr[valid_mask], minlength=n_cells)
    per_dnb_raw = np.zeros_like(raw_cell_expr)
    for c in range(n_cells):
        if cell_dnb_counts[c] > 0:
            per_dnb_raw[:, c] = raw_cell_expr[:, c] / cell_dnb_counts[c]

    # ── 8. SPARKLE ───────────────────────────────────────────────
    print("\nRunning SPARKLE...")
    t0 = time.time()
    sub_expr = dnb_expr[gene_indices, :]

    model = SPARKLE(
        bin_size=50,
        distance_metric="exponential",
        max_radius=200.0,
        n_high_genes=len(gene_indices),
        n_lambda_genes=min(30, len(gene_indices)),
        r2_threshold=0.01,
        lambda_grid=[10, 20, 30, 50, 70, 100, 150, 200, 300],
        use_local_density=False,
        cell_based=True,
        use_expr_weight=False,
        verbose=True,
    )

    corrected, diagnostics = model.fit_transform_from_dnb(
        sub_expr, dnb_coords, dnb_labels_arr
    )
    elapsed = time.time() - t0
    print(f"\nSPARKLE completed in {elapsed:.1f}s")
    print(f"  Estimated λ: {model.lambda_:.1f} μm")
    print(f"  Genes corrected: {diagnostics['n_genes_corrected']}/{len(gene_indices)}")

    # ── 9. SST diffusion analysis ────────────────────────────────
    print("\n" + "="*70)
    print("SST DIFFUSION ANALYSIS")
    print("="*70)

    sst_local_idx = gene_indices.index(sst_idx)
    if hasattr(corrected, 'toarray'):
        corrected_dense = corrected.toarray()
    else:
        corrected_dense = np.asarray(corrected)

    n_cells_corrected = corrected_dense.shape[1]
    sstin_mask_c = sstin_mask[:n_cells_corrected]
    neighbor_mask_c = neighbor_mask[:n_cells_corrected]
    other_mask_c = other_mask[:n_cells_corrected]

    sst_raw = raw_cell_expr[sst_idx][:n_cells_corrected]
    sst_per_dnb = per_dnb_raw[sst_idx][:n_cells_corrected]
    sst_corrected = corrected_dense[sst_local_idx]

    def group_stats(vals, mask, name):
        if mask.sum() == 0:
            print(f"  {name}: (no cells)")
            return 0
        m = vals[mask].mean()
        med = np.median(vals[mask])
        print(f"  {name} (n={mask.sum()}): mean={m:.4f}, median={med:.4f}")
        return m

    print(f"\n{'─'*50}")
    print("Raw SST expression (total per cell):")
    rm_sstin = group_stats(sst_raw, sstin_mask_c, "sstIN")
    rm_neighbor = group_stats(sst_raw, neighbor_mask_c, "Neighbor")
    rm_other = group_stats(sst_raw, other_mask_c, "Other")
    if rm_other > 0:
        print(f"  Neighbor/Other ratio: {rm_neighbor/rm_other:.2f}x")
        print(f"  sstIN/Neighbor ratio: {rm_sstin/rm_neighbor:.2f}x")

    print(f"\n{'─'*50}")
    print("SPARKLE-corrected SST expression (total per cell):")
    cm_sstin = group_stats(sst_corrected, sstin_mask_c, "sstIN")
    cm_neighbor = group_stats(sst_corrected, neighbor_mask_c, "Neighbor")
    cm_other = group_stats(sst_corrected, other_mask_c, "Other")
    if cm_other > 0:
        print(f"  Neighbor/Other ratio: {cm_neighbor/cm_other:.2f}x")
        print(f"  sstIN/Neighbor ratio: {cm_sstin/cm_neighbor:.2f}x")

    print(f"\n{'─'*50}")
    print("Correction effect on SST:")
    print(f"  sstIN retained:     {cm_sstin/rm_sstin*100:.1f}%" if rm_sstin > 0 else "  N/A")
    print(f"  Neighbor retained:  {cm_neighbor/rm_neighbor*100:.1f}%" if rm_neighbor > 0 else "  N/A")
    print(f"  Other retained:     {cm_other/rm_other*100:.1f}%" if rm_other > 0 else "  N/A")

    if rm_other > 0 and cm_other > 0:
        raw_nb_excess = (rm_neighbor - rm_other) / rm_other * 100
        corr_nb_excess = (cm_neighbor - cm_other) / cm_other * 100 if cm_other > 0 else 0
        print(f"\n  Neighbor excess over Other: {raw_nb_excess:.0f}% → {corr_nb_excess:.0f}%")

    if rm_sstin > 0 and rm_neighbor > 0:
        raw_ratio = rm_sstin / rm_neighbor
        corr_ratio = cm_sstin / cm_neighbor if cm_neighbor > 0 else float('inf')
        print(f"  sstIN/Neighbor ratio: {raw_ratio:.2f} → {corr_ratio:.2f}")

    print(f"\n{'─'*50}")
    print("Per-DNB SST rate:")
    pd_sstin = group_stats(sst_per_dnb, sstin_mask_c, "sstIN")
    pd_neighbor = group_stats(sst_per_dnb, neighbor_mask_c, "Neighbor")
    pd_other = group_stats(sst_per_dnb, other_mask_c, "Other")
    if pd_other > 0:
        print(f"  Neighbor excess over Other: {pd_neighbor - pd_other:.4f} per DNB")

    print(f"\nReport saved to: {report_dir}")


if __name__ == "__main__":
    main()
