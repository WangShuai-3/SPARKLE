"""SoupX baseline for SPARKLE evaluation.

Adapts SoupX (ambient RNA removal for scRNA-seq) to spatial transcriptomics
by treating DNBs as droplets and aggregating to cell-level expression.

Reference: Young & Behjati (2020), "SoupX removes ambient RNA contamination
from droplet-based single-cell RNA sequencing data", GigaScience.
"""

import numpy as np
from scipy.sparse import csr_matrix
from typing import Tuple

from soupx import SoupChannel, autoEstCont, adjustCounts


def _estimate_rho_simple(dnb_expr, dnb_labels):
    """Fallback: estimate contamination ρ from empty-DNB / cell-DNB ratio.

    Uses only high-count genes to avoid noise-dominated ratios.
    ρ ≈ median_g(mean_empty_expr[g] / mean_cell_expr[g]) for top 20% genes by total counts.
    """
    empty_mask = dnb_labels < 0
    cell_mask = dnb_labels >= 0

    if not empty_mask.any() or not cell_mask.any():
        return 0.01

    n_genes = dnb_expr.shape[0]
    total_per_gene = np.asarray(dnb_expr.sum(axis=1)).ravel()

    # Select top 20% genes by total expression (high-expr + ambient-rich)
    n_top = max(5, n_genes // 5)
    top_genes = np.argsort(total_per_gene)[-n_top:]

    dnb_dense = np.asarray(dnb_expr.todense()) if hasattr(dnb_expr, 'todense') else dnb_expr.toarray()
    mean_cell = dnb_dense[top_genes][:, cell_mask].mean(axis=1)
    mean_empty = dnb_dense[top_genes][:, empty_mask].mean(axis=1)

    # ρ = empty / (cell + empty) avoids division by zero
    valid = mean_cell > 0.01
    if valid.sum() < 3:
        return 0.01

    ratios = mean_empty[valid] / (mean_cell[valid] + mean_empty[valid])
    rho = float(np.median(ratios))
    return max(0.001, min(rho, 0.8))


def run_soupx(
    dnb_expr: np.ndarray,
    dnb_labels: np.ndarray,
    n_clusters: int = None,
    contamination_range: tuple = (0.01, 0.8),
    tfidf_min: float = 1.0,
    soup_quantile: float = 0.90,
    random_state: int = 42,
    verbose: bool = False,
) -> Tuple[np.ndarray, float]:
    """Run SoupX on spatial DNB data.

    Treats DNBs as droplets: empty DNBs (label=-1) contribute to ambient
    profile estimation; cell DNBs are aggregated to per-cell expression
    and corrected.

    Falls back to a simple ρ estimate from empty/cell ratio if autoEstCont
    fails due to insufficient marker genes (common in synthetic data).

    Args:
        dnb_expr: [genes × DNBs] DNB-level expression matrix.
        dnb_labels: [DNBs] cell IDs; -1 for empty.
        n_clusters: Number of clusters for marker gene detection.
            Default: max(3, n_cells // 20).
        contamination_range: (min, max) bounds for contamination fraction ρ.
        tfidf_min: Minimum TF-IDF score for marker genes.
        soup_quantile: Quantile threshold for soup profile filtering.
        random_state: Seed for KMeans clustering.
        verbose: Print SoupX progress.

    Returns:
        corrected_expr: [genes × n_cells] corrected per-cell expression.
        rho: global contamination fraction estimate.
    """
    n_genes, n_dnbs = dnb_expr.shape
    cell_ids = np.unique(dnb_labels[dnb_labels >= 0])
    n_cells = len(cell_ids)

    if n_clusters is None:
        n_clusters = max(3, n_cells // 20)

    # ── 1. Build tod: all DNBs as "droplets" ──────────────────────
    tod = csr_matrix(dnb_expr.astype(np.float64))

    # ── 2. Build toc: per-cell aggregated expression (sparse MM, O(nnz)) ──
    cell_mask = dnb_labels >= 0
    cell_dnb_idx = np.where(cell_mask)[0]
    cell_indices = dnb_labels[cell_dnb_idx]
    C = csr_matrix(
        (np.ones(len(cell_dnb_idx), dtype=np.float64),
         (cell_dnb_idx, cell_indices)),
        shape=(n_dnbs, n_cells)
    )
    toc_dense = np.asarray((dnb_expr @ C).todense()) if hasattr(dnb_expr @ C, 'todense') else (dnb_expr @ C).toarray()
    toc = csr_matrix(toc_dense)

    # ── 3. Setup SoupChannel ───────────────────────────────────────
    gene_names = [f"gene_{i}" for i in range(n_genes)]
    sc = SoupChannel(tod, toc, gene_names=gene_names, calcSoupProfile=True)

    # ── 4. Clustering ──────────────────────────────────────────────
    from sklearn.cluster import KMeans
    cell_expr_for_cluster = np.log1p(toc_dense.T)
    clusters = KMeans(
        n_clusters=n_clusters, random_state=random_state, n_init=10
    ).fit(cell_expr_for_cluster).labels_
    sc.setClusters(clusters)

    # ── 5. Estimate contamination fraction ──────────────────────────
    try:
        sc = autoEstCont(
            sc,
            tfidfMin=tfidf_min,
            soupQuantile=soup_quantile,
            contaminationRange=contamination_range,
            verbose=verbose,
        )
        rho = float(sc.metaData['rho'].iloc[0])
    except (ValueError, KeyError) as e:
        # Fallback: simple ρ estimation from empty DNB ratio
        if verbose:
            print(f"  autoEstCont failed ({e}), using simple ρ estimate")
        rho = _estimate_rho_simple(dnb_expr, dnb_labels)
        sc.set_contamination_fraction(rho, forceAccept=True)

    # ── 6. Correct ─────────────────────────────────────────────────
    try:
        corrected_sparse = adjustCounts(sc, roundToInt=False, verbose=int(verbose))
        corrected_dense = corrected_sparse.toarray()
    except Exception as e:
        # Fallback: simple global subtraction using estimated ρ
        if verbose:
            print(f"  adjustCounts failed ({e}), using simple global subtraction")
        corrected_dense = np.maximum(toc_dense * (1.0 - rho), 0.0)

    return corrected_dense, rho
