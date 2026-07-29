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

    No heuristic fallback is applied: if autoEstCont or adjustCounts fails,
    a RuntimeError is raised so the caller records the failure explicitly
    (NaN metrics) instead of silently substituting a fabricated correction.

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

    Raises:
        RuntimeError: If contamination estimation or count adjustment fails.
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
        raise RuntimeError(
            f"SoupX autoEstCont failed (insufficient marker genes?): {e}"
        ) from e

    # ── 6. Correct ─────────────────────────────────────────────────
    try:
        corrected_sparse = adjustCounts(sc, roundToInt=False, verbose=int(verbose))
        corrected_dense = corrected_sparse.toarray()
    except Exception as e:
        raise RuntimeError(f"SoupX adjustCounts failed: {e}") from e

    # Clip negative values that can arise from soupx cluster expansion numerical errors.
    corrected_dense = np.maximum(corrected_dense, 0.0)

    return corrected_dense, rho
