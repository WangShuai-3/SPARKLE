"""SPARKLE main class: pipeline orchestration."""

import numpy as np
from scipy.sparse import csr_matrix, issparse
from typing import Optional, Dict, Tuple, List, Any

from .binning import dnb_to_bins
from .spatial import (
    build_spatial_graph,
    compute_local_density,
    DistanceMetric,
)
from .estimation import (
    select_high_expression_genes,
    estimate_lambda_grid_search,
    estimate_alpha_per_gene,
)
from .correction import correct_expression
from .cell_pipeline import cell_pipeline_fit
from .diagnostics import compute_diagnostics, print_diagnostics
from .io_utils import check_inputs


class SPARKLE:
    """Spatial Ambient RNA Kernel-based Leakage Estimator.

    Removes ambient RNA contamination from high-resolution spatial
    transcriptomics data using empty spots as built-in ambient probes.

    Parameters
    ----------
    bin_size : int
        Number of DNBs per bin edge for spatial binning (default 50).
    distance_metric : str
        Weight function: 'exponential', 'gaussian', or 'inverse'.
    lambda_distance : float or None
        Distance decay parameter (μm). Auto-estimated if None.
    max_radius : float
        Maximum neighbor search radius (μm).
    n_high_genes : int or None
        Number of top highly-expressed genes to correct. If None, all
        genes are used.
    n_lambda_genes : int
        Number of top genes for λ estimation.
    r2_threshold : float
        Minimum weighted R² to apply correction.
    empty_purity : float
        Minimum empty DNB fraction for "pure empty" bin.
    cell_purity : float
        Minimum single-cell DNB fraction for "pure cell" bin.
    lambda_grid : list of float
        λ values (μm) for grid search.
    use_local_density : bool
        Whether to apply local density modulation (β).
        Default False: β dampens correction and reduces RMSE by 50-80%
        in benchmarks. Enable only for very dense tissue with
        suspected ambient over-estimation.
    use_expr_weight : bool
        Whether to use expression-weighted ambient prediction.
        Default False (backward compatible). When True, source
        contribution is quadratically weighted by expression level
        (s²/s_max), suppressing non-expressing cells from acting
        as ambient sources. Recommended for cell-type-specific genes.
    per_gene_lambda : bool
        Whether to estimate per-gene λ (instead of single global λ).
        Default False. When True, each gene gets its own optimal λ
        from lambda_grid via per-gene RSS minimization.
        Useful for cell-type-specific genes whose diffusion pattern
        differs from broadly expressed genes.
    cell_level : bool
        Whether to use cell-level correction (ambient computed per cell
        from other bins, excluding self-contribution). Default False.
    cell_based : bool
        If True, use cell-based pipeline: cells as intact units,
        empty space binned separately. Cells never split across bins.
        This is the recommended mode for data with cell segmentation.
    local_radius_factor : float
        Local density window radius = factor × λ.
    classification : dict or None
        Pre-computed bin classification. Overrides purity thresholds.
    verbose : bool
        Print progress messages.
    """

    def __init__(
        self,
        bin_size: int = 50,
        distance_metric: DistanceMetric = "exponential",
        lambda_distance: Optional[float] = None,
        max_radius: float = 200.0,
        n_high_genes: Optional[int] = None,
        n_lambda_genes: int = 50,
        r2_threshold: float = 0.05,
        empty_purity: float = 0.95,
        cell_purity: float = 0.80,
        lambda_grid: Optional[List[float]] = None,
        use_local_density: bool = False,
        use_expr_weight: bool = False,
        per_gene_lambda: bool = False,
        cell_level: bool = False,
        cell_based: bool = True,
        self_confidence_penalty: bool = True,
        local_radius_factor: float = 3.0,
        classification: Optional[np.ndarray] = None,
        verbose: bool = True,
    ):
        self.bin_size = bin_size
        self.distance_metric = distance_metric
        self.lambda_distance = lambda_distance
        self.max_radius = max_radius
        self.n_high_genes = n_high_genes
        self.n_lambda_genes = n_lambda_genes
        self.r2_threshold = r2_threshold
        self.empty_purity = empty_purity
        self.cell_purity = cell_purity
        self.lambda_grid = lambda_grid or [10, 20, 30, 50, 70, 100, 150, 200, 300]
        self.use_local_density = use_local_density
        self.use_expr_weight = use_expr_weight
        self.per_gene_lambda = per_gene_lambda
        self.cell_level = cell_level
        self.cell_based = cell_based
        self.self_confidence_penalty = self_confidence_penalty
        self.local_radius_factor = local_radius_factor
        self.classification = classification
        self.verbose = verbose

        # Results (populated after fit)
        self.lambda_ = None
        self.alpha_ = None
        self.r2_scores_ = None
        self.lambdas_per_gene_ = None
        self.bin_classification_ = None
        self.diagnostics_ = None
        self._gene_indices_ = None
        self._n_cell_bins_ = None
        self._n_empty_bins_ = None
        self._bin_coords_ = None
        self._bin_cell_assignment_ = None

    def fit_transform_from_dnb(
        self,
        dnb_expression: np.ndarray,
        dnb_coordinates: np.ndarray,
        dnb_cell_labels: np.ndarray,
    ) -> Tuple[csr_matrix, Dict[str, Any]]:
        """Fit the model and correct expression from DNB-level data.

        Args:
            dnb_expression: [genes × DNBs] expression matrix.
            dnb_coordinates: [DNBs × 2] coordinates in μm.
            dnb_cell_labels: [DNBs] cell IDs; -1 for empty.

        Returns:
            corrected_cell_expr: [genes × cells] corrected per-cell expression.
            diagnostics: Dictionary of diagnostic metrics.
        """
        check_inputs(dnb_expression, dnb_coordinates, dnb_cell_labels)

        if self.cell_based:
            return self._fit_cell_based(
                dnb_expression, dnb_coordinates, dnb_cell_labels
            )

        # ── Standard bin-level pipeline ────────────────────────────
        if self.verbose:
            print(f"Binning {dnb_coordinates.shape[0]} DNBs with bin_size={self.bin_size}...")

        (Y_cell, Y_empty, n_cell, n_empty, bin_coords,
         bin_cell_assignment, classification, n_total) = dnb_to_bins(
            dnb_expression, dnb_coordinates, dnb_cell_labels, self.bin_size
        )

        if self.classification is not None:
            classification = self.classification

        n_bins = len(n_cell)
        if self.verbose:
            n_empty_bins = int((classification == 0).sum())
            n_cell_bins = int((classification == 1).sum())
            n_mixed_bins = int((classification == 2).sum())
            print(f"  → {n_bins} bins: {n_empty_bins} empty, {n_cell_bins} cell, {n_mixed_bins} mixed")

        # Phase 1: Build empty-DNB observation set
        # Pure empty bins + empty component of mixed bins
        empty_bin_mask = n_empty > 0  # any bin with empty DNBs

        if self.verbose:
            n_empty_bins_obs = int(empty_bin_mask.sum())
            total_empty_dnb = int(n_empty[empty_bin_mask].sum())
            print(f"  → {n_empty_bins_obs} bins with empty DNBs ({total_empty_dnb} total empty DNBs)")

        # Phase 2: Select high-expression genes
        if self.verbose:
            print("Selecting high-expression genes...")

        gene_indices = select_high_expression_genes(
            Y_empty, n_empty, self.n_high_genes, empty_bin_mask
        )

        if len(gene_indices) == 0:
            raise RuntimeError("No high-expression genes detected. Check your data.")

        if self.verbose:
            print(f"  → Selected {len(gene_indices)} high-expression genes")

        # Phase 3: Estimate λ (if not provided)
        if self.lambda_distance is None:
            if self.verbose:
                print(f"Estimating λ via grid search over {self.lambda_grid}...")

            n_lambda_genes = min(self.n_lambda_genes, len(gene_indices))
            lambda_gene_indices = gene_indices[:n_lambda_genes]

            best_lam, best_rss, rss_per_lam, _ = estimate_lambda_grid_search(
                Y_empty, Y_cell, n_empty, n_cell, n_total, bin_coords,
                lambda_gene_indices, empty_bin_mask,
                self.lambda_grid, self.max_radius, self.distance_metric,
                use_expr_weight=self.use_expr_weight,
            )

            self.lambda_ = best_lam

            if self.verbose:
                print(f"  → Optimal λ = {best_lam:.1f} μm (RSS = {best_rss:.2f})")
        else:
            self.lambda_ = self.lambda_distance
            rss_per_lam = []
            if self.verbose:
                print(f"Using provided λ = {self.lambda_} μm")

        # Phase 4: Estimate α per gene
        if self.verbose:
            print("Estimating gene-specific leakage rates α...")

        self.alpha_, self.r2_scores_, self.lambdas_per_gene_ = estimate_alpha_per_gene(
            Y_empty, Y_cell, n_empty, n_cell, bin_coords,
            gene_indices, empty_bin_mask,
            self.lambda_, self.max_radius, self.distance_metric,
            use_expr_weight=self.use_expr_weight,
            per_gene_lambda=self.per_gene_lambda,
            lambda_grid=self.lambda_grid if self.per_gene_lambda else None,
        )

        n_corrected = int((self.r2_scores_ >= self.r2_threshold).sum())
        if self.verbose:
            print(f"  → {n_corrected}/{len(gene_indices)} genes pass R² threshold ({self.r2_threshold})")

        # Phase 5: Local density
        beta = None
        if self.use_local_density:
            if self.verbose:
                print("Computing local density β...")
            local_radius = self.local_radius_factor * self.lambda_
            beta = compute_local_density(bin_coords, n_cell, n_total, local_radius)
            if self.verbose:
                print(f"  → β range: [{beta.min():.3f}, {beta.max():.3f}]")

        # Phase 6: Correction
        if self.verbose:
            print("Applying ambient correction...")

        corrected_cell_expr, ambient_fraction, correction_magnitude = correct_expression(
            Y_cell, Y_empty, n_cell, n_empty, n_total,
            bin_coords, bin_cell_assignment,
            gene_indices, self.alpha_, self.r2_scores_,
            self.r2_threshold, self.lambda_, self.max_radius,
            beta, self.distance_metric,
            use_expr_weight=self.use_expr_weight,
            lambdas_per_gene=self.lambdas_per_gene_,
            cell_level=self.cell_level,
        )

        if self.verbose:
            print(f"  → Corrected expression matrix: {corrected_cell_expr.shape[0]} genes × {corrected_cell_expr.shape[1]} cells")

        # Store internals
        self.bin_classification_ = classification
        self._gene_indices_ = gene_indices
        self._n_cell_bins_ = n_cell
        self._n_empty_bins_ = n_empty
        self._bin_coords_ = bin_coords
        self._bin_cell_assignment_ = bin_cell_assignment

        # Diagnostics
        self.diagnostics_ = compute_diagnostics(
            self.alpha_, self.r2_scores_,
            ambient_fraction, correction_magnitude,
            self.lambda_, rss_per_lam, self.lambda_grid,
            classification, n_cell, n_empty, n_total,
            len(gene_indices), n_corrected, self.r2_threshold,
        )

        if self.verbose:
            print_diagnostics(self.diagnostics_)

        return corrected_cell_expr, self.diagnostics_

    def fit_transform(
        self,
        spot_expr: np.ndarray,
        spot_coords: np.ndarray,
        spot_labels: np.ndarray,
    ) -> Tuple[csr_matrix, Dict[str, Any]]:
        """Fit the model and correct expression from spot-level data.

        This is the recommended entry point for raw spatial data. It is a
        thin wrapper around ``fit_transform_from_dnb`` (which is kept for
        backward compatibility).

        Args:
            spot_expr: [genes × spots] expression matrix.
            spot_coords: [spots × 2] coordinates in μm.
            spot_labels: [spots] cell IDs; -1 for empty spots.

        Returns:
            corrected_cell_expr: [genes × cells] corrected per-cell expression.
            diagnostics: Dictionary of diagnostic metrics.
        """
        return self.fit_transform_from_dnb(spot_expr, spot_coords, spot_labels)

    def _fit_cell_based(
        self,
        dnb_expression: np.ndarray,
        dnb_coordinates: np.ndarray,
        dnb_cell_labels: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Cell-based pipeline: cells intact, empty DNBs binned separately."""
        corrected, diag = cell_pipeline_fit(
            dnb_expression, dnb_coordinates, dnb_cell_labels,
            bin_size=self.bin_size,
            distance_metric=self.distance_metric,
            max_radius=self.max_radius,
            n_high_genes=self.n_high_genes,
            n_lambda_genes=self.n_lambda_genes,
            r2_threshold=self.r2_threshold,
            lambda_grid=self.lambda_grid,
            use_expr_weight=self.use_expr_weight,
            self_confidence_penalty=self.self_confidence_penalty,
            verbose=self.verbose,
        )
        self.lambda_ = diag["lambda_estimated"]
        self.alpha_ = diag.get("alphas", None)
        self.r2_scores_ = diag.get("r2_scores", None)
        self._gene_indices_ = diag.get("gene_indices", None)
        self.diagnostics_ = diag
        return corrected, diag

    def fit_transform_binned(
        self,
        bin_cell_expr: np.ndarray,
        bin_empty_expr: np.ndarray,
        bin_n_cell: np.ndarray,
        bin_n_empty: np.ndarray,
        bin_coords: np.ndarray,
        bin_cell_assignment: Dict[int, Dict[int, int]],
    ) -> Tuple[csr_matrix, Dict[str, Any]]:
        """Fit and correct from pre-binned data.

        Args:
            bin_cell_expr: [genes × bins] cell DNB expression.
            bin_empty_expr: [genes × bins] empty DNB expression.
            bin_n_cell: [bins] cell DNB counts.
            bin_n_empty: [bins] empty DNB counts.
            bin_coords: [bins × 2] bin center coordinates (μm).
            bin_cell_assignment: bin_id → {cell_id: n_dnbs}.

        Returns:
            corrected_cell_expr: [genes × cells] corrected per-cell expression.
            diagnostics: Dictionary of diagnostic metrics.
        """
        if not issparse(bin_cell_expr):
            Y_cell = csr_matrix(bin_cell_expr)
        else:
            Y_cell = bin_cell_expr.tocsr()

        if not issparse(bin_empty_expr):
            Y_empty = csr_matrix(bin_empty_expr)
        else:
            Y_empty = bin_empty_expr.tocsr()

        n_total = bin_n_cell + bin_n_empty
        n_bins = len(bin_n_cell)

        if self.verbose:
            print(f"Processing {n_bins} pre-binned units...")

        # Classify bins
        if self.classification is None:
            empty_ratio = np.zeros(n_bins, dtype=np.float64)
            mask = n_total > 0
            empty_ratio[mask] = bin_n_empty[mask] / n_total[mask]

            classification = np.full(n_bins, 2, dtype=np.int64)
            for i in range(n_bins):
                if n_total[i] == 0:
                    classification[i] = 0
                    continue
                if empty_ratio[i] >= self.empty_purity:
                    classification[i] = 0
                elif bin_cell_assignment.get(i):
                    max_ratio = max(bin_cell_assignment[i].values()) / n_total[i]
                    if max_ratio >= self.cell_purity:
                        classification[i] = 1
        else:
            classification = self.classification

        if self.verbose:
            n_empty_bins = int((classification == 0).sum())
            n_cell_bins = int((classification == 1).sum())
            n_mixed_bins = int((classification == 2).sum())
            print(f"  → {n_empty_bins} empty, {n_cell_bins} cell, {n_mixed_bins} mixed")

        # Empty observation mask
        empty_bin_mask = bin_n_empty > 0

        # Select high-expression genes
        gene_indices = select_high_expression_genes(
            Y_empty, bin_n_empty, self.n_high_genes, empty_bin_mask
        )
        if self.verbose:
            print(f"  → {len(gene_indices)} high-expression genes selected")

        # Estimate λ
        if self.lambda_distance is None:
            n_lambda_genes = min(self.n_lambda_genes, len(gene_indices))
            lambda_gene_indices = gene_indices[:n_lambda_genes]
            best_lam, best_rss, rss_per_lam, _ = estimate_lambda_grid_search(
                Y_empty, Y_cell, bin_n_empty, bin_n_cell, n_total, bin_coords,
                lambda_gene_indices, empty_bin_mask,
                self.lambda_grid, self.max_radius, self.distance_metric,
                use_expr_weight=self.use_expr_weight,
            )
            self.lambda_ = best_lam
            if self.verbose:
                print(f"  → λ = {best_lam:.1f} μm")
        else:
            self.lambda_ = self.lambda_distance
            rss_per_lam = []

        # Estimate α
        self.alpha_, self.r2_scores_, self.lambdas_per_gene_ = estimate_alpha_per_gene(
            Y_empty, Y_cell, bin_n_empty, bin_n_cell, bin_coords,
            gene_indices, empty_bin_mask,
            self.lambda_, self.max_radius, self.distance_metric,
            use_expr_weight=self.use_expr_weight,
            per_gene_lambda=self.per_gene_lambda,
            lambda_grid=self.lambda_grid if self.per_gene_lambda else None,
        )

        n_corrected = int((self.r2_scores_ >= self.r2_threshold).sum())
        if self.verbose:
            print(f"  → {n_corrected} genes pass R² threshold")

        # Local density
        beta = None
        if self.use_local_density:
            local_radius = self.local_radius_factor * self.lambda_
            beta = compute_local_density(bin_coords, bin_n_cell, n_total, local_radius)

        # Correct
        corrected_cell_expr, ambient_fraction, correction_magnitude = correct_expression(
            Y_cell, Y_empty, bin_n_cell, bin_n_empty, n_total,
            bin_coords, bin_cell_assignment,
            gene_indices, self.alpha_, self.r2_scores_,
            self.r2_threshold, self.lambda_, self.max_radius,
            beta, self.distance_metric,
            use_expr_weight=self.use_expr_weight,
            lambdas_per_gene=self.lambdas_per_gene_,
            cell_level=self.cell_level,
        )

        # Store
        self.bin_classification_ = classification
        self._gene_indices_ = gene_indices

        self.diagnostics_ = compute_diagnostics(
            self.alpha_, self.r2_scores_,
            ambient_fraction, correction_magnitude,
            self.lambda_, rss_per_lam, self.lambda_grid,
            classification, bin_n_cell, bin_n_empty, n_total,
            len(gene_indices), n_corrected, self.r2_threshold,
        )

        if self.verbose:
            print_diagnostics(self.diagnostics_)

        return corrected_cell_expr, self.diagnostics_
