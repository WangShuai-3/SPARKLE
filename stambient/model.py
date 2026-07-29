"""SPARKLE main class: pipeline orchestration."""

import numpy as np
from typing import Optional, Dict, Tuple, List, Any

from .spatial import DistanceMetric
from .cell_pipeline import cell_pipeline_fit
from .io_utils import check_inputs
from .gpu import GPUExecutionError


class SPARKLE:
    """Spatial Ambient RNA Kernel-based Leakage Estimator.

    Removes ambient RNA contamination from high-resolution spatial
    transcriptomics data using empty spots as built-in ambient probes.

    The cell-based pipeline is the only supported mode: segmented cells are
    kept intact throughout the analysis while out-of-mask capture locations
    are aggregated into background bins.

    Parameters
    ----------
    bin_size : float
        Square-bin side length in μm used to aggregate out-of-mask capture
        locations (default 50).
    distance_metric : str
        Weight function: 'exponential', 'gaussian', or 'inverse'. The
        'inverse' kernel does not use λ; the λ grid search is then skipped
        and ``lambda_estimated`` is reported as None.
    max_radius : float
        Maximum neighbor search radius in μm.
    n_high_genes : int or None
        Number of top highly-expressed genes to correct. If None, all
        genes are used.
    n_lambda_genes : int
        Number of top genes for λ estimation.
    r2_threshold : float
        Minimum weighted R² to apply correction. Default 0.01.
    lambda_grid : list of float
        λ values in μm for grid search.
    use_expr_weight : bool
        Whether to use expression-weighted ambient prediction.
        Default False. When True, source contribution is weighted by
        expression level, suppressing non-expressing cells from acting
        as ambient sources. Recommended for cell-type-specific genes.
    cell_based : bool
        Kept for backward compatibility. The legacy bin-level pipeline
        (``cell_based=False``) has been removed; passing False raises
        ValueError.
    self_confidence_penalty : bool
        Protects cells with high source expression from excessive
        subtraction. Default True.
    verbose : bool
        Print progress messages.
    use_gpu : bool
        If True, accelerate the batched sparse/dense computations with
        PyTorch CUDA. If PyTorch, CUDA, or a working GPU is unavailable,
        SPARKLE warns and falls back to CPU.
    gpu_dtype : str
        GPU precision mode: 'float64' (strictest agreement), 'mixed'
        (float32 sparse products and float64 reductions), or 'float32'.
    gpu_gene_batch_size : int or None
        Genes per GPU batch. None chooses a batch from available VRAM.
    """

    def __init__(
        self,
        bin_size: float = 50.0,
        distance_metric: DistanceMetric = "exponential",
        max_radius: float = 200.0,
        n_high_genes: Optional[int] = None,
        n_lambda_genes: int = 50,
        r2_threshold: float = 0.01,
        lambda_grid: Optional[List[float]] = None,
        use_expr_weight: bool = False,
        cell_based: bool = True,
        self_confidence_penalty: bool = True,
        verbose: bool = True,
        use_gpu: bool = False,
        gpu_dtype: str = "float64",
        gpu_gene_batch_size: Optional[int] = None,
    ):
        if not cell_based:
            raise ValueError(
                "cell_based=False is no longer supported: the legacy "
                "bin-level pipeline has been removed. Use the default "
                "cell-based pipeline (cell_based=True)."
            )
        if not np.isfinite(bin_size) or bin_size <= 0:
            raise ValueError("bin_size must be a positive finite length in µm")
        if not np.isfinite(max_radius) or max_radius <= 0:
            raise ValueError("max_radius must be a positive finite length in µm")
        if distance_metric not in ("exponential", "gaussian", "inverse"):
            raise ValueError(
                "distance_metric must be one of 'exponential', 'gaussian', "
                f"'inverse'; got {distance_metric!r}"
            )
        if lambda_grid is not None and (
            len(lambda_grid) == 0
            or any(not np.isfinite(value) or value <= 0 for value in lambda_grid)
        ):
            raise ValueError(
                "lambda_grid must be a non-empty list of positive finite "
                "lengths in µm"
            )
        if n_high_genes is not None and (
            not isinstance(n_high_genes, (int, np.integer)) or n_high_genes <= 0
        ):
            raise ValueError(
                "n_high_genes must be a positive integer or None; "
                f"got {n_high_genes!r}"
            )
        if not isinstance(n_lambda_genes, (int, np.integer)) or n_lambda_genes <= 0:
            raise ValueError(
                f"n_lambda_genes must be a positive integer; got {n_lambda_genes!r}"
            )
        if not np.isfinite(r2_threshold) or not 0.0 <= r2_threshold <= 1.0:
            raise ValueError(
                f"r2_threshold must be a finite value in [0, 1]; got {r2_threshold!r}"
            )

        self.bin_size = bin_size
        self.distance_metric = distance_metric
        self.max_radius = max_radius
        self.n_high_genes = n_high_genes
        self.n_lambda_genes = n_lambda_genes
        self.r2_threshold = r2_threshold
        self.lambda_grid = lambda_grid or [10, 20, 30, 50, 70, 100, 150, 200, 300]
        self.use_expr_weight = use_expr_weight
        self.cell_based = cell_based
        self.self_confidence_penalty = self_confidence_penalty
        self.verbose = verbose
        self.use_gpu = use_gpu
        self.gpu_dtype = gpu_dtype
        self.gpu_gene_batch_size = gpu_gene_batch_size

        # Results (populated after fit)
        self.lambda_ = None
        self.alpha_ = None
        self.r2_scores_ = None
        self.diagnostics_ = None
        self._gene_indices_ = None

    def fit_transform_from_dnb(
        self,
        dnb_expression: np.ndarray,
        dnb_coordinates: np.ndarray,
        dnb_cell_labels: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Fit the model and correct expression from DNB-level data.

        Args:
            dnb_expression: [genes × DNBs] expression matrix.
            dnb_coordinates: [DNBs × 2] coordinates in μm.
            dnb_cell_labels: [DNBs] cell IDs; -1 for empty.

        Returns:
            corrected_cell_expr: [genes × cells] corrected per-cell expression
                as a dense float64 numpy.ndarray.
            diagnostics: Dictionary of diagnostic metrics.
        """
        check_inputs(dnb_expression, dnb_coordinates, dnb_cell_labels)

        kwargs = dict(
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
        try:
            corrected, diag = cell_pipeline_fit(
                dnb_expression, dnb_coordinates, dnb_cell_labels,
                use_gpu=self.use_gpu,
                gpu_dtype=self.gpu_dtype,
                gpu_gene_batch_size=self.gpu_gene_batch_size,
                **kwargs,
            )
        except (GPUExecutionError, RuntimeError) as exc:
            if not self.use_gpu:
                raise
            # A device can disappear or run out of memory after successful
            # initialization. Re-running on CPU preserves the fallback promise.
            import warnings

            warnings.warn(
                f"GPU execution failed ({exc}); restarting SPARKLE on CPU.",
                RuntimeWarning,
            )
            corrected, diag = cell_pipeline_fit(
                dnb_expression, dnb_coordinates, dnb_cell_labels,
                use_gpu=False,
                gpu_dtype=self.gpu_dtype,
                gpu_gene_batch_size=self.gpu_gene_batch_size,
                **kwargs,
            )
            diag["gpu_requested"] = True
            diag["gpu_fallback_reason"] = str(exc)
        self.lambda_ = diag["lambda_estimated"]
        self.alpha_ = diag.get("alphas", None)
        self.r2_scores_ = diag.get("r2_scores", None)
        self._gene_indices_ = diag.get("gene_indices", None)
        self.diagnostics_ = diag
        return corrected, diag

    def fit_transform(
        self,
        spot_expr: np.ndarray,
        spot_coords: np.ndarray,
        spot_labels: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Fit the model and correct expression from spot-level data.

        This is the recommended entry point for raw spatial data. It is a
        thin wrapper around ``fit_transform_from_dnb`` (which is kept for
        backward compatibility).

        Args:
            spot_expr: [genes × spots] expression matrix.
            spot_coords: [spots × 2] coordinates in μm.
            spot_labels: [spots] cell IDs; -1 for empty spots.

        Returns:
            corrected_cell_expr: [genes × cells] corrected per-cell expression
                as a dense float64 numpy.ndarray.
            diagnostics: Dictionary of diagnostic metrics.
        """
        return self.fit_transform_from_dnb(spot_expr, spot_coords, spot_labels)
