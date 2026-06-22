"""Diagnostic metrics and reporting for SPARKLE."""

import numpy as np
from typing import Dict, Any, List


def compute_diagnostics(
    alpha: np.ndarray,
    r2_scores: np.ndarray,
    ambient_fraction: np.ndarray,
    correction_magnitude: np.ndarray,
    lambda_est: float,
    rss_per_lambda: List[float],
    lambda_grid: List[float],
    bin_classification: np.ndarray,
    n_cell: np.ndarray,
    n_empty: np.ndarray,
    n_total: np.ndarray,
    n_genes_high: int,
    n_genes_corrected: int,
    r2_threshold: float,
) -> Dict[str, Any]:
    """Generate a diagnostic summary.

    Args:
        alpha: [n_high_genes] gene-specific leakage rates.
        r2_scores: [n_high_genes] R² scores.
        ambient_fraction: [n_total_genes] per-gene ambient fraction.
        correction_magnitude: [n_total_genes] mean correction per gene.
        lambda_est: Estimated λ.
        rss_per_lambda: RSS values from grid search.
        lambda_grid: λ values tried.
        bin_classification: [bins] 0=empty, 1=cell, 2=mixed.
        n_cell, n_empty, n_total: per-bin DNB counts.
        n_genes_high: Number of high-expression genes selected.
        n_genes_corrected: Number of genes actually corrected.
        r2_threshold: R² threshold used.

    Returns:
        Dictionary of diagnostic metrics.
    """
    n_bins = len(bin_classification)
    n_empty_bins = int((bin_classification == 0).sum())
    n_cell_bins = int((bin_classification == 1).sum())
    n_mixed_bins = int((bin_classification == 2).sum())

    total_dnb = n_total.sum()
    total_empty_dnb = n_empty.sum()
    total_cell_dnb = n_cell.sum()

    # Alpha statistics (for corrected genes only)
    corrected_mask = r2_scores >= r2_threshold
    alpha_corrected = alpha[corrected_mask]
    r2_corrected = r2_scores[corrected_mask]

    # Per-gene ambient fraction for corrected genes
    af_nonzero = ambient_fraction[ambient_fraction > 0]

    diagnostics = {
        "lambda_estimated": float(lambda_est),
        "lambda_grid": lambda_grid,
        "rss_per_lambda": [float(r) for r in rss_per_lambda],
        "n_total_bins": n_bins,
        "n_empty_bins": n_empty_bins,
        "n_cell_bins": n_cell_bins,
        "n_mixed_bins": n_mixed_bins,
        "total_dnb": int(total_dnb),
        "total_empty_dnb": int(total_empty_dnb),
        "total_cell_dnb": int(total_cell_dnb),
        "empty_dnb_ratio": float(total_empty_dnb / max(total_dnb, 1)),
        "n_high_genes": n_genes_high,
        "n_genes_corrected": n_genes_corrected,
        "r2_threshold": r2_threshold,
        "alpha_stats": {
            "mean": float(alpha_corrected.mean()) if len(alpha_corrected) > 0 else 0.0,
            "std": float(alpha_corrected.std()) if len(alpha_corrected) > 0 else 0.0,
            "median": float(np.median(alpha_corrected)) if len(alpha_corrected) > 0 else 0.0,
            "min": float(alpha_corrected.min()) if len(alpha_corrected) > 0 else 0.0,
            "max": float(alpha_corrected.max()) if len(alpha_corrected) > 0 else 0.0,
        },
        "r2_stats": {
            "mean": float(r2_corrected.mean()) if len(r2_corrected) > 0 else 0.0,
            "std": float(r2_corrected.std()) if len(r2_corrected) > 0 else 0.0,
            "median": float(np.median(r2_corrected)) if len(r2_corrected) > 0 else 0.0,
        },
        "ambient_fraction_stats": {
            "mean": float(af_nonzero.mean()) if len(af_nonzero) > 0 else 0.0,
            "median": float(np.median(af_nonzero)) if len(af_nonzero) > 0 else 0.0,
            "max": float(af_nonzero.max()) if len(af_nonzero) > 0 else 0.0,
        },
        "correction_magnitude_stats": {
            "mean": float(correction_magnitude[correction_magnitude > 0].mean()) if (correction_magnitude > 0).any() else 0.0,
            "max": float(correction_magnitude.max()),
        },
    }

    return diagnostics


def print_diagnostics(diag: Dict[str, Any]):
    """Print a human-readable diagnostic summary."""
    print("=" * 60)
    print("SPARKLE Diagnostic Report")
    print("=" * 60)
    print(f"\nEstimated distance decay λ: {diag['lambda_estimated']:.1f} μm")
    print(f"\nBin Summary:")
    print(f"  Total bins:        {diag['n_total_bins']}")
    print(f"  Pure empty bins:   {diag['n_empty_bins']}")
    print(f"  Pure cell bins:    {diag['n_cell_bins']}")
    print(f"  Mixed bins:        {diag['n_mixed_bins']}")
    print(f"  Empty DNB ratio:   {diag['empty_dnb_ratio']:.3f}")
    print(f"\nGene Statistics:")
    print(f"  High-expr genes selected: {diag['n_high_genes']}")
    print(f"  Genes corrected (R² ≥ {diag['r2_threshold']}): {diag['n_genes_corrected']}")
    print(f"\nLeakage Rate (α) for corrected genes:")
    a = diag['alpha_stats']
    print(f"  Mean ± SD: {a['mean']:.4f} ± {a['std']:.4f}")
    print(f"  Median:     {a['median']:.4f}")
    print(f"  Range:      [{a['min']:.4f}, {a['max']:.4f}]")
    print(f"\nModel Fit (R²) for corrected genes:")
    r = diag['r2_stats']
    print(f"  Mean ± SD: {r['mean']:.4f} ± {r['std']:.4f}")
    print(f"  Median:     {r['median']:.4f}")
    print(f"\nAmbient Fraction (corrected genes):")
    af = diag['ambient_fraction_stats']
    print(f"  Mean:   {af['mean']:.4f}")
    print(f"  Median: {af['median']:.4f}")
    print(f"  Max:    {af['max']:.4f}")
    print("=" * 60)
