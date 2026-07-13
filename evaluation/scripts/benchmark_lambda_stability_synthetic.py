#!/usr/bin/env python3
"""Test SPARKLE's sensitivity to a FORCED lambda (spatial decay) on synthetic data.

SPARKLE normally estimates the global spatial decay length ``lambda`` via grid
search.  This benchmark instead *forces* SPARKLE to use a series of fixed lambda
values (via ``lambda_distance``) on a synthetic scenario with a KNOWN ground
truth, then measures the RMSE reduction vs raw for each.  It answers: how stable
is correction quality when lambda is mis-specified, and does the auto-estimated
lambda land near the optimum / the true value?

Default scenario: S2 "Medium multi-type (25% empty)", ground-truth lambda = 50 µm.

Modelled after ``benchmark_resource.py`` / ``benchmark_binsize_axolotl.py``:
the synthetic data is generated ONCE, then SPARKLE is run for each forced lambda.
Only SPARKLE is run.

Usage:
    python evaluation/scripts/benchmark_lambda_stability_synthetic.py \
        --scenario S2 --plot
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root so `evaluation.*` imports work when run standalone.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd

from stambient import SPARKLE
from evaluation.scripts.final_comparison import (
    load_synthetic_scenario_data,
    compute_cell_expr,
    _rmse,
)


def _run_sparkle_fixed_lambda(data, forced_lambda, r2_threshold, max_radius):
    """Run SPARKLE with a forced lambda (or auto if forced_lambda is None).

    The cell-based pipeline estimates lambda via grid search and ignores the
    ``lambda_distance`` argument, so to FORCE a lambda we pass a single-element
    ``lambda_grid=[forced_lambda]`` (the grid 'search' can only pick that value,
    while alpha and the spatial weights are still fit at that lambda).  When
    ``forced_lambda`` is None we use the full default grid (auto estimation).

    Mirrors the synthetic SPARKLE recipe in final_comparison.run_synthetic_comparison
    (bin_size=25, cell_based, self_confidence_penalty=False).
    Returns (corrected_matrix, used_lambda, runtime).
    """
    dnb_expr = data["dnb_expr"]
    if forced_lambda is None:
        lambda_grid = [10, 20, 30, 50, 70, 100, 150, 200, 300]
    else:
        lambda_grid = [float(forced_lambda)]
    model = SPARKLE(
        bin_size=25,
        distance_metric="exponential",
        max_radius=max_radius,
        n_high_genes=min(80, dnb_expr.shape[0]),
        n_lambda_genes=min(50, dnb_expr.shape[0]),
        r2_threshold=r2_threshold,
        lambda_grid=lambda_grid,
        use_local_density=False,
        cell_based=True,
        self_confidence_penalty=False,
        verbose=False,
    )
    t0 = time.time()
    corrected, _ = model.fit_transform_from_dnb(
        data["dnb_expr"], data["dnb_coords"], data["dnb_labels"]
    )
    runtime = time.time() - t0
    if hasattr(corrected, "toarray"):
        corrected = corrected.toarray()
    return corrected, float(model.lambda_), runtime


def _plot_results(df, auto_row, true_lambda, output_dir, scenario_id):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = df.sort_values("forced_lambda_um")

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.plot(df["forced_lambda_um"], df["rmse_reduction_pct"],
            marker="o", color="#2980b9", lw=2, label="Forced λ")

    # True lambda reference
    if true_lambda and true_lambda > 0:
        ax.axvline(true_lambda, color="#27ae60", ls="--", lw=1.5,
                   label=f"True λ = {true_lambda:.0f} µm")
    # Auto-estimated lambda + its reduction
    if auto_row is not None:
        ax.axvline(auto_row["used_lambda_um"], color="#e67e22", ls=":", lw=1.5,
                   label=f"Auto λ = {auto_row['used_lambda_um']:.0f} µm "
                         f"({auto_row['rmse_reduction_pct']:.1f}%)")
        ax.scatter([auto_row["used_lambda_um"]], [auto_row["rmse_reduction_pct"]],
                   color="#e67e22", zorder=5, s=70, marker="*")

    ax.set_xscale("log")
    ax.set_xlabel("Forced λ (µm, log scale)")
    ax.set_ylabel("RMSE reduction vs raw (%)")
    ax.set_title(f"SPARKLE λ stability on synthetic {scenario_id}\n(RMSE reduction vs forced spatial-decay λ)")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend()
    for _, r in df.iterrows():
        ax.annotate(f"{r['rmse_reduction_pct']:.1f}",
                    (r["forced_lambda_um"], r["rmse_reduction_pct"]),
                    textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
    plt.tight_layout()
    out = output_dir / f"lambda_stability_{scenario_id}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {out}")


def main():
    parser = argparse.ArgumentParser(
        description="Test SPARKLE forced-lambda stability on synthetic data."
    )
    parser.add_argument("--scenario", type=str, default="S2",
                        help="Synthetic scenario ID (default: S2, true λ=50 µm)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lambdas-um", type=float, nargs="+", default=None,
                        help="Forced λ values in µm "
                             "(default: 5 10 20 30 50 70 100 150 200 300 500)")
    parser.add_argument("--r2-threshold", type=float, default=0.01)
    parser.add_argument("--max-radius", type=float, default=300.0)
    parser.add_argument("--output", type=str,
                        default="evaluation/reports/lambda_stability/lambda_stability_synthetic.csv")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    forced_lambdas = args.lambdas_um if args.lambdas_um is not None \
        else [5, 10, 20, 30, 50, 70, 100, 150, 200, 300, 500]

    print("=" * 60)
    print("LAMBDA STABILITY: SPARKLE on synthetic data")
    print(f"  Scenario: {args.scenario}")
    print(f"  Forced λ (µm): {forced_lambdas}")
    print("=" * 60)

    # 1. Generate synthetic scenario once.
    print("\n[1/3] Generating synthetic scenario...")
    data = load_synthetic_scenario_data(args.scenario, seed=args.seed)
    true_lambda = float(data.get("true_lambda", 0.0))

    # 2. Raw baseline RMSE.
    raw_cell = compute_cell_expr(data["dnb_expr"], data["dnb_labels"],
                                 data["true_expr"].shape[1])
    rmse_raw = _rmse(raw_cell, data["true_expr"])
    print(f"\n[2/3] Raw RMSE = {rmse_raw:.4f}; true λ = {true_lambda:.0f} µm")

    # 3. Sweep forced lambdas + one auto (grid-search) run for reference.
    print("\n[3/3] Sweeping forced λ...")
    rows = []
    for lam in forced_lambdas:
        corrected, used_lam, runtime = _run_sparkle_fixed_lambda(
            data, float(lam), args.r2_threshold, args.max_radius)
        rmse = _rmse(corrected, data["true_expr"])
        reduc = (rmse_raw - rmse) / rmse_raw * 100.0
        rows.append({
            "mode": "forced",
            "forced_lambda_um": float(lam),
            "used_lambda_um": used_lam,
            "rmse": rmse,
            "rmse_reduction_pct": reduc,
            "runtime_sec": runtime,
        })
        print(f"  forced λ={lam:>5.0f}µm -> RMSE={rmse:.4f}  reduction={reduc:6.2f}%  ({runtime:.1f}s)")

    # Auto (grid search) reference.
    corrected, auto_lam, runtime = _run_sparkle_fixed_lambda(
        data, None, args.r2_threshold, args.max_radius)
    rmse = _rmse(corrected, data["true_expr"])
    reduc = (rmse_raw - rmse) / rmse_raw * 100.0
    auto_row = {
        "mode": "auto",
        "forced_lambda_um": np.nan,
        "used_lambda_um": auto_lam,
        "rmse": rmse,
        "rmse_reduction_pct": reduc,
        "runtime_sec": runtime,
    }
    print(f"  AUTO (grid search) -> selected λ={auto_lam:.0f}µm  "
          f"RMSE={rmse:.4f}  reduction={reduc:6.2f}%  ({runtime:.1f}s)")

    df = pd.DataFrame(rows + [auto_row])
    df["scenario"] = args.scenario
    df["true_lambda_um"] = true_lambda
    df["rmse_raw"] = rmse_raw

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    # Stability summary over the forced sweep.
    forced_df = df[df["mode"] == "forced"]
    red = forced_df["rmse_reduction_pct"]
    print(f"\n{'=' * 60}")
    print(f"Saved CSV: {out_path}")
    print(f"{'=' * 60}")
    print(df.to_string(index=False))
    print(f"\nForced-λ RMSE reduction: min={red.min():.2f}%  max={red.max():.2f}%  "
          f"spread={red.max() - red.min():.2f} pts (over λ = {forced_lambdas[0]:.0f}-{forced_lambdas[-1]:.0f} µm)")
    best = forced_df.loc[red.idxmax()]
    print(f"Best forced λ = {best['forced_lambda_um']:.0f}µm ({best['rmse_reduction_pct']:.2f}%); "
          f"auto-selected λ = {auto_lam:.0f}µm ({auto_row['rmse_reduction_pct']:.2f}%); "
          f"true λ = {true_lambda:.0f}µm")

    if args.plot:
        _plot_results(forced_df, auto_row, true_lambda, out_path.parent, args.scenario)


if __name__ == "__main__":
    main()
