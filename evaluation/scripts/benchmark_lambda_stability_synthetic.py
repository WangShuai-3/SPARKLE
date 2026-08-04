#!/usr/bin/env python3
"""Test SPARKLE's sensitivity to a FORCED lambda (spatial decay) on synthetic data.

SPARKLE normally estimates the global spatial decay length ``lambda`` via grid
search.  This benchmark instead *forces* SPARKLE to use a series of fixed lambda
values (via single-element ``lambda_grid``) on a synthetic scenario with a KNOWN
ground truth, then measures the RMSE reduction vs raw for each.  It answers:
how stable is correction quality when lambda is mis-specified, and does the
auto-estimated lambda land near the optimum / the true value?

Default scenario: S2 "Medium multi-type (25% empty)", ground-truth lambda =
50 µm. S2 deliberately uses random spatial assignment
(``cluster_strength=0``); the spatially clustered condition is S6. This keeps
the lambda sweep focused on decay-scale misspecification rather than spatial
cell-type domains.

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
from evaluation.synthetic.scenarios import get_scenario


def _cell_r2(pred, true):
    """Per-cell R² (how well the corrected profile fits the true profile).

    For each cell c:
        R²_c = 1 - SS_res_c / SS_tot_c
        SS_res_c = Σ_g (pred_gc - true_gc)²
        SS_tot_c = Σ_g (true_gc - mean_c)²   (variance of the true profile)
    Returns the mean R² across cells. A perfect correction gives R² = 1;
    a correction no better than the mean profile gives R² = 0.
    """
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    ss_res = np.sum((pred - true) ** 2, axis=0)          # per cell
    ss_tot = np.sum((true - true.mean(axis=0)) ** 2, axis=0)  # per cell
    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = 1.0 - ss_res / ss_tot
    # Cells with zero variance in the true profile are undefined; exclude them.
    r2 = r2[np.isfinite(r2)]
    return float(r2.mean()) if len(r2) else float("nan")


def _run_sparkle_fixed_lambda(data, forced_lambda, r2_threshold, max_radius):
    """Run SPARKLE with a forced lambda (or auto if forced_lambda is None).

    SPARKLE estimates lambda via grid search, so to FORCE a lambda we pass a
    single-element ``lambda_grid=[forced_lambda]`` (the grid 'search' can only
    pick that value, while alpha and the spatial weights are still fit at that
    lambda).  When ``forced_lambda`` is None we use the full default grid
    (auto estimation).

    Mirrors the synthetic SPARKLE recipe in final_comparison.run_synthetic_comparison
    (bin_size=25, cell_based, self_confidence_penalty=True).
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
        cell_based=True,
        self_confidence_penalty=True,
        verbose=False,
    )
    t0 = time.time()
    corrected, diagnostics = model.fit_transform_from_dnb(
        data["dnb_expr"], data["dnb_coords"], data["dnb_labels"]
    )
    runtime = time.time() - t0
    if hasattr(corrected, "toarray"):
        corrected = corrected.toarray()
    return corrected, float(model.lambda_), runtime, diagnostics


def _plot_results(
    df,
    auto_row,
    true_lambda,
    output_dir,
    scenario_id,
    scenario_name,
    cluster_strength,
    metric="r2",
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = df.sort_values("forced_lambda_um")

    y_col = "cell_r2_gain" if metric == "r2" else "rmse_reduction_pct"
    y_label = "Per-cell R² gain vs raw" if metric == "r2" else "RMSE reduction vs raw (%)"
    y_fmt = ".4f" if metric == "r2" else ".1f"

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.plot(df["forced_lambda_um"], df[y_col],
            marker="o", color="#2980b9", lw=2, label="Forced λ")

    # True lambda reference
    if true_lambda and true_lambda > 0:
        ax.axvline(true_lambda, color="#27ae60", ls="--", lw=1.5,
                   label=f"True λ = {true_lambda:.0f} µm")
    # Auto-estimated lambda + its score
    if auto_row is not None:
        ax.axvline(auto_row["used_lambda_um"], color="#e67e22", ls=":", lw=1.5,
                   label=f"Auto λ = {auto_row['used_lambda_um']:.0f} µm "
                         f"({auto_row[y_col]:{y_fmt}})")
        ax.scatter([auto_row["used_lambda_um"]], [auto_row[y_col]],
                   color="#e67e22", zorder=5, s=70, marker="*")

    ax.set_xscale("log")
    ax.set_xlabel("Forced λ (µm, log scale)")
    ax.set_ylabel(y_label)
    spatial_assignment = (
        "random cell-type assignment"
        if cluster_strength == 0
        else f"cluster strength = {cluster_strength:g}"
    )
    ax.set_title(
        f"SPARKLE λ stability on synthetic {scenario_id}\n"
        f"{scenario_name}; {spatial_assignment}"
    )
    ax.grid(True, alpha=0.3, which="both")
    ax.legend()
    for _, r in df.iterrows():
        ax.annotate(f"{r[y_col]:{y_fmt}}",
                    (r["forced_lambda_um"], r[y_col]),
                    textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
    plt.tight_layout()
    suffix = "r2" if metric == "r2" else "rmse"
    out = output_dir / f"lambda_stability_{scenario_id}_{suffix}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {out}")


def _plot_rss(
    df,
    auto_row,
    true_lambda,
    output_dir,
    scenario_id,
    scenario_name,
    cluster_strength,
):
    """Plot the actual weighted RSS used by SPARKLE's lambda search."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = df.sort_values("forced_lambda_um")

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.plot(
        df["forced_lambda_um"],
        df["lambda_search_weighted_rss"],
        marker="o",
        color="#2980b9",
        lw=2,
    )

    selected_lambda = float(auto_row["used_lambda_um"])
    selected = df.loc[
        np.isclose(df["forced_lambda_um"], selected_lambda)
    ].iloc[0]
    if np.isclose(selected_lambda, true_lambda):
        reference_label = f"Auto-selected and true λ = {selected_lambda:.0f} µm"
    else:
        reference_label = f"Auto-selected λ = {selected_lambda:.0f} µm"
        ax.axvline(
            true_lambda,
            color="#555555",
            ls="--",
            lw=1.4,
            label=f"True λ = {true_lambda:.0f} µm",
        )
    ax.axvline(
        selected_lambda,
        color="#e67e22",
        ls=":",
        lw=1.8,
        label=reference_label,
    )
    ax.scatter(
        [selected_lambda],
        [selected["lambda_search_weighted_rss"]],
        color="#e67e22",
        edgecolor="white",
        linewidth=0.8,
        zorder=5,
        s=110,
        marker="*",
    )

    spatial_assignment = (
        "random cell-type assignment"
        if cluster_strength == 0
        else f"cluster strength = {cluster_strength:g}"
    )
    ax.set_xscale("log")
    ax.set_xlabel("Candidate λ (µm, log scale)")
    ax.set_ylabel("Weighted residual sum of squares (RSS)")
    ax.set_title(
        f"SPARKLE λ-search RSS on synthetic {scenario_id}\n"
        f"{scenario_name}; {spatial_assignment}; lower is better"
    )
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.grid(True, alpha=0.3, which="both")
    ax.legend()
    plt.tight_layout()
    out = output_dir / f"lambda_rss_{scenario_id}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved RSS plot: {out}")


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
    parser.add_argument(
        "--metric",
        type=str,
        default="r2",
        choices=["rmse", "r2"],
        help="Stability metric: 'rmse' (reduction %) or 'r2' (per-cell R² gain "
        "over raw; default: r2).",
    )
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    forced_lambdas = args.lambdas_um if args.lambdas_um is not None \
        else [5, 10, 20, 30, 50, 70, 100, 150, 200, 300, 500]

    print("=" * 60)
    print("LAMBDA STABILITY: SPARKLE on synthetic data")
    scenario = get_scenario(args.scenario)
    cluster_strength = float(scenario.get("cluster_strength", 0.0))
    if args.scenario == "S2" and cluster_strength != 0.0:
        raise ValueError(
            "The current lambda benchmark requires revised S2 to be "
            "non-clustered (cluster_strength=0). Use S6 for the clustered condition."
        )

    print(f"  Scenario: {args.scenario} ({scenario['name']})")
    print(f"  Cluster strength: {cluster_strength:g}")
    print(f"  Forced λ (µm): {forced_lambdas}")
    print("=" * 60)

    # 1. Generate synthetic scenario once.
    print("\n[1/3] Generating synthetic scenario...")
    data = load_synthetic_scenario_data(args.scenario, seed=args.seed)
    true_lambda = float(data.get("true_lambda", 0.0))

    # 2. Raw baseline metrics.
    raw_cell = compute_cell_expr(data["dnb_expr"], data["dnb_labels"],
                                 data["true_expr"].shape[1])
    rmse_raw = _rmse(raw_cell, data["true_expr"])
    r2_raw = _cell_r2(raw_cell, data["true_expr"])
    metric = args.metric
    print(f"\n[2/3] Raw RMSE = {rmse_raw:.4f}, Raw per-cell R² = {r2_raw:.4f}; "
          f"true λ = {true_lambda:.0f} µm | metric = {metric}")

    # 3. Sweep forced lambdas + one auto (grid-search) run for reference.
    print("\n[3/3] Sweeping forced λ...")
    rows = []
    for lam in forced_lambdas:
        corrected, used_lam, runtime, diagnostics = _run_sparkle_fixed_lambda(
            data, float(lam), args.r2_threshold, args.max_radius)
        rmse = _rmse(corrected, data["true_expr"])
        reduc = (rmse_raw - rmse) / rmse_raw * 100.0
        r2 = _cell_r2(corrected, data["true_expr"])
        r2_gain = r2 - r2_raw
        score = r2_gain if metric == "r2" else reduc
        rows.append({
            "mode": "forced",
            "forced_lambda_um": float(lam),
            "used_lambda_um": used_lam,
            "rmse": rmse,
            "rmse_reduction_pct": reduc,
            "cell_r2": r2,
            "cell_r2_gain": r2_gain,
            "score": score,
            "lambda_search_weighted_rss": diagnostics["lambda_search_rss"][0],
            "runtime_sec": runtime,
        })
        print(f"  forced λ={lam:>5.0f}µm -> RMSE={rmse:.4f} ({reduc:5.2f}%↓)  "
              f"R²={r2:.4f} (gain {r2_gain:+.4f})  ({runtime:.1f}s)")

    # Auto (grid search) reference.
    corrected, auto_lam, runtime, diagnostics = _run_sparkle_fixed_lambda(
        data, None, args.r2_threshold, args.max_radius)
    rmse = _rmse(corrected, data["true_expr"])
    reduc = (rmse_raw - rmse) / rmse_raw * 100.0
    r2 = _cell_r2(corrected, data["true_expr"])
    r2_gain = r2 - r2_raw
    auto_score = r2_gain if metric == "r2" else reduc
    auto_row = {
        "mode": "auto",
        "forced_lambda_um": np.nan,
        "used_lambda_um": auto_lam,
        "rmse": rmse,
        "rmse_reduction_pct": reduc,
        "cell_r2": r2,
        "cell_r2_gain": r2_gain,
        "score": auto_score,
        "lambda_search_weighted_rss": diagnostics["lambda_search_best_rss"],
        "runtime_sec": runtime,
    }
    print(f"  AUTO (grid search) -> selected λ={auto_lam:.0f}µm  "
          f"RMSE={rmse:.4f} ({reduc:5.2f}%↓)  R²={r2:.4f} (gain {r2_gain:+.4f})  ({runtime:.1f}s)")

    df = pd.DataFrame(rows + [auto_row])
    df["scenario"] = args.scenario
    df["scenario_name"] = scenario["name"]
    df["cluster_strength"] = cluster_strength
    df["spatial_type_assignment"] = (
        "random" if cluster_strength == 0 else "clustered"
    )
    df["seed"] = args.seed
    df["true_lambda_um"] = true_lambda
    df["rmse_raw"] = rmse_raw
    df["r2_raw"] = r2_raw

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    rss_path = out_path.parent / f"lambda_rss_{args.scenario}_source_data.csv"
    rss_columns = [
        "forced_lambda_um",
        "lambda_search_weighted_rss",
        "rmse",
        "rmse_reduction_pct",
        "cell_r2",
        "cell_r2_gain",
        "scenario",
        "scenario_name",
        "cluster_strength",
        "spatial_type_assignment",
        "seed",
        "true_lambda_um",
    ]
    df.loc[df["mode"] == "forced", rss_columns].to_csv(rss_path, index=False)

    # Stability summary over the forced sweep.
    forced_df = df[df["mode"] == "forced"]
    scores = forced_df["score"]
    score_label = "R² gain" if metric == "r2" else "RMSE reduction %"
    print(f"\n{'=' * 60}")
    print(f"Saved CSV: {out_path}")
    print(f"Saved RSS source data: {rss_path}")
    print(f"{'=' * 60}")
    print(df.to_string(index=False))
    print(f"\nForced-λ {score_label}: min={scores.min():.4f}  max={scores.max():.4f}  "
          f"spread={scores.max() - scores.min():.4f} (over λ = {forced_lambdas[0]:.0f}-{forced_lambdas[-1]:.0f} µm)")
    best = forced_df.loc[scores.idxmax()]
    best_col = "cell_r2_gain" if metric == "r2" else "rmse_reduction_pct"
    auto_col = "cell_r2_gain" if metric == "r2" else "rmse_reduction_pct"
    print(f"Best forced λ = {best['forced_lambda_um']:.0f}µm ({best[best_col]:.4f}); "
          f"auto-selected λ = {auto_lam:.0f}µm ({auto_row[auto_col]:.4f}); "
          f"true λ = {true_lambda:.0f}µm")

    if args.plot:
        _plot_results(
            forced_df,
            auto_row,
            true_lambda,
            out_path.parent,
            args.scenario,
            scenario["name"],
            cluster_strength,
            metric,
        )
        _plot_rss(
            forced_df,
            auto_row,
            true_lambda,
            out_path.parent,
            args.scenario,
            scenario["name"],
            cluster_strength,
        )


if __name__ == "__main__":
    main()
