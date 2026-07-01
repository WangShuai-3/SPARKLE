#!/usr/bin/env python3
"""Benchmark runtime and peak memory of correction methods across synthetic data sizes.

The gene count is kept constant (default 500) while the spatial grid size and the
number of cells are scaled together.  For each grid size and each method a fresh
subprocess is spawned so that peak RSS is not polluted by previous methods.

Usage example:
    python evaluation/scripts/benchmark_resource.py \
        --scenario S2 \
        --grid-sizes 100 150 200 250 300 350 400 \
        --n-genes 500 \
        --methods sparkle,spatial_soupx,soupx,decontx \
        --output evaluation/reports/resource_benchmark.csv \
        --plot
"""

import argparse
import multiprocessing as mp
import resource
import sys
import time
import warnings
from pathlib import Path

# Add project root to path so `evaluation.*` imports work when running standalone.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


def _run_single(args_tuple):
    """Worker: generate one synthetic dataset and run one method.

    Returns a dict with runtime and peak memory.
    """
    (grid_size, n_cells, n_genes, method, scenario_id, seed, n_high_genes) = args_tuple

    # Imports inside worker keep the parent process lightweight.
    from evaluation.scripts.final_comparison import (
        run_decontx_method,
        run_spatial_soupx_method,
        run_soupx_method,
        run_sparkle_method,
    )
    from evaluation.synthetic import generate_synthetic_data
    from evaluation.synthetic.scenarios import get_scenario

    scenario = get_scenario(scenario_id)
    try:
        data = generate_synthetic_data(
            n_cells=n_cells,
            grid_width=grid_size,
            grid_height=grid_size,
            dnb_pitch=0.5,
            cell_radius=5.0,
            cell_radius_cv=0.2,
            n_genes=n_genes,
            n_high_genes=80,
            ambient_lambda=scenario["ambient_lambda"],
            ambient_alpha=scenario["ambient_alpha"],
            empty_fraction=scenario["empty_fraction"],
            n_cell_types=scenario.get("n_cell_types", 1),
            marker_fraction=scenario.get("marker_fraction", 0.0),
            cluster_strength=scenario.get("cluster_strength", 0.5),
            seed=seed,
        )
        dnb_expr = csr_matrix(data["dnb_expr"].astype(np.float64))
        n_total_cells = data["true_expr"].shape[1]
        sub = {
            "dnb_expr": dnb_expr,
            "dnb_coords": data["dnb_coords"],
            "dnb_labels": data["dnb_labels"],
            "gene_names": np.array([f"gene_{i}" for i in range(n_genes)]),
            "cell_ids": np.arange(n_total_cells, dtype=np.int64),
        }

        t0 = time.time()
        if method == "sparkle":
            corrected, diag = run_sparkle_method(sub, n_high_genes=n_high_genes)
        elif method == "spatial_soupx":
            corrected, diag = run_spatial_soupx_method(sub)
        elif method == "soupx":
            corrected, diag = run_soupx_method(sub)
        elif method == "decontx":
            corrected, diag = run_decontx_method(sub)
        else:
            raise ValueError(f"Unknown method: {method}")
        runtime = time.time() - t0

        peak_mem_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_mem_mb = peak_mem_kb / 1024.0
        status = "ok"
    except Exception as e:
        runtime = float("nan")
        peak_mem_mb = float("nan")
        status = f"error: {e}"

    return {
        "scenario": scenario_id,
        "grid_size": grid_size,
        "n_cells": n_cells,
        "n_dnbs": grid_size * grid_size,
        "n_genes": n_genes,
        "method": method,
        "runtime_sec": runtime,
        "peak_memory_mb": peak_mem_mb,
        "status": status,
    }


def _plot_results(df, output_dir, tag):
    """Optional runtime / memory line plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = sorted(df["method"].unique())
    colors = {m: c for m, c in zip(methods, plt.cm.tab10.colors)}

    # Runtime vs n_dnbs
    fig, ax = plt.subplots(figsize=(8, 5))
    for method in methods:
        sub = df[df["method"] == method].sort_values("n_dnbs")
        ax.plot(sub["n_dnbs"], sub["runtime_sec"], marker="o", label=method, color=colors[method])
    ax.set_xlabel("Number of DNBs")
    ax.set_ylabel("Runtime (s)")
    ax.set_title(f"Runtime vs data size ({tag})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_dir / f"{tag}_runtime.png", dpi=150)
    plt.close(fig)

    # Peak memory vs n_dnbs
    fig, ax =plt.subplots(figsize=(8, 5))
    for method in methods:
        sub = df[df["method"] == method].sort_values("n_dnbs")
        ax.plot(sub["n_dnbs"], sub["peak_memory_mb"], marker="o", label=method, color=colors[method])
    ax.set_xlabel("Number of DNBs")
    ax.set_ylabel("Peak memory (MB)")
    ax.set_title(f"Peak memory vs data size ({tag})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_dir / f"{tag}_memory.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark resource consumption of correction methods on synthetic data."
    )
    parser.add_argument(
        "--scenario", type=str, default="S2",
        help="Synthetic scenario ID (default: S2)",
    )
    parser.add_argument(
        "--grid-sizes", type=int, nargs="+",
        default=[100, 150, 200, 250, 300, 350, 400],
        help="List of grid widths/heights to test (default: 100 150 200 250 300 350 400)",
    )
    parser.add_argument(
        "--n-genes", type=int, default=500,
        help="Number of genes to keep constant (default: 500)",
    )
    parser.add_argument(
        "--n-high-genes", type=int, default=None,
        help="High-expression gene count passed to SPARKLE (default: min(500, n_genes))",
    )
    parser.add_argument(
        "--methods", type=str,
        default="sparkle,spatial_soupx,soupx,decontx",
        help="Comma-separated methods to benchmark",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for synthetic data generation",
    )
    parser.add_argument(
        "--output", type=str,
        default="evaluation/reports/resource_benchmark.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Generate runtime/memory line plots",
    )
    parser.add_argument(
        "--n-jobs", type=int, default=1,
        help="Number of parallel workers (default: 1, sequential)",
    )
    args = parser.parse_args()

    from evaluation.synthetic.scenarios import get_scenario

    methods = [m.strip().lower() for m in args.methods.split(",") if m.strip()]
    scenario = get_scenario(args.scenario)
    base_cells = scenario.get("n_cells", 200)
    base_grid = 200 * 200
    n_high = args.n_high_genes if args.n_high_genes is not None else min(500, args.n_genes)

    tasks = []
    for grid_size in args.grid_sizes:
        n_dnbs = grid_size * grid_size
        # Scale cell count with grid area so density / empty_fraction stay similar.
        n_cells = max(10, int(base_cells * n_dnbs / base_grid))
        for method in methods:
            tasks.append((
                grid_size, n_cells, args.n_genes, method,
                args.scenario, args.seed, n_high,
            ))

    print(f"Benchmark: scenario={args.scenario}, genes={args.n_genes}")
    print(f"Grid sizes: {args.grid_sizes}")
    print(f"Methods: {methods}")
    print(f"Total runs: {len(tasks)}")
    print("-" * 60)

    if args.n_jobs > 1:
        with mp.Pool(processes=args.n_jobs) as pool:
            results = pool.map(_run_single, tasks)
    else:
        results = [_run_single(t) for t in tasks]

    df = pd.DataFrame(results)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nSaved CSV: {out_path}")

    # Pretty print
    print("\n" + df.to_string(index=False))

    if args.plot:
        _plot_results(df, out_path.parent, f"scenario_{args.scenario}_genes_{args.n_genes}")
        print(f"Saved plots in {out_path.parent}")


if __name__ == "__main__":
    main()
