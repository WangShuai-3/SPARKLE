#!/usr/bin/env python3
"""Benchmark SPARKLE runtime and memory on MouseBrain real data.

Loads the MouseBrain GEM once at the user-specified maximum spatial window,
then extracts progressively smaller windows by arithmetically shrinking the
side lengths (each step subtracts an equal fraction of the original side length).
This avoids the repeated cost of parsing the raw GEM file.

Only SPARKLE is benchmarked; r2_threshold can be set to 0 to measure pure
speed without any gene filtering.

SPARKLE is run in the main process with ``@profile`` from ``memory_profiler``;
the line-by-line profile for each run is saved under
``evaluation/reports/resource_profiles/`` and the maximum SPARKLE memory
increment (above the already-loaded data) is recorded in the CSV.

Usage example:
    python evaluation/scripts/benchmark_resource.py \
        --x-range 6000 20000 \
        --y-range 2000 15000 \
        --n-genes 10000 \
        --n-high-genes 10000 \
        --r2-threshold 0 \
        --n-runs 8
"""

import argparse
import gc
import io
import re
import sys
import time
from pathlib import Path

# Add project root to path so `evaluation.*` imports work when running standalone.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd

from evaluation.scripts.final_comparison import (
    load_mousebrain_data,
    subsample_data,
    run_sparkle_method,
)


def _shrink_range(x_range, y_range, n_runs, step):
    """Return a smaller spatial window centered on the original window.

    The side lengths are reduced arithmetically: at step ``step`` each side
    length is ``L * (1 - step / n_runs)``, where ``L`` is the original side
    length. With ``step`` from 0 to ``n_runs - 1``, the smallest window has
    side lengths equal to ``L / n_runs`` (i.e. the area is divided by
    ``n_runs ** 2`` relative to the maximum window).
    """
    x_min, x_max = x_range
    y_min, y_max = y_range
    x_center = (x_min + x_max) / 2.0
    y_center = (y_min + y_max) / 2.0
    x_half = (x_max - x_min) / 2.0 * (1.0 - step / n_runs)
    y_half = (y_max - y_min) / 2.0 * (1.0 - step / n_runs)
    return (x_center - x_half, x_center + x_half), (y_center - y_half, y_center + y_half)


def _extract_window(data, x_range, y_range):
    """Extract a spatial window from already-loaded data without re-reading GEM."""
    x_min, x_max = x_range
    y_min, y_max = y_range
    mask = (
        (data["dnb_coords"][:, 0] >= x_min)
        & (data["dnb_coords"][:, 0] <= x_max)
        & (data["dnb_coords"][:, 1] >= y_min)
        & (data["dnb_coords"][:, 1] <= y_max)
    )
    return {
        "dnb_expr": data["dnb_expr"][:, mask],
        "dnb_coords": data["dnb_coords"][mask],
        "dnb_labels": data["dnb_labels"][mask],
        "gene_names": data["gene_names"],
        "cell_ids": data["cell_ids"],
    }


def _parse_memory_profile(profile_text):
    """Parse memory_profiler text output and return memory statistics.

    Returns:
        dict with:
          - baseline_mb: Mem usage at the first profiled line.
          - peak_mb: max(Mem usage) during the function.
          - sparkle_increment_mb: peak_mb - baseline_mb (SPARKLE-only peak).
          - max_increment_mb: largest single-line Increment.
    """
    # Format: "Line #    Mem usage    Increment  Occurrences   Line Contents"
    mem_usages = []
    increments = []
    for line in profile_text.splitlines():
        line = line.strip()
        if not line or line.startswith("Line") or line.startswith("=="):
            continue
        # Each data line: line_no  Mem_usage  MiB  Increment  MiB  Occurrences  Line Contents
        parts = line.split()
        if len(parts) < 6:
            continue
        try:
            mem_usage = float(parts[1])
            increment = float(parts[3])
            mem_usages.append(mem_usage)
            increments.append(increment)
        except ValueError:
            continue

    if not mem_usages:
        return {
            "baseline_mb": float("nan"),
            "peak_mb": float("nan"),
            "sparkle_increment_mb": float("nan"),
            "max_increment_mb": float("nan"),
        }

    baseline = mem_usages[0]
    peak = max(mem_usages)
    return {
        "baseline_mb": baseline,
        "peak_mb": peak,
        "sparkle_increment_mb": peak - baseline,
        "max_increment_mb": max(increments) if increments else 0.0,
    }


def _run_sparkle_profiled(sub, n_high_genes, r2_threshold, lambda_grid, max_radius):
    """Run SPARKLE in the main process and capture memory_profiler output.

    Returns:
        (runtime_sec, diag, profile_text, parsed_mem_dict)
    """
    old_stdout = sys.stdout
    captured = io.StringIO()

    t0 = time.time()
    try:
        sys.stdout = captured
        corrected, diag = run_sparkle_method(
            sub,
            n_high_genes=n_high_genes,
            r2_threshold=r2_threshold,
            lambda_grid=lambda_grid,
            max_radius=max_radius,
            verbose=False,
        )
        runtime = time.time() - t0
        status = "ok"
    except Exception as e:
        runtime = time.time() - t0
        diag = {}
        status = f"error: {e}"
    finally:
        sys.stdout = old_stdout

    profile_text = captured.getvalue()
    mem_stats = _parse_memory_profile(profile_text)

    # Print status back to real stdout.
    if status == "ok":
        print(f"    Done in {runtime:.1f}s, λ={diag.get('lambda', float('nan')):.0f}μm, "
              f"SPARKLE peak +{mem_stats['sparkle_increment_mb']:.1f} MiB")
    else:
        print(f"    ERROR: {status}")

    return runtime, diag, profile_text, mem_stats, status


def _plot_results(df, output_dir):
    """Generate runtime and peak-memory plots versus number of DNBs."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = df.sort_values("n_dnbs")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["n_dnbs"], df["runtime_sec"], marker="o", color="#e74c3c")
    ax.set_xlabel("Number of DNBs")
    ax.set_ylabel("Runtime (s)")
    ax.set_title("SPARKLE runtime vs data size (MouseBrain)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_dir / "resource_benchmark_runtime.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["n_dnbs"], df["sparkle_peak_mb"], marker="o", color="#3498db")
    ax.set_xlabel("Number of DNBs")
    ax.set_ylabel("Memory (MB)")
    ax.set_title("SPARKLE peak memory increment vs data size (MouseBrain)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_dir / "resource_benchmark_memory.png", dpi=150)
    plt.close(fig)

    print(f"Saved plots in {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark SPARKLE resource consumption on MouseBrain real data."
    )
    parser.add_argument(
        "--x-range", type=int, nargs=2, required=True,
        help="Maximum X range (two integers, e.g. 10000 20000)",
    )
    parser.add_argument(
        "--y-range", type=int, nargs=2, required=True,
        help="Maximum Y range (two integers, e.g. 2000 22000)",
    )
    parser.add_argument(
        "--n-genes", type=int, default=10000,
        help="Number of top genes to keep (default: 10000)",
    )
    parser.add_argument(
        "--n-high-genes", type=int, default=10000,
        help="High-expression gene count passed to SPARKLE (default: 10000)",
    )
    parser.add_argument(
        "--r2-threshold", type=float, default=0.0,
        help="R^2 threshold for SPARKLE gene correction, 0 disables filtering (default: 0)",
    )
    parser.add_argument(
        "--n-runs", type=int, default=5,
        help="Number of arithmetic shrink steps (default: 5). "
             "Side lengths are reduced by (original / n_runs) each step.",
    )
    parser.add_argument(
        "--lambda-grid", type=int, nargs="+",
        default=[10, 20, 30, 50, 70, 100, 150, 200, 300],
        help="Lambda candidates in um for SPARKLE (default: 10 20 30 50 70 100 150 200 300)",
    )
    parser.add_argument(
        "--max-radius", type=float, default=300.0,
        help="Spatial neighborhood radius in um (default: 300)",
    )
    parser.add_argument(
        "--output", type=str,
        default="evaluation/reports/resource_benchmark_mousebrain.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--profile-dir", type=str,
        default="evaluation/reports/resource_profiles",
        help="Directory to save per-run memory_profiler text outputs",
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Generate runtime/memory line plots",
    )
    args = parser.parse_args()

    x_range = tuple(args.x_range)
    y_range = tuple(args.y_range)

    print("=" * 60)
    print("RESOURCE BENCHMARK: SPARKLE on MouseBrain")
    print(f"  Max window: x={x_range}, y={y_range}")
    print(f"  Genes: {args.n_genes}, high genes: {args.n_high_genes}")
    print(f"  R2 threshold: {args.r2_threshold}")
    print(f"  Arithmetic shrink steps: {args.n_runs}")
    print("=" * 60)

    # 1. Load the full MouseBrain data once (the expensive step).
    print("\n[1/3] Loading full MouseBrain window...")
    t_load = time.time()
    data = load_mousebrain_data(x_range=x_range, y_range=y_range)
    print(f"  Loaded in {time.time() - t_load:.1f}s")

    # 2. Subsample genes once.
    print("\n[2/3] Subsampling genes...")
    sub_full = subsample_data(data, n_genes=args.n_genes, cut_genes=True)

    # 3. Benchmark SPARKLE on arithmetically shrinking windows.
    print("\n[3/3] Benchmarking SPARKLE...")
    profile_dir = Path(args.profile_dir)
    profile_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for i in range(args.n_runs):
        x_range_i, y_range_i = _shrink_range(x_range, y_range, args.n_runs, i)
        window = _extract_window(sub_full, x_range_i, y_range_i)

        n_empty = int((window["dnb_labels"] < 0).sum())
        n_cell_dnbs = int((window["dnb_labels"] >= 0).sum())
        n_cells = len(set(window["dnb_labels"][window["dnb_labels"] >= 0].tolist()))

        print(f"\n  Run {i + 1}/{args.n_runs}: x=[{x_range_i[0]:.1f}, {x_range_i[1]:.1f}], "
              f"y=[{y_range_i[0]:.1f}, {y_range_i[1]:.1f}], "
              f"{window['dnb_expr'].shape[1]} DNBs ({n_cell_dnbs} cell + {n_empty} empty), "
              f"{n_cells} cells")

        runtime, diag, profile_text, mem_stats, status = _run_sparkle_profiled(
            window,
            n_high_genes=args.n_high_genes,
            r2_threshold=args.r2_threshold,
            lambda_grid=args.lambda_grid,
            max_radius=args.max_radius,
        )

        # Save the full memory_profiler output.
        profile_path = profile_dir / f"run_{i + 1}.txt"
        profile_path.write_text(profile_text, encoding="utf-8")

        results.append({
            "run": i + 1,
            "x_min": x_range_i[0],
            "x_max": x_range_i[1],
            "y_min": y_range_i[0],
            "y_max": y_range_i[1],
            "x_width": x_range_i[1] - x_range_i[0],
            "y_height": y_range_i[1] - y_range_i[0],
            "n_dnbs": window["dnb_expr"].shape[1],
            "n_genes": window["dnb_expr"].shape[0],
            "n_cells": n_cells,
            "n_empty_dnbs": n_empty,
            "runtime_sec": runtime,
            "baseline_mb": mem_stats["baseline_mb"],
            "sparkle_peak_mb": mem_stats["sparkle_increment_mb"],
            "max_increment_mb": mem_stats["max_increment_mb"],
            "lambda": diag.get("lambda", float("nan")) if status == "ok" else float("nan"),
            "status": status,
        })

        # Release memory before the next run.
        del window
        gc.collect()

    df = pd.DataFrame(results)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"\n{'=' * 60}")
    print(f"Saved CSV: {out_path}")
    print(f"Saved profiles: {profile_dir}")
    print(f"{'=' * 60}")
    print(df.to_string(index=False))

    if args.plot:
        _plot_results(df, out_path.parent)


if __name__ == "__main__":
    main()
