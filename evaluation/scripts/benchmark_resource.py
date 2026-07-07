#!/usr/bin/env python3
"""Benchmark SPARKLE runtime and memory on MouseBrain real data.

Loads the MouseBrain GEM once at the user-specified maximum spatial window,
then extracts progressively smaller windows by arithmetically shrinking the
side lengths (each step subtracts an equal fraction of the original side length).
This avoids the repeated cost of parsing the raw GEM file.

Only SPARKLE is benchmarked; r2_threshold can be set to 0 to measure pure
speed without any gene filtering.

Memory columns in the output CSV:
- data_memory_mb:    RSS increment from loading the data subset in the worker.
- peak_memory_mb:    RSS increment from running SPARKLE (method-only).
- total_peak_mb:     Total peak RSS of the worker process (data + SPARKLE).
                     This is the actual RAM required for the whole step.

Usage example:
    python evaluation/scripts/benchmark_resource.py \
        --x-range 10000 20000 \
        --y-range 2000 22000 \
        --n-genes 10000 \
        --n-high-genes 10000 \
        --r2-threshold 0 \
        --n-runs 5
"""

import argparse
import multiprocessing as mp
import resource
import sys
import tempfile
import time
from pathlib import Path

# Add project root to path so `evaluation.*` imports work when running standalone.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
from scipy.sparse import save_npz, load_npz

from evaluation.scripts.final_comparison import (
    load_mousebrain_data,
    subsample_data,
    run_sparkle_method,
)


# Use spawn so the child process does not inherit (copy-on-write) the parent's
# full data matrix; this gives a cleaner peak-memory measurement for SPARKLE.
mp.set_start_method("spawn", force=True)


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


def _run_sparkle_worker(temp_dir, n_high_genes, r2_threshold, lambda_grid, max_radius, result_queue):
    """Run SPARKLE in an isolated subprocess on the subset written to ``temp_dir``."""
    import resource
    import sys
    from pathlib import Path

    from scipy.sparse import load_npz

    # Ensure imports work in spawn context.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

    from evaluation.scripts.final_comparison import run_sparkle_method

    temp_dir = Path(temp_dir)

    # Measure before data load.
    before_load_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    sub = {
        "dnb_expr": load_npz(str(temp_dir / "dnb_expr.npz")),
        "dnb_coords": np.load(temp_dir / "dnb_coords.npy"),
        "dnb_labels": np.load(temp_dir / "dnb_labels.npy"),
        "gene_names": np.load(temp_dir / "gene_names.npy"),
        "cell_ids": np.load(temp_dir / "cell_ids.npy"),
    }

    # Measure after data load.
    after_load_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    t0 = time.time()
    try:
        _, diag = run_sparkle_method(
            sub,
            n_high_genes=n_high_genes,
            r2_threshold=r2_threshold,
            lambda_grid=lambda_grid,
            max_radius=max_radius,
            verbose=False,
        )
        runtime = time.time() - t0

        # Measure after SPARKLE.
        after_sparkle_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        # Memory breakdown.
        data_mem_mb = max(0.0, (after_load_kb - before_load_kb) / 1024.0)
        method_mem_mb = max(0.0, (after_sparkle_kb - after_load_kb) / 1024.0)
        total_peak_mb = after_sparkle_kb / 1024.0

        status = "ok"
        selected_lambda = diag.get("lambda", float("nan"))
    except Exception as e:
        runtime = float("nan")
        data_mem_mb = float("nan")
        method_mem_mb = float("nan")
        total_peak_mb = float("nan")
        status = f"error: {e}"
        selected_lambda = float("nan")

    result_queue.put({
        "runtime_sec": runtime,
        "data_memory_mb": data_mem_mb,
        "peak_memory_mb": method_mem_mb,
        "total_peak_mb": total_peak_mb,
        "status": status,
        "lambda": selected_lambda,
    })


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
    ax.plot(df["n_dnbs"], df["total_peak_mb"], marker="o", color="#3498db", label="Total peak RSS")
    ax.plot(df["n_dnbs"], df["peak_memory_mb"], marker="s", color="#2ecc71", label="SPARKLE increment")
    ax.set_xlabel("Number of DNBs")
    ax.set_ylabel("Memory (MB)")
    ax.set_title("SPARKLE memory vs data size (MouseBrain)")
    ax.legend()
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

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            save_npz(tmpdir / "dnb_expr.npz", window["dnb_expr"])
            np.save(tmpdir / "dnb_coords.npy", window["dnb_coords"])
            np.save(tmpdir / "dnb_labels.npy", window["dnb_labels"])
            np.save(tmpdir / "gene_names.npy", window["gene_names"])
            np.save(tmpdir / "cell_ids.npy", window["cell_ids"])

            result_queue = mp.Queue()
            p = mp.Process(
                target=_run_sparkle_worker,
                args=(
                    str(tmpdir),
                    args.n_high_genes,
                    args.r2_threshold,
                    args.lambda_grid,
                    args.max_radius,
                    result_queue,
                ),
            )
            p.start()
            p.join()
            if p.exitcode != 0:
                result = {
                    "runtime_sec": float("nan"),
                    "data_memory_mb": float("nan"),
                    "peak_memory_mb": float("nan"),
                    "total_peak_mb": float("nan"),
                    "status": f"subprocess exited with code {p.exitcode}",
                    "lambda": float("nan"),
                }
            else:
                result = result_queue.get()

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
            "runtime_sec": result["runtime_sec"],
            "data_memory_mb": result["data_memory_mb"],
            "peak_memory_mb": result["peak_memory_mb"],
            "total_peak_mb": result["total_peak_mb"],
            "lambda": result["lambda"],
            "status": result["status"],
        })

    df = pd.DataFrame(results)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"\n{'=' * 60}")
    print(f"Saved CSV: {out_path}")
    print(f"{'=' * 60}")
    print(df.to_string(index=False))

    if args.plot:
        _plot_results(df, out_path.parent)


if __name__ == "__main__":
    main()
