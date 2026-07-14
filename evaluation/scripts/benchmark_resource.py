#!/usr/bin/env python3
"""Benchmark CPU/GPU SPARKLE runtime and memory on MouseBrain real data.

Loads the MouseBrain GEM once at the user-specified maximum spatial window,
then extracts progressively smaller windows by arithmetically shrinking the
side lengths (each step subtracts an equal fraction of the original side length).
This avoids the repeated cost of parsing the raw GEM file.

Each spatial window can run on CPU, GPU, or both. In paired mode the output
contains end-to-end speedup, process RSS, GPU peak allocated memory, and GPU
peak reserved memory. ``r2_threshold`` can be set to 0 to measure pure speed
without any gene filtering.

Host RSS is sampled with ``psutil``. When ``memory_profiler`` is installed its
line-by-line output is also saved under ``evaluation/reports/resource_profiles/``.
CUDA memory comes from PyTorch's peak allocator counters.

Usage example:
    python evaluation/scripts/benchmark_resource.py \
        --x-range 6000 20000 \
        --y-range 2000 15000 \
        --n-genes 10000 \
        --n-high-genes 10000 \
        --r2-threshold 0 \
        --n-runs 8 \
        --backends cpu gpu
"""

import argparse
import gc
import io
import sys
import threading
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
from stambient.gpu import resolve_gpu


MIB = 1024 ** 2


class _RSSMonitor:
    """Sample this process' RSS in a background thread during one model run."""

    def __init__(self, interval_sec=0.02):
        self.interval_sec = interval_sec
        self.baseline_mb = float("nan")
        self.peak_mb = float("nan")
        self._process = None
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        try:
            import psutil

            self._process = psutil.Process()
            self.baseline_mb = self._process.memory_info().rss / MIB
            self.peak_mb = self.baseline_mb
            self._thread = threading.Thread(target=self._sample, daemon=True)
            self._thread.start()
        except ImportError:
            pass

    def _sample(self):
        while not self._stop.wait(self.interval_sec):
            try:
                rss_mb = self._process.memory_info().rss / MIB
                self.peak_mb = max(self.peak_mb, rss_mb)
            except Exception:
                return

    def stop(self):
        if self._thread is None:
            return
        try:
            rss_mb = self._process.memory_info().rss / MIB
            self.peak_mb = max(self.peak_mb, rss_mb)
        except Exception:
            pass
        self._stop.set()
        self._thread.join(timeout=max(1.0, self.interval_sec * 5))

    @property
    def increment_mb(self):
        if not np.isfinite(self.baseline_mb) or not np.isfinite(self.peak_mb):
            return float("nan")
        return max(self.peak_mb - self.baseline_mb, 0.0)


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


def _run_sparkle_profiled(
    sub,
    n_high_genes,
    r2_threshold,
    lambda_grid,
    max_radius,
    use_gpu=False,
    gpu_context=None,
    gpu_dtype="float64",
    gpu_gene_batch_size=None,
):
    """Run SPARKLE in the main process and capture memory_profiler output.

    Returns:
        (runtime_sec, diag, profile_text, parsed_mem_dict, gpu_mem_dict, status)
    """
    old_stdout = sys.stdout
    captured = io.StringIO()
    rss_monitor = _RSSMonitor()
    gpu_mem = {
        "baseline_allocated_mb": float("nan"),
        "peak_allocated_mb": float("nan"),
        "allocated_increment_mb": float("nan"),
        "peak_reserved_mb": float("nan"),
    }

    if use_gpu and gpu_context is not None:
        torch = gpu_context.torch
        device = gpu_context.device
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        gpu_mem["baseline_allocated_mb"] = torch.cuda.memory_allocated(device) / MIB

    t0 = time.time()
    rss_monitor.start()
    try:
        sys.stdout = captured
        corrected, diag = run_sparkle_method(
            sub,
            n_high_genes=n_high_genes,
            r2_threshold=r2_threshold,
            lambda_grid=lambda_grid,
            max_radius=max_radius,
            verbose=False,
            use_gpu=use_gpu,
            gpu_dtype=gpu_dtype,
            gpu_gene_batch_size=gpu_gene_batch_size,
        )
        if use_gpu and gpu_context is not None:
            gpu_context.torch.cuda.synchronize(gpu_context.device)
        runtime = time.time() - t0
        if corrected is None or diag.get("error"):
            status = f"error: {diag.get('error', 'SPARKLE returned no result')}"
        elif use_gpu and diag.get("compute_backend") != "gpu":
            status = "fallback"
        else:
            status = "ok"
    except Exception as e:
        runtime = time.time() - t0
        diag = {}
        status = f"error: {e}"
    finally:
        sys.stdout = old_stdout
        rss_monitor.stop()

    if use_gpu and gpu_context is not None:
        torch = gpu_context.torch
        device = gpu_context.device
        try:
            torch.cuda.synchronize(device)
            gpu_mem["peak_allocated_mb"] = torch.cuda.max_memory_allocated(device) / MIB
            gpu_mem["peak_reserved_mb"] = torch.cuda.max_memory_reserved(device) / MIB
            gpu_mem["allocated_increment_mb"] = max(
                gpu_mem["peak_allocated_mb"] - gpu_mem["baseline_allocated_mb"], 0.0
            )
        except Exception:
            pass

    profile_text = captured.getvalue()
    mem_stats = _parse_memory_profile(profile_text)
    mem_stats.update({
        "rss_baseline_mb": rss_monitor.baseline_mb,
        "rss_peak_mb": rss_monitor.peak_mb,
        "rss_increment_mb": rss_monitor.increment_mb,
    })

    # Print status back to real stdout.
    if status in {"ok", "fallback"}:
        backend = diag.get("compute_backend", "unknown")
        message = (
            f"    Done in {runtime:.1f}s ({backend}), "
            f"λ={diag.get('lambda', float('nan')):.0f}μm, "
            f"RSS peak +{mem_stats['rss_increment_mb']:.1f} MiB"
        )
        if use_gpu:
            message += f", GPU peak {gpu_mem['peak_allocated_mb']:.1f} MiB allocated"
        if status == "fallback":
            message += f" [fallback: {diag.get('gpu_fallback_reason', 'unknown')}]"
        print(message)
    else:
        print(f"    ERROR: {status}")

    return runtime, diag, profile_text, mem_stats, gpu_mem, status


def _plot_results(df, output_dir):
    """Generate paired runtime, speedup, and memory plots."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = df.sort_values("n_dnbs")

    fig, ax = plt.subplots(figsize=(8, 5))
    if "cpu_runtime_sec" in df:
        ax.plot(df["n_dnbs"], df["cpu_runtime_sec"], marker="o", label="CPU")
    if "gpu_runtime_sec" in df:
        ax.plot(df["n_dnbs"], df["gpu_runtime_sec"], marker="o", label="GPU")
    ax.set_xlabel("Number of DNBs")
    ax.set_ylabel("Runtime (s)")
    ax.set_title("SPARKLE CPU/GPU runtime vs data size (MouseBrain)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_dir / "resource_benchmark_runtime.png", dpi=150)
    plt.close(fig)

    if "gpu_speedup" in df and df["gpu_speedup"].notna().any():
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(df["n_dnbs"], df["gpu_speedup"], marker="o", color="#27ae60")
        ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
        ax.set_xlabel("Number of DNBs")
        ax.set_ylabel("CPU runtime / GPU runtime")
        ax.set_title("SPARKLE GPU speedup (MouseBrain)")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(output_dir / "resource_benchmark_gpu_speedup.png", dpi=150)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    if "cpu_host_rss_increment_mb" in df:
        ax.plot(df["n_dnbs"], df["cpu_host_rss_increment_mb"], marker="o", label="CPU RSS")
    if "gpu_peak_allocated_mb" in df:
        ax.plot(df["n_dnbs"], df["gpu_peak_allocated_mb"], marker="o", label="GPU allocated")
        ax.plot(df["n_dnbs"], df["gpu_peak_reserved_mb"], marker="o", label="GPU reserved")
    ax.set_xlabel("Number of DNBs")
    ax.set_ylabel("Memory (MB)")
    ax.set_title("SPARKLE CPU/GPU peak memory vs data size (MouseBrain)")
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
        "--profile-dir", type=str,
        default="evaluation/reports/resource_profiles",
        help="Directory to save per-run memory_profiler text outputs",
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Generate runtime/memory line plots",
    )
    parser.add_argument(
        "--backends", nargs="+", choices=["cpu", "gpu"], default=["cpu", "gpu"],
        help="Backends to benchmark (default: cpu gpu)",
    )
    parser.add_argument(
        "--require-gpu", action="store_true",
        help="Exit before loading data when a requested GPU is unavailable",
    )
    parser.add_argument(
        "--cpu-baseline", type=str, default=None,
        help="Reuse an existing CPU benchmark CSV instead of rerunning CPU",
    )
    parser.add_argument(
        "--gpu-dtype", choices=["float64", "mixed", "float32"], default="float64",
        help="GPU precision mode (default: float64)",
    )
    parser.add_argument(
        "--gpu-gene-batch-size", type=int, default=None,
        help="Override adaptive GPU gene batch size",
    )
    args = parser.parse_args()

    x_range = tuple(args.x_range)
    y_range = tuple(args.y_range)
    backends = list(dict.fromkeys(args.backends))
    cpu_baseline = pd.read_csv(args.cpu_baseline) if args.cpu_baseline else None
    if cpu_baseline is not None and len(cpu_baseline) != args.n_runs:
        parser.error(
            f"CPU baseline contains {len(cpu_baseline)} runs, expected {args.n_runs}"
        )

    gpu_context = None
    gpu_unavailable_reason = None
    if "gpu" in backends:
        gpu_context, gpu_unavailable_reason = resolve_gpu(True)
        if gpu_context is None and args.require_gpu:
            parser.error(f"GPU requested but unavailable: {gpu_unavailable_reason}")

    print("=" * 60)
    print("RESOURCE BENCHMARK: SPARKLE on MouseBrain")
    print(f"  Max window: x={x_range}, y={y_range}")
    print(f"  Genes: {args.n_genes}, high genes: {args.n_high_genes}")
    print(f"  R2 threshold: {args.r2_threshold}")
    print(f"  Arithmetic shrink steps: {args.n_runs}")
    print(f"  Backends: {', '.join(backends)}")
    if args.cpu_baseline:
        print(f"  CPU baseline: {args.cpu_baseline}")
    if gpu_context is not None:
        print(f"  GPU: {gpu_context.name} ({args.gpu_dtype}, adaptive batch={args.gpu_gene_batch_size is None})")
    elif "gpu" in backends:
        print(f"  GPU unavailable: {gpu_unavailable_reason}")
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

        row = {
            "run": i + 1,
            "x_min": x_range_i[0],
            "x_max": x_range_i[1],
            "y_min": y_range_i[0],
            "y_max": y_range_i[1],
            "x_width": x_range_i[1] - x_range_i[0],
            "y_height": y_range_i[1] - y_range_i[0],
            "physical_width_um": (x_range_i[1] - x_range_i[0]) * 0.5,
            "physical_height_um": (y_range_i[1] - y_range_i[0]) * 0.5,
            "area_um2": (x_range_i[1] - x_range_i[0]) * (y_range_i[1] - y_range_i[0]) * 0.25,
            "area_mm2": (x_range_i[1] - x_range_i[0]) * (y_range_i[1] - y_range_i[0]) * 0.25 / 1e6,
            "n_dnbs": window["dnb_expr"].shape[1],
            "n_genes": window["dnb_expr"].shape[0],
            "n_cells": n_cells,
            "n_empty_dnbs": n_empty,
        }

        if cpu_baseline is not None:
            baseline_row = cpu_baseline.iloc[i]
            for key, actual in (
                ("x_min", x_range_i[0]),
                ("x_max", x_range_i[1]),
                ("y_min", y_range_i[0]),
                ("y_max", y_range_i[1]),
                ("n_genes", window["dnb_expr"].shape[0]),
                ("n_dnbs", window["dnb_expr"].shape[1]),
                ("n_cells", n_cells),
                ("n_empty_dnbs", n_empty),
            ):
                if key not in baseline_row or not np.isclose(baseline_row[key], actual):
                    raise ValueError(
                        f"CPU baseline run {i + 1} has {key}={baseline_row.get(key)}, "
                        f"but current benchmark has {actual}"
                    )
            baseline_peak = float(baseline_row.get("sparkle_peak_mb", float("nan")))
            baseline_rss = float(baseline_row.get("baseline_mb", float("nan")))
            row.update({
                "cpu_runtime_sec": float(baseline_row["runtime_sec"]),
                "cpu_host_rss_baseline_mb": baseline_rss,
                "cpu_host_rss_peak_mb": baseline_rss + baseline_peak,
                "cpu_host_rss_increment_mb": baseline_peak,
                "cpu_profiler_increment_mb": baseline_peak,
                "cpu_lambda": float(baseline_row.get("lambda", float("nan"))),
                "cpu_compute_backend": "cpu",
                "cpu_status": str(baseline_row.get("status", "ok")),
                "cpu_baseline_source": str(args.cpu_baseline),
            })

        for backend in backends:
            if backend == "cpu" and cpu_baseline is not None:
                continue
            if backend == "gpu" and gpu_context is None:
                row.update({
                    "gpu_runtime_sec": float("nan"),
                    "gpu_host_rss_increment_mb": float("nan"),
                    "gpu_peak_allocated_mb": float("nan"),
                    "gpu_peak_reserved_mb": float("nan"),
                    "gpu_lambda": float("nan"),
                    "gpu_compute_backend": "unavailable",
                    "gpu_status": f"unavailable: {gpu_unavailable_reason}",
                })
                continue

            print(f"    [{backend.upper()}]")
            runtime, diag, profile_text, mem_stats, gpu_mem, status = _run_sparkle_profiled(
                window,
                n_high_genes=args.n_high_genes,
                r2_threshold=args.r2_threshold,
                lambda_grid=args.lambda_grid,
                max_radius=args.max_radius,
                use_gpu=(backend == "gpu"),
                gpu_context=gpu_context if backend == "gpu" else None,
                gpu_dtype=args.gpu_dtype,
                gpu_gene_batch_size=args.gpu_gene_batch_size,
            )

            profile_path = profile_dir / f"run_{i + 1}_{backend}.txt"
            profile_path.write_text(profile_text, encoding="utf-8")
            row.update({
                f"{backend}_runtime_sec": runtime,
                f"{backend}_host_rss_baseline_mb": mem_stats["rss_baseline_mb"],
                f"{backend}_host_rss_peak_mb": mem_stats["rss_peak_mb"],
                f"{backend}_host_rss_increment_mb": mem_stats["rss_increment_mb"],
                f"{backend}_profiler_increment_mb": mem_stats["sparkle_increment_mb"],
                f"{backend}_lambda": diag.get("lambda", float("nan")),
                f"{backend}_compute_backend": diag.get("compute_backend", backend),
                f"{backend}_status": status,
                f"{backend}_empty_to_cell_graph_nnz": diag.get("empty_to_cell_graph_nnz", float("nan")),
                f"{backend}_cell_to_cell_graph_nnz": diag.get("cell_to_cell_graph_nnz", float("nan")),
            })
            for stage, seconds in diag.get("timings_sec", {}).items():
                row[f"{backend}_{stage}"] = seconds
            if backend == "gpu":
                row.update({
                    "gpu_peak_allocated_mb": gpu_mem["peak_allocated_mb"],
                    "gpu_peak_reserved_mb": gpu_mem["peak_reserved_mb"],
                    "gpu_allocated_increment_mb": gpu_mem["allocated_increment_mb"],
                    "gpu_device": gpu_context.name,
                    "gpu_dtype": diag.get("gpu_dtype", args.gpu_dtype),
                    "gpu_gene_batch_size": diag.get("gpu_gene_batch_size"),
                    "gpu_sparse_format": diag.get("gpu_sparse_format"),
                })

        cpu_runtime = row.get("cpu_runtime_sec", float("nan"))
        gpu_runtime = row.get("gpu_runtime_sec", float("nan"))
        gpu_is_valid = row.get("gpu_compute_backend") == "gpu" and row.get("gpu_status") == "ok"
        row["gpu_speedup"] = (
            cpu_runtime / gpu_runtime
            if gpu_is_valid and np.isfinite(cpu_runtime) and gpu_runtime > 0
            else float("nan")
        )
        row["lambda_match"] = (
            bool(row.get("cpu_lambda") == row.get("gpu_lambda"))
            if gpu_is_valid and "cpu_lambda" in row
            else None
        )

        # Backward-compatible aliases for the original CPU-only CSV columns.
        if "cpu_runtime_sec" in row:
            row.update({
                "runtime_sec": row["cpu_runtime_sec"],
                "baseline_mb": row["cpu_host_rss_baseline_mb"],
                "sparkle_peak_mb": row["cpu_host_rss_increment_mb"],
                "max_increment_mb": row["cpu_profiler_increment_mb"],
                "lambda": row["cpu_lambda"],
                "status": row["cpu_status"],
            })
        results.append(row)

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
