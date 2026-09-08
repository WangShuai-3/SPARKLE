#!/usr/bin/env python3
"""Run one correction method on one MouseBrain window (subprocess worker).

Used by benchmark_methods_mousebrain.py: each (method, window) is executed in
its own subprocess so wall-clock runtime and peak RSS are isolated.

Usage:
    python benchmark_methods_mousebrain_worker.py \
        <method> <x_min> <x_max> <y_min> <y_max> <out_prefix>
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np

from evaluation.scripts.final_comparison import (
    load_mousebrain_data,
    subsample_data,
    run_sparkle_method,
    run_soupx_method,
    run_decontx_method,
)
from evaluation.scripts.benchmark_resource import _extract_window

N_GENES = 10000
N_HIGH = 10000


def _measure(data, x_range, y_range, method):
    sub = subsample_data(data, n_genes=N_GENES, cut_genes=True)
    window = _extract_window(sub, x_range, y_range)
    t0 = time.time()
    if method == "sparkle":
        corrected, diag = run_sparkle_method(
            window, n_high_genes=N_HIGH, r2_threshold=0.0,
            lambda_grid=[10, 20, 30, 50, 70, 100, 150, 200, 300],
            max_radius=300.0, verbose=False,
        )
    elif method == "soupx":
        corrected, diag = run_soupx_method(window, verbose=False)
    elif method == "decontx":
        corrected, diag = run_decontx_method(window, verbose=False)
    else:
        raise ValueError(f"unknown method {method}")
    elapsed = time.time() - t0
    lam = diag.get("lambda", diag.get("rho", float("nan")))
    print(f"RESULT {method} runtime={elapsed:.3f} diag={lam}")
    return elapsed


def main():
    method, x0, x1, y0, y1, out_prefix = sys.argv[1:7]
    baseline_impl = sys.argv[7] if len(sys.argv) > 7 else "python"
    import evaluation.scripts.final_comparison as _fc
    _fc.BASELINE_IMPL = baseline_impl
    x_range = (float(x0), float(x1))
    y_range = (float(y0), float(y1))
    print(f"[worker] {method} window x={x_range} y={y_range} impl={baseline_impl}", flush=True)

    data = load_mousebrain_data(x_range=x_range, y_range=y_range)
    elapsed = _measure(data, x_range, y_range, method)

    out = Path(out_prefix)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.write(f"method={method}\n")
        f.write(f"x_range={x0},{x1}\n")
        f.write(f"y_range={y0},{y1}\n")
        f.write(f"runtime_sec={elapsed:.4f}\n")


if __name__ == "__main__":
    main()
