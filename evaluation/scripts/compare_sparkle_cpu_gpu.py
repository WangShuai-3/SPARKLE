#!/usr/bin/env python3
"""Compare CPU and requested-GPU SPARKLE on one deterministic synthetic dataset.

The script exits non-zero when a real GPU run disagrees with CPU beyond the
requested tolerance. On machines without CUDA it verifies the documented CPU
fallback and reports that a real CPU/GPU comparison was skipped. Pass
``--require-gpu`` in GPU CI or on a GPU node to make missing CUDA an error.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np

from evaluation.synthetic.generator import generate_synthetic_data
from stambient import SPARKLE


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-cells", type=int, default=100)
    parser.add_argument("--n-genes", type=int, default=128)
    parser.add_argument("--n-high-genes", type=int, default=48)
    parser.add_argument("--grid-size", type=int, default=120)
    parser.add_argument("--rtol", type=float, default=None)
    parser.add_argument("--atol", type=float, default=None)
    parser.add_argument(
        "--gpu-dtype", choices=["float64", "mixed", "float32"], default="float64",
        help="GPU precision mode (default: float64).",
    )
    parser.add_argument(
        "--gpu-gene-batch-size", type=int, default=None,
        help="Override adaptive GPU gene batching.",
    )
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="Fail instead of accepting the CPU fallback when CUDA is unavailable.",
    )
    return parser.parse_args()


def _run(data, use_gpu, n_high_genes, gpu_dtype="float64", gpu_gene_batch_size=None):
    model = SPARKLE(
        cell_based=True,
        use_gpu=use_gpu,
        gpu_dtype=gpu_dtype,
        gpu_gene_batch_size=gpu_gene_batch_size,
        bin_size=20,
        max_radius=100.0,
        n_high_genes=min(n_high_genes, data["dnb_expr"].shape[0]),
        n_lambda_genes=min(50, n_high_genes, data["dnb_expr"].shape[0]),
        lambda_grid=[10, 20, 30, 50, 70, 100],
        verbose=False,
    )
    start = time.perf_counter()
    corrected, diagnostics = model.fit_transform(
        data["dnb_expr"], data["dnb_coords"], data["dnb_labels"]
    )
    elapsed = time.perf_counter() - start
    if hasattr(corrected, "toarray"):
        corrected = corrected.toarray()
    return np.asarray(corrected), diagnostics, elapsed


def _max_relative_error(actual, expected, atol):
    scale = np.maximum(np.maximum(np.abs(actual), np.abs(expected)), atol)
    return float(np.max(np.abs(actual - expected) / scale)) if actual.size else 0.0


def main():
    args = _parse_args()
    default_tolerances = {
        "float64": (1e-8, 1e-10),
        "mixed": (2e-5, 1e-4),
        "float32": (1e-4, 1e-4),
    }
    default_rtol, default_atol = default_tolerances[args.gpu_dtype]
    rtol = default_rtol if args.rtol is None else args.rtol
    atol = default_atol if args.atol is None else args.atol
    data = generate_synthetic_data(
        n_cells=args.n_cells,
        grid_width=args.grid_size,
        grid_height=args.grid_size,
        n_genes=args.n_genes,
        n_high_genes=min(args.n_high_genes, args.n_genes),
        ambient_lambda=30.0,
        ambient_alpha=0.02,
        empty_fraction=0.30,
        n_cell_types=3,
        marker_fraction=0.20,
        seed=args.seed,
    )

    cpu, cpu_diag, cpu_seconds = _run(data, use_gpu=False, n_high_genes=args.n_high_genes)
    candidate, gpu_diag, candidate_seconds = _run(
        data,
        use_gpu=True,
        n_high_genes=args.n_high_genes,
        gpu_dtype=args.gpu_dtype,
        gpu_gene_batch_size=args.gpu_gene_batch_size,
    )

    backend = gpu_diag["compute_backend"]
    print(f"CPU backend: {cpu_diag['compute_backend']} ({cpu_seconds:.3f} s)")
    print(f"Requested GPU backend: {backend} ({candidate_seconds:.3f} s)")
    if gpu_diag.get("gpu_device"):
        print(f"GPU device: {gpu_diag['gpu_device']}")
        print(
            f"GPU dtype: {gpu_diag['gpu_dtype']}; CSR; "
            f"gene batch: {gpu_diag['gpu_gene_batch_size']}"
        )

    if backend != "gpu":
        reason = gpu_diag.get("gpu_fallback_reason") or "unknown reason"
        fallback_equal = np.array_equal(cpu, candidate)
        print(f"CUDA unavailable; CPU fallback used: {reason}")
        print(f"Fallback reproduces CPU exactly: {fallback_equal}")
        if not fallback_equal:
            return 1
        return 2 if args.require_gpu else 0

    alpha_close = np.allclose(
        cpu_diag["alphas"], gpu_diag["alphas"], rtol=rtol, atol=atol
    )
    r2_close = np.allclose(
        cpu_diag["r2_scores"], gpu_diag["r2_scores"], rtol=rtol, atol=atol
    )
    output_close = np.allclose(cpu, candidate, rtol=rtol, atol=atol)
    lambda_equal = cpu_diag["lambda_estimated"] == gpu_diag["lambda_estimated"]

    abs_error = float(np.max(np.abs(cpu - candidate))) if cpu.size else 0.0
    rel_error = _max_relative_error(cpu, candidate, atol)
    speedup = cpu_seconds / candidate_seconds if candidate_seconds > 0 else np.inf
    print(f"Estimated lambda equal: {lambda_equal}")
    print(f"Alpha allclose: {alpha_close}; R2 allclose: {r2_close}")
    print(f"Corrected matrix allclose: {output_close}")
    print(f"Max absolute error: {abs_error:.3e}")
    print(f"Max relative error: {rel_error:.3e}")
    print(f"End-to-end speedup: {speedup:.2f}x")
    print(f"Tolerance: rtol={rtol:g}, atol={atol:g}")
    print(f"GPU stage timings: {gpu_diag.get('timings_sec', {})}")

    reliable = lambda_equal and alpha_close and r2_close and output_close
    print(f"CPU/GPU consistency: {'PASS' if reliable else 'FAIL'}")
    return 0 if reliable else 1


if __name__ == "__main__":
    raise SystemExit(main())
