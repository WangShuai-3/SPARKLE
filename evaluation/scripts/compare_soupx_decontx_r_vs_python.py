#!/usr/bin/env python3
"""Compare official R implementations (SoupX, celda::decontX) against the
Python ports (soupx-python, decontx-python) on synthetic data.

Generates synthetic DNB-level data, runs both implementations, and reports
per-gene / per-cell correlation and RMSE between the two versions.

Usage:
    python evaluation/scripts/compare_soupx_decontx_r_vs_python.py \
        [--scenario S1] [--seed 42] [--out-dir /tmp/r_vs_py_comparison]
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

import h5py
import numpy as np
from scipy.sparse import csr_matrix, csc_matrix

# Reuse the project's synthetic data generator and Python baselines
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from evaluation.synthetic import generate_synthetic_data
from evaluation.scripts.final_comparison import load_synthetic_scenario_data
from evaluation.baselines.soupx import run_soupx


def run_python_soupx(dnb_expr, dnb_labels, n_clusters=6, tfidf_min=0.1,
                     soup_quantile=0.4):
    """Run the Python soupx baseline.

    Parameters match the official-R invocation: on the synthetic scenarios
    (few cell types, sparse markers) the strict defaults (tfidfMin=1,
    soupQuantile=0.9, n/20 clusters) find zero marker genes in BOTH the R
    original and the Python port, so both are relaxed identically.
    """
    corrected, rho = run_soupx(dnb_expr, dnb_labels, n_clusters=n_clusters,
                               tfidf_min=tfidf_min, soup_quantile=soup_quantile,
                               verbose=True)
    return corrected, {"rho": rho}


def run_python_decontx(dnb_expr, dnb_labels, gene_names, cell_ids):
    """Run the Python decontx baseline (via final_comparison's impl)."""
    from evaluation.scripts.final_comparison import _decontx_impl
    sub = {
        "dnb_expr": dnb_expr,
        "dnb_labels": dnb_labels,
        "gene_names": gene_names,
        "cell_ids": cell_ids,
    }
    corrected, diag = _decontx_impl(sub, verbose=True)
    return corrected, diag


def export_for_r(dnb_expr, dnb_labels, gene_names, n_cells, out_dir):
    """Write DNB-level data in the format expected by the R scripts."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mat = csc_matrix(dnb_expr, dtype=np.float64)
    with h5py.File(out_dir / "counts_csc.h5", "w") as f:
        f.create_dataset("data", data=mat.data, compression="gzip")
        f.create_dataset("indices", data=mat.indices.astype(np.int32), compression="gzip")
        f.create_dataset("indptr", data=mat.indptr.astype(np.int64), compression="gzip")
        f.create_dataset("shape", data=np.asarray(mat.shape, dtype=np.int64))

    with open(out_dir / "genes.tsv", "w") as f:
        f.write("\n".join(str(g) for g in gene_names) + "\n")

    with open(out_dir / "cell_labels.tsv", "w") as f:
        f.write("\n".join(str(int(l)) for l in dnb_labels) + "\n")

    with open(out_dir / "n_cells.txt", "w") as f:
        f.write(str(n_cells) + "\n")

    return out_dir


DEFAULT_RSCRIPT = "/home/shuaiwang/miniconda3/envs/spotclean-official/bin/Rscript"


SOUPX_PARAMS = {"tfidf_min": 0.1, "soup_quantile": 0.4, "n_clusters": 6}


def run_r_script(script_name, input_dir, output_dir, rscript=DEFAULT_RSCRIPT,
                 extra_args=()):
    """Run an official R baseline script."""
    script_path = Path(__file__).resolve().parent / script_name
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    result = subprocess.run(
        [rscript, str(script_path), str(input_dir), str(output_dir),
         *[str(a) for a in extra_args]],
        capture_output=True, text=True, timeout=600,
    )
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"  R script stderr:\n{result.stderr[-2000:]}")
        raise RuntimeError(f"{script_name} failed (exit {result.returncode})")

    # Print R script messages
    for line in result.stderr.strip().split("\n"):
        if line.strip():
            print(f"  [R] {line.strip()}")

    # Read corrected matrix.  R/rhdf5 writes column-major, so a matrix that
    # was (genes × cells) in R is seen by h5py as (cells × genes).
    with h5py.File(output_dir / "decont.h5", "r") as f:
        corrected = f["X"][:].T

    # Read diagnostics
    diag = {}
    diag_file = output_dir / "diagnostics.tsv"
    if diag_file.exists():
        for line in diag_file.read_text().strip().split("\n")[1:]:
            parts = line.split("\t")
            if len(parts) == 2:
                diag[parts[0]] = parts[1]

    return corrected, diag, elapsed


def compare_matrices(py_mat, r_mat, true_expr, method_name):
    """Compute comparison metrics between Python and R results."""
    # Ensure same shape
    assert py_mat.shape == r_mat.shape, \
        f"Shape mismatch: Python {py_mat.shape} vs R {r_mat.shape}"

    results = {"method": method_name}

    # ── Python vs R comparison ──
    py_flat = py_mat.ravel()
    r_flat = r_mat.ravel()

    # Overall Pearson correlation
    from scipy.stats import pearsonr, spearmanr
    r_pearson, _ = pearsonr(py_flat, r_flat)
    r_spearman, _ = spearmanr(py_flat, r_flat)
    results["py_vs_r_pearson"] = r_pearson
    results["py_vs_r_spearman"] = r_spearman

    # RMSE between implementations
    results["py_vs_r_rmse"] = float(np.sqrt(np.mean((py_flat - r_flat) ** 2)))

    # Per-cell correlation
    n_cells = py_mat.shape[1]
    cell_corrs = []
    for c in range(n_cells):
        pc = py_mat[:, c]
        rc = r_mat[:, c]
        if pc.std() > 0 and rc.std() > 0:
            r, _ = pearsonr(pc, rc)
            cell_corrs.append(r)
    results["py_vs_r_mean_cell_pearson"] = float(np.mean(cell_corrs)) if cell_corrs else np.nan

    # ── Against ground truth ──
    if true_expr is not None:
        true_flat = true_expr.ravel()
        r_py_true, _ = pearsonr(py_flat, true_flat)
        r_r_true, _ = pearsonr(r_flat, true_flat)
        results["py_vs_true_pearson"] = r_py_true
        results["r_vs_true_pearson"] = r_r_true
        results["py_vs_true_rmse"] = float(np.sqrt(np.mean((py_flat - true_flat) ** 2)))
        results["r_vs_true_rmse"] = float(np.sqrt(np.mean((r_flat - true_flat) ** 2)))

    return results


def main():
    parser = argparse.ArgumentParser(description="Compare R vs Python SoupX/DecontX")
    parser.add_argument("--scenario", default="S1", help="Synthetic scenario ID")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="/tmp/r_vs_py_comparison")
    parser.add_argument("--rscript", default=DEFAULT_RSCRIPT,
                        help="Path to Rscript in the env with SoupX/celda")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Generate synthetic data ──
    print(f"Loading synthetic scenario {args.scenario} (seed={args.seed})...")
    data = load_synthetic_scenario_data(args.scenario, seed=args.seed)
    dnb_expr = data["dnb_expr"]
    dnb_labels = data["dnb_labels"]
    gene_names = data["gene_names"]
    cell_ids = data["cell_ids"]
    true_expr = data["true_expr"]
    n_cells = len(cell_ids)

    if hasattr(dnb_expr, "toarray"):
        dnb_expr_dense = dnb_expr.toarray()
    else:
        dnb_expr_dense = np.asarray(dnb_expr)

    print(f"  {dnb_expr_dense.shape[0]} genes × {dnb_expr_dense.shape[1]} DNBs, "
          f"{n_cells} cells, {(dnb_labels < 0).sum()} empty DNBs")

    # ── Export for R ──
    r_input_dir = out_dir / "r_input"
    export_for_r(dnb_expr, dnb_labels, gene_names, n_cells, r_input_dir)
    print(f"  Exported R input to {r_input_dir}")

    all_results = []

    # ── SoupX ──
    print("\n" + "=" * 60)
    print("SoupX: Python vs R")
    print("=" * 60)

    print("\n  Running Python soupx-python ...")
    t0 = time.time()
    py_corrected, py_diag = run_python_soupx(dnb_expr, dnb_labels)
    py_time = time.time() - t0
    print(f"    Done in {py_time:.1f}s, ρ={py_diag['rho']:.4f}")

    print("\n  Running R SoupX (official) ...")
    r_corrected, r_diag, r_time = run_r_script(
        "run_soupx_official.R", r_input_dir, out_dir / "r_soupx",
        rscript=args.rscript,
        extra_args=(SOUPX_PARAMS["tfidf_min"], SOUPX_PARAMS["soup_quantile"],
                    SOUPX_PARAMS["n_clusters"]),
    )
    print(f"    Done in {r_time:.1f}s, ρ={r_diag.get('rho', '?')}")

    results = compare_matrices(py_corrected, r_corrected, true_expr, "SoupX")
    results["py_runtime"] = py_time
    results["r_runtime"] = r_time
    results["py_rho"] = py_diag["rho"]
    results["r_rho"] = float(r_diag.get("rho", "nan"))
    all_results.append(results)

    # ── DecontX ──
    print("\n" + "=" * 60)
    print("DecontX: Python vs R")
    print("=" * 60)

    print("\n  Running Python decontx-python ...")
    t0 = time.time()
    py_corrected_dx, py_diag_dx = run_python_decontx(
        dnb_expr, dnb_labels, gene_names, cell_ids
    )
    py_time_dx = time.time() - t0
    print(f"    Done in {py_time_dx:.1f}s, "
          f"contamination={py_diag_dx.get('contamination', '?'):.4f}")

    print("\n  Running R celda::decontX (official) ...")
    r_corrected_dx, r_diag_dx, r_time_dx = run_r_script(
        "run_decontx_official.R", r_input_dir, out_dir / "r_decontx",
        rscript=args.rscript,
    )
    print(f"    Done in {r_time_dx:.1f}s, "
          f"contamination={r_diag_dx.get('mean_contamination', '?')}")

    results = compare_matrices(py_corrected_dx, r_corrected_dx, true_expr, "DecontX")
    results["py_runtime"] = py_time_dx
    results["r_runtime"] = r_time_dx
    results["py_contamination"] = py_diag_dx.get("contamination", np.nan)
    results["r_contamination"] = float(r_diag_dx.get("mean_contamination", "nan"))
    all_results.append(results)

    # ── Summary ──
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for res in all_results:
        print(f"\n  {res['method']}:")
        print(f"    Python vs R  Pearson r = {res['py_vs_r_pearson']:.6f}")
        print(f"    Python vs R  Spearman ρ = {res['py_vs_r_spearman']:.6f}")
        print(f"    Python vs R  RMSE = {res['py_vs_r_rmse']:.4f}")
        print(f"    Python vs R  mean per-cell r = {res['py_vs_r_mean_cell_pearson']:.6f}")
        if "py_vs_true_pearson" in res:
            print(f"    Python vs True  Pearson r = {res['py_vs_true_pearson']:.6f}")
            print(f"    R      vs True  Pearson r = {res['r_vs_true_pearson']:.6f}")
            print(f"    Python vs True  RMSE = {res['py_vs_true_rmse']:.4f}")
            print(f"    R      vs True  RMSE = {res['r_vs_true_rmse']:.4f}")
        print(f"    Python runtime = {res['py_runtime']:.1f}s")
        print(f"    R      runtime = {res['r_runtime']:.1f}s")
        if "py_rho" in res:
            print(f"    Python ρ = {res['py_rho']:.4f},  R ρ = {res['r_rho']:.4f}")
        if "py_contamination" in res:
            print(f"    Python contamination = {res['py_contamination']:.4f},  "
                  f"R contamination = {res['r_contamination']:.4f}")

    # Save results
    import json
    results_file = out_dir / "comparison_results.json"
    # Convert numpy types for JSON
    def _to_serializable(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    serializable = {k: {kk: _to_serializable(vv) for kk, vv in r.items()}
                    for k, r in enumerate(all_results)}
    results_file.write_text(json.dumps(serializable, indent=2))
    print(f"\n  Results saved to {results_file}")


if __name__ == "__main__":
    main()
