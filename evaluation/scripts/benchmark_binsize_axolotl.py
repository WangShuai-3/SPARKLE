#!/usr/bin/env python3
"""Benchmark SPARKLE bin-size sensitivity on the Axolotl SST-diffusion task.

Modelled after ``benchmark_resource.py``: the (expensive) Axolotl GEM is loaded
ONCE for a fixed spatial window, then SPARKLE is run repeatedly for a range of
empty-bin sizes.  For each bin size we record the SST specificity metric
``s/N`` (sstIN / Neighbour mean expression ratio) — the higher the ratio, the
better ambient SST signal is confined to the true SST interneurons rather than
leaking onto neighbouring cells.

Bin sizes are specified and passed to SPARKLE in micrometres. Axolotl
Stereo-seq coordinates are multiplied by the 0.5-µm DNB pitch before fitting.
The default sweep is 10–50 µm in 5 µm steps. Only SPARKLE is run.

Usage:
    python evaluation/scripts/benchmark_binsize_axolotl.py \
        --x-range 10500 12500 --y-range 6000 11100 \
        --n-genes 200 --plot
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root to path so `evaluation.*` imports work when running standalone.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from stambient import SPARKLE
from evaluation.scripts.final_comparison import (
    STEREOSEQ_PITCH_UM,
    load_axolotl_data_windowed,
    compute_neighbor_stats,
)

SST_GENE = "AMEX60DD003175"
DNB_PITCH_UM = STEREOSEQ_PITCH_UM  # Axolotl Stereo-seq DNB pitch
NEIGHBOR_RADIUS_UM = 50.0


def _prepare_axolotl_eval(data):
    """Precompute the cell-level SST evaluation masks (done once, method-agnostic).

    Returns a dict with everything needed to score a corrected DNB matrix:
    sst_idx, labels_0based (DNB -> 0-based cell), sstin/neighbor/other masks and
    the raw per-cell SST vector for retain/remove computation.
    """
    dnb_coords_um = data["dnb_coords"] * DNB_PITCH_UM
    dnb_labels = data["dnb_labels"]
    gene_names = list(data["gene_names"])
    cell_ids = np.array(data["cell_ids"])
    sstin_set = data["sstin_set"]
    n_cells = len(cell_ids)

    sstin_mask = np.array([cell_ids[i] in sstin_set for i in range(n_cells)])

    # Cell centroids -> neighbour / other masks (radius default 50 µm)
    cc = np.zeros((n_cells, 2))
    for c in range(n_cells):
        m = dnb_labels == cell_ids[c]
        if m.sum():
            cc[c] = dnb_coords_um[m].mean(axis=0)
    neighbor_mask, other_mask = compute_neighbor_stats(
        cc, sstin_mask, radius=NEIGHBOR_RADIUS_UM
    )

    sst_idx = gene_names.index(SST_GENE)

    # DNB -> 0-based cell index aggregation matrix
    label_to_idx = {cid: i for i, cid in enumerate(cell_ids)}
    labels_0based = np.array([label_to_idx.get(l, -1) for l in dnb_labels])
    valid = labels_0based >= 0
    C = csr_matrix(
        (np.ones(valid.sum()), (np.where(valid)[0], labels_0based[valid])),
        shape=(data["dnb_expr"].shape[1], n_cells),
    )
    raw_cell = (data["dnb_expr"] @ C).toarray()
    sst_raw = raw_cell[sst_idx]

    return {
        "sst_idx": sst_idx,
        "sstin_mask": sstin_mask,
        "neighbor_mask": neighbor_mask,
        "other_mask": other_mask,
        "sst_raw": sst_raw,
        "n_cells": n_cells,
    }


def _score(sst_vals, ev):
    """Compute the SST specificity metrics for a per-cell SST vector."""
    si = float(sst_vals[ev["sstin_mask"]].mean())
    sn = float(sst_vals[ev["neighbor_mask"]].mean())
    so = float(sst_vals[ev["other_mask"]].mean())
    r_si = float(ev["sst_raw"][ev["sstin_mask"]].mean())
    r_sn = float(ev["sst_raw"][ev["neighbor_mask"]].mean())
    return {
        "sstIN": si,
        "sstNbr": sn,
        "sstOth": so,
        "s_over_N": si / sn if sn > 0 else float("nan"),
        "N_over_O": sn / so if so > 0 else float("nan"),
        "retain_pct": si / r_si * 100 if r_si > 0 else float("nan"),
        "remove_pct": (1 - sn / r_sn) * 100 if r_sn > 0 else float("nan"),
    }


def _run_sparkle_binsize(data, bin_size_um, n_high_genes, lambda_grid,
                         r2_threshold, max_radius):
    """Run SPARKLE for one bin size; return corrected matrix and diagnostics."""
    model = SPARKLE(
        bin_size=bin_size_um,
        max_radius=max_radius,
        n_high_genes=n_high_genes,
        n_lambda_genes=min(100, data["dnb_expr"].shape[0]),
        r2_threshold=r2_threshold,
        lambda_grid=lambda_grid,
        cell_based=True,
        verbose=False,
    )
    t0 = time.time()
    corrected, diag = model.fit_transform_from_dnb(
        data["dnb_expr"],
        data["dnb_coords"] * DNB_PITCH_UM,
        data["dnb_labels"],
    )
    runtime = time.time() - t0
    if hasattr(corrected, "toarray"):
        corrected = corrected.toarray()
    return corrected, diag, runtime, float(model.lambda_)


def _plot_results(df, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = df.sort_values("bin_size_um")

    # Main figure: s/N vs bin size (the requested metric)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["bin_size_um"], df["s_over_N"], marker="o", color="#8e44ad", lw=2)
    if "raw_s_over_N" in df.columns and df["raw_s_over_N"].notna().any():
        ax.axhline(df["raw_s_over_N"].iloc[0], color="#7f8c8d", ls="--",
                   label=f"Raw ({df['raw_s_over_N'].iloc[0]:.2f}x)")
        ax.legend()
    ax.set_xlabel("Empty-bin size (µm)")
    ax.set_ylabel("SST specificity  s/N  (sstIN / Neighbour)")
    ax.set_title("SPARKLE bin-size sensitivity on Axolotl SST diffusion")
    ax.grid(True, alpha=0.3)
    for _, r in df.iterrows():
        ax.annotate(f"{r['s_over_N']:.2f}", (r["bin_size_um"], r["s_over_N"]),
                    textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
    plt.tight_layout()
    fig.savefig(output_dir / "binsize_axolotl_sN.png", dpi=150)
    plt.close(fig)

    # Secondary figure: retain% and remove% vs bin size
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["bin_size_um"], df["retain_pct"], marker="o", color="#27ae60", label="Signal retained %")
    ax.plot(df["bin_size_um"], df["remove_pct"], marker="s", color="#e67e22", label="Neighbour removed %")
    ax.set_xlabel("Empty-bin size (µm)")
    ax.set_ylabel("Percent")
    ax.set_title("SPARKLE retain / remove vs bin size (Axolotl)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_dir / "binsize_axolotl_retain_remove.png", dpi=150)
    plt.close(fig)

    print(f"Saved plots in {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark SPARKLE bin-size sensitivity on Axolotl SST task."
    )
    parser.add_argument("--x-range", type=int, nargs=2, default=[10500, 12500],
                        help="X range (default: 10500 12500)")
    parser.add_argument("--y-range", type=int, nargs=2, default=[6000, 11100],
                        help="Y range (default: 6000 11100)")
    parser.add_argument("--n-genes", type=int, default=200,
                        help="Top-N genes SPARKLE corrects (default: 200)")
    parser.add_argument("--bin-sizes-um", type=float, nargs="+", default=None,
                        help="Bin sizes in µm (default: 10 15 20 25 30 35 40 45 50)")
    parser.add_argument("--lambda-grid", type=int, nargs="+",
                        default=[10, 20, 30, 50, 70, 100, 150, 200],
                        help="Lambda candidates in µm (default axolotl grid)")
    parser.add_argument("--r2-threshold", type=float, default=0.01,
                        help="R^2 threshold for SPARKLE gene correction (default: 0.01)")
    parser.add_argument("--max-radius", type=float, default=200.0,
                        help="Spatial neighbourhood radius in µm (default: 200)")
    parser.add_argument("--output", type=str,
                        default="evaluation/reports/binsize_axolotl/binsize_axolotl_sensitivity.csv",
                        help="Output CSV path")
    parser.add_argument("--plot", action="store_true", help="Generate line plots")
    args = parser.parse_args()

    x_range = tuple(args.x_range)
    y_range = tuple(args.y_range)
    bin_sizes_um = args.bin_sizes_um if args.bin_sizes_um is not None \
        else list(np.arange(10, 51, 5).astype(float))
    n_high = args.n_genes

    print("=" * 60)
    print("BIN-SIZE SENSITIVITY: SPARKLE on Axolotl SST diffusion")
    print(f"  Window: x={x_range}, y={y_range}")
    print(f"  Bin sizes (µm): {bin_sizes_um}")
    print(f"  n-genes (corrected): {args.n_genes}")
    print("=" * 60)

    # 1. Load Axolotl data once (expensive).
    print("\n[1/3] Loading Axolotl window...")
    t_load = time.time()
    data = load_axolotl_data_windowed(x_range, y_range)
    print(f"  Loaded in {time.time() - t_load:.0f}s")

    # 2. Precompute evaluation masks once.
    print("\n[2/3] Preparing SST evaluation masks...")
    ev = _prepare_axolotl_eval(data)
    raw_metrics = _score(ev["sst_raw"], ev)
    print(f"  Raw: sstIN={raw_metrics['sstIN']:.1f} Nbr={raw_metrics['sstNbr']:.1f} "
          f"Oth={raw_metrics['sstOth']:.1f} s/N={raw_metrics['s_over_N']:.2f}x")

    # 3. Sweep bin sizes.
    print("\n[3/3] Sweeping bin sizes...")
    results = []
    for bs_um in bin_sizes_um:
        bs_dnb = max(1, int(round(bs_um / DNB_PITCH_UM)))
        corrected, diag, runtime, lam = _run_sparkle_binsize(
            data, bs_um, n_high, args.lambda_grid, args.r2_threshold, args.max_radius,
        )
        m = _score(corrected[ev["sst_idx"]], ev)
        row = {
            "bin_size_um": bs_um,
            "bin_size_dnb": bs_dnb,
            "lambda_um": lam,
            "runtime_sec": runtime,
            "raw_s_over_N": raw_metrics["s_over_N"],
            **m,
        }
        results.append(row)
        print(f"  bin={bs_um:>4.0f}µm ({bs_dnb:>3d} DNB): "
              f"sstIN={m['sstIN']:6.1f} Nbr={m['sstNbr']:5.2f} "
              f"s/N={m['s_over_N']:6.2f}x  retain={m['retain_pct']:5.1f}% "
              f"remove={m['remove_pct']:5.1f}%  λ={lam:.0f}µm  ({runtime:.0f}s)")

    df = pd.DataFrame(results)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"\n{'=' * 60}")
    print(f"Saved CSV: {out_path}")
    print(f"{'=' * 60}")
    print(df[["bin_size_um", "bin_size_dnb", "sstIN", "sstNbr", "s_over_N",
              "retain_pct", "remove_pct", "lambda_um", "runtime_sec"]].to_string(index=False))

    if args.plot:
        _plot_results(df, out_path.parent)


if __name__ == "__main__":
    main()
