#!/usr/bin/env python3
"""Single-variable ablation: which generator change biased SPARKLE's lambda?

History: SPARKLE estimated the true lambda (S1=50, S5=500) exactly up to the
expression-gradient commit, but estimates one grid step low (30) since the
multi-type/imbalanced/spatial-clustering commit.  That commit changed three
things at once:

  (a) n_cell_types 3 -> 6
  (b) type_size_ratio 1.0 -> 0.5 (imbalanced proportions)
  (c) cluster_strength became effective (quota-based spatial domains)

This script generates S1-like data (no dropout) toggling one factor at a
time and records SPARKLE's estimated lambda over two seeds.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stambient import SPARKLE
from evaluation.synthetic import generate_synthetic_data


CONFIGS = [
    ("A0 3types/balanced/random",      dict(n_cell_types=3, type_size_ratio=1.0, cluster_strength=0.0)),
    ("A1 6types/balanced/random",      dict(n_cell_types=6, type_size_ratio=1.0, cluster_strength=0.0)),
    ("A2 3types/imbalanced/random",    dict(n_cell_types=3, type_size_ratio=0.5, cluster_strength=0.0)),
    ("A3 3types/balanced/clustered",   dict(n_cell_types=3, type_size_ratio=1.0, cluster_strength=0.6)),
    ("A4 3types/imbalanced/clustered", dict(n_cell_types=3, type_size_ratio=0.5, cluster_strength=0.6)),
    ("A5 6types/imbalanced/clustered (current)", dict(n_cell_types=6, type_size_ratio=0.5, cluster_strength=0.6)),
]
SEEDS = [42, 7]
TRUE_LAMBDA = 50.0


def run_one(name, kw, seed):
    data = generate_synthetic_data(
        n_cells=600, grid_width=500, grid_height=500, dnb_pitch=0.5,
        cell_radius=5.0, cell_radius_cv=0.2, n_genes=500, n_high_genes=80,
        ambient_lambda=TRUE_LAMBDA, ambient_alpha=0.01,
        empty_fraction=0.40, marker_fraction=0.20,
        dropout_rate=0.0, seed=seed, **kw,
    )
    model = SPARKLE(
        bin_size=25, distance_metric="exponential", max_radius=300.0,
        n_high_genes=80, n_lambda_genes=50, r2_threshold=0.01,
        lambda_grid=[10, 20, 30, 50, 70, 100, 150, 200, 300, 500],
        cell_based=True, self_confidence_penalty=True, verbose=False,
    )
    corrected, diag = model.fit_transform_from_dnb(
        data["dnb_expr"], data["dnb_coords"], data["dnb_labels"]
    )
    return {
        "config": name, "seed": seed,
        "lambda_est": float(diag["lambda_estimated"]),
        "true_lambda": TRUE_LAMBDA,
        "n_types": kw["n_cell_types"],
        "type_size_ratio": kw["type_size_ratio"],
        "cluster_strength": kw["cluster_strength"],
    }


def main():
    rows = []
    t0 = time.time()
    for name, kw in CONFIGS:
        for seed in SEEDS:
            row = run_one(name, kw, seed)
            rows.append(row)
            print(f"[{time.time() - t0:6.0f}s] {name} seed={seed}: "
                  f"λ={row['lambda_est']:.0f} (真值 {TRUE_LAMBDA:.0f})", flush=True)

    df = pd.DataFrame(rows)
    out = PROJECT_ROOT / "evaluation" / "reports" / "dropout_control" / "lambda_ablation.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nSaved {out}")
    print("\n=== λ 估计汇总（2 个种子） ===")
    piv = df.pivot_table(index="config", columns="seed", values="lambda_est")
    print(piv.to_string())


if __name__ == "__main__":
    main()
