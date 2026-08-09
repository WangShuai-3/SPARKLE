#!/usr/bin/env python3
"""Run official SpotClean on one MouseBrain window (subprocess worker).

Steps:
  1. load the window with load_mousebrain_data(x_range, y_range)
  2. write the per-cell RAW h5ad (template) that run_real requires
  3. remove 'tile_grid' from the mousebrain config so run_real takes the
     axolotl-style non-tiled path for this window
  4. call rso.run_real('mousebrain', args) -> official R package

/usr/bin/time -v on THIS process captures the whole-tree peak RSS (Python + R).

Usage:
    python benchmark_spotclean_worker.py <x_min> <x_max> <y_min> <y_max> <out_prefix>
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import run_spotclean_official as rso
from evaluation.scripts.final_comparison import (
    load_mousebrain_data,
    compute_cell_expr,
    save_result_h5ad,
)

N_GENES = 10000


def _write_window_raw_h5ad(x_range, y_range, tag):
    data = load_mousebrain_data(x_range=x_range, y_range=y_range)
    n_cells = len(data["cell_ids"])
    raw_cell = compute_cell_expr(data["dnb_expr"], data["dnb_labels"], n_cells)
    # gene_names: load_mousebrain_data returns full gene list; subset like
    # run_real expects the RAW h5ad to be the "final-comparison" template.
    out_path = rso.REPORTS / "h5ad" / f"{tag}_raw.h5ad"
    save_result_h5ad(raw_cell, data["gene_names"], data["cell_ids"],
                     data.get("ann_map", {}), out_path, "RAW", save_h5ad=True)
    print(f"[worker] wrote RAW template: {out_path} ({raw_cell.shape})", flush=True)
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("x_min", type=float)
    parser.add_argument("x_max", type=float)
    parser.add_argument("y_min", type=float)
    parser.add_argument("y_max", type=float)
    parser.add_argument("out_prefix", type=str)
    args = parser.parse_args()

    x_range = (args.x_min, args.x_max)
    y_range = (args.y_min, args.y_max)

    config = rso.REAL_CONFIG["mousebrain"]
    config["x_range"] = x_range
    config["y_range"] = y_range
    tag = rso._tag("mousebrain", config)

    _write_window_raw_h5ad(x_range, y_range, tag)

    # axolotl-style: no tiling for a single small window
    config.pop("tile_grid", None)
    config.pop("tile_halo", None)

    ns = argparse.Namespace(
        datasets=["mousebrain"], scenarios=[],
        maxit=30, tol=1.0, rscript=rso.DEFAULT_RSCRIPT,
        memory_fraction=0.75, force_memory=False,
        reuse_prepared_input=False, prepare_only=False, overwrite=True,
    )
    print(f"[worker] SpotClean window x={x_range} y={y_range} tag={tag}", flush=True)
    result = rso.run_real("mousebrain", ns)
    runtime = result.get("end_to_end_runtime_seconds",
                         result.get("runtime", result.get("total_seconds", float("nan"))))
    out = Path(args.out_prefix)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(f"method=SpotClean\nruntime_sec={runtime}\n")
    print(f"[worker] done, runtime={runtime}s", flush=True)


if __name__ == "__main__":
    main()
