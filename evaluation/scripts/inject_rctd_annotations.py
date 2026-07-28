#!/usr/bin/env python3
"""
Inject RCTD first_type annotations into per-method h5ad files.

For each h5ad file matching a given dataset tag, reads the corresponding
RCTD first_type CSV, maps ``cell_name`` to ``first_type`` via ``cell_id``,
and replaces the ``annotation`` column in ``h5ad.obs``.

Usage
-----
python evaluation/scripts/inject_rctd_annotations.py \\
    --tag ovarian_x1000-1800_y300-1100 \\
    --input-dir evaluation/reports/h5ad \\
    --first-type-dir evaluation/reports/rctd_ovarian/first_type \\
    --output-dir evaluation/reports/h5ad_ovarian_annotated
"""

import argparse
import os
import sys
from pathlib import Path

import anndata
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Inject RCTD first_type annotations into h5ad files."
    )
    parser.add_argument(
        "--tag",
        default="ovarian_x1000-1800_y300-1100",
        help="Dataset tag prefix for h5ad files (default: ovarian_x1000-1800_y300-1100).",
    )
    parser.add_argument(
        "--input-dir",
        default="evaluation/reports/h5ad",
        help="Directory containing input h5ad files (default: evaluation/reports/h5ad).",
    )
    parser.add_argument(
        "--first-type-dir",
        default="evaluation/reports/rctd_ovarian/first_type",
        help="Directory containing RCTD first_type CSV files (default: evaluation/reports/rctd_ovarian/first_type).",
    )
    parser.add_argument(
        "--output-dir",
        default="evaluation/reports/h5ad_ovarian_annotated",
        help="Directory to write annotated h5ad files (default: evaluation/reports/h5ad_ovarian_annotated).",
    )
    parser.add_argument(
        "--annotation-method",
        default=None,
        help="If set (e.g. 'RAW'), use this single method's first_type CSV for ALL "
        "h5ad files, giving every method a fixed shared annotation. This isolates "
        "the effect of correction on expression under a common cell grouping "
        "(recommended for the expression-based single-cell comparison). Default: "
        "per-method (each h5ad uses its own RCTD first_type).",
    )
    parser.add_argument(
        "--project-root",
        default=None,
        help="Project root directory.  If given, all relative paths are resolved "
        "against this directory.  Defaults to the directory two levels above this "
        "script (i.e. the project root).",
    )
    args = parser.parse_args()

    # Resolve project root
    if args.project_root:
        project_root = Path(args.project_root).resolve()
    else:
        project_root = Path(__file__).resolve().parents[2]

    input_dir = (project_root / args.input_dir).resolve()
    first_type_dir = (project_root / args.first_type_dir).resolve()
    output_dir = (project_root / args.output_dir).resolve()

    if not input_dir.is_dir():
        sys.exit(f"ERROR: input directory not found: {input_dir}")
    if not first_type_dir.is_dir():
        sys.exit(f"ERROR: first_type directory not found: {first_type_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover h5ad files matching the tag
    prefix = f"{args.tag}_"
    h5ad_files = sorted(
        p for p in input_dir.iterdir()
        if p.name.startswith(prefix) and p.suffix == ".h5ad"
    )
    if not h5ad_files:
        sys.exit(f"ERROR: no .h5ad files found matching tag '{args.tag}' in {input_dir}")

    print(f"Tag:              {args.tag}")
    print(f"Input directory:  {input_dir}")
    print(f"First-type dir:   {first_type_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Found {len(h5ad_files)} h5ad file(s):")
    for p in h5ad_files:
        print(f"  {p.name}")

    # ------------------------------------------------------------------
    # Method name mapping:  stem -> Method used in first_type CSV names
    # ------------------------------------------------------------------
    # The R script uses these method names for first_type CSV filenames:
    #   RAW, SPARKLE, SoupX, DecontX, SpotClean
    METHOD_MAP = {
        "raw": "RAW",
        "sparkle": "SPARKLE",
        "soupx": "SoupX",
        "decontx": "DecontX",
        "spotcleanofficial": "SpotClean",
    }

    # ------------------------------------------------------------------
    # Process each h5ad
    # ------------------------------------------------------------------
    stats = []
    for h5ad_path in h5ad_files:
        stem = h5ad_path.stem[len(prefix):]  # e.g. "raw", "DecontX", ...
        if stem.lower() not in METHOD_MAP:
            print(f"  [{stem}] SKIP: method is outside the final manuscript scope")
            stats.append((stem, "SKIP (out of scope)", 0, 0))
            continue
        method = METHOD_MAP.get(stem.lower(), stem)
        # Optionally use a single shared first_type source for all methods.
        source_method = args.annotation_method if args.annotation_method else method

        csv_path = first_type_dir / f"{source_method}_first_type.csv"
        if not csv_path.is_file():
            print(f"  [{stem}] SKIP: first_type CSV not found: {csv_path}")
            stats.append((stem, "SKIP (no CSV)", 0, 0))
            continue

        print(f"\n[{stem}] Reading {h5ad_path.name} ...")
        adata = anndata.read_h5ad(h5ad_path)

        print(f"  h5ad: {adata.n_obs} cells, annotation unique: {adata.obs['annotation'].unique().tolist()}")

        # Ensure annotation column is object dtype (may be categorical with only "Unknown")
        if hasattr(adata.obs['annotation'].dtype, 'categories'):
            adata.obs['annotation'] = adata.obs['annotation'].astype(object)

        # Read first_type CSV
        rctd = pd.read_csv(csv_path)
        print(f"  CSV:  {rctd.shape[0]} rows, columns: {rctd.columns.tolist()}")

        # Build lookup via cell_id -> enrich multiple obs columns
        rctd = rctd.set_index("cell_id")
        # First_type -> annotation (the primary column the eval script reads)
        mapped = adata.obs["cell_id"].map(rctd["first_type"])
        n_missing = int(mapped.isna().sum())
        n_mapped = int(mapped.notna().sum())
        adata.obs["annotation"] = mapped.fillna(adata.obs["annotation"]).astype(object).values
        # Additional RCTD confidence columns (if present in the CSV)
        bonus_numeric = {"singlet_score": "rctd_singlet_score",
                         "min_score": "rctd_min_score",
                         "score_delta": "rctd_score_delta"}
        for csv_col, obs_col in bonus_numeric.items():
            if csv_col in rctd.columns:
                adata.obs[obs_col] = adata.obs["cell_id"].map(rctd[csv_col]).astype(float)

        print(f"  Mapped: {n_mapped}, missing (kept Unknown): {n_missing}")
        print(f"  Post-injection annotation counts:\n{adata.obs['annotation'].value_counts().to_string()}")

        # Write
        out_path = output_dir / h5ad_path.name
        adata.write_h5ad(out_path)
        print(f"  Written: {out_path}")

        stats.append((stem, "OK", n_mapped, n_missing))

    print("\n" + "=" * 60)
    print("Summary:")
    print(f"{'Method':<20} {'Status':<16} {'Mapped':>8} {'Missing':>8}")
    print("-" * 56)
    for stem, status, n_mapped, n_missing in stats:
        print(f"{stem:<20} {status:<16} {n_mapped:>8} {n_missing:>8}")
    print("Done.")


if __name__ == "__main__":
    main()
