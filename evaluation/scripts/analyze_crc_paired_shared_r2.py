#!/usr/bin/env python3
"""Paired CRC RAW-vs-SPARKLE comparison across Proseg and StarDist.

This analysis deliberately removes three sources of denominator drift that can
make the two segmentation conditions difficult to compare:

1. Proseg and StarDist cells are paired geometrically rather than by ``cell_id``.
   The IDs are local to each segmentation and therefore do not identify the same
   physical cell across conditions.
2. Only genes passing SPARKLE's R² threshold in *both* segmentations are used.
3. RAW and SPARKLE use the exact same paired cells and the consensus RAW-RCTD
   annotation for pseudobulk construction.

The output reports the mean per-cell-type Pearson correlation to the Pelka CRC
single-cell pseudobulk reference.  A paired bootstrap resamples the same cells
for RAW and SPARKLE, providing an uncertainty interval for the correlation
change rather than relying only on one point estimate.

Default inputs correspond to the 800 x 800 um full-gene CRC evaluation window.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import cKDTree


SEGMENTATIONS = ("proseg", "stardist")
METHOD_SUFFIX = {"RAW": "raw", "SPARKLE": "SPARKLE"}


def _h5ad_path(input_dir: Path, tag: str, method: str) -> Path:
    """Return one method path, respecting RAW's lowercase filename suffix."""
    return input_dir / f"{tag}_{METHOD_SUFFIX[method]}.h5ad"


def _load_obs(path: Path) -> pd.DataFrame:
    """Read only cell identity, annotation, and centroid metadata."""
    adata = ad.read_h5ad(path, backed="r")
    required = {"cell_id", "annotation", "x", "y"}
    missing = required.difference(adata.obs.columns)
    if missing:
        adata.file.close()
        raise KeyError(f"{path} is missing obs columns: {sorted(missing)}")
    obs = adata.obs[["cell_id", "annotation", "x", "y"]].copy()
    obs["cell_id"] = obs["cell_id"].astype(str)
    obs["annotation"] = obs["annotation"].astype(str)
    adata.file.close()
    if obs["cell_id"].duplicated().any():
        raise ValueError(f"Duplicate cell_id values in {path}")
    return obs


def _load_r2_genes(path: Path, threshold: float) -> set[str]:
    """Return genes whose stored SPARKLE weighted R² passes ``threshold``."""
    adata = ad.read_h5ad(path, backed="r")
    if "sparkle_r2" not in adata.var.columns:
        adata.file.close()
        raise KeyError(f"{path} has no var['sparkle_r2']")
    r2 = adata.var["sparkle_r2"].to_numpy(dtype=float)
    genes = set(adata.var_names[r2 >= threshold].astype(str))
    adata.file.close()
    return genes


def _mutual_nearest_pairs(
    proseg_obs: pd.DataFrame,
    stardist_obs: pd.DataFrame,
    max_distance_um: float,
) -> pd.DataFrame:
    """Create one-to-one mutual-nearest-neighbor centroid pairs.

    A simple nearest-neighbor join can map multiple cells from one segmentation
    to the same cell in the other.  Requiring the match to be mutual guarantees
    a one-to-one mapping.  The distance cutoff then rejects geometrically weak
    matches near crowded or segmentation-discordant regions.
    """
    pro_coords = proseg_obs[["x", "y"]].to_numpy(dtype=float)
    star_coords = stardist_obs[["x", "y"]].to_numpy(dtype=float)
    pro_to_star_dist, pro_to_star = cKDTree(star_coords).query(pro_coords, k=1)
    _, star_to_pro = cKDTree(pro_coords).query(star_coords, k=1)

    pro_index = np.arange(len(proseg_obs), dtype=int)
    mutual = star_to_pro[pro_to_star] == pro_index
    accepted = mutual & (pro_to_star_dist <= max_distance_um)
    pro_index = pro_index[accepted]
    star_index = pro_to_star[accepted]

    pairs = pd.DataFrame({
        "proseg_row": pro_index,
        "stardist_row": star_index,
        "proseg_cell_id": proseg_obs.iloc[pro_index]["cell_id"].to_numpy(),
        "stardist_cell_id": stardist_obs.iloc[star_index]["cell_id"].to_numpy(),
        "proseg_x": pro_coords[pro_index, 0],
        "proseg_y": pro_coords[pro_index, 1],
        "stardist_x": star_coords[star_index, 0],
        "stardist_y": star_coords[star_index, 1],
        "distance_um": pro_to_star_dist[accepted],
        "proseg_annotation": proseg_obs.iloc[pro_index]["annotation"].to_numpy(),
        "stardist_annotation": stardist_obs.iloc[star_index]["annotation"].to_numpy(),
    })
    pairs["both_known"] = (
        pairs["proseg_annotation"].ne("Unknown")
        & pairs["stardist_annotation"].ne("Unknown")
    )
    pairs["annotation_agrees"] = (
        pairs["both_known"]
        & pairs["proseg_annotation"].eq(pairs["stardist_annotation"])
    )
    pairs["consensus_annotation"] = np.where(
        pairs["annotation_agrees"], pairs["proseg_annotation"], "Unknown"
    )
    return pairs


def _normalized_expression(
    path: Path,
    cell_ids: list[str],
    genes: list[str],
    target_sum: float,
) -> np.ndarray:
    """Return log1p-CP-target expression for selected cells and genes.

    Library-size normalization is computed from *all* 18,085 genes before the
    shared-R² subset is selected.  Subsetting first would change the denominator
    and make this analysis incomparable with the main CRC evaluator.
    """
    adata = ad.read_h5ad(path)
    ids = pd.Index(adata.obs["cell_id"].astype(str))
    if ids.has_duplicates:
        raise ValueError(f"Duplicate cell_id values in {path}")
    row_index = ids.get_indexer(cell_ids)
    if np.any(row_index < 0):
        missing = np.asarray(cell_ids, dtype=object)[row_index < 0][:5]
        raise KeyError(f"Cells missing from {path}: {missing.tolist()}")
    col_index = adata.var_names.astype(str).get_indexer(genes)
    if np.any(col_index < 0):
        missing = np.asarray(genes, dtype=object)[col_index < 0][:5]
        raise KeyError(f"Genes missing from {path}: {missing.tolist()}")

    matrix = adata.X
    if sparse.issparse(matrix):
        matrix = matrix.tocsr().astype(np.float64, copy=True)
        # Corrected matrices can contain tiny floating-point negatives, which
        # have no count interpretation and must not enter library-size totals.
        matrix.data = np.maximum(matrix.data, 0.0)
        totals = np.asarray(matrix.sum(axis=1)).ravel()
        scales = np.divide(
            target_sum, totals,
            out=np.zeros_like(totals, dtype=float), where=totals > 0,
        )
        matrix = matrix.multiply(scales[:, None]).tocsr()
        matrix.data = np.log1p(matrix.data)
        selected = matrix[row_index][:, col_index].toarray()
    else:
        matrix = np.maximum(np.asarray(matrix, dtype=np.float64), 0.0)
        totals = matrix.sum(axis=1)
        scales = np.divide(
            target_sum, totals,
            out=np.zeros_like(totals, dtype=float), where=totals > 0,
        )
        matrix = np.log1p(matrix * scales[:, None])
        selected = matrix[np.ix_(row_index, col_index)]
    return np.asarray(selected, dtype=np.float64)


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Fast scalar Pearson correlation with explicit constant-vector handling."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x - x.mean()
    y = y - y.mean()
    denominator = np.sqrt(np.dot(x, x) * np.dot(y, y))
    return float(np.dot(x, y) / denominator) if denominator > 0 else float("nan")


def _correlations_by_type(
    expression: np.ndarray,
    labels: np.ndarray,
    groups: list[str],
    reference: pd.DataFrame,
) -> dict[str, float]:
    """Correlate each spatial type pseudobulk with its reference profile."""
    correlations = {}
    for group in groups:
        profile = expression[labels == group].mean(axis=0)
        correlations[group] = _pearson(profile, reference[group].to_numpy())
    return correlations


def _paired_bootstrap(
    raw: np.ndarray,
    sparkle: np.ndarray,
    labels: np.ndarray,
    groups: list[str],
    reference: pd.DataFrame,
    n_bootstrap: int,
    seed: int,
) -> np.ndarray:
    """Bootstrap the SPARKLE-minus-RAW mean correlation difference.

    For every cell type and replicate, exactly the same sampled cell indices are
    used for RAW and SPARKLE.  This preserves the pairing and isolates the change
    caused by correction rather than cell-composition noise.
    """
    rng = np.random.default_rng(seed)
    group_indices = {group: np.flatnonzero(labels == group) for group in groups}
    deltas = np.empty(n_bootstrap, dtype=float)
    for iteration in range(n_bootstrap):
        raw_corr = []
        sparkle_corr = []
        for group in groups:
            candidates = group_indices[group]
            sampled = rng.choice(candidates, size=len(candidates), replace=True)
            ref = reference[group].to_numpy()
            raw_corr.append(_pearson(raw[sampled].mean(axis=0), ref))
            sparkle_corr.append(_pearson(sparkle[sampled].mean(axis=0), ref))
        deltas[iteration] = np.nanmean(sparkle_corr) - np.nanmean(raw_corr)
    return deltas


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    reports = project_root / "evaluation" / "reports"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--proseg-dir",
        default=str(reports / "h5ad_crc_proseg_annotated_x14200-15000_y2750-3550"),
    )
    parser.add_argument(
        "--stardist-dir",
        default=str(reports / "h5ad_crc_stardist_annotated_x14200-15000_y2750-3550"),
    )
    parser.add_argument("--proseg-tag", default="crc_proseg_x14200-15000_y2750-3550")
    parser.add_argument("--stardist-tag", default="crc_stardist_x14200-15000_y2750-3550")
    parser.add_argument(
        "--reference",
        default=str(
            project_root / "evaluation" / "data" / "CRC" / "scrna_reference"
            / "prepared_cluster_midway" / "crc_cluster_midway_pseudobulk.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(reports / "crc_eval_paired_shared_r2"),
    )
    parser.add_argument("--r2-threshold", type=float, default=0.01)
    parser.add_argument(
        "--max-match-distance-um", type=float, default=5.0,
        help="Maximum mutual-nearest-neighbor centroid distance (default: 5 um).",
    )
    parser.add_argument("--min-cells-per-type", type=int, default=3)
    parser.add_argument("--target-sum", type=float, default=1e4)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_dirs = {
        "proseg": Path(args.proseg_dir),
        "stardist": Path(args.stardist_dir),
    }
    tags = {"proseg": args.proseg_tag, "stardist": args.stardist_tag}
    paths = {
        seg: {
            method: _h5ad_path(input_dirs[seg], tags[seg], method)
            for method in METHOD_SUFFIX
        }
        for seg in SEGMENTATIONS
    }
    for method_paths in paths.values():
        for path in method_paths.values():
            if not path.exists():
                raise FileNotFoundError(path)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reference = pd.read_csv(args.reference, index_col=0)
    reference.index = reference.index.astype(str)

    # The paired cohort is defined once from RAW centroids and RAW-RCTD labels,
    # then reused unchanged for both RAW and SPARKLE expression matrices.
    obs = {seg: _load_obs(paths[seg]["RAW"]) for seg in SEGMENTATIONS}
    pairs = _mutual_nearest_pairs(
        obs["proseg"], obs["stardist"], args.max_match_distance_um
    )
    pairs.to_csv(output_dir / "matched_pairs_all.csv", index=False)
    consensus = pairs[pairs["annotation_agrees"]].copy().reset_index(drop=True)
    if consensus.empty:
        raise ValueError("No matched cells have concordant non-Unknown annotations")

    type_counts = consensus["consensus_annotation"].value_counts()
    groups = sorted(
        group for group, count in type_counts.items()
        if count >= args.min_cells_per_type and group in reference.columns
    )
    consensus = consensus[consensus["consensus_annotation"].isin(groups)].copy()
    consensus.to_csv(output_dir / "matched_pairs_consensus.csv", index=False)
    labels = consensus["consensus_annotation"].to_numpy(dtype=str)

    # Use the intersection of actual R² values, rather than assuming the boolean
    # flags are identical between independently fitted segmentation conditions.
    r2_genes = {
        seg: _load_r2_genes(paths[seg]["SPARKLE"], args.r2_threshold)
        for seg in SEGMENTATIONS
    }
    shared_r2 = r2_genes["proseg"].intersection(r2_genes["stardist"])
    genes = sorted(shared_r2.intersection(reference.index))
    if len(genes) < 10:
        raise ValueError(f"Only {len(genes)} shared R²/reference genes")
    reference = reference.loc[genes, groups]

    cell_ids = {
        "proseg": consensus["proseg_cell_id"].astype(str).tolist(),
        "stardist": consensus["stardist_cell_id"].astype(str).tolist(),
    }
    summary_rows = []
    per_type_rows = []
    bootstrap_arrays = {}

    for seg in SEGMENTATIONS:
        matrices = {
            method: _normalized_expression(
                paths[seg][method], cell_ids[seg], genes, args.target_sum
            )
            for method in METHOD_SUFFIX
        }
        correlations = {
            method: _correlations_by_type(
                matrices[method], labels, groups, reference
            )
            for method in METHOD_SUFFIX
        }
        raw_mean = float(np.nanmean(list(correlations["RAW"].values())))
        sparkle_mean = float(np.nanmean(list(correlations["SPARKLE"].values())))
        delta = sparkle_mean - raw_mean

        bootstrap = _paired_bootstrap(
            matrices["RAW"], matrices["SPARKLE"], labels, groups, reference,
            n_bootstrap=args.n_bootstrap, seed=args.seed,
        )
        bootstrap_arrays[seg] = bootstrap
        ci_low, ci_high = np.quantile(bootstrap, [0.025, 0.975])
        summary_rows.append({
            "segmentation": seg,
            "n_matched_cells": len(consensus),
            "n_cell_types": len(groups),
            "n_shared_r2_genes": len(genes),
            "raw_mean_snrna_corr": raw_mean,
            "sparkle_mean_snrna_corr": sparkle_mean,
            "sparkle_minus_raw": delta,
            "bootstrap_ci_low": float(ci_low),
            "bootstrap_ci_high": float(ci_high),
            "bootstrap_probability_positive": float(np.mean(bootstrap > 0)),
        })
        for group in groups:
            per_type_rows.append({
                "segmentation": seg,
                "cell_type": group,
                "n_cells": int(np.sum(labels == group)),
                "raw_snrna_corr": correlations["RAW"][group],
                "sparkle_snrna_corr": correlations["SPARKLE"][group],
                "sparkle_minus_raw": (
                    correlations["SPARKLE"][group] - correlations["RAW"][group]
                ),
            })

    summary = pd.DataFrame(summary_rows)
    per_type = pd.DataFrame(per_type_rows)
    summary.to_csv(output_dir / "paired_shared_r2_summary.csv", index=False)
    per_type.to_csv(output_dir / "paired_shared_r2_per_celltype.csv", index=False)
    np.savez_compressed(
        output_dir / "paired_bootstrap_deltas.npz", **bootstrap_arrays
    )

    metadata = {
        "r2_threshold": args.r2_threshold,
        "max_match_distance_um": args.max_match_distance_um,
        "n_mutual_pairs_within_distance": int(len(pairs)),
        "n_consensus_pairs_before_min_type_filter": int(
            pairs["annotation_agrees"].sum()
        ),
        "n_analyzed_pairs": int(len(consensus)),
        "cell_types": groups,
        "cells_per_type": {
            group: int(np.sum(labels == group)) for group in groups
        },
        "n_r2_genes_proseg": len(r2_genes["proseg"]),
        "n_r2_genes_stardist": len(r2_genes["stardist"]),
        "n_shared_r2_genes_before_reference": len(shared_r2),
        "n_shared_r2_reference_genes": len(genes),
        "normalization": f"log1p-CP{int(args.target_sum)} using all genes for totals",
        "n_bootstrap": args.n_bootstrap,
        "seed": args.seed,
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)

    print("\nStrict paired CRC comparison")
    print(f"  Mutual pairs <= {args.max_match_distance_um:g} um: {len(pairs):,}")
    print(f"  Consensus annotated pairs analyzed: {len(consensus):,}")
    print(f"  Cell types: {len(groups)}")
    print(f"  Shared R2/reference genes: {len(genes):,}")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(f"\nOutputs: {output_dir}")


if __name__ == "__main__":
    main()
