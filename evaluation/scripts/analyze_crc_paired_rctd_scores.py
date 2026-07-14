#!/usr/bin/env python3
"""Compare CRC RCTD scores on the same Proseg/StarDist physical cells.

The input cohort comes from ``analyze_crc_paired_shared_r2.py``: Proseg and
StarDist cells are mutual-nearest-neighbor centroid matches with concordant,
non-Unknown RAW-RCTD annotations.  This script further requires every physical
cell pair to have RCTD results for all four combinations:

    Proseg/RAW, Proseg/SPARKLE, StarDist/RAW, StarDist/SPARKLE.

RCTD doublet mode reports ``min_score`` (best doublet fit) and
``singlet_score`` (best single-type fit).  spacexr classifies a cell as singlet
when ``singlet_score - min_score < 25``.  We therefore use this score margin as
the primary continuous metric: a smaller margin is more singlet-like.  Absolute
scores depend strongly on UMI depth and segmentation, so cross-segmentation
score differences are descriptive and are not treated as an accuracy ranking.

RAW-to-SPARKLE changes are paired cell by cell.  Stratified bootstrap resampling
preserves the consensus cell-type composition, and an exact McNemar/binomial
test evaluates discordant singlet transitions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon


SEGMENTATIONS = ("proseg", "stardist")
METHODS = ("RAW", "SPARKLE")


def _load_rctd(path: Path) -> dict[str, pd.DataFrame]:
    """Load RAW/SPARKLE RCTD rows and index them by spatial cell barcode."""
    results = pd.read_csv(path)
    required = {
        "cell_barcode", "spot_class", "first_type", "second_type",
        "min_score", "singlet_score", "method",
    }
    missing = required.difference(results.columns)
    if missing:
        raise KeyError(f"{path} is missing columns: {sorted(missing)}")
    results["cell_barcode"] = results["cell_barcode"].astype(str)
    results["score_margin"] = results["singlet_score"] - results["min_score"]

    by_method = {}
    for method in METHODS:
        subset = results[results["method"].eq(method)].copy()
        if subset.empty:
            raise ValueError(f"No {method} rows in {path}")
        if subset["cell_barcode"].duplicated().any():
            raise ValueError(f"Duplicate {method} cell_barcode values in {path}")
        by_method[method] = subset.set_index("cell_barcode", drop=False)
    return by_method


def _stratified_bootstrap(
    margin_difference: np.ndarray,
    singlet_difference: np.ndarray,
    labels: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Bootstrap paired mean-margin and singlet-percentage-point changes.

    Sampling occurs independently within every consensus cell type, preserving
    the observed type counts.  Because the inputs are already paired
    SPARKLE-minus-RAW differences, the same sampled cells contribute to both
    methods by construction.
    """
    rng = np.random.default_rng(seed)
    group_indices = {
        group: np.flatnonzero(labels == group) for group in np.unique(labels)
    }
    margin_bootstrap = np.empty(n_bootstrap, dtype=float)
    singlet_bootstrap = np.empty(n_bootstrap, dtype=float)
    for iteration in range(n_bootstrap):
        sampled = np.concatenate([
            rng.choice(indices, size=len(indices), replace=True)
            for indices in group_indices.values()
        ])
        margin_bootstrap[iteration] = margin_difference[sampled].mean()
        singlet_bootstrap[iteration] = 100.0 * singlet_difference[sampled].mean()
    return margin_bootstrap, singlet_bootstrap


def _method_metrics(frame: pd.DataFrame) -> dict[str, float]:
    """Return descriptive metrics for one method on a fixed cell cohort."""
    singlet = frame["spot_class"].eq("singlet")
    return {
        "mean_min_score": float(frame["min_score"].mean()),
        "mean_singlet_score": float(frame["singlet_score"].mean()),
        "mean_score_margin": float(frame["score_margin"].mean()),
        "median_score_margin": float(frame["score_margin"].median()),
        "pct_singlet": float(100.0 * singlet.mean()),
    }


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    reports = project_root / "evaluation" / "reports"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matched-pairs",
        default=str(
            reports / "crc_eval_paired_shared_r2" / "matched_pairs_consensus.csv"
        ),
        help="Consensus physical-cell pairs from analyze_crc_paired_shared_r2.py.",
    )
    parser.add_argument(
        "--proseg-rctd",
        default=str(reports / "rctd_crc" / "proseg" / "rctd_doublet_results.csv"),
    )
    parser.add_argument(
        "--stardist-rctd",
        default=str(reports / "rctd_crc" / "stardist" / "rctd_doublet_results.csv"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(reports / "crc_eval_paired_rctd"),
    )
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    pairs = pd.read_csv(args.matched_pairs)
    required_pair_columns = {
        "proseg_cell_id", "stardist_cell_id", "consensus_annotation",
        "distance_um",
    }
    missing = required_pair_columns.difference(pairs.columns)
    if missing:
        raise KeyError(f"Matched-pair file is missing columns: {sorted(missing)}")
    if pairs["proseg_cell_id"].duplicated().any() or pairs["stardist_cell_id"].duplicated().any():
        raise ValueError("Matched-pair input is not one-to-one")

    rctd_paths = {
        "proseg": Path(args.proseg_rctd),
        "stardist": Path(args.stardist_rctd),
    }
    rctd = {seg: _load_rctd(path) for seg, path in rctd_paths.items()}

    # RCTD barcodes use the h5ad obs_name form ``Cell_<cell_id>``.  Requiring
    # availability in all four result tables yields one denominator for every
    # within- and cross-segmentation comparison below.
    common = np.ones(len(pairs), dtype=bool)
    barcode_series = {}
    for seg in SEGMENTATIONS:
        barcodes = "Cell_" + pairs[f"{seg}_cell_id"].astype(str)
        barcode_series[seg] = barcodes
        for method in METHODS:
            common &= barcodes.isin(rctd[seg][method].index).to_numpy()
    cohort = pairs.loc[common].copy().reset_index(drop=True)
    if cohort.empty:
        raise ValueError("No physical cell pairs have all four RCTD results")
    labels = cohort["consensus_annotation"].astype(str).to_numpy()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cohort.to_csv(output_dir / "paired_rctd_common_cells.csv", index=False)

    aligned: dict[str, dict[str, pd.DataFrame]] = {}
    per_cell = cohort[[
        "proseg_cell_id", "stardist_cell_id", "distance_um",
        "consensus_annotation",
    ]].copy()
    for seg in SEGMENTATIONS:
        barcodes = "Cell_" + cohort[f"{seg}_cell_id"].astype(str)
        aligned[seg] = {}
        for method in METHODS:
            frame = rctd[seg][method].loc[barcodes].reset_index(drop=True)
            aligned[seg][method] = frame
            prefix = f"{seg}_{method.lower()}"
            for column in (
                "spot_class", "first_type", "second_type", "min_score",
                "singlet_score", "score_margin",
            ):
                per_cell[f"{prefix}_{column}"] = frame[column].to_numpy()
    per_cell.to_csv(output_dir / "paired_rctd_scores_per_cell.csv", index=False)

    summary_rows = []
    per_type_rows = []
    bootstrap_outputs = {}
    for seg in SEGMENTATIONS:
        raw = aligned[seg]["RAW"]
        sparkle = aligned[seg]["SPARKLE"]
        raw_metrics = _method_metrics(raw)
        sparkle_metrics = _method_metrics(sparkle)

        raw_singlet = raw["spot_class"].eq("singlet").to_numpy()
        sparkle_singlet = sparkle["spot_class"].eq("singlet").to_numpy()
        margin_difference = (
            sparkle["score_margin"].to_numpy() - raw["score_margin"].to_numpy()
        )
        singlet_difference = sparkle_singlet.astype(float) - raw_singlet.astype(float)
        margin_bootstrap, singlet_bootstrap = _stratified_bootstrap(
            margin_difference, singlet_difference, labels,
            n_bootstrap=args.n_bootstrap, seed=args.seed,
        )
        bootstrap_outputs[f"{seg}_score_margin_delta"] = margin_bootstrap
        bootstrap_outputs[f"{seg}_singlet_pp_delta"] = singlet_bootstrap

        gained_singlet = int(np.sum(~raw_singlet & sparkle_singlet))
        lost_singlet = int(np.sum(raw_singlet & ~sparkle_singlet))
        discordant = gained_singlet + lost_singlet
        mcnemar_p = (
            float(binomtest(gained_singlet, discordant, 0.5).pvalue)
            if discordant else 1.0
        )
        try:
            wilcoxon_p = float(wilcoxon(margin_difference).pvalue)
        except ValueError:
            wilcoxon_p = float("nan")

        summary_rows.append({
            "segmentation": seg,
            "n_common_cells": len(cohort),
            **{f"raw_{key}": value for key, value in raw_metrics.items()},
            **{f"sparkle_{key}": value for key, value in sparkle_metrics.items()},
            "sparkle_minus_raw_mean_score_margin": float(margin_difference.mean()),
            "score_margin_delta_ci_low": float(np.quantile(margin_bootstrap, 0.025)),
            "score_margin_delta_ci_high": float(np.quantile(margin_bootstrap, 0.975)),
            "bootstrap_probability_margin_lower": float(
                np.mean(margin_bootstrap < 0)
            ),
            "wilcoxon_score_margin_p": wilcoxon_p,
            "sparkle_minus_raw_singlet_pp": float(100.0 * singlet_difference.mean()),
            "singlet_pp_delta_ci_low": float(np.quantile(singlet_bootstrap, 0.025)),
            "singlet_pp_delta_ci_high": float(np.quantile(singlet_bootstrap, 0.975)),
            "gained_singlet_cells": gained_singlet,
            "lost_singlet_cells": lost_singlet,
            "mcnemar_exact_p": mcnemar_p,
            "first_type_agreement_raw_to_sparkle": float(
                np.mean(
                    raw["first_type"].astype(str).to_numpy()
                    == sparkle["first_type"].astype(str).to_numpy()
                )
            ),
        })

        for cell_type in sorted(np.unique(labels)):
            mask = labels == cell_type
            raw_type = raw.loc[mask]
            sparkle_type = sparkle.loc[mask]
            raw_type_singlet = raw_type["spot_class"].eq("singlet").to_numpy()
            sparkle_type_singlet = sparkle_type["spot_class"].eq("singlet").to_numpy()
            per_type_rows.append({
                "segmentation": seg,
                "cell_type": cell_type,
                "n_cells": int(mask.sum()),
                "raw_mean_score_margin": float(raw_type["score_margin"].mean()),
                "sparkle_mean_score_margin": float(
                    sparkle_type["score_margin"].mean()
                ),
                "sparkle_minus_raw_score_margin": float(
                    (sparkle_type["score_margin"].to_numpy()
                     - raw_type["score_margin"].to_numpy()).mean()
                ),
                "raw_pct_singlet": float(100.0 * raw_type_singlet.mean()),
                "sparkle_pct_singlet": float(100.0 * sparkle_type_singlet.mean()),
                "sparkle_minus_raw_singlet_pp": float(
                    100.0 * (sparkle_type_singlet.mean() - raw_type_singlet.mean())
                ),
                "first_type_agreement_raw_to_sparkle": float(
                    np.mean(
                        raw_type["first_type"].astype(str).to_numpy()
                        == sparkle_type["first_type"].astype(str).to_numpy()
                    )
                ),
            })

    summary = pd.DataFrame(summary_rows)
    per_type = pd.DataFrame(per_type_rows)
    summary.to_csv(output_dir / "paired_rctd_score_summary.csv", index=False)
    per_type.to_csv(output_dir / "paired_rctd_score_per_celltype.csv", index=False)
    np.savez_compressed(
        output_dir / "paired_rctd_bootstrap_deltas.npz", **bootstrap_outputs
    )

    # Direct cross-segmentation comparisons use the same physical cell pairs,
    # but remain descriptive because the score scale changes with UMI depth and
    # the number of spots assigned to each segmented cell.
    cross_rows = []
    for method in METHODS:
        proseg = aligned["proseg"][method]
        stardist = aligned["stardist"][method]
        proseg_singlet = proseg["spot_class"].eq("singlet").to_numpy()
        stardist_singlet = stardist["spot_class"].eq("singlet").to_numpy()
        cross_rows.append({
            "method": method,
            "n_common_cells": len(cohort),
            "proseg_mean_score_margin": float(proseg["score_margin"].mean()),
            "stardist_mean_score_margin": float(stardist["score_margin"].mean()),
            "stardist_minus_proseg_score_margin": float(
                (stardist["score_margin"].to_numpy()
                 - proseg["score_margin"].to_numpy()).mean()
            ),
            "proseg_pct_singlet": float(100.0 * proseg_singlet.mean()),
            "stardist_pct_singlet": float(100.0 * stardist_singlet.mean()),
            "stardist_minus_proseg_singlet_pp": float(
                100.0 * (stardist_singlet.mean() - proseg_singlet.mean())
            ),
            "first_type_agreement_between_segmentations": float(
                np.mean(
                    proseg["first_type"].astype(str).to_numpy()
                    == stardist["first_type"].astype(str).to_numpy()
                )
            ),
        })
    cross = pd.DataFrame(cross_rows)
    cross.to_csv(output_dir / "paired_rctd_cross_segmentation.csv", index=False)

    metadata = {
        "input_consensus_pairs": int(len(pairs)),
        "all_four_rctd_common_pairs": int(len(cohort)),
        "coverage_fraction": float(len(cohort) / len(pairs)),
        "cell_types": sorted(np.unique(labels).tolist()),
        "cells_per_type": {
            cell_type: int(np.sum(labels == cell_type))
            for cell_type in sorted(np.unique(labels))
        },
        "score_margin_definition": "singlet_score - min_score",
        "score_margin_direction": "lower is more singlet-like; spacexr threshold is 25",
        "n_bootstrap": args.n_bootstrap,
        "bootstrap": "paired and stratified by consensus cell type",
        "seed": args.seed,
        "cross_segmentation_caveat": (
            "Absolute RCTD scores depend on UMI depth and segmentation; "
            "cross-segmentation score differences are descriptive."
        ),
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)

    display_columns = [
        "segmentation", "n_common_cells", "raw_mean_score_margin",
        "sparkle_mean_score_margin", "sparkle_minus_raw_mean_score_margin",
        "score_margin_delta_ci_low", "score_margin_delta_ci_high",
        "raw_pct_singlet", "sparkle_pct_singlet",
        "sparkle_minus_raw_singlet_pp", "singlet_pp_delta_ci_low",
        "singlet_pp_delta_ci_high", "first_type_agreement_raw_to_sparkle",
    ]
    print("\nStrict paired CRC RCTD comparison")
    print(f"  Consensus input pairs: {len(pairs):,}")
    print(f"  Pairs with all four RCTD results: {len(cohort):,}")
    print(summary[display_columns].to_string(
        index=False, float_format=lambda value: f"{value:.6f}"
    ))
    print("\nCross-segmentation descriptive comparison")
    print(cross.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(f"\nOutputs: {output_dir}")


if __name__ == "__main__":
    main()
