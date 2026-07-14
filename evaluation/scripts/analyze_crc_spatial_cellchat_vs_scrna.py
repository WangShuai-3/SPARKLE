#!/usr/bin/env python3
"""Validate CRC spatial CellChat interactions against the Pelka scRNA reference.

``run_cellchat_spatial_crc.R`` uses CellChat v2 for both inputs: CRC spatial
expression uses segmentation-specific centroid distances, while the dissociated
Pelka single-cell reference is analysed without a distance term.  This script
compares interaction keys ``(source, target, ligand, receptor)`` and reports:

* reference precision/recall for every spatial method;
* removal of RAW reference-negative interactions (putative false positives);
* retention of RAW reference-positive interactions;
* newly introduced reference-negative interactions;
* the same metrics for EpiT-involving (tumour) communication.

The single-cell result is a reference, not an infallible biological truth:
donor composition, dissociation and sampling can hide genuine spatial signals.
Consequently, spatial interactions absent from the reference are called
``reference-negative`` rather than definitively false.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


METHODS = ("RAW", "SPARKLE", "SpatialSoupX", "SoupX", "DecontX")
KEY_COLUMNS = ("source", "target", "ligand", "receptor")
TUMOR_TYPES = frozenset({"EpiT"})
P_VALUE_THRESHOLD = 0.05


def _normalize_table(path: Path) -> pd.DataFrame:
    """Load one CellChat table and enforce a unique interaction-key grain."""
    frame = pd.read_csv(path)
    required = set(KEY_COLUMNS) | {"prob", "pval"}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"{path} is missing columns: {sorted(missing)}")
    for column in KEY_COLUMNS:
        frame[column] = frame[column].astype(str)
    frame["prob"] = pd.to_numeric(frame["prob"], errors="coerce")
    frame["pval"] = pd.to_numeric(frame["pval"], errors="coerce")

    # CellChat normally emits one row per key.  If a database version contains
    # duplicate annotations, keep the strongest probability and lowest p-value
    # so set arithmetic below cannot double count an interaction.
    if frame.duplicated(list(KEY_COLUMNS)).any():
        frame = (
            frame.sort_values(["pval", "prob"], ascending=[True, False])
            .drop_duplicates(list(KEY_COLUMNS), keep="first")
        )
    return frame.reset_index(drop=True)


def _key_set(frame: pd.DataFrame) -> set[tuple[str, str, str, str]]:
    """Convert a CellChat table to a set at the documented interaction grain."""
    return set(frame.loc[:, KEY_COLUMNS].itertuples(index=False, name=None))


def _in_scope(
    keys: set[tuple[str, str, str, str]], scope: str
) -> set[tuple[str, str, str, str]]:
    if scope == "all":
        return keys
    if scope != "tumor":
        raise ValueError(f"Unknown scope: {scope}")
    return {key for key in keys if key[0] in TUMOR_TYPES or key[1] in TUMOR_TYPES}


def _safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else float("nan")


def _evaluate_scope(
    spatial: dict[str, pd.DataFrame],
    reference_all: set[tuple[str, str, str, str]],
    reference_significant: set[tuple[str, str, str, str]],
    scope: str,
) -> tuple[list[dict[str, float | int | str]], dict[str, set]]:
    """Calculate method metrics and RAW-anchored transitions for one scope."""
    reference_all = _in_scope(reference_all, scope)
    reference_significant = _in_scope(reference_significant, scope)
    method_significant = {
        method: _in_scope(
            _key_set(frame.loc[frame["pval"].lt(P_VALUE_THRESHOLD)]), scope
        )
        for method, frame in spatial.items()
    }

    raw_testable = method_significant["RAW"] & reference_all
    raw_tp = raw_testable & reference_significant
    raw_fp = raw_testable - reference_significant

    rows: list[dict[str, float | int | str]] = []
    detail: dict[str, set] = {
        "raw_tp": raw_tp,
        "raw_fp": raw_fp,
        "reference_all": reference_all,
        "reference_significant": reference_significant,
    }
    for method in METHODS:
        significant = method_significant[method]
        testable = significant & reference_all
        not_testable = significant - reference_all
        true_positive = testable & reference_significant
        false_positive = testable - reference_significant
        false_positive_removed = raw_fp - significant
        true_positive_retained = raw_tp & significant
        new_false_positive = false_positive - raw_fp
        new_true_positive = true_positive - raw_tp
        precision = _safe_ratio(len(true_positive), len(testable))
        recall = _safe_ratio(len(true_positive), len(reference_significant))
        f1 = (
            2 * precision * recall / (precision + recall)
            if np.isfinite(precision + recall) and precision + recall > 0
            else float("nan")
        )
        rows.append({
            "segmentation": "",  # filled by the caller
            "scope": scope,
            "method": method,
            "n_spatial_significant": len(significant),
            "n_reference_testable": len(testable),
            "n_not_testable_in_reference": len(not_testable),
            "TP": len(true_positive),
            "FP": len(false_positive),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "RAW_TP": len(raw_tp),
            "RAW_FP": len(raw_fp),
            "raw_fp_removed": len(false_positive_removed),
            "raw_fp_removal_rate": _safe_ratio(
                len(false_positive_removed), len(raw_fp)
            ),
            "raw_tp_retained": len(true_positive_retained),
            "raw_tp_retention_rate": _safe_ratio(
                len(true_positive_retained), len(raw_tp)
            ),
            "new_reference_negative": len(new_false_positive),
            "new_reference_positive": len(new_true_positive),
            "net_reference_negative_change": len(false_positive) - len(raw_fp),
        })
        detail[f"{method}_significant"] = significant
        detail[f"{method}_tp"] = true_positive
        detail[f"{method}_fp"] = false_positive
    return rows, detail


def _keys_to_frame(
    keys: set[tuple[str, str, str, str]], source_frame: pd.DataFrame
) -> pd.DataFrame:
    """Recover probabilities/pathways for a selected interaction-key set."""
    if not keys:
        return pd.DataFrame(columns=[*KEY_COLUMNS, "prob", "pval"])
    selected = pd.DataFrame(sorted(keys), columns=KEY_COLUMNS)
    return selected.merge(source_frame, on=list(KEY_COLUMNS), how="left")


def _probability_lookup(frame: pd.DataFrame) -> dict[tuple[str, ...], float]:
    """Map interaction keys to CellChat probability after duplicate handling."""
    return {
        tuple(row[column] for column in KEY_COLUMNS): float(row["prob"])
        for _, row in frame.iterrows()
    }


def _anchored_probability_concordance(
    spatial: dict[str, pd.DataFrame],
    reference_all_frame: pd.DataFrame,
    scope: str,
    n_bootstrap: int,
    seed: int,
) -> list[dict[str, float | int | str]]:
    """Compare spatial/reference probability ranks on fixed RAW-detected keys.

    Absolute CellChat probabilities are not comparable between dissociated and
    spatial data because the latter includes a distance kernel.  Spearman rank
    correlation is therefore used.  The denominator is fixed to RAW-significant
    interactions testable in the reference; an interaction absent after
    correction receives probability zero.  Paired bootstrap samples the same
    interaction keys for RAW and SPARKLE, stratified by source-target pair.
    """
    reference_probability = _probability_lookup(reference_all_frame)
    reference_keys = set(reference_probability)
    spatial_probability = {
        method: _probability_lookup(frame) for method, frame in spatial.items()
    }
    anchor = set(spatial_probability["RAW"]) & reference_keys
    anchor = _in_scope(anchor, scope)
    ordered = sorted(anchor)
    reference_values = np.asarray(
        [reference_probability[key] for key in ordered], dtype=float
    )
    method_values = {
        method: np.asarray(
            [probability.get(key, 0.0) for key in ordered], dtype=float
        )
        for method, probability in spatial_probability.items()
    }
    correlations = {
        method: float(spearmanr(reference_values, values).statistic)
        if len(ordered) >= 3 and np.unique(values).size > 1
        else float("nan")
        for method, values in method_values.items()
    }

    # Bootstrap only the focal RAW-to-SPARKLE contrast.  Other baselines retain
    # point estimates in the output without multiplying runtime unnecessarily.
    rng = np.random.default_rng(seed)
    groups: dict[tuple[str, str], list[int]] = {}
    for index, key in enumerate(ordered):
        groups.setdefault((key[0], key[1]), []).append(index)
    group_indices = [np.asarray(indices, dtype=int) for indices in groups.values()]
    deltas = np.empty(n_bootstrap, dtype=float)
    for iteration in range(n_bootstrap):
        sampled = np.concatenate([
            rng.choice(indices, size=len(indices), replace=True)
            for indices in group_indices
        ])
        raw_correlation = spearmanr(
            reference_values[sampled], method_values["RAW"][sampled]
        ).statistic
        sparkle_correlation = spearmanr(
            reference_values[sampled], method_values["SPARKLE"][sampled]
        ).statistic
        deltas[iteration] = sparkle_correlation - raw_correlation
    finite_delta = deltas[np.isfinite(deltas)]

    rows = []
    for method in METHODS:
        row: dict[str, float | int | str] = {
            "scope": scope,
            "method": method,
            "n_raw_anchor_interactions": len(ordered),
            "reference_probability_spearman": correlations[method],
            "sparkle_minus_raw_spearman": (
                correlations[method] - correlations["RAW"]
                if method == "SPARKLE" else float("nan")
            ),
            "delta_bootstrap_ci_low": float("nan"),
            "delta_bootstrap_ci_high": float("nan"),
            "bootstrap_probability_delta_positive": float("nan"),
        }
        if method == "SPARKLE" and finite_delta.size:
            row["delta_bootstrap_ci_low"] = float(
                np.quantile(finite_delta, 0.025)
            )
            row["delta_bootstrap_ci_high"] = float(
                np.quantile(finite_delta, 0.975)
            )
            row["bootstrap_probability_delta_positive"] = float(
                np.mean(finite_delta > 0)
            )
        rows.append(row)
    return rows


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    default_root = (
        project_root / "evaluation" / "reports" / "crc_eval_cellchat_spatial"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default=str(default_root))
    parser.add_argument(
        "--segmentations", default="proseg,stardist",
        help="Comma-separated segmentation conditions.",
    )
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_root = Path(args.input_root)
    segmentations = tuple(
        item.strip() for item in args.segmentations.split(",") if item.strip()
    )
    reference_all_frame = _normalize_table(
        input_root / "reference" / "Pelka_scRNA_cellchat_all_tested.csv"
    )
    reference_significant_frame = reference_all_frame.loc[
        reference_all_frame["pval"].lt(P_VALUE_THRESHOLD)
    ].copy()
    reference_all = _key_set(reference_all_frame)
    reference_significant = _key_set(reference_significant_frame)

    combined_rows = []
    concordance_rows = []
    for segmentation_index, segmentation in enumerate(segmentations):
        output_dir = input_root / segmentation
        spatial = {
            method: _normalize_table(
                output_dir / f"{method}_cellchat_spatial.csv"
            )
            for method in METHODS
        }
        for scope in ("all", "tumor"):
            rows, detail = _evaluate_scope(
                spatial, reference_all, reference_significant, scope
            )
            for row in rows:
                row["segmentation"] = segmentation
            combined_rows.extend(rows)
            concordance = _anchored_probability_concordance(
                spatial,
                reference_all_frame,
                scope,
                n_bootstrap=args.n_bootstrap,
                seed=args.seed + 10 * segmentation_index + (scope == "tumor"),
            )
            for row in concordance:
                row["segmentation"] = segmentation
            concordance_rows.extend(concordance)

            # Save auditable SPARKLE transition lists.  RAW provides probability
            # and pathway metadata for removals; SPARKLE supplies newly detected
            # interactions.  These files make headline rates traceable to keys.
            sparkle_significant = detail["SPARKLE_significant"]
            removed_fp = detail["raw_fp"] - sparkle_significant
            lost_tp = detail["raw_tp"] - sparkle_significant
            retained_tp = detail["raw_tp"] & sparkle_significant
            new_fp = detail["SPARKLE_fp"] - detail["raw_fp"]
            _keys_to_frame(removed_fp, spatial["RAW"]).to_csv(
                output_dir / f"SPARKLE_removed_RAW_reference_negative_{scope}.csv",
                index=False,
            )
            _keys_to_frame(lost_tp, spatial["RAW"]).to_csv(
                output_dir / f"SPARKLE_lost_RAW_reference_positive_{scope}.csv",
                index=False,
            )
            _keys_to_frame(retained_tp, spatial["SPARKLE"]).to_csv(
                output_dir / f"SPARKLE_retained_RAW_reference_positive_{scope}.csv",
                index=False,
            )
            _keys_to_frame(new_fp, spatial["SPARKLE"]).to_csv(
                output_dir / f"SPARKLE_new_reference_negative_{scope}.csv",
                index=False,
            )

    comparison = pd.DataFrame(combined_rows)
    comparison.to_csv(
        input_root / "spatial_cellchat_vs_scrna_summary.csv", index=False
    )
    concordance = pd.DataFrame(concordance_rows)
    concordance.to_csv(
        input_root / "spatial_cellchat_probability_concordance.csv", index=False
    )
    metadata = pd.DataFrame([{
        "reference_tested_interactions": len(reference_all),
        "reference_significant_interactions": len(reference_significant),
        "p_value_rule": f"pval < {P_VALUE_THRESHOLD}",
        "interaction_key": "source|target|ligand|receptor",
        "tumor_types": ",".join(sorted(TUMOR_TYPES)),
        "probability_concordance_bootstrap": args.n_bootstrap,
        "bootstrap_seed": args.seed,
    }])
    metadata.to_csv(input_root / "reference_comparison_metadata.csv", index=False)

    display = comparison.loc[
        comparison["method"].isin(["RAW", "SPARKLE"]),
        [
            "segmentation", "scope", "method", "n_spatial_significant",
            "TP", "FP", "precision", "recall", "raw_fp_removal_rate",
            "raw_tp_retention_rate", "new_reference_negative",
        ],
    ]
    print("\nCRC spatial CellChat vs Pelka single-cell reference")
    print(display.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nFixed-RAW-anchor probability rank concordance")
    print(
        concordance.loc[
            concordance["method"].isin(["RAW", "SPARKLE"])
        ].to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )
    print(f"\nOutputs: {input_root}")


if __name__ == "__main__":
    main()
