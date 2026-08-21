#!/usr/bin/env python3
"""SPARKLE 2.x Phase-3 evidence calibration (E_g score analysis).

Quantifies how well the spatial-CV evidence score separates genes with
genuine local leakage from genes without it, and how the score and the
evidence-gated correction (E1 hard / E2 linear) behave under weak leakage,
low sequencing depth and dense tissue:

  positives: S2 (alpha=0.01) and M3 (weak, alpha=0.002) panel genes —
             all have true local leakage.
  negatives: M2 (diffuse-only) panel genes — zero local leakage; any
             positive evidence/weight is a false positive.

Outputs:
    evaluation/reports/v2/evidence/evidence_calibration.csv
"""

import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATASETS = [
    # (tag, label, description)
    ("synthetic_S2", 1, "strong local leakage (alpha=0.01)"),
    ("mismatch_M3", 1, "weak local leakage (alpha=0.002)"),
    ("mismatch_M2", 0, "no local leakage (diffuse-only)"),
]
SCORE_DATASETS = ["mismatch_M4", "synthetic_S2", "synthetic_S3"]


def _v2_reports_root():
    return PROJECT_ROOT / "evaluation" / "reports" / "v2"


def _evidence_scores(tag, method="SPARKLEv2E2"):
    path = _v2_reports_root() / "h5ad" / f"{tag}_{method}.h5ad"
    if not path.exists():
        return None
    adata = ad.read_h5ad(path)
    if "sparkle_evidence" not in adata.var:
        return None
    ev = np.asarray(adata.var["sparkle_evidence"], dtype=np.float64)
    return ev[~np.isnan(ev)]


def _roc_auc(pos, neg):
    """Rank-based AUC = P(pos > neg) + 0.5 * P(pos == neg)."""
    pos = np.sort(pos)
    ranks = np.searchsorted(pos, neg, side="right")
    ties = np.searchsorted(pos, neg, side="left")
    frac_below = ((ranks + ties) / 2.0) / len(pos)  # P(pos <= neg) w/ ties
    return float(1.0 - frac_below.mean())


def main():
    out_dir = _v2_reports_root() / "evidence"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. ROC: evidence score separates leaked vs non-leaked genes.
    scores = {}
    for tag, label, desc in DATASETS:
        ev = _evidence_scores(tag)
        if ev is None:
            print(f"  [skip] {tag}: no evidence score found")
            continue
        scores[tag] = ev
        print(f"[{tag}] {desc}: n={len(ev)}, E mean={ev.mean():.4f}, "
              f"median={np.median(ev):.4f}, P(E>0)={(ev > 0).mean():.3f}, "
              f"P(E>0.05)={(ev > 0.05).mean():.3f}")

    rows = []
    if "synthetic_S2" in scores and "mismatch_M2" in scores:
        rows.append({
            "analysis": "AUC S2 (leak) vs M2 (no leak)",
            "value": _roc_auc(scores["synthetic_S2"], scores["mismatch_M2"]),
        })
    if "mismatch_M3" in scores and "mismatch_M2" in scores:
        rows.append({
            "analysis": "AUC M3 (weak leak) vs M2 (no leak)",
            "value": _roc_auc(scores["mismatch_M3"], scores["mismatch_M2"]),
        })

    # 2. Evidence under depth / coverage stress (from metrics JSONs).
    import json
    for tag in SCORE_DATASETS:
        mpath = _v2_reports_root() / "metrics" / f"{tag}_metrics_v2.json"
        if not mpath.exists():
            continue
        stored = json.loads(mpath.read_text(encoding="utf-8"))
        for method in ("SPARKLEv2E1", "SPARKLEv2E2"):
            entry = stored.get("methods", {}).get(method, {})
            ev = entry.get("evidence") or {}
            if ev:
                rows.append({
                    "analysis": f"{tag} {method} evidence_median",
                    "value": ev.get("evidence_median"),
                })
                rows.append({
                    "analysis": f"{tag} {method} positive_rate",
                    "value": ev.get("evidence_positive_rate"),
                })
                rows.append({
                    "analysis": f"{tag} {method} n_corrected",
                    "value": ev.get("n_genes_corrected"),
                })

    df = pd.DataFrame(rows)
    out_csv = out_dir / "evidence_calibration.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved {out_csv}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
