#!/usr/bin/env python3
"""Analyze how well-known ovarian cancer markers are improved by SPARKLE correction.

Reads h5ad files for a given spatial window, normalizes expression to log1p-CPM,
and computes tumor-to-stromal fold-change metrics for three gene panels
(epithelial/tumor, stromal/fibroblast, immune) across all methods.

Outputs (under evaluation/reports/ovarian_eval/marker_analysis/):
    - marker_foldchanges.csv          : per-gene × method fc, tumor_mean, stromal_mean
    - marker_summary.csv              : per-method aggregate metrics
    - marker_foldchange_delta.csv     : per-gene fc delta vs RAW
"""

import warnings
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


# ── Constants ────────────────────────────────────────────────────────────────

TAG = "ovarian_x1000-1800_y300-1100"

INPUT_DIR = Path("evaluation/reports/h5ad_ovarian_annotated")
OUTPUT_DIR = Path("evaluation/reports/ovarian_eval/marker_analysis")

METHODS = ["RAW", "SPARKLE", "SpatialSoupX", "SoupX", "DecontX"]

# Cell-type → category mapping  (16 cell types total)
TUMOR_TYPES = {
    "Tumor Cells",
    "Proliferative Tumor Cells",
    "VEGFA+ Tumor Cells",
    "MT-High, Jun+-Fos+ Tumor Cells",
    "Inflammatory Tumor Cells",
    "Malignant Cells Lining Cyst",
}

STROMAL_TYPES = {
    "Tumor Associated Fibroblasts",
    "Stromal Associated Fibroblasts",
    "Endothelial Cells",
    "Macrophages",
    "T & NK Cells",
    "Pericytes",
    "Smooth Muscle Cells",
    "Granulosa Cells",
    "Fallopian Tube Epithelium",
    "Ciliated Epithelial Cells",
}

# Gene panels -----------------------------------------------------------------
EPITHELIAL_TUMOR_MARKERS = [
    "MUC16", "WFDC2", "PAX8", "MSLN", "MUC1", "FOLR1", "EPCAM",
    "KRT7", "KRT19", "WT1", "CD24", "CLDN3", "CLDN4", "CLDN6",
    "LSR", "TACSTD2", "EPHA2",
]

STROMAL_FIBROBLAST_MARKERS = [
    "COL1A1", "COL1A2", "DCN", "ACTA2", "VIM", "FAP",
    "PECAM1", "VWF", "CDH5",
]

IMMUNE_MARKERS = [
    "CD68", "CD163", "CD3E", "CD4", "CD8A", "NKG7",
]

# Helper: gene → expected direction ("up" = higher in tumor is better)
GENE_DIRECTION = {}
for g in EPITHELIAL_TUMOR_MARKERS:
    GENE_DIRECTION[g] = "up"       # higher fc = better
for g in STROMAL_FIBROBLAST_MARKERS:
    GENE_DIRECTION[g] = "down"     # lower fc = better
for g in IMMUNE_MARKERS:
    GENE_DIRECTION[g] = "down"


# ── Utility functions ────────────────────────────────────────────────────────

def normalize_adata(adata: ad.AnnData) -> np.ndarray:
    """Normalize expression to log1p-CPM.

    Clips negative values to 0, scales each cell to 10 000 total counts,
    then applies log1p.  Returns a dense numpy array (cells × genes).
    """
    if hasattr(adata.X, "toarray"):
        mat = adata.X.toarray()
    else:
        mat = np.asarray(adata.X[:], dtype=np.float64)

    mat = np.maximum(mat, 0.0)

    # CPM normalization (sum-to-1e4 per cell); avoid division by zero
    lib_sizes = mat.sum(axis=1)
    lib_sizes = np.where(lib_sizes == 0, 1.0, lib_sizes)
    mat = mat / lib_sizes[:, None] * 1e4

    mat = np.log1p(mat)
    return mat


def filter_and_sort_genes(expr: np.ndarray, var_names: pd.Index,
                          gene_list: list) -> tuple:
    """Return (expr_subset, idx_map) where idx_map maps gene → column index."""
    idx_map = {}
    valid_indices = []
    valid_genes = []
    for g in gene_list:
        loc = var_names.get_loc(g)
        if isinstance(loc, (int, np.integer)):
            idx_map[g] = loc
            valid_indices.append(loc)
            valid_genes.append(g)
    return expr[:, valid_indices], valid_genes


def compute_group_means(expr: np.ndarray, obs: pd.DataFrame,
                        group_col: str = "annotation") -> pd.DataFrame:
    """Return DataFrame of per-group mean expression [genes × cell_groups]."""
    groups = obs[group_col].astype(str).values
    unique_groups = sorted(set(groups))
    n_genes = expr.shape[1]

    means = {}
    for g in unique_groups:
        mask = groups == g
        if mask.sum() == 0:
            means[g] = np.full(n_genes, np.nan)
        else:
            means[g] = expr[mask].mean(axis=0)
    return pd.DataFrame(means, index=pd.RangeIndex(n_genes))


def safe_divide(num, denom):
    """Element-wise division with small epsilon to avoid div-by-zero."""
    return num / (denom + 1e-6)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Load and normalize all h5ad files ─────────────────────────────────
    methods_data = {}   # method → (expr, obs, var_names)
    sparkle_corrected_genes = None

    for method in METHODS:
        path = INPUT_DIR / f"{TAG}_{method}.h5ad"
        if method == "RAW":
            alt = INPUT_DIR / f"{TAG}_raw.h5ad"
            if not path.exists() and alt.exists():
                path = alt

        if not path.exists():
            warnings.warn(f"File not found: {path}")
            continue

        print(f"\n{'='*60}")
        print(f"Loading {method} ...")
        adata = ad.read_h5ad(path, backed="r")

        # Capture SPARKLE-corrected gene names
        if method == "SPARKLE" and "sparkle_corrected" in adata.var.columns:
            sparkle_corrected_genes = set(
                adata.var_names[adata.var["sparkle_corrected"].values]
            )
            print(f"  SPARKLE-corrected genes: {len(sparkle_corrected_genes)}")

        var_names = adata.var_names
        obs = adata.obs[["annotation"]].copy()

        # Print cell-type counts
        print(f"  Total cells: {adata.n_obs}")
        print("  Cell-type counts:")
        counts = obs["annotation"].value_counts()
        for ct, cnt in counts.items():
            cat = "TUMOR" if ct in TUMOR_TYPES else (
                "STROMAL" if ct in STROMAL_TYPES else "UNKNOWN"
            )
            print(f"    {ct:45s} {cnt:6d}  [{cat}]")

        # Normalize
        expr = normalize_adata(adata)
        methods_data[method] = (expr, obs, var_names)

    if len(methods_data) == 0:
        raise FileNotFoundError(f"No h5ad files found for tag {TAG}")

    # ── 2. Build cell-type category masks (shared across methods) ────────────
    # Use RAW obs to determine which cell types appear
    _, raw_obs, _ = methods_data["RAW"]
    raw_groups = raw_obs["annotation"].astype(str).values

    tumor_mask = np.isin(raw_groups, list(TUMOR_TYPES))
    stromal_mask = np.isin(raw_groups, list(STROMAL_TYPES))

    n_tumor = tumor_mask.sum()
    n_stromal = stromal_mask.sum()
    print(f"\n{'='*60}")
    print(f"Tumor cells: {n_tumor}  |  Stromal cells: {n_stromal}")

    if n_tumor == 0 or n_stromal == 0:
        raise ValueError("Insufficient tumor or stromal cells for analysis.")

    # ── 3. Analysis 1 & 3: fold-change per gene per method ───────────────────
    all_genes = (
        EPITHELIAL_TUMOR_MARKERS +
        STROMAL_FIBROBLAST_MARKERS +
        IMMUNE_MARKERS
    )

    fc_rows = []       # for marker_foldchanges.csv
    summary_rows = []  # for marker_summary.csv
    delta_rows = []    # for marker_foldchange_delta.csv

    # Store RAW fc for delta computation
    raw_fc = {}

    for method in METHODS:
        if method not in methods_data:
            continue

        expr, obs, var_names = methods_data[method]

        # Map cell-type categories to indices
        groups = obs["annotation"].astype(str).values
        t_mask = np.isin(groups, list(TUMOR_TYPES))
        s_mask = np.isin(groups, list(STROMAL_TYPES))

        # Per-gene fold change for the marker panel
        epi_fcs = []
        stromal_fcs = []
        immune_fcs = []

        for gene in all_genes:
            try:
                gidx = var_names.get_loc(gene)
            except KeyError:
                continue
            if isinstance(gidx, (np.ndarray, list)):
                continue  # duplicate gene name — skip

            tumor_mean = float(expr[t_mask, gidx].mean())
            stromal_mean = float(expr[s_mask, gidx].mean())
            fc = safe_divide(tumor_mean, stromal_mean)

            fc_rows.append({
                "method": method,
                "gene": gene,
                "panel": (
                    "epithelial_tumor" if gene in EPITHELIAL_TUMOR_MARKERS else
                    "stromal_fibroblast" if gene in STROMAL_FIBROBLAST_MARKERS else
                    "immune"
                ),
                "direction": GENE_DIRECTION[gene],
                "tumor_mean": tumor_mean,
                "stromal_mean": stromal_mean,
                "fold_change": fc,
            })

            if method == "RAW":
                raw_fc[gene] = fc

            # Collect per-panel FCs for summary
            if gene in EPITHELIAL_TUMOR_MARKERS:
                epi_fcs.append(fc)
            elif gene in STROMAL_FIBROBLAST_MARKERS:
                stromal_fcs.append(fc)
            else:
                immune_fcs.append(fc)

        # Summary metrics
        tumor_marker_mean_fc = np.mean(epi_fcs) if epi_fcs else np.nan
        stromal_marker_mean_fc = np.mean(stromal_fcs) if stromal_fcs else np.nan
        immune_marker_mean_fc = np.mean(immune_fcs) if immune_fcs else np.nan
        contrast = (
            tumor_marker_mean_fc / (stromal_marker_mean_fc + 1e-6)
            if stromal_marker_mean_fc and stromal_marker_mean_fc > 0
            else np.nan
        )

        summary_rows.append({
            "method": method,
            "tumor_marker_mean_fc": tumor_marker_mean_fc,
            "stromal_marker_mean_fc": stromal_marker_mean_fc,
            "immune_marker_mean_fc": immune_marker_mean_fc,
            "tumor_stromal_contrast": contrast,
        })

        print(f"\n{method}: tumor_marker_fc={tumor_marker_mean_fc:.3f}, "
              f"stromal_marker_fc={stromal_marker_mean_fc:.3f}, "
              f"immune_marker_fc={immune_marker_mean_fc:.3f}, "
              f"contrast={contrast:.3f}")

    # ── 3b. Delta vs RAW ─────────────────────────────────────────────────────
    fc_df = pd.DataFrame(fc_rows)
    raw_fc_df = fc_df[fc_df["method"] == "RAW"].set_index("gene")["fold_change"]

    for method in METHODS:
        if method == "RAW" or method not in methods_data:
            continue
        m_df = fc_df[fc_df["method"] == method].set_index("gene")
        common = m_df.index.intersection(raw_fc_df.index)
        for gene in common:
            fc_raw = raw_fc_df.loc[gene]
            fc_method = m_df.loc[gene, "fold_change"]
            direction = GENE_DIRECTION[gene]
            abs_delta = fc_method - fc_raw
            pct_delta = (fc_method - fc_raw) / (abs(fc_raw) + 1e-6) * 100
            # For "up" markers, positive delta is improvement
            # For "down" markers, negative delta is improvement
            improved = (direction == "up" and abs_delta > 0) or \
                       (direction == "down" and abs_delta < 0)

            delta_rows.append({
                "method": method,
                "gene": gene,
                "panel": (
                    "epithelial_tumor" if gene in EPITHELIAL_TUMOR_MARKERS else
                    "stromal_fibroblast" if gene in STROMAL_FIBROBLAST_MARKERS else
                    "immune"
                ),
                "direction": direction,
                "fc_raw": fc_raw,
                "fc_method": fc_method,
                "fc_delta": abs_delta,
                "fc_pct_delta": pct_delta,
                "improved": improved,
            })

    # ── 4. Analysis 2: SPARKLE-corrected genes, top/bottom 40 ────────────────
    if sparkle_corrected_genes is not None and "SPARKLE" in methods_data:
        print(f"\n{'='*60}")
        print("Analysis 2: SPARKLE-corrected gene specificity")

        sp_expr, sp_obs, sp_vn = methods_data["SPARKLE"]
        raw_expr, raw_obs, raw_vn = methods_data["RAW"]

        # Filter to SPARKLE-corrected genes only
        sc_mask = np.array([g in sparkle_corrected_genes for g in sp_vn])
        sc_expr_sp = sp_expr[:, sc_mask]
        sc_genes = np.array(sp_vn)[sc_mask]

        sc_mask_raw = np.array([g in sparkle_corrected_genes for g in raw_vn])
        sc_expr_raw = raw_expr[:, sc_mask_raw]

        # Tumor / stromal masks on SPARKLE data
        sp_groups = sp_obs["annotation"].astype(str).values
        sp_tmask = np.isin(sp_groups, list(TUMOR_TYPES))
        sp_smask = np.isin(sp_groups, list(STROMAL_TYPES))

        # RAW masks
        raw_groups_arr = raw_obs["annotation"].astype(str).values
        raw_tmask = np.isin(raw_groups_arr, list(TUMOR_TYPES))
        raw_smask = np.isin(raw_groups_arr, list(STROMAL_TYPES))

        # Compute tumor/stromal FC for all SPARKLE-corrected genes
        sp_tumor_mean = sc_expr_sp[sp_tmask].mean(axis=0)
        sp_stromal_mean = sc_expr_sp[sp_smask].mean(axis=0)
        sp_fc = sp_tumor_mean / (sp_stromal_mean + 1e-6)

        raw_tumor_mean = sc_expr_raw[raw_tmask].mean(axis=0)
        raw_stromal_mean = sc_expr_raw[raw_smask].mean(axis=0)
        raw_fc_all = raw_tumor_mean / (raw_stromal_mean + 1e-6)

        # Top 40 tumor markers: highest RAW tumor expression
        tumor_rank = np.argsort(raw_tumor_mean)[::-1]  # descending
        top40_tumor_idx = tumor_rank[:40]
        top40_tumor_genes = sc_genes[top40_tumor_idx]

        tumor_specificity = pd.DataFrame({
            "gene": top40_tumor_genes,
            "raw_tumor_mean": raw_tumor_mean[top40_tumor_idx],
            "raw_stromal_mean": raw_stromal_mean[top40_tumor_idx],
            "raw_fc": raw_fc_all[top40_tumor_idx],
            "sparkle_tumor_mean": sp_tumor_mean[top40_tumor_idx],
            "sparkle_stromal_mean": sp_stromal_mean[top40_tumor_idx],
            "sparkle_fc": sp_fc[top40_tumor_idx],
        })
        tumor_specificity["fc_delta"] = (
            tumor_specificity["sparkle_fc"] - tumor_specificity["raw_fc"]
        )
        tumor_path = OUTPUT_DIR / "sparkle_top40_tumor_specificity.csv"
        tumor_specificity.to_csv(tumor_path, index=False)
        print(f"  Saved top-40 tumor-specific SPARKLE genes → {tumor_path}")

        # Bottom 40: genes with highest RAW stromal expression
        stromal_rank = np.argsort(raw_stromal_mean)[::-1]
        bottom40_stromal_idx = stromal_rank[:40]
        bottom40_stromal_genes = sc_genes[bottom40_stromal_idx]

        stromal_specificity = pd.DataFrame({
            "gene": bottom40_stromal_genes,
            "raw_tumor_mean": raw_tumor_mean[bottom40_stromal_idx],
            "raw_stromal_mean": raw_stromal_mean[bottom40_stromal_idx],
            "raw_fc": raw_fc_all[bottom40_stromal_idx],
            "sparkle_tumor_mean": sp_tumor_mean[bottom40_stromal_idx],
            "sparkle_stromal_mean": sp_stromal_mean[bottom40_stromal_idx],
            "sparkle_fc": sp_fc[bottom40_stromal_idx],
        })
        stromal_specificity["fc_delta"] = (
            stromal_specificity["sparkle_fc"] - stromal_specificity["raw_fc"]
        )
        stromal_path = OUTPUT_DIR / "sparkle_top40_stromal_specificity.csv"
        stromal_specificity.to_csv(stromal_path, index=False)
        print(f"  Saved top-40 stromal-specific SPARKLE genes → {stromal_path}")

    # ── 5. Save outputs ──────────────────────────────────────────────────────
    fc_df = pd.DataFrame(fc_rows)
    fc_path = OUTPUT_DIR / "marker_foldchanges.csv"
    fc_df.to_csv(fc_path, index=False)
    print(f"\nSaved → {fc_path}")

    summary_df = pd.DataFrame(summary_rows)
    summary_path = OUTPUT_DIR / "marker_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved → {summary_path}")

    delta_df = pd.DataFrame(delta_rows)
    delta_path = OUTPUT_DIR / "marker_foldchange_delta.csv"
    delta_df.to_csv(delta_path, index=False)
    print(f"Saved → {delta_path}")

    # ── 6. Print formatted summary table ─────────────────────────────────────
    print(f"\n{'='*70}")
    print("SUMMARY: Marker Fold-Change Comparison")
    print(f"{'='*70}")
    header = (
        f"{'Method':<16s}"
        f"{'Tumor FC':>10s}"
        f"{'Stromal FC':>11s}"
        f"{'Immune FC':>10s}"
        f"{'Contrast':>10s}"
    )
    print(header)
    print("-" * 70)

    raw_row = summary_df[summary_df["method"] == "RAW"].iloc[0]
    raw_contrast = raw_row["tumor_stromal_contrast"]

    for _, row in summary_df.iterrows():
        method = row["method"]
        marker = " ← RAW" if method == "RAW" else ""
        print(
            f"{method:<16s}"
            f"{row['tumor_marker_mean_fc']:>10.3f}"
            f"{row['stromal_marker_mean_fc']:>11.3f}"
            f"{row['immune_marker_mean_fc']:>10.3f}"
            f"{row['tumor_stromal_contrast']:>10.3f}"
            f"{marker}"
        )

    # ── 7. Highlight top improvers and worseners ─────────────────────────────
    print(f"\n{'='*70}")
    print("FC IMPROVEMENT ANALYSIS (>20% change vs RAW)")
    print(f"{'='*70}")

    for method in METHODS:
        if method == "RAW" or method not in methods_data:
            continue
        m_delta = delta_df[delta_df["method"] == method]

        # Improved genes
        improved = m_delta[
            (m_delta["improved"]) &
            (m_delta["fc_pct_delta"].abs() > 20)
        ].sort_values("fc_pct_delta", ascending=False, key=abs)

        # Worsened genes
        worsened = m_delta[
            (~m_delta["improved"]) &
            (m_delta["fc_pct_delta"].abs() > 20)
        ].sort_values("fc_pct_delta", ascending=True, key=abs)

        print(f"\n--- {method} ---")
        if len(improved) > 0:
            print(f"  Improved ({len(improved)} genes with >20% |Δ|):")
            for _, r in improved.iterrows():
                direction = "↑" if r["direction"] == "up" else "↓"
                print(
                    f"    {r['gene']:<20s} {direction}  "
                    f"RAW={r['fc_raw']:.3f} → {method}={r['fc_method']:.3f}  "
                    f"Δ={r['fc_pct_delta']:+.1f}%"
                )
        else:
            print("  No genes improved >20%.")

        if len(worsened) > 0:
            print(f"  Worsened ({len(worsened)} genes with >20% |Δ|):")
            for _, r in worsened.iterrows():
                direction = "↑" if r["direction"] == "up" else "↓"
                print(
                    f"    {r['gene']:<20s} {direction}  "
                    f"RAW={r['fc_raw']:.3f} → {method}={r['fc_method']:.3f}  "
                    f"Δ={r['fc_pct_delta']:+.1f}%"
                )
        else:
            print("  No genes worsened >20%.")

    print(f"\n{'='*70}")
    print("DONE. All outputs in:", OUTPUT_DIR.resolve())
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
