#!/usr/bin/env python3
"""Analyze how well-known ovarian cancer markers are improved by SPARKLE correction.

Reads h5ad files for a given spatial window, normalizes expression to log1p-CP10K,
and computes tumor-to-non-tumor log2 fold-change metrics for three gene panels
(epithelial/tumor, stromal/fibroblast, immune) across all methods.  The method-
level summary intentionally excludes the unstable immune-panel aggregate.

Outputs (under evaluation/reports/ovarian_eval/marker_analysis/):
    - marker_foldchanges.csv          : per-gene × method log2fc and expression means
    - marker_log2fc_final_core.csv    : final 17 tumor + 9 stromal marker results
    - marker_summary.csv              : per-method aggregate metrics
    - marker_foldchange_delta.csv     : per-gene log2fc delta vs RAW
"""

import warnings
from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# ── Constants ────────────────────────────────────────────────────────────────

TAG = "ovarian_x1000-1800_y300-1100"

INPUT_DIR = Path("evaluation/reports/h5ad_ovarian_annotated")
OUTPUT_DIR = Path("evaluation/reports/ovarian_eval/marker_analysis")

METHODS = ["RAW", "SPARKLE", "SpatialSoupX", "SoupX", "DecontX", "SpotClean"]
PLOT_METHOD_ORDER = ["RAW", "SPARKLE", "SpotClean", "SoupX", "SpatialSoupX", "DecontX"]
METHOD_SUFFIXES = {
    "RAW": "raw",
    "SpotClean": "SpotCleanOfficial",
}

# Pseudocount in log1p-CP10K units, matching the detailed specificity analysis.
LOG2FC_PSEUDOCOUNT = 0.01

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
    """Normalize expression to log1p-CP10K.

    Clips negative values to 0, scales each cell to 10 000 total counts,
    then applies log1p.  Returns a dense numpy array (cells × genes).
    """
    if hasattr(adata.X, "toarray"):
        mat = adata.X.toarray()
    else:
        mat = np.asarray(adata.X[:], dtype=np.float64)

    mat = np.maximum(mat, 0.0)

    # CP10K normalization (sum-to-1e4 per cell); avoid division by zero
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


def compute_log2fc(tumor_mean, non_tumor_mean):
    """Tumor/non-tumor log2FC with a fixed log1p-CP10K pseudocount."""
    return np.log2(
        (tumor_mean + LOG2FC_PSEUDOCOUNT)
        / (non_tumor_mean + LOG2FC_PSEUDOCOUNT)
    )


def save_figure(fig: plt.Figure, path_stem: Path) -> None:
    fig.savefig(path_stem.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(path_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_marker_log2fc(
    log2fc_df: pd.DataFrame,
    output_dir: Path,
    expression_label: str = "log1p-CP10K",
) -> None:
    """Plot signed per-marker log2FC as mean±SEM bars and boxplots."""
    pseudocounts = log2fc_df["log2fc_pseudocount"].drop_duplicates()
    if len(pseudocounts) != 1:
        raise ValueError("Plot input must use one shared log2FC pseudocount")
    pseudocount = float(pseudocounts.iloc[0])
    panel_order = ["epithelial_tumor", "stromal_fibroblast"]
    panel_labels = {
        "epithelial_tumor": "Tumor markers (n=17)",
        "stromal_fibroblast": "Stromal markers (n=9)",
    }
    panel_colors = {
        "epithelial_tumor": "#4477AA",
        "stromal_fibroblast": "#CC6677",
    }
    core = log2fc_df[log2fc_df["panel"].isin(panel_order)].copy()
    core["panel_label"] = core["panel"].map(panel_labels)

    stats = (
        core.groupby(["method", "panel"], observed=True)["log2fc"]
        .agg(mean="mean", std="std", n="size")
        .reset_index()
    )
    stats["sem"] = stats["std"] / np.sqrt(stats["n"])

    x = np.arange(len(PLOT_METHOD_ORDER))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    for index, panel in enumerate(panel_order):
        values = stats[stats["panel"] == panel].set_index("method").reindex(
            PLOT_METHOD_ORDER
        )
        positions = x + (index - 0.5) * width
        bars = ax.bar(
            positions, values["mean"], width,
            yerr=values["sem"], capsize=4,
            color=panel_colors[panel], edgecolor="#333333", linewidth=0.7,
            label=panel_labels[panel],
        )
        for bar, mean in zip(bars, values["mean"]):
            offset = 0.045 if mean >= 0 else -0.065
            ax.text(
                bar.get_x() + bar.get_width() / 2, mean + offset,
                f"{mean:+.2f}", ha="center",
                va="bottom" if mean >= 0 else "top", fontsize=8,
            )
    ax.axhline(0, color="#333333", linewidth=0.9)
    ax.set_xticks(x, PLOT_METHOD_ORDER, rotation=20, ha="right")
    ax.set_xlabel("")
    ax.set_ylabel("Tumor vs non-tumor log2FC")
    ax.set_title(
        f"Ovarian marker log2FC by correction method: mean ± SEM ({expression_label})",
        pad=58,
    )
    ax.legend(
        frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.01)
    )
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.7, alpha=0.7)
    ax.set_axisbelow(True)
    fig.text(
        0.5, -0.015,
        f"Each observation is one marker; pseudocount={pseudocount:g} {expression_label}. "
        "Positive is expected for tumor markers and negative for stromal markers.",
        ha="center", fontsize=9,
    )
    fig.tight_layout()
    save_figure(fig, output_dir / "marker_log2fc_mean_sem_bar")

    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    hue_order = [panel_labels[panel] for panel in panel_order]
    label_palette = {
        panel_labels[panel]: panel_colors[panel] for panel in panel_order
    }
    sns.boxplot(
        data=core, x="method", y="log2fc", hue="panel_label",
        order=PLOT_METHOD_ORDER, hue_order=hue_order, palette=label_palette,
        width=0.72, showfliers=False, linewidth=1.0, ax=ax,
    )
    sns.stripplot(
        data=core, x="method", y="log2fc", hue="panel_label",
        order=PLOT_METHOD_ORDER, hue_order=hue_order, palette=label_palette,
        dodge=True, jitter=0.16, size=3.2, alpha=0.58,
        edgecolor="white", linewidth=0.25, legend=False, ax=ax,
    )
    ax.axhline(0, color="#333333", linewidth=0.9)
    ax.set_xlabel("")
    ax.set_ylabel("Tumor vs non-tumor log2FC")
    ax.set_title(
        f"Ovarian marker log2FC distributions by correction method ({expression_label})",
        pad=58,
    )
    ax.tick_params(axis="x", rotation=20)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    ax.legend(
        frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.01)
    )
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.7, alpha=0.7)
    ax.set_axisbelow(True)
    fig.text(
        0.5, -0.015,
        "Boxes show median and IQR; whiskers use 1.5×IQR; points are individual markers. "
        f"Pseudocount={pseudocount:g} {expression_label}.",
        ha="center", fontsize=9,
    )
    fig.tight_layout()
    save_figure(fig, output_dir / "marker_log2fc_boxplot")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Load and normalize all h5ad files ─────────────────────────────────
    methods_data = {}   # method → (expr, obs, var_names)
    sparkle_corrected_genes = None
    reference_obs_names = None
    reference_var_names = None
    reference_annotations = None

    for method in METHODS:
        suffix = METHOD_SUFFIXES.get(method, method)
        path = INPUT_DIR / f"{TAG}_{suffix}.h5ad"

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

        var_names = adata.var_names.copy()
        obs = adata.obs[["annotation"]].copy()

        if reference_obs_names is None:
            reference_obs_names = adata.obs_names.copy()
            reference_var_names = var_names.copy()
            reference_annotations = obs["annotation"].astype(str).to_numpy()
        else:
            if not adata.obs_names.equals(reference_obs_names):
                raise ValueError(f"{method}: cell order differs from RAW")
            if not var_names.equals(reference_var_names):
                raise ValueError(f"{method}: gene order differs from RAW")
            if not np.array_equal(
                obs["annotation"].astype(str).to_numpy(), reference_annotations
            ):
                raise ValueError(f"{method}: annotations differ from RAW")

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
        adata.file.close()

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

    log2fc_rows = []   # for marker_foldchanges.csv
    summary_rows = []  # for marker_summary.csv
    delta_rows = []    # for marker_foldchange_delta.csv

    # Store RAW log2FC for delta computation
    raw_log2fc = {}

    for method in METHODS:
        if method not in methods_data:
            continue

        expr, obs, var_names = methods_data[method]

        # Map cell-type categories to indices
        groups = obs["annotation"].astype(str).values
        t_mask = np.isin(groups, list(TUMOR_TYPES))
        s_mask = np.isin(groups, list(STROMAL_TYPES))

        # Per-gene signed tumor/non-tumor log2FC for the marker panel.
        epi_log2fcs = []
        stromal_log2fcs = []

        for gene in all_genes:
            try:
                gidx = var_names.get_loc(gene)
            except KeyError:
                continue
            if isinstance(gidx, (np.ndarray, list)):
                continue  # duplicate gene name — skip

            tumor_mean = float(expr[t_mask, gidx].mean())
            stromal_mean = float(expr[s_mask, gidx].mean())
            log2fc = float(compute_log2fc(tumor_mean, stromal_mean))

            log2fc_rows.append({
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
                "log2fc": log2fc,
                "log2fc_pseudocount": LOG2FC_PSEUDOCOUNT,
            })

            if method == "RAW":
                raw_log2fc[gene] = log2fc

            # The method summary deliberately excludes the immune panel.
            if gene in EPITHELIAL_TUMOR_MARKERS:
                epi_log2fcs.append(log2fc)
            elif gene in STROMAL_FIBROBLAST_MARKERS:
                stromal_log2fcs.append(log2fc)

        # Summary metrics
        tumor_marker_mean_log2fc = (
            float(np.mean(epi_log2fcs)) if epi_log2fcs else np.nan
        )
        stromal_marker_mean_log2fc = (
            float(np.mean(stromal_log2fcs)) if stromal_log2fcs else np.nan
        )
        # Tumor markers should be positive and stromal markers negative, so
        # their difference is an expected-direction separation score.
        contrast = tumor_marker_mean_log2fc - stromal_marker_mean_log2fc

        summary_rows.append({
            "method": method,
            "tumor_marker_mean_log2fc": tumor_marker_mean_log2fc,
            "stromal_marker_mean_log2fc": stromal_marker_mean_log2fc,
            "tumor_stromal_contrast": contrast,
            "log2fc_pseudocount": LOG2FC_PSEUDOCOUNT,
        })

        print(f"\n{method}: tumor_marker_mean_log2fc={tumor_marker_mean_log2fc:.3f}, "
              f"stromal_marker_mean_log2fc={stromal_marker_mean_log2fc:.3f}, "
              f"contrast={contrast:.3f}")

    # ── 3b. Delta vs RAW ─────────────────────────────────────────────────────
    log2fc_df = pd.DataFrame(log2fc_rows)
    raw_log2fc_df = log2fc_df[
        log2fc_df["method"] == "RAW"
    ].set_index("gene")["log2fc"]

    for method in METHODS:
        if method == "RAW" or method not in methods_data:
            continue
        m_df = log2fc_df[log2fc_df["method"] == method].set_index("gene")
        common = m_df.index.intersection(raw_log2fc_df.index)
        for gene in common:
            log2fc_raw = raw_log2fc_df.loc[gene]
            log2fc_method = m_df.loc[gene, "log2fc"]
            direction = GENE_DIRECTION[gene]
            log2fc_delta = log2fc_method - log2fc_raw
            # For "up" markers, positive delta is improvement
            # For "down" markers, negative delta is improvement
            improved = (direction == "up" and log2fc_delta > 0) or \
                       (direction == "down" and log2fc_delta < 0)

            delta_rows.append({
                "method": method,
                "gene": gene,
                "panel": (
                    "epithelial_tumor" if gene in EPITHELIAL_TUMOR_MARKERS else
                    "stromal_fibroblast" if gene in STROMAL_FIBROBLAST_MARKERS else
                    "immune"
                ),
                "direction": direction,
                "log2fc_raw": log2fc_raw,
                "log2fc_method": log2fc_method,
                "log2fc_delta": log2fc_delta,
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

        # Compute tumor/non-tumor log2FC for all SPARKLE-corrected genes
        sp_tumor_mean = sc_expr_sp[sp_tmask].mean(axis=0)
        sp_stromal_mean = sc_expr_sp[sp_smask].mean(axis=0)
        sp_log2fc = compute_log2fc(sp_tumor_mean, sp_stromal_mean)

        raw_tumor_mean = sc_expr_raw[raw_tmask].mean(axis=0)
        raw_stromal_mean = sc_expr_raw[raw_smask].mean(axis=0)
        raw_log2fc_all = compute_log2fc(raw_tumor_mean, raw_stromal_mean)

        # Top 40 tumor markers: highest RAW tumor expression
        tumor_rank = np.argsort(raw_tumor_mean)[::-1]  # descending
        top40_tumor_idx = tumor_rank[:40]
        top40_tumor_genes = sc_genes[top40_tumor_idx]

        tumor_specificity = pd.DataFrame({
            "gene": top40_tumor_genes,
            "raw_tumor_mean": raw_tumor_mean[top40_tumor_idx],
            "raw_stromal_mean": raw_stromal_mean[top40_tumor_idx],
            "raw_log2fc": raw_log2fc_all[top40_tumor_idx],
            "sparkle_tumor_mean": sp_tumor_mean[top40_tumor_idx],
            "sparkle_stromal_mean": sp_stromal_mean[top40_tumor_idx],
            "sparkle_log2fc": sp_log2fc[top40_tumor_idx],
        })
        tumor_specificity["log2fc_delta"] = (
            tumor_specificity["sparkle_log2fc"]
            - tumor_specificity["raw_log2fc"]
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
            "raw_log2fc": raw_log2fc_all[bottom40_stromal_idx],
            "sparkle_tumor_mean": sp_tumor_mean[bottom40_stromal_idx],
            "sparkle_stromal_mean": sp_stromal_mean[bottom40_stromal_idx],
            "sparkle_log2fc": sp_log2fc[bottom40_stromal_idx],
        })
        stromal_specificity["log2fc_delta"] = (
            stromal_specificity["sparkle_log2fc"]
            - stromal_specificity["raw_log2fc"]
        )
        stromal_path = OUTPUT_DIR / "sparkle_top40_stromal_specificity.csv"
        stromal_specificity.to_csv(stromal_path, index=False)
        print(f"  Saved top-40 stromal-specific SPARKLE genes → {stromal_path}")

    # ── 5. Save outputs ──────────────────────────────────────────────────────
    log2fc_df = pd.DataFrame(log2fc_rows)
    fc_path = OUTPUT_DIR / "marker_foldchanges.csv"
    log2fc_df.to_csv(fc_path, index=False)
    print(f"\nSaved → {fc_path}")
    final_core = log2fc_df[log2fc_df["panel"].isin([
        "epithelial_tumor", "stromal_fibroblast"
    ])].copy()
    final_core_path = OUTPUT_DIR / "marker_log2fc_final_core.csv"
    final_core.to_csv(final_core_path, index=False)
    print(f"Saved → {final_core_path}")
    plot_marker_log2fc(log2fc_df, OUTPUT_DIR)
    print(
        "Saved → "
        f"{OUTPUT_DIR / 'marker_log2fc_mean_sem_bar.png'} and "
        f"{OUTPUT_DIR / 'marker_log2fc_boxplot.png'}"
    )

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
    print("SUMMARY: Marker log2FC Comparison")
    print(f"{'='*70}")
    header = (
        f"{'Method':<16s}"
        f"{'Tumor log2FC':>15s}"
        f"{'Stromal log2FC':>17s}"
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
            f"{row['tumor_marker_mean_log2fc']:>15.3f}"
            f"{row['stromal_marker_mean_log2fc']:>17.3f}"
            f"{row['tumor_stromal_contrast']:>10.3f}"
            f"{marker}"
        )

    # ── 7. Highlight top improvers and worseners ─────────────────────────────
    print(f"\n{'='*70}")
    print("log2FC IMPROVEMENT ANALYSIS (|delta| > 0.1 vs RAW)")
    print(f"{'='*70}")

    for method in METHODS:
        if method == "RAW" or method not in methods_data:
            continue
        m_delta = delta_df[delta_df["method"] == method]

        # Improved genes
        improved = m_delta[
            (m_delta["improved"]) &
            (m_delta["log2fc_delta"].abs() > 0.1)
        ].sort_values("log2fc_delta", ascending=False, key=abs)

        # Worsened genes
        worsened = m_delta[
            (~m_delta["improved"]) &
            (m_delta["log2fc_delta"].abs() > 0.1)
        ].sort_values("log2fc_delta", ascending=True, key=abs)

        print(f"\n--- {method} ---")
        if len(improved) > 0:
            print(f"  Improved ({len(improved)} genes with |delta| > 0.1):")
            for _, r in improved.iterrows():
                direction = "↑" if r["direction"] == "up" else "↓"
                print(
                    f"    {r['gene']:<20s} {direction}  "
                    f"RAW={r['log2fc_raw']:.3f} → {method}={r['log2fc_method']:.3f}  "
                    f"delta={r['log2fc_delta']:+.3f}"
                )
        else:
            print("  No genes improved with |delta| > 0.1.")

        if len(worsened) > 0:
            print(f"  Worsened ({len(worsened)} genes with |delta| > 0.1):")
            for _, r in worsened.iterrows():
                direction = "↑" if r["direction"] == "up" else "↓"
                print(
                    f"    {r['gene']:<20s} {direction}  "
                    f"RAW={r['log2fc_raw']:.3f} → {method}={r['log2fc_method']:.3f}  "
                    f"delta={r['log2fc_delta']:+.3f}"
                )
        else:
            print("  No genes worsened with |delta| > 0.1.")

    print(f"\n{'='*70}")
    print("DONE. All outputs in:", OUTPUT_DIR.resolve())
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
