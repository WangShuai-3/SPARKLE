# SPARKLE: Spatial Ambient RNA Kernel-based Leakage Estimator

A method for removing ambient RNA contamination in high-resolution spatial transcriptomics data (Stereo-seq, Visium HD).

## Overview

SPARKLE leverages **empty spots** — spatial locations outside cell segmentation masks — as built-in ambient RNA probes. Since empty spots contain no cells, any expression detected there is purely ambient RNA from neighboring cells.

### Architecture

Two pipelines available:

| Pipeline | Architecture | Best for |
|----------|-------------|----------|
| **Cell-based** (recommended) | Cells intact + empty space binned separately. Cell-to-cell ambient prediction with zero diagonal → natural self-exclusion. | Data with cell segmentation |
| Bin-level | Uniform grid bins mixing cell and empty spots. Bin-to-bin ambient prediction. | Legacy / no cell masks |

### Key Features

- **No external reference needed** — empty spots serve as built-in ambient probes
- **Cell-based architecture** — cells never split across bins, ambient only from other cells
- **Self-confidence penalty** — `f(s)=1/(1+(s/p90)²)` protects true expression of high-expressors
- **Gene-specific leakage rates (α)** — different genes leak at different rates
- **Global spatial decay (λ)** — physical diffusion distance estimated via grid search

## Quick Start

```python
from stambient import SPARKLE
from stambient.io_utils import load_stereoseq

data = load_stereoseq("sample.gem.gz")

model = SPARKLE(cell_based=True)  # recommended config

# spot_expr: [genes × spots], spot_coords: [spots × 2], spot_labels: [spots] (-1 = empty)
corrected, diagnostics = model.fit_transform(
    data["spot_expr"], data["spot_coords"], data["spot_labels"]
)
```

## Loading Data

SPARKLE provides convenience loaders that return a standard dictionary with
keys ``spot_expr``, ``spot_coords``, ``spot_labels``, ``gene_names``, and
``cell_ids``.

### Stereo-seq GEM

```python
from stambient.io_utils import load_stereoseq

data = load_stereoseq(
    "sample.gem.gz",
    gene_col="geneID",        # or "gene"
    count_col="MIDCounts",    # or "umi_count"
    cell_label_col="cell",     # or "cell_label"
)
```

The GEM loader supports plain text or ``.gz`` files, skips ``#`` comment lines,
and auto-detects tab or comma delimiters.  You can override the default empty
label values with ``empty_labels={0}`` (or any int/list/set).

### Visium HD

```python
from stambient.io_utils import load_visiumhd

data = load_visiumhd("Visium_HD_..._feature_slice.h5")
```

Visium HD pixels are 2 µm, whereas Stereo-seq DNBs are 0.5 µm.  When running
SPARKLE on Visium HD, adjust ``bin_size`` so that the physical empty-bin size
is comparable (e.g., ``bin_size=13`` gives ~26 µm bins, similar to
``bin_size=50`` at 0.5 µm):

```python
model = SPARKLE(cell_based=True, bin_size=13)
corrected, diagnostics = model.fit_transform(
    data["spot_expr"], data["spot_coords"], data["spot_labels"]
)
```

### RYTools GEM + scGEM

```python
from stambient.io_utils import load_RYTools_data

data = load_RYTools_data(
    "sample.gem.gz",          # full GEM (all DNBs, no cell labels)
    "sample.scgem.gz",        # cell-labelled scGEM (no background DNBs)
    gene_col="geneID",
    count_col="MIDCounts",
    cell_label_col="cell",
)
```

The RYTools loader merges an unlabeled full GEM with a matching scGEM file.
DNBs present in the scGEM are assigned their cell labels; all other DNBs are
marked as empty (`-1`) and used as built-in ambient probes.  Column names and
delimiters are configurable.  See `examples/RYTools_example.md` for a complete
workflow including h5ad export and downstream analysis in R/Seurat v4.

## Algorithm

### Cell-Based Pipeline (recommended)

1. **Cells**: extracted intact from spot labels — centroid (x,y), area (spot count), expression/area.
2. **Empty bins**: non-cell spots binned spatially into irregular bins.
3. **λ estimation**: grid search minimizing empty-bin RSS with cell sources.
4. **α estimation**: per-gene weighted OLS on empty-bin observations.
5. **Correction**: for cell c, ambient from **other cells only** (W with zero diagonal), scaled by self-confidence penalty:

$$A_{gc} = \alpha_g \cdot \text{area}_c \cdot f(s_{gc}) \cdot \sum_{c' \neq c} w(d_{cc'}) \cdot \frac{\text{expr}_{gc'}}{\text{area}_{c'}}$$

where $$f(s) = \frac{1}{1 + (s / p_{90})^2}$$ is the self-confidence penalty: high-expressors get near-zero penalty (signal preserved), low-expressors get near-1 penalty (ambient subtracted).

## Final Comparison

### Axolotl SST Diffusion (window x10500-12500 y6000-11100, 200 genes, 4772 cells)

| Method | sstIN | Neighbor | sstIN/Neighbor | Retain | Remove | Runtime |
|--------|:--:|:--:|:--:|:--:|:--:|:--:|
| Raw | 73.0 | 9.7 | 7.53x | — | — | — |
| **SPARKLE** | **71.5** | **5.8** | **12.32x** | **97.8%** | **40.2%** | ~7s |
| DecontX | 62.9 | 7.6 | 8.29x | 86.1% | 21.8% | 1.1s |
| SoupX | 67.3 | 8.0 | 8.45x | 92.1% | 18.0% | ~150s |
| Spatial SoupX | 44.3 | 8.5 | 5.19x | 60.7% | 11.9% | ~30s |

SPARKLE achieves the best signal retention (97.8%) and neighbor removal (40.2%). DecontX offers moderate removal with a fast runtime (1.1s). SoupX retains signal well but removes little noise. Spatial SoupX overcorrects due to global ρ with spatial kernel.

### Synthetic Data (10 scenarios, RMSE reduction)

Reproduced via `evaluation/scripts/final_comparison.py --dataset synthetic --all-scenarios --methods sparkle,spatial_soupx,soupx,decontx`. **All scenarios contain 3–5 cell types with cell-type-specific marker genes**, so the benchmark tests correction in a complex cellular environment rather than homogeneous tissue. 

### MOSTA Cortical Layers (window x10000-14000 y8000-17000, 2219 layer cells)


#### 2000 genes

| Method | DE genes | Pearson | Spearman | Runtime |
|------|:--:|:--:|:--:|:--:|
| RAW | 707 | 0.2740 | 0.1467 | — |
| DecontX | **1012** | nan | 0.1230 | 220.2s |
| **SPARKLE** | **906** | **0.1039** | **0.0424** | 59.8s |
| Spatial SoupX | 672 | 0.1938 | 0.0721 | 51.5s |
| SoupX | 542 | nan | 0.1222 | 4301.7s |

SPARKLE achieves the best overall balance: 906 DE genes (+28% vs raw), lowest Pearson (0.1039, −62%), and lowest Spearman (0.0424, −71%), delivering the strongest cell-type separation. DecontX finds the most DE genes (1012) but its Spearman (0.1230) drops only 16% from raw (0.1467), indicating weak ambient removal — the uniform ~33% contamination estimate over-corrects most genes to near-zero, collapsing cross-type variance. Spatial SoupX cuts Spearman by 51% (0.0721) but with fewer DE genes (672). SoupX produces NaN Pearson at both gene counts due to global ρ with large panels.

### Excluded Methods

**CellBender** — VAE prior estimation fails on spatial spot data (empty spots have 1-2 UMI → zero division / NaN). Falls back to empirical subtraction on full datasets.

**CellClear** — Requires ≥2000 genes for NMF statistical power, but spatial background bins fail the `contamined_genes_detection` step: background expression profiles are too sparse to match any cell cluster (core algorithm designed for scRNA-seq droplet data, not spatial).

## Optimization History

| # | Attempt | Outcome |
|:--:|------|:--:|
| 1 | Remove β (local density) | ✅ RMSE -50~80% |
| 2 | Cell-based architecture | ✅ sstIN 61→74% |
| 3 | Self-confidence penalty | ✅ sstIN 74→97% |
| 4 | Spatial SoupX baseline | ✅ Proves kernel > parameter granularity |
| 5 | EWAP (expression-weight source) | ⚠️ Marginal gain, optional |
| 6 | Per-gene λ | ❌ α instability |

### Why Bin-Level Fails for Clustered Cell Types

In bin-level mode, two bins both containing sstIN cells contribute strongly to each other's ambient prediction — treating real SST expression as contamination. The cell-based architecture solves this: each cell's ambient only comes from **other cells**.

### Why Self-Confidence Penalty

Per-gene α is estimated from empty-bin data where all expression is ambient. But when applied to cells, a high-expressing cell's own signal shouldn't be penalized. The penalty `f(s)=1/(1+(s/p90)²)` is near-zero for truly-expressing cells and near-1 for non-expressing cells.

### Why Spatial Kernel > Global Subtraction

Spatial SoupX (bin-level, global ρ, spatial kernel) achieves 89.9% RMSE↓ on synthetic while original SoupX achieves only 34.2% — the spatial kernel alone provides the majority of the gain. Cell-based architecture + penalty provide the remaining lift.

## API Reference

### `SPARKLE` Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `cell_based` | `True` | Use cell-based pipeline (recommended for segmented data) |
| `bin_size` | 50 | Spots per bin edge (cell-based: empty bin size) |
| `distance_metric` | `'exponential'` | `'exponential'`, `'gaussian'`, or `'inverse'` |
| `lambda_distance` | `None` | Distance decay (μm); auto-estimated if None |
| `max_radius` | 200.0 | Max neighbor search radius (μm) |
| `n_high_genes` | `None` | Number of top genes by empty-bin expression; `None` uses all genes |
| `n_lambda_genes` | 50 | Top genes for λ grid search |
| `r2_threshold` | 0.05 | Minimum weighted R² to apply correction |
| `lambda_grid` | `[10,20,30,50,70,100,150,200,300]` | λ candidates |
| `use_local_density` | `False` | β modulation (disabled — harmful) |
| `use_expr_weight` | `False` | EWAP (experimental) |
| `per_gene_lambda` | `False` | Per-gene λ (rejected — unstable) |

### Methods

- `fit_transform(spot_expr, spot_coords, spot_labels)` → `(corrected_matrix, diagnostics)` (recommended)
- `fit_transform_from_dnb(expr, coords, labels)` → `(corrected_matrix, diagnostics)` (backward-compatible alias)
- `fit_transform_binned(bin_cell_expr, bin_empty_expr, bin_n_cell, bin_n_empty, bin_coords, bin_cell_assignment)` → `(corrected_matrix, diagnostics)`

### Baselines

- `evaluation/baselines/spatial_soupx.py` — Spatial SoupX (kernel-weighted global ρ)
- `evaluation/baselines/soupx.py` — Original SoupX wrapper (global ρ, no spatial info)
- `evaluation/baselines/cellbender.py` — CellBender wrapper (excluded from comparison, see above)
- `evaluation/scripts/standalone_decontx.py` — DecontX standalone (EM-based, integrated via `decontx-python`)

### Unified Entry Point

```bash
# Axolotl
python evaluation/scripts/final_comparison.py \
    --dataset axolotl --x-range 10500 12500 --y-range 6000 11100 \
    --n-genes 200 --methods sparkle,spatial_soupx,soupx,decontx

# MOSTA
python evaluation/scripts/final_comparison.py \
    --dataset mosta --x-range 10000 14000 --y-range 8000 17000 \
    --n-genes 2000 --n-high-genes 2000 \
    --methods sparkle,spatial_soupx,soupx,decontx

# MouseBrain (T304, cell_subclass annotations)
python evaluation/scripts/final_comparison.py \
    --dataset mousebrain --x-range 12500 17500 --y-range 2000 5000 \
    --n-genes 2000 --n-high-genes 2000 \
    --methods sparkle,spatial_soupx,soupx,decontx

# Synthetic
python evaluation/scripts/final_comparison.py \
    --dataset synthetic --all-scenarios \
    --methods sparkle,spatial_soupx,soupx,decontx
```

> **CellBender** and **CellClear** are excluded from `final_comparison.py` due to fundamental incompatibility with spatial spot data. CellBender's VAE requires ≥50 UMI per background barcode (spatial: 1-2). CellClear's NMF-based gene detection fails because background expression profiles are too sparse to match foreground clusters. See "Excluded Methods" above for details.

> **DecontX** is cell-level only (no spatial information, gene-specific α). It runs directly in the base environment via `pip install decontx-python`. On MOSTA, uniform contamination estimates produce NaN correlations for between-type analysis.

## Additional Evaluation Scripts

- **`evaluation/scripts/evaluate_mousebrain_h5ad.py`** — post-hoc evaluation of corrected MouseBrain h5ad files: cell-subclass correlation heatmaps, silhouette score, and comparison against an snRNA-seq reference.
- **`evaluation/scripts/prepare_mousebrain_snrna_reference.R`** — extract `cell_subclass` pseudobulk profiles from the provided Seurat RDS reference for the script above.
- **`evaluation/scripts/benchmark_resource.py`** — benchmark runtime and peak RSS memory of all methods across synthetic data sizes while keeping gene count constant.

Run any script with `--help` for detailed options.

## Installation

```bash
pip install -e .
```

## Dependencies

- numpy, scipy, scikit-learn
- anndata (optional, for h5ad I/O in evaluation)
