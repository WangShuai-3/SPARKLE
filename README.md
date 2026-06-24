# SPARKLE: Spatial Ambient RNA Kernel-based Leakage Estimator

A method for removing ambient RNA contamination in high-resolution spatial transcriptomics data (Stereo-seq, Visium HD).

## Overview

SPARKLE leverages **empty spots** — spatial locations outside cell segmentation masks — as built-in ambient RNA probes. Since empty spots contain no cells, any expression detected there is purely ambient RNA from neighboring cells.

### Architecture

Two pipelines available:

| Pipeline | Architecture | Best for |
|----------|-------------|----------|
| **Cell-based** (recommended) | Cells intact + empty space binned separately. Cell-to-cell ambient prediction with zero diagonal → natural self-exclusion. | Data with cell segmentation |
| Bin-level | Uniform grid bins mixing cell and empty DNBs. Bin-to-bin ambient prediction. | Legacy / no cell masks |

### Key Features

- **No external reference needed** — empty spots serve as built-in ambient probes
- **Cell-based architecture** — cells never split across bins, ambient only from other cells
- **Self-confidence penalty** — `f(s)=1/(1+(s/p90)²)` protects true expression of high-expressors
- **Gene-specific leakage rates (α)** — different genes leak at different rates
- **Global spatial decay (λ)** — physical diffusion distance estimated via grid search

## Quick Start

```python
from stambient import SPARKLE

model = SPARKLE(cell_based=True)  # recommended config

# dnb_expr: [genes × DNBs], dnb_coords: [DNBs × 2], dnb_labels: [DNBs] (-1 = empty)
corrected, diagnostics = model.fit_transform_from_dnb(
    dnb_expr, dnb_coords, dnb_labels
)
```

## Algorithm

### Cell-Based Pipeline (recommended)

1. **Cells**: extracted intact from DNB labels — centroid (x,y), area (DNB count), expression/area.
2. **Empty bins**: non-cell DNBs binned spatially into irregular bins.
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

Reproduced via `evaluation/scripts/final_comparison.py --dataset synthetic --all-scenarios`. The synthetic benchmark evaluates raw RMSE against ground-truth per-cell expression, so SPARKLE is run with `self_confidence_penalty=False` to maximise ambient removal (real-data runs keep the penalty on by default to protect biological signal).

| Scenario | SPARKLE | SoupX | Spatial SoupX |
|----------|:---:|:---:|:---:|
| S1 Sparse (40% empty) | **69.5%** | 28.3% | 61.6% |
| S2 Medium (25%) | **77.0%** | 35.3% | 69.6% |
| S3 Dense (10%) | **85.2%** | 14.7% | 76.4% |
| S4 Short λ=20µm | **50.6%** | 7.4% | 37.7% |
| S5 Long λ=100µm | **82.8%** | 24.4% | 77.8% |
| S6 Weak α≤0.005 | **60.8%** | 12.4% | 49.8% |
| S7 Strong α≤0.10 | **94.7%** | 20.0% | 91.2% |
| S8 Very Sparse (>50%) | **55.6%** | 9.3% | 46.4% |
| S9 Multi-Cell-Type | **60.9%** | 28.2% | −14.2% |
| S10 Marker Benchmark | **74.7%** | 24.0% | 12.2% |
| **Average** | **71.2%** | **20.4%** | **50.8%** |

SPARKLE leads in all scenarios. The multi-type scenarios (S9–S10) expose the limitation of a global contamination fraction ρ: Spatial SoupX over-subtracts cell-type-specific marker genes, producing negative RMSE reductions, while SPARKLE’s cell-based, gene-specific model preserves the true signal.

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

**CellBender** — VAE prior estimation fails on spatial DNB data (empty DNBs have 1-2 UMI → zero division / NaN). Falls back to empirical subtraction on full datasets.

**CellClear** — Requires ≥2000 genes for NMF statistical power, but spatial DNB background bins fail the `contamined_genes_detection` step: background expression profiles are too sparse to match any cell cluster (core algorithm designed for scRNA-seq droplet data, not spatial).

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
| `cell_based` | `False` | Use cell-based pipeline (recommended for segmented data) |
| `bin_size` | 50 | DNBs per bin edge (cell-based: empty bin size) |
| `distance_metric` | `'exponential'` | `'exponential'`, `'gaussian'`, or `'inverse'` |
| `lambda_distance` | `None` | Distance decay (μm); auto-estimated if None |
| `max_radius` | 200.0 | Max neighbor search radius (μm) |
| `n_high_genes` | 500 | Top genes by empty-bin expression |
| `n_lambda_genes` | 50 | Top genes for λ grid search |
| `r2_threshold` | 0.05 | Minimum weighted R² to apply correction |
| `lambda_grid` | `[10,20,30,50,70,100,150,200,300]` | λ candidates |
| `use_local_density` | `False` | β modulation (disabled — harmful) |
| `use_expr_weight` | `False` | EWAP (experimental) |
| `per_gene_lambda` | `False` | Per-gene λ (rejected — unstable) |

### Methods

- `fit_transform_from_dnb(expr, coords, labels)` → `(corrected_matrix, diagnostics)`
- `fit_transform(...)` → `(corrected_matrix, diagnostics)`

### Baselines

- `evaluation/baselines/spatial_soupx.py` — Spatial SoupX (kernel-weighted global ρ)
- `evaluation/baselines/soupx.py` — Original SoupX wrapper (global ρ, no spatial info)
- `evaluation/baselines/cellbender.py` — CellBender wrapper (excluded from comparison, see above)
- `evaluation/scripts/standalone_decontx.py` — DecontX standalone (EM-based, integrated via `decontx-python`)

### Unified Entry Point

```bash
python evaluation/scripts/final_comparison.py \
    --dataset {axolotl,mosta} --x-range X1 X2 --y-range Y1 Y2 \
    --n-genes 200 --methods sparkle,spatial_soupx,soupx,decontx
```

> **CellBender** and **CellClear** are excluded from `final_comparison.py` due to fundamental incompatibility with spatial DNB data. CellBender's VAE requires ≥50 UMI per background barcode (spatial: 1-2). CellClear's NMF-based gene detection fails because background expression profiles are too sparse to match foreground clusters. See "Excluded Methods" above for details.

> **DecontX** is cell-level only (no spatial information, gene-specific α). It runs directly in the base environment via `pip install decontx-python`. On MOSTA, uniform contamination estimates produce NaN correlations for between-type analysis.

## Installation

```bash
pip install -e .
```

## Dependencies

- numpy, scipy, scikit-learn
- anndata (optional, for h5ad I/O in evaluation)
