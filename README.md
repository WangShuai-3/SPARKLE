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

### GPU acceleration

The recommended cell-based pipeline can offload its batched sparse matrix
multiplications, lambda search, per-gene regression, and correction to a CUDA
GPU through PyTorch:

```python
model = SPARKLE(
    cell_based=True,
    use_gpu=True,
    gpu_dtype="float64",       # "mixed" or "float32" for approximate fast mode
    gpu_gene_batch_size=None,  # adapt automatically to available VRAM
)
corrected, diagnostics = model.fit_transform(
    data["spot_expr"], data["spot_coords"], data["spot_labels"]
)
print(diagnostics["compute_backend"], diagnostics["gpu_device"])
```

Install a PyTorch build compatible with the machine's CUDA runtime (or use
`pip install 'stambient[gpu]'`). If PyTorch, CUDA, or the device is unavailable,
SPARKLE emits a warning and transparently runs on CPU. The default `float64`
mode keeps fitted parameters and corrected expression consistent with the
NumPy/SciPy implementation. `mixed` uses float32 CSR multiplication with
float64 regression/correction reductions, while `float32` keeps all GPU math
in float32. The latter two are approximate fast modes; validate them for the
target dataset. The legacy `cell_based=False` pipeline remains CPU only.

The optimized GPU path builds each KDTree distance topology once, transfers
its CSR row/column indices and distances once, and computes all lambda weights
on-device. Gene batch size is selected from currently free VRAM unless it is
overridden. Diagnostics expose `gpu_dtype`, `gpu_gene_batch_size`,
`gpu_sparse_format`, both graph `*_nnz` values, and detailed `timings_sec` for
aggregation, binning, graph construction, lambda search, alpha estimation, and
correction.

CPU/GPU consistency was validated in the `scvi` environment on an NVIDIA RTX
4090. On the 300-cell/600-gene validation, all three modes selected the same
lambda and passed alpha, R², and corrected-matrix checks. Maximum corrected
matrix absolute error was `3.126e-13` for `float64` (`rtol=1e-8`,
`atol=1e-10`), `3.822e-5` for `mixed`, and `4.955e-5` for `float32` (both
checked with `atol=1e-4`). Small synthetic inputs are slower on GPU because
CUDA startup and preprocessing dominate; use the resource benchmark below to
determine the crossover point for a real dataset.

### MouseBrain CPU/GPU resource benchmark (pre-optimization baseline)

The table below records the original COO/fixed-batch implementation so the
effect of the CSR optimization remains auditable. The existing 8-window CPU
benchmark was paired with an RTX 4090 GPU run using
the same `x=6000–20000`, `y=2000–15000`, 10,000-gene, 10,000-high-gene,
`r2_threshold=0` configuration. GPU alpha estimation and correction process 512
genes per batch, so the 71,476-cell maximum window remains well below 24 GB of
VRAM. Times below exclude the shared one-time GEM loading step.

| Run | DNBs | Cells | CPU (s) | GPU (s) | Speedup | GPU allocated peak | GPU reserved peak |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 58,381,645 | 71,476 | 1,192.1 | 1,329.4 | 0.90× | 3,215.9 MiB | 4,510 MiB |
| 2 | 50,313,652 | 68,092 | 763.5 | 1,731.5 | 0.44× | 2,952.1 MiB | 4,278 MiB |
| 3 | 38,032,645 | 56,539 | 472.6 | 392.3 | 1.20× | 2,553.5 MiB | 3,624 MiB |
| 4 | 25,035,453 | 39,770 | 264.0 | 241.9 | 1.09× | 1,791.7 MiB | 2,548 MiB |
| 5 | 15,087,630 | 25,384 | 137.2 | 52.1 | 2.63× | 1,145.8 MiB | 1,674 MiB |
| 6 | 8,047,416 | 14,577 | 61.3 | 14.3 | 4.28× | 654.3 MiB | 960 MiB |
| 7 | 3,409,044 | 6,804 | 25.8 | 4.2 | 6.17× | 302.6 MiB | 414 MiB |
| 8 | 893,581 | 1,831 | 8.7 | 1.0 | 8.69× | 81.5 MiB | 128 MiB |

CPU and GPU selected the same lambda in all eight runs. GPU acceleration is
strongest here for the 1,831–25,384-cell windows; the two largest/high-density
windows are slower on the RTX 4090 because CPU-side preprocessing, float64
sparse kernels, transfers, and 512-gene batching dominate. Consequently the
sum across all windows is 2,925.2 seconds on CPU versus 3,766.7 seconds on GPU
(0.78× overall), even though six individual windows are faster or near parity.
Use the measured crossover rather than assuming GPU is always faster.

The paired results and plots are stored in
`evaluation/reports/resource_benchmark_mousebrain_cpu_gpu.csv` and
`evaluation/reports/resource_benchmark_{runtime,gpu_speedup,memory}.png`.

### MouseBrain benchmark after the six GPU optimizations (before preprocessing fix)

The same eight windows were rerun in strict `float64` mode after introducing
one-time distance topology construction, one-time GPU graph transfer, GPU CSR,
adaptive gene batches, selectable precision, and stage/nnz diagnostics.

| Run | Cells | CPU (s) | Old GPU (s) | Optimized GPU (s) | Optimized speedup | Adaptive batch | GPU allocated peak |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 71,476 | 1,192.1 | 1,329.4 | 927.7 | 1.29× | 1,568 | 7,120.2 MiB |
| 2 | 68,092 | 763.5 | 1,731.5 | 546.5 | 1.40× | 1,760 | 8,261.8 MiB |
| 3 | 56,539 | 472.6 | 392.3 | 262.6 | 1.80× | 2,240 | 7,730.7 MiB |
| 4 | 39,770 | 264.0 | 241.9 | 124.0 | 2.13× | 3,232 | 6,257.5 MiB |
| 5 | 25,384 | 137.2 | 52.1 | 51.4 | 2.67× | 5,056 | 8,151.4 MiB |
| 6 | 14,577 | 61.3 | 14.3 | 13.4 | 4.57× | 8,928 | 4,569.4 MiB |
| 7 | 6,804 | 25.8 | 4.2 | 4.4 | 5.85× | 9,984 | 2,305.9 MiB |
| 8 | 1,831 | 8.7 | 1.0 | 1.0 | 8.24× | 9,984 | 686.6 MiB |

All lambda choices match CPU. Total GPU runtime fell from 3,766.7 to 1,931.0
seconds (48.7% reduction), giving a 1.51× aggregate speedup over the 2,925.2
second CPU baseline. Small windows are essentially unchanged because launch
and preprocessing overhead dominate; the largest gains occur where the old
implementation repeatedly rebuilt/transferred graphs or split 10,000 genes
into many 512-gene batches.

The stage diagnostics showed that empty-DNB binning consumed 1,768.9 of
1,927.8 measured pipeline seconds across the eight GPU runs. Alpha estimation
and correction consumed 42.6 and 53.8 seconds respectively, while both graph
build stages plus lambda search consumed 7.1 seconds. This identified the
remaining CPU-side preprocessing bottleneck, addressed below.
Adaptive batching trades more VRAM (up to 8,261.8 MiB allocated and 10,798 MiB
reserved here) for fewer launches; set `gpu_gene_batch_size` explicitly when
memory must be capped.

The optimized CSV is
`evaluation/reports/resource_benchmark_mousebrain_cpu_gpu_optimized.csv`.

### Vectorized empty-DNB preprocessing

The previous implementation materialized each bin with
`empty_indices[inverse == bin_id]`, scanning every empty DNB once per bin. It
has been replaced by one aligned `(DNB index, bin index)` mapping and direct
sparse-matrix construction. Bin areas and centroids now use `bincount`, making
assignment linear in the number of empty DNBs instead of approximately
`O(n_empty_DNBs × n_bins)`.

The complete strict-float64 MouseBrain benchmark was rerun for both backends
after this change, using the same 8 windows and parameters:

| Run | Cells | Old empty binning (s) | New empty binning (s) | New CPU (s) | New GPU (s) | GPU speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 71,476 | 885.3 | 6.87 | 132.0 | 49.3 | 2.68× |
| 2 | 68,092 | 507.8 | 5.39 | 92.0 | 44.5 | 2.07× |
| 3 | 56,539 | 229.6 | 3.74 | 77.8 | 36.6 | 2.13× |
| 4 | 39,770 | 101.5 | 2.36 | 53.3 | 24.6 | 2.16× |
| 5 | 25,384 | 37.5 | 1.43 | 31.9 | 15.1 | 2.11× |
| 6 | 14,577 | 5.86 | 0.73 | 17.3 | 8.2 | 2.11× |
| 7 | 6,804 | 1.20 | 0.29 | 6.6 | 3.5 | 1.91× |
| 8 | 1,831 | 0.11 | 0.04 | 1.7 | 1.0 | 1.68× |

Across all windows, empty preprocessing fell from 1,768.9 to 20.9 seconds
(98.8% reduction). Current CPU runtime is 412.6 seconds versus 2,925.2 seconds
before the preprocessing fix (7.1× faster). Current GPU runtime is 182.8
seconds versus 1,931.0 seconds before the fix (10.6× faster), and is 2.26×
faster than the current CPU under the same code. Lambda and corrected-gene
counts are unchanged in every window.

The detailed current results are stored in
`evaluation/reports/resource_benchmark_mousebrain_cpu_preprocessing_optimized.csv`
and
`evaluation/reports/resource_benchmark_mousebrain_cpu_gpu_preprocessing_optimized.csv`.

```bash
# Numerical reliability (use --require-gpu on a GPU node/CI worker)
conda run -n scvi python evaluation/scripts/compare_sparkle_cpu_gpu.py \
    --require-gpu

# Paired current CPU/GPU runtime, process RSS, and peak CUDA memory on MouseBrain
conda run -n scvi python evaluation/scripts/benchmark_resource.py \
    --x-range 6000 20000 --y-range 2000 15000 \
    --n-genes 10000 --n-high-genes 10000 --r2-threshold 0 \
    --n-runs 8 --backends cpu gpu --require-gpu --plot \
    --gpu-dtype float64 \
    --output evaluation/reports/resource_benchmark_mousebrain_cpu_gpu_final.csv
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
| `r2_threshold` | 0.01 | Minimum weighted R² to apply correction |
| `lambda_grid` | `[10,20,30,50,70,100,150,200,300]` | λ candidates |
| `use_local_density` | `False` | β modulation (disabled — harmful) |
| `use_expr_weight` | `False` | EWAP (experimental) |
| `per_gene_lambda` | `False` | Per-gene λ (rejected — unstable) |
| `use_gpu` | `False` | Use PyTorch CUDA in the cell-based pipeline; automatically fall back to CPU |
| `gpu_dtype` | `'float64'` | GPU precision: `'float64'`, `'mixed'`, or `'float32'` |
| `gpu_gene_batch_size` | `None` | Genes per GPU batch; `None` adapts to free VRAM |

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

# VisiumHD (Human Colon Cancer 6.5mm, segmentation embedded in feature_slice.h5)
python evaluation/scripts/final_comparison.py \
    --dataset visiumhd --x-range 1250 1650 --y-range 500 900 \
    --n-genes 200 --n-high-genes 200 \
    --methods sparkle,spatial_soupx,soupx,decontx

# Ovarian (Visium HD Human Ovarian Cancer FF, same feature_slice.h5 layout with embedded segmentation)
python evaluation/scripts/final_comparison.py \
    --dataset ovarian --x-range 1250 1650 --y-range 500 900 \
    --n-genes 200 --n-high-genes 200 \
    --methods sparkle,spatial_soupx,soupx,decontx

# CRC (same window, separate Proseg and StarDist conditions)
python evaluation/scripts/final_comparison.py \
    --dataset crc --crc-segmentation both \
    --x-range 14300 14900 --y-range 2850 3450 \
    --n-genes 2000 --n-high-genes 2000 --cut-genes \
    --methods sparkle,spatial_soupx

# Synthetic
python evaluation/scripts/final_comparison.py \
    --dataset synthetic --all-scenarios \
    --methods sparkle,spatial_soupx,soupx,decontx --use-gpu
```

> **Note on Visium HD segmentation:** the `visiumhd`/`ovarian` loaders read cell labels from `segmentations/cell_segmentation_mask` embedded in the feature_slice.h5. Some Visium HD samples (e.g. Human Colon Cancer P1) ship a feature_slice.h5 *without* an embedded `segmentations` group; those cannot be evaluated unless the corresponding 10x segmented outputs (cell_segmentations.geojson) are available. The loader raises a clear error in that case.

> **CellBender** and **CellClear** are excluded from `final_comparison.py` due to fundamental incompatibility with spatial spot data. CellBender's VAE requires ≥50 UMI per background barcode (spatial: 1-2). CellClear's NMF-based gene detection fails because background expression profiles are too sparse to match foreground clusters. See "Excluded Methods" above for details.

> **DecontX** is cell-level only (no spatial information, gene-specific α). It runs directly in the base environment via `pip install decontx-python`. On MOSTA, uniform contamination estimates produce NaN correlations for between-type analysis.

## Additional Evaluation Scripts

- **`evaluation/scripts/evaluate_mousebrain_h5ad.py`** — post-hoc evaluation of corrected MouseBrain h5ad files: cell-subclass correlation heatmaps, silhouette score, and comparison against an snRNA-seq reference.
- **`evaluation/scripts/prepare_mousebrain_snrna_reference.R`** — extract `cell_subclass` pseudobulk profiles from the provided Seurat RDS reference for the script above.
- **`evaluation/scripts/compare_sparkle_cpu_gpu.py`** — verify CPU/GPU lambda, alpha, R², and corrected-expression consistency on deterministic synthetic data.
- **`evaluation/scripts/benchmark_resource.py`** — paired CPU/GPU SPARKLE benchmark across shrinking MouseBrain windows. Reports runtime, speedup, host RSS, CUDA peak allocated/reserved memory, and produces runtime/speedup/memory plots.

Run any script with `--help` for detailed options.

## Installation

```bash
pip install -e .
```

## Dependencies

- numpy, scipy, scikit-learn
- anndata (optional, for h5ad I/O in evaluation)
