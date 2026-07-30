# SPARKLE evaluation

This directory contains the evaluation code, input adapters, and reproducible
outputs used for the final manuscript. The algorithm is implemented in
`stambient/`. This document defines the evaluation design and execution
entry points; it does not report numerical results or conclusions.

## Scope

- Synthetic data: predefined scenarios S1–S10.
- Real data: axolotl brain, MouseBrain T304, and human ovarian cancer.
- Methods: uncorrected data (RAW), SPARKLE, SoupX, DecontX, and SpotClean.
- Synthetic metrics: RMSE on raw counts and cell-wise Pearson \(R^2\).
- Axolotl: SST expression in source cells, neighbouring cells, and other
  cells.
- MouseBrain: correlations among cell-type pseudobulks, correlations with an
  snRNA-seq reference, and RCTD singlet confidence at the `Cell_group` level.
- Ovarian: cell-type pseudobulk and single-cell-reference comparisons, RCTD,
  curated tumour/stromal markers using `log1p(CP10K)` expression and log2 fold
  change, and CellChat analysis of COL1A2–SDC4.
- Stability and computational performance: the candidate range for the
  shared decay scale, out-of-mask aggregation size, and CPU/GPU resource
  measurements on MouseBrain windows.


## Layout

```text
evaluation/
├── synthetic/
│   ├── generator.py                  # Cell-resolved synthetic-data generator
│   └── scenarios.py                  # Definitions of S1–S10
├── baselines/
│   ├── soupx.py
│   └── spotclean_official.py         # Python adapter for the official R package
├── scripts/                          # Execution, summary, and plotting scripts
├── data/                             # Local inputs and single-cell references
└── reports/                          # Generated at runtime; not a source of README results
```

Run all commands from the repository root. Python dependencies are described
in the root README. SoupX, DecontX, RCTD, CellChat, and SpotClean additionally
require their corresponding R/Bioconductor environments. Before running an
evaluation, verify the input paths, software versions, random seed, and
command-line parameters, and retain the generated configuration or provenance
files.

## Synthetic data

The generator uses a \(500\times500\) DNB grid, 0.5-µm DNB spacing, 500 genes,
and 80 highly expressed genes. Per-gene baseline and high-expression rates
follow log-spaced gradients, so most genes are expressed very low and only a
few are abundant. Scenarios contain 6 spatially clustered cell types with
imbalanced proportions (8 in S10); marker genes are strictly
cell-type-specific in the ground truth, and a 95% UMI dropout is applied
(binomial thinning of the clean and ambient parts independently); the
correction ground truth is the observed clean expression. Every scenario
varies one principal source of difficulty:

| Scenario | Principal variation |
|---|---|
| S1–S3 | Increasing cell coverage from sparse to dense |
| S4–S5 | Shorter or longer leakage-decay scale |
| S6–S7 | Weaker or stronger leakage fraction |
| S8 | High out-of-cell fraction |
| S9 | Higher marker-gene fraction |
| S10 | More cell types |

Generate the SPARKLE, SoupX, and DecontX corrected matrices through the shared
entry point:

```bash
python evaluation/scripts/final_comparison.py \
  --dataset synthetic \
  --all-scenarios \
  --methods sparkle,soupx,decontx
```

SPARKLE must always be run with `self_confidence_penalty=True` (the default)
on synthetic scenarios; do not disable it. Cell types form spatial domains,
so marker-owning cells are each other's strongest leakage sources — without
the penalty they over-subtract one another and marker specificity degrades.

For synthetic scenarios SoupX is run with `tfidfMin=0.2`: the scenarios use
3–5 balanced cell types, so the best achievable marker tf-idf is
\(\log(n_\mathrm{types})\approx 1.1\), and the default `tfidfMin=1.0` leaves
no headroom once any background expression is present. If SoupX fails on a
dataset, the failure is recorded explicitly as NaN metrics rather than
substituted with a heuristic fallback.

Run SpotClean through the Python wrapper for the official R package:

```bash
python evaluation/scripts/run_spotclean_official.py \
  --datasets synthetic \
  --scenarios S1 S2 S3 S4 S5 S6 S7 S8 S9 S10 \
  --overwrite
```

RMSE is calculated on unnormalised raw counts. Cell-wise Pearson \(R^2\)
compares each corrected cell with its uncontaminated ground truth across all
simulated genes; cell-type clustering accuracy (ARI of KMeans on
log1p(CP10K) PCs against the ground-truth cell types) measures how well
correction preserves cell-type structure:

```bash
python evaluation/scripts/evaluate_synthetic_cell_r2.py
python evaluation/scripts/evaluate_synthetic_clustering.py
python evaluation/scripts/visualize_synthetic_rmse.py
```


## Correcting the real datasets

The final manuscript uses the following fixed windows:

| Dataset | X range | Y range |
|---|---:|---:|
| Axolotl | 10500–12500 | 6000–11100 |
| MouseBrain T304 | 12500–20000 | 2000–10000 |
| Ovarian | 1000–1800 | 300–1100 |

Generate the SPARKLE, SoupX, and DecontX cell-expression matrices:

```bash
python evaluation/scripts/final_comparison.py \
  --dataset axolotl \
  --x-range 10500 12500 --y-range 6000 11100 \
  --no-cut-genes \
  --methods sparkle,soupx,decontx

python evaluation/scripts/final_comparison.py \
  --dataset mousebrain \
  --x-range 12500 20000 --y-range 2000 10000 \
  --n-genes 30000 --n-high-genes 30000 \
  --no-cut-genes --annotation-level cell_group \
  --methods sparkle,soupx,decontx

python evaluation/scripts/final_comparison.py \
  --dataset ovarian \
  --x-range 1000 1800 --y-range 300 1100 \
  --n-genes 30000 --n-high-genes 30000 \
  --no-cut-genes \
  --methods sparkle,soupx,decontx
```

Add `--no-save-h5ad` to `final_comparison.py` when only summary metrics are
needed. Full manuscript reproduction requires the h5ad files for downstream
analyses, so the default output should normally be retained.

### Official SpotClean

`run_spotclean_official.py` aggregates each intact segmented cell into one
tissue spot and aggregates out-of-mask DNBs into background spots. The
centroid of each aggregate is used as its spot coordinate. Python prepares
the inputs, invokes R, and writes the h5ad files; model fitting and correction
are performed by the official SpotClean R package.

```bash
python evaluation/scripts/run_spotclean_official.py \
  --datasets axolotl mousebrain ovarian \
  --overwrite
```

The official implementation constructs large distance objects. After
full-window aggregation, the MouseBrain adapter partitions the space into
\(3\times3\) cores and adds a halo equal to the maximum candidate radius
around every core. Only core cells are retained after each block is corrected,
and the blocks are concatenated in the original cell order. Neither cells nor
background spots are randomly sampled. The prepared input can be reviewed
before running R and then reused:

```bash
python evaluation/scripts/run_spotclean_official.py \
  --datasets mousebrain \
  --prepare-only

python evaluation/scripts/run_spotclean_official.py \
  --datasets mousebrain \
  --reuse-prepared-input \
  --overwrite
```

## Axolotl: local SST dispersion

Cells are assigned to three groups according to their relationship to
SST-positive source cells: annotated sstINs, non-sstIN cells within the
specified radius of an sstIN, and all remaining non-sstIN cells. Every method
uses the same cells and spatial-neighbourhood definition.

```bash
python evaluation/scripts/visualize_axolotl_sst_boxplots.py \
  --tag axolotl_x10500-12500_y6000-11100 \
  --input-dir evaluation/reports/h5ad \
  --output-dir evaluation/reports/axolotl_figures \
  --neighbor-radius 50
```

## MouseBrain: reference concordance and RCTD

Prepare the snRNA-seq `Cell_group` pseudobulk reference when it is not already
available, and run RCTD at the same hierarchy:

```bash
Rscript evaluation/scripts/prepare_mousebrain_snrna_reference.R

RCTD_DATASET_TAG=mousebrain_x12500-20000_y2000-10000 \
RCTD_REFERENCE_LEVEL=Cell_group \
RCTD_METHODS=RAW,SPARKLE,SoupX,DecontX,SpotClean \
RCTD_MAX_CORES=16 \
Rscript evaluation/scripts/run_rctd_mousebrain.R

Rscript evaluation/scripts/enrich_first_type_scores_mousebrain.R
python evaluation/scripts/inject_rctd_mousebrain.py
```

The final comparison must consistently use `Cell_group`; do not combine RCTD
labels or denominators from different hierarchy levels. Restrict expression
evaluation explicitly to the final-manuscript methods:

```bash
python evaluation/scripts/evaluate_mousebrain_h5ad.py \
  --tag mousebrain_x12500-20000_y2000-10000 \
  --input-dir evaluation/reports/h5ad_mousebrain_annotated \
  --snrna-ref evaluation/data/mousebrain/snrna_cell_group_pseudobulk.csv \
  --output-dir evaluation/reports/mousebrain_eval \
  --methods RAW,SPARKLE,SoupX,DecontX,SpotClean

python evaluation/scripts/visualize_mousebrain_comparison.py
```

## Ovarian: single-cell reference, markers, and CellChat

First construct the scRNA-seq cell-type pseudobulk reference and run RCTD.
Expression comparisons use the RAW `first_type` assignment as a shared cell
grouping for every method, preventing changes in inferred labels from being
confounded with changes in expression:

```bash
python evaluation/scripts/prepare_ovarian_scrna_reference.py

RCTD_DATASET_TAG=ovarian_x1000-1800_y300-1100 \
RCTD_METHODS=RAW,SPARKLE,SoupX,DecontX,SpotClean \
RCTD_MAX_CORES=16 \
Rscript evaluation/scripts/run_rctd_ovarian.R

python evaluation/scripts/inject_rctd_annotations.py \
  --tag ovarian_x1000-1800_y300-1100 \
  --input-dir evaluation/reports/h5ad \
  --first-type-dir evaluation/reports/rctd_ovarian/first_type \
  --output-dir evaluation/reports/h5ad_ovarian_annotated \
  --annotation-method RAW

python evaluation/scripts/evaluate_mousebrain_h5ad.py \
  --tag ovarian_x1000-1800_y300-1100 \
  --input-dir evaluation/reports/h5ad_ovarian_annotated \
  --snrna-ref evaluation/data/ovarian/scrna_celltype_pseudobulk.csv \
  --output-dir evaluation/reports/ovarian_eval \
  --methods RAW,SPARKLE,SoupX,DecontX,SpotClean
```

Generate the curated tumour/stromal marker comparison used in the manuscript:

```bash
python evaluation/scripts/analyze_ovarian_cancer_markers.py
```

Run spatial CellChat and the non-spatial scRNA-seq CellChat reference
separately:

```bash
CELLCHAT_METHODS=RAW,SPARKLE,SoupX,DecontX,SpotClean \
Rscript evaluation/scripts/run_cellchat_spatial_ovarian.R --preflight-only

CELLCHAT_METHODS=RAW,SPARKLE,SoupX,DecontX,SpotClean \
Rscript evaluation/scripts/run_cellchat_spatial_ovarian.R

Rscript evaluation/scripts/run_cellchat_ovarian_scrna_reference.R
python evaluation/scripts/plot_ovarian_col1a2_sdc4_comparison.py
```

The focused comparison uses COL1A2–SDC4 and reports communication
probabilities in the main figure.

## Parameter stability and computational performance

The candidate range for the shared decay scale determines which propagation
distances the model can represent and is therefore varied in a synthetic
scenario. Out-of-mask aggregation size trades spatial resolution against
computational cost and is therefore tested separately in the fixed axolotl
window. Each analysis changes only its target parameter while holding the
other settings constant.

```bash
python evaluation/scripts/benchmark_lambda_stability_synthetic.py --plot
python evaluation/scripts/benchmark_binsize_axolotl.py --plot
```

The CPU/GPU benchmark uses identical inputs and model parameters on
progressively larger MouseBrain windows. Strict numerical comparison uses
`float64`; `--require-gpu` prevents a silent CPU fallback on GPU nodes:

```bash
conda run -n scvi python evaluation/scripts/compare_sparkle_cpu_gpu.py \
  --gpu-dtype float64 \
  --require-gpu

conda run -n scvi python evaluation/scripts/benchmark_resource.py \
  --x-range 6000 20000 --y-range 2000 15000 \
  --n-genes 10000 --n-high-genes 10000 \
  --r2-threshold 0 \
  --n-runs 8 \
  --backends cpu gpu \
  --require-gpu \
  --gpu-dtype float64 \
  --plot \
  --output evaluation/reports/resource_benchmark_mousebrain_cpu_gpu_final.csv
```

The resource table records runtime, host RSS, GPU allocated/reserved memory,
and stage-level timings.
