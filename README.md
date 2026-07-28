# SPARKLE: evidence-constrained correction of local RNA leakage in high-resolution spatial transcriptomics

SPARKLE (Spatial Ambient RNA Kernel-based Leakage Estimator) corrects local
RNA leakage in cell-resolved, sequencing-based spatial transcriptomic data.
It uses capture locations outside cell-segmentation masks as within-sample
spatial evidence and returns a corrected gene-by-cell expression matrix.

The package implements the cell-based framework described in the manuscript.
Segmented cells remain intact throughout the analysis; only out-of-mask
capture locations are spatially aggregated.

## Model overview

SPARKLE separates the input into two observation layers:

- **Segmented cells** are potential leakage sources and the final correction
  targets.
- **Out-of-mask capture locations** are aggregated into background bins and
  used to estimate local leakage.

The standard workflow:

1. aggregates raw counts, effective capture area and centroids for each cell;
2. aggregates out-of-mask capture locations into background bins;
3. builds a sparse background-bin–cell graph;
4. estimates one sample-level spatial decay scale, \(\lambda\), from
   informative genes;
5. estimates a gene-specific leakage coefficient, \(\alpha_g\), and
   goodness-of-fit score, \(R_g^2\);
6. retains only genes whose out-of-mask fit passes the configured threshold;
7. predicts leakage on a zero-diagonal cell–cell graph; and
8. applies expression-dependent conservative shrinkage before subtracting the
   predicted leakage and truncating corrected values at zero.

The standard distance kernel is a truncated exponential:

```text
Kλ(d) = exp(-d / λ),  d ≤ R
        0,             d > R
```

For a corrected gene \(g\) and target cell \(c\), the leakage estimate is
proportional to

```text
αg × target-cell area × conservative-shrinkage weight
   × kernel-weighted expression from neighbouring cells
```

The target cell is excluded from its own leakage prediction.

## Installation

From PyPI:

```bash
pip install stambient
```

Install optional CUDA support:

```bash
pip install "stambient[gpu]"
```

For local development:

```bash
git clone https://github.com/WangShuai-3/SPARKLE.git
cd SPARKLE
pip install -e ".[dev]"
```

SPARKLE requires Python 3.9 or later.

## Input contract

The recommended entry point accepts three aligned objects:

| Input | Shape | Description |
|---|---:|---|
| `spot_expr` | genes × capture locations | Raw, non-negative counts |
| `spot_coords` | capture locations × 2 | Two-dimensional coordinates |
| `spot_labels` | capture locations | Cell identifier; `-1` denotes an out-of-mask location |

Coordinates, `bin_size`, `lambda_grid` and `max_radius` must use the same
units. Convert coordinates to micrometres before fitting when possible.

Cell labels may be arbitrary non-negative integers. The returned matrix is
ordered by the sorted unique non-negative labels. Keep the corresponding
`cell_ids` from the loader or record the label order before fitting.

## Quick start

```python
from stambient import SPARKLE
from stambient.io_utils import load_stereoseq

data = load_stereoseq("sample.gem.gz")

model = SPARKLE(
    bin_size=50,
    max_radius=200,
    lambda_grid=[10, 30, 50, 70, 100, 150, 200, 300],
    r2_threshold=0.01,
)

corrected, diagnostics = model.fit_transform(
    data["spot_expr"],
    data["spot_coords"],
    data["spot_labels"],
)

print(corrected.shape)
print(diagnostics["lambda_estimated"])
```

`corrected` is a non-negative gene-by-cell matrix. Because the estimated
leakage is continuous, corrected values are not restricted to integers.

### GPU execution

```python
model = SPARKLE(
    bin_size=50,
    max_radius=200,
    lambda_grid=[10, 30, 50, 70, 100, 150, 200, 300],
    r2_threshold=0.01,
    use_gpu=True,
    gpu_dtype="float64",
)

corrected, diagnostics = model.fit_transform(
    data["spot_expr"],
    data["spot_coords"],
    data["spot_labels"],
)
```

The GPU backend accelerates sparse graph operations, scale selection,
gene-specific regression and correction. `float64` is the recommended mode
for reproducible CPU/GPU comparison. `mixed` and `float32` are optional
approximate modes and should be validated on the target dataset.

If CUDA execution cannot be initialized or fails during fitting, SPARKLE
restarts the analysis on CPU and records the fallback in `diagnostics`.

## Data loading

Loaders return a common dictionary containing `spot_expr`, `spot_coords`,
`spot_labels`, `gene_names` and `cell_ids`.

### Stereo-seq GEM with cell labels

```python
from stambient.io_utils import load_stereoseq

data = load_stereoseq(
    "sample.gem.gz",
    gene_col="geneID",
    count_col="MIDCounts",
    cell_label_col="cell",
)
```

The loader accepts plain-text or gzip-compressed files, skips comment lines
beginning with `#` and detects tab- or comma-delimited input. Configure
platform-specific background labels through `empty_labels`.

### Full Stereo-seq GEM plus segmented scGEM

```python
from stambient.io_utils import load_RYTools_data

data = load_RYTools_data(
    "sample.gem.gz",
    "sample.scgem.csv.gz",
    gene_col="geneID",
    count_col="MIDCounts",
    cell_label_col="cell",
)
```

Capture locations present in the scGEM receive their segmented cell label.
Locations present only in the full GEM are assigned `-1` and form the
out-of-mask observation layer.

### Visium HD feature slice

```python
from stambient.io_utils import load_visiumhd

data = load_visiumhd("Visium_HD_feature_slice.h5")
```

The feature-slice file must contain a cell-segmentation mask. Capture squares
not covered by the mask are assigned `-1`.

Platform coordinate scales differ. For example, a bin side of 50 units on a
0.5-µm Stereo-seq grid corresponds to 25 µm, whereas a comparable side on a
2-µm Visium HD grid is approximately 12–13 squares.

## Main parameters

| Parameter | Default | Role |
|---|---:|---|
| `bin_size` | `50` | Side length used only to aggregate out-of-mask capture locations |
| `max_radius` | `200.0` | Truncation radius of the sparse spatial graphs |
| `lambda_grid` | `[10,20,30,50,70,100,150,200,300]` | Candidate sample-level spatial decay scales |
| `n_high_genes` | `None` | Highest-information genes eligible for fitting and correction; `None` uses all genes |
| `n_lambda_genes` | `50` | Top informative genes used to select the shared spatial scale |
| `r2_threshold` | `0.01` | Minimum out-of-mask \(R_g^2\) required for correction |
| `self_confidence_penalty` | `True` | Protects cells with high source expression from excessive subtraction |
| `use_gpu` | `False` | Requests the CUDA backend |
| `gpu_dtype` | `"float64"` | GPU precision mode |
| `gpu_gene_batch_size` | `None` | Gene batch size; `None` adapts to available GPU memory |
| `verbose` | `True` | Prints progress and model diagnostics |

The manuscript uses the standard cell-based framework, a shared
sample-level \(\lambda\), gene-specific \(\alpha_g\), goodness-of-fit gating
and expression-dependent conservative shrinkage.

## Diagnostics

The returned diagnostics include:

- the selected spatial decay scale;
- the candidate-scale residual profile;
- gene-specific leakage coefficients and \(R_g^2\) values;
- the genes entering correction;
- cell and background-bin counts;
- graph sizes and non-zero edge counts;
- stage-level timing information;
- CPU/GPU backend and precision metadata; and
- GPU fallback information, when applicable.

Inspect these values before using a corrected matrix downstream. Limited
out-of-mask coverage, low per-gene fit, an optimum at the edge of the scale
grid or poor graph connectivity may indicate that the data do not support
local leakage correction.

## Interpretation and limitations

Out-of-mask counts are not assumed to be pure technical negatives. They may
contain missed cytoplasm, unsegmented cells, tissue debris, nonspecific
capture or genuine extracellular RNA. SPARKLE therefore corrects only the
component that is locally predictable from observed neighbouring cells and
passes the per-gene evidence threshold.

The current model uses one shared, isotropic spatial scale and requires:

- reliable cell-segmentation labels;
- sufficient out-of-mask spatial coverage;
- coordinates at a meaningful and internally consistent scale; and
- a study objective in which extracellular signal is considered unwanted.

SPARKLE should not be interpreted as a method for removing genuine
extracellular RNA biology.

## Evaluation

The workflows used in the manuscript are documented in
[`evaluation/README.md`](evaluation/README.md). That document describes the
frozen datasets, methods and analysis stages without reproducing manuscript
results.

## Citation

If you use SPARKLE, please cite:

> SPARKLE: evidence-constrained correction of local RNA leakage in
> high-resolution spatial transcriptomics.

Publication metadata will be added after publication.

## License

SPARKLE is released under the MIT License.
