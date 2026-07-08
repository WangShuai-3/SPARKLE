# RYTools Stereo-seq data: end-to-end SPARKLE example

This example shows how to run **SPARKLE** on the output of the **RYTools** Stereo-seq pipeline.

RYTools typically produces two files:

1. **`<sample>.gem.gz`** — a full, unlabeled GEM file with every DNB (including background/empty DNBs):

   ```
   x\ty\tgeneID\tMIDCounts
   100\t200\tGeneA\t3
   100\t201\tGeneB\t1
   ...
   ```

2. **`<sample>.scgem.gz`** — a single-cell GEM file that contains only DNBs that belong to a cell:

   ```
   x\ty\tgeneID\tMIDCounts\tcell
   100\t200\tGeneA\t3\tCell_1
   102\t205\tGeneC\t2\tCell_2
   ...
   ```

   The scGEM file does **not** contain background DNBs. DNBs that appear in the full GEM but are missing from the scGEM will be automatically labelled as empty (`-1`) by SPARKLE and used as built-in ambient probes.

> All paths, column names, and parameters below are placeholders. Replace them with values matching your experiment.

---

## Python: load, run SPARKLE, and save h5ad

```python
import numpy as np
import anndata as ad
from stambient import SPARKLE
from stambient.io_utils import load_RYTools_data

# ---------------------------------------------------------------------------
# 1. Load the RYTools GEM + scGEM pair
# ---------------------------------------------------------------------------
data = load_RYTools_data(
    gem_path="path/to/sample.gem.gz",       # full GEM, no cell labels
    scgem_path="path/to/sample.scgem.gz",   # cell-labelled scGEM
    gene_col="geneID",
    x_col="x",
    y_col="y",
    count_col="MIDCounts",
    cell_label_col="cell",
    empty_labels={0, -1},     # only relevant if your scGEM happens to contain background rows
    gem_sep=None,             # auto-detect tab/comma
    scgem_sep=None,           # auto-detect tab/comma
    pitch_um=None,            # set to 0.5 if x/y are raw Stereo-seq DNB indices
    verbose=True,
)

spot_expr = data["spot_expr"]      # [genes × DNBs] sparse CSR
spot_coords = data["spot_coords"]  # [DNBs × 2]
spot_labels = data["spot_labels"]   # [DNBs], -1 = empty/background

n_genes, n_dnbs = spot_expr.shape
n_cells = len(data["cell_ids"])
print(f"Loaded {n_genes} genes, {n_dnbs} DNBs, {n_cells} cells.")

# ---------------------------------------------------------------------------
# 2. Run SPARKLE (cell-based pipeline is recommended for segmented data)
# ---------------------------------------------------------------------------
model = SPARKLE(
    cell_based=True,          # keep cells intact, bin empty DNBs separately
    bin_size=50,              # empty-bin size in DNB units (25 µm if pitch=0.5)
    max_radius=200.0,           # µm
    n_high_genes=500,
    n_lambda_genes=50,
    r2_threshold=0.05,
    lambda_grid=[10, 20, 30, 50, 70, 100, 150, 200, 300],
    use_local_density=False,  # recommended default
    verbose=True,
)

corrected, diagnostics = model.fit_transform_from_dnb(
    spot_expr, spot_coords, spot_labels
)

# `corrected` is [genes × cells] (dense or sparse)
if hasattr(corrected, "toarray"):
    corrected = corrected.toarray()

print(f"SPARKLE finished: λ={model.lambda_:.1f} µm, "
      f"{diagnostics.get('n_genes_corrected', '?')} genes corrected")

# ---------------------------------------------------------------------------
# 3. Build per-gene R² annotation from diagnostics
# ---------------------------------------------------------------------------
gene_indices = diagnostics.get("gene_indices")
r2_scores = diagnostics.get("r2_scores")

sparkle_r2 = np.full(n_genes, np.nan, dtype=np.float64)
sparkle_selected = np.zeros(n_genes, dtype=bool)
sparkle_corrected = np.zeros(n_genes, dtype=bool)

if gene_indices is not None and r2_scores is not None:
    gene_indices = np.asarray(gene_indices, dtype=int)
    r2_scores = np.asarray(r2_scores)
    sparkle_r2[gene_indices] = r2_scores
    sparkle_selected[gene_indices] = True
    sparkle_corrected[gene_indices] = r2_scores >= model.r2_threshold

var_data = {
    "sparkle_selected": sparkle_selected,
    "sparkle_corrected": sparkle_corrected,
    "sparkle_r2": sparkle_r2,
}

# ---------------------------------------------------------------------------
# 4. Save raw and corrected cell-level expression as h5ad
# ---------------------------------------------------------------------------
# First compute the raw per-cell expression by summing DNBs per cell.
from scipy.sparse import csr_matrix

cell_dnb_idx = np.where(spot_labels >= 0)[0]
cell_indices = spot_labels[cell_dnb_idx]
C = csr_matrix(
    (np.ones(len(cell_dnb_idx), dtype=np.float64),
     (cell_dnb_idx, cell_indices)),
    shape=(n_dnbs, n_cells),
)
raw_cell = (spot_expr @ C).toarray()   # [genes × cells]

cell_ids = data["cell_ids"]


def save_cell_h5ad(expr_matrix, gene_names, cell_ids, var_data, out_path, method_name=""):
    """Save a [genes × cells] matrix as a cell-level h5ad file."""
    n_cells_expr = expr_matrix.shape[1]
    cell_ids_use = np.asarray(cell_ids)[:n_cells_expr]

    X = expr_matrix.T
    if hasattr(X, "toarray"):
        X = X.astype(np.float32)
    else:
        X = np.asarray(X, dtype=np.float32)

    adata = ad.AnnData(X=X)
    adata.var_names = [str(g) for g in gene_names]
    adata.obs_names = [f"Cell_{cid}" for cid in cell_ids_use]
    adata.obs["cell_id"] = cell_ids_use

    if var_data is not None:
        for key, vals in var_data.items():
            adata.var[key] = vals

    if method_name:
        adata.uns["method"] = str(method_name)

    adata.write_h5ad(out_path)
    print(f"Saved {method_name} h5ad: {out_path}")


save_cell_h5ad(
    raw_cell,
    data["gene_names"],
    cell_ids,
    var_data=None,
    out_path="./sparkle_output/raw_cell.h5ad",
    method_name="RAW",
)

save_cell_h5ad(
    corrected,
    data["gene_names"],
    cell_ids,
    var_data=var_data,
    out_path="./sparkle_output/corrected_cell.h5ad",
    method_name="SPARKLE",
)

# ---------------------------------------------------------------------------
# 5. (Optional) add external annotations to the h5ad
# ---------------------------------------------------------------------------
# If you have a CSV/TSV mapping cell_id -> annotation, you can add it before
# writing. For example:
# import pandas as pd
# meta = pd.read_csv("cell_annotations.csv")
# ann_map = dict(zip(meta["cell_id"], meta["annotation"]))
# adata.obs["annotation"] = [str(ann_map.get(int(cid), "Unknown")) for cid in adata.obs["cell_id"]]
```

---

## R: read h5ad into Seurat v4 and re-run SCTransform

The following R script uses `reticulate` to call Python's `scanpy` directly from R, extracts the corrected counts, replaces the `spatial` assay in an existing Seurat v4 object, and re-runs `SCTransform`.

```r
library(reticulate)
library(Seurat)

# ---------------------------------------------------------------------------
# 1. Use Python/scanpy from R to read the corrected h5ad
# ---------------------------------------------------------------------------
sc <- import("scanpy")
np <- import("numpy")

# Path to the SPARKLE-corrected h5ad written by Python
h5ad_path <- "./sparkle_output/corrected_cell.h5ad"
adata <- sc$read_h5ad(h5ad_path)

# Extract the corrected counts matrix (cells × genes in AnnData)
counts <- adata$X
if (!is.matrix(counts)) {
  counts <- as.matrix(counts)   # convert sparse/dense to base R matrix
}

# Extract feature and cell names
gene_names <- adata$var_names$to_list()
cell_names <- adata$obs_names$to_list()

# If X is stored as float32 and you want integer counts, round and cast.
# SPARKLE returns non-negative corrected counts, but they are not integers.
counts <- round(counts)
storage.mode(counts) <- "integer"

rownames(counts) <- gene_names
colnames(counts) <- cell_names

# ---------------------------------------------------------------------------
# 2. Load your existing Seurat v4 RDS object (e.g. built from the raw data)
# ---------------------------------------------------------------------------
seu <- readRDS("path/to/your_seurat_v4_object.rds")

# ---------------------------------------------------------------------------
# 3. Match cells and genes between the h5ad and the Seurat object
# ---------------------------------------------------------------------------
# Only keep cells that exist in both objects
common_cells <- intersect(Cells(seu, assay = "spatial"), colnames(counts))
counts <- counts[, common_cells, drop = FALSE]

# Re-order cells to match the Seurat object ordering
common_cells <- common_cells[match(Cells(seu, assay = "spatial"), common_cells)]
common_cells <- common_cells[!is.na(common_cells)]
counts <- counts[, common_cells, drop = FALSE]

# Keep only genes that exist in the Seurat object (or its spatial assay)
existing_features <- rownames(seu[["spatial"]])
common_features <- intersect(existing_features, rownames(counts))
counts <- counts[common_features, , drop = FALSE]

# ---------------------------------------------------------------------------
# 4. Replace the spatial assay counts with the SPARKLE-corrected matrix
# ---------------------------------------------------------------------------
# Create a new Seurat assay with the corrected counts
assay_spatial_corrected <- CreateAssayObject(counts = counts)

# Add it back to the Seurat object. If you want to overwrite the original
# "spatial" assay, first remove the old one or store it under a new name.
seu[["spatial_sparkle"]] <- assay_spatial_corrected

# Optionally set the new assay as the default
DefaultAssay(seu) <- "spatial_sparkle"

# If your original object has spatial coordinates/image information, you can
# copy the spatial image/coordinates from the original spatial assay.
# (Seurat v4 image handling depends on the platform; adjust as needed.)

# ---------------------------------------------------------------------------
# 5. Re-run SCTransform on the corrected spatial assay
# ---------------------------------------------------------------------------
seu <- SCTransform(
  seu,
  assay = "spatial_sparkle",      # corrected counts
  new.assay.name = "SCT_sparkle",
  vst.flavor = "v2",              # or "v1" for Seurat v4 compatibility
  return.only.var.genes = FALSE,
  verbose = TRUE
)

# ---------------------------------------------------------------------------
# 6. Save the updated Seurat object
# ---------------------------------------------------------------------------
saveRDS(seu, "path/to/your_seurat_v4_sparkle_corrected.rds")
```

### Notes on the R workflow

- **Coordinate system**: The h5ad created by SPARKLE is cell-level; it does not store DNB-level coordinates. If your downstream Seurat analysis needs spatial coordinates, keep them from the original `spatial` assay object or add them to `seu@images` as appropriate for your platform (Stereo-seq, Visium HD, etc.).
- **Count rounding**: SPARKLE returns corrected expression values that are non-negative but continuous. The example rounds them to integers before creating a Seurat assay because many Seurat functions expect integer counts. If you prefer to keep the continuous values, use `CreateAssayObject(counts = counts)` with a `data` slot instead, but note that `SCTransform` expects counts.
- **Feature/Cell matching**: Always match by name before overwriting the Seurat assay to avoid silent reordering errors.

---

## Full Python script (condensed)

```python
import numpy as np
import anndata as ad
from scipy.sparse import csr_matrix
from stambient import SPARKLE
from stambient.io_utils import load_RYTools_data

data = load_RYTools_data(
    gem_path="sample.gem.gz",
    scgem_path="sample.scgem.gz",
    cell_label_col="cell",
)

model = SPARKLE(cell_based=True, bin_size=50, max_radius=200.0)
corrected, diag = model.fit_transform_from_dnb(
    data["spot_expr"], data["spot_coords"], data["spot_labels"]
)
if hasattr(corrected, "toarray"):
    corrected = corrected.toarray()

n_genes = data["spot_expr"].shape[0]
sparkle_r2 = np.full(n_genes, np.nan)
sparkle_selected = np.zeros(n_genes, dtype=bool)
sparkle_corrected = np.zeros(n_genes, dtype=bool)
gi = diag.get("gene_indices")
r2 = diag.get("r2_scores")
if gi is not None and r2 is not None:
    gi = np.asarray(gi, dtype=int)
    r2 = np.asarray(r2)
    sparkle_r2[gi] = r2
    sparkle_selected[gi] = True
    sparkle_corrected[gi] = r2 >= model.r2_threshold

var_data = {
    "sparkle_selected": sparkle_selected,
    "sparkle_corrected": sparkle_corrected,
    "sparkle_r2": sparkle_r2,
}

# raw cell expression
n_cells = len(data["cell_ids"])
idx = np.where(data["spot_labels"] >= 0)[0]
C = csr_matrix(
    (np.ones(len(idx)), (idx, data["spot_labels"][idx])),
    shape=(data["spot_expr"].shape[1], n_cells),
)
raw_cell = (data["spot_expr"] @ C).toarray()

# save corrected h5ad
adata = ad.AnnData(X=corrected.T.astype(np.float32))
adata.var_names = [str(g) for g in data["gene_names"]]
adata.obs_names = [f"Cell_{cid}" for cid in data["cell_ids"]]
adata.obs["cell_id"] = data["cell_ids"]
for k, v in var_data.items():
    adata.var[k] = v
adata.uns["method"] = "SPARKLE"
adata.write_h5ad("corrected_cell.h5ad")
```
