#!/usr/bin/env Rscript
#
# Run RCTD (doublet mode) on Visium HD cell-level h5ad outputs and compare
# mapping purity/quality across correction methods (RAW, SPARKLE, SpatialSoupX).
#
# Inputs:
#   - evaluation/data/visiumhd/scrna/HumanColonCancer_Flex_Multiplex_count_filtered_feature_bc_matrix.h5
#   - evaluation/data/visiumhd/scrna/SingleCell_MetaData.csv.gz
#   - evaluation/data/visiumhd/Visium_HD_6p5mm_Human_Colon_Cancer_feature_slice.h5
#   - evaluation/reports/h5ad_visiumhd/*.h5ad
#
# Outputs (written to evaluation/reports/rctd_visiumhd/):
#   - rctd_doublet_results.csv       per-cell RCTD predictions per method
#   - rctd_summary_metrics.csv       comparison metrics per method
#   - rctd_shared_metrics.csv        shared-cell comparison against RAW
#
# Required R packages (install first if missing):
#   /home/shuaiwang/miniconda3/envs/r-env/bin/Rscript -e "install.packages(c('remotes','data.table','Matrix','hdf5r'), repos='https://cloud.r-project.org/')"
#   /home/shuaiwang/miniconda3/envs/r-env/bin/Rscript -e "remotes::install_github('dmcable/spacexr', upgrade='never')"
#
# Run:
#   conda activate r-env
#   Rscript evaluation/scripts/run_rctd_visiumhd.R

suppressPackageStartupMessages({
  library(hdf5r)
  library(Matrix)
  library(data.table)
  library(spacexr)
})

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
# Determine project root from script location (works with Rscript)
args <- commandArgs(trailingOnly = FALSE)
script_arg <- args[grep("^--file=", args)]
if (length(script_arg) > 0) {
  script_path <- normalizePath(sub("^--file=", "", script_arg))
  PROJECT_ROOT <- file.path(dirname(script_path), "..", "..")
} else {
  PROJECT_ROOT <- normalizePath(".")
}
if (!dir.exists(file.path(PROJECT_ROOT, "evaluation"))) {
  PROJECT_ROOT <- normalizePath(".")
}

SCRNA_H5 <- file.path(
  PROJECT_ROOT, "evaluation", "data", "visiumhd", "scrna",
  "HumanColonCancer_Flex_Multiplex_count_filtered_feature_bc_matrix.h5"
)
SCRNA_META <- file.path(
  PROJECT_ROOT, "evaluation", "data", "visiumhd", "scrna",
  "SingleCell_MetaData.csv.gz"
)
FEATURE_SLICE_H5 <- file.path(
  PROJECT_ROOT, "evaluation", "data", "visiumhd",
  "Visium_HD_6p5mm_Human_Colon_Cancer_feature_slice.h5"
)
H5AD_DIR <- file.path(PROJECT_ROOT, "evaluation", "reports", "h5ad_visiumhd")
MTX_DIR <- file.path(PROJECT_ROOT, "evaluation", "reports", "rctd_visiumhd", "mtx")
OUT_DIR <- file.path(PROJECT_ROOT, "evaluation", "reports", "rctd_visiumhd")

H5AD_FILES <- c("raw.h5ad", "sparkle.h5ad", "spatial_soupx.h5ad")
MTX_PREFIXES <- c("raw", "sparkle", "spatial_soupx")
METHOD_NAMES <- c("RAW", "SPARKLE", "SpatialSoupX")

MAX_CORES <- as.integer(Sys.getenv("RCTD_MAX_CORES", "16"))

stopifnot(dir.exists(H5AD_DIR))
if (!dir.exists(MTX_DIR)) {
  message("MTX inputs not found. Running Python conversion first...")
  py_script <- file.path(PROJECT_ROOT, "evaluation", "scripts", "convert_h5ad_to_mtx.py")
  for (i in seq_along(H5AD_FILES)) {
    h5ad_file <- file.path(H5AD_DIR, H5AD_FILES[i])
    cmd <- sprintf("python3 '%s' '%s' '%s'", py_script, h5ad_file, MTX_DIR)
    message(sprintf("  %s", cmd))
    system(cmd)
  }
}
stopifnot(dir.exists(MTX_DIR))
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
read_mtx_input <- function(prefix, mtx_dir) {
  # Read a pre-converted MTX + obs/var CSV set into a genes x cells dgCMatrix.
  # Returns a list with counts, cell_ids, cell_types, gene_names, cell_names.
  counts <- readMM(file.path(mtx_dir, paste0(prefix, "_counts.mtx")))
  counts <- as(counts, "CsparseMatrix")

  obs <- fread(file.path(mtx_dir, paste0(prefix, "_obs.csv")))
  var <- fread(file.path(mtx_dir, paste0(prefix, "_var.csv")))

  gene_names <- make.names(var$gene_name, unique = TRUE)
  cell_names <- as.character(obs$cell_name)
  cell_ids <- as.integer(obs$cell_id)
  cell_types <- as.character(obs$cell_type)

  rownames(counts) <- gene_names
  colnames(counts) <- cell_names

  list(
    counts = counts,
    cell_ids = cell_ids,
    cell_types = cell_types,
    gene_names = gene_names,
    cell_names = cell_names
  )
}

read_segmentation_coords <- function(feature_slice_h5) {
  # Compute mean (x, y) coordinate per cell from segmentation mask.
  h5 <- H5File$new(feature_slice_h5, mode = "r")
  on.exit(h5$close_all())

  seg <- h5[["segmentations/cell_segmentation_mask"]]
  rows <- as.integer(seg[["row"]]$read())
  cols <- as.integer(seg[["col"]]$read())
  data <- as.integer(seg[["data"]]$read())

  dt <- data.table(cell_id = data, x = cols * 2L, y = rows * 2L)
  coords <- dt[, .(x = mean(x), y = mean(y)), by = cell_id]
  setkey(coords, cell_id)
  coords
}

compute_rctd_metrics <- function(res, weights, method_name) {
  # Compute RCTD quality/purity metrics from a results_df and weights matrix.
  if (is(res, "RCTD")) {
    weights <- res@results$weights
    res <- res@results$results_df
  }

  n_cells <- nrow(res)
  spot_class <- as.character(res$spot_class)

  pct_singlet <- mean(spot_class == "singlet") * 100
  pct_doublet_certain <- mean(spot_class == "doublet_certain") * 100
  pct_doublet_uncertain <- mean(spot_class == "doublet_uncertain") * 100
  pct_doublet <- pct_doublet_certain + pct_doublet_uncertain

  w <- as.matrix(weights[rownames(res), , drop = FALSE])
  w <- w / rowSums(w)
  sorted_w <- t(apply(w, 1, sort, decreasing = TRUE))
  top1 <- sorted_w[, 1]
  top2 <- sorted_w[, 2]
  entropy <- -rowSums(w * log2(w + 1e-12), na.rm = TRUE)

  singlet_mask <- spot_class == "singlet"
  doublet_mask <- spot_class != "singlet"

  data.table(
    method = method_name,
    n_cells = n_cells,
    pct_singlet = pct_singlet,
    pct_doublet = pct_doublet,
    pct_doublet_certain = pct_doublet_certain,
    pct_doublet_uncertain = pct_doublet_uncertain,
    mean_max_weight = mean(top1, na.rm = TRUE),
    mean_top2_gap = mean(top1 - top2, na.rm = TRUE),
    mean_entropy = mean(entropy, na.rm = TRUE),
    singlet_purity = mean(top1[singlet_mask], na.rm = TRUE),
    doublet_balance = mean(top2[doublet_mask] / top1[doublet_mask], na.rm = TRUE)
  )
}

# -----------------------------------------------------------------------------
# 1. Build scRNA reference (class1 / Level1 annotations)
# -----------------------------------------------------------------------------
message("[1/5] Loading scRNA reference and metadata ...")

read_10x_h5 <- function(h5_path) {
  # Read 10x Genomics HDF5 feature-barcode matrix into a genes x cells dgCMatrix.
  h5 <- H5File$new(h5_path, mode = "r")
  on.exit(h5$close_all())

  mat <- h5[["matrix"]]
  data <- mat[["data"]]$read()
  indices <- mat[["indices"]]$read() + 1L  # 0-based -> 1-based
  indptr <- mat[["indptr"]]$read()
  shape <- mat[["shape"]]$read()
  barcodes <- as.character(mat[["barcodes"]]$read())
  feature_names <- as.character(mat[["features/name"]]$read())

  counts <- sparseMatrix(
    i = indices,
    p = indptr,
    x = as.numeric(data),
    dims = as.integer(shape),
    index1 = TRUE  # i is 1-based; p is always 0-based
  )
  # 10x h5 shape is already [genes x cells]
  # Make gene names unique (required by RCTD)
  feature_names <- make.names(feature_names, unique = TRUE)
  rownames(counts) <- feature_names
  colnames(counts) <- barcodes
  as(counts, "dgCMatrix")
}

ref_counts <- read_10x_h5(SCRNA_H5)  # genes x cells

meta <- as.data.table(read.csv(gzfile(SCRNA_META), stringsAsFactors = FALSE))
meta <- meta[QCFilter == "Keep"]
meta <- meta[Barcode %in% colnames(ref_counts)]
ref_counts <- ref_counts[, meta$Barcode]

cell_types <- factor(setNames(meta$Level1, meta$Barcode))
nUMI_ref <- colSums(ref_counts)

reference <- Reference(ref_counts, cell_types, nUMI_ref)
message(sprintf("  Reference: %d genes x %d cells, %d class1 types",
                nrow(ref_counts), ncol(ref_counts),
                length(unique(cell_types))))

# -----------------------------------------------------------------------------
# 2. Load cell coordinates from segmentation mask
# -----------------------------------------------------------------------------
message("[2/5] Loading cell coordinates from segmentation mask ...")
coords <- read_segmentation_coords(FEATURE_SLICE_H5)
message(sprintf("  Coordinates loaded for %d segmented cells", nrow(coords)))

# -----------------------------------------------------------------------------
# 3. Run RCTD doublet mode on each h5ad
# -----------------------------------------------------------------------------
message("[3/5] Running RCTD doublet mode on each h5ad ...")
results_list <- list()
weights_list <- list()
metrics_list <- list()

for (i in seq_along(H5AD_FILES)) {
  h5ad_file <- file.path(H5AD_DIR, H5AD_FILES[i])
  method <- METHOD_NAMES[i]

  if (!file.exists(h5ad_file)) {
    message(sprintf("  Skipping %s: file not found (%s)", method, h5ad_file))
    next
  }

  message(sprintf("  Processing %s (%s) ...", method, h5ad_file))

  h5ad <- read_mtx_input(MTX_PREFIXES[i], MTX_DIR)
  counts <- h5ad$counts
  cell_ids <- h5ad$cell_ids

  # Map to coordinates
  coords_sub <- coords[J(cell_ids), nomatch = 0L]
  if (nrow(coords_sub) != length(cell_ids)) {
    missing <- length(cell_ids) - nrow(coords_sub)
    message(sprintf("    Warning: %d cells missing coordinates", missing))
    keep <- cell_ids %in% coords_sub$cell_id
    cell_ids <- cell_ids[keep]
    counts <- counts[, keep, drop = FALSE]
    h5ad$cell_names <- h5ad$cell_names[keep]
  }
  coords_sub <- coords_sub[match(cell_ids, coords_sub$cell_id)]
  rownames(coords_sub) <- h5ad$cell_names

  # Ensure non-negative and intersect genes with reference
  counts@x[counts@x < 0] <- 0
  common_genes <- intersect(rownames(counts), rownames(ref_counts))
  if (length(common_genes) == 0) {
    stop(sprintf("No shared genes between %s and reference", method))
  }
  counts <- counts[common_genes, , drop = FALSE]
  colnames(counts) <- h5ad$cell_names
  # RCTD expects whole-number counts; round corrected float values
  counts <- round(counts)
  message(sprintf("    Shared genes with reference: %d", length(common_genes)))

  nUMI_sp <- setNames(colSums(counts), h5ad$cell_names)

  coords_df <- as.data.frame(coords_sub[, .(x, y)])
  rownames(coords_df) <- h5ad$cell_names

  spatialRNA <- SpatialRNA(
    counts = counts,
    coords = coords_df,
    nUMI = nUMI_sp
  )

  rctd <- create.RCTD(spatialRNA, reference, max_cores = MAX_CORES)
  rctd <- run.RCTD(rctd, doublet_mode = "doublet")

  res_df <- as.data.table(rctd@results$results_df, keep.rownames = "cell_barcode")
  res_df[, method := method]
  # Keep barcode as row name for matching with weights matrix
  rownames(res_df) <- res_df$cell_barcode
  results_list[[method]] <- res_df
  weights_list[[method]] <- rctd@results$weights

  metrics_list[[method]] <- compute_rctd_metrics(res_df, rctd@results$weights, method)
}

# -----------------------------------------------------------------------------
# 4. Compute shared-cell metrics against RAW
# -----------------------------------------------------------------------------
message("[4/5] Computing shared-cell metrics against RAW ...")
raw_barcodes <- results_list[["RAW"]]$cell_barcode
shared_metrics_list <- list()

for (method in METHOD_NAMES) {
  method_barcodes <- results_list[[method]]$cell_barcode
  shared_barcodes <- intersect(raw_barcodes, method_barcodes)

  raw_res <- results_list[["RAW"]][cell_barcode %in% shared_barcodes]
  method_res <- results_list[[method]][cell_barcode %in% shared_barcodes]
  rownames(raw_res) <- raw_res$cell_barcode
  rownames(method_res) <- method_res$cell_barcode

  raw_weights <- weights_list[["RAW"]]
  method_weights <- weights_list[[method]]

  shared_metrics_list[[paste0("RAW_vs_", method)]] <- rbind(
    compute_rctd_metrics(raw_res, raw_weights, "RAW_shared")[, compared_to := method],
    compute_rctd_metrics(method_res, method_weights, method)[, compared_to := method]
  )
}

# -----------------------------------------------------------------------------
# 5. Save combined results and comparison tables
# -----------------------------------------------------------------------------
message("[5/5] Saving results ...")

all_results <- rbindlist(results_list, use.names = TRUE, fill = TRUE)
fwrite(all_results, file.path(OUT_DIR, "rctd_doublet_results.csv"))
message(sprintf("  Per-cell results: %s", file.path(OUT_DIR, "rctd_doublet_results.csv")))

summary_metrics <- rbindlist(metrics_list, use.names = TRUE, fill = TRUE)
fwrite(summary_metrics, file.path(OUT_DIR, "rctd_summary_metrics.csv"))
message(sprintf("  Summary metrics: %s", file.path(OUT_DIR, "rctd_summary_metrics.csv")))

shared_metrics <- rbindlist(shared_metrics_list, use.names = TRUE, fill = TRUE)
fwrite(shared_metrics, file.path(OUT_DIR, "rctd_shared_metrics.csv"))
message(sprintf("  Shared-cell metrics: %s", file.path(OUT_DIR, "rctd_shared_metrics.csv")))

message("\nRCTD doublet-mode comparison (all retained cells):")
print(summary_metrics)

message("\nRCTD doublet-mode comparison (shared cells with RAW):")
print(shared_metrics)

message("\nDone.")
