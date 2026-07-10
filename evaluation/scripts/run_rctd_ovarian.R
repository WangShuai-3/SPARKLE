#!/usr/bin/env Rscript
#
# Run RCTD (doublet mode) on Ovarian Visium HD cell-level h5ad outputs and
# compare mapping quality across correction methods (RAW, SPARKLE, SpatialSoupX,
# SoupX, DecontX).
#
# Mirrors ``run_rctd_visiumhd.R`` (colon cancer 6p5mm) but uses the Ovarian
# scFFPE single-cell reference and its FLEX annotation.
#
# Inputs:
#   - evaluation/data/ovarian/17k_Ovarian_Cancer_scFFPE_count_filtered_feature_bc_matrix.h5
#   - evaluation/data/ovarian/FLEX_Ovarian_Barcode_Cluster_Annotation.csv
#   - evaluation/data/ovarian/Visium_HD_Human_Ovarian_Cancer_FF_feature_slice.h5
#   - evaluation/reports/h5ad/ovarian_x1000-1800_y300-1100_{RAW,SPARKLE,...}.h5ad
#
# Outputs (written to evaluation/reports/rctd_ovarian/):
#   - rctd_doublet_results.csv       per-cell RCTD predictions per method
#   - rctd_summary_metrics.csv       comparison metrics per method
#   - rctd_shared_metrics.csv        shared-cell comparison against RAW
#   - {Method}_first_type.csv        per-cell first_type annotated back to cell_id
#
# Run:
#   conda activate r-env
#   Rscript evaluation/scripts/run_rctd_ovarian.R

suppressPackageStartupMessages({
  library(hdf5r)
  library(Matrix)
  library(data.table)
  library(spacexr)
})

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
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

SCRN_H5 <- file.path(
  PROJECT_ROOT, "evaluation", "data", "ovarian",
  "17k_Ovarian_Cancer_scFFPE_count_filtered_feature_bc_matrix.h5"
)
SCRN_META <- file.path(
  PROJECT_ROOT, "evaluation", "data", "ovarian",
  "FLEX_Ovarian_Barcode_Cluster_Annotation.csv"
)
FEATURE_SLICE_H5 <- file.path(
  PROJECT_ROOT, "evaluation", "data", "ovarian",
  "Visium_HD_Human_Ovarian_Cancer_FF_feature_slice.h5"
)
H5AD_DIR <- file.path(PROJECT_ROOT, "evaluation", "reports", "h5ad")
MTX_DIR <- file.path(PROJECT_ROOT, "evaluation", "reports", "rctd_ovarian", "mtx")
OUT_DIR <- file.path(PROJECT_ROOT, "evaluation", "reports", "rctd_ovarian")
FIRST_TYPE_DIR <- file.path(PROJECT_ROOT, "evaluation", "reports", "rctd_ovarian", "first_type")

DATASET_TAG <- Sys.getenv("RCTD_DATASET_TAG", "ovarian_x1000-1800_y300-1100")

method_name_map <- list(
  raw = "RAW",
  sparkle = "SPARKLE",
  SPARKLE = "SPARKLE",
  spatial_soupx = "SpatialSoupX",
  SpatialSoupX = "SpatialSoupX",
  soupx = "SoupX",
  SoupX = "SoupX",
  decontx = "DecontX",
  DecontX = "DecontX"
)

method_name_from_stem <- function(stem) {
  if (stem %in% names(method_name_map)) {
    return(method_name_map[[stem]])
  }
  # Fallback: keep the stem as-is (already a readable method name).
  stem
}

# Discover h5ad files
prefix <- paste0(DATASET_TAG, "_")
h5ad_files <- sort(list.files(H5AD_DIR, pattern = paste0("^", prefix, ".*\\.h5ad$")))
if (length(h5ad_files) == 0) {
  stop(sprintf("No h5ad files found for tag '%s' in %s", DATASET_TAG, H5AD_DIR))
}
method_stems <- tools::file_path_sans_ext(h5ad_files)
method_stems <- ifelse(startsWith(method_stems, prefix),
                       substring(method_stems, nchar(prefix) + 1),
                       method_stems)
MTX_PREFIXES <- tools::file_path_sans_ext(h5ad_files)
METHOD_NAMES <- sapply(method_stems, method_name_from_stem)

message(sprintf("Using dataset tag '%s'. Discovered %d methods: %s",
                DATASET_TAG, length(METHOD_NAMES),
                paste(METHOD_NAMES, collapse = ", ")))

MAX_CORES <- as.integer(Sys.getenv("RCTD_MAX_CORES", "16"))
REF_MAX_CELLS_PER_TYPE <- as.integer(Sys.getenv("RCTD_REF_MAX_CELLS_PER_TYPE", "1000"))
REF_MIN_CELLS_PER_TYPE <- as.integer(Sys.getenv("RCTD_REF_MIN_CELLS_PER_TYPE", "25"))

for (p in c(H5AD_DIR, MTX_DIR, OUT_DIR, FIRST_TYPE_DIR)) {
  dir.create(p, recursive = TRUE, showWarnings = FALSE)
}

# Convert h5ad -> MTX if needed
py_script <- file.path(PROJECT_ROOT, "evaluation", "scripts",
                       "convert_h5ad_to_mtx_mousebrain.py")
if (!file.exists(py_script)) {
  py_script <- file.path(PROJECT_ROOT, "evaluation", "scripts", "convert_h5ad_to_mtx.py")
}
for (i in seq_along(h5ad_files)) {
  prefix_i <- MTX_PREFIXES[i]
  mtx_file <- file.path(MTX_DIR, paste0(prefix_i, "_counts.mtx"))
  if (file.exists(mtx_file)) next
  message(sprintf("MTX inputs missing for %s. Running Python conversion ...", prefix_i))
  h5ad_file <- file.path(H5AD_DIR, h5ad_files[i])
  cmd <- sprintf("python3 '%s' '%s' '%s'", py_script, h5ad_file, MTX_DIR)
  message(sprintf("  %s", cmd))
  system(cmd)
}
stopifnot(dir.exists(MTX_DIR))

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
read_mtx_input <- function(prefix, mtx_dir) {
  counts <- readMM(file.path(mtx_dir, paste0(prefix, "_counts.mtx")))
  counts <- as(counts, "CsparseMatrix")

  obs <- fread(file.path(mtx_dir, paste0(prefix, "_obs.csv")))
  var <- fread(file.path(mtx_dir, paste0(prefix, "_var.csv")))

  gene_names <- make.names(var$gene_name, unique = TRUE)
  cell_names <- as.character(obs$cell_name)
  cell_ids <- as.integer(obs$cell_id)
  # For ovarian h5ad, cell_type is "Unknown" (from convert_h5ad_to_mtx.py) or
  # "annotation" (from convert_h5ad_to_mtx_mousebrain.py).  We only use cell_id
  # here; cell_types are filled by RCTD.

  rownames(counts) <- gene_names
  colnames(counts) <- cell_names

  list(
    counts = counts,
    cell_ids = cell_ids,
    gene_names = gene_names,
    cell_names = cell_names
  )
}

read_segmentation_coords <- function(feature_slice_h5) {
  h5 <- H5File$new(feature_slice_h5, mode = "r")
  on.exit(h5$close_all())

  seg <- h5[["segmentations/cell_segmentation_mask"]]
  rows <- as.integer(seg[["row"]]$read())
  cols <- as.integer(seg[["col"]]$read())
  data <- as.integer(seg[["data"]]$read())

  # x = col * 2, y = row * 2 (2 um pixel pitch)
  dt <- data.table(cell_id = data, x = cols * 2L, y = rows * 2L)
  coords <- dt[, .(x = mean(x), y = mean(y)), by = cell_id]
  setkey(coords, cell_id)
  coords
}

compute_rctd_metrics <- function(res, weights, method_name) {
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

compute_entropy_from_weights <- function(weights, barcodes) {
  if (is.null(weights) || length(barcodes) == 0) return(NA_real_)
  available <- intersect(barcodes, rownames(weights))
  if (length(available) == 0) return(NA_real_)
  w <- as.matrix(weights[available, , drop = FALSE])
  w <- w / rowSums(w)
  entropy <- -rowSums(w * log2(w + 1e-12), na.rm = TRUE)
  mean(entropy, na.rm = TRUE)
}

# -----------------------------------------------------------------------------
# 1. Build scRNA reference
# -----------------------------------------------------------------------------
message("[1/5] Loading scRNA reference and annotation ...")

read_10x_h5 <- function(h5_path) {
  h5 <- H5File$new(h5_path, mode = "r")
  on.exit(h5$close_all())
  mat <- h5[["matrix"]]
  data <- mat[["data"]]$read()
  indices <- mat[["indices"]]$read() + 1L
  indptr <- mat[["indptr"]]$read()
  shape <- mat[["shape"]]$read()
  barcodes <- as.character(mat[["barcodes"]]$read())
  feature_names <- as.character(mat[["features/name"]]$read())
  counts <- sparseMatrix(
    i = indices, p = indptr, x = as.numeric(data),
    dims = as.integer(shape), index1 = TRUE
  )
  feature_names <- make.names(feature_names, unique = TRUE)
  rownames(counts) <- feature_names
  colnames(counts) <- barcodes
  as(counts, "dgCMatrix")
}

ref_counts <- read_10x_h5(SCRN_H5)  # genes x cells

# FLEX annotation: keep only barcodes with an annotation
meta <- fread(SCRN_META, check.names = FALSE)
if ("Cell Annotation" %in% names(meta)) {
  meta <- meta[, .(Barcode, `Cell Annotation`)]
  setnames(meta, "Cell Annotation", "cell_type")
} else if ("Cell.Annotation" %in% names(meta)) {
  meta <- meta[, .(Barcode, Cell.Annotation)]
  setnames(meta, "Cell.Annotation", "cell_type")
}
meta <- meta[Barcode %in% colnames(ref_counts)]
meta <- meta[!is.na(cell_type) & cell_type != ""]

# RCTD prohibits '/' in cell type names
meta[, cell_type := gsub("/", "-", cell_type)]

ref_counts <- ref_counts[, meta$Barcode]
cell_types <- factor(setNames(meta$cell_type, meta$Barcode))

# Filter types with too few cells
ct_table <- table(cell_types)
keep_types <- names(ct_table)[ct_table >= REF_MIN_CELLS_PER_TYPE]
cell_types <- cell_types[cell_types %in% keep_types]
ref_counts <- ref_counts[, names(cell_types)]

# Downsample
if (REF_MAX_CELLS_PER_TYPE > 0) {
  set.seed(42)
  keep_idx <- unlist(lapply(keep_types, function(ct) {
    idx <- which(cell_types == ct)
    if (length(idx) > REF_MAX_CELLS_PER_TYPE) sample(idx, REF_MAX_CELLS_PER_TYPE) else idx
  }))
  ref_counts <- ref_counts[, keep_idx]
  cell_types <- cell_types[keep_idx]
}

nUMI_ref <- colSums(ref_counts)
reference <- Reference(ref_counts, cell_types, nUMI_ref)
message(sprintf("  Reference: %d genes x %d cells, %d cell types",
                nrow(ref_counts), ncol(ref_counts), length(unique(cell_types))))

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
first_type_list <- list()

for (i in seq_along(h5ad_files)) {
  h5ad_file <- file.path(H5AD_DIR, h5ad_files[i])
  method <- METHOD_NAMES[i]

  if (!file.exists(h5ad_file)) {
    message(sprintf("  Skipping %s: file not found (%s)", method, h5ad_file))
    next
  }

  message(sprintf("  Processing %s ...", method))

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

  counts@x[counts@x < 0] <- 0
  common_genes <- intersect(rownames(counts), rownames(ref_counts))
  if (length(common_genes) == 0) {
    stop(sprintf("No shared genes between %s and reference", method))
  }
  counts <- counts[common_genes, , drop = FALSE]
  colnames(counts) <- h5ad$cell_names
  counts <- round(counts)
  message(sprintf("    Shared genes: %d / %d cells", length(common_genes), ncol(counts)))

  nUMI_sp <- setNames(colSums(counts), h5ad$cell_names)
  coords_df <- as.data.frame(coords_sub[, .(x, y)])
  rownames(coords_df) <- h5ad$cell_names

  spatialRNA <- SpatialRNA(counts = counts, coords = coords_df, nUMI = nUMI_sp)
  rctd <- create.RCTD(spatialRNA, reference, max_cores = MAX_CORES)
  rctd <- run.RCTD(rctd, doublet_mode = "doublet")

  res_df <- as.data.table(rctd@results$results_df, keep.rownames = "cell_barcode")
  res_df[, method := method]
  rownames(res_df) <- res_df$cell_barcode
  results_list[[method]] <- res_df
  weights_list[[method]] <- rctd@results$weights

  metrics_list[[method]] <- compute_rctd_metrics(res_df, rctd@results$weights, method)

  # Export first_type mapped back to cell_id
  ft_lookup <- setNames(as.character(res_df$first_type), res_df$cell_barcode)
  ft <- data.table(
    cell_name = h5ad$cell_names,
    cell_id = cell_ids,
    first_type = unname(ft_lookup[h5ad$cell_names])
  )
  first_type_list[[method]] <- ft
  fwrite(ft, file.path(FIRST_TYPE_DIR, paste0(method, "_first_type.csv")))
  message(sprintf("    Exported first_type to %s", paste0(method, "_first_type.csv")))
}

# -----------------------------------------------------------------------------
# 4. Shared-cell metrics
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

  raw_entropy <- compute_entropy_from_weights(weights_list[["RAW"]], raw_res$cell_barcode)
  method_entropy <- compute_entropy_from_weights(weights_list[[method]], method_res$cell_barcode)

  shared_metrics_list[[paste0("RAW_vs_", method)]] <- rbind(
    data.table(method = "RAW_shared", compared_to = method, n_cells = nrow(raw_res),
               pct_doublet = mean(raw_res$spot_class != "singlet") * 100,
               mean_entropy = raw_entropy),
    data.table(method = method, compared_to = method, n_cells = nrow(method_res),
               pct_doublet = mean(method_res$spot_class != "singlet") * 100,
               mean_entropy = method_entropy)
  )
}

# -----------------------------------------------------------------------------
# 5. Save
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
