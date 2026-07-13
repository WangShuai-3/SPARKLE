#!/usr/bin/env Rscript
#
# Run RCTD doublet-mode evaluation for one CRC segmentation condition.
#
# ``final_comparison.py`` writes Proseg and StarDist results under distinct tags
# but with identical raw expression and spatial windows.  Run this script once
# per condition (or launch the two commands concurrently) to obtain directly
# comparable RCTD quality metrics and per-cell ``first_type`` annotations.
#
# Required inputs:
#   evaluation/reports/h5ad/{DATASET_TAG}_{raw,SPARKLE,...}.h5ad
#   evaluation/data/CRC/scrna_reference/prepared_cluster_midway/*
#
# The compact reference is created automatically from the 3.75-GB Pelka h5ad if
# absent.  Cell coordinates come from h5ad obs x/y, where final_comparison.py
# stores centroids computed from the exact cropped segmentation labels.
#
# Example (chosen 600 x 600 um evaluation window):
#   CRC_SEGMENTATION=proseg \
#   RCTD_DATASET_TAG=crc_proseg_x14300-14900_y2850-3450 \
#   RCTD_MAX_CORES=16 Rscript evaluation/scripts/run_rctd_crc.R

suppressPackageStartupMessages({
  library(Matrix)
  library(data.table)
  library(spacexr)
})

# -----------------------------------------------------------------------------
# Configuration and file discovery
# -----------------------------------------------------------------------------
args <- commandArgs(trailingOnly = FALSE)
script_arg <- args[grep("^--file=", args)]
if (length(script_arg) > 0) {
  script_path <- normalizePath(sub("^--file=", "", script_arg))
  PROJECT_ROOT <- normalizePath(file.path(dirname(script_path), "..", ".."))
} else {
  PROJECT_ROOT <- normalizePath(".")
}

SEGMENTATION <- tolower(Sys.getenv("CRC_SEGMENTATION", "proseg"))
if (!SEGMENTATION %in% c("proseg", "stardist")) {
  stop("CRC_SEGMENTATION must be 'proseg' or 'stardist'")
}
DEFAULT_TAG <- sprintf("crc_%s_x14300-14900_y2850-3450", SEGMENTATION)
DATASET_TAG <- Sys.getenv("RCTD_DATASET_TAG", DEFAULT_TAG)
MAX_CORES <- as.integer(Sys.getenv("RCTD_MAX_CORES", "16"))

H5AD_DIR <- file.path(PROJECT_ROOT, "evaluation", "reports", "h5ad")
OUT_DIR <- file.path(PROJECT_ROOT, "evaluation", "reports", "rctd_crc", SEGMENTATION)
MTX_DIR <- file.path(OUT_DIR, "mtx")
FIRST_TYPE_DIR <- file.path(OUT_DIR, "first_type")
REF_DIR <- file.path(
  PROJECT_ROOT, "evaluation", "data", "CRC", "scrna_reference",
  "prepared_cluster_midway"
)
REF_COUNTS <- file.path(REF_DIR, "rctd_counts.mtx")
REF_GENES <- file.path(REF_DIR, "rctd_genes.csv")
REF_CELLS <- file.path(REF_DIR, "rctd_cells.csv")

for (path in c(OUT_DIR, MTX_DIR, FIRST_TYPE_DIR, REF_DIR)) {
  dir.create(path, recursive = TRUE, showWarnings = FALSE)
}

# Build the compact reference on demand.  Sampling is deterministic and shared
# across both segmentation conditions.
if (!all(file.exists(c(REF_COUNTS, REF_GENES, REF_CELLS)))) {
  prep_script <- file.path(
    PROJECT_ROOT, "evaluation", "scripts", "prepare_crc_scrna_reference.py"
  )
  message("Compact CRC reference is missing; preparing it now ...")
  status <- system2("python3", prep_script)
  if (status != 0) stop("prepare_crc_scrna_reference.py failed")
}

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
  if (stem %in% names(method_name_map)) return(method_name_map[[stem]])
  stem
}

prefix <- paste0(DATASET_TAG, "_")
h5ad_files <- sort(list.files(
  H5AD_DIR, pattern = paste0("^", prefix, ".*\\.h5ad$")
))
if (length(h5ad_files) == 0) {
  stop(sprintf("No h5ad files found for tag '%s' in %s", DATASET_TAG, H5AD_DIR))
}
method_stems <- tools::file_path_sans_ext(h5ad_files)
method_stems <- ifelse(
  startsWith(method_stems, prefix),
  substring(method_stems, nchar(prefix) + 1),
  method_stems
)
MTX_PREFIXES <- tools::file_path_sans_ext(h5ad_files)
METHOD_NAMES <- sapply(method_stems, method_name_from_stem)
message(sprintf(
  "CRC/%s tag '%s': discovered %d methods: %s",
  SEGMENTATION, DATASET_TAG, length(METHOD_NAMES),
  paste(METHOD_NAMES, collapse = ", ")
))

# Convert h5ad to sparse Matrix Market.  The MouseBrain converter is used
# because it preserves obs cell_id and physical x/y columns required by RCTD.
converter <- file.path(
  PROJECT_ROOT, "evaluation", "scripts", "convert_h5ad_to_mtx_mousebrain.py"
)
for (i in seq_along(h5ad_files)) {
  mtx_file <- file.path(MTX_DIR, paste0(MTX_PREFIXES[i], "_counts.mtx"))
  if (file.exists(mtx_file)) next
  h5ad_file <- file.path(H5AD_DIR, h5ad_files[i])
  message(sprintf("Converting %s for RCTD ...", h5ad_files[i]))
  status <- system2("python3", c(converter, h5ad_file, MTX_DIR))
  if (status != 0) stop(sprintf("h5ad conversion failed: %s", h5ad_file))
}

# -----------------------------------------------------------------------------
# Input helpers and metrics
# -----------------------------------------------------------------------------
read_mtx_input <- function(prefix_name) {
  counts <- as(readMM(file.path(MTX_DIR, paste0(prefix_name, "_counts.mtx"))),
               "CsparseMatrix")
  obs <- fread(file.path(MTX_DIR, paste0(prefix_name, "_obs.csv")))
  var <- fread(file.path(MTX_DIR, paste0(prefix_name, "_var.csv")))
  required_obs <- c("cell_id", "cell_name", "x", "y")
  missing_obs <- setdiff(required_obs, names(obs))
  if (length(missing_obs) > 0) {
    stop(sprintf("MTX obs is missing columns: %s", paste(missing_obs, collapse = ", ")))
  }
  rownames(counts) <- make.names(var$gene_name, unique = TRUE)
  colnames(counts) <- as.character(obs$cell_name)
  list(counts = counts, obs = obs)
}

compute_rctd_metrics <- function(results_df, weights, method_name) {
  spot_class <- as.character(results_df$spot_class)
  w <- as.matrix(weights[rownames(results_df), , drop = FALSE])
  # Numerical optimization can occasionally return tiny negative weights.
  # They are not valid proportions and would make log2() emit NaNs, so clip
  # them exactly as expression counts are clipped before RCTD and renormalize.
  w[!is.finite(w) | w < 0] <- 0
  w <- w / pmax(rowSums(w), 1e-12)
  sorted_w <- t(apply(w, 1, sort, decreasing = TRUE))
  top1 <- sorted_w[, 1]
  top2 <- sorted_w[, 2]
  entropy <- -rowSums(w * log2(w + 1e-12), na.rm = TRUE)
  singlet <- spot_class == "singlet"
  doublet <- !singlet

  data.table(
    segmentation = SEGMENTATION,
    method = method_name,
    n_cells = nrow(results_df),
    pct_singlet = mean(singlet) * 100,
    pct_doublet = mean(doublet) * 100,
    pct_doublet_certain = mean(spot_class == "doublet_certain") * 100,
    pct_doublet_uncertain = mean(spot_class == "doublet_uncertain") * 100,
    mean_max_weight = mean(top1, na.rm = TRUE),
    mean_top2_gap = mean(top1 - top2, na.rm = TRUE),
    mean_entropy = mean(entropy, na.rm = TRUE),
    singlet_purity = if (any(singlet)) mean(top1[singlet], na.rm = TRUE) else NA_real_,
    doublet_balance = if (any(doublet)) {
      mean(top2[doublet] / top1[doublet], na.rm = TRUE)
    } else NA_real_
  )
}

mean_weight_entropy <- function(weights, barcodes) {
  available <- intersect(barcodes, rownames(weights))
  if (length(available) == 0) return(NA_real_)
  w <- as.matrix(weights[available, , drop = FALSE])
  w[!is.finite(w) | w < 0] <- 0
  w <- w / pmax(rowSums(w), 1e-12)
  mean(-rowSums(w * log2(w + 1e-12), na.rm = TRUE), na.rm = TRUE)
}

# -----------------------------------------------------------------------------
# Build the compact RCTD Reference object
# -----------------------------------------------------------------------------
message("[1/4] Loading compact Pelka CRC reference ...")
ref_counts <- as(readMM(REF_COUNTS), "CsparseMatrix")
ref_genes <- fread(REF_GENES)
ref_cells <- fread(REF_CELLS)
if (nrow(ref_counts) != nrow(ref_genes) || ncol(ref_counts) != nrow(ref_cells)) {
  stop("Prepared CRC reference matrix/metadata dimensions do not agree")
}
rownames(ref_counts) <- make.names(ref_genes$gene_name, unique = TRUE)
colnames(ref_counts) <- as.character(ref_cells$cell_name)
cell_types <- factor(setNames(
  gsub("/", "-", as.character(ref_cells$cell_type)),
  as.character(ref_cells$cell_name)
))
nUMI_ref <- colSums(ref_counts)
reference <- Reference(ref_counts, cell_types, nUMI_ref)
message(sprintf(
  "  Reference: %d genes x %d cells, %d cell types",
  nrow(ref_counts), ncol(ref_counts), length(unique(cell_types))
))

# -----------------------------------------------------------------------------
# Run RCTD independently for each correction method
# -----------------------------------------------------------------------------
message("[2/4] Running RCTD doublet mode ...")
results_list <- list()
weights_list <- list()
metrics_list <- list()

for (i in seq_along(h5ad_files)) {
  method <- METHOD_NAMES[i]
  message(sprintf("  Processing %s ...", method))
  spatial <- read_mtx_input(MTX_PREFIXES[i])
  counts <- spatial$counts
  obs <- spatial$obs

  # Corrected methods can contain small floating-point negatives.  RCTD expects
  # non-negative integer-like UMI counts, matching the Ovarian evaluation.
  counts@x[counts@x < 0] <- 0
  counts <- round(counts)
  common_genes <- intersect(rownames(counts), rownames(ref_counts))
  if (length(common_genes) == 0) {
    stop(sprintf("No shared genes between %s and CRC reference", method))
  }
  counts <- counts[common_genes, , drop = FALSE]
  cell_names <- as.character(obs$cell_name)
  colnames(counts) <- cell_names
  coords <- as.data.frame(obs[, .(x, y)])
  rownames(coords) <- cell_names
  nUMI_spatial <- setNames(colSums(counts), cell_names)
  message(sprintf("    %d shared genes x %d cells", length(common_genes), ncol(counts)))

  spatial_rna <- SpatialRNA(counts = counts, coords = coords, nUMI = nUMI_spatial)
  rctd <- create.RCTD(spatial_rna, reference, max_cores = MAX_CORES)
  rctd <- run.RCTD(rctd, doublet_mode = "doublet")

  result <- as.data.table(rctd@results$results_df, keep.rownames = "cell_barcode")
  result[, `:=`(method = method, segmentation = SEGMENTATION)]
  rownames(result) <- result$cell_barcode
  results_list[[method]] <- result
  weights_list[[method]] <- rctd@results$weights
  metrics_list[[method]] <- compute_rctd_metrics(
    result, rctd@results$weights, method
  )

  first_type_lookup <- setNames(as.character(result$first_type), result$cell_barcode)
  first_type <- data.table(
    cell_name = cell_names,
    cell_id = as.character(obs$cell_id),
    first_type = unname(first_type_lookup[cell_names]),
    segmentation = SEGMENTATION
  )
  fwrite(first_type, file.path(FIRST_TYPE_DIR, paste0(method, "_first_type.csv")))
}

# -----------------------------------------------------------------------------
# Compare every method on the cells retained in both it and RAW
# -----------------------------------------------------------------------------
message("[3/4] Computing shared-cell metrics against RAW ...")
if (is.null(results_list[["RAW"]])) stop("RCTD RAW result is required")
raw_barcodes <- results_list[["RAW"]]$cell_barcode
shared_metrics_list <- list()

for (method in names(results_list)) {
  shared <- intersect(raw_barcodes, results_list[[method]]$cell_barcode)
  raw_result <- results_list[["RAW"]][cell_barcode %in% shared]
  method_result <- results_list[[method]][cell_barcode %in% shared]
  shared_metrics_list[[method]] <- rbind(
    data.table(
      segmentation = SEGMENTATION, method = "RAW_shared", compared_to = method,
      n_cells = nrow(raw_result),
      pct_doublet = mean(raw_result$spot_class != "singlet") * 100,
      mean_entropy = mean_weight_entropy(weights_list[["RAW"]], raw_result$cell_barcode)
    ),
    data.table(
      segmentation = SEGMENTATION, method = method, compared_to = method,
      n_cells = nrow(method_result),
      pct_doublet = mean(method_result$spot_class != "singlet") * 100,
      mean_entropy = mean_weight_entropy(weights_list[[method]], method_result$cell_barcode)
    )
  )
}

# -----------------------------------------------------------------------------
# Persist results using the same schema as the Ovarian workflow
# -----------------------------------------------------------------------------
message("[4/4] Saving RCTD results ...")
all_results <- rbindlist(results_list, use.names = TRUE, fill = TRUE)
summary_metrics <- rbindlist(metrics_list, use.names = TRUE, fill = TRUE)
shared_metrics <- rbindlist(shared_metrics_list, use.names = TRUE, fill = TRUE)
fwrite(all_results, file.path(OUT_DIR, "rctd_doublet_results.csv"))
fwrite(summary_metrics, file.path(OUT_DIR, "rctd_summary_metrics.csv"))
fwrite(shared_metrics, file.path(OUT_DIR, "rctd_shared_metrics.csv"))

message("\nRCTD summary:")
print(summary_metrics)
message("\nDone: ", OUT_DIR)
