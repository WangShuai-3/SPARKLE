#!/usr/bin/env Rscript
#
# Run RCTD (doublet mode) on MouseBrain cell-level h5ad outputs and compare
# doublet rate and cell-type confidence across correction methods.
#
# Inputs:
#   - evaluation/data/mousebrain/mouseBrain.snRNAseq.308Clusters.seurat.20230607.rds
#   - evaluation/reports/h5ad/mousebrain_x..._{raw,SPARKLE,...}.h5ad
#
# Outputs (written to evaluation/reports/rctd_mousebrain/):
#   - rctd_doublet_results.csv
#   - rctd_summary_metrics.csv
#   - rctd_shared_metrics.csv
#   - rctd_all_methods_shared_metrics.csv
#
# Required R packages:
#   /home/shuaiwang/miniconda3/envs/r-env/bin/Rscript -e "install.packages(c('remotes','data.table','Matrix','Seurat'), repos='https://cloud.r-project.org/')"
#   /home/shuaiwang/miniconda3/envs/r-env/bin/Rscript -e "remotes::install_github('dmcable/spacexr', upgrade='never')"
#
# Run:
#   conda activate r-env
#   RCTD_DATASET_TAG=mousebrain_x12500-20000_y2000-10000 Rscript evaluation/scripts/run_rctd_mousebrain.R

suppressPackageStartupMessages({
  library(Matrix)
  library(data.table)
  library(spacexr)
  library(Seurat)
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

RDS_PATH <- file.path(
  PROJECT_ROOT, "evaluation", "data", "mousebrain",
  "mouseBrain.snRNAseq.308Clusters.seurat.20230607.rds"
)
H5AD_DIR <- file.path(PROJECT_ROOT, "evaluation", "reports", "h5ad")
MTX_DIR <- file.path(PROJECT_ROOT, "evaluation", "reports", "rctd_mousebrain", "mtx")
OUT_DIR <- file.path(PROJECT_ROOT, "evaluation", "reports", "rctd_mousebrain")

DATASET_TAG <- Sys.getenv("RCTD_DATASET_TAG", "mousebrain_x12500-20000_y2000-10000")

method_name_map <- c(
  raw = "RAW",
  sparkle = "SPARKLE",
  soupx = "SoupX",
  decontx = "DecontX",
  spotcleanofficial = "SpotClean"
)

method_name_from_stem <- function(stem) {
  key <- tolower(stem)
  if (key %in% names(method_name_map)) {
    return(unname(method_name_map[[key]]))
  }
  if (!grepl("_", stem)) {
    return(toupper(stem))
  }
  parts <- strsplit(stem, "_")[[1]]
  paste0(toupper(substring(parts, 1, 1)), substring(parts, 2), collapse = "")
}

prefix <- paste0(DATASET_TAG, "_")
H5AD_FILES <- sort(list.files(H5AD_DIR, pattern = paste0("^", prefix, ".*\\.h5ad$")))
if (length(H5AD_FILES) == 0) {
  stop(sprintf("No h5ad files found for tag '%s' in %s", DATASET_TAG, H5AD_DIR))
}

method_stems <- tools::file_path_sans_ext(H5AD_FILES)
method_stems <- ifelse(startsWith(method_stems, prefix),
                       substring(method_stems, nchar(prefix) + 1),
                       method_stems)
MTX_PREFIXES <- tools::file_path_sans_ext(H5AD_FILES)
METHOD_NAMES <- sapply(method_stems, method_name_from_stem)
FINAL_COMPARISON_METHODS <- c("RAW", "SPARKLE", "SoupX", "DecontX", "SpotClean")

# Optionally run only selected methods while preserving the rows already
# present in the aggregate CSV files. This is intended for adding a newly
# completed baseline without rerunning the existing methods.
requested_methods <- trimws(strsplit(Sys.getenv("RCTD_METHODS", ""), ",")[[1]])
requested_methods <- requested_methods[nzchar(requested_methods)]
partial_run <- length(requested_methods) > 0
if (partial_run) {
  keep <- METHOD_NAMES %in% requested_methods
  missing_requested <- setdiff(requested_methods, METHOD_NAMES)
  if (length(missing_requested) > 0) {
    stop(sprintf("Requested RCTD methods not found: %s",
                 paste(missing_requested, collapse = ", ")))
  }
  H5AD_FILES <- H5AD_FILES[keep]
  method_stems <- method_stems[keep]
  MTX_PREFIXES <- MTX_PREFIXES[keep]
  METHOD_NAMES <- METHOD_NAMES[keep]
}

message(sprintf("Using dataset tag '%s'. Discovered %d methods: %s",
                DATASET_TAG, length(METHOD_NAMES),
                paste(METHOD_NAMES, collapse = ", ")))

MAX_CORES <- as.integer(Sys.getenv("RCTD_MAX_CORES", "8"))
REFERENCE_LEVEL <- Sys.getenv("RCTD_REFERENCE_LEVEL", "Cell_subclass")  # or Cell_group
REF_MAX_CELLS_PER_TYPE <- as.integer(Sys.getenv("RCTD_REF_MAX_CELLS_PER_TYPE", "1000"))
REF_MIN_CELLS_PER_TYPE <- as.integer(Sys.getenv("RCTD_REF_MIN_CELLS_PER_TYPE", "25"))
HVG_N <- as.integer(Sys.getenv("RCTD_HVG_N", "5000"))
PYTHON_BIN <- Sys.getenv("RCTD_PYTHON", "python3")

stopifnot(file.exists(RDS_PATH))
dir.create(MTX_DIR, recursive = TRUE, showWarnings = FALSE)
py_script <- file.path(PROJECT_ROOT, "evaluation", "scripts", "convert_h5ad_to_mtx_mousebrain.py")
for (i in seq_along(H5AD_FILES)) {
  prefix_i <- MTX_PREFIXES[i]
  h5ad_file <- file.path(H5AD_DIR, H5AD_FILES[i])
  mtx_file <- file.path(MTX_DIR, paste0(prefix_i, "_counts.mtx"))
  obs_file <- file.path(MTX_DIR, paste0(prefix_i, "_obs.csv"))
  var_file <- file.path(MTX_DIR, paste0(prefix_i, "_var.csv"))
  cache_files <- c(mtx_file, obs_file, var_file)
  cache_complete <- all(file.exists(cache_files))
  cache_fresh <- cache_complete &&
    min(file.info(cache_files)$mtime) >= file.info(h5ad_file)$mtime
  if (cache_fresh) {
    next
  }
  cache_reason <- if (cache_complete) "stale" else "missing"
  message(sprintf(
    "MTX inputs %s for %s. Running Python conversion ...",
    cache_reason, prefix_i
  ))
  cmd <- sprintf("%s '%s' '%s' '%s'", PYTHON_BIN, py_script, h5ad_file, MTX_DIR)
  message(sprintf("  %s", cmd))
  system(cmd)
}
stopifnot(dir.exists(MTX_DIR))
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

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
  cell_types <- as.character(obs$annotation)
  coords <- as.data.frame(obs[, .(x, y)])
  rownames(coords) <- cell_names

  rownames(counts) <- gene_names
  colnames(counts) <- cell_names

  list(
    counts = counts,
    cell_ids = cell_ids,
    cell_types = cell_types,
    gene_names = gene_names,
    cell_names = cell_names,
    coords = coords
  )
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
  w[is.na(w) | is.nan(w)] <- 0
  rs <- rowSums(w)
  w <- w / rs
  w[rs == 0 | is.nan(w)] <- 0
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
  if (is.null(weights) || length(barcodes) == 0) {
    return(NA_real_)
  }
  available <- intersect(barcodes, rownames(weights))
  if (length(available) == 0) {
    return(NA_real_)
  }
  w <- as.matrix(weights[available, , drop = FALSE])
  w[is.na(w) | is.nan(w)] <- 0
  rs <- rowSums(w)
  w <- w / rs
  w[rs == 0 | is.nan(w)] <- 0
  entropy <- -rowSums(w * log2(w + 1e-12), na.rm = TRUE)
  mean(entropy, na.rm = TRUE)
}

# -----------------------------------------------------------------------------
# 1. Build scRNA reference from Seurat RDS
# -----------------------------------------------------------------------------
message("[1/5] Loading MouseBrain snRNA reference ...")
obj <- readRDS(RDS_PATH)

# Extract counts
assay <- DefaultAssay(obj)
if (.hasSlot(obj[[assay]], "layers") && "counts" %in% names(obj[[assay]]@layers)) {
  ref_counts <- obj[[assay]]@layers$counts
} else if (.hasSlot(obj[[assay]], "counts")) {
  ref_counts <- obj[[assay]]@counts
} else if (.hasSlot(obj[[assay]], "data")) {
  ref_counts <- obj[[assay]]@data
} else {
  stop("Cannot find counts slot in Seurat object")
}

# Ensure genes x cells orientation
if (ncol(ref_counts) != ncol(obj)) {
  ref_counts <- t(ref_counts)
}

# Cell type labels
cell_types <- as.character(obj@meta.data[[REFERENCE_LEVEL]])
# RCTD prohibits '/' in cell type names; replace with '-'
cell_types <- gsub("/", "-", cell_types)
names(cell_types) <- colnames(obj)

# Remove the full Seurat object from memory; we only need the extracted counts
rm(obj)
invisible(gc())

# Remove cells with NA labels and empty types
valid_cells <- !is.na(cell_types) & cell_types != ""

# RCTD requires at least REF_MIN_CELLS_PER_TYPE cells per cell type
if (REF_MIN_CELLS_PER_TYPE > 0) {
  ct_counts <- table(cell_types[valid_cells])
  keep_types <- names(ct_counts)[ct_counts >= REF_MIN_CELLS_PER_TYPE]
  removed_types <- names(ct_counts)[ct_counts < REF_MIN_CELLS_PER_TYPE]
  n_removed <- sum(ct_counts[ct_counts < REF_MIN_CELLS_PER_TYPE])
  valid_cells <- valid_cells & cell_types %in% keep_types
  message(sprintf(
    "  Filtered %d %s types with <%d cells (%d cells removed); %d types retained",
    length(removed_types), REFERENCE_LEVEL, REF_MIN_CELLS_PER_TYPE, n_removed, length(keep_types)
  ))
}

ref_counts <- ref_counts[, valid_cells]
cell_types <- cell_types[valid_cells]
nUMI_ref <- setNames(colSums(ref_counts), colnames(ref_counts))

gene_names <- make.names(rownames(ref_counts), unique = TRUE)
rownames(ref_counts) <- gene_names

cell_types <- factor(setNames(cell_types, colnames(ref_counts)))

# Downsample the reference to speed up RCTD while preserving cell-type balance
if (REF_MAX_CELLS_PER_TYPE > 0) {
  set.seed(42)
  ct_table <- table(cell_types)
  keep_idx <- unlist(lapply(names(ct_table), function(ct) {
    idx <- which(cell_types == ct)
    if (length(idx) > REF_MAX_CELLS_PER_TYPE) {
      sample(idx, REF_MAX_CELLS_PER_TYPE)
    } else {
      idx
    }
  }))
  ref_counts <- ref_counts[, keep_idx]
  cell_types <- cell_types[keep_idx]
  nUMI_ref <- nUMI_ref[keep_idx]
  message(sprintf("  Downsampled reference: %d genes x %d cells, %d %s types",
                  nrow(ref_counts), ncol(ref_counts),
                  length(unique(cell_types)), REFERENCE_LEVEL))
}

# Select highly variable genes to further reduce runtime
if (HVG_N > 0 && HVG_N < nrow(ref_counts)) {
  suppressMessages({
    ref_obj <- CreateSeuratObject(counts = ref_counts, min.cells = 0, min.features = 0)
    ref_obj <- FindVariableFeatures(ref_obj, selection.method = "vst", nfeatures = HVG_N, verbose = FALSE)
  })
  hvg_genes <- VariableFeatures(ref_obj)
  if (length(hvg_genes) > 0) {
    ref_counts <- ref_counts[hvg_genes, , drop = FALSE]
    gene_names <- rownames(ref_counts)
    message(sprintf("  Selected %d HVGs for reference", length(hvg_genes)))
  }
  rm(ref_obj)
  invisible(gc())
}

# Recompute nUMI so it matches the final subset reference counts
nUMI_ref <- setNames(colSums(ref_counts), colnames(ref_counts))

reference <- Reference(ref_counts, cell_types, nUMI_ref)
message(sprintf("  Reference: %d genes x %d cells, %d %s types",
                nrow(ref_counts), ncol(ref_counts),
                length(unique(cell_types)), REFERENCE_LEVEL))

# -----------------------------------------------------------------------------
# 2. Run RCTD doublet mode on each h5ad
# -----------------------------------------------------------------------------
message("[2/5] Running RCTD doublet mode on each h5ad ...")
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

  counts@x[counts@x < 0] <- 0
  common_genes <- intersect(rownames(counts), rownames(ref_counts))
  if (length(common_genes) == 0) {
    stop(sprintf("No shared genes between %s and reference", method))
  }
  counts <- counts[common_genes, , drop = FALSE]
  counts <- round(counts)
  message(sprintf("    Shared genes with reference: %d", length(common_genes)))

  nUMI_sp <- setNames(colSums(counts), h5ad$cell_names)

  spatialRNA <- SpatialRNA(
    counts = counts,
    coords = h5ad$coords,
    nUMI = nUMI_sp
  )

  rctd <- create.RCTD(spatialRNA, reference, max_cores = MAX_CORES)
  rctd <- run.RCTD(rctd, doublet_mode = "doublet")

  res_df <- as.data.table(rctd@results$results_df, keep.rownames = "cell_barcode")
  res_df[, method := method]
  rownames(res_df) <- res_df$cell_barcode
  results_list[[method]] <- res_df
  weights_list[[method]] <- rctd@results$weights

  metrics_list[[method]] <- compute_rctd_metrics(res_df, rctd@results$weights, method)

  # Clean up large objects before the next method
  rm(rctd, spatialRNA, counts)
  invisible(gc())
}

# -----------------------------------------------------------------------------
# 3. Compute shared-cell metrics against RAW
# -----------------------------------------------------------------------------
message("[3/5] Computing shared-cell metrics against RAW ...")
shared_metrics_list <- list()

out_prefix <- paste0("rctd_", REFERENCE_LEVEL, "_")
results_path <- file.path(OUT_DIR, paste0(out_prefix, "doublet_results.csv"))
raw_res_all <- results_list[["RAW"]]
raw_weights <- weights_list[["RAW"]]
if (is.null(raw_res_all) && partial_run && file.exists(results_path)) {
  previous_results <- fread(results_path)
  raw_res_all <- previous_results[method == "RAW"]
}

if (!is.null(raw_res_all) && nrow(raw_res_all) > 0) {
  raw_barcodes <- raw_res_all$cell_barcode
  for (method in METHOD_NAMES) {
    method_barcodes <- results_list[[method]]$cell_barcode
    shared_barcodes <- intersect(raw_barcodes, method_barcodes)

    raw_res <- raw_res_all[cell_barcode %in% shared_barcodes]
    method_res <- results_list[[method]][cell_barcode %in% shared_barcodes]
    rownames(raw_res) <- raw_res$cell_barcode
    rownames(method_res) <- method_res$cell_barcode

    raw_entropy <- compute_entropy_from_weights(raw_weights, raw_res$cell_barcode)
    method_entropy <- compute_entropy_from_weights(
      weights_list[[method]], method_res$cell_barcode
    )

    shared_metrics_list[[paste0("RAW_vs_", method)]] <- rbind(
      data.table(
        method = "RAW_shared",
        compared_to = method,
        n_cells = nrow(raw_res),
        pct_doublet = mean(raw_res$spot_class != "singlet") * 100,
        mean_entropy = raw_entropy
      ),
      data.table(
        method = method,
        compared_to = method,
        n_cells = nrow(method_res),
        pct_doublet = mean(method_res$spot_class != "singlet") * 100,
        mean_entropy = method_entropy
      )
    )
  }
} else {
  message("  RAW results unavailable; preserving existing shared-cell metrics.")
}

# -----------------------------------------------------------------------------
# 4. Save results
# -----------------------------------------------------------------------------
message("[4/5] Saving results ...")

merge_incremental <- function(new_data, output_file, key_column = "method",
                              replace_values = METHOD_NAMES) {
  if (partial_run && file.exists(output_file)) {
    old_data <- fread(output_file)
    if (key_column %in% names(old_data)) {
      old_data <- old_data[!get(key_column) %in% replace_values]
    }
    return(rbindlist(list(old_data, new_data), use.names = TRUE, fill = TRUE))
  }
  new_data
}

all_results <- merge_incremental(
  rbindlist(results_list, use.names = TRUE, fill = TRUE), results_path
)
fwrite(all_results, results_path)
message(sprintf("  Per-cell results: %s", results_path))

# Figure 4C uses one fixed denominator: the intersection of cells retained by
# all five final-manuscript methods. This avoids conflating method performance
# with method-specific filtering or failed RCTD fits.
all_methods_shared_path <- file.path(
  OUT_DIR, paste0(out_prefix, "all_methods_shared_metrics.csv")
)
available_methods <- unique(as.character(all_results$method))
if (all(FINAL_COMPARISON_METHODS %in% available_methods)) {
  final_results <- all_results[method %in% FINAL_COMPARISON_METHODS]
  duplicate_rows <- final_results[, .N, by = .(method, cell_barcode)][N > 1]
  if (nrow(duplicate_rows) > 0) {
    stop("Duplicate method/cell_barcode rows prevent a valid shared-cell comparison")
  }

  common_barcodes <- Reduce(
    intersect,
    lapply(
      FINAL_COMPARISON_METHODS,
      function(method_name) final_results[method == method_name, cell_barcode]
    )
  )
  if (length(common_barcodes) == 0) {
    stop("The five final-comparison methods have no shared retained cells")
  }

  all_methods_shared_metrics <- rbindlist(lapply(
    FINAL_COMPARISON_METHODS,
    function(method_name) {
      x <- final_results[
        method == method_name & cell_barcode %in% common_barcodes
      ]
      classes <- as.character(x$spot_class)
      is_singlet <- !is.na(classes) & classes == "singlet"
      is_certain <- !is.na(classes) & classes == "doublet_certain"
      is_uncertain <- !is.na(classes) & classes == "doublet_uncertain"
      is_other <- !(is_singlet | is_certain | is_uncertain)
      data.table(
        method = method_name,
        n_cells = nrow(x),
        n_singlet = sum(is_singlet),
        pct_singlet = mean(is_singlet) * 100,
        n_doublet_certain = sum(is_certain),
        pct_doublet_certain = mean(is_certain) * 100,
        n_doublet_uncertain = sum(is_uncertain),
        pct_doublet_uncertain = mean(is_uncertain) * 100,
        n_reject_or_other = sum(is_other),
        pct_reject_or_other = mean(is_other) * 100,
        mean_singlet_score = mean(x$singlet_score, na.rm = TRUE),
        median_singlet_score = median(x$singlet_score, na.rm = TRUE)
      )
    }
  ))
  fwrite(all_methods_shared_metrics, all_methods_shared_path)
  message(sprintf(
    "  All-method shared-cell metrics (%d cells): %s",
    length(common_barcodes), all_methods_shared_path
  ))
} else {
  missing_methods <- setdiff(FINAL_COMPARISON_METHODS, available_methods)
  warning(sprintf(
    "Not writing all-method shared-cell metrics; missing methods: %s",
    paste(missing_methods, collapse = ", ")
  ))
  all_methods_shared_metrics <- data.table()
}

summary_path <- file.path(OUT_DIR, paste0(out_prefix, "summary_metrics.csv"))
summary_metrics <- merge_incremental(
  rbindlist(metrics_list, use.names = TRUE, fill = TRUE), summary_path
)
fwrite(summary_metrics, summary_path)
message(sprintf("  Summary metrics: %s", summary_path))

shared_path <- file.path(OUT_DIR, paste0(out_prefix, "shared_metrics.csv"))
if (length(shared_metrics_list) > 0) {
  new_shared <- rbindlist(shared_metrics_list, use.names = TRUE, fill = TRUE)
  if (partial_run && file.exists(shared_path)) {
    old_shared <- fread(shared_path)
    old_shared <- old_shared[!compared_to %in% METHOD_NAMES]
    shared_metrics <- rbindlist(
      list(old_shared, new_shared), use.names = TRUE, fill = TRUE
    )
  } else {
    shared_metrics <- new_shared
  }
  fwrite(shared_metrics, shared_path)
} else if (file.exists(shared_path)) {
  shared_metrics <- fread(shared_path)
} else {
  shared_metrics <- data.table()
}
message(sprintf("  Shared-cell metrics: %s", shared_path))

message("\nRCTD doublet-mode comparison (all retained cells):")
print(summary_metrics)

message("\nRCTD doublet-mode comparison (shared cells with RAW):")
print(shared_metrics)

message("\nRCTD doublet-mode comparison (cells shared by all five methods):")
print(all_methods_shared_metrics)
