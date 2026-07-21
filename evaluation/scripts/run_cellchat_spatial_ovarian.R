#!/usr/bin/env Rscript
###############################################################################
# run_cellchat_spatial_ovarian.R
# Spatial CellChat v2 analysis on ovarian Visium HD data (RAW + 4 correction
# methods), comparing whether cancer cell communication changes after ambient
# RNA correction. Key difference from prior non-spatial liana run:
#   computeCommunProb(distance.use=TRUE) — only spatially proximal cell types
#   communicate.
#
# Usage:
#   /home/shuaiwang/miniconda3/envs/r-env/bin/Rscript \
#     evaluation/scripts/run_cellchat_spatial_ovarian.R \
#     > evaluation/logs/cellchat_spatial_ovarian.log 2>&1
# Preflight only (library/finite-value audit, no CellChat inference):
#   /home/shuaiwang/miniconda3/envs/r-env/bin/Rscript \
#     evaluation/scripts/run_cellchat_spatial_ovarian.R --preflight-only
###############################################################################

suppressPackageStartupMessages({
  library(Matrix)
  library(CellChat)
  library(future)
})

# ── Paths ────────────────────────────────────────────────────────────────────
args <- commandArgs(trailingOnly = FALSE)
script_arg <- args[grep("^--file=", args)]
if (length(script_arg) > 0) {
  script_path <- normalizePath(sub("^--file=", "", script_arg))
  PROJECT_ROOT <- normalizePath(file.path(dirname(script_path), "..", ".."))
} else {
  PROJECT_ROOT <- normalizePath(".")
}
trailing_args <- commandArgs(trailingOnly = TRUE)
PREFLIGHT_ONLY <- "--preflight-only" %in% trailing_args

MTX_DIR      <- file.path(PROJECT_ROOT, "evaluation", "reports", "rctd_ovarian", "mtx")
FIRST_TYPE   <- file.path(PROJECT_ROOT, "evaluation", "reports", "rctd_ovarian", "first_type", "RAW_first_type.csv")
COORDS_FILE  <- file.path(PROJECT_ROOT, "evaluation", "reports", "rctd_ovarian", "cell_spatial_coords.csv")
OUT_DIR      <- file.path(PROJECT_ROOT, "evaluation", "reports", "ovarian_eval", "cellchat_spatial")
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

STEM_MAP <- c(
  raw          = "RAW",
  SPARKLE      = "SPARKLE",
  SpatialSoupX = "SpatialSoupX",
  SoupX        = "SoupX",
  DecontX      = "DecontX",
  SpotCleanOfficial = "SpotClean"
)
requested_methods <- trimws(strsplit(Sys.getenv("CELLCHAT_METHODS", ""), ",")[[1]])
requested_methods <- requested_methods[nzchar(requested_methods)]
partial_run <- length(requested_methods) > 0
if (partial_run) {
  missing_requested <- setdiff(requested_methods, unname(STEM_MAP))
  if (length(missing_requested) > 0) {
    stop(sprintf("Requested CellChat methods not found: %s",
                 paste(missing_requested, collapse = ", ")))
  }
  RUN_STEMS <- names(STEM_MAP)[STEM_MAP %in% requested_methods]
} else {
  RUN_STEMS <- names(STEM_MAP)
}

# ── Global params ────────────────────────────────────────────────────────────
NB    <- 20       # lower nboot for speed (documented in output)
INTERACTION_RANGE <- 250  # um – Visium spot diameter ~55 um, this captures local neighbourhoods
SCALE_DISTANCE    <- 1    # coordinates already in microns

# ── Parallel safety ──────────────────────────────────────────────────────────
plan("sequential")
options(future.globals.maxSize = 4 * 1024^3)

# ── Load shared annotation & coordinates ─────────────────────────────────────
cat("[setup] loading first_type …\n")
ft <- read.csv(FIRST_TYPE, stringsAsFactors = FALSE)
ft <- ft[!is.na(ft$first_type) & ft$first_type != "", ]
cat(sprintf("[setup] %d cells with valid first_type\n", nrow(ft)))

cat("[setup] loading spatial coordinates …\n")
coord_df <- read.csv(COORDS_FILE, stringsAsFactors = FALSE)
cat(sprintf("[setup] %d cells with coordinates\n", nrow(coord_df)))

# Build lookup: cell_id → c(x_um, y_um)
coord_map <- setNames(
  lapply(seq_len(nrow(coord_df)), function(i) c(coord_df$x_um[i], coord_df$y_um[i])),
  as.character(coord_df$cell_id)
)

# ── Helpers: load and validate counts ────────────────────────────────────────
input_paths <- function(stem) {
  list(
    mtx = file.path(MTX_DIR,
      sprintf("ovarian_x1000-1800_y300-1100_%s_counts.mtx", stem)),
    var = file.path(MTX_DIR,
      sprintf("ovarian_x1000-1800_y300-1100_%s_var.csv", stem)),
    obs = file.path(MTX_DIR,
      sprintf("ovarian_x1000-1800_y300-1100_%s_obs.csv", stem))
  )
}

assert_finite_matrix <- function(x, method_name, stage) {
  values <- if (is(x, "sparseMatrix")) x@x else as.numeric(x)
  n_bad <- sum(!is.finite(values))
  if (n_bad > 0) {
    stop(sprintf(
      "[%s] %s contains %d NaN/Inf values; refusing to write placeholder outputs",
      method_name, stage, n_bad
    ))
  }
  invisible(TRUE)
}

load_counts <- function(stem, cell_ids, canonical_cell_names = NULL) {
  paths <- input_paths(stem)
  missing_paths <- unlist(paths)[!file.exists(unlist(paths))]
  if (length(missing_paths) > 0) {
    stop(sprintf("[%s] missing input files: %s", stem,
                 paste(missing_paths, collapse = ", ")))
  }

  mtx_file <- file.path(MTX_DIR,
    sprintf("ovarian_x1000-1800_y300-1100_%s_counts.mtx", stem))
  var_file <- file.path(MTX_DIR,
    sprintf("ovarian_x1000-1800_y300-1100_%s_var.csv", stem))
  obs_file <- file.path(MTX_DIR,
    sprintf("ovarian_x1000-1800_y300-1100_%s_obs.csv", stem))

  var_df  <- read.csv(var_file, stringsAsFactors = FALSE)
  obs_df  <- read.csv(obs_file, stringsAsFactors = FALSE)
  counts  <- readMM(mtx_file)

  if (nrow(counts) != nrow(var_df) || ncol(counts) != nrow(obs_df)) {
    stop(sprintf(
      "[%s] matrix dimensions %d x %d do not match var/obs rows %d/%d",
      stem, nrow(counts), ncol(counts), nrow(var_df), nrow(obs_df)
    ))
  }
  assert_finite_matrix(counts, stem, "raw count matrix")

  # Clip negatives
  if (any(counts@x < 0)) {
    n_neg <- sum(counts@x < 0)
    counts@x[counts@x < 0] <- 0
    cat(sprintf("  [%s] clipped %d negative entries\n", stem, n_neg))
  }

  rownames(counts) <- make.names(var_df$gene_name, unique = TRUE)
  colnames(counts) <- obs_df$cell_name

  # Subset by stable cell_id rather than assuming cell_name is identical in all
  # converted files.  Assign RAW names afterwards as canonical CellChat names.
  idx <- match(cell_ids, obs_df$cell_id)
  if (any(is.na(idx))) {
    stop(sprintf("[%s] missing %d cells from count matrix", stem, sum(is.na(idx))))
  }
  counts <- counts[, idx, drop = FALSE]
  if (!is.null(canonical_cell_names)) {
    colnames(counts) <- canonical_cell_names
  }
  counts <- drop0(counts)
  counts
}

# ── Build a cross-method valid cell set ──────────────────────────────────────
cat("[setup] determining cells shared by annotation, coordinates, and all methods …\n")
if (anyDuplicated(ft$cell_id)) stop("first_type contains duplicated cell_id values")
if (anyDuplicated(coord_df$cell_id)) stop("coordinate table contains duplicated cell_id values")

method_obs <- lapply(names(STEM_MAP), function(stem) {
  path <- input_paths(stem)$obs
  if (!file.exists(path)) stop(sprintf("[%s] missing obs file: %s", stem, path))
  obs <- read.csv(path, stringsAsFactors = FALSE)
  if (anyDuplicated(obs$cell_id)) stop(sprintf("[%s] obs has duplicated cell_id values", stem))
  obs
})
names(method_obs) <- names(STEM_MAP)

shared_id_sets <- c(
  list(ft$cell_id, coord_df$cell_id),
  lapply(method_obs, function(obs) obs$cell_id)
)
shared_ids <- Reduce(intersect, shared_id_sets)
# Preserve the annotation order for deterministic matrices and outputs.
candidate_cell_ids <- unique(ft$cell_id[ft$cell_id %in% shared_ids])
raw_obs <- method_obs[["raw"]]
id_to_name <- setNames(raw_obs$cell_name, raw_obs$cell_id)
candidate_cell_names <- unname(id_to_name[as.character(candidate_cell_ids)])
if (any(is.na(candidate_cell_names))) {
  stop("RAW obs is missing names for candidate cells")
}

ft_lookup <- setNames(ft$first_type, ft$cell_id)
candidate_types <- unname(ft_lookup[as.character(candidate_cell_ids)])
cat(sprintf("[setup] candidate common cells before library QC: %d\n",
            length(candidate_cell_ids)))

# Audit every method independently, then remove the union of zero-library cells
# from every method so the benchmark uses exactly the same biological units.
library_sizes <- list()
qc_rows <- list()
for (stem in names(STEM_MAP)) {
  method_name <- STEM_MAP[[stem]]
  cat(sprintf("[preflight] auditing %s libraries …\n", method_name))
  counts_qc <- load_counts(stem, candidate_cell_ids, candidate_cell_names)
  sizes <- as.numeric(colSums(counts_qc))
  if (any(!is.finite(sizes))) {
    stop(sprintf("[%s] library sizes contain NaN/Inf", method_name))
  }
  zero <- sizes <= 0
  zero_type_counts <- sort(table(candidate_types[zero]), decreasing = TRUE)
  zero_type_text <- if (length(zero_type_counts)) {
    paste(sprintf("%s:%d", names(zero_type_counts), as.integer(zero_type_counts)),
          collapse = "; ")
  } else {
    ""
  }
  library_sizes[[method_name]] <- sizes
  qc_rows[[method_name]] <- data.frame(
    method = method_name,
    n_candidate_cells = length(sizes),
    n_zero_library_cells = sum(zero),
    zero_library_cell_ids = paste(candidate_cell_ids[zero], collapse = ";"),
    zero_library_cell_types = zero_type_text,
    total_counts = sum(sizes),
    min_cell_counts = min(sizes),
    median_cell_counts = median(sizes),
    max_cell_counts = max(sizes),
    stringsAsFactors = FALSE
  )
  rm(counts_qc)
  gc(verbose = FALSE)
}

zero_matrix <- do.call(cbind, lapply(library_sizes, function(x) x <= 0))
colnames(zero_matrix) <- names(library_sizes)
excluded <- rowSums(zero_matrix) > 0
zero_methods <- apply(zero_matrix, 1, function(x) {
  paste(colnames(zero_matrix)[x], collapse = ";")
})

cell_qc <- data.frame(
  cell_id = candidate_cell_ids,
  cell_name = candidate_cell_names,
  first_type = candidate_types,
  included_in_cellchat = !excluded,
  exclusion_reason = ifelse(excluded,
                            paste0("zero_library:", zero_methods), ""),
  stringsAsFactors = FALSE
)
for (method_name in names(library_sizes)) {
  cell_qc[[paste0(method_name, "_library_size")]] <- library_sizes[[method_name]]
}

qc_summary <- do.call(rbind, qc_rows)
qc_summary$n_shared_analyzed_cells <- sum(!excluded)
write.csv(qc_summary, file.path(OUT_DIR, "cellchat_spatial_input_qc.csv"),
          row.names = FALSE)
write.csv(cell_qc, file.path(OUT_DIR, "cellchat_spatial_cell_qc.csv"),
          row.names = FALSE)

cat(sprintf("[preflight] excluded %d/%d cells with a zero library in any method\n",
            sum(excluded), length(excluded)))
print(qc_summary[, c("method", "n_candidate_cells", "n_zero_library_cells",
                     "n_shared_analyzed_cells")], row.names = FALSE)

common_cell_ids <- candidate_cell_ids[!excluded]
common_cell_names <- candidate_cell_names[!excluded]
common_types <- candidate_types[!excluded]
if (length(common_cell_ids) == 0) stop("No cells remain after cross-method library QC")

# Build coordinates matrix (rows = cells, columns = x, y) for the final set.
coord_mat <- do.call(rbind, coord_map[as.character(common_cell_ids)])
colnames(coord_mat) <- c("x", "y")
rownames(coord_mat) <- common_cell_names
if (any(!is.finite(coord_mat))) stop("Final coordinate matrix contains NaN/Inf")

cat(sprintf("[setup] final common cell set: %d\n", length(common_cell_ids)))
cat(sprintf("[setup] coord x range: %.0f–%.0f um, y range: %.0f–%.0f um\n",
            min(coord_mat[,1]), max(coord_mat[,1]),
            min(coord_mat[,2]), max(coord_mat[,2])))

if (PREFLIGHT_ONLY) {
  cat(sprintf("[preflight] complete; QC files written to %s\n", OUT_DIR))
  quit(save = "no", status = 0)
}

# ── Run CellChat pipeline ─────────────────────────────────────────────────────
run_cellchat_spatial <- function(method_name, counts, cells, types, coords, out_dir) {

  cat(sprintf("\n========== [%s] Starting CellChat spatial pipeline ==========\n", method_name))

  if (ncol(counts) != length(cells) || length(types) != length(cells) ||
      nrow(coords) != length(cells)) {
    stop(sprintf("[%s] counts/meta/coordinate cell dimensions are not aligned", method_name))
  }
  assert_finite_matrix(counts, method_name, "CellChat input counts")
  library_size <- as.numeric(colSums(counts))
  if (any(!is.finite(library_size)) || any(library_size <= 0)) {
    bad <- which(!is.finite(library_size) | library_size <= 0)
    stop(sprintf(
      "[%s] %d non-positive/non-finite libraries reached normalization: %s",
      method_name, length(bad), paste(cells[bad], collapse = ", ")
    ))
  }

  # Normalize (log-normalized library-size)
  cat(sprintf("  [%s] normalizing …\n", method_name))
  data_input <- normalizeData(counts)
  assert_finite_matrix(data_input, method_name, "normalized expression")

  # Meta data frame
  meta <- data.frame(labels = types, row.names = cells, stringsAsFactors = FALSE)

  # Spatial factors (coordinates in microns → ratio = 1)
  spatial_factors <- data.frame(ratio = 1, tol = 5)  # tol ~ cell radius

  # Create CellChat object
  cat(sprintf("  [%s] creating CellChat object …\n", method_name))
  cellchat <- createCellChat(
    object          = data_input,
    meta            = meta,
    group.by        = "labels",
    datatype        = "spatial",
    coordinates     = coords,
    spatial.factors = spatial_factors
  )

  # Set database
  data("CellChatDB.human", package = "CellChat")
  cellchat@DB <- CellChatDB.human

  # Subset data to ligand/receptor genes present
  cat(sprintf("  [%s] subsetData …\n", method_name))
  cellchat <- subsetData(cellchat)

  # Identify over-expressed genes & interactions
  cat(sprintf("  [%s] identifyOverExpressedGenes …\n", method_name))
  cellchat <- identifyOverExpressedGenes(cellchat)
  cat(sprintf("  [%s] identifyOverExpressedInteractions …\n", method_name))
  cellchat <- identifyOverExpressedInteractions(cellchat)

  # A zero-LR result is non-estimable for this benchmark.  Fail explicitly
  # instead of encoding a pipeline failure as an empty table plus zero matrices.
  if (nrow(cellchat@LR$LRsig) == 0) {
    stop(sprintf(paste0(
      "[%s] identifyOverExpressedInteractions retained 0 LR pairs; ",
      "no placeholder outputs were written"
    ), method_name))
  }

  # ── SPATIAL communication inference (KEY STEP) ──
  cat(sprintf("  [%s] computeCommunProb (spatial, distance.use=TRUE) …\n", method_name))
  cellchat <- computeCommunProb(
    cellchat,
    type                 = "truncatedMean",
    trim                 = 0.1,
    distance.use         = TRUE,
    interaction.range    = INTERACTION_RANGE,
    scale.distance       = SCALE_DISTANCE,
    contact.dependent    = TRUE,
    contact.range        = 100,
    nboot                = NB,
    seed.use             = 42
  )

  # Filter
  cat(sprintf("  [%s] filterCommunication …\n", method_name))
  cellchat <- filterCommunication(cellchat, min.cells = 10)

  # Extract significant LR pairs
  cat(sprintf("  [%s] subsetCommunication …\n", method_name))
  df_net <- subsetCommunication(cellchat)
  cat(sprintf("  [%s] significant LR interactions: %d\n", method_name, nrow(df_net)))

  # Save
  out_csv <- file.path(out_dir, sprintf("%s_cellchat_spatial.csv", method_name))
  write.csv(df_net, out_csv, row.names = FALSE)
  cat(sprintf("  [%s] saved %s\n", method_name, out_csv))

  # Aggregate and save count/weight matrices
  cellchat <- aggregateNet(cellchat)
  agg_count  <- cellchat@net$count
  agg_weight <- cellchat@net$weight
  write.csv(as.matrix(agg_count),
            file.path(out_dir, sprintf("%s_agg_count.csv", method_name)))
  write.csv(as.matrix(agg_weight),
            file.path(out_dir, sprintf("%s_agg_weight.csv", method_name)))

  df_net
}

# ── Run pipeline for each method ─────────────────────────────────────────────
all_results <- list()
summary_path <- file.path(OUT_DIR, "cellchat_spatial_summary.csv")
if (!partial_run && file.exists(summary_path)) unlink(summary_path)

for (stem in RUN_STEMS) {
  method_name <- STEM_MAP[[stem]]
  cat(sprintf("\n############################################################\n"))
  cat(sprintf("# Method: %s (stem=%s)\n", method_name, stem))
  cat(sprintf("############################################################\n"))

  # Remove stale per-method artifacts before starting.  If this run fails, a
  # previous empty/zero result cannot be mistaken for the new analysis.
  stale_outputs <- file.path(OUT_DIR, c(
    sprintf("%s_cellchat_spatial.csv", method_name),
    sprintf("%s_agg_count.csv", method_name),
    sprintf("%s_agg_weight.csv", method_name)
  ))
  unlink(stale_outputs[file.exists(stale_outputs)])

  counts <- load_counts(stem, common_cell_ids, common_cell_names)
  df_net <- run_cellchat_spatial(method_name, counts, common_cell_names,
                                 common_types, coord_mat, OUT_DIR)
  all_results[[method_name]] <- df_net
}

# ── Analysis / comparison ────────────────────────────────────────────────────
cat("\n\n========== Comparison Summary ==========\n\n")

tumor_types <- c(
  "Tumor Cells",
  "Proliferative Tumor Cells",
  "VEGFA+ Tumor Cells",
  "MT-High, Jun+-Fos+ Tumor Cells",
  "Inflammatory Tumor Cells",
  "Malignant Cells Lining Cyst"
)

summary_rows <- list()

for (method_name in names(all_results)) {
  df <- all_results[[method_name]]

  sig <- df[df$pval <= 0.05, , drop = FALSE]
  n_sig    <- nrow(sig)
  tot_prob <- sum(sig$prob, na.rm = TRUE)

  # Autocrine
  auto <- sig[sig$source == sig$target, , drop = FALSE]
  auto_prob <- sum(auto$prob, na.rm = TRUE)
  auto_n    <- nrow(auto)

  # Tumor involving
  tumor_sig <- sig[sig$source %in% tumor_types | sig$target %in% tumor_types, , drop = FALSE]
  tumor_n   <- nrow(tumor_sig)
  tumor_p   <- sum(tumor_sig$prob, na.rm = TRUE)

  # Tumor → Stromal
  t2s <- sig[sig$source %in% tumor_types & !sig$target %in% tumor_types, , drop = FALSE]
  t2s_n <- nrow(t2s)
  t2s_p <- sum(t2s$prob, na.rm = TRUE)

  # Stromal → Tumor
  s2t <- sig[!sig$source %in% tumor_types & sig$target %in% tumor_types, , drop = FALSE]
  s2t_n <- nrow(s2t)
  s2t_p <- sum(s2t$prob, na.rm = TRUE)

  summary_rows[[method_name]] <- data.frame(
    Method                       = method_name,
    n_significant                = n_sig,
    total_prob                   = tot_prob,
    autocrine_n                  = auto_n,
    autocrine_prob               = auto_prob,
    tumor_involving_n            = tumor_n,
    tumor_involving_prob         = tumor_p,
    tumor_to_stromal_n           = t2s_n,
    tumor_to_stromal_prob        = t2s_p,
    stromal_to_tumor_n           = s2t_n,
    stromal_to_tumor_prob        = s2t_p,
    stringsAsFactors             = FALSE
  )
}

summary_df <- do.call(rbind, summary_rows)
rownames(summary_df) <- NULL

if (partial_run && file.exists(summary_path)) {
  previous_summary <- read.csv(summary_path, stringsAsFactors = FALSE)
  previous_summary <- previous_summary[
    !previous_summary$Method %in% names(all_results), , drop = FALSE
  ]
  summary_df <- rbind(previous_summary, summary_df)
}

write.csv(summary_df, summary_path, row.names = FALSE)

# ── Print comparison table ───────────────────────────────────────────────────
cat("\n--- Per-method summary ---\n")
print(summary_df, row.names = FALSE)

# % change vs RAW
cat("\n--- % Change vs RAW ---\n")
raw_row <- summary_df[summary_df$Method == "RAW", ]
for (method_name in names(all_results)) {
  if (method_name == "RAW") next
  row <- summary_df[summary_df$Method == method_name, ]
  pct <- function(new, old) ifelse(old == 0, NA, round(100 * (new - old) / old, 1))
  cat(sprintf("\n  %s:\n", method_name))
  cat(sprintf("    n_sig:            %d → %d  (%+.1f%%)\n", raw_row$n_significant, row$n_significant,
              pct(row$n_significant, raw_row$n_significant)))
  cat(sprintf("    total_prob:       %.4f → %.4f  (%+.1f%%)\n", raw_row$total_prob, row$total_prob,
              pct(row$total_prob, raw_row$total_prob)))
  cat(sprintf("    tumor_involving_n: %d → %d  (%+.1f%%)\n", raw_row$tumor_involving_n, row$tumor_involving_n,
              pct(row$tumor_involving_n, raw_row$tumor_involving_n)))
  cat(sprintf("    tumor_involving_prob: %.4f → %.4f  (%+.1f%%)\n",
              raw_row$tumor_involving_prob, row$tumor_involving_prob,
              pct(row$tumor_involving_prob, raw_row$tumor_involving_prob)))
  cat(sprintf("    tumor→stromal_n:  %d → %d  (%+.1f%%)\n", raw_row$tumor_to_stromal_n, row$tumor_to_stromal_n,
              pct(row$tumor_to_stromal_n, raw_row$tumor_to_stromal_n)))
  cat(sprintf("    stromal→tumor_n:  %d → %d  (%+.1f%%)\n", raw_row$stromal_to_tumor_n, row$stromal_to_tumor_n,
              pct(row$stromal_to_tumor_n, raw_row$stromal_to_tumor_n)))
}

cat("\n=== CellChat spatial analysis complete ===\n")
