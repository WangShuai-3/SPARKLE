#!/usr/bin/env Rscript
###############################################################################
# Spatial CellChat v2 analysis for the CRC Visium HD evaluation.
#
# The script mirrors the ovarian spatial analysis while preserving the two CRC
# segmentation results as independent experimental conditions.  Within each
# condition all five expression methods use:
#
#   * the exact same cells;
#   * RAW-RCTD first_type annotations;
#   * segmentation-specific cell-centroid coordinates;
#   * the same CellChatDB.human database and inference parameters.
#
# A balanced Pelka CRC single-cell reference is also analysed once with the
# same R CellChat implementation (without spatial distance, because dissociated
# cells have no coordinates).  The companion Python script compares spatial
# significant interactions with this reference.
#
# Usage (run once for each segmentation):
#
#   CRC_SEGMENTATION=proseg \
#     /home/shuaiwang/miniconda3/envs/r-env/bin/Rscript \
#     evaluation/scripts/run_cellchat_spatial_crc.R
#
#   CRC_SEGMENTATION=stardist \
#     /home/shuaiwang/miniconda3/envs/r-env/bin/Rscript \
#     evaluation/scripts/run_cellchat_spatial_crc.R
#
# Optional environment variables:
#   CELLCHAT_NBOOT=20
#   CELLCHAT_INTERACTION_RANGE=250
#   CELLCHAT_MIN_CELLS=10
#   CELLCHAT_METHODS=RAW,SPARKLE,SpatialSoupX,SoupX,DecontX
#   CELLCHAT_FORCE_REFERENCE=1
#   CELLCHAT_REFERENCE_ONLY=1
###############################################################################

suppressPackageStartupMessages({
  library(Matrix)
  library(CellChat)
  library(future)
})

# Resolve all paths relative to the repository instead of the caller's cwd.
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

TAG <- sprintf("crc_%s_x14200-15000_y2750-3550", SEGMENTATION)
RCTD_DIR <- file.path(
  PROJECT_ROOT, "evaluation", "reports", "rctd_crc", SEGMENTATION
)
MTX_DIR <- file.path(RCTD_DIR, "mtx")
FIRST_TYPE_FILE <- file.path(RCTD_DIR, "first_type", "RAW_first_type.csv")
OUT_ROOT <- file.path(
  PROJECT_ROOT, "evaluation", "reports", "crc_eval_cellchat_spatial"
)
OUT_DIR <- file.path(OUT_ROOT, SEGMENTATION)
REF_OUT_DIR <- file.path(OUT_ROOT, "reference")
REF_DIR <- file.path(
  PROJECT_ROOT, "evaluation", "data", "CRC", "scrna_reference",
  "prepared_cluster_midway"
)
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(REF_OUT_DIR, recursive = TRUE, showWarnings = FALSE)

# Match the ovarian analysis defaults.  Twenty bootstrap permutations keeps the
# ten CRC method/segmentation runs tractable; the seed makes them reproducible.
NBOOT <- as.integer(Sys.getenv("CELLCHAT_NBOOT", "20"))
INTERACTION_RANGE <- as.numeric(
  Sys.getenv("CELLCHAT_INTERACTION_RANGE", "250")
)
MIN_CELLS <- as.integer(Sys.getenv("CELLCHAT_MIN_CELLS", "10"))
SEED <- 42L
FORCE_REFERENCE <- Sys.getenv("CELLCHAT_FORCE_REFERENCE", "0") == "1"
REFERENCE_ONLY <- Sys.getenv("CELLCHAT_REFERENCE_ONLY", "0") == "1"

STEM_MAP <- c(
  raw = "RAW",
  SPARKLE = "SPARKLE",
  SpatialSoupX = "SpatialSoupX",
  SoupX = "SoupX",
  DecontX = "DecontX"
)
requested_methods <- strsplit(
  Sys.getenv("CELLCHAT_METHODS", paste(STEM_MAP, collapse = ",")), ",",
  fixed = TRUE
)[[1]]
requested_methods <- trimws(requested_methods)
if (!all(requested_methods %in% unname(STEM_MAP))) {
  stop("CELLCHAT_METHODS contains an unknown method")
}
STEM_MAP <- STEM_MAP[STEM_MAP %in% requested_methods]

plan("sequential")
options(future.globals.maxSize = 6 * 1024^3)

TUMOR_TYPES <- c("EpiT")

# Matrix Market files are genes x cells.  Gene symbols must remain unchanged so
# CellChatDB identifiers still match; make.unique only disambiguates duplicates.
load_counts <- function(stem, cells = NULL) {
  prefix <- file.path(MTX_DIR, sprintf("%s_%s", TAG, stem))
  counts <- as(readMM(paste0(prefix, "_counts.mtx")), "CsparseMatrix")
  obs <- read.csv(paste0(prefix, "_obs.csv"), stringsAsFactors = FALSE)
  var <- read.csv(paste0(prefix, "_var.csv"), stringsAsFactors = FALSE)
  if (nrow(counts) != nrow(var) || ncol(counts) != nrow(obs)) {
    stop(sprintf("Dimension mismatch for %s", prefix))
  }
  rownames(counts) <- make.unique(as.character(var$gene_name))
  colnames(counts) <- as.character(obs$cell_name)

  # Corrected baselines may contain tiny numerical negatives.  CellChat expects
  # non-negative expression; clipping is the same input handling used by RCTD.
  if (length(counts@x) > 0 && any(counts@x < 0)) {
    n_negative <- sum(counts@x < 0)
    counts@x[counts@x < 0] <- 0
    counts <- drop0(counts)
    cat(sprintf("  [%s] clipped %d negative matrix entries\n", stem, n_negative))
  }

  if (!is.null(cells)) {
    index <- match(cells, colnames(counts))
    if (any(is.na(index))) {
      stop(sprintf("[%s] is missing %d fixed-cohort cells", stem, sum(is.na(index))))
    }
    counts <- counts[, index, drop = FALSE]
    obs <- obs[index, , drop = FALSE]
  }
  list(counts = counts, obs = obs)
}

write_empty_result <- function(method_name, groups) {
  empty <- data.frame(
    source = character(), target = character(), ligand = character(),
    receptor = character(), prob = numeric(), pval = numeric(),
    interaction_name = character(), interaction_name_2 = character(),
    pathway_name = character(), annotation = character(),
    evidence = character(), stringsAsFactors = FALSE
  )
  write.csv(
    empty, file.path(OUT_DIR, sprintf("%s_cellchat_spatial.csv", method_name)),
    row.names = FALSE
  )
  zero <- matrix(0, length(groups), length(groups), dimnames = list(groups, groups))
  write.csv(zero, file.path(OUT_DIR, sprintf("%s_agg_count.csv", method_name)))
  write.csv(zero, file.path(OUT_DIR, sprintf("%s_agg_weight.csv", method_name)))
  empty
}

run_spatial_cellchat <- function(method_name, counts, cells, types, coords) {
  cat(sprintf("\n========== [%s/%s] spatial CellChat ==========\n",
              SEGMENTATION, method_name))
  data_input <- normalizeData(counts)
  meta <- data.frame(labels = types, row.names = cells, stringsAsFactors = FALSE)

  # CRC h5ad x/y fields are segmentation-mask centroid coordinates in microns.
  # ratio=1 therefore preserves physical distance; tol=5 approximates cell radius.
  spatial_factors <- data.frame(ratio = 1, tol = 5)
  cellchat <- createCellChat(
    object = data_input,
    meta = meta,
    group.by = "labels",
    datatype = "spatial",
    coordinates = coords,
    spatial.factors = spatial_factors
  )
  data("CellChatDB.human", package = "CellChat")
  cellchat@DB <- CellChatDB.human
  cellchat <- subsetData(cellchat)
  cellchat <- identifyOverExpressedGenes(cellchat)
  cellchat <- identifyOverExpressedInteractions(cellchat)

  if (nrow(cellchat@LR$LRsig) == 0) {
    warning(sprintf("%s/%s has zero over-expressed LR pairs", SEGMENTATION, method_name))
    return(write_empty_result(method_name, sort(unique(types))))
  }

  cellchat <- computeCommunProb(
    cellchat,
    type = "truncatedMean",
    trim = 0.1,
    distance.use = TRUE,
    interaction.range = INTERACTION_RANGE,
    scale.distance = 1,
    contact.dependent = TRUE,
    contact.range = 100,
    nboot = NBOOT,
    seed.use = SEED
  )
  cellchat <- filterCommunication(cellchat, min.cells = MIN_CELLS)
  result <- subsetCommunication(cellchat, thresh = 0.05)
  write.csv(
    result, file.path(OUT_DIR, sprintf("%s_cellchat_spatial.csv", method_name)),
    row.names = FALSE
  )

  cellchat <- aggregateNet(cellchat)
  write.csv(
    as.matrix(cellchat@net$count),
    file.path(OUT_DIR, sprintf("%s_agg_count.csv", method_name))
  )
  write.csv(
    as.matrix(cellchat@net$weight),
    file.path(OUT_DIR, sprintf("%s_agg_weight.csv", method_name))
  )
  cat(sprintf("  significant spatial interactions: %d\n", nrow(result)))
  result
}

run_reference_cellchat <- function() {
  output_sig <- file.path(REF_OUT_DIR, "Pelka_scRNA_cellchat.csv")
  output_all <- file.path(REF_OUT_DIR, "Pelka_scRNA_cellchat_all_tested.csv")
  output_parameters <- file.path(REF_OUT_DIR, "run_parameters.csv")
  cache_matches <- FALSE
  if (file.exists(output_parameters)) {
    cached <- read.csv(output_parameters, stringsAsFactors = FALSE)
    cache_matches <- nrow(cached) == 1 &&
      cached$nboot[1] == NBOOT && cached$seed[1] == SEED &&
      cached$min_cells_per_type[1] == MIN_CELLS
  }
  if (!FORCE_REFERENCE && file.exists(output_sig) && file.exists(output_all) &&
      cache_matches) {
      cat(sprintf("[reference] reusing %s\n", output_sig))
      return(invisible(NULL))
  }

  cat("[reference] loading balanced Pelka single-cell count matrix ...\n")
  counts <- as(
    readMM(file.path(REF_DIR, "rctd_counts.mtx")), "CsparseMatrix"
  )
  genes <- read.csv(
    file.path(REF_DIR, "rctd_genes.csv"), stringsAsFactors = FALSE
  )
  cells <- read.csv(
    file.path(REF_DIR, "rctd_cells.csv"), stringsAsFactors = FALSE
  )
  if (nrow(counts) != nrow(genes) || ncol(counts) != nrow(cells)) {
    stop("Pelka reference matrix and metadata dimensions do not agree")
  }
  rownames(counts) <- make.unique(as.character(genes$gene_name))
  colnames(counts) <- as.character(cells$cell_name)
  labels <- as.character(cells$cell_type)
  meta <- data.frame(labels = labels, row.names = colnames(counts))

  cellchat <- createCellChat(
    object = normalizeData(counts), meta = meta, group.by = "labels",
    datatype = "RNA"
  )
  data("CellChatDB.human", package = "CellChat")
  cellchat@DB <- CellChatDB.human
  cellchat <- subsetData(cellchat)
  cellchat <- identifyOverExpressedGenes(cellchat)
  cellchat <- identifyOverExpressedInteractions(cellchat)
  cellchat <- computeCommunProb(
    cellchat,
    type = "truncatedMean",
    trim = 0.1,
    distance.use = FALSE,
    nboot = NBOOT,
    seed.use = SEED
  )
  cellchat <- filterCommunication(cellchat, min.cells = MIN_CELLS)

  # The all-tested table is required to distinguish a reference-negative
  # interaction from one that cannot be assessed in the single-cell data.
  all_tested <- subsetCommunication(cellchat, thresh = 1.01)
  # subsetCommunication(thresh=0.05) uses a strict cutoff.  With NBOOT=20,
  # p-values are multiples of 0.05, so matching that rule matters in practice.
  significant <- all_tested[all_tested$pval < 0.05, , drop = FALSE]
  write.csv(all_tested, output_all, row.names = FALSE)
  write.csv(significant, output_sig, row.names = FALSE)
  write.csv(
    as.data.frame(table(labels), stringsAsFactors = FALSE),
    file.path(REF_OUT_DIR, "Pelka_scRNA_cell_type_counts.csv"), row.names = FALSE
  )
  write.csv(
    data.frame(
      n_cells = ncol(counts), n_cell_types = length(unique(labels)),
      nboot = NBOOT, seed = SEED, min_cells_per_type = MIN_CELLS,
      p_value_rule = "pval < 0.05", n_tested = nrow(all_tested),
      n_significant = nrow(significant)
    ),
    output_parameters, row.names = FALSE
  )
  cat(sprintf(
    "[reference] %d tested, %d significant interactions\n",
    nrow(all_tested), nrow(significant)
  ))
}

# Build a segmentation-specific, method-invariant cell cohort.  Cell types with
# fewer than MIN_CELLS are excluded before any method sees the data, preventing
# group-size changes from masquerading as expression-correction effects.
cat(sprintf("[setup] CRC/%s tag=%s\n", SEGMENTATION, TAG))
raw_input <- load_counts("raw")
first_type <- read.csv(FIRST_TYPE_FILE, stringsAsFactors = FALSE)
annotation_lookup <- setNames(first_type$first_type, first_type$cell_name)
raw_obs <- raw_input$obs
raw_obs$first_type <- annotation_lookup[as.character(raw_obs$cell_name)]
raw_obs <- raw_obs[
  !is.na(raw_obs$first_type) & raw_obs$first_type != "", , drop = FALSE
]
type_counts_initial <- table(raw_obs$first_type)
eligible_types <- names(type_counts_initial[type_counts_initial >= MIN_CELLS])
cohort <- raw_obs[raw_obs$first_type %in% eligible_types, , drop = FALSE]
cohort <- cohort[!duplicated(cohort$cell_name), , drop = FALSE]
if (nrow(cohort) == 0) stop("No CRC cells passed annotation/type-size filters")

fixed_cells <- as.character(cohort$cell_name)
fixed_types <- as.character(cohort$first_type)
coords <- as.matrix(cohort[, c("x", "y")])
storage.mode(coords) <- "double"
rownames(coords) <- fixed_cells
colnames(coords) <- c("x", "y")

write.csv(cohort, file.path(OUT_DIR, "fixed_cell_cohort.csv"), row.names = FALSE)
write.csv(
  data.frame(cell_type = names(table(fixed_types)),
             n_cells = as.integer(table(fixed_types))),
  file.path(OUT_DIR, "fixed_cell_type_counts.csv"), row.names = FALSE
)
write.csv(
  data.frame(
    segmentation = SEGMENTATION,
    tag = TAG,
    n_cells = length(fixed_cells),
    n_cell_types = length(unique(fixed_types)),
    nboot = NBOOT,
    seed = SEED,
    interaction_range_um = INTERACTION_RANGE,
    contact_range_um = 100,
    min_cells_per_type = MIN_CELLS,
    x_min_um = min(coords[, 1]), x_max_um = max(coords[, 1]),
    y_min_um = min(coords[, 2]), y_max_um = max(coords[, 2])
  ),
  file.path(OUT_DIR, "run_parameters.csv"), row.names = FALSE
)
cat(sprintf(
  "[setup] fixed cohort: %d cells, %d types; x %.1f-%.1f, y %.1f-%.1f um\n",
  length(fixed_cells), length(unique(fixed_types)),
  min(coords[, 1]), max(coords[, 1]), min(coords[, 2]), max(coords[, 2])
))

# The single-cell reference is shared by both segmentation runs and is cached.
run_reference_cellchat()
if (REFERENCE_ONLY) {
  cat("[reference] reference-only run complete\n")
  quit(save = "no", status = 0)
}

all_results <- list()
for (stem in names(STEM_MAP)) {
  method_name <- unname(STEM_MAP[[stem]])
  input <- load_counts(stem, fixed_cells)
  all_results[[method_name]] <- run_spatial_cellchat(
    method_name, input$counts, fixed_cells, fixed_types, coords
  )
}

# Compact per-method summary.  EpiT is the Pelka tumor-epithelial ClusterMidway
# label; Epi is normal epithelial and is deliberately not called tumor.
summary_rows <- list()
for (method_name in names(all_results)) {
  result <- all_results[[method_name]]
  significant <- result[result$pval <= 0.05, , drop = FALSE]
  autocrine <- significant[significant$source == significant$target, , drop = FALSE]
  tumor <- significant[
    significant$source %in% TUMOR_TYPES |
      significant$target %in% TUMOR_TYPES, , drop = FALSE
  ]
  t2n <- significant[
    significant$source %in% TUMOR_TYPES &
      !significant$target %in% TUMOR_TYPES, , drop = FALSE
  ]
  n2t <- significant[
    !significant$source %in% TUMOR_TYPES &
      significant$target %in% TUMOR_TYPES, , drop = FALSE
  ]
  summary_rows[[method_name]] <- data.frame(
    segmentation = SEGMENTATION,
    method = method_name,
    n_significant = nrow(significant),
    total_prob = sum(significant$prob, na.rm = TRUE),
    autocrine_n = nrow(autocrine),
    autocrine_prob = sum(autocrine$prob, na.rm = TRUE),
    tumor_involving_n = nrow(tumor),
    tumor_involving_prob = sum(tumor$prob, na.rm = TRUE),
    tumor_to_nontumor_n = nrow(t2n),
    tumor_to_nontumor_prob = sum(t2n$prob, na.rm = TRUE),
    nontumor_to_tumor_n = nrow(n2t),
    nontumor_to_tumor_prob = sum(n2t$prob, na.rm = TRUE),
    stringsAsFactors = FALSE
  )
}
summary_df <- do.call(rbind, summary_rows)
rownames(summary_df) <- NULL
write.csv(
  summary_df, file.path(OUT_DIR, "cellchat_spatial_summary.csv"),
  row.names = FALSE
)
print(summary_df, row.names = FALSE)
cat(sprintf("\nCRC/%s spatial CellChat complete: %s\n", SEGMENTATION, OUT_DIR))
