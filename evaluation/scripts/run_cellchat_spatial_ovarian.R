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
###############################################################################

suppressPackageStartupMessages({
  library(Matrix)
  library(CellChat)
  library(future)
})

# ── Paths ────────────────────────────────────────────────────────────────────
MTX_DIR      <- "evaluation/reports/rctd_ovarian/mtx"
FIRST_TYPE   <- "evaluation/reports/rctd_ovarian/first_type/RAW_first_type.csv"
COORDS_FILE  <- "evaluation/reports/rctd_ovarian/cell_spatial_coords.csv"
OUT_DIR      <- "evaluation/reports/ovarian_eval/cellchat_spatial"
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

STEM_MAP <- c(
  raw          = "RAW",
  SPARKLE      = "SPARKLE",
  SpatialSoupX = "SpatialSoupX",
  SoupX        = "SoupX",
  DecontX      = "DecontX"
)

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

# ── Compute common cell set from RAW (used for all methods) ──────────────────
cat("[setup] determining common cell set from RAW …\n")

# Read RAW obs to get cell_name <-> cell_id mapping
raw_obs <- read.csv(file.path(MTX_DIR, "ovarian_x1000-1800_y300-1100_raw_obs.csv"),
                    stringsAsFactors = FALSE)

# Cells with annotation AND coordinates
cells_with_annot <- ft$cell_id     # integer
cells_with_coord <- coord_df$cell_id

common_cell_ids <- intersect(cells_with_annot, cells_with_coord)
cat(sprintf("[setup] common cells (annot + coords): %d\n", length(common_cell_ids)))

# Map cell_id → cell_name for the common set
id_to_name <- setNames(raw_obs$cell_name, raw_obs$cell_id)
common_cell_names <- id_to_name[as.character(common_cell_ids)]
# Remove any NA (shouldn't happen)
common_cell_ids   <- common_cell_ids[!is.na(common_cell_names)]
common_cell_names <- common_cell_names[!is.na(common_cell_names)]
cat(sprintf("[setup] final common cell set: %d\n", length(common_cell_ids)))

# Build first_type vector (mapped by cell_id) for the common set
ft_lookup <- setNames(ft$first_type, ft$cell_id)
common_types <- ft_lookup[as.character(common_cell_ids)]

# Build coordinates matrix (rows = cells, columns = x, y) for common set
coord_mat <- do.call(rbind, coord_map[as.character(common_cell_ids)])
colnames(coord_mat) <- c("x", "y")
rownames(coord_mat) <- common_cell_names

# Report coordinate range
cat(sprintf("[setup] coord x range: %.0f–%.0f um, y range: %.0f–%.0f um\n",
            min(coord_mat[,1]), max(coord_mat[,1]),
            min(coord_mat[,2]), max(coord_mat[,2])))

# ── Helper: load counts for a method, subset to common cells ─────────────────
load_counts <- function(stem, cells) {
  mtx_file <- file.path(MTX_DIR,
    sprintf("ovarian_x1000-1800_y300-1100_%s_counts.mtx", stem))
  var_file <- file.path(MTX_DIR,
    sprintf("ovarian_x1000-1800_y300-1100_%s_var.csv", stem))
  obs_file <- file.path(MTX_DIR,
    sprintf("ovarian_x1000-1800_y300-1100_%s_obs.csv", stem))

  var_df  <- read.csv(var_file, stringsAsFactors = FALSE)
  obs_df  <- read.csv(obs_file, stringsAsFactors = FALSE)
  counts  <- readMM(mtx_file)

  # Clip negatives
  if (any(counts@x < 0)) {
    n_neg <- sum(counts@x < 0)
    counts@x[counts@x < 0] <- 0
    cat(sprintf("  [%s] clipped %d negative entries\n", stem, n_neg))
  }

  rownames(counts) <- make.names(var_df$gene_name, unique = TRUE)
  colnames(counts) <- obs_df$cell_name

  # Subset to common cells (in order)
  idx <- match(cells, colnames(counts))
  if (any(is.na(idx))) {
    stop(sprintf("[%s] missing %d cells from count matrix", stem, sum(is.na(idx))))
  }
  counts <- counts[, idx, drop = FALSE]
  counts
}

# ── Run CellChat pipeline ─────────────────────────────────────────────────────
run_cellchat_spatial <- function(method_name, counts, cells, types, coords, out_dir) {

  cat(sprintf("\n========== [%s] Starting CellChat spatial pipeline ==========\n", method_name))

  # Normalize (log-normalized library-size)
  cat(sprintf("  [%s] normalizing …\n", method_name))
  data_input <- normalizeData(counts)

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

  # Edge case: zero significant LR pairs (e.g. DecontX with near-zero expression)
  if (nrow(cellchat@LR$LRsig) == 0) {
    cat(sprintf("  [%s] WARNING: 0 significant LR pairs — returning empty result\n", method_name))
    empty_df <- data.frame(
      source=character(), target=character(), ligand=character(), receptor=character(),
      prob=numeric(), pval=numeric(), interaction_name=character(),
      interaction_name_2=character(), pathway_name=character(),
      annotation=character(), evidence=character(),
      stringsAsFactors=FALSE
    )
    write.csv(empty_df, file.path(out_dir, sprintf("%s_cellchat_spatial.csv", method_name)), row.names=FALSE)
    # Write empty aggregate matrices too
    ug <- sort(unique(types))
    write.csv(matrix(0,length(ug),length(ug),dimnames=list(ug,ug)),
              file.path(out_dir, sprintf("%s_agg_count.csv", method_name)))
    write.csv(matrix(0,length(ug),length(ug),dimnames=list(ug,ug)),
              file.path(out_dir, sprintf("%s_agg_weight.csv", method_name)))
    return(empty_df)
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

for (stem in names(STEM_MAP)) {
  method_name <- STEM_MAP[[stem]]
  cat(sprintf("\n############################################################\n"))
  cat(sprintf("# Method: %s (stem=%s)\n", method_name, stem))
  cat(sprintf("############################################################\n"))

  counts <- load_counts(stem, common_cell_names)
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

write.csv(summary_df,
          file.path(OUT_DIR, "cellchat_spatial_summary.csv"),
          row.names = FALSE)

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
