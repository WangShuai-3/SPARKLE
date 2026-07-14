#!/usr/bin/env Rscript
#
# Generate enriched first_type CSVs from existing mousebrain RCTD doublet results.
# Mirrors enrich_first_type_scores.R (ovarian) but with mousebrain's file naming
# convention (rctd_Cell_subclass_doublet_results.csv).
#
# Run:
#   Rscript evaluation/scripts/enrich_first_type_scores_mousebrain.R
#
suppressPackageStartupMessages(library(data.table))

PROJECT_ROOT <- file.path(dirname(normalizePath(
  sub("^--file=", "", grep("^--file=", commandArgs(trailingOnly=FALSE), value=TRUE))
)), "..", "..")
if (!dir.exists(file.path(PROJECT_ROOT, "evaluation"))) PROJECT_ROOT <- "."

RCTD_CSV <- file.path(PROJECT_ROOT, "evaluation", "reports", "rctd_mousebrain",
                      "rctd_Cell_subclass_doublet_results.csv")
FIRST_DIR <- file.path(PROJECT_ROOT, "evaluation", "reports", "rctd_mousebrain",
                       "first_type")

stopifnot(file.exists(RCTD_CSV))
dir.create(FIRST_DIR, recursive=TRUE, showWarnings=FALSE)

all <- fread(RCTD_CSV)
message(sprintf("Read %d rows from %s", nrow(all), RCTD_CSV))
message("Methods found: ", paste(sort(unique(all$method)), collapse=", "))

# H5ad cell_name -> cell_id mapping (use RAW obs)
obs <- fread(file.path(PROJECT_ROOT, "evaluation", "reports",
                       "rctd_mousebrain", "mtx",
                       "mousebrain_x12500-17500_y2000-5000_raw_obs.csv"))
cell_name_id <- setNames(obs$cell_id, obs$cell_name)

for (m in sort(unique(all$method))) {
  sub <- all[method == m]
  sub[, score_delta := singlet_score - min_score]

  out <- data.table(
    cell_name = sub$cell_barcode,
    cell_id   = as.integer(cell_name_id[sub$cell_barcode]),
    first_type   = sub$first_type,
    second_type  = sub$second_type,
    singlet_score = sub$singlet_score,
    min_score     = sub$min_score,
    score_delta   = sub$score_delta,
    spot_class    = sub$spot_class
  )

  fpath <- file.path(FIRST_DIR, paste0(m, "_first_type.csv"))
  fwrite(out, fpath)
  msg <- sprintf("  %-14s %6d cells  mean singlet=%.1f  min=%.1f  delta=%.1f",
                 m, nrow(out),
                 mean(out$singlet_score, na.rm=TRUE),
                 mean(out$min_score, na.rm=TRUE),
                 mean(out$score_delta, na.rm=TRUE))
  message(msg)
}

message("Done. Enriched first_type CSVs written to ", FIRST_DIR)
