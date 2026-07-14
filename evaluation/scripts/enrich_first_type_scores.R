#!/usr/bin/env Rscript
#
# Re-generate enriched first_type CSVs from the already-computed
# rctd_doublet_results.csv — adds singlet_score, min_score, and
# score_delta (= singlet_score - min_score, the improvement of the
# full singlet fit over the single-type baseline).
#
# The existing per-method *_first_type.csv files only had
# cell_name, cell_id, first_type.  This script replaces them with
# versions that include the RCTD confidence columns.
#
# Run:
#   Rscript evaluation/scripts/enrich_first_type_scores.R
#
suppressPackageStartupMessages(library(data.table))

PROJECT_ROOT <- file.path(dirname(normalizePath(
  sub("^--file=", "", grep("^--file=", commandArgs(trailingOnly=FALSE), value=TRUE))
)), "..", "..")
if (!dir.exists(file.path(PROJECT_ROOT, "evaluation"))) PROJECT_ROOT <- "."

RCTD_CSV   <- file.path(PROJECT_ROOT, "evaluation", "reports", "rctd_ovarian",
                        "rctd_doublet_results.csv")
FIRST_DIR  <- file.path(PROJECT_ROOT, "evaluation", "reports", "rctd_ovarian",
                        "first_type")

stopifnot(file.exists(RCTD_CSV))
dir.create(FIRST_DIR, recursive=TRUE, showWarnings=FALSE)

all <- fread(RCTD_CSV)
message(sprintf("Read %d rows from %s", nrow(all), RCTD_CSV))
message("Methods found: ", paste(sort(unique(all$method)), collapse=", "))

# H5ad cell_name -> cell_id mapping (same for all methods; use RAW)
raw_h5ad <- fread(file.path(PROJECT_ROOT, "evaluation", "reports",
                            "rctd_ovarian", "mtx",
                            "ovarian_x1000-1800_y300-1100_raw_obs.csv"))
cell_name_id <- setNames(raw_h5ad$cell_id, raw_h5ad$cell_name)

for (m in sort(unique(all$method))) {
  sub <- all[method == m]
  sub[, score_delta := singlet_score - min_score]

  # Map cell_barcode -> cell_id
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
