#!/usr/bin/env Rscript
# Extract cell-subclass pseudobulk profiles from the MouseBrain snRNA-seq Seurat object.
#
# Usage:
#   Rscript evaluation/scripts/prepare_mousebrain_snrna_reference.R \
#       [path/to/mouseBrain.snRNAseq.308Clusters.seurat.20230607.rds] \
#       [output.csv]
#
# Requires the Seurat package and enough memory to load the ~23 GB RDS file.

library(Seurat)

args <- commandArgs(trailingOnly = TRUE)
rds_path <- ifelse(length(args) >= 1, args[1], "evaluation/data/mousebrain/mouseBrain.snRNAseq.308Clusters.seurat.20230607.rds")
out_path <- ifelse(length(args) >= 2, args[2], "evaluation/data/mousebrain/snrna_cell_subclass_pseudobulk.csv")

message("Loading RDS: ", rds_path)
obj <- readRDS(rds_path)

# Normalize if a normalized data layer/slot is not already present.
# Seurat v5 uses @layers; Seurat v4 uses @data in the assay slot.
assay <- DefaultAssay(obj)
norm_present <- FALSE
if (.hasSlot(obj[[assay]], "layers")) {
  norm_present <- "data" %in% names(obj[[assay]]@layers)
} else if (.hasSlot(obj[[assay]], "data")) {
  norm_present <- length(obj[[assay]]@data) > 0
}
if (!norm_present) {
  message("Normalized data not found; running NormalizeData ...")
  obj <- NormalizeData(obj)
}

# Try to locate a cell-subclass metadata column.
meta_cols <- colnames(obj@meta.data)
subclass_col <- grep("subclass", meta_cols, ignore.case = TRUE, value = TRUE)[1]
if (is.na(subclass_col)) {
  subclass_col <- "cell_subclass"
}
if (!(subclass_col %in% meta_cols)) {
  stop("Cannot find a cell_subclass metadata column. Available columns: ", paste(meta_cols, collapse = ", "))
}
message("Using grouping column: ", subclass_col)

# AverageExpression by subclass.
# Try Seurat v5 'layer' argument first, fall back to v4 'slot'.
avg <- tryCatch(
  AverageExpression(obj, group.by = subclass_col, layer = "data")[[assay]],
  error = function(e) {
    AverageExpression(obj, group.by = subclass_col, slot = "data")[[assay]]
  }
)

avg_df <- as.data.frame(avg)
message("Reference shape: ", nrow(avg_df), " genes x ", ncol(avg_df), " subclasses")
write.csv(avg_df, out_path)
message("Saved: ", out_path)
