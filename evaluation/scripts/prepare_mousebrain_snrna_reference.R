#!/usr/bin/env Rscript
# Extract cell-group pseudobulk profiles from the MouseBrain snRNA-seq Seurat object.
#
# Usage:
#   Rscript evaluation/scripts/prepare_mousebrain_snrna_reference.R \
#       [path/to/mouseBrain.snRNAseq.308Clusters.seurat.20230607.rds] \
#       [output.csv] \
#       [group_column]
#
#   group_column: metadata column to average by. Default is "Cell_group" (50 groups),
#   which is finer than "Cell_subclass" (18 groups) but coarser than "Cell_cluster"
#   (308 clusters).
#
# Requires the Seurat package and enough memory to load the ~23 GB RDS file.

library(Seurat)

args <- commandArgs(trailingOnly = TRUE)
rds_path <- ifelse(length(args) >= 1, args[1], "evaluation/data/mousebrain/mouseBrain.snRNAseq.308Clusters.seurat.20230607.rds")
out_path <- ifelse(length(args) >= 2, args[2], "evaluation/data/mousebrain/snrna_cell_group_pseudobulk.csv")
group_col <- ifelse(length(args) >= 3, args[3], "Cell_group")

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

# Validate requested grouping column.
meta_cols <- colnames(obj@meta.data)
if (!(group_col %in% meta_cols)) {
  stop("Cannot find metadata column: ", group_col,
       ". Available columns: ", paste(meta_cols, collapse = ", "))
}
message("Using grouping column: ", group_col)

# AverageExpression by group.
# Try Seurat v5 'layer' argument first, fall back to v4 'slot'.
avg <- tryCatch(
  AverageExpression(obj, group.by = group_col, layer = "data")[[assay]],
  error = function(e) {
    AverageExpression(obj, group.by = group_col, slot = "data")[[assay]]
  }
)

avg_df <- as.data.frame(avg)
message("Reference shape: ", nrow(avg_df), " genes x ", ncol(avg_df), " groups")
write.csv(avg_df, out_path)
message("Saved: ", out_path)
