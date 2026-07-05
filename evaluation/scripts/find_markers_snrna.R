#!/usr/bin/env Rscript
# Find markers in snRNA reference using Wilcoxon rank-sum test.
# Memory-efficient approach: down-sample cells from metadata first, then
# extract only the required gene x cell count sub-matrix and build a slim
# Seurat object for FindAllMarkers.

suppressPackageStartupMessages({
  library(Seurat)
  library(data.table)
})

args <- commandArgs(trailingOnly=TRUE)
sparkle_genes_file <- ifelse(length(args) >= 1, args[1], "evaluation/reports/mousebrain_eval/sparkle_corrected_genes.txt")
rds_file <- ifelse(length(args) >= 2, args[2], "evaluation/data/mousebrain/mouseBrain.snRNAseq.308Clusters.seurat.20230607.rds")
spatial_groups_file <- ifelse(length(args) >= 3, args[3], "evaluation/data/mousebrain/snrna_cell_group_pseudobulk.csv")
output_file <- ifelse(length(args) >= 4, args[4], "evaluation/reports/mousebrain_eval/gene_level_analysis/snrna_wilcox_markers_sparkle_genes.csv")
max_cells_per_group <- ifelse(length(args) >= 5, as.integer(args[5]), 500)
top_n <- ifelse(length(args) >= 6, as.integer(args[6]), 100)

cat("Loading sparkle corrected genes from", sparkle_genes_file, "\n")
gene_list <- scan(sparkle_genes_file, what="character", quiet=TRUE)
cat("Using", length(gene_list), "genes\n")

cat("Loading spatial group names from", spatial_groups_file, "\n")
if (tolower(tools::file_ext(spatial_groups_file)) == "txt") {
  spatial_groups <- scan(spatial_groups_file, what="character", quiet=TRUE, sep="\n")
} else {
  spatial_ref <- fread(spatial_groups_file)
  spatial_groups <- colnames(spatial_ref)[2:ncol(spatial_ref)]
}
spatial_groups <- trimws(spatial_groups)
spatial_groups <- spatial_groups[spatial_groups != ""]
cat("Spatial groups:", length(spatial_groups), "\n")
cat("First few groups:", head(spatial_groups, 5), "\n")

# Map spatial group names (hyphens) to snRNA Cell_group names (underscores)
spatial_groups_in_rna <- gsub("-", "_", spatial_groups)
cat("Mapped to snRNA names:", head(spatial_groups_in_rna, 5), "\n")

cat("Loading RDS from", rds_file, "\n")
seu <- readRDS(rds_file)
cat("RDS dims:", nrow(seu), "genes x", ncol(seu), "cells\n")

# Use RNA assay (raw counts) and drop everything else to save memory
DefaultAssay(seu) <- "RNA"
seu <- DietSeurat(seu, assays = "RNA", counts = TRUE, data = TRUE, scale.data = FALSE,
                  features = NULL, dimreducs = NULL, layers = c("counts", "data"))
cat("DietSeurat dims:", nrow(seu), "genes x", ncol(seu), "cells\n")

# Set active identity to Cell_group
seu@meta.data$Cell_group <- as.character(seu@meta.data[["Cell_group"]])
Idents(seu) <- seu@meta.data$Cell_group
cat("Identity levels:", length(levels(Idents(seu))), "\n")

# Keep only cells whose Cell_group is present in spatial data
meta <- seu@meta.data
cells_keep <- rownames(meta)[meta$Cell_group %in% spatial_groups_in_rna]
cat("Cells in shared groups:", length(cells_keep), "\n")

# Down-sample per group based on metadata
set.seed(42)
cells_downsampled <- unlist(lapply(spatial_groups_in_rna, function(g) {
  g_cells <- cells_keep[meta[cells_keep, "Cell_group"] == g]
  if (length(g_cells) > max_cells_per_group) {
    sample(g_cells, max_cells_per_group)
  } else {
    g_cells
  }
}))
cat("Down-sampled cells:", length(cells_downsampled), "\n")

# Restrict to sparkle corrected genes present in object
gene_list <- intersect(gene_list, rownames(seu))
cat("Genes present in RDS:", length(gene_list), "\n")

# Extract the small count sub-matrix directly (genes x cells)
cat("Extracting count sub-matrix...\n")
counts_sub <- GetAssayData(seu, assay = "RNA", layer = "counts")[gene_list, cells_downsampled]
cat("Sub-matrix dims:", nrow(counts_sub), "genes x", ncol(counts_sub), "cells\n")

# Build slim Seurat object from sub-matrix
meta_sub <- meta[cells_downsampled, c("Cell_group"), drop = FALSE]
seu_sub <- CreateSeuratObject(counts = counts_sub, meta.data = meta_sub)
Idents(seu_sub) <- seu_sub@meta.data$Cell_group
cat("Slim object identity levels:", length(levels(Idents(seu_sub))), "\n")

# Normalize for FindAllMarkers (Wilcoxon uses data slot)
seu_sub <- NormalizeData(seu_sub, verbose = FALSE)

# Optional: clean up original large object to free memory
rm(seu, counts_sub)
invisible(gc(verbose = FALSE))

cat("Running Wilcoxon FindAllMarkers...\n")
markers <- FindAllMarkers(
  seu_sub,
  only.pos = TRUE,
  method = "wilcox",
  min.pct = 0.1,
  logfc.threshold = 0,
  return.thresh = 0.05,
  verbose = TRUE
)

# Keep top N per group by score = -log10(padj) * avg_log2FC
markers$score <- -log10(markers$p_val_adj + 1e-300) * markers$avg_log2FC
markers <- as.data.table(markers)
markers <- markers[order(-score), .SD[1:min(.N, top_n)], by = cluster]

# Map cluster names back to hyphenated spatial group names for direct comparison
name_map <- setNames(spatial_groups, spatial_groups_in_rna)
markers$cluster <- name_map[as.character(markers$cluster)]

fwrite(markers, output_file)
cat("Saved:", output_file, "\n")
cat("Done.\n")
