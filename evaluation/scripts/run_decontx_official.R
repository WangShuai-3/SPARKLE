#!/usr/bin/env Rscript

# Standalone adapter around the official Bioconductor celda::decontX().
# All statistical estimation and decontamination is performed by
# celda::decontX(); this file only handles I/O and DNB-level aggregation.
#
# The input directory must contain:
#   counts_csc.h5   — CSC sparse matrix (genes × DNBs)
#   genes.tsv       — one gene name per line
#   cell_labels.tsv — one integer label per DNB column (-1 = empty/ambient)
#   n_cells.txt     — single integer, total number of cells
#
# The output directory will contain:
#   decont.h5           — dense corrected matrix (genes × cells)
#   diagnostics.tsv     — key-value diagnostics
#   session_info.txt    — R session info for reproducibility

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) {
    stop("Usage: run_decontx_official.R INPUT_DIR OUTPUT_DIR")
}

input_dir <- normalizePath(args[[1]], mustWork = TRUE)
output_dir <- args[[2]]
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

suppressPackageStartupMessages({
    library(Matrix)
    library(rhdf5)
    library(celda)
})

# ── Read input ──────────────────────────────────────────────────────────
input_h5 <- file.path(input_dir, "counts_csc.h5")
matrix_data <- h5read(input_h5, "data")
matrix_indices <- as.integer(h5read(input_h5, "indices"))
matrix_indptr <- as.integer(h5read(input_h5, "indptr"))
matrix_shape <- as.integer(h5read(input_h5, "shape"))
count_mat <- new(
    "dgCMatrix",
    x = as.numeric(matrix_data),
    i = matrix_indices,
    p = matrix_indptr,
    Dim = matrix_shape
)
rm(matrix_data, matrix_indices, matrix_indptr)

gene_names <- readLines(file.path(input_dir, "genes.tsv"), warn = FALSE)
cell_labels <- as.integer(readLines(file.path(input_dir, "cell_labels.tsv"), warn = FALSE))
n_cells <- as.integer(readLines(file.path(input_dir, "n_cells.txt"), warn = FALSE))

if (nrow(count_mat) != length(gene_names)) stop("Gene-name count does not match matrix")
if (ncol(count_mat) != length(cell_labels)) stop("Cell-label count does not match matrix columns")

rownames(count_mat) <- gene_names

# ── DNB → cell aggregation ─────────────────────────────────────────────
cell_col_idx <- which(cell_labels >= 0L)
cell_ids <- cell_labels[cell_col_idx]
indicator <- Matrix::sparseMatrix(
    i = cell_col_idx,
    j = cell_ids + 1L,
    x = 1.0,
    dims = c(ncol(count_mat), n_cells)
)
counts_cell <- count_mat %*% indicator
rownames(counts_cell) <- gene_names
colnames(counts_cell) <- paste0("Cell_", seq_len(n_cells) - 1L)

message(
    "Official celda::decontX ", as.character(packageVersion("celda")), ": ",
    nrow(counts_cell), " genes x ", n_cells, " cells"
)

# ── Run DecontX ─────────────────────────────────────────────────────────
# decontX normalizes counts internally; cells with zero library size make
# size factors non-positive and abort the run.  Drop them up front and
# re-insert zero columns afterwards so the output keeps n_cells columns.
library_sizes <- Matrix::colSums(counts_cell)
keep_cells <- library_sizes > 0
n_dropped <- sum(!keep_cells)
if (n_dropped > 0) {
    message("  Dropping ", n_dropped, " zero-library cells before decontX")
}
counts_run <- counts_cell[, keep_cells, drop = FALSE]

# DecontX requires >= 2 clusters; use kmeans on log1p expression
n_clusters <- max(2L, ncol(counts_run) %/% 20L)
log_mat <- log1p(as.matrix(t(counts_run)))
set.seed(42L)  # kmeans is random-initialised; fix seed for reproducibility
km <- stats::kmeans(log_mat, centers = n_clusters, iter.max = 100L)
clusters <- km$cluster

started <- proc.time()[["elapsed"]]

result <- celda::decontX(
    x = as.matrix(counts_run),
    z = clusters,
    maxIter = 200L,
    seed = 12345L,
    verbose = FALSE
)

runtime_seconds <- proc.time()[["elapsed"]] - started

corrected_run <- result$decontXcounts
mean_contamination <- mean(result$contamination)

# Re-insert dropped zero-library cells as all-zero columns
corrected <- matrix(0.0, nrow = nrow(counts_cell), ncol = n_cells)
rownames(corrected) <- gene_names
colnames(corrected) <- colnames(counts_cell)
corrected[, keep_cells] <- as.matrix(corrected_run)

message(
    "  mean contamination = ", round(mean_contamination, 4),
    ", runtime = ", round(runtime_seconds, 1), "s"
)

# ── Write output ────────────────────────────────────────────────────────
corrected_dense <- as.matrix(corrected)
output_h5 <- file.path(output_dir, "decont.h5")
if (file.exists(output_h5)) file.remove(output_h5)
h5createFile(output_h5)
n_genes <- nrow(corrected_dense)
n_out_cells <- ncol(corrected_dense)
h5createDataset(
    output_h5, "X",
    dims = c(n_genes, n_out_cells),
    chunk = c(min(128L, n_genes), min(1024L, n_out_cells)),
    storage.mode = "double", level = 4
)
batch_size <- 128L
for (start in seq.int(1L, n_genes, by = batch_size)) {
    end <- min(start + batch_size - 1L, n_genes)
    h5write(
        corrected_dense[start:end, , drop = FALSE],
        output_h5, "X",
        index = list(start:end, seq_len(n_out_cells))
    )
}
h5closeAll()

diagnostics <- data.frame(
    key = c(
        "official_package", "celda_version", "runtime_seconds",
        "mean_contamination", "n_genes", "n_dnb", "n_cells",
        "n_clusters", "max_iter", "seed", "n_zero_library_cells"
    ),
    value = c(
        "celda", as.character(packageVersion("celda")), runtime_seconds,
        mean_contamination, n_genes, ncol(count_mat), n_cells,
        n_clusters, 200L, 12345L, n_dropped
    )
)
write.table(diagnostics, file.path(output_dir, "diagnostics.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)
writeLines(capture.output(sessionInfo()), file.path(output_dir, "session_info.txt"))
message("Official DecontX completed in ", round(runtime_seconds, 1), " seconds")
