#!/usr/bin/env Rscript

# Standalone adapter around the official Bioconductor SpotClean package.
# All statistical estimation and decontamination is performed by
# SpotClean::createSlide() and SpotClean::spotclean(); this file only handles I/O.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 5) {
    stop(
        "Usage: run_spotclean_official.R INPUT_DIR OUTPUT_DIR MAXIT TOL ",
        "CANDIDATE_RADIUS_CSV"
    )
}

input_dir <- normalizePath(args[[1]], mustWork = TRUE)
output_dir <- args[[2]]
maxit <- as.integer(args[[3]])
tol <- as.numeric(args[[4]])
candidate_radius <- as.numeric(strsplit(args[[5]], ",", fixed = TRUE)[[1]])
if (is.na(maxit) || maxit <= 1) stop("MAXIT must be an integer greater than one")
if (is.na(tol) || tol < 0) stop("TOL must be non-negative")
if (anyNA(candidate_radius) || any(candidate_radius <= 0)) {
    stop("All candidate radii must be positive numbers")
}
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

suppressPackageStartupMessages({
    library(Matrix)
    library(rhdf5)
    library(S4Vectors)
    library(SummarizedExperiment)
    library(SpotClean)
})

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
gene_keep <- readLines(file.path(input_dir, "gene_keep.tsv"), warn = FALSE)
if (length(gene_keep) == 0) gene_keep <- NULL
slide_info <- read.csv(
    file.path(input_dir, "slide.csv"),
    stringsAsFactors = FALSE,
    check.names = FALSE
)
if (nrow(count_mat) != length(gene_names)) stop("Gene-name count does not match matrix")
if (ncol(count_mat) != nrow(slide_info)) stop("Slide rows do not match matrix columns")
rownames(count_mat) <- gene_names
colnames(count_mat) <- slide_info$barcode

message(
    "Official SpotClean ", as.character(packageVersion("SpotClean")), ": ",
    nrow(count_mat), " genes x ", ncol(count_mat), " spots (",
    sum(slide_info$tissue == 1), " tissue + ",
    sum(slide_info$tissue == 0), " background)"
)
started <- proc.time()[["elapsed"]]
slide_obj <- SpotClean::createSlide(
    count_mat = count_mat,
    slide_info = slide_info,
    gene_cutoff = 0,
    verbose = TRUE
)
rm(count_mat)
gc()

decont_obj <- SpotClean::spotclean(
    slide_obj,
    gene_keep = gene_keep,
    maxit = maxit,
    tol = tol,
    candidate_radius = candidate_radius,
    kernel = "gaussian",
    verbose = TRUE
)
runtime_seconds <- proc.time()[["elapsed"]] - started
decont <- SummarizedExperiment::assays(decont_obj)$decont
meta <- S4Vectors::metadata(decont_obj)

# A dense, chunked HDF5 dataset is straightforward for Python to stream into
# the existing h5ad template and avoids a very large text Matrix Market file.
output_h5 <- file.path(output_dir, "decont.h5")
if (file.exists(output_h5)) file.remove(output_h5)
h5createFile(output_h5)
n_genes <- nrow(decont)
n_tissue <- ncol(decont)
h5createDataset(
    output_h5,
    "X",
    dims = c(n_genes, n_tissue),
    chunk = c(min(128L, n_genes), min(1024L, n_tissue)),
    storage.mode = "double",
    level = 4
)
batch_size <- 128L
for (start in seq.int(1L, n_genes, by = batch_size)) {
    end <- min(start + batch_size - 1L, n_genes)
    h5write(
        as.matrix(decont[start:end, , drop = FALSE]),
        output_h5,
        "X",
        index = list(start:end, seq_len(n_tissue))
    )
}
h5closeAll()
writeLines(rownames(decont), file.path(output_dir, "genes.tsv"))
writeLines(colnames(decont), file.path(output_dir, "tissue_barcodes.tsv"))
writeLines(meta$decontaminated_genes, file.path(output_dir, "decontaminated_genes.tsv"))
write.table(
    data.frame(iteration = seq_along(meta$loglh), log_likelihood = meta$loglh),
    file.path(output_dir, "log_likelihood.tsv"),
    sep = "\t", quote = FALSE, row.names = FALSE
)
write.table(
    data.frame(barcode = colnames(decont), contamination_rate = meta$contamination_rate),
    file.path(output_dir, "contamination_rate.tsv"),
    sep = "\t", quote = FALSE, row.names = FALSE
)
diagnostics <- data.frame(
    key = c(
        "official_package", "spotclean_version", "runtime_seconds",
        "bleeding_rate", "distal_rate", "contamination_radius", "ARC_score",
        "n_input_genes", "n_output_genes", "n_tissue_spots",
        "n_background_spots", "n_decontaminated_genes", "em_iterations",
        "maxit", "tol", "candidate_radius", "kernel"
    ),
    value = c(
        "SpotClean", as.character(packageVersion("SpotClean")), runtime_seconds,
        meta$bleeding_rate, meta$distal_rate, meta$contamination_radius,
        meta$ARC_score, length(gene_names), n_genes, n_tissue,
        sum(slide_info$tissue == 0), length(meta$decontaminated_genes),
        length(meta$loglh), maxit, tol, paste(candidate_radius, collapse = ","),
        "gaussian"
    )
)
write.table(
    diagnostics,
    file.path(output_dir, "diagnostics.tsv"),
    sep = "\t", quote = FALSE, row.names = FALSE
)
writeLines(capture.output(sessionInfo()), file.path(output_dir, "session_info.txt"))
message("Official SpotClean completed in ", round(runtime_seconds, 1), " seconds")
