#!/usr/bin/env Rscript

# Standalone adapter around the official CRAN SoupX package.
# All statistical estimation and decontamination is performed by
# SoupX::SoupChannel(), SoupX::autoEstCont(), and SoupX::adjustCounts();
# this file only handles I/O and DNB-level aggregation.
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
if (length(args) < 2) {
    stop("Usage: run_soupx_official.R INPUT_DIR OUTPUT_DIR ",
         "[TFIDF_MIN] [SOUP_QUANTILE] [N_CLUSTERS] [QMK_FDR] [FORCE_ACCEPT]")
}

input_dir <- normalizePath(args[[1]], mustWork = TRUE)
output_dir <- args[[2]]
tfidf_min <- if (length(args) >= 3) as.numeric(args[[3]]) else 1.0
soup_quantile <- if (length(args) >= 4) as.numeric(args[[4]]) else 0.9
# Number of clusters for marker-gene detection.  SoupX's autoEstCont uses
# quickMarkers with a strict hypergeometric FDR; on low-heterogeneity data
# (e.g. synthetic benchmarks with a handful of cell types) too many clusters
# dilute the marker signal and yield zero markers.  6 works well for the
# synthetic scenarios; override when the data have many real clusters.
n_clusters_arg <- if (length(args) >= 5) as.integer(args[[5]]) else NA_integer_
# FDR threshold for the hypergeometric marker-enrichment test inside
# quickMarkers.  SoupX::autoEstCont hardcodes quickMarkers' FDR at 0.01;
# relax it (mirroring the Python port's monkey-patch) by overriding
# quickMarkers in the SoupX namespace for the duration of this session.
qmk_fdr <- if (length(args) >= 6) as.numeric(args[[6]]) else 0.01
force_accept <- if (length(args) >= 7) toupper(args[[7]]) == "TRUE" else FALSE
# Upper bound of the contamination search range; mousebrain-scale data can
# need >0.8 for estimateNonExpressingCells to find usable cells.
cont_max <- if (length(args) >= 8) as.numeric(args[[8]]) else 0.8
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

suppressPackageStartupMessages({
    library(Matrix)
    library(rhdf5)
    library(SoupX)
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
# tod = table of droplets (all DNBs, genes × DNBs)
# toc = table of counts  (genes × cells, aggregated from DNBs)
tod <- count_mat

cell_col_idx <- which(cell_labels >= 0L)
cell_ids <- cell_labels[cell_col_idx]
# Efficient aggregation: build a (DNBs × cells) indicator matrix and multiply
indicator <- Matrix::sparseMatrix(
    i = cell_col_idx,
    j = cell_ids + 1L,
    x = 1.0,
    dims = c(ncol(tod), n_cells)
)
toc <- tod %*% indicator
rownames(toc) <- gene_names
colnames(toc) <- paste0("Cell_", seq_len(n_cells) - 1L)

message(
    "Official SoupX ", as.character(packageVersion("SoupX")), ": ",
    nrow(tod), " genes x ", ncol(tod), " DNBs -> ", n_cells, " cells"
)

# ── Run SoupX pipeline ─────────────────────────────────────────────────
started <- proc.time()[["elapsed"]]

sc <- SoupX::SoupChannel(tod, toc, calcSoupProfile = TRUE)

# Clustering: quickMarkers requires clusters; use a simple kmeans on log1p.
# Default: few clusters so hypergeometric marker tests retain power.
n_clusters <- if (is.na(n_clusters_arg)) {
    min(6L, max(3L, n_cells %/% 20L))
} else {
    n_clusters_arg
}
set.seed(42L)  # kmeans is random-initialised; fix seed for reproducibility
log_toc <- log1p(as.matrix(t(toc)))
km <- stats::kmeans(log_toc, centers = n_clusters, iter.max = 100L)
sc <- SoupX::setClusters(sc, km$cluster)

# Estimate contamination
if (qmk_fdr != 0.01) {
    ns <- asNamespace("SoupX")
    orig_quickMarkers <- get("quickMarkers", envir = ns)
    patched_quickMarkers <- function(toc, clusters, N = 10, FDR = qmk_fdr, ...) {
        orig_quickMarkers(toc, clusters, N = N, FDR = qmk_fdr, ...)
    }
    unlockBinding("quickMarkers", ns)
    assign("quickMarkers", patched_quickMarkers, envir = ns)
    lockBinding("quickMarkers", ns)
}
sc <- SoupX::autoEstCont(
    sc,
    tfidfMin = tfidf_min,
    soupQuantile = soup_quantile,
    doPlot = FALSE,
    forceAccept = force_accept,
    contaminationRange = c(0.01, cont_max)
)

# Adjust counts
corrected <- SoupX::adjustCounts(sc, roundToInt = FALSE)

runtime_seconds <- proc.time()[["elapsed"]] - started

# Clip negatives (same as Python baseline)
corrected[corrected < 0] <- 0

rho <- sc$metaData$rho[1]
message("  rho = ", round(rho, 4), ", runtime = ", round(runtime_seconds, 1), "s")

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
        "official_package", "soupx_version", "runtime_seconds",
        "rho", "n_genes", "n_dnb", "n_cells", "n_clusters",
        "n_empty_dnb", "tfidf_min", "soup_quantile", "n_clusters_arg",
        "qmk_fdr", "force_accept", "cont_max"
    ),
    value = c(
        "SoupX", as.character(packageVersion("SoupX")), runtime_seconds,
        rho, n_genes, ncol(tod), n_cells, n_clusters,
        sum(cell_labels < 0L), tfidf_min, soup_quantile, n_clusters_arg,
        qmk_fdr, force_accept, cont_max
    )
)
write.table(diagnostics, file.path(output_dir, "diagnostics.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)
writeLines(capture.output(sessionInfo()), file.path(output_dir, "session_info.txt"))
message("Official SoupX completed in ", round(runtime_seconds, 1), " seconds")
