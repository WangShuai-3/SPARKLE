#!/usr/bin/env Rscript
###############################################################################
# Official CellChat v2 reference analysis for the ovarian scFFPE single-cell
# dataset.  The communication estimator matches the ovarian spatial analysis
# except that spatial distance/contact weighting is disabled for dissociated
# cells.
#
# Usage:
#   /home/shuaiwang/miniconda3/envs/r-env/bin/Rscript \
#     evaluation/scripts/run_cellchat_ovarian_scrna_reference.R
#
# Optional environment variables:
#   CELLCHAT_NBOOT=20
#   CELLCHAT_MIN_CELLS=10
#   CELLCHAT_FORCE_REFERENCE=1
###############################################################################

suppressPackageStartupMessages({
  library(CellChat)
  library(future)
  library(Matrix)
})

args <- commandArgs(trailingOnly = FALSE)
script_arg <- args[grep("^--file=", args)]
if (length(script_arg) > 0) {
  script_path <- normalizePath(sub("^--file=", "", script_arg))
  PROJECT_ROOT <- normalizePath(file.path(dirname(script_path), "..", ".."))
} else {
  PROJECT_ROOT <- normalizePath(".")
}

INPUT_DIR <- file.path(PROJECT_ROOT, "evaluation", "data", "ovarian")
H5_FILE <- file.path(
  INPUT_DIR, "17k_Ovarian_Cancer_scFFPE_count_filtered_feature_bc_matrix.h5"
)
ANNOTATION_FILE <- file.path(
  INPUT_DIR, "FLEX_Ovarian_Barcode_Cluster_Annotation.csv"
)
OUT_DIR <- file.path(
  PROJECT_ROOT, "evaluation", "reports", "ovarian_eval", "cellchat",
  "official_reference"
)
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

OUTPUT_ALL <- file.path(OUT_DIR, "scRNA_cellchat_official_all_tested.csv")
OUTPUT_SIG <- file.path(OUT_DIR, "scRNA_cellchat_official_significant.csv")
OUTPUT_ROUTES <- file.path(
  OUT_DIR, "scRNA_cellchat_official_COL1A2_SDC4.csv"
)
OUTPUT_PARAMETERS <- file.path(OUT_DIR, "run_parameters.csv")
OUTPUT_COUNTS <- file.path(OUT_DIR, "cell_type_counts.csv")

NBOOT <- as.integer(Sys.getenv("CELLCHAT_NBOOT", "20"))
MIN_CELLS <- as.integer(Sys.getenv("CELLCHAT_MIN_CELLS", "10"))
SEED <- 42L
FORCE <- Sys.getenv("CELLCHAT_FORCE_REFERENCE", "0") == "1"

if (!FORCE && file.exists(OUTPUT_ALL) && file.exists(OUTPUT_PARAMETERS)) {
  cached <- read.csv(OUTPUT_PARAMETERS, stringsAsFactors = FALSE)
  cache_matches <- nrow(cached) == 1 &&
    cached$nboot[1] == NBOOT &&
    cached$min_cells_per_type[1] == MIN_CELLS &&
    cached$seed[1] == SEED &&
    cached$expression_summary[1] == "truncatedMean" &&
    cached$trim[1] == 0.1
  if (cache_matches) {
    cat(sprintf("[reference] reusing %s\n", OUTPUT_ALL))
    quit(save = "no", status = 0)
  }
}

if (!file.exists(H5_FILE)) stop(sprintf("Missing 10x H5: %s", H5_FILE))
if (!file.exists(ANNOTATION_FILE)) {
  stop(sprintf("Missing annotation: %s", ANNOTATION_FILE))
}
if (!requireNamespace("Seurat", quietly = TRUE)) {
  stop("The Seurat package is required to read the 10x H5 file")
}

cat("[reference] reading ovarian scFFPE counts ...\n")
counts <- Seurat::Read10X_h5(
  H5_FILE, use.names = TRUE, unique.features = TRUE
)
if (is.list(counts)) {
  if (!"Gene Expression" %in% names(counts)) {
    stop("10x H5 contains multiple assays but no 'Gene Expression' matrix")
  }
  counts <- counts[["Gene Expression"]]
}
counts <- as(counts, "CsparseMatrix")
if (any(!is.finite(counts@x)) || any(counts@x < 0)) {
  stop("Single-cell count matrix contains negative or non-finite values")
}
cat(sprintf(
  "[reference] raw matrix: %d genes x %d cells (%d nonzero entries)\n",
  nrow(counts), ncol(counts), length(counts@x)
))

annotation <- read.csv(ANNOTATION_FILE, stringsAsFactors = FALSE)
required_columns <- c("Barcode", "Cell.Annotation")
if (!all(required_columns %in% colnames(annotation))) {
  stop(sprintf(
    "Annotation must contain columns: %s",
    paste(required_columns, collapse = ", ")
  ))
}
annotation <- annotation[
  !is.na(annotation$Cell.Annotation) & annotation$Cell.Annotation != "", ,
  drop = FALSE
]
annotation$Cell.Annotation <- gsub(
  "/", "-", annotation$Cell.Annotation, fixed = TRUE
)
if (anyDuplicated(annotation$Barcode)) {
  stop("Annotation contains duplicated barcodes")
}

shared_cells <- intersect(colnames(counts), annotation$Barcode)
if (length(shared_cells) == 0) {
  stop("No shared barcodes between the 10x matrix and annotation")
}
counts <- counts[, shared_cells, drop = FALSE]
annotation_lookup <- setNames(
  annotation$Cell.Annotation, annotation$Barcode
)
labels <- unname(annotation_lookup[colnames(counts)])
if (any(is.na(labels))) stop("Missing labels after barcode alignment")

cell_type_counts <- as.data.frame(
  table(labels), stringsAsFactors = FALSE
)
colnames(cell_type_counts) <- c("cell_type", "n_cells")
cell_type_counts <- cell_type_counts[
  order(cell_type_counts$n_cells, decreasing = TRUE), ,
  drop = FALSE
]
write.csv(cell_type_counts, OUTPUT_COUNTS, row.names = FALSE)
cat(sprintf(
  "[reference] retained %d cells across %d annotated cell types\n",
  ncol(counts), length(unique(labels))
))
print(cell_type_counts, row.names = FALSE)

meta <- data.frame(
  labels = labels, row.names = colnames(counts), stringsAsFactors = FALSE
)

plan("sequential")
options(future.globals.maxSize = 8 * 1024^3)

cat("[reference] normalizing and creating official CellChat object ...\n")
normalized <- CellChat::normalizeData(counts)
cellchat <- createCellChat(
  object = normalized,
  meta = meta,
  group.by = "labels",
  datatype = "RNA"
)
rm(normalized)
gc(verbose = FALSE)

data("CellChatDB.human", package = "CellChat")
cellchat@DB <- CellChatDB.human

cat("[reference] identifying expressed genes and interactions ...\n")
cellchat <- subsetData(cellchat)
cellchat <- identifyOverExpressedGenes(cellchat)
cellchat <- identifyOverExpressedInteractions(cellchat)
if (nrow(cellchat@LR$LRsig) == 0) {
  stop("identifyOverExpressedInteractions retained no ligand-receptor pairs")
}
cat(sprintf(
  "[reference] retained %d overexpressed ligand-receptor pairs\n",
  nrow(cellchat@LR$LRsig)
))

cat("[reference] computing non-spatial communication probabilities ...\n")
cellchat <- computeCommunProb(
  cellchat,
  type = "truncatedMean",
  trim = 0.1,
  distance.use = FALSE,
  nboot = NBOOT,
  seed.use = SEED
)
cellchat <- filterCommunication(cellchat, min.cells = MIN_CELLS)

# Export all tested edges; significance is retained only as metadata and is not
# used by the focused COL1A2-SDC4 comparison figure.
all_tested <- subsetCommunication(cellchat, thresh = 1.01)
significant <- all_tested[
  !is.na(all_tested$pval) & all_tested$pval < 0.05, ,
  drop = FALSE
]
col1a2_sdc4 <- all_tested[
  all_tested$ligand == "COL1A2" & all_tested$receptor == "SDC4", ,
  drop = FALSE
]

write.csv(all_tested, OUTPUT_ALL, row.names = FALSE)
write.csv(significant, OUTPUT_SIG, row.names = FALSE)
write.csv(col1a2_sdc4, OUTPUT_ROUTES, row.names = FALSE)
write.csv(
  data.frame(
    n_cells = ncol(counts),
    n_genes = nrow(counts),
    n_cell_types = length(unique(labels)),
    nboot = NBOOT,
    seed = SEED,
    min_cells_per_type = MIN_CELLS,
    expression_summary = "truncatedMean",
    trim = 0.1,
    distance_use = FALSE,
    database = "CellChatDB.human",
    cellchat_version = as.character(packageVersion("CellChat")),
    stringsAsFactors = FALSE
  ),
  OUTPUT_PARAMETERS,
  row.names = FALSE
)

cat(sprintf(
  "[reference] saved %d all-tested and %d significant interactions\n",
  nrow(all_tested), nrow(significant)
))
cat("\n[reference] official COL1A2-SDC4 routes:\n")
print(
  col1a2_sdc4[
    , intersect(
      c("source", "target", "ligand", "receptor", "prob", "pval"),
      colnames(col1a2_sdc4)
    ),
    drop = FALSE
  ],
  row.names = FALSE
)
cat(sprintf("[reference] complete: %s\n", OUT_DIR))
