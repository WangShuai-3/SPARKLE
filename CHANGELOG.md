# Changelog

All notable changes to `stambient` are documented in this file.

## [Unreleased]

### Changed

- CPU path memory: the estimation stages (λ search, α estimation,
  correction) now process genes in batches like the GPU path, and the
  aggregated gene × cell / gene × bin matrices stay sparse with only the
  active batch densified. Peak RAM no longer grows with the number of
  genes, which makes all-gene (`n_high_genes=None`) runs on large datasets
  feasible; on a 25k-gene × 1.1M-DNB synthetic benchmark peak RSS dropped
  from 27.8 GiB to 7.4 GiB (now dominated by the dense corrected output)
  while runtime improved 2.4x. New `cpu_gene_batch_size` parameter
  (default: adaptive, ~1 GiB dense working set per batch); results are
  unchanged up to floating-point rounding from the batched sparse products.

### Added

- Native BGI GEF I/O (`stambient.gef_io`, h5py-only; stereopy stays
  optional): `read_cellbin_gef`/`load_cellbin_gef` reads official
  cellbin GEF files (current `/cellBin` layout of SAW >= 7.1 and the
  legacy `cellExp`/`cellData` layout), `read_bin_gef` reads raw/tissue
  GEF DNB layers with optional µm-region cropping, and
  `load_bgi_gef` pairs the two so DNBs are labelled from the cell
  border polygons (out-of-mask DNBs become the `-1` ambient layer
  required by the SPARKLE input contract).
- `save_cellbin_gef` writes corrected gene-by-cell matrices back to an
  official-layout cellbin GEF readable by stereopy and StereoMap; exact
  floating-point values are preserved in a non-standard `/sparkleInfo`
  group restored automatically by the reader.
- Optional dependency extras: `gef` (h5py) and `stereopy`
  (Python < 3.11).
- Validated against the official SAW 8.1 demo dataset (mouse whole
  brain, `C04042E3`), including value-exact round-trips through
  gefpy's `CgefR` reader.

## [0.1.3] - 2026-09-08

### Changed

- Refactor: split the monolithic `cell_pipeline` module into per-stage
  modules (`aggregation`, `alpha_estimation`, `binning`, `correction`,
  `graphs`, `lambda_search`, `weights`). No behaviour change.

### Evaluation

- Add official-R SoupX/DecontX baselines and assorted manuscript figure
  scripts; no impact on the library.

## [0.1.2] - 2026-08-10

### Changed

- Synthetic benchmark: replace the unreasonable 95% UMI dropout with a
  reasonable 20% in all S1–S10 scenarios, keeping the count level high
  enough for the benchmark to remain informative.
- SoupX synthetic protocol: cluster with the true cell-type count and relax
  the marker-detection thresholds (`tfidfMin=0.05`, `soupQuantile=0.5`,
  quickMarkers FDR=0.1); add `fdr` and `force_accept` options to
  `run_soupx` so an extremely high estimated contamination is accepted
  instead of failing the run.
- Synthetic evaluation metrics: switch RMSE to per-cell library-normalised
  `log1p(CP10K)` RMSE; the two primary synthetic metrics are now
  `log1p(CP10K)` RMSE and cell-wise Pearson R².

### Documentation

- Update the evaluation README to describe the current synthetic pipeline
  (20% dropout, tuned SoupX protocol, and the two primary metrics).

## [0.1.1] - 2026-08-09

### Changed

- Bump version to 0.1.1 and publish the wheel + sdist to PyPI (92 tests
  pass).

## [0.1.0] - Initial release

- Evidence-constrained correction of local RNA leakage in high-resolution
  spatial transcriptomics (`stambient.SPARKLE`).
