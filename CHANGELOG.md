# Changelog

All notable changes to `stambient` are documented in this file.

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
