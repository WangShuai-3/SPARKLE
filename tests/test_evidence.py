"""Tests for the spatial-CV evidence score (2.x Phase 3)."""

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from stambient import SPARKLE
from stambient.evidence import (
    poisson_deviance,
    spatial_block_fold_ids,
    spatial_cv_evidence,
)
from stambient.spatial import distance_graph_to_weights

from tests.test_legacy_freeze import build_freeze_fixture

FIXTURE_KW = dict(
    bin_size=10.0,
    max_radius=100.0,
    lambda_grid=[10.0, 25.0, 50.0],
    r2_threshold=0.01,
    verbose=False,
)


def test_fold_ids_cover_and_deterministic():
    rng = np.random.RandomState(0)
    coords = rng.rand(200, 2) * 100.0
    folds = spatial_block_fold_ids(coords, block_size=25.0, n_splits=2)
    assert folds.shape == (200,)
    assert set(np.unique(folds)) <= {0, 1, 2, 3}
    assert len(np.unique(folds)) == 4
    np.testing.assert_array_equal(
        folds, spatial_block_fold_ids(coords, block_size=25.0, n_splits=2)
    )


def test_poisson_deviance_nonnegative_and_zero_at_saturation():
    rng = np.random.RandomState(1)
    y = rng.poisson(5.0, (30, 4)).astype(np.float64)
    mu = rng.gamma(2.0, 2.0, (30, 4))
    assert (poisson_deviance(y, mu) >= 0).all()
    np.testing.assert_allclose(poisson_deviance(y, y), 0.0, atol=1e-10)


def _mini_leakage_dataset(seed=3):
    """Aggregated inputs with known local-leakage truth for 8 genes.

    Genes 0-3 have genuine local leakage (rho=0.4); genes 4-7 are pure
    diffuse (rho=0).  Bins sit on a spatial grid so the block CV has
    spatially separated folds.
    """
    rng = np.random.RandomState(seed)
    n_genes, n_cells = 8, 6
    cell_centroids = np.array(
        [[20, 20], [20, 60], [60, 20], [60, 60], [100, 20], [100, 60]],
        dtype=np.float64,
    )
    cell_areas = np.full(n_cells, 120.0)
    cell_expr = rng.poisson(2.0, (n_genes, n_cells)).astype(np.float64)
    cell_expr[:4] += rng.poisson(60.0, (4, n_cells))
    # Bin grid interleaved with cells (background only).
    gx, gy = np.meshgrid(np.arange(0, 130, 10.0), np.arange(0, 70, 10.0))
    bin_coords = np.column_stack([gx.ravel(), gy.ravel()])
    n_bins = len(bin_coords)
    bin_areas = np.full(n_bins, 40.0)
    # Distances bin->cell, truncated.
    d = np.linalg.norm(
        bin_coords[:, None, :] - cell_centroids[None, :, :], axis=2
    )
    d[d > 60.0] = 0.0
    e2c = csr_matrix(d)
    lam = 25.0
    W = distance_graph_to_weights(e2c, lam, "exponential")
    sources = cell_expr / cell_areas[None, :]
    S = W.dot(sources.T)  # [bins x genes]
    true_rho = np.array([0.4] * 4 + [0.0] * 4)
    true_beta = np.full(n_genes, 0.05)
    mu = bin_areas[:, None] * (true_beta[None, :] + true_rho[None, :] * S)
    y = rng.poisson(mu).astype(np.float64)
    empty_bin_expr = y.T  # [genes x bins]
    return dict(
        gene_indices=np.arange(n_genes),
        cell_expr=cell_expr,
        cell_areas=cell_areas,
        empty_bin_expr=empty_bin_expr,
        empty_bin_areas=bin_areas,
        empty_bin_coords=bin_coords,
        empty_to_cell_distances=e2c,
        lam_weights=lam,
        true_rho=true_rho,
    )


@pytest.mark.parametrize("fit_diffuse", [True, False])
def test_evidence_higher_for_true_leakage_genes(fit_diffuse):
    ds = _mini_leakage_dataset()
    out = spatial_cv_evidence(
        gene_indices=ds["gene_indices"],
        cell_expr=ds["cell_expr"],
        cell_areas=ds["cell_areas"],
        empty_bin_expr=ds["empty_bin_expr"],
        empty_bin_areas=ds["empty_bin_areas"],
        empty_bin_coords=ds["empty_bin_coords"],
        empty_to_cell_distances=ds["empty_to_cell_distances"],
        lam_weights=ds["lam_weights"],
        fit_diffuse=fit_diffuse,
        block_size=20.0,
        gpu=None,
        storage_dtype=None,
        reduction_dtype=None,
    )
    leak = ds["true_rho"] > 0
    assert (out["deviance_null"] >= 0).all()
    # True leakage genes must show clearly higher held-out evidence.
    assert out["evidence"][leak].mean() > out["evidence"][~leak].mean() + 0.01
    assert (out["delta_deviance"][leak] > 0).mean() >= 0.75


def _evidence_pipeline(weight, fixture_kw=FIXTURE_KW):
    expr, coords, labels = build_freeze_fixture()
    return SPARKLE(
        inference_mode="latent",
        latent_refit_rounds=1,
        observation_model="poisson",
        fit_diffuse=True,
        evidence_mode="cv_deviance",
        evidence_weight=weight,
        **fixture_kw,
    ).fit_transform(expr, coords, labels)


def test_pipeline_evidence_hard_vs_linear_weights():
    c_hard, d_hard = _evidence_pipeline("hard")
    c_lin, d_lin = _evidence_pipeline("linear")
    w_hard = d_hard["correction_weights"]
    w_lin = d_lin["correction_weights"]
    assert set(np.unique(w_hard)) <= {0.0, 1.0}
    assert (w_lin >= 0).all() and (w_lin <= 1).all()
    # With threshold 0, the linear weight can only be weaker than hard.
    assert (w_lin <= w_hard + 1e-12).all()
    assert (c_hard >= 0).all() and (c_lin >= 0).all()
    # Same genes are gated (linear weight is positive exactly where hard is).
    assert ((w_hard > 0) == (w_lin > 0)).all()
    # Weaker weights subtract less in aggregate.
    assert c_lin.sum() >= c_hard.sum() - 1e-6
    ev = d_hard["evidence"]
    assert ev["deviance_null"].shape == ev["evidence"].shape


def test_pipeline_evidence_requires_poisson():
    expr, coords, labels = build_freeze_fixture()
    with pytest.raises(ValueError, match="observation_model='poisson'"):
        SPARKLE(
            inference_mode="latent",
            evidence_mode="cv_deviance",
            **FIXTURE_KW,
        ).fit_transform(expr, coords, labels)
    with pytest.raises(ValueError, match="evidence_mode"):
        SPARKLE(evidence_mode="bogus", **FIXTURE_KW)
    with pytest.raises(ValueError, match="evidence_weight"):
        SPARKLE(
            inference_mode="latent",
            observation_model="poisson",
            evidence_mode="cv_deviance",
            evidence_weight="bogus",
            **FIXTURE_KW,
        ).fit_transform(expr, coords, labels)


def test_pipeline_evidence_deterministic():
    c1, _ = _evidence_pipeline("linear")
    c2, _ = _evidence_pipeline("linear")
    np.testing.assert_array_equal(c1, c2)


@pytest.mark.skipif(
    not pytest.importorskip("torch").cuda.is_available(),
    reason="CUDA unavailable",
)
def test_gpu_cpu_evidence_parity_float64():
    expr, coords, labels = build_freeze_fixture()
    common = dict(
        inference_mode="latent",
        latent_refit_rounds=1,
        observation_model="poisson",
        fit_diffuse=True,
        evidence_mode="cv_deviance",
        evidence_weight="linear",
        **FIXTURE_KW,
    )
    c_cpu, _ = SPARKLE(**common).fit_transform(expr, coords, labels)
    c_gpu, d_gpu = SPARKLE(
        use_gpu=True, gpu_dtype="float64", **common
    ).fit_transform(expr, coords, labels)
    assert d_gpu["compute_backend"] == "gpu"
    np.testing.assert_allclose(c_cpu, c_gpu, rtol=1e-9, atol=1e-9)
