"""Tests for the latent-X self-consistent inference mode (2.0-alpha1)."""

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from stambient import SPARKLE
from stambient.latent_correction import correct_cells_latent

from tests.test_legacy_freeze import build_freeze_fixture

FIXTURE_KW = dict(
    bin_size=10.0,
    max_radius=100.0,
    lambda_grid=[10.0, 25.0, 50.0],
    r2_threshold=0.01,
    verbose=False,
)


def _mock_inputs(n_genes=3, n_cells=4, alpha=0.1):
    """Minimal direct inputs for correct_cells_latent."""
    rng = np.random.RandomState(7)
    cell_expr = rng.poisson(5.0, size=(n_genes, n_cells)).astype(np.float64)
    cell_areas = np.full(n_cells, 10.0)
    coords = np.array(
        [[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]]
    )[:n_cells]
    D = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
    np.fill_diagonal(D, 0.0)
    dist_graph = csr_matrix(D)
    return dict(
        gene_indices=np.arange(n_genes),
        r2_scores=np.full(n_genes, 0.5),
        r2_threshold=0.01,
        n_genes=n_genes,
        n_cells=n_cells,
        cell_expr=cell_expr,
        cell_areas=cell_areas,
        alphas=np.full(n_genes, alpha),
        cell_distance_graph=dist_graph,
        lam_weights=25.0,
        distance_metric="exponential",
        use_expr_weight=False,
        self_confidence_penalty=True,
        penalty_mode="1/(1+(s/p90)²)",
        gpu=None,
        storage_dtype=None,
        reduction_dtype=None,
        effective_gpu_gene_batch_size=None,
    )


def test_zero_alpha_returns_observed_expression():
    """With alpha = 0 there is no leakage: latent X must equal Y."""
    inputs = _mock_inputs(alpha=0.0)
    corrected, info = correct_cells_latent(**inputs)
    np.testing.assert_allclose(corrected, inputs["cell_expr"])
    assert info["n_iterations"] == [1]
    assert info["converged"]


def test_iteration_converges_and_respects_bounds():
    inputs = _mock_inputs()
    corrected, info = correct_cells_latent(
        **inputs, eta=0.5, max_iter=50, tol=1e-6
    )
    assert info["converged"]
    history = info["rel_change_history_max"]
    assert all(b <= a for a, b in zip(history, history[1:]))
    assert (corrected >= 0).all()
    # Correction only subtracts.
    assert (corrected <= inputs["cell_expr"] + 1e-12).all()


def test_fixed_point_residual_is_small():
    """At convergence X ≈ max(Y - L(X), 0) (penalty-free fixed point)."""
    inputs = {**_mock_inputs(), "self_confidence_penalty": False}
    corrected, info = correct_cells_latent(
        **inputs, eta=0.5, max_iter=100, tol=1e-10
    )
    X = corrected
    sources = X / inputs["cell_areas"][None, :]
    W = inputs["cell_distance_graph"].toarray()
    K = np.where(W > 0, np.exp(-W / 25.0), 0.0)  # zero diagonal stays zero
    neighbor = K @ sources.T
    ambient = inputs["alphas"][None, :] * inputs["cell_areas"][:, None] * neighbor
    residual = X - np.maximum(inputs["cell_expr"] - ambient.T, 0.0)
    assert np.abs(residual).max() < 1e-6


def test_penalty_reduces_subtraction():
    """The penalty shrinks the ambient term (weights <= 1), so the penalised
    run subtracts less in aggregate than the unpenalised run."""
    base = _mock_inputs()
    with_penalty, _ = correct_cells_latent(**base)
    without_penalty, _ = correct_cells_latent(
        **{**base, "self_confidence_penalty": False}
    )
    assert with_penalty.sum() > without_penalty.sum()
    assert not np.allclose(with_penalty, without_penalty)


def test_invalid_latent_parameters_raise():
    inputs = _mock_inputs()
    with pytest.raises(ValueError):
        correct_cells_latent(**inputs, eta=0.0)
    with pytest.raises(ValueError):
        correct_cells_latent(**inputs, eta=1.5)
    with pytest.raises(ValueError):
        correct_cells_latent(**inputs, max_iter=0)
    with pytest.raises(ValueError):
        correct_cells_latent(**inputs, tol=0.0)
    with pytest.raises(ValueError):
        SPARKLE(inference_mode="unknown")


def test_pipeline_latent_mode_converges_on_fixture():
    expr, coords, labels = build_freeze_fixture()
    corrected, diag = SPARKLE(
        inference_mode="latent", **FIXTURE_KW
    ).fit_transform(expr, coords, labels)
    info = diag["latent"]
    assert diag["inference_mode"] == "latent"
    assert info["converged"]
    assert (corrected >= 0).all()
    # λ and α are inherited from the 1.x pipeline unchanged.
    legacy_corrected, legacy_diag = SPARKLE(**FIXTURE_KW).fit_transform(
        expr, coords, labels
    )
    assert diag["lambda_estimated"] == legacy_diag["lambda_estimated"]
    np.testing.assert_array_equal(diag["alphas"], legacy_diag["alphas"])
    np.testing.assert_array_equal(diag["r2_scores"], legacy_diag["r2_scores"])
    # Latent mode differs from legacy only through the correction stage.
    assert not np.allclose(corrected, legacy_corrected)


def test_pipeline_latent_defaults_match_roadmap():
    model = SPARKLE()
    assert model.inference_mode == "legacy"
    assert model.latent_eta == 0.5
    assert model.latent_max_iter == 20
    assert model.latent_tol == 1e-4


@pytest.mark.skipif(
    not pytest.importorskip("torch").cuda.is_available(),
    reason="CUDA unavailable",
)
def test_gpu_cpu_latent_parity_float64():
    expr, coords, labels = build_freeze_fixture()
    for penalty in (True, False):
        c_cpu, _ = SPARKLE(
            inference_mode="latent", self_confidence_penalty=penalty,
            **FIXTURE_KW,
        ).fit_transform(expr, coords, labels)
        c_gpu, d_gpu = SPARKLE(
            inference_mode="latent", self_confidence_penalty=penalty,
            use_gpu=True, gpu_dtype="float64", **FIXTURE_KW,
        ).fit_transform(expr, coords, labels)
        assert d_gpu["compute_backend"] == "gpu"
        np.testing.assert_allclose(c_cpu, c_gpu, rtol=1e-10, atol=1e-10)
