"""Tests for the Poisson background count model (2.x Phase 2)."""

import hashlib

import numpy as np
import pytest

from stambient import SPARKLE
from stambient.count_model import (
    _fit_poisson_batch_cpu,
    estimate_leakage_poisson,
)

from tests.test_legacy_freeze import FROZEN_SHA256, build_freeze_fixture

FIXTURE_KW = dict(
    bin_size=10.0,
    max_radius=100.0,
    lambda_grid=[10.0, 25.0, 50.0],
    r2_threshold=0.01,
    verbose=False,
)


def _simulate_poisson_bins(seed=1):
    rng = np.random.RandomState(seed)
    n_bins, batch = 300, 6
    A = rng.uniform(50, 150, n_bins)
    S = rng.gamma(1.0, 0.4, (n_bins, batch))
    S[rng.rand(n_bins, batch) < 0.3] = 0.0
    true_beta = np.array([0.0, 0.05, 0.2, 0.0, 0.1, 0.3])
    true_rho = np.array([0.1, 0.05, 0.02, 0.08, 0.0, 0.04])
    mu = A[:, None] * (true_beta[None, :] + true_rho[None, :] * S)
    y = rng.poisson(mu).astype(np.float64)
    return y, A, S, true_beta, true_rho


@pytest.mark.parametrize("seed", [1, 7, 13])
def test_poisson_solver_recovers_parameters(seed):
    y, A, S, true_beta, true_rho = _simulate_poisson_bins(seed)
    beta, rho, loglik, null_loglik = _fit_poisson_batch_cpu(
        y, A, S, fit_diffuse=True, max_iter=25, tol=1e-6
    )
    np.testing.assert_allclose(beta, true_beta, atol=0.03)
    np.testing.assert_allclose(rho, true_rho, atol=0.06)
    # The full model must beat the diffuse-only null on genes with signal.
    assert (loglik[true_rho > 0] > null_loglik[true_rho > 0]).all()


def test_local_only_fit_absorbs_diffuse_into_rho():
    """B1 vs B2: with true diffuse background, the local-only Poisson fit
    must overestimate rho; the diffuse fit debiases it."""
    y, A, S, true_beta, true_rho = _simulate_poisson_bins()
    has_diffuse = true_beta > 0
    _, rho_local, _, _ = _fit_poisson_batch_cpu(
        y, A, S, fit_diffuse=False, max_iter=25, tol=1e-6
    )
    beta_full, rho_full, _, _ = _fit_poisson_batch_cpu(
        y, A, S, fit_diffuse=True, max_iter=25, tol=1e-6
    )
    assert (rho_local[has_diffuse] > true_rho[has_diffuse] + 0.05).all()
    # The diffuse fit does not have this systematic upward bias.
    err_full = np.abs(rho_full[has_diffuse] - true_rho[has_diffuse])
    err_local = np.abs(rho_local[has_diffuse] - true_rho[has_diffuse])
    assert (err_full < err_local).all()


def test_degenerate_genes_are_safe():
    y, A, S, _, _ = _simulate_poisson_bins()
    zero_y = np.zeros((y.shape[0], 1))
    beta, rho, _, _ = _fit_poisson_batch_cpu(
        zero_y, A, S[:, :1], fit_diffuse=True, max_iter=25, tol=1e-6
    )
    assert beta[0] == 0.0 and rho[0] == 0.0
    # No spatial signal: rho unidentifiable -> 0, beta = mean rate.
    S0 = np.zeros((y.shape[0], 1))
    beta, rho, _, _ = _fit_poisson_batch_cpu(
        y[:, :1], A, S0, fit_diffuse=True, max_iter=25, tol=1e-6
    )
    assert rho[0] == 0.0
    np.testing.assert_allclose(beta[0], y[:, :1].sum() / A.sum(), rtol=1e-6)


def test_pipeline_poisson_modes_and_validation():
    expr, coords, labels = build_freeze_fixture()
    # Poisson requires the latent framework.
    with pytest.raises(ValueError):
        SPARKLE(inference_mode="legacy", observation_model="poisson")
    with pytest.raises(ValueError):
        SPARKLE(inference_mode="latent", observation_model="foo")

    totals = {}
    for fd, sd in [(False, False), (True, False), (True, True)]:
        corrected, diag = SPARKLE(
            inference_mode="latent",
            latent_refit_rounds=1,
            observation_model="poisson",
            fit_diffuse=fd,
            subtract_diffuse=sd,
            **FIXTURE_KW,
        ).fit_transform(expr, coords, labels)
        assert diag["latent"]["converged"]
        assert (corrected >= 0).all()
        assert diag["observation_model"] == "poisson"
        totals[(fd, sd)] = corrected.sum()
    # Fitting a diffuse component changes the correction (through rho),
    # and subtracting it can only reduce the total further.
    assert abs(totals[(True, False)] - totals[(False, False)]) > 1e-6
    assert totals[(True, True)] <= totals[(True, False)] + 1e-6


def test_pipeline_legacy_unaffected_by_poisson_addition():
    expr, coords, labels = build_freeze_fixture()
    corrected, _ = SPARKLE(**FIXTURE_KW).fit_transform(expr, coords, labels)
    digest = hashlib.sha256(np.ascontiguousarray(corrected).tobytes()).hexdigest()
    assert digest == FROZEN_SHA256


@pytest.mark.skipif(
    not pytest.importorskip("torch").cuda.is_available(),
    reason="CUDA unavailable",
)
def test_gpu_cpu_poisson_parity_float64():
    expr, coords, labels = build_freeze_fixture()
    for fd, sd in [(False, False), (True, False), (True, True)]:
        common = dict(
            inference_mode="latent", latent_refit_rounds=1,
            observation_model="poisson", fit_diffuse=fd, subtract_diffuse=sd,
            **FIXTURE_KW,
        )
        c_cpu, _ = SPARKLE(**common).fit_transform(expr, coords, labels)
        c_gpu, d_gpu = SPARKLE(
            use_gpu=True, gpu_dtype="float64", **common
        ).fit_transform(expr, coords, labels)
        assert d_gpu["compute_backend"] == "gpu"
        np.testing.assert_allclose(c_cpu, c_gpu, rtol=1e-9, atol=1e-9)
