"""Numerical freeze of the SPARKLE 1.x (legacy) pipeline.

The expectations below were recorded at tag ``sparkle-v1-baseline`` with the
default ``inference_mode="legacy"``. Any future 2.x change must keep the
legacy path numerically identical on this fixture.
"""

import hashlib

import numpy as np

from stambient import SPARKLE

FROZEN_LAMBDA = 25.0
FROZEN_SHA256 = (
    "f29f45a80aefa61fcdc6443a0a644cf2c0ddd087b26043f89ae3c6b81711baeb"
)
FROZEN_ALPHAS = [
    0.023158, 0.018332, 0.021429, 0.018106, 0.016973,
    0.020444, 0.018308, 0.019577, 0.017754, 0.018032,
    0.019304, 0.021107, 0.019856, 0.01623, 0.019938,
    0.012392, 0.02091, 0.018884, 0.01274, 0.010871,
]
FROZEN_R2 = [
    0.313542, 0.104164, 0.246612, 0.235675, 0.245095,
    0.08809, 0.161827, 0.329699, 0.340518, 0.243653,
    0.013038, -0.066271, -0.001827, 0.034289, -0.022874,
    -0.019875, -0.010925, -0.006903, 0.02927, -0.009335,
]


def build_freeze_fixture(seed=20260814):
    """Small deterministic DNB-level dataset with known leakage structure."""
    rng = np.random.RandomState(seed)
    # 24 cells on a jittered 6x4 grid over a 120x80 um FOV, pitch 1.0 um.
    xs = np.linspace(10, 110, 6)
    ys = np.linspace(10, 70, 4)
    xx, yy = np.meshgrid(xs, ys)
    centers = np.column_stack([xx.ravel(), yy.ravel()]).astype(float)
    centers += rng.uniform(-2, 2, centers.shape)
    gx = np.arange(0, 120, 1.0)
    gy = np.arange(0, 80, 1.0)
    gxx, gyy = np.meshgrid(gx, gy)
    coords = np.column_stack([gxx.ravel(), gyy.ravel()])
    labels = np.full(len(coords), -1, dtype=np.int64)
    for ci, c in enumerate(centers):
        d = np.linalg.norm(coords - c, axis=1)
        labels[d <= 4.0] = ci
    n_genes = 20
    lam_true = 25.0
    n_cells = len(centers)
    true_expr = np.zeros((n_genes, n_cells))
    for g in range(n_genes):
        if g < 10:
            owners = np.arange(n_cells) % 3 == (g % 3)
            true_expr[g, owners] = rng.uniform(20, 60, owners.sum())
        else:
            true_expr[g, :] = rng.uniform(0.5, 3, n_cells)
    areas = np.bincount(labels[labels >= 0], minlength=n_cells).astype(float)
    src = true_expr / areas[None, :]
    D = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
    K = np.exp(-D / lam_true)
    np.fill_diagonal(K, 0.0)
    alpha = 0.02
    cell_ambient = alpha * areas[None, :] * (src @ K)
    dnb_expr = np.zeros((n_genes, len(coords)))
    for ci in range(n_cells):
        idx = np.flatnonzero(labels == ci)
        per = (true_expr[:, ci] + cell_ambient[:, ci]) / len(idx)
        dnb_expr[:, idx] = per[:, None]
    De = np.linalg.norm(coords[:, None, :] - centers[None, :, :], axis=2)
    Ke = np.where(De <= 3 * lam_true, np.exp(-De / lam_true), 0.0)
    empty_ambient = alpha * (src @ Ke.T)
    empty_idx = np.flatnonzero(labels < 0)
    dnb_expr[:, empty_idx] += empty_ambient[:, empty_idx]
    dnb_expr = rng.poisson(dnb_expr).astype(np.float64)
    return dnb_expr, coords, labels


def run_legacy_fixture():
    dnb_expr, coords, labels = build_freeze_fixture()
    model = SPARKLE(
        bin_size=10.0,
        max_radius=100.0,
        lambda_grid=[10.0, 25.0, 50.0],
        r2_threshold=0.01,
        verbose=False,
    )
    return model.fit_transform(dnb_expr, coords, labels)


def test_legacy_pipeline_matches_frozen_baseline():
    corrected, diag = run_legacy_fixture()
    assert diag["lambda_estimated"] == FROZEN_LAMBDA
    np.testing.assert_allclose(
        np.round(diag["alphas"], 6), FROZEN_ALPHAS, atol=1e-6
    )
    np.testing.assert_allclose(
        np.round(diag["r2_scores"], 6), FROZEN_R2, atol=1e-6
    )
    digest = hashlib.sha256(
        np.ascontiguousarray(corrected).tobytes()
    ).hexdigest()
    assert digest == FROZEN_SHA256
