"""Tests for optional GPU execution and fallback."""

import numpy as np
import pytest

from stambient import SPARKLE
from stambient.gpu import GPUContext


def _small_spatial_data(seed=7):
    rng = np.random.RandomState(seed)
    x, y = np.meshgrid(np.arange(24, dtype=float), np.arange(24, dtype=float))
    coords = np.column_stack((x.ravel(), y.ravel()))
    centers = np.array([[5, 5], [18, 5], [5, 18], [18, 18]], dtype=float)
    labels = np.full(len(coords), -1, dtype=np.int64)
    for cell_id, center in enumerate(centers):
        inside = np.linalg.norm(coords - center, axis=1) <= 4.5
        labels[inside] = cell_id

    expr = rng.poisson(0.2, size=(8, len(coords))).astype(np.float64)
    for cell_id in range(4):
        mask = labels == cell_id
        expr[cell_id % 4, mask] += rng.poisson(2.0, size=mask.sum())
    return expr, coords, labels


def _run(use_gpu, gpu_dtype="float64", gpu_gene_batch_size=None):
    expr, coords, labels = _small_spatial_data()
    model = SPARKLE(
        use_gpu=use_gpu,
        gpu_dtype=gpu_dtype,
        gpu_gene_batch_size=gpu_gene_batch_size,
        cell_based=True,
        bin_size=4,
        max_radius=30,
        n_high_genes=8,
        n_lambda_genes=4,
        lambda_grid=[5, 10, 20],
        verbose=False,
    )
    return model.fit_transform(expr, coords, labels)


def test_gpu_request_runs_or_falls_back_and_matches_cpu():
    cpu, cpu_diag = _run(use_gpu=False)
    candidate, candidate_diag = _run(use_gpu=True)

    assert candidate_diag["gpu_requested"] is True
    assert candidate_diag["compute_backend"] in {"cpu", "gpu"}
    np.testing.assert_allclose(candidate, cpu, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(
        candidate_diag["alphas"], cpu_diag["alphas"], rtol=1e-8, atol=1e-10
    )


def test_torch_sparse_path_matches_numpy_on_cpu(monkeypatch):
    torch = pytest.importorskip("torch")
    import stambient.cell_pipeline as cell_pipeline

    cpu, cpu_diag = _run(use_gpu=False)
    monkeypatch.setattr(
        cell_pipeline,
        "resolve_gpu",
        lambda use_gpu: (
            GPUContext(torch=torch, device=torch.device("cpu"), name="torch-cpu-test"),
            None,
        ),
    )
    accelerated, accelerated_diag = _run(use_gpu=True)

    assert accelerated_diag["compute_backend"] == "gpu"
    np.testing.assert_allclose(accelerated, cpu, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(
        accelerated_diag["r2_scores"], cpu_diag["r2_scores"], rtol=1e-8, atol=1e-10
    )
    assert accelerated_diag["gpu_sparse_format"] == "csr"
    assert accelerated_diag["empty_to_cell_graph_nnz"] > 0
    assert accelerated_diag["cell_to_cell_graph_nnz"] > 0
    assert accelerated_diag["timings_sec"]["total_sec"] > 0
    assert accelerated_diag["timings_sec"]["empty_bin_assignment_sec"] > 0
    assert accelerated_diag["timings_sec"]["empty_expression_aggregation_sec"] > 0


@pytest.mark.parametrize(
    ("gpu_dtype", "rtol", "atol"),
    [("mixed", 2e-5, 1e-7), ("float32", 1e-4, 1e-5)],
)
def test_reduced_precision_and_small_batches_match_cpu(
    monkeypatch, gpu_dtype, rtol, atol
):
    torch = pytest.importorskip("torch")
    import stambient.cell_pipeline as cell_pipeline

    cpu, _ = _run(use_gpu=False)
    monkeypatch.setattr(
        cell_pipeline,
        "resolve_gpu",
        lambda use_gpu: (
            GPUContext(torch=torch, device=torch.device("cpu"), name="torch-cpu-test"),
            None,
        ),
    )
    accelerated, diagnostics = _run(
        use_gpu=True,
        gpu_dtype=gpu_dtype,
        gpu_gene_batch_size=3,
    )

    assert diagnostics["gpu_dtype"] == gpu_dtype
    assert diagnostics["gpu_gene_batch_size"] == 3
    np.testing.assert_allclose(accelerated, cpu, rtol=rtol, atol=atol)
