"""Tests for gene batching and sparse aggregation on the CPU path."""

import numpy as np
import pytest
from scipy.sparse import csr_matrix, issparse

from stambient import SPARKLE
from stambient.batching import choose_cpu_gene_batch_size, densify_gene_rows
from stambient.aggregation import (
    aggregate_empty_bin_expression,
    extract_cells,
    select_high_expression_genes,
)


def _spatial_data(seed=11, n_genes=12, grid=28):
    rng = np.random.RandomState(seed)
    x, y = np.meshgrid(np.arange(grid, dtype=float), np.arange(grid, dtype=float))
    coords = np.column_stack((x.ravel(), y.ravel()))
    labels = np.full(len(coords), -1, dtype=np.int64)
    centers = [(7, 7), (20, 7), (7, 20), (20, 20)]
    for cell_id, center in enumerate(centers):
        inside = np.linalg.norm(coords - center, axis=1) <= 4.5
        labels[inside] = cell_id
    expr = rng.poisson(0.3, size=(n_genes, len(coords))).astype(np.float64)
    for cell_id in range(len(centers)):
        mask = labels == cell_id
        expr[cell_id % n_genes, mask] += rng.poisson(3.0, size=mask.sum())
    return expr, coords, labels


def _fit(expr, coords, labels, **kwargs):
    model = SPARKLE(
        bin_size=4,
        max_radius=30,
        lambda_grid=[5, 10, 20],
        n_lambda_genes=4,
        r2_threshold=0.0,  # every selected gene is corrected
        verbose=False,
        **kwargs,
    )
    return model.fit_transform(expr, coords, labels)


@pytest.mark.parametrize("batch", [1, 2, 3, 7, 12])
def test_cpu_gene_batches_match_unbatched(batch):
    """Batching only changes kernel dispatch, not the estimator."""
    expr, coords, labels = _spatial_data()
    base_corr, base_diag = _fit(expr, coords, labels)
    corr, diag = _fit(
        expr, coords, labels, cpu_gene_batch_size=batch
    )

    assert diag["cpu_gene_batch_size"] == batch
    assert diag["lambda_estimated"] == base_diag["lambda_estimated"]
    np.testing.assert_allclose(
        diag["alphas"], base_diag["alphas"], rtol=1e-12, atol=1e-14
    )
    np.testing.assert_allclose(
        diag["r2_scores"], base_diag["r2_scores"], rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(corr, base_corr, rtol=1e-12, atol=1e-12)


def test_adaptive_cpu_batch_size_is_reported_and_bounded():
    expr, coords, labels = _spatial_data()
    _, diag = _fit(expr, coords, labels)

    assert diag["compute_backend"] == "cpu"
    assert diag["cpu_gene_batch_size"] == expr.shape[0]  # tiny problem: one batch
    assert diag["gpu_gene_batch_size"] is None


def test_sparse_input_matches_dense_input():
    """The pipeline contract accepts both; results must agree."""
    expr, coords, labels = _spatial_data()
    dense_corr, dense_diag = _fit(expr, coords, labels)
    sparse_corr, sparse_diag = _fit(csr_matrix(expr), coords, labels)

    np.testing.assert_array_equal(sparse_corr, dense_corr)
    np.testing.assert_array_equal(sparse_diag["alphas"], dense_diag["alphas"])
    np.testing.assert_array_equal(
        sparse_diag["r2_scores"], dense_diag["r2_scores"]
    )


def test_torch_backend_matches_cpu_with_sparse_aggregates(monkeypatch):
    """The torch (test-mocked CUDA) path also densifies per batch."""
    torch = pytest.importorskip("torch")
    import stambient.cell_pipeline as cell_pipeline
    from stambient.gpu import GPUContext

    expr, coords, labels = _spatial_data()
    cpu_corr, cpu_diag = _fit(expr, coords, labels)
    monkeypatch.setattr(
        cell_pipeline,
        "resolve_gpu",
        lambda use_gpu: (
            GPUContext(
                torch=torch, device=torch.device("cpu"), name="torch-cpu-test"
            ),
            None,
        ),
    )
    gpu_corr, gpu_diag = _fit(
        expr, coords, labels, use_gpu=True, gpu_gene_batch_size=3
    )

    assert gpu_diag["compute_backend"] == "gpu"
    assert gpu_diag["cpu_gene_batch_size"] is None
    assert gpu_diag["gpu_gene_batch_size"] == 3
    np.testing.assert_allclose(gpu_corr, cpu_corr, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(
        gpu_diag["alphas"], cpu_diag["alphas"], rtol=1e-8, atol=1e-10
    )


def test_aggregation_returns_sparse_matrices():
    """Aggregated gene matrices stay sparse until a batch is densified."""
    expr, coords, labels = _spatial_data()
    expr_sp = csr_matrix(expr).tocsc()
    empty = np.where(labels < 0)[0]

    unique_cells, cell_areas, centroids, cell_expr = extract_cells(
        expr_sp, coords, labels
    )
    assert issparse(cell_expr)
    assert cell_expr.format == "csr"
    np.testing.assert_allclose(
        cell_expr.toarray(), expr @ _cell_indicator(labels, unique_cells)
    )

    from stambient.binning import bin_empty_dnbs

    _, empty_areas, empty_rows, empty_cols = bin_empty_dnbs(coords, labels, 4.0)
    empty_bin_expr = aggregate_empty_bin_expression(
        expr_sp, empty_rows, empty_cols, len(empty_areas), expr.shape[0]
    )
    assert issparse(empty_bin_expr)
    assert empty_bin_expr.format == "csr"

    # Gene selection works directly on the sparse matrix.
    gene_indices = select_high_expression_genes(
        empty_bin_expr, empty_areas, n_high_genes=5
    )
    dense_indices = select_high_expression_genes(
        empty_bin_expr.toarray(), empty_areas, n_high_genes=5
    )
    np.testing.assert_array_equal(gene_indices, dense_indices)


def _cell_indicator(labels, unique_cells):
    """[n_dnbs × n_cells] 0/1 map matching extract_cells' construction."""
    cell_ids = labels[labels >= 0]
    lookup = {cid: i for i, cid in enumerate(unique_cells)}
    indicator = np.zeros((len(labels), len(unique_cells)))
    dnb_idx = np.where(labels >= 0)[0]
    for row, cid in zip(dnb_idx, cell_ids):
        indicator[row, lookup[cid]] = 1.0
    return indicator


def test_densify_gene_rows_returns_dense_batch():
    rng = np.random.RandomState(0)
    dense = rng.poisson(2.0, size=(6, 9)).astype(float)
    sparse = csr_matrix(dense)
    idx = np.array([4, 1, 1, 3])

    for matrix in (dense, sparse):
        rows = densify_gene_rows(matrix, idx)
        assert isinstance(rows, np.ndarray)
        assert rows.dtype == np.float64
        assert rows.shape == (4, 9)
        np.testing.assert_array_equal(rows, dense[idx])


@pytest.mark.parametrize(
    "n_rows,n_cols,max_batch,expected",
    [
        (100, 50, 64, 64),  # budget allows the maximum
        # 1 GiB / (6 × 1e6 × 8 B) rounds down to 22 genes
        (10**6, 10**6, 4096, 22),
        (1000, 200, 4096, 4096),
        (0, 0, 0, 1),  # degenerate inputs are clamped
    ],
)
def test_choose_cpu_gene_batch_size_bounds(n_rows, n_cols, max_batch, expected):
    assert choose_cpu_gene_batch_size(n_rows, n_cols, max_batch) == expected


def test_choose_cpu_gene_batch_size_scales_with_budget():
    small = choose_cpu_gene_batch_size(
        100_000, 100_000, 4096, working_set_bytes=1 << 28
    )
    large = choose_cpu_gene_batch_size(
        100_000, 100_000, 4096, working_set_bytes=1 << 32
    )
    assert 1 <= small < large <= 4096


@pytest.mark.parametrize("value", [0, -3, 2.5, float("nan")])
def test_invalid_cpu_gene_batch_size_raises(value):
    with pytest.raises(ValueError, match="cpu_gene_batch_size"):
        SPARKLE(cpu_gene_batch_size=value)
