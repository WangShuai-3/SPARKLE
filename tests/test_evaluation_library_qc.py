import numpy as np
import pytest
from scipy.sparse import csr_matrix

from evaluation.scripts.final_comparison import summarize_cell_libraries


@pytest.mark.parametrize("as_sparse", [False, True])
def test_summarize_cell_libraries_matches_downstream_negative_clipping(as_sparse):
    expr = np.array(
        [
            [1.0, 0.0, -5.0],
            [2.0, 0.0, 4.0],
        ]
    )
    if as_sparse:
        expr = csr_matrix(expr)

    summary = summarize_cell_libraries(expr, cell_ids=[101, 102, 103])

    assert summary["n_cells"] == 3
    assert summary["n_zero_library_cells"] == 1
    assert summary["zero_library_cell_ids"] == [102]
    assert summary["n_negative_values_clipped_for_qc"] == 1
    assert summary["total_counts"] == pytest.approx(7.0)


def test_summarize_cell_libraries_rejects_nonfinite_expression():
    expr = np.array([[1.0, np.nan], [2.0, 3.0]])

    with pytest.raises(ValueError, match="non-finite"):
        summarize_cell_libraries(expr)


def test_summarize_cell_libraries_checks_cell_id_alignment():
    with pytest.raises(ValueError, match="cell_ids has length"):
        summarize_cell_libraries(np.ones((2, 3)), cell_ids=[1, 2])
