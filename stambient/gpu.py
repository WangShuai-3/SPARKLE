"""Optional PyTorch CUDA backend used by the cell-based SPARKLE pipeline.

PyTorch is intentionally imported lazily so the base NumPy/SciPy installation
does not acquire a mandatory GPU dependency.
"""

from dataclasses import dataclass
from typing import Any, Optional, Tuple
import warnings

import numpy as np
from scipy.sparse import spmatrix


class GPUExecutionError(RuntimeError):
    """Raised when a CUDA operation fails after GPU execution was selected."""


@dataclass(frozen=True)
class GPUContext:
    """The lazily resolved PyTorch CUDA objects needed by the pipeline."""

    torch: Any
    device: Any
    name: str


@dataclass(frozen=True)
class GPUCSRDistanceGraph:
    """CSR topology and distances resident on the selected device."""

    crow_indices: Any
    col_indices: Any
    distances: Any
    shape: Tuple[int, int]


def resolve_gpu(use_gpu: bool) -> Tuple[Optional[GPUContext], Optional[str]]:
    """Resolve a usable CUDA device, returning a CPU fallback reason if needed."""
    if not use_gpu:
        return None, None

    try:
        import torch
    except (ImportError, OSError) as exc:
        reason = f"PyTorch could not be imported ({exc})"
        warnings.warn(f"GPU requested but {reason}; falling back to CPU.", RuntimeWarning)
        return None, reason

    try:
        if not torch.cuda.is_available():
            reason = "PyTorch reports that CUDA is unavailable"
            warnings.warn(f"GPU requested but {reason}; falling back to CPU.", RuntimeWarning)
            return None, reason

        device = torch.device("cuda")
        # Force CUDA context creation here so driver/device failures trigger the
        # documented CPU fallback before the expensive preprocessing starts.
        torch.empty(1, dtype=torch.float64, device=device)
        name = torch.cuda.get_device_name(device)
        return GPUContext(torch=torch, device=device, name=name), None
    except Exception as exc:
        reason = f"CUDA initialization failed ({exc})"
        warnings.warn(f"GPU requested but {reason}; falling back to CPU.", RuntimeWarning)
        return None, reason


def to_gpu(array: np.ndarray, context: GPUContext, dtype=None):
    """Copy a NumPy array to the selected device with an explicit dtype."""
    try:
        dtype = dtype or context.torch.float64
        return context.torch.as_tensor(
            np.asarray(array), dtype=dtype, device=context.device
        )
    except Exception as exc:
        raise GPUExecutionError(f"Failed to transfer an array to CUDA: {exc}") from exc


def sparse_distance_graph_to_gpu_csr(
    matrix: spmatrix, context: GPUContext, dtype=None
) -> GPUCSRDistanceGraph:
    """Transfer CSR topology and distance values once to the selected device."""
    try:
        dtype = dtype or context.torch.float64
        csr = matrix.tocsr()
        crow = context.torch.as_tensor(
            csr.indptr.astype(np.int64, copy=False),
            dtype=context.torch.int64,
            device=context.device,
        )
        columns = context.torch.as_tensor(
            csr.indices.astype(np.int64, copy=False),
            dtype=context.torch.int64,
            device=context.device,
        )
        distances = context.torch.as_tensor(
            csr.data, dtype=dtype, device=context.device
        )
        return GPUCSRDistanceGraph(crow, columns, distances, csr.shape)
    except Exception as exc:
        raise GPUExecutionError(f"Failed to transfer a CSR distance graph: {exc}") from exc


def weighted_gpu_csr(
    graph: GPUCSRDistanceGraph,
    lam: float,
    metric: str,
    context: GPUContext,
):
    """Create a weighted CSR tensor without retransferring graph indices."""
    try:
        if metric == "exponential":
            values = context.torch.exp(-graph.distances / lam)
        elif metric == "gaussian":
            values = context.torch.exp(
                -(graph.distances.square()) / (2.0 * lam * lam)
            )
        elif metric == "inverse":
            values = 1.0 / (graph.distances + 1e-6)
        else:
            raise ValueError(f"Unknown distance metric: {metric}")
        return context.torch.sparse_csr_tensor(
            graph.crow_indices,
            graph.col_indices,
            values,
            size=graph.shape,
            dtype=values.dtype,
            device=context.device,
        )
    except Exception as exc:
        raise GPUExecutionError(f"Failed to construct a weighted GPU CSR graph: {exc}") from exc


def sparse_to_gpu(matrix: spmatrix, context: GPUContext, dtype=None):
    """Convert an already weighted SciPy sparse matrix to GPU CSR.

    Retained for compatibility; optimized code should transfer a distance
    graph with :func:`sparse_distance_graph_to_gpu_csr` and reweight in place.
    """
    graph = sparse_distance_graph_to_gpu_csr(matrix, context, dtype=dtype)
    try:
        return context.torch.sparse_csr_tensor(
            graph.crow_indices,
            graph.col_indices,
            graph.distances,
            size=graph.shape,
            dtype=graph.distances.dtype,
            device=context.device,
        )
    except Exception as exc:
        raise GPUExecutionError(f"Failed to construct a GPU CSR tensor: {exc}") from exc


def choose_gpu_gene_batch_size(
    context: GPUContext,
    n_rows: int,
    n_cells: int,
    dtype,
    max_batch_size: int = 4096,
) -> int:
    """Choose a conservative batch size from currently available VRAM."""
    max_batch_size = max(1, int(max_batch_size))
    try:
        if context.device.type != "cuda":
            return min(512, max_batch_size)
        free_bytes, _ = context.torch.cuda.mem_get_info(context.device)
        element_size = context.torch.tensor([], dtype=dtype).element_size()
        # Sources, observations, products, residuals and correction temporaries
        # coexist. Reserve most free VRAM for CSR graphs and allocator overhead.
        bytes_per_gene = max(1, (3 * n_rows + 5 * n_cells) * element_size)
        estimate = int((free_bytes * 0.30) // bytes_per_gene)
        estimate = max(1, min(max_batch_size, estimate))
        return max(32, (estimate // 32) * 32) if estimate >= 32 else estimate
    except Exception:
        return min(512, max_batch_size)


def sparse_mm(sparse_matrix, dense_matrix, context: GPUContext):
    """Run sparse-dense multiplication and normalize CUDA failures."""
    try:
        return context.torch.sparse.mm(sparse_matrix, dense_matrix)
    except Exception as exc:
        raise GPUExecutionError(f"CUDA sparse matrix multiplication failed: {exc}") from exc


def to_cpu(tensor, context: GPUContext) -> np.ndarray:
    """Copy a CUDA tensor back to a NumPy array."""
    try:
        return tensor.detach().cpu().numpy()
    except Exception as exc:
        raise GPUExecutionError(f"Failed to copy a CUDA result to CPU: {exc}") from exc
