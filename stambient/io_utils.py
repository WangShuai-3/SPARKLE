"""I/O utilities for SPARKLE."""

import gzip
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Union, List, Set

import numpy as np
from scipy.sparse import csr_matrix, issparse


def _pack_xy(x: int, y: int) -> int:
    """Pack two int32 coordinates into one integer key for fast dict lookup."""
    return ((int(x) & 0xFFFFFFFF) << 32) | (int(y) & 0xFFFFFFFF)


def _load_RYTools_label_map(
    scgem_path: Union[str, Path],
    x_col: str = "x",
    y_col: str = "y",
    cell_label_col: str = "cell",
    empty_labels: Optional[Union[int, List[int], Set[int]]] = None,
    sep: Optional[str] = None,
    verbose: bool = True,
) -> Dict[int, int]:
    """Build a (x, y) -> original cell-id map from an RYTools-style scGEM file.

    The scGEM file is expected to contain cell-labelled DNBs only; any rows
    whose ``cell_label_col`` value is in ``empty_labels`` are skipped.  DNBs
    not present in the map will be treated as empty/background when the full
    GEM is loaded.

    Args:
        scgem_path: Path to the scGEM file (plain text or ``.gz``).
        x_col: Name of the x-coordinate column.
        y_col: Name of the y-coordinate column.
        cell_label_col: Name of the cell-label column.
        empty_labels: Values that indicate an empty/background DNB. Defaults to
            ``{0, -1}``. Can be a single int or a list/set of ints.
        sep: Field delimiter. Auto-detected if None.
        verbose: Print progress messages.

    Returns:
        Dictionary mapping packed ``(x, y)`` integer keys to original cell IDs.
    """
    scgem_path = Path(scgem_path)
    if not scgem_path.exists():
        raise FileNotFoundError(f"scGEM file not found: {scgem_path}")

    empty_set = {str(x) for x in _normalize_empty_labels(empty_labels)}

    if verbose:
        print(f"Building cell label map from {scgem_path.name}...")

    t0 = time.time()
    with _open_text_file(scgem_path) as f:
        header_line = None
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            header_line = stripped
            break

    if header_line is None:
        raise ValueError(f"No valid header found in {scgem_path}")

    sep = _detect_delimiter(header_line, sep)
    header = [h.strip() for h in header_line.split(sep)]
    required_cols = [x_col, y_col, cell_label_col]
    missing = [c for c in required_cols if c not in header]
    if missing:
        raise ValueError(
            f"Missing required columns in scGEM: {missing}. "
            f"Available columns: {header}"
        )

    col_idx = {c: i for i, c in enumerate(header)}
    ix = col_idx[x_col]
    iy = col_idx[y_col]
    il = col_idx[cell_label_col]

    label_map: Dict[int, int] = {}
    n_rows = 0
    with _open_text_file(scgem_path) as f:
        # skip header and any comments before it
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            break

        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            parts = stripped.split(sep)
            if len(parts) <= max(ix, iy, il):
                continue

            x = int(parts[ix])
            y = int(parts[iy])
            label = parts[il]

            if label in empty_set:
                continue

            label_map[_pack_xy(x, y)] = label
            n_rows += 1
            if verbose and n_rows % 5_000_000 == 0:
                print(f"  {n_rows:,} labelled DNBs, {len(label_map):,} unique positions...")

    if verbose:
        print(f"  {len(label_map):,} unique DNB positions with cell labels "
              f"({n_rows:,} rows) in {time.time() - t0:.1f}s")

    return label_map


def _load_gem_chunked(
    gem_path: Union[str, Path],
    x_col: str = "x",
    y_col: str = "y",
    gene_col: str = "geneID",
    count_col: str = "MIDCounts",
    sep: Optional[str] = None,
    nrows_per_chunk: int = 500_000,
):
    """Generator yielding pandas DataFrames of a GEM file in chunks.

    Args:
        gem_path: Path to the GEM file (plain text or ``.gz``).
        x_col, y_col, gene_col, count_col: Column names.
        sep: Field delimiter. Auto-detected if None.
        nrows_per_chunk: Number of rows per chunk.

    Yields:
        pandas.DataFrame chunks with all original columns as strings.
    """
    import pandas as pd

    gem_path = Path(gem_path)
    if not gem_path.exists():
        raise FileNotFoundError(f"GEM file not found: {gem_path}")

    with _open_text_file(gem_path) as f:
        header_line = None
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            header_line = stripped
            break

    if header_line is None:
        raise ValueError(f"No valid header found in {gem_path}")

    sep = _detect_delimiter(header_line, sep)
    header = [h.strip() for h in header_line.split(sep)]
    required_cols = [x_col, y_col, gene_col, count_col]
    missing = [c for c in required_cols if c not in header]
    if missing:
        raise ValueError(
            f"Missing required columns in GEM: {missing}. "
            f"Available columns: {header}"
        )

    with _open_text_file(gem_path) as f:
        # skip header
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            break

        chunk_rows = []
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            chunk_rows.append(stripped.split(sep))
            if len(chunk_rows) >= nrows_per_chunk:
                yield pd.DataFrame(chunk_rows, columns=header)
                chunk_rows = []
        if chunk_rows:
            yield pd.DataFrame(chunk_rows, columns=header)


def _build_dnb_matrix_from_gem(
    gem_path: Union[str, Path],
    label_map: Dict[int, Any],
    gene_col: str = "geneID",
    x_col: str = "x",
    y_col: str = "y",
    count_col: str = "MIDCounts",
    sep: Optional[str] = None,
    nrows_per_chunk: int = 500_000,
    verbose: bool = True,
) -> Tuple[csr_matrix, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build a genes × DNBs sparse matrix from a GEM file and a label map.

    DNBs present in ``label_map`` are assigned their original cell ID; all
    other DNBs are marked as empty (``-1``).  Original cell IDs are then
    remapped to a contiguous 0..n_cells-1 range for SPARKLE.

    Returns:
        - spot_expr: [n_genes × n_dnbs] CSR matrix of UMI counts.
        - spot_coords: [n_dnbs × 2] coordinates.
        - spot_labels: [n_dnbs] remapped cell indices; -1 for empty.
        - gene_names: array of gene names.
        - cell_ids: sorted array of original cell IDs.
    """
    import pandas as pd
    from scipy.sparse import coo_matrix

    gem_path = Path(gem_path)
    if not gem_path.exists():
        raise FileNotFoundError(f"GEM file not found: {gem_path}")

    if verbose:
        print(f"Single-pass loading GEM from {gem_path.name}...")

    t0 = time.time()
    gene_to_idx: Dict[str, int] = {}  # gene 名到矩阵行号的动态映射，避免先扫一遍收集所有 gene
    genes: List[str] = []  # 按首次出现顺序保存 gene 名，后续作为 gene_names 返回
    dnb_to_idx: Dict[int, int] = {}  # packed DNB 坐标到矩阵列号的动态映射，避免千万级 tuple key 开销
    dnb_x: List[int] = []  # 按首次出现顺序保存 DNB x 坐标，后续和 y 合并为 spot_coords
    dnb_y: List[int] = []  # 按首次出现顺序保存 DNB y 坐标，后续和 x 合并为 spot_coords
    row_chunks: List[np.ndarray] = []  # 每个 chunk 的 COO 行索引数组；用 numpy 数组比逐元素 Python list 更省内存
    col_chunks: List[np.ndarray] = []  # 每个 chunk 的 COO 列索引数组；最后一次性 concatenate
    data_chunks: List[np.ndarray] = []  # 每个 chunk 的 count 数组；COO 转 CSR 时会自动合并重复项
    total_rows = 0

    for chunk in _load_gem_chunked(gem_path, x_col, y_col, gene_col, count_col, sep, nrows_per_chunk):
        n_chunk = len(chunk)
        x_arr = chunk[x_col].astype(int).values
        y_arr = chunk[y_col].astype(int).values
        count_arr = chunk[count_col].astype(np.float64).values
        gene_arr = chunk[gene_col].astype(str).values

        gene_codes, gene_uniques = pd.factorize(gene_arr, sort=False)  # 当前 chunk 内 gene 编码，避免逐行查 gene 字典
        gene_chunk_map = np.empty(len(gene_uniques), dtype=np.int32)  # chunk 内 gene code -> 全局 gene index
        for local_idx, gene in enumerate(gene_uniques):
            gene = str(gene)  # pandas unique 可能返回 numpy 字符串，统一成 Python str 作为字典 key
            g_idx = gene_to_idx.get(gene)  # 查询当前 gene 是否已经分配过矩阵行号
            if g_idx is None:
                g_idx = len(genes)  # 新 gene 使用下一个行号
                gene_to_idx[gene] = g_idx  # 记录 gene -> 行号
                genes.append(gene)  # 保存 gene 名
            gene_chunk_map[local_idx] = g_idx  # 保存当前 chunk gene code 对应的全局行号
        rows = gene_chunk_map[gene_codes]  # 向量化得到当前 chunk 每一行的 COO 行索引

        x_u64 = x_arr.astype(np.uint64, copy=False)  # x 坐标转为 uint64，便于向量化 packed key
        y_u64 = y_arr.astype(np.uint64, copy=False)  # y 坐标转为 uint64，便于向量化 packed key
        dnb_keys = ((x_u64 & np.uint64(0xFFFFFFFF)) << np.uint64(32)) | (y_u64 & np.uint64(0xFFFFFFFF))  # 向量化 packed DNB key
        dnb_codes, dnb_uniques = pd.factorize(dnb_keys, sort=False)  # 当前 chunk 内 DNB 编码，减少逐行查 DNB 字典
        _, first_pos = np.unique(dnb_codes, return_index=True)  # 每个 chunk-local DNB 第一次出现的位置，用于记录原始 x/y
        dnb_chunk_map = np.empty(len(dnb_uniques), dtype=np.int32)  # chunk 内 DNB code -> 全局 DNB index
        for local_idx, key in enumerate(dnb_uniques):
            dnb_key = int(key)  # numpy uint64 转 Python int，作为全局 dnb_to_idx 字典 key
            d_idx = dnb_to_idx.get(dnb_key)  # 查询当前 DNB 是否已经分配过矩阵列号
            if d_idx is None:
                d_idx = len(dnb_x)  # 新 DNB 使用下一个列号
                dnb_to_idx[dnb_key] = d_idx  # 记录 DNB packed key -> 列号
                pos = first_pos[local_idx]  # 当前 unique DNB 在 chunk 中第一次出现的行号
                dnb_x.append(int(x_arr[pos]))  # 保存 DNB x 坐标
                dnb_y.append(int(y_arr[pos]))  # 保存 DNB y 坐标
            dnb_chunk_map[local_idx] = d_idx  # 保存当前 chunk DNB code 对应的全局列号

        cols = dnb_chunk_map[dnb_codes]  # 向量化得到当前 chunk 每一行的 COO 列索引

        row_chunks.append(rows)  # 保存当前 chunk 的行索引数组
        col_chunks.append(cols)  # 保存当前 chunk 的列索引数组
        data_chunks.append(count_arr)  # 保存当前 chunk 的表达计数数组
        total_rows += n_chunk
        if verbose and total_rows % 5_000_000 == 0:
            print(
                f"  Processed {total_rows:,} rows; "
                f"{len(genes):,} genes; {len(dnb_x):,} DNBs; "
                f"{time.time() - t0:.1f}s"
            )

    n_genes = len(genes)
    n_dnbs = len(dnb_x)
    if n_dnbs == 0:
        raise ValueError("No DNBs found in the GEM file.")

    if verbose:
        print(f"Building sparse matrix: {n_genes:,} genes × {n_dnbs:,} DNBs...")

    rows_all = np.concatenate(row_chunks) if row_chunks else np.array([], dtype=np.int32)
    cols_all = np.concatenate(col_chunks) if col_chunks else np.array([], dtype=np.int32)
    data_all = np.concatenate(data_chunks) if data_chunks else np.array([], dtype=np.float64)
    dnb_expr = coo_matrix(
        (data_all, (rows_all, cols_all)),
        shape=(n_genes, n_dnbs),
        dtype=np.float64,
    ).tocsr()

    spot_coords = np.column_stack(
        (np.asarray(dnb_x, dtype=np.float64), np.asarray(dnb_y, dtype=np.float64))
    )
    all_cells = sorted(set(label_map.values()))
    cell_to_idx = {c: i for i, c in enumerate(all_cells)}
    spot_labels = np.full(n_dnbs, -1, dtype=np.int32)
    n_matched = 0
    for i, dnb_key in enumerate(dnb_to_idx.keys()):
        label = label_map.get(dnb_key)
        if label is not None:
            spot_labels[i] = cell_to_idx[label]
            n_matched += 1

    n_empty = n_dnbs - n_matched

    if verbose:
        print(
            f"  Final: {n_genes:,} genes, {n_dnbs:,} DNBs "
            f"({n_matched:,} cell, {n_empty:,} empty), "
            f"{len(all_cells):,} cells"
        )
        print(
            f"  Total rows: {total_rows:,}, nonzeros: {dnb_expr.nnz:,}, "
            f"time: {time.time() - t0:.1f}s"
        )

    return dnb_expr, spot_coords, spot_labels, np.asarray(genes), np.asarray(all_cells)


def load_RYTools_data(
    gem_path: Union[str, Path],
    scgem_path: Union[str, Path],
    gene_col: str = "geneID",
    x_col: str = "x",
    y_col: str = "y",
    count_col: str = "MIDCounts",
    cell_label_col: str = "cell",
    empty_labels: Optional[Union[int, List[int], Set[int]]] = None,
    gem_sep: Optional[str] = None,
    scgem_sep: Optional[str] = None,
    pitch_um: Optional[float] = None,
    nrows_per_chunk: int = 500_000,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Load RYTools-style Stereo-seq data from a GEM + scGEM file pair.

    RYTools pipelines typically produce:
      - A full GEM file without cell labels (all DNBs, including background).
      - A scGEM file that contains only cell-labelled DNBs (no background rows).

    This loader merges the two files: it builds the full genes × DNBs expression
    matrix from the GEM file, assigns cell labels from the scGEM file, and
    marks any DNB absent from the scGEM as empty (``-1``).

    Args:
        gem_path: Path to the full GEM file (plain text or ``.gz``).
        scgem_path: Path to the scGEM file with cell labels (plain text or
            ``.gz``).
        gene_col: Name of the gene-ID column in both files.
        x_col: Name of the x-coordinate column in both files.
        y_col: Name of the y-coordinate column in both files.
        count_col: Name of the UMI count column in the GEM file.
        cell_label_col: Name of the cell-label column in the scGEM file.
        empty_labels: Values that indicate an empty/background DNB in the
            scGEM file. Defaults to ``{0, -1}``. Only relevant if the scGEM
            happens to contain background rows.
        gem_sep: Field delimiter for the GEM file. Auto-detected if None.
        scgem_sep: Field delimiter for the scGEM file. Auto-detected if None.
        pitch_um: If provided, scale integer coordinates by this factor to
            convert them to micrometres. If None, coordinates are assumed to
            already be in micrometres and are unchanged. Use ``0.5`` for raw
            Stereo-seq DNB-index coordinates.
        nrows_per_chunk: Number of rows to read per chunk when scanning the
            (potentially very large) GEM file.
        verbose: Print progress messages.

    Returns:
        Dictionary with keys:

        - ``spot_expr``: [n_genes × n_spots] ``csr_matrix`` of UMI counts.
        - ``spot_coords``: [n_spots × 2] spot coordinates (in µm if ``pitch_um``
          was provided, otherwise raw units).
        - ``spot_labels``: [n_spots] cell indices; ``-1`` for empty spots.
        - ``gene_names``: array of gene names.
        - ``cell_ids``: sorted array of original cell IDs.
    """
    t0 = time.time()

    label_map = _load_RYTools_label_map(
        scgem_path,
        x_col=x_col,
        y_col=y_col,
        cell_label_col=cell_label_col,
        empty_labels=empty_labels,
        sep=scgem_sep,
        verbose=verbose,
    )

    spot_expr, spot_coords, spot_labels, gene_names, cell_ids = _build_dnb_matrix_from_gem(
        gem_path,
        label_map,
        gene_col=gene_col,
        x_col=x_col,
        y_col=y_col,
        count_col=count_col,
        sep=gem_sep,
        nrows_per_chunk=nrows_per_chunk,
        verbose=verbose,
    )

    if pitch_um is not None:
        if not np.isfinite(pitch_um) or pitch_um <= 0:
            raise ValueError("pitch_um must be a positive finite scale")
        if pitch_um != 1.0:
            if verbose:
                print(f"Scaling coordinates by pitch_um={pitch_um}...")
            spot_coords = spot_coords * pitch_um

    if verbose:
        n_empty = int((spot_labels < 0).sum())
        n_cell_dnb = int(spot_labels.shape[0]) - n_empty
        print(f"Loaded {spot_expr.shape[0]} genes, {spot_expr.shape[1]} spots "
              f"({n_cell_dnb} cell + {n_empty} empty), "
              f"{len(cell_ids)} cells in {time.time() - t0:.1f}s")

    return {
        "spot_expr": spot_expr,
        "spot_coords": spot_coords,
        "spot_labels": spot_labels,
        "gene_names": gene_names,
        "cell_ids": cell_ids,
    }

def check_inputs(
    expression: np.ndarray,
    coordinates: np.ndarray,
    labels: np.ndarray,
) -> None:
    """Validate inputs against the SPARKLE input contract.

    Beyond shape alignment this enforces the README content requirements:
    raw non-negative finite counts, finite [N × 2] coordinates,
    integer-valued labels, and at least one cell (label >= 0) plus one
    out-of-mask location (label < 0).

    Args:
        expression: [genes × spots] expression matrix (dense or sparse).
        coordinates: [spots × 2] coordinates.
        labels: [spots] cell labels; -1 denotes an out-of-mask location.

    Raises:
        ValueError: If any input violates the contract.
    """
    if getattr(expression, "ndim", None) != 2:
        raise ValueError(
            f"expression must be 2-D [genes × spots], got shape "
            f"{getattr(expression, 'shape', None)}"
        )
    coordinates = np.asarray(coordinates)
    labels = np.asarray(labels)

    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError(
            f"coordinates must be [N × 2], got shape {coordinates.shape}"
        )
    n_spots = coordinates.shape[0]
    if expression.shape[1] != n_spots:
        raise ValueError(
            f"expression has {expression.shape[1]} spots but coordinates "
            f"have {n_spots}"
        )
    if labels.ndim != 1 or len(labels) != n_spots:
        raise ValueError(
            f"labels must be 1-D with one entry per spot, got shape "
            f"{labels.shape} for {n_spots} spots"
        )

    if not np.issubdtype(coordinates.dtype, np.number) or not np.all(
        np.isfinite(coordinates)
    ):
        raise ValueError("coordinates must contain only finite numeric values")

    values = expression.data if issparse(expression) else np.asarray(expression)
    if not np.issubdtype(values.dtype, np.number) or not np.all(
        np.isfinite(values)
    ):
        raise ValueError("expression must contain only finite numeric values")
    if np.any(values < 0):
        raise ValueError(
            "expression must be raw non-negative counts; found negative values"
        )

    if not np.issubdtype(labels.dtype, np.number) or not np.all(
        np.isfinite(labels)
    ):
        raise ValueError("labels must contain only finite numeric values")
    if not np.issubdtype(labels.dtype, np.integer):
        non_integral = labels != np.floor(labels)
        if np.any(non_integral):
            bad = labels[np.flatnonzero(non_integral)[0]]
            raise ValueError(
                f"labels must be integer-valued cell ids (-1 = out-of-mask); "
                f"found non-integer value {bad}"
            )
    if not np.any(labels >= 0):
        raise ValueError(
            "labels contain no cell (label >= 0); segmented cells are "
            "required as correction targets"
        )
    if not np.any(labels < 0):
        raise ValueError(
            "labels contain no out-of-mask location (label -1); empty "
            "locations are required as ambient probes"
        )


def sparse_to_dense_if_needed(mat) -> np.ndarray:
    """Convert a matrix to dense numpy if sparse."""
    if issparse(mat):
        return mat.toarray()
    return np.asarray(mat)


def save_results(
    corrected_expr,
    diagnostics: dict,
    output_prefix: str,
):
    """Save corrected expression and diagnostics to files.

    Args:
        corrected_expr: [genes × cells] corrected per-cell expression, dense
            numpy array (the ``fit_transform`` return type) or scipy sparse.
        diagnostics: Diagnostic dictionary.
        output_prefix: File path prefix.
    """
    import json

    # save_npz expects a sparse matrix; densify the sparse-on-disk contract.
    from scipy.sparse import save_npz

    matrix = (
        corrected_expr if issparse(corrected_expr) else csr_matrix(corrected_expr)
    )
    save_npz(f"{output_prefix}_corrected.npz", matrix)

    # Save diagnostics as JSON
    with open(f"{output_prefix}_diagnostics.json", "w") as f:
        json.dump(diagnostics, f, indent=2, default=str)


def load_visiumhd(
    h5_path: Union[str, Path],
    pixel_size_um: float = 2.0,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Load Visium HD data from a 10x Genomics-style HDF5 file.

    The expected HDF5 structure mirrors the output produced by Visium HD
    Space Ranger post-processing used in this project:

        masks/filtered              -> tissue pixel rows/cols
        feature_slices/<gene_id>    -> per-gene sparse pixel data
        features/name               -> gene names
        segmentations/cell_segmentation_mask -> cell segmentation

    Each pixel is treated as a DNB (like Stereo-seq's DNB concept), with
    coordinates returned in micrometers. Pixels that are not covered by any
    cell in the segmentation mask are labelled as empty (-1).

    Args:
        h5_path: Path to the Visium HD ``.h5`` file.
        pixel_size_um: Size of one Visium HD pixel in µm (default 2.0).
        verbose: Print progress messages.

    Returns:
        Dictionary with keys:

        - ``spot_expr``: [n_genes × n_spots] ``csr_matrix`` of UMI counts.
        - ``spot_coords``: [n_spots × 2] pixel center coordinates in µm.
        - ``spot_labels``: [n_spots] cell IDs; ``-1`` for empty pixels.
        - ``gene_names``: list of gene names.
        - ``cell_ids``: sorted array of original cell IDs.
    """
    try:
        import h5py
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "load_visiumhd requires h5py. Install with: pip install h5py"
        ) from e

    h5_path = Path(h5_path)
    if not h5_path.exists():
        raise FileNotFoundError(f"Visium HD file not found: {h5_path}")

    if verbose:
        print(f"Loading Visium HD data from {h5_path.name}...")

    f = h5py.File(h5_path, "r")
    t0 = time.time()

    # ── Step 1: tissue mask and pixel coordinates ────────────────────
    filt = f["masks/filtered"]
    pixel_rows = filt["row"][:].astype(np.int32)
    pixel_cols = filt["col"][:].astype(np.int32)
    n_pixels_total = len(pixel_rows)

    max_row = int(pixel_rows.max())
    max_col = int(pixel_cols.max())
    if verbose:
        print(f"  Tissue mask: {max_row + 1} x {max_col + 1}, "
              f"{n_pixels_total:,} pixels")

    pixel_to_idx = np.full((max_row + 1, max_col + 1), -1, dtype=np.int32)
    kept_indices = np.arange(n_pixels_total, dtype=np.int32)
    pixel_to_idx[pixel_rows, pixel_cols] = kept_indices
    spot_coords = np.column_stack([
        pixel_cols.astype(np.float64) * pixel_size_um + pixel_size_um / 2.0,
        pixel_rows.astype(np.float64) * pixel_size_um + pixel_size_um / 2.0,
    ])

    # ── Step 2: gene vocabulary and sparse matrix build ───────────────
    fs = f["feature_slices"]
    all_gene_indices = sorted(int(k) for k in fs.keys())
    n_all_genes = len(all_gene_indices)
    all_gene_names = [n.decode() for n in f["features/name"][:]]

    gene_indices = all_gene_indices
    n_genes_use = n_all_genes
    gene_to_new = {g: i for i, g in enumerate(gene_indices)}
    gene_names_arr = np.array([all_gene_names[g] for g in gene_indices])

    if verbose:
        print(f"  Loading all {n_genes_use} genes")

    # Count nonzeros for preallocation
    gene_nnz = np.zeros(n_genes_use, dtype=np.int32)
    for gi, gene_id in enumerate(all_gene_indices):
        g = fs[str(gene_id)]
        rows = g["row"][:].astype(np.int32)
        cols = g["col"][:].astype(np.int32)
        pix_indices = pixel_to_idx[rows, cols]
        gene_nnz[gi] = int((pix_indices >= 0).sum())

    total_nnz = int(gene_nnz.sum())
    if verbose:
        print(f"  Total nonzeros: {total_nnz:,}")

    coo_rows = np.empty(total_nnz, dtype=np.int32)
    coo_cols = np.empty(total_nnz, dtype=np.int32)
    coo_data = np.empty(total_nnz, dtype=np.float64)
    offset = 0

    if verbose:
        print("  Building sparse matrix...")
    for gene_id_orig in all_gene_indices:
        gi = gene_to_new[gene_id_orig]
        g = fs[str(gene_id_orig)]
        rows = g["row"][:].astype(np.int32)
        cols = g["col"][:].astype(np.int32)
        data = g["data"][:].astype(np.float64)

        pix_indices = pixel_to_idx[rows, cols]
        mask = pix_indices >= 0
        n_add = int(mask.sum())

        if n_add > 0:
            end = offset + n_add
            coo_rows[offset:end] = gi
            coo_cols[offset:end] = pix_indices[mask]
            coo_data[offset:end] = data[mask]
            offset = end

    spot_expr = csr_matrix(
        (coo_data, (coo_rows, coo_cols)),
        shape=(n_genes_use, n_pixels_total),
        dtype=np.float64,
    )

    # ── Step 3: assign cell labels from segmentation mask ─────────────
    if verbose:
        print("  Assigning cell labels from segmentation mask...")
    seg = f["segmentations/cell_segmentation_mask"]
    seg_rows = seg["row"][:].astype(np.int32)
    seg_cols = seg["col"][:].astype(np.int32)
    seg_data = seg["data"][:].astype(np.int64)

    pix_indices = pixel_to_idx[seg_rows, seg_cols]
    mask = pix_indices >= 0
    kept_pix = pix_indices[mask]
    kept_cell_ids = seg_data[mask]

    cell_ids_list = sorted(int(x) for x in np.unique(kept_cell_ids))
    cell_to_idx = {cid: i for i, cid in enumerate(cell_ids_list)}
    n_cells = len(cell_ids_list)

    spot_labels = np.full(n_pixels_total, -1, dtype=np.int32)
    cell_indices_mapped = np.array(
        [cell_to_idx[int(cid)] for cid in kept_cell_ids], dtype=np.int32
    )
    spot_labels[kept_pix.astype(np.int32)] = cell_indices_mapped

    f.close()

    if verbose:
        n_cell_pixels = int((spot_labels >= 0).sum())
        n_empty_pixels = int((spot_labels < 0).sum())
        print(f"  {n_cells:,} cells, {n_cell_pixels:,} cell pixels, "
              f"{n_empty_pixels:,} empty pixels")
        print(f"  Loaded in {time.time() - t0:.1f}s")

    return {
        "spot_expr": spot_expr,
        "spot_coords": spot_coords,
        "spot_labels": spot_labels,
        "gene_names": gene_names_arr,
        "cell_ids": np.array(cell_ids_list, dtype=np.int64),
    }


def _open_text_file(path: Union[str, Path]) -> Any:
    """Open a text file, transparently handling gzip compression."""
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def _detect_delimiter(header_line: str, sep: Optional[str] = None) -> str:
    """Detect CSV/TSV delimiter from a header line."""
    if sep is not None:
        return sep
    stripped = header_line.lstrip("#").strip()
    if "\t" in stripped:
        return "\t"
    if "," in stripped:
        return ","
    raise ValueError(
        f"Could not detect delimiter from header line: {header_line!r}. "
        "Specify sep explicitly."
    )


def _normalize_empty_labels(
    empty_labels: Optional[Union[int, List[int], Set[int]]]
) -> Set[int]:
    """Convert various empty-label specifications to a set of ints."""
    if empty_labels is None:
        return {0, -1}
    if isinstance(empty_labels, int):
        return {empty_labels}
    return set(int(x) for x in empty_labels)


def load_stereoseq(
    gem_path: Union[str, Path],
    gene_col: str = "geneID",
    x_col: str = "x",
    y_col: str = "y",
    count_col: str = "MIDCounts",
    cell_label_col: str = "cell",
    empty_labels: Optional[Union[int, List[int], Set[int]]] = None,
    sep: Optional[str] = None,
    pitch_um: Optional[float] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Load Stereo-seq GEM data from a text file.

    Supports plain text or gzip-compressed files (``.gz``).  Comment lines
    starting with ``#`` are skipped.  The delimiter is auto-detected as tab
    or comma unless ``sep`` is provided.

    Args:
        gem_path: Path to the GEM file (e.g. ``.txt``, ``.tsv``, ``.csv``,
            optionally ``.gz``).
        gene_col: Name of the gene-ID column.
        x_col: Name of the x-coordinate column.
        y_col: Name of the y-coordinate column.
        count_col: Name of the UMI count column.
        cell_label_col: Name of the cell-label column.  If this column is not
            present in the header, a ``ValueError`` is raised.
        empty_labels: Values that indicate an empty/background DNB.  Defaults
            to ``{0, -1}``.  Can be a single int or a list/set of ints.
        sep: Field delimiter.  Auto-detected if None.
        pitch_um: Spatial size of one raw coordinate unit in µm. Set to
            ``0.5`` for unscaled Stereo-seq DNB-index coordinates. If None,
            coordinates are assumed to already be in µm and are unchanged.
        verbose: Print progress messages.

    Returns:
        Dictionary with keys:

        - ``spot_expr``: [n_genes × n_spots] ``csr_matrix`` of UMI counts.
        - ``spot_coords``: [n_spots × 2] spot coordinates in µm.
        - ``spot_labels``: [n_spots] cell IDs; ``-1`` for empty spots.
        - ``gene_names``: array of gene names.
        - ``cell_ids``: sorted array of original cell IDs.
    """
    gem_path = Path(gem_path)
    if not gem_path.exists():
        raise FileNotFoundError(f"GEM file not found: {gem_path}")

    empty_set = _normalize_empty_labels(empty_labels)

    if verbose:
        print(f"Loading Stereo-seq GEM from {gem_path.name}...")

    t0 = time.time()

    with _open_text_file(gem_path) as f:
        # Skip comment lines and read the header
        header_line = None
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            header_line = stripped
            break

    if header_line is None:
        raise ValueError(f"No valid header found in {gem_path}")

    sep = _detect_delimiter(header_line, sep)
    header = header_line.split(sep)
    header = [h.strip() for h in header]

    required_cols = [gene_col, x_col, y_col, count_col, cell_label_col]
    missing = [c for c in required_cols if c not in header]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. Available columns: {header}"
        )

    col_idx = {c: i for i, c in enumerate(header)}
    ig = col_idx[gene_col]
    ix = col_idx[x_col]
    iy = col_idx[y_col]
    ic = col_idx[count_col]
    il = col_idx[cell_label_col]

    gene_to_idx: Dict[str, int] = {}
    dnb_to_idx: Dict[Tuple[int, int], int] = {}
    dnb_coords_list: List[Tuple[int, int]] = []
    dnb_orig_label: Dict[int, int] = {}
    counts: defaultdict = defaultdict(float)

    n_rows = 0
    with _open_text_file(gem_path) as f:
        # Skip header and comments
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            # first non-comment, non-empty line is the header -> skip it
            break

        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            parts = stripped.split(sep)
            if len(parts) <= max(ig, ix, iy, ic, il):
                continue  # malformed line

            gene = parts[ig]
            x = int(parts[ix])
            y = int(parts[iy])
            count = float(parts[ic])
            label = int(parts[il])

            if gene not in gene_to_idx:
                gene_to_idx[gene] = len(gene_to_idx)
            gidx = gene_to_idx[gene]

            d = (x, y)
            didx = dnb_to_idx.get(d)
            if didx is None:
                didx = len(dnb_to_idx)
                dnb_to_idx[d] = didx
                dnb_coords_list.append((x, y))
                dnb_orig_label[didx] = -1 if label in empty_set else label
            else:
                if dnb_orig_label[didx] == -1 and label not in empty_set:
                    dnb_orig_label[didx] = label

            counts[(gidx, didx)] += count
            n_rows += 1
            if verbose and n_rows % 5_000_000 == 0:
                print(f"  {n_rows/1e6:.1f}M rows, {len(gene_to_idx)} genes, "
                      f"{len(dnb_to_idx)} DNBs...")

    n_genes = len(gene_to_idx)
    n_dnbs = len(dnb_to_idx)
    if n_dnbs == 0:
        raise ValueError("No DNBs found in the GEM file.")

    gene_names = [None] * n_genes
    for g, idx in gene_to_idx.items():
        gene_names[idx] = g
    gene_names = np.array(gene_names)
    dnb_coords = np.array(dnb_coords_list, dtype=np.float64)

    # Remap original cell labels to 0..n_cells-1
    orig_labels = sorted({lbl for lbl in dnb_orig_label.values() if lbl >= 0})
    label_to_idx = {lbl: i for i, lbl in enumerate(orig_labels)}
    spot_labels = np.array(
        [label_to_idx.get(dnb_orig_label[i], -1) for i in range(n_dnbs)],
        dtype=np.int32,
    )
    cell_ids = np.array(orig_labels, dtype=np.int64)

    n_counts = len(counts)
    rows = np.fromiter((k[0] for k in counts.keys()), dtype=np.int32, count=n_counts)
    cols = np.fromiter((k[1] for k in counts.keys()), dtype=np.int32, count=n_counts)
    data = np.fromiter(counts.values(), dtype=np.float64, count=n_counts)
    spot_expr = csr_matrix((data, (rows, cols)), shape=(n_genes, n_dnbs))

    if pitch_um is not None:
        if not np.isfinite(pitch_um) or pitch_um <= 0:
            raise ValueError("pitch_um must be a positive finite scale")
        if pitch_um != 1.0:
            if verbose:
                print(f"  Scaling coordinates by pitch_um={pitch_um}...")
            dnb_coords = dnb_coords * pitch_um

    if verbose:
        n_empty = int((spot_labels < 0).sum())
        print(f"  Loaded {n_genes} genes, {n_dnbs} spots "
              f"({n_dnbs - n_empty} cell + {n_empty} empty), "
              f"{len(cell_ids)} cells in {time.time() - t0:.1f}s")

    return {
        "spot_expr": spot_expr,
        "spot_coords": dnb_coords,
        "spot_labels": spot_labels,
        "gene_names": gene_names,
        "cell_ids": cell_ids,
    }
