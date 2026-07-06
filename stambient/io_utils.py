"""I/O utilities for SPARKLE."""

import gzip
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Union, List, Set

import numpy as np
from scipy.sparse import csr_matrix, issparse


def check_inputs(
    expression: np.ndarray,
    coordinates: np.ndarray,
    labels: np.ndarray,
) -> None:
    """Validate input dimensions and types.

    Args:
        expression: [genes × spots] expression matrix.
        coordinates: [spots × 2] coordinates.
        labels: [spots] cell labels.

    Raises:
        ValueError: If inputs are inconsistent.
    """
    n_spots = coordinates.shape[0]

    if expression.shape[1] != n_spots:
        raise ValueError(
            f"Expression has {expression.shape[1]} spots but coordinates have {n_spots}"
        )
    if len(labels) != n_spots:
        raise ValueError(
            f"Labels has {len(labels)} entries but coordinates have {n_spots}"
        )
    if coordinates.shape[1] != 2:
        raise ValueError(f"Coordinates must be [N × 2], got {coordinates.shape}")


def sparse_to_dense_if_needed(mat) -> np.ndarray:
    """Convert a matrix to dense numpy if sparse."""
    if issparse(mat):
        return mat.toarray()
    return np.asarray(mat)


def save_results(
    corrected_expr: csr_matrix,
    diagnostics: dict,
    output_prefix: str,
):
    """Save corrected expression and diagnostics to files.

    Args:
        corrected_expr: [genes × cells] corrected per-cell expression.
        diagnostics: Diagnostic dictionary.
        output_prefix: File path prefix.
    """
    import json

    # Save sparse matrix
    from scipy.sparse import save_npz
    save_npz(f"{output_prefix}_corrected.npz", corrected_expr)

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
        verbose: Print progress messages.

    Returns:
        Dictionary with keys:

        - ``spot_expr``: [n_genes × n_spots] ``csr_matrix`` of UMI counts.
        - ``spot_coords``: [n_spots × 2] spot coordinates.
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
