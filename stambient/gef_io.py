"""Native BGI Stereo-seq GEF (HDF5) support for SPARKLE.

This module reads and writes the GEF file family used by BGI's SAW /
Stereopy ecosystem without requiring any compiled dependency (only
``h5py``, imported lazily):

- **Cellbin GEF** (``*.cellbin.gef``): per-cell expression produced by
  SAW cell segmentation.  Both the current ``/cellBin`` layout (GEF
  format versions 3/4, SAW >= 7.1, read by current gefpy/stereopy) and
  the legacy ``/cellExp`` + ``/cellData`` layout (SAW 5.x-7.0) are
  supported.
- **Bin GEF** (``*.raw.gef`` / ``*.tissue.gef`` / ``*.gef``): capture-
  location ("DNB") level expression under ``/geneExp/binN``.

``load_bgi_gef`` pairs a cellbin GEF with a bin GEF: DNB-level counts
come from the bin GEF and each DNB receives the label of the cell whose
border polygon contains it (border polygons are the 32-point
approximations stored in the cellbin GEF).  DNBs outside every polygon
form the out-of-mask observation layer (label ``-1``) required by the
SPARKLE input contract.

``save_cellbin_gef`` writes corrected gene-by-cell matrices back to a
current-layout cellbin GEF that stereopy (``st.io.read_gef(...,
bin_type='cell_bins')``) and StereoMap can open.  Because the official
format stores integer counts, corrected values are rounded; the exact
floating-point values can optionally be preserved in a non-standard
``/sparkleInfo`` group that official tools ignore.

The GEF layout implemented here follows the official "GEF (Cell Bin)
v2" specification and was validated against a SAW 8.1 demo file
(mouse whole brain, sample ``C04042E3``).
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.sparse import csr_matrix, issparse

__all__ = [
    "load_bgi_gef",
    "load_cellbin_gef",
    "read_cellbin_gef",
    "read_bin_gef",
    "save_cellbin_gef",
]

# Sentinel used by the official format to pad unused border vertices.
_BORDER_SENTINEL = 32767

# Compound dtypes of the current /cellBin layout (GEF versions 3/4).
_CELL_DTYPE = np.dtype([
    ("id", "<u4"),
    ("x", "<i4"),
    ("y", "<i4"),
    ("offset", "<u4"),
    ("geneCount", "<u2"),
    ("expCount", "<u2"),
    ("dnbCount", "<u2"),
    ("area", "<u2"),
    ("cellTypeID", "<u2"),
    ("clusterID", "<u2"),
])
_GENE_DTYPE = np.dtype([
    ("geneID", "S64"),
    ("geneName", "S64"),
    ("offset", "<u4"),
    ("cellCount", "<u4"),
    ("expCount", "<u4"),
    ("maxMIDcount", "<u2"),
])
_CELL_EXP_DTYPE = np.dtype([("geneID", "<u4"), ("count", "<u2")])
_GENE_EXP_DTYPE = np.dtype([("cellID", "<u4"), ("count", "<u2")])
_FLOAT_EXP_DTYPE = np.dtype([("geneID", "<u4"), ("value", "<f4")])

# Row-chunk size when streaming large bin-GEF expression datasets.
_SLICE_ROWS = 20_000_000


def _require_h5py():
    try:
        import h5py
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "GEF support requires h5py. Install with: pip install h5py"
        ) from e
    return h5py


def _decode(arr) -> np.ndarray:
    """Decode a bytes/string numpy array to a Python-str array."""
    if arr.dtype.kind in ("S", "O"):
        return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in arr])
    return arr


def _attr_int(attrs: Dict, key: str, default: Optional[int] = None) -> Optional[int]:
    if key not in attrs:
        return default
    return int(np.asarray(attrs[key]).ravel()[0])


def _um_per_unit(file_attrs: Dict, default_nm: int = 500) -> float:
    """Micrometres per raw coordinate unit from the ``resolution`` attr."""
    nm = _attr_int(file_attrs, "resolution", default_nm)
    if nm is None or nm <= 0:
        nm = default_nm
    return nm / 1000.0


def _resolve_pitch(file_attrs: Dict, pitch_um: Optional[float]) -> float:
    if pitch_um is not None:
        if not np.isfinite(pitch_um) or pitch_um <= 0:
            raise ValueError("pitch_um must be a positive finite scale")
        return float(pitch_um)
    return _um_per_unit(file_attrs)


def _borders_to_absolute_um(
    borders_raw: np.ndarray, centers_raw: np.ndarray, um_per_unit: float
) -> np.ndarray:
    """Convert stored (n, P, 2) int16 relative borders to absolute µm.

    Stored vertices are relative to each cell's centre of mass with the
    ``[32767, 32767]`` sentinel padding unused slots; the result is in
    µm with ``NaN`` padding.
    """
    sentinel = (borders_raw[:, :, 0] == _BORDER_SENTINEL) & (
        borders_raw[:, :, 1] == _BORDER_SENTINEL
    )
    abs_borders = borders_raw.astype(np.float64)
    abs_borders[:, :, 0] += centers_raw[:, 0:1]
    abs_borders[:, :, 1] += centers_raw[:, 1:2]
    abs_borders[sentinel] = np.nan
    return abs_borders * um_per_unit


def _borders_to_relative_int16(
    borders_um: np.ndarray, centers_um: np.ndarray, um_per_unit: float
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Convert absolute µm borders to the stored relative int16 layout.

    Returns the (n, P, 2) int16 array and the absolute border bounding
    box in raw units ``(min_x, max_x, min_y, max_y)`` for the attrs.
    """
    n, n_pts, _ = borders_um.shape
    centers_raw = np.rint(centers_um / um_per_unit)
    rel = np.rint(borders_um / um_per_unit) - centers_raw[:, None, :]
    pad = np.isnan(rel[:, :, 0]) | np.isnan(rel[:, :, 1])
    limit = _BORDER_SENTINEL - 1
    rel = np.clip(rel, -limit, limit)
    stored = np.full((n, n_pts, 2), _BORDER_SENTINEL, dtype=np.int16)
    valid = ~pad
    stored[valid, 0] = rel[valid, 0].astype(np.int16)
    stored[valid, 1] = rel[valid, 1].astype(np.int16)
    valid_coords = borders_um[valid] / um_per_unit
    if valid_coords.size:
        bbox = (
            int(np.floor(valid_coords[:, 0].min())),
            int(np.ceil(valid_coords[:, 0].max())),
            int(np.floor(valid_coords[:, 1].min())),
            int(np.ceil(valid_coords[:, 1].max())),
        )
    else:
        bbox = (0, 0, 0, 0)
    return stored, bbox


# ---------------------------------------------------------------------------
# Reading: cellbin GEF
# ---------------------------------------------------------------------------

def read_cellbin_gef(
    gef_path: Union[str, Path],
    pitch_um: Optional[float] = None,
    keep_borders: bool = True,
    prefer_stored_float: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Read a BGI cellbin GEF file.

    Both the current ``/cellBin`` layout (SAW >= 7.1 / GEF versions 3-4)
    and the legacy ``/cellExp`` + ``/cellData`` layout are recognised
    automatically.

    Args:
        gef_path: Path to the ``.cellbin.gef`` file.
        pitch_um: Micrometres per raw coordinate unit. If None, derived
            from the file's ``resolution`` attribute (nm), falling back
            to 0.5 µm (the Stereo-seq DNB pitch).
        keep_borders: Load cell border polygons (absolute coordinates in
            µm, ``NaN``-padded) into ``cell_borders``.
        prefer_stored_float: If the file contains a ``/sparkleInfo``
            sidecar written by :func:`save_cellbin_gef`, use its exact
            floating-point values instead of the rounded integer counts.
        verbose: Print progress messages.

    Returns:
        Dictionary with keys:

        - ``spot_expr``: [n_genes × n_cells] ``csr_matrix`` of counts
          (one column per cell).
        - ``spot_coords``: [n_cells × 2] cell centroids in µm.
        - ``spot_labels``: [n_cells] ``0..n_cells-1`` — every entry is a
          cell. NOTE: this dictionary alone does not satisfy the SPARKLE
          input contract, which also needs out-of-mask locations
          (``-1``); pair with a bin GEF via :func:`load_bgi_gef`.
        - ``gene_names`` / ``gene_ids``: gene symbols and accession ids.
        - ``cell_ids``: original cell ids (sorted for the legacy layout).
        - ``cell_areas``: cell areas in pixels (``-1`` when unknown).
        - ``cell_dnb_counts``: mRNA-captured DNB counts (``-1`` unknown).
        - ``cell_borders``: [n_cells × 32 × 2] border polygons in µm
          (``NaN`` padding), or ``None``.
        - ``resolution_nm``: stored resolution attribute.
        - ``format``: ``"cellBin"`` (current) or ``"legacy"``.
    """
    h5py = _require_h5py()
    gef_path = Path(gef_path)
    if not gef_path.exists():
        raise FileNotFoundError(f"Cellbin GEF not found: {gef_path}")

    if verbose:
        print(f"Reading cellbin GEF {gef_path.name}...")
    t0 = time.time()

    with h5py.File(gef_path, "r") as f:
        file_attrs = dict(f.attrs)
        if "cellBin" in f:
            parsed = _read_cellbin_current(f, gef_path)
            fmt = "cellBin"
        elif "cellExp" in f and "cellData" in f:
            parsed = _read_cellbin_legacy(f, gef_path, verbose)
            fmt = "legacy"
        else:
            raise ValueError(
                f"{gef_path} is not a recognised cellbin GEF: expected a "
                "'/cellBin' group (SAW >= 7.1) or '/cellExp' + '/cellData' "
                "groups (legacy layout)."
            )

        coords_raw = parsed.pop("coords_raw")
        borders_raw = parsed.pop("borders_raw", None)
        um_per_unit = _resolve_pitch(file_attrs, pitch_um)
        parsed["spot_coords"] = coords_raw * um_per_unit
        if borders_raw is not None:
            parsed["cell_borders"] = (
                _borders_to_absolute_um(borders_raw, coords_raw, um_per_unit)
                if keep_borders
                else None
            )
        else:
            parsed["cell_borders"] = None

        if prefer_stored_float and "sparkleInfo" in f:
            sidecar = f["sparkleInfo"].get("cellExpFloat")
            if sidecar is not None:
                parsed["spot_expr"] = _apply_float_sidecar(
                    parsed["spot_expr"], sidecar[:], verbose
                )
                parsed["has_stored_float"] = True

    parsed["resolution_nm"] = _attr_int(file_attrs, "resolution", 500)
    parsed["format"] = fmt
    if verbose:
        n_genes, n_cells = parsed["spot_expr"].shape
        print(
            f"  {n_genes:,} genes x {n_cells:,} cells, resolution "
            f"{parsed['resolution_nm']} nm, format {fmt}, "
            f"{time.time() - t0:.1f}s"
        )
    return parsed


#: Public alias matching the naming style of the other SPARKLE loaders.
load_cellbin_gef = read_cellbin_gef


def _read_cellbin_current(f, gef_path: Path) -> Dict[str, Any]:
    """Parse the current /cellBin layout (GEF versions 3/4)."""
    cb = f["cellBin"]
    cell = cb["cell"][:]
    gene = cb["gene"][:]
    cell_exp = cb["cellExp"][:]

    n_cells = len(cell)
    n_genes = len(gene)
    if n_cells == 0:
        raise ValueError(f"No cells found in {gef_path}")

    gene_counts = cell["geneCount"].astype(np.int64)
    if int(gene_counts.sum()) != len(cell_exp):
        raise ValueError(
            f"Corrupt cellbin GEF {gef_path}: cell geneCount entries "
            f"({int(gene_counts.sum()):,}) do not match cellExp rows "
            f"({len(cell_exp):,})"
        )
    cell_idx = np.repeat(np.arange(n_cells, dtype=np.int64), gene_counts)

    parsed = {
        "spot_expr": csr_matrix(
            (
                cell_exp["count"].astype(np.float64),
                (cell_exp["geneID"].astype(np.int64), cell_idx),
            ),
            shape=(n_genes, n_cells),
            dtype=np.float64,
        ),
        "spot_labels": np.arange(n_cells, dtype=np.int32),
        "gene_names": _decode(gene["geneName"]),
        "gene_ids": _decode(gene["geneID"]),
        "cell_ids": cell["id"].astype(np.int64),
        "cell_areas": cell["area"].astype(np.int64),
        "cell_dnb_counts": cell["dnbCount"].astype(np.int64),
        "coords_raw": np.column_stack(
            [cell["x"].astype(np.float64), cell["y"].astype(np.float64)]
        ),
    }
    if "cellBorder" in cb:
        parsed["borders_raw"] = cb["cellBorder"][:]
    return parsed


def _read_cellbin_legacy(f, gef_path: Path, verbose: bool) -> Dict[str, Any]:
    """Parse the legacy /cellExp + /cellData layout (SAW 5.x-7.0).

    Best effort: the layout predates the public specification.

    - ``/cellData``: ``cellId``, ``x``, ``y``, ``area``.
    - ``/cellExp``: ``cellId``, ``geneid``, ``count`` (+``geneName``).
    - ``/geneExp`` (optional): gene vocabulary ``geneName``.
    - ``/cellBorderList`` (optional): ``x``/``y`` int16 datasets with 32
      points per cell, relative to the centroid, sentinel padded.
    """
    cd = f["cellData"]
    ce = f["cellExp"]

    cell_ids_raw = cd["cellId"][:].astype(np.int64)
    coords_raw = np.column_stack(
        [cd["x"][:].astype(np.float64), cd["y"][:].astype(np.float64)]
    )
    n_cells = len(cell_ids_raw)
    if n_cells == 0:
        raise ValueError(f"No cells found in {gef_path}")
    areas = (
        cd["area"][:].astype(np.int64) if "area" in cd else np.full(n_cells, -1)
    )
    # Canonical order: cells sorted by original id, with their rows.
    cell_order = np.argsort(cell_ids_raw, kind="stable")
    cell_ids_raw = cell_ids_raw[cell_order]
    coords_raw = coords_raw[cell_order]
    areas = areas[cell_order]

    exp_cell_raw = ce["cellId"][:].astype(np.int64)
    gene_idx = ce["geneid"][:].astype(np.int64)
    counts = ce["count"][:].astype(np.float64)

    cell_pos = {int(c): i for i, c in enumerate(cell_ids_raw)}
    cell_idx = np.array([cell_pos[int(c)] for c in exp_cell_raw], dtype=np.int64)

    if "geneExp" in f and "geneName" in f["geneExp"]:
        vocab = _decode(f["geneExp/geneName"][:])
    elif "geneName" in ce:
        vocab = _decode(ce["geneName"][:])
    else:
        vocab = np.array([], dtype=object)
    n_genes = max(len(vocab), int(gene_idx.max()) + 1 if gene_idx.size else 0, 1)
    gene_names = np.array(
        [
            str(vocab[i]) if i < len(vocab) and str(vocab[i]) else f"gene_{i}"
            for i in range(n_genes)
        ]
    )

    parsed = {
        "spot_expr": csr_matrix(
            (counts, (gene_idx, cell_idx)), shape=(n_genes, n_cells)
        ),
        "spot_labels": np.arange(n_cells, dtype=np.int32),
        "gene_names": gene_names,
        "gene_ids": gene_names,
        "cell_ids": cell_ids_raw,
        "cell_areas": areas,
        "cell_dnb_counts": np.full(n_cells, -1, dtype=np.int64),
        "coords_raw": coords_raw,
    }
    if "cellBorderList" in f and "x" in f["cellBorderList"]:
        bl = f["cellBorderList"]
        bx = bl["x"][:]
        by = bl["y"][:]
        borders = np.full((n_cells, 32, 2), _BORDER_SENTINEL, dtype=np.int16)
        for i in range(n_cells):
            for p in range(32):
                k = i * 32 + p
                if k >= len(bx) or k >= len(by):
                    break
                borders[i, p, 0] = bx[k]
                borders[i, p, 1] = by[k]
        parsed["borders_raw"] = borders
    if verbose:
        print(f"  Legacy cellbin layout: {n_genes:,} genes x {n_cells:,} cells")
    return parsed


def _apply_float_sidecar(
    expr: csr_matrix, sidecar: np.ndarray, verbose: bool
) -> csr_matrix:
    """Substitute integer counts with exact float sidecar values.

    The sidecar mirrors the cellExp row order of the file, which is
    cell-major (cell rows in order, geneID ascending within each cell);
    ``expr`` was built from that same order, so the columns of its CSC
    representation reproduce it.
    """
    n_genes, n_cells = expr.shape
    csc = expr.tocsc()
    rows_per_cell = np.diff(csc.indptr)
    if int(rows_per_cell.sum()) != len(sidecar):
        if verbose:
            print(
                "  WARNING: float sidecar length mismatch; keeping integer counts"
            )
        return expr
    col_of_row = np.repeat(np.arange(n_cells, dtype=np.int64), rows_per_cell)
    return csr_matrix(
        (
            sidecar["value"].astype(np.float64),
            (sidecar["geneID"].astype(np.int64), col_of_row),
        ),
        shape=(n_genes, n_cells),
        dtype=np.float64,
    )


# ---------------------------------------------------------------------------
# Reading: bin GEF (DNB / square-bin level)
# ---------------------------------------------------------------------------

def read_bin_gef(
    gef_path: Union[str, Path],
    bin_size: Optional[int] = None,
    region_um: Optional[Sequence[float]] = None,
    pitch_um: Optional[float] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Read a BGI bin GEF (``.raw.gef`` / ``.tissue.gef`` / ``.gef``).

    Loads the expression matrix of one ``geneExp/binN`` group at the
    capture-location level.  Bin-GEF files carry no cell labels, so all
    locations are labelled ``-1``; use :func:`load_bgi_gef` to pair this
    layer with cellbin labels.

    Args:
        gef_path: Path to the bin GEF file.
        bin_size: Bin group to read (``1`` = DNB level). If None, the
            smallest available bin is used.
        region_um: Optional crop ``[min_x, max_x, min_y, max_y]`` in µm
            applied to the loaded locations (inclusive bounds). Strongly
            recommended for large chips; without it the whole bin group
            is loaded into memory.
        pitch_um: Micrometres per raw coordinate unit; defaults to the
            file's ``resolution`` attribute (nm), else 0.5 µm.
        verbose: Print progress messages.

    Returns:
        Dictionary with the standard loader keys plus ``bin_size`` and
        ``resolution_nm``. ``spot_labels`` is all ``-1``.
    """
    h5py = _require_h5py()
    gef_path = Path(gef_path)
    if not gef_path.exists():
        raise FileNotFoundError(f"Bin GEF not found: {gef_path}")

    if verbose:
        print(f"Reading bin GEF {gef_path.name}...")
    t0 = time.time()

    with h5py.File(gef_path, "r") as f:
        if "geneExp" not in f:
            raise ValueError(
                f"{gef_path} has no '/geneExp' group — it may be a cellbin "
                "GEF; use read_cellbin_gef instead."
            )
        file_attrs = dict(f.attrs)

        available = sorted(
            int(k[3:]) for k in f["geneExp"].keys() if k.startswith("bin")
        )
        if not available:
            raise ValueError(f"No geneExp/binN groups in {gef_path}")
        if bin_size is None:
            bin_size = available[0]
        if bin_size not in available:
            raise ValueError(
                f"bin_size={bin_size} not available in {gef_path}; "
                f"available bins: {available}"
            )

        grp = f[f"geneExp/bin{bin_size}"]
        gene = grp["gene"][:]
        expression = grp["expression"]
        gene_names = _decode(gene["geneName"])
        gene_ids = _decode(gene["geneID"])
        n_genes = len(gene)

        # Some writers (e.g. SAW raw.gef) store the resolution on the
        # expression dataset instead of the file attributes.
        if "resolution" not in file_attrs and "resolution" in expression.attrs:
            file_attrs["resolution"] = expression.attrs["resolution"]
        um_per_unit = _resolve_pitch(file_attrs, pitch_um)

        region_raw = None
        if region_um is not None:
            if len(region_um) != 4:
                raise ValueError("region_um must be [min_x, max_x, min_y, max_y]")
            region_raw = np.asarray(region_um, dtype=np.float64) / um_per_unit

        gene_idx_list: List[np.ndarray] = []
        x_list: List[np.ndarray] = []
        y_list: List[np.ndarray] = []
        count_list: List[np.ndarray] = []

        for gi in range(n_genes):
            start = int(gene[gi]["offset"])
            n_rows = int(gene[gi]["count"])
            pos = start
            while pos < start + n_rows:
                stop = min(pos + _SLICE_ROWS, start + n_rows)
                block = expression[pos:stop]
                xs = block["x"].astype(np.int64)
                ys = block["y"].astype(np.int64)
                cs = block["count"].astype(np.float64)
                if region_raw is not None:
                    keep = (
                        (xs >= region_raw[0])
                        & (xs <= region_raw[1])
                        & (ys >= region_raw[2])
                        & (ys <= region_raw[3])
                    )
                    if not np.any(keep):
                        pos = stop
                        continue
                    xs, ys, cs = xs[keep], ys[keep], cs[keep]
                gene_idx_list.append(np.full(xs.size, gi, dtype=np.int32))
                x_list.append(xs)
                y_list.append(ys)
                count_list.append(cs)
                pos = stop
            if verbose and (gi + 1) % 5000 == 0:
                print(f"  scanned {gi + 1:,}/{n_genes:,} genes...")

    if not gene_idx_list:
        raise ValueError(
            f"No expression rows found in {gef_path}"
            + (f" within region_um={list(region_um)}" if region_um is not None else "")
        )

    gene_idx_all = np.concatenate(gene_idx_list)
    x_all = np.concatenate(x_list)
    y_all = np.concatenate(y_list)
    counts_all = np.concatenate(count_list)
    del gene_idx_list, x_list, y_list, count_list

    # One column per unique capture location (x, y), matching the DNB
    # semantics of the GEM loaders.
    packed = (x_all.astype(np.uint64) << np.uint64(32)) | y_all.astype(np.uint64)
    uniq, inverse = np.unique(packed, return_inverse=True)
    n_spots = len(uniq)
    spot_coords = np.column_stack(
        [
            (uniq >> np.uint64(32)).astype(np.float64),
            (uniq & np.uint64(0xFFFFFFFF)).astype(np.float64),
        ]
    ) * um_per_unit

    spot_expr = csr_matrix(
        (counts_all, (gene_idx_all.astype(np.int64), inverse.astype(np.int64))),
        shape=(n_genes, n_spots),
        dtype=np.float64,
    )

    if verbose:
        print(
            f"  bin{bin_size}: {n_genes:,} genes x {n_spots:,} locations, "
            f"{spot_expr.nnz:,} counts, {time.time() - t0:.1f}s"
        )

    return {
        "spot_expr": spot_expr,
        "spot_coords": spot_coords,
        "spot_labels": np.full(n_spots, -1, dtype=np.int32),
        "gene_names": gene_names,
        "gene_ids": gene_ids,
        "cell_ids": np.array([], dtype=np.int64),
        "cell_areas": np.array([], dtype=np.int64),
        "cell_borders": None,
        "bin_size": bin_size,
        "resolution_nm": _attr_int(file_attrs, "resolution", 500),
        "format": "bin",
    }


# ---------------------------------------------------------------------------
# Point-in-polygon cell assignment
# ---------------------------------------------------------------------------

def _points_in_polygon(
    px: np.ndarray, py: np.ndarray, vx: np.ndarray, vy: np.ndarray
) -> np.ndarray:
    """Even-odd rule point-in-polygon test, vectorised over points."""
    inside = np.zeros(px.shape, dtype=bool)
    j = len(vx) - 1
    for i in range(len(vx)):
        yi, yj = vy[i], vy[j]
        if yi != yj:
            cond = (yi > py) != (yj > py)
            x_int = vx[i] + (py - yi) * (vx[j] - vx[i]) / (yj - yi)
            inside ^= cond & (px < x_int)
        j = i
    return inside


def _assign_points_to_cells(
    points: np.ndarray,
    cell_centers: np.ndarray,
    cell_borders: np.ndarray,
    grid_size: Optional[float] = None,
) -> np.ndarray:
    """Label points with the cell whose border polygon contains them.

    Args:
        points: [N × 2] coordinates (µm).
        cell_centers: [M × 2] cell centroids (µm).
        cell_borders: [M × P × 2] border polygons (µm, NaN padding).
        grid_size: Spatial-grid tile side in µm. Defaults to twice the
            median polygon width, keeping candidate lists small.

    Returns:
        [N] int64 labels; ``-1`` outside every polygon. Overlap between
        the simplified polygons is resolved to the nearest centroid.
    """
    n_points = points.shape[0]
    labels = np.full(n_points, -1, dtype=np.int64)
    n_cells = cell_centers.shape[0]
    if n_points == 0 or n_cells == 0:
        return labels

    valid = ~np.isnan(cell_borders[:, :, 0]) & ~np.isnan(cell_borders[:, :, 1])
    has_poly = valid.sum(axis=1) >= 3
    if not np.any(has_poly):
        return labels

    poly_min = np.where(valid[:, :, None], cell_borders, np.inf).min(axis=1)
    poly_max = np.where(valid[:, :, None], cell_borders, -np.inf).max(axis=1)
    finite = has_poly & np.isfinite(poly_min).all(axis=1) & np.isfinite(poly_max).all(axis=1)

    if grid_size is None:
        widths = poly_max[finite, 0] - poly_min[finite, 0]
        grid_size = float(2.0 * max(np.median(widths), 1e-6))

    origin = points.min(axis=0)
    cell_tile_min = np.floor((poly_min - origin) / grid_size).astype(np.int64)
    cell_tile_max = np.floor((poly_max - origin) / grid_size).astype(np.int64)

    tiles: Dict[Tuple[int, int], List[int]] = {}
    for ci in np.flatnonzero(finite):
        tx0, ty0 = cell_tile_min[ci]
        tx1, ty1 = cell_tile_max[ci]
        for tx in range(int(tx0), int(tx1) + 1):
            for ty in range(int(ty0), int(ty1) + 1):
                tiles.setdefault((int(tx), int(ty)), []).append(int(ci))

    point_tile = np.floor((points - origin) / grid_size).astype(np.int64)
    tile_key = point_tile[:, 0] * np.int64(2**31) + point_tile[:, 1]
    order = np.argsort(tile_key, kind="stable")
    sorted_key = tile_key[order]
    uniq_keys, starts = np.unique(sorted_key, return_index=True)
    ends = np.append(starts[1:], len(sorted_key))

    for uk, start, end in zip(uniq_keys, starts, ends):
        tx = int(uk // np.int64(2**31))
        ty = int(uk % np.int64(2**31))
        candidates = tiles.get((tx, ty))
        if not candidates:
            continue
        idx = order[start:end]
        px = points[idx, 0]
        py = points[idx, 1]
        best_label = np.full(len(idx), -1, dtype=np.int64)
        best_dist = np.full(len(idx), np.inf)
        for ci in candidates:
            vmask = valid[ci]
            poly = cell_borders[ci][vmask]
            hit = _points_in_polygon(px, py, poly[:, 0], poly[:, 1])
            if not np.any(hit):
                continue
            dist = (px - cell_centers[ci, 0]) ** 2 + (py - cell_centers[ci, 1]) ** 2
            closer = hit & (dist < best_dist)
            best_dist[closer] = dist[closer]
            best_label[closer] = ci
        labels[idx] = best_label
    return labels


# ---------------------------------------------------------------------------
# Paired loading (SPARKLE-ready)
# ---------------------------------------------------------------------------

def load_bgi_gef(
    cellbin_gef_path: Union[str, Path],
    raw_gef_path: Optional[Union[str, Path]] = None,
    bin_size: int = 1,
    region_um: Optional[Sequence[float]] = None,
    pitch_um: Optional[float] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Load BGI GEF data in the SPARKLE input format.

    With ``raw_gef_path`` provided (recommended), capture-location
    (DNB) level counts are loaded from the bin GEF and each location is
    assigned to the cell whose border polygon contains it; locations
    outside every polygon form the out-of-mask layer (``-1``).  The
    returned dictionary is ready for ``SPARKLE.fit_transform``.

    Without ``raw_gef_path`` the cellbin GEF alone is returned at cell
    level (every label >= 0), which does not satisfy the SPARKLE input
    contract — ambient-evidence locations only exist at the DNB layer.

    Border polygons are the 32-point approximations stored in the
    cellbin GEF, so labels near cell boundaries follow the simplified
    polygons rather than the exact segmentation masks.

    Args:
        cellbin_gef_path: Path to ``*.cellbin.gef``.
        raw_gef_path: Optional path to ``*.raw.gef`` / ``*.tissue.gef``.
        bin_size: Bin group of the raw GEF to use (1 = DNB level).
        region_um: Optional crop ``[min_x, max_x, min_y, max_y]`` in µm
            restricting the DNB layer; only cells whose borders meet the
            region are used for assignment.
        pitch_um: Micrometres per raw unit for both files. Defaults to
            each file's ``resolution`` attribute.
        verbose: Print progress messages.

    Returns:
        Dictionary with keys:

        - ``spot_expr``: [genes × locations] ``csr_matrix`` (raw GEF
          gene universe).
        - ``spot_coords``: [locations × 2] coordinates in µm.
        - ``spot_labels``: [locations] cell indices; ``-1`` out-of-mask.
        - ``gene_names`` / ``gene_ids``: raw-GEF gene symbols / ids.
        - ``cell_ids``: original cellbin ids of the cells carrying at
          least one in-mask DNB, in label order (i.e. matching the
          columns of the matrix returned by ``fit_transform``).
        - ``cell_coords``, ``cell_areas``, ``cell_dnb_counts``,
          ``cell_borders``: cellbin metadata (subset to cells inside the
          region when ``region_um`` is given).
        - ``n_mask_locations`` / ``n_out_of_mask_locations``: counts.
    """
    t0 = time.time()
    cellbin = read_cellbin_gef(cellbin_gef_path, pitch_um=pitch_um, verbose=verbose)

    if raw_gef_path is None:
        if verbose:
            print(
                "  NOTE: no raw GEF given; returning cell-level data only. "
                "SPARKLE fitting needs the DNB layer: pass raw_gef_path."
            )
        return cellbin

    raw = read_bin_gef(
        raw_gef_path,
        bin_size=bin_size,
        region_um=region_um,
        pitch_um=pitch_um,
        verbose=verbose,
    )

    borders = cellbin["cell_borders"]
    if borders is None:
        raise ValueError(
            "The cellbin GEF carries no cell border polygons "
            "('/cellBin/cellBorder' or '/cellBorderList'); DNB labels "
            "cannot be assigned. Use a GEM-based loader (load_stereoseq) "
            "for this sample."
        )

    # Restrict candidate cells to those whose borders meet the region.
    candidate = np.arange(len(borders))
    if region_um is not None:
        minx, maxx, miny, maxy = region_um
        valid = ~np.isnan(borders[:, :, 0]) & ~np.isnan(borders[:, :, 1])
        poly_min = np.where(valid[:, :, None], borders, np.inf).min(axis=1)
        poly_max = np.where(valid[:, :, None], borders, -np.inf).max(axis=1)
        candidate = np.flatnonzero(
            (poly_min[:, 0] <= maxx) & (poly_max[:, 0] >= minx)
            & (poly_min[:, 1] <= maxy) & (poly_max[:, 1] >= miny)
        )

    if verbose:
        print(f"Assigning DNB labels from border polygons "
              f"({len(candidate):,} candidate cells)...")
    raw_labels = _assign_points_to_cells(
        raw["spot_coords"],
        cellbin["spot_coords"][candidate],
        borders[candidate],
    )
    spot_labels = np.where(
        raw_labels >= 0, candidate[raw_labels], -1
    ).astype(np.int32)

    # Compact the labels to 0..K-1 over cells with >=1 in-mask DNB and
    # subset the cell metadata to the same order, so the columns of the
    # corrected matrix returned by SPARKLE.fit_transform align exactly
    # with cell_ids / cell_coords / ... .
    hit_cells = np.unique(spot_labels[spot_labels >= 0])
    remap = np.full(len(borders), -1, dtype=np.int32)
    remap[hit_cells] = np.arange(len(hit_cells), dtype=np.int32)
    # Guard against -1 labels wrapping around the fancy index.
    spot_labels = np.where(
        spot_labels >= 0, remap[np.clip(spot_labels, 0, None)], -1
    ).astype(np.int32)

    n_mask = int((spot_labels >= 0).sum())
    n_empty = int((spot_labels < 0).sum())
    if verbose:
        print(
            f"  {n_mask:,} in-mask / {n_empty:,} out-of-mask locations "
            f"({len(hit_cells):,} cells with >=1 DNB), {time.time() - t0:.1f}s total"
        )

    return {
        "spot_expr": raw["spot_expr"],
        "spot_coords": raw["spot_coords"],
        "spot_labels": spot_labels,
        "gene_names": raw["gene_names"],
        "gene_ids": raw["gene_ids"],
        "cell_ids": cellbin["cell_ids"][hit_cells],
        "cell_coords": cellbin["spot_coords"][hit_cells],
        "cell_areas": cellbin["cell_areas"][hit_cells],
        "cell_dnb_counts": cellbin["cell_dnb_counts"][hit_cells],
        "cell_borders": borders[hit_cells],
        "n_mask_locations": n_mask,
        "n_out_of_mask_locations": n_empty,
        "n_cells": int(len(hit_cells)),
        "bin_size_loaded": raw["bin_size"],
        "resolution_nm": raw["resolution_nm"],
    }


# ---------------------------------------------------------------------------
# Writing: cellbin GEF
# ---------------------------------------------------------------------------

def save_cellbin_gef(
    output_path: Union[str, Path],
    expr,
    gene_names: Sequence[str],
    cell_coords: np.ndarray,
    cell_ids: Optional[Sequence[int]] = None,
    cell_areas: Optional[np.ndarray] = None,
    cell_borders: Optional[np.ndarray] = None,
    cell_dnb_counts: Optional[np.ndarray] = None,
    gene_ids: Optional[Sequence[str]] = None,
    pitch_um: Optional[float] = None,
    resolution_nm: int = 500,
    sn: Optional[str] = None,
    omics: str = "Transcriptomics",
    store_float: Optional[bool] = None,
    source_info: Optional[Dict[str, Any]] = None,
    block_len: int = 256,
    verbose: bool = True,
) -> Path:
    """Write a gene-by-cell matrix as a current-layout cellbin GEF.

    The output follows the official ``/cellBin`` layout (GEF version 4)
    and is readable by stereopy (``st.io.read_gef(...,
    bin_type='cell_bins')``) and StereoMap.

    The official format stores unsigned 16-bit integer counts, so
    expression values are rounded to the nearest integer (negative
    values are clipped to 0).  When ``store_float`` is True the exact
    values are additionally kept as float32 in a non-standard
    ``/sparkleInfo`` group that official tools ignore and
    :func:`read_cellbin_gef` restores automatically.

    Args:
        output_path: Destination ``.cellbin.gef`` path.
        expr: [genes × cells] expression matrix (dense or scipy sparse),
            typically the corrected matrix from ``fit_transform``.
        gene_names: Gene symbols ordered like the rows of ``expr``.
        cell_coords: [cells × 2] cell centroids in µm.
        cell_ids: Original cell ids; defaults to ``0..n-1``.
        cell_areas: Cell areas in pixels; unknown entries pass ``-1``
            (a polygon-matching or default area is substituted).
        cell_borders: [cells × 32 × 2] border polygons in µm with NaN
            padding. When None, circular polygons matching each cell
            area are synthesised.
        cell_dnb_counts: mRNA-captured DNB counts per cell; defaults to
            the areas as an approximation when areas are known.
        gene_ids: Gene accession ids (e.g. Ensembl); defaults to
            ``gene_names``.
        pitch_um: Micrometres per raw coordinate unit written to the
            file. Defaults to ``resolution_nm / 1000``.
        resolution_nm: Raw-unit pitch in nanometres (default 500).
        sn: Chip serial number stored in the file attributes.
        omics: Omics label (default ``"Transcriptomics"``).
        store_float: Preserve exact values in ``/sparkleInfo``. ``None``
            (default) writes the sidecar only when non-integer values are
            present.
        source_info: Extra provenance attributes for ``/sparkleInfo``.
        block_len: Spatial block length in raw units (default 256).
        verbose: Print progress messages.

    Returns:
        The output path.
    """
    h5py = _require_h5py()
    output_path = Path(output_path)
    if output_path.parent and not output_path.parent.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)

    expr_csr = csr_matrix(expr) if issparse(expr) else csr_matrix(np.asarray(expr))
    expr_csr.data = np.clip(expr_csr.data, 0, None)
    expr_csr.eliminate_zeros()
    expr_csr = expr_csr.astype(np.float64)
    n_genes, n_cells = expr_csr.shape

    gene_names = np.asarray([str(g) for g in gene_names])
    if len(gene_names) != n_genes:
        raise ValueError(
            f"gene_names has {len(gene_names)} entries but expr has {n_genes} rows"
        )
    cell_coords = np.asarray(cell_coords, dtype=np.float64)
    if cell_coords.shape != (n_cells, 2):
        raise ValueError(
            f"cell_coords must be [{n_cells} × 2], got {cell_coords.shape}"
        )
    if cell_ids is None:
        cell_ids = np.arange(n_cells, dtype=np.int64)
    else:
        cell_ids = np.asarray(cell_ids, dtype=np.int64)
        if len(cell_ids) != n_cells:
            raise ValueError("cell_ids length must match the number of cells")
    if gene_ids is None:
        gene_ids_arr = gene_names
    else:
        gene_ids_arr = np.asarray([str(g) for g in gene_ids])
    um_per_unit = float(pitch_um) if pitch_um is not None else resolution_nm / 1000.0

    has_fractional = not np.all(np.equal(np.mod(expr_csr.data, 1), 0))
    if has_fractional and verbose:
        print(
            "  NOTE: non-integer values rounded to uint16 counts; exact "
            "floats stored in /sparkleInfo (store_float=True)"
        )
    if has_fractional and store_float is None:
        store_float = True
    elif not has_fractional and store_float is None:
        store_float = False
    if expr_csr.data.size and np.rint(expr_csr.data).max() > 65535:
        raise ValueError(
            "Expression values exceed the uint16 count range (65535) of the "
            "GEF format."
        )

    # ---- cell ordering: row-major over spatial blocks, then (y, x) ----
    x_raw = np.rint(cell_coords[:, 0] / um_per_unit).astype(np.int64)
    y_raw = np.rint(cell_coords[:, 1] / um_per_unit).astype(np.int64)
    min_x, min_y = int(x_raw.min()), int(y_raw.min())
    max_x, max_y = int(x_raw.max()), int(y_raw.max())
    block_cnt_x = max(1, int(np.ceil((max_x - min_x + 1) / block_len)))
    block_cnt_y = max(1, int(np.ceil((max_y - min_y + 1) / block_len)))
    bx = (x_raw - min_x) // block_len
    by = (y_raw - min_y) // block_len
    order = np.lexsort((x_raw, y_raw, by * block_cnt_x + bx))
    x_raw, y_raw = x_raw[order], y_raw[order]
    cell_ids = cell_ids[order]
    cell_coords_sorted = cell_coords[order]
    expr_csr = expr_csr[:, order]
    if cell_areas is not None:
        cell_areas = np.asarray(cell_areas, dtype=np.int64)[order]
    if cell_borders is not None:
        cell_borders = np.asarray(cell_borders, dtype=np.float64)[order]
    if cell_dnb_counts is not None:
        cell_dnb_counts = np.asarray(cell_dnb_counts, dtype=np.int64)[order]

    # ---- areas, borders, dnbCount ----
    if cell_areas is None:
        areas = np.full(n_cells, -1, dtype=np.int64)
    else:
        areas = np.asarray(cell_areas, dtype=np.int64).copy()
    unknown = areas < 0
    if np.any(unknown):
        if cell_borders is not None:
            valid = ~np.isnan(cell_borders[unknown, :, 0]) & ~np.isnan(
                cell_borders[unknown, :, 1]
            )
            for row, i in enumerate(np.flatnonzero(unknown)):
                poly = cell_borders[i][valid[row]]
                if len(poly) >= 3:
                    areas[i] = _polygon_area(poly)
        known = areas >= 0
        if not np.any(known):
            areas[:] = 100
        else:
            still_unknown = areas < 0
            if np.any(still_unknown):
                default_area = max(int(np.median(areas[known])), 1)
                areas[still_unknown] = default_area

    borders_um = (
        cell_borders if cell_borders is not None
        else _synthesise_borders(cell_coords_sorted, areas, um_per_unit)
    )
    border_ds, border_bbox = _borders_to_relative_int16(
        borders_um, cell_coords_sorted, um_per_unit
    )

    if cell_dnb_counts is None:
        dnb_counts = areas.copy()
    else:
        dnb_counts = np.asarray(cell_dnb_counts, dtype=np.int64).copy()
        dnb_counts[dnb_counts < 0] = 0

    # ---- expression rows, cell-major with geneID ascending per cell ----
    expr_csc = expr_csr.tocsc()
    rows_per_cell = np.diff(expr_csc.indptr).astype(np.int64)
    n_exp = int(rows_per_cell.sum())
    cell_of_row = np.repeat(np.arange(n_cells, dtype=np.int64), rows_per_cell)
    exp_gene = expr_csc.indices.astype(np.int64)
    exp_float = expr_csc.data.astype(np.float64)
    row_order = np.lexsort((exp_gene, cell_of_row))
    cell_of_row = cell_of_row[row_order]
    exp_gene = exp_gene[row_order]
    exp_float = exp_float[row_order]
    exp_count = np.rint(exp_float).astype(np.uint16)
    # geneExp rows: gene-major; the stable sort keeps cell ids ascending
    # within each gene because rows are already cell-major.
    gene_row_order = np.argsort(exp_gene, kind="stable")

    cell_gene_counts = rows_per_cell
    if cell_gene_counts.max(initial=0) > 65535:
        raise ValueError("A cell exceeds the uint16 geneCount range of GEF")
    cell_offsets = np.zeros(n_cells, dtype=np.uint32)
    if n_cells > 1:
        cell_offsets[1:] = np.cumsum(cell_gene_counts)[:-1]

    cell_exp_totals = np.zeros(n_cells, dtype=np.int64)
    np.add.at(cell_exp_totals, cell_of_row, exp_count.astype(np.int64))
    if cell_exp_totals.max(initial=0) > 65535:
        raise ValueError("A cell's total counts exceed the uint16 expCount range")

    # ---- gene-side index ----
    gene_cell_counts = np.asarray((expr_csr != 0).sum(axis=1)).ravel().astype(np.int64)
    gene_exp_totals = np.zeros(n_genes, dtype=np.int64)
    np.add.at(gene_exp_totals, exp_gene, exp_count.astype(np.int64))
    gene_max_mid = np.zeros(n_genes, dtype=np.int64)
    np.maximum.at(gene_max_mid, exp_gene, exp_count.astype(np.int64))
    if gene_exp_totals.max(initial=0) > 2**32 - 1:
        raise ValueError("A gene's total counts exceed the uint32 range")
    gene_offsets = np.zeros(n_genes, dtype=np.uint32)
    if n_genes > 1:
        gene_offsets[1:] = np.cumsum(gene_cell_counts)[:-1]

    cell_exp_ds = np.empty(n_exp, dtype=_CELL_EXP_DTYPE)
    cell_exp_ds["geneID"] = exp_gene.astype(np.uint32)
    cell_exp_ds["count"] = exp_count
    gene_exp_ds = np.empty(n_exp, dtype=_GENE_EXP_DTYPE)
    gene_exp_ds["cellID"] = cell_of_row[gene_row_order].astype(np.uint32)
    gene_exp_ds["count"] = exp_count[gene_row_order]

    gene_ds = np.empty(n_genes, dtype=_GENE_DTYPE)
    gene_ds["geneID"] = gene_ids_arr.astype("S64")
    gene_ds["geneName"] = gene_names.astype("S64")
    gene_ds["offset"] = gene_offsets
    gene_ds["cellCount"] = gene_cell_counts.astype(np.uint32)
    gene_ds["expCount"] = gene_exp_totals.astype(np.uint32)
    gene_ds["maxMIDcount"] = np.minimum(gene_max_mid, 65535).astype(np.uint16)

    # ---- block index (row-major over blocks) ----
    block_ids = by * block_cnt_x + bx
    counts_per_block = np.bincount(block_ids, minlength=block_cnt_x * block_cnt_y)
    block_index = np.concatenate([[0], np.cumsum(counts_per_block)]).astype(np.uint32)
    block_size = np.array(
        [block_len, block_len, block_cnt_x, block_cnt_y], dtype=np.uint32
    )

    # ---- cell compound ----
    cell_ds = np.empty(n_cells, dtype=_CELL_DTYPE)
    cell_ds["id"] = cell_ids.astype(np.uint32)
    cell_ds["x"] = x_raw.astype(np.int32)
    cell_ds["y"] = y_raw.astype(np.int32)
    cell_ds["offset"] = cell_offsets
    cell_ds["geneCount"] = cell_gene_counts.astype(np.uint16)
    cell_ds["expCount"] = cell_exp_totals.astype(np.uint16)
    cell_ds["dnbCount"] = np.minimum(dnb_counts, 65535).astype(np.uint16)
    cell_ds["area"] = np.minimum(areas, 65535).astype(np.uint16)
    cell_ds["cellTypeID"] = 0
    cell_ds["clusterID"] = 0

    t0 = time.time()
    if verbose:
        print(f"Writing cellbin GEF {output_path.name} "
              f"({n_genes:,} genes x {n_cells:,} cells)...")
    with h5py.File(output_path, "w") as f:
        # Numeric and string attributes are stored as 1-element arrays,
        # matching the official writer (readers index attrs[0]).
        f.attrs.create("bin_type", np.array([b"CellBin"], dtype="S32"))
        f.attrs.create("version", np.array([4], dtype=np.uint32))
        f.attrs.create("geftool_ver", np.array([1, 2, 2], dtype=np.uint32))
        f.attrs.create("omics", np.array([omics.encode()], dtype="S32"))
        if sn is not None:
            f.attrs.create("sn", str(sn))
        f.attrs.create("resolution", np.array([resolution_nm], dtype=np.uint32))
        f.attrs.create("offsetX", np.array([0], dtype=np.int32))
        f.attrs.create("offsetY", np.array([0], dtype=np.int32))
        f.attrs.create("maxX", np.array([max(max_x + 1, 0)], dtype=np.uint32))
        f.attrs.create("maxY", np.array([max(max_y + 1, 0)], dtype=np.uint32))

        cb = f.create_group("cellBin")
        _attach_cell_attrs(
            cb.create_dataset("cell", data=cell_ds),
            areas=areas, dnb=dnb_counts, exp=cell_exp_totals,
            genes=cell_gene_counts, x_raw=x_raw, y_raw=y_raw,
        )
        ce = cb.create_dataset("cellExp", data=cell_exp_ds)
        ce.attrs.create(
            "maxCount", np.array([int(exp_count.max(initial=0))], dtype=np.uint16)
        )
        g = cb.create_dataset("gene", data=gene_ds)
        expressed = gene_cell_counts > 0
        g.attrs.create(
            "maxCellCount",
            np.array([int(gene_cell_counts.max(initial=0))], dtype=np.uint32),
        )
        g.attrs.create(
            "maxExpCount",
            np.array([int(gene_exp_totals.max(initial=0))], dtype=np.uint32),
        )
        g.attrs.create(
            "minCellCount",
            np.array(
                [int(gene_cell_counts[expressed].min()) if np.any(expressed) else 0],
                dtype=np.uint32,
            ),
        )
        g.attrs.create(
            "minExpCount",
            np.array(
                [int(gene_exp_totals[expressed].min()) if np.any(expressed) else 0],
                dtype=np.uint32,
            ),
        )
        ge = cb.create_dataset("geneExp", data=gene_exp_ds)
        ge.attrs.create(
            "maxCount", np.array([int(exp_count.max(initial=0))], dtype=np.uint16)
        )
        bds = cb.create_dataset("cellBorder", data=border_ds)
        bds.attrs.create("minX", np.array([border_bbox[0]], dtype=np.int32))
        bds.attrs.create("maxX", np.array([border_bbox[1]], dtype=np.int32))
        bds.attrs.create("minY", np.array([border_bbox[2]], dtype=np.int32))
        bds.attrs.create("maxY", np.array([border_bbox[3]], dtype=np.int32))
        cb.create_dataset("blockIndex", data=block_index)
        cb.create_dataset("blockSize", data=block_size)
        cb.create_dataset(
            "cellTypeList", data=np.array([np.bytes_("default")], dtype="S32")
        )

        if store_float:
            info = f.create_group("sparkleInfo")
            float_ds = np.empty(n_exp, dtype=_FLOAT_EXP_DTYPE)
            float_ds["geneID"] = exp_gene.astype(np.uint32)
            float_ds["value"] = exp_float.astype(np.float32)
            fd = info.create_dataset("cellExpFloat", data=float_ds)
            fd.attrs.create(
                "description",
                "Exact expression values written by stambient "
                "save_cellbin_gef; row order matches /cellBin/cellExp.",
            )
            meta: Dict[str, str] = {"created_by": "stambient"}
            if source_info:
                meta.update({k: str(v) for k, v in source_info.items()})
            for k, v in meta.items():
                info.attrs.create(k, np.bytes_(v))

    if verbose:
        print(f"  Wrote {output_path} ({output_path.stat().st_size:,} bytes) "
              f"in {time.time() - t0:.1f}s")
    return output_path


def _polygon_area(pts: np.ndarray) -> int:
    x, y = pts[:, 0], pts[:, 1]
    return int(round(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))))


def _synthesise_borders(
    centers_um: np.ndarray, areas_px: np.ndarray, um_per_unit: float
) -> np.ndarray:
    """Circular 32-point polygons matching each cell's pixel area."""
    radii_um = np.sqrt(np.maximum(areas_px, 1) / np.pi) * um_per_unit
    angles = np.linspace(0, 2 * np.pi, 32, endpoint=False)
    borders = np.full((len(centers_um), 32, 2), np.nan)
    borders[:, :, 0] = (
        centers_um[:, 0:1] + radii_um[:, None] * np.cos(angles)[None, :]
    )
    borders[:, :, 1] = (
        centers_um[:, 1:2] + radii_um[:, None] * np.sin(angles)[None, :]
    )
    return borders


def _attach_cell_attrs(dataset, *, areas, dnb, exp, genes, x_raw, y_raw) -> None:
    """Attach the summary attributes of the /cellBin/cell dataset.

    Attributes are stored as 1-element arrays with the official dtypes
    (uint16 for per-cell statistics, int32 for the coordinate bounds,
    float32 for averages/medians).
    """

    def u16(name, arr, reducer):
        value = int(max(int(reducer(arr)) if arr.size else 0, 0))
        dataset.attrs.create(name, np.array([value], dtype=np.uint16))

    def f32(name, arr, statistic):
        value = float(statistic(arr)) if arr.size else 0.0
        dataset.attrs.create(name, np.array([value], dtype=np.float32))

    def i32(name, value):
        dataset.attrs.create(name, np.array([int(value)], dtype=np.int32))

    u16("maxArea", areas, np.max); u16("maxDnbCount", dnb, np.max)
    u16("maxExpCount", exp, np.max); u16("maxGeneCount", genes, np.max)
    u16("minArea", areas, np.min); u16("minDnbCount", dnb, np.min)
    u16("minExpCount", exp, np.min); u16("minGeneCount", genes, np.min)
    f32("averageArea", areas, np.mean); f32("averageDnbCount", dnb, np.mean)
    f32("averageExpCount", exp, np.mean); f32("averageGeneCount", genes, np.mean)
    f32("medianArea", areas, np.median)
    f32("medianDnbCount", dnb, np.median)
    f32("medianExpCount", exp, np.median)
    f32("medianGeneCount", genes, np.median)
    i32("minX", x_raw.min()); i32("maxX", x_raw.max())
    i32("minY", y_raw.min()); i32("maxY", y_raw.max())
