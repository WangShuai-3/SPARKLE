#!/usr/bin/env python3
"""Run the official R SpotClean package on synthetic and final real windows.

Python prepares cells as tissue spots and aggregated empty DNBs as background
spots, calls the standalone R wrapper, then writes h5ad output. It does not
implement or modify any SpotClean statistical routine.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import anndata as ad
import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.baselines.spotclean_official import (
    prepare_spotclean_spots,
    read_key_value_tsv,
    run_official_r,
    write_official_input,
)
from evaluation.scripts.final_comparison import (
    compute_cell_expr,
    load_axolotl_data_windowed,
    load_mousebrain_data,
    load_ovarian_data,
    load_synthetic_scenario_data,
)


R_SCRIPT = PROJECT_ROOT / "evaluation" / "scripts" / "run_spotclean_official.R"
REPORTS = PROJECT_ROOT / "evaluation" / "reports"
_CONDA = shutil.which("conda")
_RSCRIPT_CANDIDATES = [
    Path(sys.prefix) / "envs" / "spotclean-official" / "bin" / "Rscript",
    Path(sys.prefix) / "bin" / "Rscript",
]
if _CONDA:
    _RSCRIPT_CANDIDATES.insert(
        0,
        Path(_CONDA).resolve().parent.parent
        / "envs"
        / "spotclean-official"
        / "bin"
        / "Rscript",
    )
DEFAULT_RSCRIPT = str(next((path for path in _RSCRIPT_CANDIDATES if path.is_file()), "Rscript"))
PREPARATION_CONTRACT = "physical_um_independent_xy_bins_v2"

REAL_CONFIG = {
    "axolotl": {
        "x_range": (10500, 12500),
        "y_range": (6000, 11100),
        "candidate_radius": (10, 20, 30, 50, 70, 100, 150, 200),
        "empty_bin_size_um": 25.0,
        "source_coordinate_scale_to_um": 0.5,
    },
    "mousebrain": {
        "x_range": (12500, 20000),
        "y_range": (2000, 10000),
        "candidate_radius": (10, 20, 30, 50, 70, 100, 150, 200, 300),
        "empty_bin_size_um": 25.0,
        "source_coordinate_scale_to_um": 0.5,
        # Run nine spatial cores sequentially. Each core receives a halo equal
        # to the largest candidate radius, and only core cells are retained
        # during stitching. No cell or background bin is randomly discarded.
        "tile_grid": (3, 3),
        "tile_halo": 300.0,
    },
    "ovarian": {
        "x_range": (1000, 1800),
        "y_range": (300, 1100),
        "candidate_radius": (10, 20, 30, 50, 70, 100, 150, 200, 300),
        "empty_bin_size_um": 25.0,
        "source_coordinate_scale_to_um": 1.0,
    },
}


def _tag(name: str, config: dict) -> str:
    x0, x1 = config["x_range"]
    y0, y1 = config["y_range"]
    return f"{name}_x{x0}-{x1}_y{y0}-{y1}"


def _replace_string_dataset(group, key: str, value: str) -> None:
    if key in group:
        del group[key]
    dataset = group.create_dataset(key, data=value, dtype=h5py.string_dtype("utf-8"))
    dataset.attrs["encoding-type"] = "string"
    dataset.attrs["encoding-version"] = "0.2.0"


def _official_peak_memory_gb(n_all: int, n_tissue: int) -> float:
    """Conservative source-level estimate for official dense matrix operations."""
    # Distance and kernel (2*N^2), W_y and I1_y (2*N*T), and about five
    # tissue-square matrices are live during parameter estimation. The 1.5
    # multiplier accounts for R temporaries/copy-on-modify but not expression.
    dense_values = 2 * n_all**2 + 2 * n_all * n_tissue + 5 * n_tissue**2
    return dense_values * 8 * 1.5 / 1024**3


def _official_crossprod_flops(n_all: int, n_tissue: int, n_candidates: int) -> float:
    """Lower-bound FLOPs for only the official W_y cross-products."""
    return float(2 * n_candidates * n_all * n_tissue**2)


def _available_memory_gb() -> float:
    try:
        import psutil

        return psutil.virtual_memory().available / 1024**3
    except ImportError:
        return float("inf")


def _read_template_ids(raw_path: Path):
    raw = ad.read_h5ad(raw_path, backed="r")
    try:
        if "cell_id" not in raw.obs:
            raise ValueError(f"RAW template lacks obs['cell_id']: {raw_path}")
        return (
            raw.obs["cell_id"].to_numpy().copy(),
            raw.var_names.astype(str).to_numpy().copy(),
            raw.shape,
        )
    finally:
        raw.file.close()


def reuse_synthetic_h5ad_metrics(
    tag: str, true_expr: np.ndarray, rmse_raw: float, metrics: dict
) -> None:
    """Recover prior-method RMSEs from existing h5ad files without rerunning them."""
    for method in ("SPARKLE", "SoupX", "DecontX"):
        existing = metrics.get("methods", {}).get(method, {})
        if existing.get("error"):
            # The method failed on the CURRENT data; an h5ad from an earlier
            # run (different data realisation, same shape) must not
            # resurrect it.  Keep the explicit NaN/error record.
            continue
        path = REPORTS / "h5ad" / f"{tag}_{method}.h5ad"
        if not path.exists():
            continue
        result = ad.read_h5ad(path)
        matrix = result.X
        if hasattr(matrix, "toarray"):
            matrix = matrix.toarray()
        matrix = np.asarray(matrix, dtype=np.float64).T
        if matrix.shape != true_expr.shape:
            raise ValueError(f"Reused {method} h5ad has shape {matrix.shape}, expected {true_expr.shape}")
        rmse = float(np.sqrt(np.mean((matrix - true_expr) ** 2)))
        entry = metrics.setdefault("methods", {}).setdefault(method, {})
        entry["rmse"] = rmse
        entry["reduction_pct"] = (rmse_raw - rmse) / rmse_raw * 100.0
        entry["result_source"] = "reused_existing_h5ad"


def _write_real_h5ad(
    raw_path: Path,
    output_path: Path,
    official_dir: Path,
    diagnostics: dict,
    *,
    overwrite: bool,
) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite: {output_path}")
    template_cells, template_genes, template_shape = _read_template_ids(raw_path)
    output_genes = np.asarray(
        (official_dir / "genes.tsv").read_text(encoding="utf-8").splitlines()
    )
    output_barcodes = np.asarray(
        (official_dir / "tissue_barcodes.tsv").read_text(encoding="utf-8").splitlines()
    )
    expected_barcodes = np.asarray([f"cell_{value}" for value in template_cells])
    if not np.array_equal(output_barcodes, expected_barcodes):
        raise ValueError("Official SpotClean tissue barcode order differs from RAW h5ad")
    gene_lookup = {gene: index for index, gene in enumerate(template_genes)}
    try:
        template_columns = np.asarray([gene_lookup[gene] for gene in output_genes])
    except KeyError as exc:
        raise ValueError(f"Official output contains an unknown gene: {exc}") from exc
    if np.any(np.diff(template_columns) <= 0):
        raise ValueError("Official output genes are not in RAW-template order")

    temporary = output_path.with_name(f".{output_path.name}.tmp")
    if temporary.exists():
        temporary.unlink()
    shutil.copy2(raw_path, temporary)
    try:
        with h5py.File(official_dir / "decont.h5", "r") as source, h5py.File(
            temporary, "r+"
        ) as target:
            # rhdf5 presents R matrices as gene x tissue, while the same HDF5
            # dataset is tissue x gene through h5py (dimension order reversal).
            if source["X"].shape != (len(template_cells), len(output_genes)):
                raise ValueError("Official corrected matrix has an unexpected shape")
            if target["X"].shape != template_shape:
                raise ValueError("RAW h5ad X shape is inconsistent with AnnData metadata")
            # createSlide(gene_cutoff=0) can omit genes absent from all tissue
            # cells. The copied RAW template retains those zero/raw columns.
            batch = 128
            for start in range(0, len(output_genes), batch):
                end = min(start + batch, len(output_genes))
                columns = template_columns[start:end]
                values = source["X"][:, start:end].astype(np.float32, copy=False)
                if np.array_equal(columns, np.arange(columns[0], columns[0] + len(columns))):
                    target["X"][:, columns[0] : columns[-1] + 1] = values
                else:
                    for offset, column in enumerate(columns):
                        target["X"][:, column] = values[:, offset]
            _replace_string_dataset(target["uns"], "method", "SpotClean")
            _replace_string_dataset(target["uns"], "implementation", "official_R_package")
            _replace_string_dataset(
                target["uns"],
                "spotclean_diagnostics_json",
                json.dumps(diagnostics, sort_keys=True),
            )
            target.flush()
        os.replace(temporary, output_path)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def _make_spatial_tile_specs(
    spot_coords: np.ndarray,
    tissue: np.ndarray,
    config: dict,
) -> list[dict]:
    """Return deterministic core/halo membership for a tiled official run."""
    n_x, n_y = config["tile_grid"]
    scale = float(config["source_coordinate_scale_to_um"])
    x_edges = np.linspace(
        config["x_range"][0] * scale,
        config["x_range"][1] * scale,
        n_x + 1,
    )
    y_edges = np.linspace(
        config["y_range"][0] * scale,
        config["y_range"][1] * scale,
        n_y + 1,
    )
    halo = float(config["tile_halo"])
    tissue_indices = np.flatnonzero(tissue == 1)
    tissue_coords = spot_coords[tissue_indices]
    if np.any(tissue_coords[:, 0] < x_edges[0]) or np.any(
        tissue_coords[:, 0] > x_edges[-1]
    ):
        raise ValueError("Tissue centroids fall outside MouseBrain x tile bounds")
    if np.any(tissue_coords[:, 1] < y_edges[0]) or np.any(
        tissue_coords[:, 1] > y_edges[-1]
    ):
        raise ValueError("Tissue centroids fall outside MouseBrain y tile bounds")

    x_membership = np.searchsorted(x_edges[1:-1], tissue_coords[:, 0], side="right")
    y_membership = np.searchsorted(y_edges[1:-1], tissue_coords[:, 1], side="right")
    assigned = np.zeros(len(tissue_indices), dtype=np.int8)
    specs: list[dict] = []
    for row in range(n_y):
        for column in range(n_x):
            core_local = (x_membership == column) & (y_membership == row)
            assigned += core_local.astype(np.int8)
            core_tissue_indices = tissue_indices[core_local]
            x0, x1 = float(x_edges[column]), float(x_edges[column + 1])
            y0, y1 = float(y_edges[row]), float(y_edges[row + 1])
            context = (
                (spot_coords[:, 0] >= x0 - halo)
                & (spot_coords[:, 0] <= x1 + halo)
                & (spot_coords[:, 1] >= y0 - halo)
                & (spot_coords[:, 1] <= y1 + halo)
            )
            context_indices = np.flatnonzero(context)
            if len(core_tissue_indices) == 0:
                raise ValueError(f"MouseBrain tile r{row}c{column} has no core cells")
            if not np.all(np.isin(core_tissue_indices, context_indices)):
                raise ValueError(f"MouseBrain tile r{row}c{column} core is outside its halo")
            if not np.any(tissue[context_indices] == 0):
                raise ValueError(f"MouseBrain tile r{row}c{column} has no background spots")
            specs.append(
                {
                    "tile_id": f"r{row}c{column}",
                    "row": row,
                    "column": column,
                    "core_bounds": [x0, x1, y0, y1],
                    "halo": halo,
                    "context_indices": context_indices,
                    "core_tissue_indices": core_tissue_indices,
                }
            )
    if not np.all(assigned == 1):
        raise ValueError("Every MouseBrain tissue cell must belong to exactly one tile core")
    return specs


def _write_tiled_real_h5ad(
    raw_path: Path,
    output_path: Path,
    tiles_dir: Path,
    tile_records: list[dict],
    diagnostics: dict,
    *,
    overwrite: bool,
) -> None:
    """Stitch core-only tissue outputs from spatial tiles into the RAW template."""
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite: {output_path}")
    template_cells, template_genes, template_shape = _read_template_ids(raw_path)
    expected_barcodes = np.asarray([f"cell_{value}" for value in template_cells])
    barcode_to_row = {barcode: row for row, barcode in enumerate(expected_barcodes)}
    gene_to_column = {gene: column for column, gene in enumerate(template_genes)}
    covered = np.zeros(len(template_cells), dtype=bool)

    temporary = output_path.with_name(f".{output_path.name}.tmp")
    if temporary.exists():
        temporary.unlink()
    shutil.copy2(raw_path, temporary)
    try:
        with h5py.File(temporary, "r+") as target:
            if target["X"].shape != template_shape:
                raise ValueError("RAW h5ad X shape is inconsistent with AnnData metadata")
            for record in tile_records:
                tile_dir = tiles_dir / record["tile_id"]
                official_dir = tile_dir / "output"
                output_genes = np.asarray(
                    (official_dir / "genes.tsv").read_text(encoding="utf-8").splitlines()
                )
                output_barcodes = np.asarray(
                    (official_dir / "tissue_barcodes.tsv")
                    .read_text(encoding="utf-8")
                    .splitlines()
                )
                core_barcodes = np.asarray(
                    (tile_dir / "core_tissue_barcodes.tsv")
                    .read_text(encoding="utf-8")
                    .splitlines()
                )
                source_lookup = {
                    barcode: row for row, barcode in enumerate(output_barcodes)
                }
                try:
                    source_rows = np.asarray(
                        [source_lookup[barcode] for barcode in core_barcodes], dtype=np.int64
                    )
                    target_rows = np.asarray(
                        [barcode_to_row[barcode] for barcode in core_barcodes], dtype=np.int64
                    )
                    target_columns = np.asarray(
                        [gene_to_column[gene] for gene in output_genes], dtype=np.int64
                    )
                except KeyError as exc:
                    raise ValueError(
                        f"Unknown barcode/gene while stitching {record['tile_id']}: {exc}"
                    ) from exc
                if np.any(covered[target_rows]):
                    raise ValueError(f"Duplicate core cells while stitching {record['tile_id']}")
                if np.any(np.diff(source_rows) <= 0) or np.any(np.diff(target_rows) <= 0):
                    raise ValueError(f"Non-monotonic cell order in {record['tile_id']}")
                if np.any(np.diff(target_columns) <= 0):
                    raise ValueError(f"Non-monotonic gene order in {record['tile_id']}")

                with h5py.File(official_dir / "decont.h5", "r") as source:
                    if source["X"].shape != (len(output_barcodes), len(output_genes)):
                        raise ValueError(
                            f"Official corrected matrix has an unexpected shape in {record['tile_id']}"
                        )
                    batch = 128
                    for start in range(0, len(output_genes), batch):
                        end = min(start + batch, len(output_genes))
                        columns = target_columns[start:end]
                        values = source["X"][source_rows, start:end].astype(
                            np.float32, copy=False
                        )
                        if np.array_equal(
                            columns, np.arange(columns[0], columns[0] + len(columns))
                        ):
                            target["X"][target_rows, columns[0] : columns[-1] + 1] = values
                        else:
                            for offset, column in enumerate(columns):
                                target["X"][target_rows, column] = values[:, offset]
                covered[target_rows] = True

            if not np.all(covered):
                raise ValueError(f"Tiled stitching missed {np.sum(~covered)} MouseBrain cells")
            _replace_string_dataset(target["uns"], "method", "SpotClean")
            _replace_string_dataset(target["uns"], "implementation", "official_R_package")
            _replace_string_dataset(
                target["uns"],
                "spotclean_diagnostics_json",
                json.dumps(diagnostics, sort_keys=True),
            )
            target.flush()
        os.replace(temporary, output_path)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def _run_mousebrain_tiled(args) -> dict:
    """Run official SpotClean sequentially on 3x3 MouseBrain spatial tiles."""
    name = "mousebrain"
    config = REAL_CONFIG[name]
    tag = _tag(name, config)
    raw_path = REPORTS / "h5ad" / f"{tag}_raw.h5ad"
    output_path = REPORTS / "h5ad" / f"{tag}_SpotCleanOfficial.h5ad"
    if not raw_path.exists():
        raise FileNotFoundError(f"Existing RAW final-comparison h5ad not found: {raw_path}")

    print(f"\n{'=' * 78}\nOFFICIAL SPOTCLEAN TILED: {tag}\n{'=' * 78}", flush=True)
    started = time.time()
    work_dir = REPORTS / "spotclean_official" / tag
    tiles_dir = work_dir / "tiles"
    manifest_path = work_dir / "manifest.json"
    diagnostics = None
    if args.reuse_prepared_input and manifest_path.exists():
        candidate = json.loads(manifest_path.read_text(encoding="utf-8"))
        required = [
            tiles_dir / record["tile_id"] / "input" / "counts_csc.h5"
            for record in candidate.get("tiles", [])
        ]
        required.extend(
            tiles_dir / record["tile_id"] / "core_tissue_barcodes.tsv"
            for record in candidate.get("tiles", [])
        )
        if (
            candidate.get("preparation_contract") == PREPARATION_CONTRACT
            and
            candidate.get("tiling_strategy") == "3x3_core_with_radius_halo"
            and len(candidate.get("tiles", [])) == 9
            and all(path.exists() for path in required)
        ):
            diagnostics = candidate
            diagnostics.pop("reason", None)
            diagnostics.pop("status", None)
            print("Reusing nine validated MouseBrain tile inputs.", flush=True)

    if diagnostics is None:
        data = _load_real(name, config)
        template_cells, template_genes, _ = _read_template_ids(raw_path)
        if not np.array_equal(np.asarray(data["cell_ids"]), template_cells):
            raise ValueError("mousebrain: loader cell order differs from existing RAW h5ad")
        if not np.array_equal(
            np.asarray(data["gene_names"], dtype=str), template_genes
        ):
            raise ValueError("mousebrain: loader gene order differs from existing RAW h5ad")
        coordinate_scale_to_um = float(config["source_coordinate_scale_to_um"])
        coords_um = (
            np.asarray(data["dnb_coords"], dtype=np.float64)
            * coordinate_scale_to_um
        )
        spot_expr, coords, tissue, barcodes, prep = prepare_spotclean_spots(
            data["dnb_expr"],
            coords_um,
            data["dnb_labels"],
            data["cell_ids"],
            empty_bin_size=config["empty_bin_size_um"],
            coordinate_scale=1.0,
        )
        prep["spatial_unit"] = "micrometre"
        prep["source_coordinate_scale_to_um"] = coordinate_scale_to_um
        prep["empty_bin_size_um"] = float(config["empty_bin_size_um"])
        specs = _make_spatial_tile_specs(coords, tissue, config)
        tile_records = []
        for spec in specs:
            tile_dir = tiles_dir / spec["tile_id"]
            selected = spec["context_indices"]
            tile_tissue = tissue[selected]
            tile_barcodes = barcodes[selected]
            write_official_input(
                tile_dir / "input",
                spot_expr[:, selected],
                coords[selected],
                tile_tissue,
                tile_barcodes,
                template_genes,
                None,
            )
            core_barcodes = barcodes[spec["core_tissue_indices"]]
            (tile_dir / "core_tissue_barcodes.tsv").write_text(
                "\n".join(core_barcodes) + "\n", encoding="utf-8"
            )
            n_all = int(len(selected))
            n_tissue = int(np.sum(tile_tissue == 1))
            record = {
                "tile_id": spec["tile_id"],
                "row": spec["row"],
                "column": spec["column"],
                "core_bounds": spec["core_bounds"],
                "halo": spec["halo"],
                "n_core_tissue_spots": int(len(core_barcodes)),
                "n_all_spots": n_all,
                "n_tissue_spots": n_tissue,
                "n_background_spots": n_all - n_tissue,
                "estimated_official_peak_memory_gb": _official_peak_memory_gb(
                    n_all, n_tissue
                ),
                "estimated_crossprod_flops_lower_bound": _official_crossprod_flops(
                    n_all, n_tissue, len(config["candidate_radius"])
                ),
                "status": "prepared",
            }
            tile_records.append(record)
            print(
                f"Prepared {record['tile_id']}: {n_all:,} context spots "
                f"({n_tissue:,} tissue + {n_all - n_tissue:,} background), "
                f"{record['n_core_tissue_spots']:,} core cells, "
                f"~{record['estimated_official_peak_memory_gb']:.1f} GiB",
                flush=True,
            )
        diagnostics = {
            "dataset": tag,
            "preparation_contract": PREPARATION_CONTRACT,
            "implementation": "official_R_package",
            "r_wrapper": str(R_SCRIPT.relative_to(PROJECT_ROOT)),
            "rscript": args.rscript,
            "candidate_radius": list(config["candidate_radius"]),
            "candidate_radius_units": "micrometres_via_unit_coordinate_slope",
            "gene_keep_selection": "official_default_keepHighGene_per_tile",
            "tiling_strategy": "3x3_core_with_radius_halo",
            "tile_grid": list(config["tile_grid"]),
            "tile_halo": float(config["tile_halo"]),
            "tile_execution": "sequential",
            "stitching": "retain_core_tissue_cells_only_in_original_order",
            "n_input_genes": int(spot_expr.shape[0]),
            "n_tissue_spots": prep["n_tissue_spots"],
            "n_background_spots": prep["n_background_spots"],
            "n_all_spots_full_window": prep["n_all_spots"],
            "empty_bin_size_input_units": prep["empty_bin_size_input_units"],
            "coordinate_scale": prep["coordinate_scale"],
            "source_coordinate_scale_to_um": prep[
                "source_coordinate_scale_to_um"
            ],
            "spatial_unit": prep["spatial_unit"],
            "background_sampling_fraction": 1.0,
            "tiles": tile_records,
            "status": "prepared",
        }
        diagnostics["estimated_official_peak_memory_gb"] = max(
            record["estimated_official_peak_memory_gb"] for record in tile_records
        )
        diagnostics["estimated_crossprod_flops_lower_bound"] = sum(
            record["estimated_crossprod_flops_lower_bound"] for record in tile_records
        )
        work_dir.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(diagnostics, indent=2, sort_keys=True), encoding="utf-8"
        )
        del data, spot_expr

    max_official_spots = int(np.floor(np.sqrt(np.iinfo(np.int32).max)))
    available = _available_memory_gb()
    diagnostics["available_memory_before_R_gb"] = available
    diagnostics["official_dense_distance_max_spots"] = max_official_spots
    too_large = [
        record["tile_id"]
        for record in diagnostics["tiles"]
        if int(record["n_all_spots"]) > max_official_spots
    ]
    if too_large:
        raise RuntimeError(f"Official integer distance limit exceeded in tiles: {too_large}")
    peak = float(diagnostics["estimated_official_peak_memory_gb"])
    if args.prepare_only:
        diagnostics["status"] = "prepared_only"
        manifest_path.write_text(
            json.dumps(diagnostics, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(
            f"Prepared nine tiles; max estimated peak {peak:.1f} GiB, "
            f"total lower-bound work {diagnostics['estimated_crossprod_flops_lower_bound']:.2e} FLOPs.",
            flush=True,
        )
        return diagnostics
    if peak > available * args.memory_fraction and not args.force_memory:
        diagnostics["status"] = "skipped_memory_guard"
        diagnostics["reason"] = (
            f"Largest tile needs an estimated {peak:.1f} GiB, above the configured "
            "safe fraction of available memory."
        )
        manifest_path.write_text(
            json.dumps(diagnostics, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(f"SKIPPED by memory guard: {diagnostics['reason']}", flush=True)
        return diagnostics

    for record in diagnostics["tiles"]:
        tile_dir = tiles_dir / record["tile_id"]
        official_dir = tile_dir / "output"
        completed_files = [
            official_dir / "decont.h5",
            official_dir / "diagnostics.tsv",
            official_dir / "genes.tsv",
            official_dir / "tissue_barcodes.tsv",
        ]
        if args.reuse_prepared_input and not args.overwrite and all(
            path.exists() for path in completed_files
        ):
            print(f"Reusing completed {record['tile_id']} output.", flush=True)
            record["status"] = "completed"
            continue
        if args.overwrite:
            for stale in completed_files:
                if stale.exists():
                    stale.unlink()
        print(f"Running official SpotClean for {record['tile_id']} ...", flush=True)
        tile_started = time.time()
        run_official_r(
            tile_dir / "input",
            official_dir,
            r_script=R_SCRIPT,
            candidate_radius=config["candidate_radius"],
            maxit=args.maxit,
            tol=args.tol,
            rscript=args.rscript,
        )
        record.update(read_key_value_tsv(official_dir / "diagnostics.tsv"))
        record["tile_end_to_end_runtime_seconds"] = time.time() - tile_started
        record["status"] = "completed"
        manifest_path.write_text(
            json.dumps(diagnostics, indent=2, sort_keys=True), encoding="utf-8"
        )

    diagnostics["status"] = "completed"
    diagnostics["end_to_end_runtime_seconds"] = time.time() - started
    _write_tiled_real_h5ad(
        raw_path,
        output_path,
        tiles_dir,
        diagnostics["tiles"],
        diagnostics,
        overwrite=args.overwrite,
    )
    manifest_path.write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"Saved stitched official result: {output_path}", flush=True)
    return diagnostics


def _load_real(name: str, config: dict):
    if name == "axolotl":
        return load_axolotl_data_windowed(config["x_range"], config["y_range"])
    if name == "mousebrain":
        return load_mousebrain_data(config["x_range"], config["y_range"])
    # A large n_genes requests all genes while preserving the same total-count
    # ordering used by the existing full-gene ovarian final-comparison output.
    return load_ovarian_data(
        config["x_range"], config["y_range"], n_genes=10**9, verbose=True
    )


def run_real(name: str, args) -> dict:
    if name == "mousebrain" and "tile_grid" in REAL_CONFIG[name]:
        return _run_mousebrain_tiled(args)
    config = REAL_CONFIG[name]
    tag = _tag(name, config)
    raw_path = REPORTS / "h5ad" / f"{tag}_raw.h5ad"
    output_path = REPORTS / "h5ad" / f"{tag}_SpotCleanOfficial.h5ad"
    if not raw_path.exists():
        raise FileNotFoundError(f"Existing RAW final-comparison h5ad not found: {raw_path}")

    print(f"\n{'=' * 78}\nOFFICIAL SPOTCLEAN: {tag}\n{'=' * 78}", flush=True)
    started = time.time()
    work_dir = REPORTS / "spotclean_official" / tag
    input_dir = work_dir / "input"
    official_dir = work_dir / "output"
    manifest_path = work_dir / "manifest.json"
    if args.overwrite:
        for stale in (official_dir / "decont.h5", official_dir / "diagnostics.tsv"):
            if stale.exists():
                stale.unlink()
    required_prepared = [
        input_dir / "counts_csc.h5",
        input_dir / "genes.tsv",
        input_dir / "gene_keep.tsv",
        input_dir / "slide.csv",
        manifest_path,
    ]
    reuse_is_valid = False
    if args.reuse_prepared_input and all(path.exists() for path in required_prepared):
        candidate = json.loads(manifest_path.read_text(encoding="utf-8"))
        reuse_is_valid = (
            candidate.get("preparation_contract") == PREPARATION_CONTRACT
        )
        if not reuse_is_valid:
            print(
                "Prepared input uses an obsolete spatial-unit/binning contract; "
                "rebuilding it.",
                flush=True,
            )
    if reuse_is_valid:
        diagnostics = candidate
        diagnostics.setdefault(
            "n_input_genes",
            len((input_dir / "genes.tsv").read_text(encoding="utf-8").splitlines()),
        )
        diagnostics.pop("reason", None)
        diagnostics.pop("status", None)
        print(
            f"Reusing prepared input: {diagnostics['n_input_genes']:,} genes x "
            f"{diagnostics['n_all_spots']:,} spots",
            flush=True,
        )
    else:
        data = _load_real(name, config)
        template_cells, template_genes, _ = _read_template_ids(raw_path)
        if not np.array_equal(np.asarray(data["cell_ids"]), template_cells):
            raise ValueError(f"{name}: loader cell order differs from existing RAW h5ad")
        if not np.array_equal(np.asarray(data["gene_names"], dtype=str), template_genes):
            raise ValueError(f"{name}: loader gene order differs from existing RAW h5ad")

        coordinate_scale_to_um = float(config["source_coordinate_scale_to_um"])
        coords_um = (
            np.asarray(data["dnb_coords"], dtype=np.float64)
            * coordinate_scale_to_um
        )
        spot_expr, coords, tissue, barcodes, prep = prepare_spotclean_spots(
            data["dnb_expr"],
            coords_um,
            data["dnb_labels"],
            data["cell_ids"],
            empty_bin_size=config["empty_bin_size_um"],
            coordinate_scale=1.0,
        )
        prep["spatial_unit"] = "micrometre"
        prep["source_coordinate_scale_to_um"] = coordinate_scale_to_um
        prep["empty_bin_size_um"] = float(config["empty_bin_size_um"])
        write_official_input(
            input_dir, spot_expr, coords, tissue, barcodes, template_genes, None
        )
        memory_estimate = _official_peak_memory_gb(
            prep["n_all_spots"], prep["n_tissue_spots"]
        )
        diagnostics = {
            "dataset": tag,
            "preparation_contract": PREPARATION_CONTRACT,
            "implementation": "official_R_package",
            "r_wrapper": str(R_SCRIPT.relative_to(PROJECT_ROOT)),
            "rscript": args.rscript,
            "candidate_radius": list(config["candidate_radius"]),
            "candidate_radius_units": "micrometres_via_unit_coordinate_slope",
            "gene_keep_selection": "official_default_keepHighGene",
            "estimated_official_peak_memory_gb": memory_estimate,
            "estimated_crossprod_flops_lower_bound": _official_crossprod_flops(
                prep["n_all_spots"],
                prep["n_tissue_spots"],
                len(config["candidate_radius"]),
            ),
            "n_input_genes": int(spot_expr.shape[0]),
            **prep,
        }
        work_dir.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(diagnostics, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(
            f"Prepared {spot_expr.shape[0]:,} genes x {spot_expr.shape[1]:,} spots; "
            f"official dense peak estimate {memory_estimate:.1f} GiB",
            flush=True,
        )
        del data, spot_expr

    memory_estimate = float(diagnostics["estimated_official_peak_memory_gb"])
    available = _available_memory_gb()
    diagnostics["available_memory_before_R_gb"] = available
    diagnostics["rscript"] = args.rscript
    diagnostics.setdefault(
        "estimated_crossprod_flops_lower_bound",
        _official_crossprod_flops(
            int(diagnostics["n_all_spots"]),
            int(diagnostics["n_tissue_spots"]),
            len(config["candidate_radius"]),
        ),
    )

    # stats::as.matrix.dist() in the official implementation forms size^2
    # with a 32-bit integer. Fail clearly before entering R if preprocessing
    # ever produces more spots than that routine can address.
    max_official_spots = int(np.floor(np.sqrt(np.iinfo(np.int32).max)))
    diagnostics["official_dense_distance_max_spots"] = max_official_spots
    if int(diagnostics["n_all_spots"]) > max_official_spots:
        diagnostics["status"] = "blocked_official_integer_index_limit"
        diagnostics["reason"] = (
            "Official SpotClean calls as.matrix(dist(.)); its 32-bit size^2 "
            f"index cannot represent {diagnostics['n_all_spots']:,} spots "
            f"(maximum {max_official_spots:,})."
        )
        manifest_path.write_text(
            json.dumps(diagnostics, indent=2, sort_keys=True), encoding="utf-8"
        )
        raise RuntimeError(diagnostics["reason"])

    if args.prepare_only:
        flops = float(diagnostics["estimated_crossprod_flops_lower_bound"])
        if memory_estimate > available * args.memory_fraction:
            diagnostics["status"] = "blocked_official_memory"
            diagnostics["reason"] = (
                "The unmodified official package requires more dense-matrix RAM "
                "than the configured safe fraction of currently available memory."
            )
        elif flops > 1e13:
            diagnostics["status"] = "blocked_official_runtime"
            diagnostics["reason"] = (
                "The lower-bound W_y cross-products alone exceed 1e13 FLOPs; "
                "finishing the unchanged official candidate grid is not practical."
            )
        else:
            diagnostics["status"] = "prepared_only"
            diagnostics.pop("reason", None)
        manifest_path.write_text(
            json.dumps(diagnostics, indent=2, sort_keys=True), encoding="utf-8"
        )
        print("Prepared input only; official R execution was not requested.", flush=True)
        return diagnostics

    if memory_estimate > available * args.memory_fraction and not args.force_memory:
        diagnostics["status"] = "skipped_memory_guard"
        diagnostics["reason"] = (
            "Official SpotClean materializes full N-by-N and tissue-by-tissue "
            "dense matrices; estimated peak exceeds the configured safe fraction."
        )
        manifest_path.write_text(
            json.dumps(diagnostics, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(f"SKIPPED by memory guard: {diagnostics['reason']}", flush=True)
        return diagnostics

    run_official_r(
        input_dir,
        official_dir,
        r_script=R_SCRIPT,
        candidate_radius=config["candidate_radius"],
        maxit=args.maxit,
        tol=args.tol,
        rscript=args.rscript,
    )
    diagnostics.update(read_key_value_tsv(official_dir / "diagnostics.tsv"))
    diagnostics["status"] = "completed"
    diagnostics["end_to_end_runtime_seconds"] = time.time() - started
    _write_real_h5ad(
        raw_path, output_path, official_dir, diagnostics, overwrite=args.overwrite
    )
    manifest_path.write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"Saved official result: {output_path}", flush=True)
    return diagnostics


def run_synthetic(scenario_id: str, args) -> dict:
    print(f"\n{'=' * 78}\nOFFICIAL SPOTCLEAN: synthetic_{scenario_id}\n{'=' * 78}")
    data = load_synthetic_scenario_data(scenario_id, seed=42)
    cell_ids = np.asarray(data["cell_ids"])
    gene_names = np.asarray(data["gene_names"], dtype=str)
    radii = [10, 20, 30, 50, 70, 100, 150, 200, 300]
    true_lambda = int(round(float(data.get("true_lambda", 0))))
    if true_lambda > max(radii):
        radii.append(true_lambda)
        radii.sort()
    spot_expr, coords, tissue, barcodes, prep = prepare_spotclean_spots(
        data["dnb_expr"],
        data["dnb_coords"],
        data["dnb_labels"],
        cell_ids,
        empty_bin_size=25.0,
        coordinate_scale=1.0,
    )
    tag = f"synthetic_{scenario_id}"
    work_dir = REPORTS / "spotclean_official" / tag
    input_dir = work_dir / "input"
    official_dir = work_dir / "output"
    write_official_input(
        input_dir, spot_expr, coords, tissue, barcodes, gene_names, gene_names
    )
    started = time.time()
    run_official_r(
        input_dir,
        official_dir,
        r_script=R_SCRIPT,
        candidate_radius=radii,
        maxit=args.maxit,
        tol=args.tol,
        rscript=args.rscript,
    )
    r_diag = read_key_value_tsv(official_dir / "diagnostics.tsv")
    output_genes = (official_dir / "genes.tsv").read_text(encoding="utf-8").splitlines()
    output_barcodes = (
        official_dir / "tissue_barcodes.tsv"
    ).read_text(encoding="utf-8").splitlines()
    if output_barcodes != [f"cell_{value}" for value in cell_ids]:
        raise ValueError("Synthetic official output cell order is inconsistent")
    raw_cell = compute_cell_expr(data["dnb_expr"], data["dnb_labels"], len(cell_ids))
    corrected = raw_cell.copy()
    gene_lookup = {gene: index for index, gene in enumerate(gene_names)}
    with h5py.File(official_dir / "decont.h5", "r") as handle:
        for row, gene in enumerate(output_genes):
            corrected[gene_lookup[gene], :] = handle["X"][:, row]
    true_expr = np.asarray(data["true_expr"], dtype=np.float64)
    rmse_raw = float(np.sqrt(np.mean((raw_cell - true_expr) ** 2)))
    rmse = float(np.sqrt(np.mean((corrected - true_expr) ** 2)))
    reduction = (rmse_raw - rmse) / rmse_raw * 100.0
    diagnostics = {
        "dataset": tag,
        "implementation": "official_R_package",
        "rscript": args.rscript,
        "status": "completed",
        "candidate_radius": radii,
        "candidate_radius_units": "micrometres_via_unit_coordinate_slope",
        "gene_keep_selection": "all_genes_as_in_official_simulation_evaluation",
        "rmse_raw": rmse_raw,
        "rmse": rmse,
        "reduction_pct": reduction,
        "end_to_end_runtime_seconds": time.time() - started,
        **prep,
        **r_diag,
    }
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "manifest.json").write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True), encoding="utf-8"
    )

    h5ad_path = REPORTS / "h5ad" / f"{tag}_SpotCleanOfficial.h5ad"
    if h5ad_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite: {h5ad_path}")
    result = ad.AnnData(X=corrected.T.astype(np.float32, copy=False))
    result.obs_names = [f"Cell_{value}" for value in cell_ids]
    result.obs["cell_id"] = cell_ids
    result.var_names = gene_names
    result.uns["method"] = "SpotClean"
    result.uns["implementation"] = "official_R_package"
    result.uns["spotclean_diagnostics_json"] = json.dumps(diagnostics, sort_keys=True)
    h5ad_path.parent.mkdir(parents=True, exist_ok=True)
    result.write_h5ad(h5ad_path, compression="gzip")

    metrics_path = REPORTS / "metrics" / f"{tag}_metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    else:
        metrics = {
            "dataset": tag,
            "n_cells": len(cell_ids),
            "n_genes": len(gene_names),
            "rmse_raw": rmse_raw,
            "raw": {"rmse": rmse_raw},
            "methods": {},
        }
    reuse_synthetic_h5ad_metrics(tag, true_expr, rmse_raw, metrics)
    metrics.setdefault("methods", {})["SpotClean"] = {
        "rmse": rmse,
        "reduction_pct": reduction,
        "runtime": float(r_diag["runtime_seconds"]),
        "implementation": "official_R_package",
        "background_used": True,
        "n_background_spots": prep["n_background_spots"],
        "empty_bin_size": 25.0,
        "empty_exposure_normalized": False,
        "bleeding_rate": float(r_diag["bleeding_rate"]),
        "distal_rate": float(r_diag["distal_rate"]),
        "contamination_radius": float(r_diag["contamination_radius"]),
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(
        f"Saved {h5ad_path}; RMSE={rmse:.4f}, reduction={reduction:.1f}%",
        flush=True,
    )
    return diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["synthetic", "axolotl", "mousebrain", "ovarian"],
        choices=["synthetic", *REAL_CONFIG],
    )
    parser.add_argument("--scenarios", nargs="+", default=[f"S{i}" for i in range(1, 11)])
    parser.add_argument("--maxit", type=int, default=30)
    parser.add_argument("--tol", type=float, default=1.0)
    parser.add_argument(
        "--rscript",
        default=DEFAULT_RSCRIPT,
        help="Rscript executable containing the official SpotClean package",
    )
    parser.add_argument(
        "--memory-fraction",
        type=float,
        default=0.75,
        help="Skip a real run if estimated official peak exceeds this fraction of available RAM",
    )
    parser.add_argument(
        "--force-memory",
        action="store_true",
        help="Bypass the official dense-matrix memory guard (may OOM the host)",
    )
    parser.add_argument(
        "--reuse-prepared-input",
        action="store_true",
        help="Reuse a previously validated real-data CSC/slide input instead of rescanning DNBs",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Prepare or validate real-data input and write its feasibility manifest without running R",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.maxit <= 1:
        parser.error("--maxit must be greater than one")
    if not 0 < args.memory_fraction <= 1:
        parser.error("--memory-fraction must be in (0, 1]")

    summary = {}
    if "synthetic" in args.datasets:
        for scenario in args.scenarios:
            summary[f"synthetic_{scenario}"] = run_synthetic(scenario, args)
    for name in REAL_CONFIG:
        if name in args.datasets:
            summary[name] = run_real(name, args)
    summary_path = REPORTS / "spotclean_official" / "run_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    if summary_path.exists():
        previous = json.loads(summary_path.read_text(encoding="utf-8"))
        previous.update(summary)
        summary = previous
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nRun summary: {summary_path}")


if __name__ == "__main__":
    main()
