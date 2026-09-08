"""Adapters that run the *official R packages* SoupX and celda::decontX as
baselines for SPARKLE evaluation.

These functions are pure I/O adapters: all statistical estimation is done by
``evaluation/scripts/run_soupx_official.R`` (SoupX) and
``evaluation/scripts/run_decontx_official.R`` (celda).  They exist so the
formal benchmark can swap the community Python ports (soupx-python /
decontx-python) for the reference implementations while keeping the
``(corrected, diagnostics)`` return contract of the existing baselines.

Shared input contract with the R wrappers (mirrors spotclean_official.py):
``counts_csc.h5`` (genes × DNBs, CSC), ``genes.tsv``, ``cell_labels.tsv``
(-1 = empty DNB), ``n_cells.txt``.  Output: ``decont.h5`` + ``diagnostics.tsv``.
R/rhdf5 writes matrices column-major, so the (genes × cells) R matrix appears
as (cells × genes) through h5py and is transposed back on read.
"""

import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
from scipy.sparse import csc_matrix, issparse

from evaluation.baselines.spotclean_official import read_key_value_tsv

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
SOUPX_R_SCRIPT = SCRIPTS_DIR / "run_soupx_official.R"
DECONTX_R_SCRIPT = SCRIPTS_DIR / "run_decontx_official.R"

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
DEFAULT_RSCRIPT = str(
    next((path for path in _RSCRIPT_CANDIDATES if path.is_file()), "Rscript")
)


def _contiguous_labels(dnb_labels: np.ndarray) -> tuple[np.ndarray, int]:
    """Remap arbitrary cell IDs to 0-based contiguous indices.

    Some loaders (e.g. axolotl) return original cell IDs with gaps; the R
    wrappers build a (DNB x cells) indicator from label values directly, so
    non-contiguous IDs would create phantom empty cells.  Returns
    ``(remapped_labels, n_cells)``; empty DNBs keep label -1.
    """
    labels = np.asarray(dnb_labels, dtype=np.int64)
    unique = np.unique(labels[labels >= 0])
    if len(unique) == 0:
        return labels, 0
    mapping = {int(value): i for i, value in enumerate(unique)}
    remapped = np.array(
        [mapping.get(int(value), -1) for value in labels], dtype=np.int64
    )
    return remapped, len(unique)


def write_official_input(
    input_dir: Path,
    dnb_expr,
    dnb_labels: np.ndarray,
    gene_names: Iterable[str],
    n_cells: int,
) -> None:
    """Write DNB-level data in the format expected by the R wrappers."""
    input_dir = Path(input_dir)
    input_dir.mkdir(parents=True, exist_ok=True)
    matrix = dnb_expr if issparse(dnb_expr) else csc_matrix(dnb_expr)
    matrix = csc_matrix(matrix, dtype=np.float64)
    genes = np.asarray([str(value) for value in gene_names], dtype=object)
    labels = np.asarray(dnb_labels, dtype=np.int64)
    if matrix.shape != (len(genes), len(labels)):
        raise ValueError(
            "Expression matrix, genes, and cell labels have inconsistent shapes"
        )

    with h5py.File(input_dir / "counts_csc.h5", "w") as handle:
        handle.create_dataset("data", data=matrix.data, compression="gzip")
        handle.create_dataset(
            "indices", data=matrix.indices.astype(np.int32), compression="gzip"
        )
        handle.create_dataset(
            "indptr", data=matrix.indptr.astype(np.int64), compression="gzip"
        )
        handle.create_dataset(
            "shape", data=np.asarray(matrix.shape, dtype=np.int64)
        )
    (input_dir / "genes.tsv").write_text("\n".join(genes) + "\n", encoding="utf-8")
    (input_dir / "cell_labels.tsv").write_text(
        "\n".join(str(int(value)) for value in labels) + "\n", encoding="utf-8"
    )
    (input_dir / "n_cells.txt").write_text(f"{int(n_cells)}\n", encoding="utf-8")


def read_official_output(output_dir: Path) -> tuple[np.ndarray, dict]:
    """Read the corrected (genes × cells) matrix and R-side diagnostics."""
    output_dir = Path(output_dir)
    with h5py.File(output_dir / "decont.h5", "r") as handle:
        corrected = handle["X"][:].astype(np.float64).T  # h5py sees cells × genes
    diag = read_key_value_tsv(output_dir / "diagnostics.tsv")
    return corrected, diag


def _run_r(r_script: Path, input_dir: Path, output_dir: Path,
           extra_args: list, rscript: str) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Reuse a completed run when the R script and arguments match: the
    # mousebrain/ovarian decontX EM can take hours, so rerunning the whole
    # comparison after an unrelated failure must not redo finished work.
    import hashlib
    import json as _json
    manifest_key = hashlib.sha256(
        _json.dumps(
            {"script": r_script.name,
             "script_sha": hashlib.sha256(r_script.read_bytes()).hexdigest(),
             "args": [str(a) for a in extra_args]},
            sort_keys=True,
        ).encode()
    ).hexdigest()
    manifest_path = output_dir / "manifest.json"
    outputs_complete = (
        (output_dir / "decont.h5").exists()
        and (output_dir / "diagnostics.tsv").exists()
    )
    if (
        outputs_complete
        and manifest_path.exists()
        and _json.loads(manifest_path.read_text()).get("key") == manifest_key
    ):
        print(f"  [reuse] {r_script.name} output already complete: {output_dir}")
        return

    command = [rscript, str(r_script), str(Path(input_dir)), str(output_dir)]
    command += [str(arg) for arg in extra_args]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"{r_script.name} failed (exit {result.returncode}):\n"
            f"{result.stderr[-2000:]}"
        )
    manifest_path.write_text(_json.dumps({"key": manifest_key}))


def run_soupx_official(
    dnb_expr,
    dnb_labels: np.ndarray,
    gene_names,
    work_dir: Path,
    *,
    n_clusters: int = None,
    tfidf_min: float = 1.0,
    soup_quantile: float = 0.9,
    qmk_fdr: float = None,
    force_accept: bool = False,
    cont_max: float = 0.8,
    rscript: str = DEFAULT_RSCRIPT,
    verbose: bool = False,
) -> tuple[np.ndarray, float]:
    """Run official R SoupX on spatial DNB data.

    Mirrors the signature/behaviour of
    ``evaluation.baselines.soupx.run_soupx`` so call sites can swap
    implementations with identical parameters.  ``qmk_fdr`` relaxes the
    hypergeometric marker FDR (the Python port monkey-patches quickMarkers;
    the R wrapper overrides it in the SoupX namespace).

    Returns ``(corrected [genes × cells], rho)`` like the Python baseline.

    Raises:
        RuntimeError: if the R wrapper fails (the caller records NaN metrics).
    """
    labels, n_cells = _contiguous_labels(dnb_labels)
    if n_cells <= 0:
        raise RuntimeError("No labelled cells; SoupX cannot run")

    work_dir = Path(work_dir)
    input_dir = work_dir / "input"
    output_dir = work_dir / "output"
    write_official_input(input_dir, dnb_expr, labels, gene_names, n_cells)

    extra_args = [
        f"{tfidf_min:g}",
        f"{soup_quantile:g}",
        str(n_clusters) if n_clusters is not None else "NA",
        f"{(qmk_fdr if qmk_fdr is not None else 0.01):g}",
        "TRUE" if force_accept else "FALSE",
        f"{cont_max:g}",
    ]
    if verbose:
        print(f"  [SoupX R] input={input_dir} args={extra_args}")

    _run_r(SOUPX_R_SCRIPT, input_dir, output_dir, extra_args, rscript)
    corrected, diag = read_official_output(output_dir)
    rho = float(diag.get("rho", float("nan")))
    return corrected, rho


def run_decontx_official(
    dnb_expr,
    dnb_labels: np.ndarray,
    gene_names,
    work_dir: Path,
    *,
    rscript: str = DEFAULT_RSCRIPT,
    verbose: bool = False,
) -> tuple[np.ndarray, dict]:
    """Run official celda::decontX on spatial DNB data.

    The R wrapper aggregates DNBs to cells, clusters with kmeans on log1p,
    and runs ``celda::decontX(maxIter=200, seed=12345)``.

    Returns ``(corrected [genes × cells], diagonal dict)``.
    """
    labels, n_cells = _contiguous_labels(dnb_labels)
    if n_cells <= 0:
        raise RuntimeError("No labelled cells; DecontX cannot run")

    work_dir = Path(work_dir)
    input_dir = work_dir / "input"
    output_dir = work_dir / "output"
    write_official_input(input_dir, dnb_expr, labels, gene_names, n_cells)

    t0 = time.time()
    _run_r(DECONTX_R_SCRIPT, input_dir, output_dir, [], rscript)
    wall = time.time() - t0
    corrected, diag = read_official_output(output_dir)
    if verbose:
        print(
            f"  [DecontX R] mean contamination="
            f"{float(diag.get('mean_contamination', float('nan'))):.4f}"
        )
    return corrected, {
        "contamination": float(diag.get("mean_contamination", float("nan"))),
        "runtime": float(diag.get("runtime_seconds", wall)),
        "wall_runtime": wall,
    }
