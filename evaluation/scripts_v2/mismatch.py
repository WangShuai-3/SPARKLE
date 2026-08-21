#!/usr/bin/env python3
"""Diffusion-mismatch synthetic scenarios (M1/M2) for SPARKLE 2.x Phase 2.

The v1/2.x observation model assumes background-bin counts arise only from
spatially *local* leakage,  mu_b = A_b * alpha_g * S_gb.  Real tissue also
contains a *diffuse* (non-local) ambient component that the local model
either absorbs into an inflated alpha or misses entirely.  These scenarios
inject a uniform per-DNB Poisson component into the S2 layout so the
Poisson count model (B1 local-only vs B2 diffuse+local) can be evaluated
against a known truth.

    M1  base S2 (local leakage alpha=0.01) + uniform diffuse component
        calibrated at 0.5x the per-gene local ambient level in empty space.
    M2  clean S2 layout (ambient_alpha=0) + the same diffuse component:
        a world with *only* diffuse ambient.  Any local leakage detected
        here is a false positive.
    M3  S2 geometry with weak local leakage (alpha=0.002): evidence
        calibration near the detection limit (Phase 3).
    M4  S2 with 50%% UMI dropout: tests whether evidence degrades
        gracefully at low sequencing depth (Phase 3).

Ground truth per cell excludes the injected diffuse counts (they are
ambient, not cellular), so the same true_expr target applies.

The dict contract matches ``load_synthetic_scenario_data`` plus:
    true_beta:  [n_genes] per-DNB injected diffuse rate d_g.  Because the
        SPARKLE bin "area" equals the DNB count per bin, beta in the count
        model mu = A * (beta + rho * S) is exactly d_g.
"""

import numpy as np
from scipy.sparse import csr_matrix

from evaluation.scripts.final_comparison import SYNTHETIC_GRID_SIZE_DNB
from evaluation.synthetic.generator import generate_synthetic_data
from evaluation.synthetic.scenarios import get_scenario

MISMATCH_SCENARIOS = {
    "M1": {
        "name": "Local + diffuse (0.5x local level)",
        "base": "S2",
        "diffuse_fraction": 0.5,
        "drop_local": False,
    },
    "M2": {
        "name": "Diffuse-only (no local leakage)",
        "base": "S2",
        "diffuse_fraction": 0.5,
        "drop_local": True,
    },
    "M3": {
        "name": "Weak local leakage (alpha=0.002)",
        "base": "S2",
        "ambient_alpha": 0.002,
    },
    "M4": {
        "name": "Deep dropout (0.5, low evidence depth)",
        "base": "S2",
        "dropout_rate": 0.5,
    },
}

_DIFFUSE_SEED_OFFSET = 7919


def _generate_base(scenario, seed, ambient_alpha=None, dropout_rate=None):
    return generate_synthetic_data(
        n_cells=scenario.get("n_cells", 200),
        grid_width=SYNTHETIC_GRID_SIZE_DNB,
        grid_height=SYNTHETIC_GRID_SIZE_DNB,
        dnb_pitch=0.5,
        cell_radius=5.0,
        cell_radius_cv=0.2,
        n_genes=500,
        n_high_genes=80,
        ambient_lambda=scenario["ambient_lambda"],
        ambient_alpha=(
            scenario["ambient_alpha"] if ambient_alpha is None
            else ambient_alpha
        ),
        empty_fraction=scenario["empty_fraction"],
        n_cell_types=scenario.get("n_cell_types", 1),
        marker_fraction=scenario.get("marker_fraction", 0.0),
        cluster_strength=scenario.get("cluster_strength", 0.5),
        type_size_ratio=scenario.get("type_size_ratio", 1.0),
        dropout_rate=(
            scenario.get("dropout_rate", 0.0) if dropout_rate is None
            else dropout_rate
        ),
        seed=seed,
    )


def load_mismatch_scenario_data(mid, seed=42):
    spec = MISMATCH_SCENARIOS[mid]
    scenario = get_scenario(spec["base"])
    print(f"[Mismatch] Scenario {mid}: {spec['name']} (base={spec['base']})")

    # The full local-leakage run is RNG-identical to
    # load_synthetic_scenario_data(spec['base'], seed=seed).
    base = _generate_base(scenario, seed)
    empty_mask = base["dnb_labels"] < 0
    # Clean DNB expression at empty DNBs is exactly zero in the generator,
    # so the empty-DNB mean is the local ambient level per DNB.
    local_rate = base["dnb_expr"][:, empty_mask].mean(axis=1)
    true_beta = spec.get("diffuse_fraction", 0.0) * local_rate

    rng = np.random.RandomState(seed + _DIFFUSE_SEED_OFFSET)
    diffuse = rng.poisson(
        true_beta[:, None], base["dnb_expr"].shape
    ).astype(np.float64)

    if spec.get("drop_local", False):
        # Same layout/RNG stream but without local leakage; the injected
        # diffuse level is still calibrated on the local run above.
        clean = _generate_base(scenario, seed, ambient_alpha=0.0)
        source = clean
        true_alpha = np.zeros_like(base["true_alpha"])
        assert np.array_equal(clean["dnb_labels"], base["dnb_labels"])
    elif "ambient_alpha" in spec or "dropout_rate" in spec:
        # Generator-parameter override scenario (no diffuse injection).
        source = _generate_base(
            scenario, seed,
            ambient_alpha=spec.get("ambient_alpha"),
            dropout_rate=spec.get("dropout_rate"),
        )
        true_alpha = source["true_alpha"]
        assert np.array_equal(source["dnb_labels"], base["dnb_labels"])
    else:
        source = base
        true_alpha = base["true_alpha"]

    dnb_expr = source["dnb_expr"] + diffuse
    n_genes = dnb_expr.shape[0]
    n_cells = source["true_expr"].shape[1]

    print(f"  Injected diffuse: mean per-DNB rate {true_beta.mean():.4f} "
          f"(local level {local_rate.mean():.4f}, "
          f"fraction={spec.get('diffuse_fraction', 0.0)})")

    return {
        "dnb_expr": csr_matrix(dnb_expr.astype(np.float64)),
        "dnb_coords": base["dnb_coords"],
        "dnb_labels": base["dnb_labels"],
        "gene_names": np.array([f"gene_{i}" for i in range(n_genes)]),
        "cell_ids": np.arange(n_cells, dtype=np.int64),
        "true_expr": source["true_expr"],
        "scenario_id": mid,
        "gene_is_high": base["gene_is_high"],
        "cell_types": base["cell_types"],
        "marker_types": base["marker_types"],
        "true_alpha": true_alpha,
        "true_beta": true_beta,
        "true_lambda": float(scenario["ambient_lambda"]),
        "params": {**source["params"],
                   "mismatch_base": spec["base"],
                   "diffuse_fraction": spec.get("diffuse_fraction", 0.0),
                   "drop_local": spec.get("drop_local", False)},
    }
