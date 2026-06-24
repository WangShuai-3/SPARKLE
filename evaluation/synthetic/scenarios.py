"""Predefined synthetic benchmark scenarios (S1–S10).

Specifications shared across scenarios:
    - DNB grid: 200 x 200 (40,000 DNBs)
    - DNB pitch: 0.5 µm  -> field of view 100 µm x 100 µm
    - Genes: 500 (80 high-expression genes)
    - Cells: ~200 (after empty-cell culling ~108–306 remain)

Each scenario overrides a subset of the generator defaults to test a specific
aspect of ambient-RNA correction.
"""

SCENARIOS = {
    "S1": {
        "name": "Sparse (40% empty)",
        "empty_fraction": 0.40,
        "ambient_lambda": 50.0,
        "ambient_alpha": 0.01,
        "n_cell_types": 1,
    },
    "S2": {
        "name": "Medium (25% empty)",
        "empty_fraction": 0.25,
        "ambient_lambda": 50.0,
        "ambient_alpha": 0.01,
        "n_cell_types": 1,
    },
    "S3": {
        "name": "Dense (10% empty)",
        "empty_fraction": 0.10,
        "ambient_lambda": 50.0,
        "ambient_alpha": 0.01,
        "n_cell_types": 1,
    },
    "S4": {
        "name": "Short lambda (20 µm)",
        "empty_fraction": 0.25,
        "ambient_lambda": 20.0,
        "ambient_alpha": 0.01,
        "n_cell_types": 1,
    },
    "S5": {
        "name": "Long lambda (100 µm)",
        "empty_fraction": 0.25,
        "ambient_lambda": 100.0,
        "ambient_alpha": 0.01,
        "n_cell_types": 1,
    },
    "S6": {
        "name": "Weak alpha (<=0.005)",
        "empty_fraction": 0.25,
        "ambient_lambda": 50.0,
        "ambient_alpha": 0.005,
        "n_cell_types": 1,
    },
    "S7": {
        "name": "Strong alpha (<=0.10)",
        "empty_fraction": 0.25,
        "ambient_lambda": 50.0,
        "ambient_alpha": 0.10,
        "n_cell_types": 1,
    },
    "S8": {
        "name": "Very sparse (>50% empty)",
        "empty_fraction": 0.60,
        "ambient_lambda": 50.0,
        "ambient_alpha": 0.01,
        "n_cell_types": 1,
    },
    "S9": {
        "name": "Multi cell-type",
        "empty_fraction": 0.25,
        "ambient_lambda": 50.0,
        "ambient_alpha": 0.01,
        "n_cell_types": 3,
    },
    "S10": {
        "name": "Marker benchmark",
        "empty_fraction": 0.25,
        "ambient_lambda": 50.0,
        "ambient_alpha": 0.01,
        "n_cell_types": 3,
        "marker_fraction": 0.20,  # 20% of high genes are cell-type markers
    },
}


def list_scenarios():
    """Return scenario IDs in sorted order."""
    return sorted(SCENARIOS.keys())


def get_scenario(scenario_id):
    """Return a copy of the parameter dict for a given scenario ID."""
    if scenario_id not in SCENARIOS:
        raise ValueError(f"Unknown scenario '{scenario_id}'. Available: {list_scenarios()}")
    return dict(SCENARIOS[scenario_id])
