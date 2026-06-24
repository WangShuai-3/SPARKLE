"""Predefined synthetic benchmark scenarios (S1–S10).

Each scenario specifies the physical parameters used by the generator. The
`n_cells` parameter controls tissue density (sparse vs dense), while
`empty_fraction` is the final target fraction of empty DNBs (enforced globally
by the generator, either by removing covered DNBs or by assigning uncovered
DNBs to their nearest cell).
"""

SCENARIOS = {
    "S1": {
        "name": "Sparse (40% empty)",
        "n_cells": 120,
        "empty_fraction": 0.40,
        "ambient_lambda": 50.0,
        "ambient_alpha": 0.01,
        "n_cell_types": 1,
        "cluster_strength": 0.0,
    },
    "S2": {
        "name": "Medium (25% empty)",
        "n_cells": 180,
        "empty_fraction": 0.25,
        "ambient_lambda": 50.0,
        "ambient_alpha": 0.01,
        "n_cell_types": 1,
        "cluster_strength": 0.0,
    },
    "S3": {
        "name": "Dense (10% empty)",
        "n_cells": 280,
        "empty_fraction": 0.10,
        "ambient_lambda": 50.0,
        "ambient_alpha": 0.01,
        "n_cell_types": 1,
        "cluster_strength": 0.0,
    },
    "S4": {
        "name": "Short lambda (20 µm)",
        "n_cells": 180,
        "empty_fraction": 0.25,
        "ambient_lambda": 20.0,
        "ambient_alpha": 0.01,
        "n_cell_types": 1,
        "cluster_strength": 0.0,
    },
    "S5": {
        "name": "Long lambda (100 µm)",
        "n_cells": 180,
        "empty_fraction": 0.25,
        "ambient_lambda": 100.0,
        "ambient_alpha": 0.01,
        "n_cell_types": 1,
        "cluster_strength": 0.0,
    },
    "S6": {
        "name": "Weak alpha (<=0.005)",
        "n_cells": 180,
        "empty_fraction": 0.25,
        "ambient_lambda": 50.0,
        "ambient_alpha": 0.005,
        "n_cell_types": 1,
        "cluster_strength": 0.0,
    },
    "S7": {
        "name": "Strong alpha (<=0.10)",
        "n_cells": 180,
        "empty_fraction": 0.25,
        "ambient_lambda": 50.0,
        "ambient_alpha": 0.10,
        "n_cell_types": 1,
        "cluster_strength": 0.0,
    },
    "S8": {
        "name": "Very sparse (>50% empty)",
        "n_cells": 80,
        "empty_fraction": 0.60,
        "ambient_lambda": 50.0,
        "ambient_alpha": 0.01,
        "n_cell_types": 1,
        "cluster_strength": 0.0,
    },
    "S9": {
        "name": "Multi cell-type",
        "n_cells": 180,
        "empty_fraction": 0.25,
        "ambient_lambda": 50.0,
        "ambient_alpha": 0.01,
        "n_cell_types": 3,
        "cluster_strength": 0.6,
    },
    "S10": {
        "name": "Marker benchmark",
        "n_cells": 180,
        "empty_fraction": 0.25,
        "ambient_lambda": 50.0,
        "ambient_alpha": 0.01,
        "n_cell_types": 3,
        "marker_fraction": 0.20,
        "cluster_strength": 0.6,
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
