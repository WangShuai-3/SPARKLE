"""Predefined synthetic benchmark scenarios (S1–S10).

Every scenario now includes multiple cell types and marker genes, so the
benchmark tests ambient-RNA correction in a complex cellular environment
rather than only in homogeneous tissue. Each scenario still isolates one
additional variable (density, decay length, leakage rate, marker fraction,
or spatial clustering strength).

The scenarios are generated on a 500×500 DNB canvas (250 µm side at 0.5 µm
pitch); cell counts are scaled so that even the sparsest scenario provides
a few hundred cells.
"""

SCENARIOS = {
    "S1": {
        "name": "Sparse multi-type (40% empty)",
        "n_cells": 600,
        "empty_fraction": 0.40,
        "ambient_lambda": 50.0,
        "ambient_alpha": 0.01,
        "n_cell_types": 3,
        "marker_fraction": 0.20,
        "cluster_strength": 0.6,
    },
    "S2": {
        "name": "Medium multi-type (25% empty)",
        "n_cells": 900,
        "empty_fraction": 0.25,
        "ambient_lambda": 50.0,
        "ambient_alpha": 0.01,
        "n_cell_types": 3,
        "marker_fraction": 0.20,
        "cluster_strength": 0.6,
    },
    "S3": {
        "name": "Dense multi-type (10% empty)",
        "n_cells": 1400,
        "empty_fraction": 0.10,
        "ambient_lambda": 50.0,
        "ambient_alpha": 0.01,
        "n_cell_types": 3,
        "marker_fraction": 0.20,
        "cluster_strength": 0.6,
    },
    "S4": {
        "name": "Short lambda multi-type (20 µm)",
        "n_cells": 900,
        "empty_fraction": 0.25,
        "ambient_lambda": 20.0,
        "ambient_alpha": 0.01,
        "n_cell_types": 3,
        "marker_fraction": 0.20,
        "cluster_strength": 0.6,
    },
    "S5": {
        "name": "Long lambda multi-type (500 µm)",
        "n_cells": 900,
        "empty_fraction": 0.25,
        "ambient_lambda": 500.0,
        "ambient_alpha": 0.01,
        "n_cell_types": 3,
        "marker_fraction": 0.20,
        "cluster_strength": 0.6,
    },
    "S6": {
        "name": "Weak alpha multi-type (<=0.005)",
        "n_cells": 900,
        "empty_fraction": 0.25,
        "ambient_lambda": 50.0,
        "ambient_alpha": 0.005,
        "n_cell_types": 3,
        "marker_fraction": 0.20,
        "cluster_strength": 0.6,
    },
    "S7": {
        "name": "Strong alpha multi-type (<=0.10)",
        "n_cells": 900,
        "empty_fraction": 0.25,
        "ambient_lambda": 50.0,
        "ambient_alpha": 0.10,
        "n_cell_types": 3,
        "marker_fraction": 0.20,
        "cluster_strength": 0.6,
    },
    "S8": {
        "name": "Very sparse multi-type (>50% empty)",
        "n_cells": 400,
        "empty_fraction": 0.60,
        "ambient_lambda": 50.0,
        "ambient_alpha": 0.01,
        "n_cell_types": 3,
        "marker_fraction": 0.20,
        "cluster_strength": 0.6,
    },
    "S9": {
        "name": "High marker fraction (50% markers)",
        "n_cells": 900,
        "empty_fraction": 0.25,
        "ambient_lambda": 50.0,
        "ambient_alpha": 0.01,
        "n_cell_types": 3,
        "marker_fraction": 0.50,
        "cluster_strength": 0.6,
    },
    "S10": {
        "name": "Many cell types (5 types)",
        "n_cells": 900,
        "empty_fraction": 0.25,
        "ambient_lambda": 50.0,
        "ambient_alpha": 0.01,
        "n_cell_types": 5,
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
