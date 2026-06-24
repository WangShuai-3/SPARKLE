"""Synthetic data generation and benchmark scenarios for SPARKLE."""

from .generator import generate_synthetic_data
from .scenarios import SCENARIOS, list_scenarios, get_scenario

__all__ = ["generate_synthetic_data", "SCENARIOS", "list_scenarios", "get_scenario"]
