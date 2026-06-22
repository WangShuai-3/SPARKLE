"""
SPARKLE: Spatial Ambient RNA Kernel-based Leakage Estimator

A method for removing ambient RNA contamination in high-resolution
spatial transcriptomics data (Stereo-seq, Visium HD) using empty
spots as built-in ambient probes.
"""

from .model import SPARKLE

__version__ = "0.1.0"
__all__ = ["SPARKLE"]
