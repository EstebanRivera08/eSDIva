"""Spatial Impulse Response computation engine."""

from .sir_spectral import compute_h_sir_spectrum
from .sir_temporal import compute_h_sir

__all__ = [
    "compute_h_sir",
    "compute_h_sir_spectrum",
]
