"""Spatial Impulse Response computation engine."""

from .farfield_rect_patch import compute_h_sir
from .h_sir import h_sir

__all__ = ["h_sir", "compute_h_sir"]
