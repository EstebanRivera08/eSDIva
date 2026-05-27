"""Spatial Impulse Response computation engine."""

from .farfield_rect_patch import compute_h_sir
from .h_sir import h_sir
from .sir_derivatives import (
    compute_d2h,
    compute_d2h_per_element,
    compute_dh,
    compute_dh_per_element,
    compute_pe_sdi,
    integrate_d2h_to_dh,
    integrate_dh_to_h,
)

__all__ = [
    "h_sir",
    "compute_h_sir",
    "compute_d2h",
    "compute_dh",
    "compute_d2h_per_element",
    "compute_dh_per_element",
    "compute_pe_sdi",
    "integrate_d2h_to_dh",
    "integrate_dh_to_h",
]
