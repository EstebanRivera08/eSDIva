"""Causal power-law attenuation transfer functions and distance utilities."""

from .attenuation import (
    causal_attenuation_tf,
    compute_attenuation_distances,
    compute_reception_distances,
    convert_alpha0_to_nepers,
    reduce_patch_distances_to_element,
)

__all__ = [
    "causal_attenuation_tf",
    "compute_attenuation_distances",
    "compute_reception_distances",
    "convert_alpha0_to_nepers",
    "reduce_patch_distances_to_element",
]
