"""Plotting and helper utility functions."""

from .bg_atlas import BG_Atlas
from .helper_functions import (
    align_to_common_time,
    compute_sub_elem_attributes,
    create_spatial_grid_from_dict,
    reshape_to_mapped_points,
    to_dB,
)
from .matlab import explore_mat, mat_struct_fields, mat_struct_to_dict
from .surface_subdivision import subdivide_parametric_surface, subdivide_spherical_cap

__all__ = [
    # Brain Atlas
    "BG_Atlas",
    # Helpers
    "align_to_common_time",
    "to_dB",
    "compute_sub_elem_attributes",
    "create_spatial_grid_from_dict",
    "subdivide_parametric_surface",
    "subdivide_spherical_cap",
    "reshape_to_mapped_points",
    # MATLAB .mat exploration
    "explore_mat",
    "mat_struct_fields",
    "mat_struct_to_dict",
]
