"""Plotting and helper utility functions."""

from typing import TYPE_CHECKING

from .helper_functions import (
    align_to_common_time,
    compute_sub_elem_attributes,
    create_spatial_grid_from_dict,
    reshape_to_mapped_points,
    to_dB,
)
from .matlab import explore_mat, mat_struct_fields, mat_struct_to_dict
from .phantom import make_phantom
from .surface_subdivision import subdivide_parametric_surface, subdivide_spherical_cap

if TYPE_CHECKING:
    from .bg_atlas import BG_Atlas


def __getattr__(name: str):
    """Import `BG_Atlas` only when it is first used.

    The brain-atlas wrapper needs the BrainGlobe packages, which ship as the
    optional `esdiva[atlas]` extra. Importing it lazily keeps `import esdiva`
    working on a core-only install, where nobody maps a field onto anatomy.
    """
    if name == "BG_Atlas":
        try:
            from .bg_atlas import BG_Atlas
        except ImportError as exc:  # BrainGlobe not installed.
            raise ImportError(
                "BG_Atlas needs the BrainGlobe packages, which ship as an "
                'optional extra. Install them with:  pip install "esdiva[atlas]"'
            ) from exc
        return BG_Atlas
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Brain Atlas
    "BG_Atlas",
    # Helpers
    "align_to_common_time",
    "to_dB",
    "compute_sub_elem_attributes",
    "create_spatial_grid_from_dict",
    "make_phantom",
    "subdivide_parametric_surface",
    "subdivide_spherical_cap",
    "reshape_to_mapped_points",
    # MATLAB .mat exploration
    "explore_mat",
    "mat_struct_fields",
    "mat_struct_to_dict",
]
