"""Plotting and helper utility functions."""

from .helper_functions import (
    compute_sub_elem_attributes,
    create_spatial_grid_from_dict,
    to_dB,
)
from .plotting import (
    plot_deltak_distribution,
    plot_pressure_2D,
    plot_pressure_field,
    plot_pressure_planes,
    plot_slices_2d,
)
from .plotting_pyvista import (
    add_2D_image,
    add_3D_vol,
    add_markers,
    add_pressure_vol,
    add_regions_mesh,
    add_stl_mesh,
    add_transducer_mesh,
    create_vol_mesh,
    load_stl_mesh,
    recompute_bounds,
)
from .surface_subdivision import subdivide_parametric_surface, subdivide_spherical_cap
from .transformation_functions import (
    compute_affine_from_markers,
    get_LabToTransducer,
)

__all__ = [
    # Matplotlib plotting
    "plot_pressure_field",
    "plot_pressure_planes",
    "plot_pressure_2D",
    "plot_slices_2d",
    "plot_deltak_distribution",
    # PyVista plotting
    "add_transducer_mesh",
    "add_pressure_vol",
    "add_regions_mesh",
    "add_3D_vol",
    "add_2D_image",
    "add_markers",
    "add_stl_mesh",
    "create_vol_mesh",
    "load_stl_mesh",
    "recompute_bounds",
    # Helpers
    "to_dB",
    "compute_sub_elem_attributes",
    "create_spatial_grid_from_dict",
    "subdivide_parametric_surface",
    "subdivide_spherical_cap",
    # Transforms
    "get_LabToTransducer",
    "compute_affine_from_markers",
]
