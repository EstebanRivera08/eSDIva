"""
Example 6: Circular Transducer Pressure Fields

Computes and visualises the monochromatic pressure field for all three
circular transducer types available in PyField:

  1. FlatCircularTransducer    — flat piston disc
  2. ConcaveCircularTransducer — spherically focused bowl (TUS / HIFU)
  3. FocusedCircularTransducer — cylindrical line-focus

The XZ pressure plane is plotted for each transducer so the beam shape
and focal zone can be inspected.

Run with:
    uv run example6_circular_transducer.py
"""

import numpy as np

from pyfield.psimulation import PyField
from pyfield.transducers import (
    ConcaveCircularTransducer,
    ConvexCircularTransducer,
    FlatCircularTransducer,
    FocusedCircularTransducer,
)
from pyfield.utilities import plot_pressure_planes, plot_slices_2d

print("\n --- Example 6: Circular Transducer Pressure Fields --- \n")

C = 1540.0  # speed of sound (m/s)
FREQ_HZ = 1e6  # centre frequency
FREQ_SAMPLING_HZ = 100e6
DB_SCALE = False
VMAX = 1
VMIN = 0
COMPUTE_PRESSURE_FIELD = True
FIGSIZE = (6, 8)

# Common XZ simulation plane (Y fixed at 0)
XZ_GRID = {
    "x_extent": [-20, 20],
    "y_extent": [0, 0],
    "z_extent": [1, 80],
    "dx": 0.5,
    "dy": 0,
    "dz": 0.5,
}

# ===========================================================================
# 1. FlatCircularTransducer — flat piston
# ===========================================================================
print("\n--- 1. FlatCircularTransducer (flat piston, D=25 mm) ---")

flat = FlatCircularTransducer(
    diameter_mm=25.0,
    no_sub=30,
    frequency_Hz=FREQ_HZ,
)
flat.show()

if COMPUTE_PRESSURE_FIELD:
    sim_flat = PyField(flat, c=C, fs=FREQ_SAMPLING_HZ)
    x, y, z, p_flat = sim_flat(XZ_GRID, method="auto")

    plot_pressure_planes(
        x,
        y,
        z,
        p_flat / p_flat.max(),
        db_scale=DB_SCALE,
        vmin=VMIN,
        vmax=VMAX,
        figsize=FIGSIZE,
    )

# ===========================================================================
# 2. ConcaveCircularTransducer — spherical bowl
# ===========================================================================
print("\n--- 2. ConcaveCircularTransducer (bowl, D=40 mm, R=60 mm) ---")

bowl = ConcaveCircularTransducer(
    diameter_mm=40.0,
    radius_of_curvature_mm=60.0,  # geometric focus at 60 mm
    no_sub=30,
    frequency_Hz=FREQ_HZ,
)
bowl.show()

if COMPUTE_PRESSURE_FIELD:
    sim_bowl = PyField(bowl, c=C, fs=FREQ_SAMPLING_HZ)
    x, y, z, p_bowl = sim_bowl(XZ_GRID, method="auto")

    plot_pressure_planes(
        x,
        y,
        z,
        p_bowl / p_bowl.max(),
        db_scale=DB_SCALE,
        vmin=VMIN,
        vmax=VMAX,
        figsize=FIGSIZE,
    )

# ===========================================================================
# 3. FocusedCircularTransducer — cylindrical line focus
# ===========================================================================
print("\n--- 3. FocusedCircularTransducer (line focus, D=20 mm, R=40 mm, axis=y) ---")

cyl = FocusedCircularTransducer(
    diameter_mm=20.0,
    radius_of_curvature_mm=40.0,  # line focus at 40 mm
    no_sub=20,
    focus_axis="y",
    frequency_Hz=FREQ_HZ,
)
cyl.show()

if COMPUTE_PRESSURE_FIELD:
    sim_cyl = PyField(cyl, c=C, fs=FREQ_SAMPLING_HZ)
    x, y, z, p_cyl = sim_cyl(XZ_GRID, method="auto")

    plot_pressure_planes(
        x,
        y,
        z,
        p_cyl / p_cyl.max(),
        db_scale=DB_SCALE,
        vmin=VMIN,
        vmax=VMAX,
        figsize=FIGSIZE,
    )

# ===========================================================================
# 4. ConvexCircularTransducer -- lenses
# ===========================================================================
print("\n--- 4. ConvexCircularTransducer (line focus, D=20 mm, R=20 mm, axis=y) ---")

conv = ConvexCircularTransducer(
    diameter_mm=20.0,
    radius_of_curvature_mm=10.0,  # line focus at 40 mm
    no_sub=30,
    frequency_Hz=FREQ_HZ,
    border_refine=3,
    patch_fill=1,
    filled_radius_with_big_patches=0.8,
)
conv.show()

if COMPUTE_PRESSURE_FIELD:
    sim_conv = PyField(conv, c=C, fs=FREQ_SAMPLING_HZ)
    x, y, z, p_conv = sim_conv(XZ_GRID, method="auto")

    plot_pressure_planes(
        x,
        y,
        z,
        p_conv / p_conv.max(),
        db_scale=DB_SCALE,
        vmin=VMIN,
        vmax=VMAX,
        figsize=FIGSIZE,
    )


print("\nDone.")
