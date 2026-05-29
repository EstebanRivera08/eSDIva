"""
Example 2: Mono-element Circular Transducer Pressure Fields

Computes and visualises the monochromatic pressure field for all circular
transducer types available in PyField:

  1. FlatCircularTransducer    — flat piston disc
  2. ConcaveCircularTransducer — spherically focused bowl (TUS / HIFU)
  3. FocusedCircularTransducer — cylindrical line-focus
  4. ConvexCircularTransducer  — convex dome (lens)

Steps
-----
1. Create each circular transducer with representative parameters
2. Compute the CW pressure field on an XZ plane
3. Plot the normalised pressure field for each transducer

Run with:
    uv run examples/example2_monoelement_transducers.py
"""

from config import FIG_FOLDER, SAVE_FIG, SCALE

from pyfield.emission import Emission
from pyfield.plotting import plot2D_pressure_slices
from pyfield.transducers import (
    ConcaveCircularTransducer,
    ConvexCircularTransducer,
    FlatCircularTransducer,
    FocusedCircularTransducer,
)

# ============================================================================
# CONFIGURATION
# ============================================================================
# Physics constants
C = 1540.0  # Speed of sound (m/s)
FREQ_HZ = 1e6  # Centre frequency
FREQ_SAMPLING_HZ = 100e6  # Sampling frequency
DB_SCALE = False
VMAX = 1
VMIN = 0
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

print("\n --- Example 2: Mono-element Circular Transducer Pressure Fields --- \n")

scale = 1
if SAVE_FIG:
    FIG_FOLDER.mkdir(exist_ok=True)
    scale = SCALE

# ============================================================================
# STEP 1: FLAT CIRCULAR TRANSDUCER — FLAT PISTON
# ============================================================================
print("\n--- 1. FlatCircularTransducer (flat piston, D=25 mm) ---")

flat = FlatCircularTransducer(
    diameter_mm=25.0,
    no_sub_diameter=30,
    frequency_Hz=FREQ_HZ,
)
flat.show()

sim_flat = Emission(flat, c=C, fs=FREQ_SAMPLING_HZ, monochromatic=True)
p_flat, coords = sim_flat(XZ_GRID, method="auto")

plot2D_pressure_slices(
    p_flat / p_flat.max(),
    coords=coords,
    db_scale=DB_SCALE,
    vmin=VMIN,
    vmax=VMAX,
    figsize=FIGSIZE,
    save_path=str(FIG_FOLDER / "mono_flat.png") if SAVE_FIG else None,
)

# ============================================================================
# STEP 2: CONCAVE CIRCULAR TRANSDUCER — SPHERICAL BOWL
# ============================================================================
print("\n--- 2. ConcaveCircularTransducer (bowl, D=40 mm, R=60 mm) ---")

bowl = ConcaveCircularTransducer(
    diameter_mm=40.0,
    focus_mm=60.0,
    no_sub_diameter=30,
    frequency_Hz=FREQ_HZ,
)
bowl.show()

sim_bowl = Emission(bowl, c=C, fs=FREQ_SAMPLING_HZ, monochromatic=True)
p_bowl, coords = sim_bowl(XZ_GRID, method="auto")

plot2D_pressure_slices(
    p_bowl / p_bowl.max(),
    coords=coords,
    db_scale=DB_SCALE,
    vmin=VMIN,
    vmax=VMAX,
    figsize=FIGSIZE,
    save_path=str(FIG_FOLDER / "mono_concave.png") if SAVE_FIG else None,
)

# ============================================================================
# STEP 3: FOCUSED CIRCULAR TRANSDUCER — CYLINDRICAL LINE FOCUS
# ============================================================================
print("\n--- 3. FocusedCircularTransducer (line focus, D=20 mm, R=40 mm, axis=y) ---")

cyl = FocusedCircularTransducer(
    diameter_mm=20.0,
    focus_mm=40.0,
    no_sub_diameter=20,
    focus_axis="x",
    frequency_Hz=FREQ_HZ,
)
cyl.show()

sim_cyl = Emission(cyl, c=C, fs=FREQ_SAMPLING_HZ, monochromatic=True)
p_cyl, coords = sim_cyl(XZ_GRID, method="auto")

plot2D_pressure_slices(
    p_cyl / p_cyl.max(),
    coords=coords,
    db_scale=DB_SCALE,
    vmin=VMIN,
    vmax=VMAX,
    figsize=FIGSIZE,
    save_path=str(FIG_FOLDER / "mono_focused.png") if SAVE_FIG else None,
)

# ============================================================================
# STEP 4: CONVEX CIRCULAR TRANSDUCER — DOME LENS
# ============================================================================
print("\n--- 4. ConvexCircularTransducer (dome, D=20 mm, hemisphere) ---")

conv = ConvexCircularTransducer(
    diameter_mm=20.0,
    focus_mm=0,
    no_sub_diameter=30,
    frequency_Hz=FREQ_HZ,
)
conv.show()

sim_conv = Emission(conv, c=C, fs=FREQ_SAMPLING_HZ, monochromatic=True)
p_conv, coords = sim_conv(XZ_GRID, method="auto")

plot2D_pressure_slices(
    p_conv / p_conv.max(),
    coords=coords,
    db_scale=DB_SCALE,
    vmin=VMIN,
    vmax=VMAX,
    figsize=FIGSIZE,
    save_path=str(FIG_FOLDER / "mono_convex.png") if SAVE_FIG else None,
)

print("\nDone.")
