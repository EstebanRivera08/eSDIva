"""
Example 3: Linear Array — Monochromatic (CW) Pressure Field

Demonstrates continuous-wave simulation with a linear array transducer using
diverging-wave transmission.  It shows:

  1. Transducer setup with a virtual (behind-array) focus for diverging waves
  2. Monochromatic pressure field computation on an XZ plane
  3. 2-D visualisation in dB scale

Steps
-----
1. Create a Domino linear-array transducer
2. Apply diverging-wave delays (virtual source behind the array)
3. Compute the CW field on an XZ plane
4. Plot the pressure field

Run with:
    uv run examples/example3_lineartx_monochromatic.py
"""

import numpy as np

import pyfield.transducers as transducers
from pyfield.psimulation import PyField
from pyfield.utilities import plot_slices_2d

# ============================================================================
# CONFIGURATION
# ============================================================================
from config import FIG_FOLDER, SAVE_FIG, SCALE

# Steering configuration
VIRTUAL_FOCUS_MM = [0, 0, -1]  # Behind the array → diverging wave

# Simulation grid (all in mm)
PLANE_X_EXTENT_MM = [-10, 10]
PLANE_Z_EXTENT_MM = [0, 15]
GRID_SPACING_X_MM = 0.05
GRID_SPACING_Z_MM = 0.05

# Visualisation
FIGURE_SIZE = (9, 5)

print("\n --- Example 3: Linear Array — Monochromatic (CW) --- \n")

# ============================================================================
# STEP 1: CREATE TRANSDUCER AND APPLY DIVERGING-WAVE DELAYS
# ============================================================================
tx = transducers.Domino()
tx.compute_delays(focus_mm=VIRTUAL_FOCUS_MM)
tx.compute_apodization(focus_mm=VIRTUAL_FOCUS_MM, FoverD=1)
tx.plot_delays_apodization()

# ============================================================================
# STEP 2: DEFINE SIMULATION PLANE
# ============================================================================
plane_config = {
    "x_extent": PLANE_X_EXTENT_MM,
    "y_extent": [0, 0],
    "z_extent": PLANE_Z_EXTENT_MM,
    "dx": GRID_SPACING_X_MM,
    "dy": 0,
    "dz": GRID_SPACING_Z_MM,
}

# ============================================================================
# STEP 3: COMPUTE MONOCHROMATIC PRESSURE FIELD
# ============================================================================
simulator = PyField(tx)
x, y, z, p_field = simulator(plane_config)

# ============================================================================
# STEP 4: VISUALISE
# ============================================================================
plot_slices_2d(
    x,
    y,
    z,
    p_field,
    db_scale=True,
    figsize=FIGURE_SIZE,
    vmin=-30,
    save_path=str(FIG_FOLDER) if SAVE_FIG else None,
    file_name=f"lineartx_monochromatic",
)
