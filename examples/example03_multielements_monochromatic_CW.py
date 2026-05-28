"""
Example 03: Multi-Element Linear Array — Monochromatic (CW) Pressure Field

Demonstrates continuous-wave simulation with a focused linear array using
`Emission(tx, monochromatic=True)`.  Returns the CW amplitude |H(r, ωc)|
at the transducer centre frequency — a 3-D spatial map with no time axis.

  1. Focused linear array with electronic delays and apodization
  2. Monochromatic pressure field on the XZ plane (single Y slice)
  3. 2-D dB visualisation of the focused beam
  4. Runtime parameter update via `sim.set()`

Run with:
    uv run examples/example03_multielements_monochromatic_CW.py
"""

from config import FIG_FOLDER, SAVE_FIG

import pyfield.transducers as transducers
from pyfield.plotting import plot2D_pressure_slices
from pyfield.psimulation import Emission

# ============================================================================
# CONFIGURATION
# ============================================================================
FOCUS_MM = [0, 0, 30]  # TX focal point [x, y, z] in mm
FOVERD = 2.0  # F/D ratio for apodization window

# XZ simulation plane (y = 0 elevation slice)
PLANE = {
    "x_extent": [-10, 10],
    "y_extent": [0, 0],
    "z_extent": [5, 55],
    "dx": 0.1,
    "dy": 0,
    "dz": 0.2,
}

FIGSIZE = (9, 5)

print("\n--- Example 03: Multi-Element Linear Array — Monochromatic CW ---\n")

# ============================================================================
# STEP 1: CREATE TRANSDUCER WITH FOCUSED DELAYS
# ============================================================================
tx = transducers.Domino()
tx.compute_delays(focus_mm=FOCUS_MM)
tx.compute_apodization(focus_mm=FOCUS_MM, FoverD=FOVERD)

# ============================================================================
# STEP 2: COMPUTE CW PRESSURE FIELD
# ============================================================================
# monochromatic=True → returns |H(r, fc)|, shape (Nx, Ny, Nz)
sim = Emission(tx, monochromatic=True)
p_cw, coords = sim(PLANE)

plot2D_pressure_slices(
    p_cw,
    coords=coords,
    db_scale=True,
    figsize=FIGSIZE,
    vmin=-40,
    title=f"Focused CW — focus at z={FOCUS_MM[2]} mm",
    save_path=str(FIG_FOLDER) if SAVE_FIG else None,
    file_name="linear_cw_focused.png",
)

# ============================================================================
# STEP 3: RUNTIME UPDATE — shift focus deeper and rerun (no new object needed)
# ============================================================================
FOCUS_DEEP_MM = [0, 0, 50]
tx.compute_delays(focus_mm=FOCUS_DEEP_MM)
tx.compute_apodization(focus_mm=FOCUS_DEEP_MM, FoverD=FOVERD)

sim.set("monochromatic", True)  # sim.set() updates in place without reconstruction
p_deep, coords_deep = sim(PLANE)

plot2D_pressure_slices(
    p_deep,
    coords=coords_deep,
    db_scale=True,
    figsize=FIGSIZE,
    vmin=-40,
    title=f"Focused CW — focus at z={FOCUS_DEEP_MM[2]} mm",
    save_path=str(FIG_FOLDER) if SAVE_FIG else None,
    file_name="linear_cw_focused_deep.png",
)
