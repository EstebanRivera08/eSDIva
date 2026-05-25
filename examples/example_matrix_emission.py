"""
Matrix Array — Emission API: Monochromatic, Pulsed, and Excitation Modes

Demonstrates the three emission modes of the `Emission` class on a focused
matrix array (Zeus_Matrix transducer):

  1. Monochromatic (CW)  — `Emission(tx, monochromatic=True)` → (Nx, Ny, Nz)
  2. Pulsed              — `Emission(tx)` with no excitation  → (Nt, Nx, Ny, Nz)
  3. With excitation     — `Emission(tx, excitation=pulse)`   → (Nt, Nx, Ny, Nz)

The matrix array has many more patches than a linear array, so the field grid
is kept deliberately coarse to keep runtime manageable.

Run with:
    uv run examples/example_matrix_emission.py
"""

import numpy as np

from config import FIG_FOLDER, SAVE_FIG

import pyfield.transducers as transducers
from pyfield.plotting import plot2D_pressure_slices
from pyfield.psimulation import Emission

# ============================================================================
# CONFIGURATION
# ============================================================================
FOCUS_MM = [0, 0, 20]
FOVERD = 2.0

# XZ plane at y=0 — coarse grid to keep matrix runtime reasonable
PLANE = {
    "x_extent": [-8, 8],
    "y_extent": [0, 0],
    "z_extent": [5, 30],
    "dx": 0.15,
    "dy": 0,
    "dz": 0.15,
}

PULSE_CYCLES = 2
FIGSIZE = (9, 5)

print("\n--- Matrix Array: Emission API ---\n")

if SAVE_FIG:
    FIG_FOLDER.mkdir(exist_ok=True)

# ============================================================================
# TRANSDUCER SETUP (shared across all three modes)
# ============================================================================
tx = transducers.Zeus_Matrix()
tx.compute_delays(focus_mm=FOCUS_MM)
tx.compute_apodization(focus_mm=FOCUS_MM, FoverD=FOVERD)

# ============================================================================
# EXCITATION PULSE (used by mode 3 only)
# ============================================================================
fs = 200e6
fc = tx.fc
t_pulse = np.arange(0, PULSE_CYCLES / fc, 1.0 / fs)
pulse = (np.sin(2 * np.pi * fc * t_pulse) * np.hanning(len(t_pulse))).astype(
    np.float32
)

# ============================================================================
# MODE 1 — MONOCHROMATIC (CW)
# ============================================================================
print("Mode 1: Monochromatic CW")
sim_cw = Emission(tx, monochromatic=True)
p_cw, coords_cw = sim_cw(PLANE)
# p_cw.shape == (Nx, Ny, Nz)

plot2D_pressure_slices(
    p_cw,
    coords=coords_cw,
    db_scale=True,
    figsize=FIGSIZE,
    vmin=-30,
    title="Matrix — Monochromatic CW",
    save_path=str(FIG_FOLDER) if SAVE_FIG else None,
    file_name="matrix_emission_monochromatic.png",
)

# ============================================================================
# MODE 2 — PULSED (raw SIR, no excitation)
# ============================================================================
print("Mode 2: Pulsed (raw SIR)")
sim_pulsed = Emission(tx)  # monochromatic=False, excitation=None
p_pulsed, coords_pulsed = sim_pulsed(PLANE)
# p_pulsed.shape == (Nt, Nx, Ny, Nz)

t_pulsed = coords_pulsed["t0"] + np.arange(p_pulsed.shape[0]) * coords_pulsed["dt"]

plot2D_pressure_slices(
    p_pulsed,
    coords=coords_pulsed,
    time_array=t_pulsed,
    db_scale=True,
    figsize=FIGSIZE,
    vmin=-40,
    title="Matrix — Pulsed (raw SIR)",
    save_path=str(FIG_FOLDER / "matrix_emission_pulsed.gif") if SAVE_FIG else None,
)

# ============================================================================
# MODE 3 — WITH EXCITATION (Hanning-windowed sine burst)
# ============================================================================
print("Mode 3: Transient with excitation")
sim_exc = Emission(tx, excitation=pulse)
p_exc, coords_exc = sim_exc(PLANE)
# p_exc.shape == (Nt, Nx, Ny, Nz)

t_exc = coords_exc["t0"] + np.arange(p_exc.shape[0]) * coords_exc["dt"]

plot2D_pressure_slices(
    p_exc,
    coords=coords_exc,
    time_array=t_exc,
    db_scale=True,
    figsize=FIGSIZE,
    vmin=-40,
    title="Matrix — Transient with excitation",
    save_path=str(FIG_FOLDER / "matrix_emission_excitation.gif") if SAVE_FIG else None,
)
