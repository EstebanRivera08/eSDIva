"""
Linear Array — Emission API: Monochromatic, Pulsed, Excitation, and Per-Element Modes

Demonstrates the four emission modes of the `Emission` class on a focused
linear array (Domino transducer):

  1. Monochromatic (CW)       — `Emission(tx, monochromatic=True)` → (Nx, Ny, Nz)
  2. Pulsed                   — `Emission(tx)` with no excitation  → (Nt, Nx, Ny, Nz)
  3. With excitation          — `Emission(tx, excitation=pulse)`   → (Nt, Nx, Ny, Nz)
  4. Per-element excitation   — `Emission(tx, excitation=(L, E))`  → (Nt, Nx, Ny, Nz)

Run with:
    uv run examples/example_linear_emission.py
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

PLANE = {
    "x_extent": [-10, 10],
    "y_extent": [0, 0],
    "z_extent": [5, 35],
    "dx": 0.1,
    "dy": 0,
    "dz": 0.1,
}

PULSE_CYCLES = 10
FIGSIZE = (9, 5)

print("\n--- Linear Array: Emission API ---\n")

if SAVE_FIG:
    FIG_FOLDER.mkdir(exist_ok=True)

# ============================================================================
# TRANSDUCER SETUP (shared across all three modes)
# ============================================================================
tx = transducers.Domino()
tx.compute_delays(focus_mm=FOCUS_MM)
tx.compute_apodization(focus_mm=FOCUS_MM, FoverD=FOVERD)

# ============================================================================
# EXCITATION PULSE (used by mode 3 only)
# ============================================================================
fs = 200e6
fc = tx.fc
t_pulse = np.arange(0, PULSE_CYCLES / fc, 1.0 / fs)
pulse = (np.sin(2 * np.pi * fc * t_pulse) * np.hanning(len(t_pulse))).astype(np.float32)

# ============================================================================
# MODE 1 — MONOCHROMATIC (CW)
# ============================================================================
print("Mode 1: Monochromatic CW")
sim_cw = Emission(tx, monochromatic=True, fs=fs)
p_cw, coords_cw = sim_cw(PLANE)
# p_cw.shape == (Nx, Ny, Nz)

plot2D_pressure_slices(
    p_cw,
    coords=coords_cw,
    db_scale=True,
    figsize=FIGSIZE,
    vmin=-30,
    title="Linear — Monochromatic CW",
    save_path=str(FIG_FOLDER) if SAVE_FIG else None,
    file_name="linear_emission_monochromatic.png",
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
    title="Linear — Pulsed (raw SIR)",
    save_path=str(FIG_FOLDER / "linear_emission_pulsed.gif") if SAVE_FIG else None,
)

# ============================================================================
# MODE 3 — WITH EXCITATION (Hanning-windowed sine burst)
# ============================================================================
print("Mode 3: Transient with excitation")
sim_exc = Emission(tx, excitation=pulse, verbose=True)
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
    title="Linear — Transient with excitation",
    save_path=str(FIG_FOLDER / "linear_emission_excitation.gif") if SAVE_FIG else None,
)

# ============================================================================
# MODE 4 — PER-ELEMENT EXCITATION (different pulse amplitude per element)
# ============================================================================
print("Mode 4: Transient with per-element excitation")

# Apply a secondary amplitude taper across elements: left half at 70%, right at 100%.
# This is independent of the focusing apodization already baked into the SIR.
# excitations_LE shape must be (len(pulse), n_elements).
n_elements = tx.n_elements
secondary_amp = np.ones(n_elements, dtype=np.float32)
secondary_amp[: n_elements // 2] = 0.7  # left half at 70% amplitude
exc_per_elem = (pulse[:, np.newaxis] * secondary_amp[np.newaxis, :]).astype(np.float32)

sim_pe = Emission(tx, excitation=exc_per_elem, fs=fs)
p_pe, coords_pe = sim_pe(PLANE)
# p_pe.shape == (Nt, Nx, Ny, Nz)

t_pe = coords_pe["t0"] + np.arange(p_pe.shape[0]) * coords_pe["dt"]

plot2D_pressure_slices(
    p_pe,
    coords=coords_pe,
    time_array=t_pe,
    db_scale=True,
    figsize=FIGSIZE,
    vmin=-40,
    title="Linear — Per-element excitation (asymmetric taper)",
    save_path=str(FIG_FOLDER / "linear_emission_per_element.gif") if SAVE_FIG else None,
)
