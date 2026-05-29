"""
Example 04: Linear Array — Diverging Wave with Excitation (Transient)

Demonstrates pulsed emission for diverging-wave (DW) transmission using a
virtual source behind the array.  Shows the full transient API:

  1. Diverging-wave delays (virtual focus at z < 0 → unfocused wide beam)
  2. Hanning-windowed sine excitation pulse
  3. Transient `Emission` call → pressure field shape ``(Nt, Nx, Ny, Nz)``
  4. Time-vector reconstruction from `coords["t0"]` and `coords["dt"]`
  5. Animated 2-D wavefront visualisation

Run with:
    uv run examples/example04_lineararray_excitation_DW.py
"""

import numpy as np

from config import FIG_FOLDER, SAVE_FIG

import pyfield.transducers as transducers
from pyfield.plotting import plot2D_pressure_slices
from pyfield.emission import Emission

# ============================================================================
# CONFIGURATION
# ============================================================================
VIRTUAL_FOCUS_MM = [0, 0, -10]  # Behind array → diverging wave
SPEED_OF_SOUND = 1540.0  # m/s

FS = 200e6  # Sampling frequency (Hz)
PULSE_CYCLES = 2  # Hanning-windowed sine burst length

# Simulation plane (XZ, y = 0)
PLANE = {
    "x_extent": [-10, 10],
    "y_extent": [0, 0],
    "z_extent": [0.5, 20],
    "dx": 0.05,
    "dy": 0,
    "dz": 0.05,
}

FIGSIZE = (9, 5)

print("\n--- Example 04: Linear Array — Diverging Wave (Transient) ---\n")

if SAVE_FIG:
    FIG_FOLDER.mkdir(exist_ok=True)

# ============================================================================
# STEP 1: CREATE TRANSDUCER WITH DIVERGING-WAVE DELAYS
# ============================================================================
tx = transducers.Domino()
# Virtual focus behind array creates a diverging spherical wavefront.
tx.compute_delays(focus_mm=VIRTUAL_FOCUS_MM)
tx.compute_apodization(focus_mm=VIRTUAL_FOCUS_MM, FoverD=1)

print(f"Virtual focus: {VIRTUAL_FOCUS_MM} mm  (behind array → diverging wave)")

# ============================================================================
# STEP 2: BUILD HANNING-WINDOWED EXCITATION PULSE
# ============================================================================
fc = tx.fc
t_pulse = np.arange(0, PULSE_CYCLES / fc, 1.0 / FS)
excitation = (np.sin(2 * np.pi * fc * t_pulse) * np.hanning(len(t_pulse))).astype(
    np.float32
)

print(f"Pulse: {PULSE_CYCLES}-cycle Hanning sine at fc={fc / 1e6:.1f} MHz")

# ============================================================================
# STEP 3: COMPUTE TRANSIENT PRESSURE FIELD
# ============================================================================
# Emission with excitation → returns (Nt, Nx, Ny, Nz)
sim = Emission(tx, fs=FS, excitation=excitation)
p, coords = sim(PLANE)

print(f"\nPressure shape: {p.shape}  (Nt, Nx, Ny, Nz)")

# Reconstruct absolute time vector for animation axis label
t = coords["t0"] + np.arange(p.shape[0]) * coords["dt"]
print(f"Time range: {t[0] * 1e6:.2f} – {t[-1] * 1e6:.2f} µs")

# ============================================================================
# STEP 4: ANIMATE THE PROPAGATING WAVEFRONT
# ============================================================================
plot2D_pressure_slices(
    p,
    coords=coords,
    time_array=t,
    db_scale=True,
    figsize=FIGSIZE,
    vmin=-40,
    title="Diverging Wave — transient propagation",
    save_path=str(FIG_FOLDER / "dw_transient.gif") if SAVE_FIG else None,
)
