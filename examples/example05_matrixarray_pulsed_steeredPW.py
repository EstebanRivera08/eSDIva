"""
Example 05: Matrix Array — Pulsed Steered Plane Wave (3-D Transient)

Demonstrates transient emission for a steered plane wave (PW) on a matrix
array.  PW delays are computed analytically from the steering direction unit
vector.  Output is animated in 3-D using `plot3D_transient_slices`.

  1. Small custom matrix array (manageable runtime)
  2. Plane-wave steering delays for angle (θx, θy)
  3. Hanning-windowed excitation
  4. Transient field (Nt, Nx, Ny, Nz) on a 3-D grid
  5. 3-D PyVista animation of the wavefront

Run with:
    uv run examples/example05_matrixarray_pulsed_steeredPW.py
"""

import numpy as np

from config import FIG_FOLDER, SAVE_FIG

from pyfield.plotting import plot3D_transient_slices
from pyfield.emission import Emission
from pyfield.transducers import MatrixArrayTransducer

# ============================================================================
# CONFIGURATION
# ============================================================================
# Steering angles (degrees) — small matrix array for a quick demo
THETA_X_DEG = 10.0  # lateral steering
THETA_Y_DEG = 5.0  # elevation steering
SPEED_OF_SOUND = 1540.0  # m/s
FS = 100e6  # Hz (lower fs → shorter FFT → faster)
PULSE_CYCLES = 2

# 3-D field grid — coarse for reasonable runtime
VOLUME = {
    "x_extent": [-8, 8],
    "y_extent": [-8, 8],
    "z_extent": [5, 30],
    "dx": 1.0,
    "dy": 1.0,
    "dz": 0.5,
}

print("\n--- Example 05: Matrix Array — Steered PW (3-D Transient) ---\n")

if SAVE_FIG:
    FIG_FOLDER.mkdir(exist_ok=True)

# ============================================================================
# STEP 1: CREATE MATRIX ARRAY TRANSDUCER
# ============================================================================
# Small 11 × 11 matrix for manageable compute
tx = MatrixArrayTransducer(
    n_elements_x=11,
    n_elements_y=11,
    element_width_mm=0.5,
    element_height_mm=0.5,
    kerf_x_mm=0.1,
    kerf_y_mm=0.1,
    no_sub_x=1,
    no_sub_y=1,
    frequency_Hz=5e6,
)

# ============================================================================
# STEP 2: COMPUTE PLANE-WAVE STEERING DELAYS
# ============================================================================
# Steering direction unit vector.
# Element with maximum projection along n fires first (zero delay).
theta_x = np.deg2rad(THETA_X_DEG)
theta_y = np.deg2rad(THETA_Y_DEG)
nx = np.sin(theta_x)
ny = np.sin(theta_y)
nz = np.sqrt(max(0.0, 1.0 - nx**2 - ny**2))
n = np.array([nx, ny, nz])

d_e = tx.element_centers @ n  # projection per element (m)
delays = (d_e.max() - d_e) / SPEED_OF_SOUND  # element with max projection fires first
tx.set_delays(delays)

print(f"Steering: θx={THETA_X_DEG}°, θy={THETA_Y_DEG}°")
print(f"Max delay: {delays.max() * 1e6:.3f} µs")

# ============================================================================
# STEP 3: BUILD EXCITATION PULSE
# ============================================================================
fc = tx.fc
t_pulse = np.arange(0, PULSE_CYCLES / fc, 1.0 / FS)
excitation = (np.sin(2 * np.pi * fc * t_pulse) * np.hanning(len(t_pulse))).astype(
    np.float32
)

# ============================================================================
# STEP 4: COMPUTE 3-D TRANSIENT PRESSURE FIELD
# ============================================================================
sim = Emission(tx, fs=FS, excitation=excitation)
p, coords = sim(VOLUME)

print(f"\nPressure shape: {p.shape}  (Nt, Nx, Ny, Nz)")
t = coords["t0"] + np.arange(p.shape[0]) * coords["dt"]
print(f"Time range: {t[0] * 1e6:.2f} – {t[-1] * 1e6:.2f} µs")

# ============================================================================
# STEP 5: ANIMATE IN 3-D (PyVista)
# ============================================================================
plot3D_transient_slices(
    p,
    coords=coords,
    time_array=t,
    db_scale=True,
    save_path=str(FIG_FOLDER) if SAVE_FIG else None,
    file_name="matrix_pw_3d.mp4",
)
