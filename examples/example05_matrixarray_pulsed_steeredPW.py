"""
Example 05: Matrix Array — Pulsed Steered Plane Wave (3-D Transient)

Demonstrates transient emission for a steered plane wave (PW) on a matrix
array.  PW delays are computed analytically from the steering direction unit
vector.  Output is animated in 3-D using `plot3D_transient_slices`.

  1. 32 × 32 matrix array, 3 MHz, 0.3 mm pitch (sub-λ → no grating lobes)
  2. Plane-wave steering delays for angle (θx, θy)
  3. Hanning-windowed excitation
  4. Transient field on two orthogonal planes (full 3-D volumes are heavy;
     two planes carry the same physics and render faster)
  5. 3-D PyVista animation of the wavefront crossing both planes

Run with:
    uv run examples/example05_matrixarray_pulsed_steeredPW.py
"""

import numpy as np
from config import FIG_FOLDER, SAVE_FIG

from pyfield.emission import Emission
from pyfield.plotting import add_transducer_mesh, plot3D_transient_slices
from pyfield.transducers import MatrixArrayTransducer
from pyfield.utilities import align_to_common_time

# ============================================================================
# CONFIGURATION
# ============================================================================
THETA_X_DEG = 10.0  # lateral steering
THETA_Y_DEG = 0.0  # elevation steering
FC_HZ = 3e6  # centre frequency (λ ≈ 0.51 mm)
FS = 50e6  # Hz (lower fs → shorter FFT → faster)
PULSE_CYCLES = 2

# Two orthogonal planes through the steering axis (cheaper than a volume)
dx = dy = dz = 0.05  # mm
CENTER_PLANES = (0, 0, 20)
PLANE_XZ = {
    "x_extent": [-8, 8],
    "y_extent": [CENTER_PLANES[1], CENTER_PLANES[1]],
    "z_extent": [5, 30],
    "dx": dx,
    "dy": dy,
    "dz": dz,
}
PLANE_YZ = {
    "x_extent": [CENTER_PLANES[0], CENTER_PLANES[0]],
    "y_extent": [-8, 8],
    "z_extent": [5, 30],
    "dx": dx,
    "dy": dy,
    "dz": dz,
}

print("\n--- Example 05: Matrix Array — Steered PW (3-D Transient) ---\n")

if SAVE_FIG:
    FIG_FOLDER.mkdir(exist_ok=True)

# ============================================================================
# STEP 1: CREATE MATRIX ARRAY TRANSDUCER
# ============================================================================
# 32 × 32 elements at 0.3 mm pitch → 9.6 mm square aperture, pitch < λ
tx = MatrixArrayTransducer(
    n_elements_x=32,
    n_elements_y=32,
    element_width_mm=0.275,
    element_height_mm=0.275,
    kerf_x_mm=0.025,
    kerf_y_mm=0.025,
    no_sub_x=1,
    no_sub_y=1,
    frequency_Hz=FC_HZ,
)

# ============================================================================
# STEP 2: COMPUTE PLANE-WAVE STEERING DELAYS
# ============================================================================
tx.compute_delays(angle_steering_deg=(THETA_X_DEG, THETA_Y_DEG))
print(f"Steering: θx = {THETA_X_DEG}°, θy = {THETA_Y_DEG}°")

# ============================================================================
# STEP 3: BUILD EXCITATION PULSE
# ============================================================================
fc = tx.fc
t_pulse = np.arange(0, PULSE_CYCLES / fc, 1.0 / FS)
excitation = (np.sin(2 * np.pi * fc * t_pulse) * np.hanning(len(t_pulse))).astype(
    np.float32
)

# ============================================================================
# STEP 4: COMPUTE THE TRANSIENT PRESSURE FIELD ON BOTH PLANES
# ============================================================================
sim = Emission(tx, fs=FS, excitation=excitation)
p_xz, coords_xz = sim(PLANE_XZ)
p_yz, coords_yz = sim(PLANE_YZ)

# The two planes start at different times of flight — resample onto a shared
# time axis so the animation shows one consistent wavefront.
t, [p_xza, p_yza] = align_to_common_time(
    [(p_xz, coords_xz), (p_yz, coords_yz)],
    align_to_shorter=False,
)

coords = {
    "x": coords_xz["x"],
    "y": coords_yz["y"],
    "z": coords_xz["z"],
}

# Planes list with offset tracking — squeeze singleton spatial dims → (Nt, N1, N2)
planes = [
    {"plane": "xz", "data": p_xza.squeeze(), "translation": (0, CENTER_PLANES[1], 0)},
    {"plane": "yz", "data": p_yza.squeeze(), "translation": (CENTER_PLANES[0], 0, 0)},
]

print(f"Time range: {t[0] * 1e6:.2f} – {t[-1] * 1e6:.2f} µs")

# ============================================================================
# STEP 5: ANIMATE IN 3-D (PyVista)
# ============================================================================
plotter = add_transducer_mesh(tx.get_mesh(), scalars="Delays")
plotter = plot3D_transient_slices(
    planes,
    coords=coords,
    plotter=plotter,
    time_array=t,
    db_scale=True,
    save_path=str(FIG_FOLDER) if SAVE_FIG else None,
    file_name="matrix_pw_3d.mp4",
)

del plotter
