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

from pyfield.emission import Emission
from pyfield.plotting import add_transducer_mesh, plot3D_transient_slices
from pyfield.transducers import MatrixArrayTransducer
from pyfield.utilities import align_to_common_time

# ============================================================================
# CONFIGURATION
# ============================================================================
# Steering angles (degrees) — small matrix array for a quick demo
THETA_X_DEG = 10.0  # lateral steering
THETA_Y_DEG = 5.0  # elevation steering
SPEED_OF_SOUND = 1540.0  # m/s
FS = 50e6  # Hz (lower fs -> shorter FFT -> faster)
PULSE_CYCLES = 2

# 3-D field grid — coarse for reasonable runtime
dx = dy = dz = 0.05  # mm
# Volume grid can be computed but the storage and compute requirements for a 3-D
# transient field are high, so we will compute three planes instead and animate them in
# 3-D with the transducer mesh.
# VOLUME = {
#     "x_extent": [-8, 8],
#     "y_extent": [-8, 8],
#     "z_extent": [5, 30],
#     "dx": dx,
#     "dy": dy,
#     "dz": dz,
# }
CENTER_PLANES = (0, 0, 20)
# PLANE_XY = {
#     "x_extent": [-8, 8],
#     "y_extent": [-8, 8],
#     "z_extent": [CENTER_PLANES[2], CENTER_PLANES[2]],
#     "dx": dx,
#     "dy": dy,
#     "dz": dz,
# }
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
# Small 11 × 11 matrix for manageable compute
tx = MatrixArrayTransducer(
    n_elements_x=33,
    n_elements_y=33,
    element_width_mm=0.99,
    element_height_mm=0.99,
    kerf_x_mm=0.01,
    kerf_y_mm=0.01,
    no_sub_x=2,
    no_sub_y=2,
    frequency_Hz=5e6,
)

# ============================================================================
# STEP 2: COMPUTE PLANE-WAVE STEERING DELAYS
# ============================================================================
tx.compute_delays(angle_steering_deg=(10, 0))

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
# p, coords = sim(VOLUME)
# p_xy, coords_xy = sim(PLANE_XY)
p_xz, coords_xz = sim(PLANE_XZ)
p_yz, coords_yz = sim(PLANE_YZ)

# t = coords["t0"] + np.arange(p.shape[0]) * coords["dt"]
t, [p_xza, p_yza] = align_to_common_time(  # , p_xya]
    [(p_xz, coords_xz), (p_yz, coords_yz)],
    # (p_xy, coords_xy)],
    align_to_shorter=False,
)

coords = {
    "x": coords_xz["x"],
    "y": coords_yz["y"],
    "z": coords_xz["z"],
}

# Build planes list with offset tracking — squeeze singleton spatial dims → (Nt, N1, N2)
planes = [
    {"plane": "xz", "data": p_xza.squeeze(), "translation": (0, CENTER_PLANES[1], 0)},
    {"plane": "yz", "data": p_yza.squeeze(), "translation": (CENTER_PLANES[0], 0, 0)},
    # {"plane": "xy", "data": p_xya.squeeze(), "translation": (0, 0, CENTER_PLANES[2])},
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
