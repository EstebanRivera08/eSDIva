"""
Example 18: Custom Sparse Array — transform() and 3-D Plane Slices

Builds a research-style sparse 2-D array (64 disc elements on a Fermat
spiral — a common layout for 3-D imaging and transcranial ultrasound
because it avoids grating lobes with few elements), then:

  1. Assembles it with `CustomTransducer` from one mono-element template
  2. Rigidly repositions it with `transform()` (tilt + shift, as when a
     probe is mounted at an angle over the target)
  3. Refocuses electronically on a fixed world-frame target
  4. Computes the CW field on three orthogonal planes through the focus —
     the full volume would be large, three planes carry the same physics
  5. Renders planes + transducer mesh together in one PyVista scene

Run with:
    uv run examples/example18_customtransducer_3Dplanes.py
"""

import numpy as np
import pyvista as pv
from config import FIG_FOLDER, SAVE_FIG, SCALE

from pyfield.emission import Emission
from pyfield.plotting import add_2D_image, add_transducer_mesh, create_2Dimage_mesh
from pyfield.transducers import CustomTransducer, FlatCircularTransducer
from pyfield.utilities import to_dB

# ============================================================================
# CONFIGURATION
# ============================================================================
N_ELEMENTS = 64
ELEMENT_DIAMETER_MM = 3.0
APERTURE_RADIUS_MM = 25.0
FC_HZ = 1e6  # 1 MHz — TUS-range frequency
TARGET_MM = np.array([0.0, 0.0, 60.0])  # world-frame focus (fixed)
TILT_DEG = 20.0  # probe mounted tilted about the x-axis
SHIFT_MM = [0.0, -15.0, 0.0]  # and shifted in elevation

# Three orthogonal planes through the target
HALF_XY_MM, DXY_MM = 15.0, 0.3
Z_EXTENT_MM, DZ_MM = [35.0, 85.0], 0.4

print("\n--- Example 18: Custom Sparse Array — transform() + 3-D Planes ---\n")

if SAVE_FIG:
    FIG_FOLDER.mkdir(exist_ok=True)

# ============================================================================
# STEP 1: SPARSE SPIRAL LAYOUT + CUSTOM TRANSDUCER
# ============================================================================
# Fermat spiral: uniform element density, no periodicity → no grating lobes.
k = np.arange(N_ELEMENTS)
golden = np.pi * (3.0 - np.sqrt(5.0))
r = APERTURE_RADIUS_MM * np.sqrt((k + 0.5) / N_ELEMENTS)
positions_mm = np.column_stack(
    [r * np.cos(k * golden), r * np.sin(k * golden), np.zeros(N_ELEMENTS)]
)

disc = FlatCircularTransducer(
    diameter_mm=ELEMENT_DIAMETER_MM,
    no_sub_diameter=6,
    frequency_Hz=FC_HZ,
)
tx = CustomTransducer(
    elements=[disc] * N_ELEMENTS,
    positions_mm=positions_mm,
    frequency_Hz=FC_HZ,
)
print(
    f"Sparse array: {N_ELEMENTS} discs Ø{ELEMENT_DIAMETER_MM} mm, "
    f"aperture Ø{2 * APERTURE_RADIUS_MM} mm"
)

# ============================================================================
# STEP 2: SIMULATOR + RIGID REPOSITIONING
# ============================================================================
sim = Emission(tx, monochromatic=True, verbose=False)

# Mount the probe tilted over the target: rotation about x, then translation.
th = np.deg2rad(TILT_DEG)
T = np.array(
    [
        [1, 0, 0, SHIFT_MM[0]],
        [0, np.cos(th), -np.sin(th), SHIFT_MM[1]],
        [0, np.sin(th), np.cos(th), SHIFT_MM[2]],
        [0, 0, 0, 1],
    ]
)
tx.transform(T)

# Delays are firing-time state, not geometry: refocus on the (unchanged)
# world-frame target from the new pose.
tx.compute_delays(focus_mm=TARGET_MM)

# The simulator snapshots geometry at construction — refresh it after the move.
sim.set("transducer", tx)

# ============================================================================
# STEP 3: THREE ORTHOGONAL PLANES THROUGH THE TARGET
# ============================================================================
x0, y0, z0 = TARGET_MM
grids = {
    "xz": {
        "x_extent": [x0 - HALF_XY_MM, x0 + HALF_XY_MM],
        "y_extent": [y0, y0],
        "z_extent": Z_EXTENT_MM,
        "dx": DXY_MM,
        "dy": 0,
        "dz": DZ_MM,
    },
    "yz": {
        "x_extent": [x0, x0],
        "y_extent": [y0 - HALF_XY_MM, y0 + HALF_XY_MM],
        "z_extent": Z_EXTENT_MM,
        "dx": 0,
        "dy": DXY_MM,
        "dz": DZ_MM,
    },
    "xy": {
        "x_extent": [x0 - HALF_XY_MM, x0 + HALF_XY_MM],
        "y_extent": [y0 - HALF_XY_MM, y0 + HALF_XY_MM],
        "z_extent": [z0, z0],
        "dx": DXY_MM,
        "dy": DXY_MM,
        "dz": 0,
    },
}

planes, coords = {}, {}
for name, g in grids.items():
    print(f"Computing {name.upper()} plane ...")
    p, c = sim(g, method="auto")
    planes[name], coords[name] = p.squeeze(), c

# Shared dB scale referenced to the global peak of all three planes.
p_max = max(p.max() for p in planes.values())
planes_db = {k: to_dB(v, vmax=p_max) for k, v in planes.items()}

# ============================================================================
# STEP 4: 3-D SCENE — PLANES + TILTED TRANSDUCER MESH
# ============================================================================
scale = SCALE if SAVE_FIG else 1
plotter = pv.Plotter(window_size=(800 * scale, 700 * scale), off_screen=SAVE_FIG)

# imshow-style (row, col) → world: rows = first grid axis, cols = second.
mesh_specs = [
    ("xz", {"y": y0}, (coords["xz"]["x"], coords["xz"]["z"])),
    ("yz", {"x": x0}, (coords["yz"]["y"], coords["yz"]["z"])),
    ("xy", {"z": z0}, (coords["xy"]["x"], coords["xy"]["y"])),
]
for name, offset, (c1, c2) in mesh_specs:
    mesh = create_2Dimage_mesh(
        planes_db[name],
        extent=(c1.min(), c1.max(), c2.min(), c2.max()),
        plane_offset=offset,
        scalars="Pressure (dB)",
    )
    plotter = add_2D_image(
        mesh,
        plotter=plotter,
        scale=scale,
        cmap="jet",
        clim=[-40, 0],
        colorbar_title="Pressure (dB)",
    )

plotter = add_transducer_mesh(
    tx.get_mesh(), plotter=plotter, scalars="Delays", scale=scale
)
plotter.add_axes()
plotter.show_grid(
    font_size=10 * scale, xtitle="X (mm)", ytitle="Y (mm)", ztitle="Z (mm)"
)
plotter.camera.up = (0, 0, -1)

if SAVE_FIG:
    plotter.screenshot(str(FIG_FOLDER / "custom_sparse_3dplanes.png"))
else:
    plotter.show()
plotter.close()

print("\nDone.")
