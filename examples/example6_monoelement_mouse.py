"""
Example 6: Mouse Brain Atlas with Focused Ultrasound

Demonstrates how to combine a BrainGlobe mouse brain atlas with a
ConcaveCircularTransducer and its simulated pressure field in a single
3-D PyVista scene.

Steps
-----
1. Create a concave bowl transducer and compute a small focal volume
2. Load a mouse brain atlas (``allen_mouse_25um``) via BrainGlobe
3. Transform the atlas into the transducer coordinate frame
4. Render anatomy, transducer, and pressure together

Requirements
------------
BrainGlobe atlas data is downloaded on first run (~500 MB for the mouse).

Run with:
    uv run examples/example6_monoelement_mouse.py
"""

import gc

import numpy as np
import pyvista as pv

from pyfield.psimulation import PyField
from pyfield.transducers import ConcaveCircularTransducer
from pyfield.utilities import BG_Atlas
from pyfield.plotting import (
    add_pressure_vol,
    add_regions_mesh,
    add_transducer_mesh,
    create_3Dvol_mesh,
)

# ============================================================================
# CONFIGURATION
# ============================================================================
from config import FIG_FOLDER, SAVE_FIG, SCALE

WIN_W, WIN_H = 800, 600

# Transducer parameters
FOCUS_DEPTH_MM = 10
DIAMETER_MM = 10.0
FREQ_HZ = 5e6

# Atlas configuration
ATLAS_NAME = "allen_mouse_25um"
REGION_NAMES = ("root", "Isocortex", "CA1")

# Coordinate transform parameters
LAMBDA_BREGMA_MM = 5.0  # Lambda-bregma distance for mouse
CORTEX2PROBE_MM = FOCUS_DEPTH_MM - 1.0

print("\n --- Example 6: Mouse Brain Atlas + Focused Ultrasound --- \n")

if SAVE_FIG:
    FIG_FOLDER.mkdir(exist_ok=True)

# ============================================================================
# STEP 1: CREATE TRANSDUCER AND COMPUTE PRESSURE FIELD
# ============================================================================
print("Creating transducer and computing pressure field...")

mouse_tx = ConcaveCircularTransducer(
    diameter_mm=DIAMETER_MM,
    focus_mm=FOCUS_DEPTH_MM,
    no_sub_diameter=20,
    frequency_Hz=FREQ_HZ,
)

mouse_sim = PyField(mouse_tx, verbose=False)
p, coords = mouse_sim(
    {
        "x_extent": [-1, 1],
        "y_extent": [-1, 1],
        "z_extent": [-1 + FOCUS_DEPTH_MM, 1 + FOCUS_DEPTH_MM],
        "dx": 0.04,
        "dy": 0.04,
        "dz": 0.04,
    },
    method="auto",
)

pressure_vol = create_3Dvol_mesh(
    coords["x"], coords["y"], coords["z"], p / p.max(), scalars="Pressure"
)
del p, coords, mouse_sim
gc.collect()

# ============================================================================
# STEP 2: LOAD MOUSE BRAIN ATLAS
# ============================================================================
print("Loading mouse brain atlas...")

mouse_atlas = BG_Atlas(ATLAS_NAME, region_names=REGION_NAMES)

# ============================================================================
# STEP 3: TRANSFORM ATLAS INTO TRANSDUCER COORDINATE FRAME
# ============================================================================
print("Transforming atlas coordinates...")

# Scale by lambda-bregma distance
scale_mat = np.eye(4)
scale_mat[:3, :3] *= LAMBDA_BREGMA_MM

# Translate so that atlas cortex top sits CORTEX2PROBE_MM below the transducer
atlas_z_max = mouse_atlas.pv_mesh["root"].bounds[5]
trans_depth = np.eye(4)
trans_depth[2, 3] = -CORTEX2PROBE_MM - atlas_z_max * LAMBDA_BREGMA_MM

# Invert z-axis (transducer convention: z positive = depth)
inv_z = np.diag([1.0, 1.0, -1.0, 1.0])

T_matrix = inv_z @ trans_depth @ scale_mat
mouse_atlas.transform(T_matrix=T_matrix, inplace=True)

# ============================================================================
# STEP 4: RENDER 3-D SCENE
# ============================================================================
print("Building 3-D visualisation...")

if SAVE_FIG:
    pl = pv.Plotter(window_size=(WIN_W * SCALE, WIN_H * SCALE), off_screen=True)
else:
    pl = pv.Plotter(window_size=(WIN_W, WIN_H))

pl = add_regions_mesh(
    mouse_atlas.pv_mesh,
    plotter=pl,
    kwargs_dict={
        REGION_NAMES[0]: {"color": "lightgray", "opacity": 0.2},
        REGION_NAMES[1]: {"color": "lightblue", "opacity": 0.2},
        REGION_NAMES[2]: {"color": "salmon", "opacity": 0.2},
    },
)
pl = add_transducer_mesh(mouse_tx.get_mesh(), plotter=pl, scale=SCALE)
pl = add_pressure_vol(
    pressure_vol, plotter=pl, plot_focal_spot=False, contour_levels=8, scale=SCALE
)
pl.show_bounds(
    grid="back",
    location="outer",
    xtitle="X (mm)",
    ytitle="Y (mm)",
    ztitle="Z (mm)",
    font_size=10 * SCALE,
    use_3d_text=False,
)
pl.add_axes()
pl.camera.up = (0, 0, -1)
pl.reset_camera()

if SAVE_FIG:
    pl.screenshot(str(FIG_FOLDER / "brain_mouse_scene.png"))
else:
    pl.show()

pl.close()
del pl, mouse_atlas, pressure_vol, mouse_tx
gc.collect()

print("\nDone.")
