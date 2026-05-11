from pathlib import Path

import numpy as np
import pyvista as pv

import pyfield.transducers as Transducers
from pyfield.psimulation import PyField
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
FIG_FOLDER = Path(r"./figures/")
SAVE_FIG = True
SCALE = 3
SHOW_TX = False
SCALARS = "Apodization"
SCALARS = "Delays"

WIN_W, WIN_H = 800, 600

THEME = "dark"

# Atlas parameters
ATLAS_NAME = "whs_sd_rat_39um"
REGION_NAMES = ("root", "V1", "V2")

# Transducer / focus parameters
FOCUS_MM = np.array([-4.5, 0, 7.5])
FOVERD = 1

# Coordinate transform
LAMBDA_BREGMA_MM = 8  # mm (rat)
CORTEX2PROBE_MM = 4.5
X_TRANSLATION = 0  # mm
Y_TRANSLATION = 4  # mm

# PyVista theme
if THEME == "dark":
    pv.set_plot_theme("dark")
    pv.global_theme.anti_aliasing = "ssaa"
    pv.global_theme.background = "black"
    COLOR = "white"
    pv.global_theme.font.color = "white"
    AMBIENT_TX = 0.1
    AMBIENT_PR = 0.55
else:
    pv.set_plot_theme("default")
    pv.global_theme.anti_aliasing = "ssaa"
    COLOR = "black"
    AMBIENT_TX = 1
    AMBIENT_PR = 0.5


if SAVE_FIG:
    FIG_FOLDER.mkdir(exist_ok=True)

# ============================================================================
# STEP 1: LOAD BRAIN ATLAS
# ============================================================================
print("Loading brain atlas...")

brain_atlas = BG_Atlas(ATLAS_NAME, region_names=REGION_NAMES)

# ============================================================================
# STEP 2: CREATE TRANSDUCER AND COMPUTE PRESSURE FIELD
# ============================================================================
print("Creating transducer and computing pressure field...")

domino = Transducers.Domino()
domino.compute_delays(focus_mm=FOCUS_MM)
domino.compute_apodization(focus_mm=FOCUS_MM, FoverD=FOVERD, apodization_type="rect")
tx_mesh = domino.get_mesh()

sim = PyField(domino)
field_info_mm = {
    "x_extent": [-0.25 + FOCUS_MM[0], 0.25 + FOCUS_MM[0]],
    "y_extent": [-0.5 + FOCUS_MM[1], 0.5 + FOCUS_MM[1]],
    "z_extent": [-2 + FOCUS_MM[2], 2 + FOCUS_MM[2]],
    "dx": 0.0125,
    "dy": 0.025,
    "dz": 0.05,
}
pr, coords = sim(field_info_mm)
pressure_vol_mesh = create_3Dvol_mesh(
    coords["x"], coords["y"], coords["z"], pr / pr.max(), scalars="Pressure"
)

# ============================================================================
# STEP 3: TRANSFORM ATLAS INTO TRANSDUCER COORDINATE FRAME
# ============================================================================
print("Transforming atlas coordinates...")

invertz = np.diag([1, 1, -1, 1])

scale2animalsize = np.eye(4, dtype=float)
scale2animalsize[0, 0] = LAMBDA_BREGMA_MM
scale2animalsize[1, 1] = LAMBDA_BREGMA_MM
scale2animalsize[2, 2] = LAMBDA_BREGMA_MM

atlas_z_max = brain_atlas.pv_mesh["root"].bounds[5]

translate2xycenter = np.eye(4, dtype=float)
translate2xycenter[0, 3] = X_TRANSLATION
translate2xycenter[1, 3] = Y_TRANSLATION

translate2depth = np.eye(4, dtype=float)
translate2depth[2, 3] = -float(CORTEX2PROBE_MM) - atlas_z_max * LAMBDA_BREGMA_MM

T_matrix = invertz @ translate2depth @ translate2xycenter @ scale2animalsize
brain_atlas.transform(T_matrix=T_matrix, inplace=True)

# ============================================================================
# STEP 4: RENDER 3-D SCENE
# ============================================================================
print("Building 3-D visualisation...")

scale = SCALE if SAVE_FIG else 1
off_screen = SAVE_FIG

plotter = pv.Plotter(
    window_size=(int(WIN_W * scale), int(WIN_H * scale)), off_screen=off_screen
)
plotter = add_regions_mesh(
    brain_atlas.pv_mesh,
    plotter=plotter,
    kwargs_dict={
        REGION_NAMES[0]: {"color": "lightgray", "opacity": 0.4},
        REGION_NAMES[1]: {"color": "cadmiumlemon", "opacity": 0.3},
        REGION_NAMES[2]: {"color": "permanentgreen", "opacity": 0.3},
    },
    label="Brain Atlas",
)
if SHOW_TX:
    plotter = add_transducer_mesh(
        tx_mesh,
        plotter=plotter,
        show_edges=False,
        lighting=True,
        ambient=AMBIENT_TX,
        scalars=SCALARS,
        scalar_bar_args={
            "title": "Delays (s)",
            "title_font_size": int(16 * scale),
            "label_font_size": int(12 * scale),
            "vertical": True,
            "position_x": 0.85,
            "position_y": 0.6,
            "height": 0.3,
            "color": COLOR,
        },
    )
plotter = add_pressure_vol(
    pressure_vol_mesh,
    plotter=plotter,
    plot_focal_spot=True,
    lighting=True,
    ambient=AMBIENT_PR,
    scalar_bar_args={
        "title": "Pressure",
        "title_font_size": int(16 * scale),
        "label_font_size": int(12 * scale),
        "vertical": True,
        "position_x": 0.85,
        "position_y": 0.2,
        "height": 0.3,
        "color": COLOR,
    },
)

plotter.add_legend(loc="upper left")
plotter.show_grid(
    grid="back",
    color=COLOR,
    font_size=int(12 * scale),
    location="outer",
    xtitle="X (mm)",
    ytitle="Y (mm)",
    ztitle="Z (mm)",
    n_xlabels=5,
    n_ylabels=7,
    n_zlabels=5,
    use_3d_text=False,
)
plotter.add_axes(color=COLOR)
plotter.camera.up = (0, 0, -1)
if SHOW_TX:
    plotter.camera_position = [
        (-15.25, 35.68, -13.25),
        (2.36, -6.75, 10.33),
        (0.20, -0.41, -0.89),
    ]
else:
    # Upper view
    # plotter.set_viewup([0, 1, 0])
    plotter.camera_position = [
        (6, 3, -42),
        (6, 3, 0),
        (0, 1, 0),
    ]

comment = 1 if SHOW_TX else 0
if SAVE_FIG:
    plotter.screenshot(
        str(FIG_FOLDER / f"rat_brain_Vis_{SCALARS}_FoverD{FOVERD}_TX{comment}.png")
    )
else:
    plotter.show()

print(plotter.camera_position)
plotter.close()
try:
    pv.close_all()
except Exception:
    pass

del tx_mesh, brain_atlas, pressure_vol_mesh, domino, plotter

print("\nDone.")
