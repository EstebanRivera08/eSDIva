"""
Example 9: STL Mesh with Acoustic Simulation

Demonstrates how to combine an STL model (experimental setup) with a PyField
acoustic simulation to visualise the complete experimental configuration in 3-D.

Steps
-----
1. Create a concave bowl transducer (0.5" diameter, 1" focal length)
2. Compute the monochromatic pressure field around the geometric focus
3. Load and position a Petri-dish STL mesh in the simulation space
4. Render transducer + pressure + STL in a single PyVista scene

Run with:
    uv run examples/example9_monoelement_petridish.py
"""

from pathlib import Path

import pyvista as pv

from pyfield.psimulation import PyField
from pyfield.transducers import ConcaveCircularTransducer
from pyfield.plotting import (
    add_pressure_vol,
    add_stl_mesh,
    add_transducer_mesh,
    create_3Dvol_mesh,
    load_mesh_from_stl,
)

# ============================================================================
# CONFIGURATION
# ============================================================================
from config import FIG_FOLDER, SAVE_FIG, SCALE

WIN_W, WIN_H = 500, 600

THEME = "dark"
SCRIPT_DIR = Path(__file__).parent
STL_FILE = SCRIPT_DIR / "Petri_dish.stl"

# PyVista theme
if THEME == "dark":
    pv.set_plot_theme("dark")
    pv.global_theme.background = "black"
    pv.global_theme.font.color = "white"
    pv.global_theme.anti_aliasing = "ssaa"
else:
    pv.set_plot_theme("default")

print("=" * 70)
print("PyField + STL Mesh Visualisation Example")
print("=" * 70)

if SAVE_FIG:
    FIG_FOLDER.mkdir(exist_ok=True)

# ============================================================================
# STEP 1: CREATE TRANSDUCER
# ============================================================================
print("\n Creating transducer...")

transducer = ConcaveCircularTransducer(
    diameter_mm=0.5 * 25.4,  # 0.5 inch diameter
    focus_mm=1 * 25.4,  # 1 inch focal length
    no_sub_diameter=20,
    frequency_Hz=5e6,
)

# ============================================================================
# STEP 2: COMPUTE PRESSURE FIELD
# ============================================================================
print(" Running acoustic simulation...")

field_points = {
    "x_extent": [-2, 2],
    "y_extent": [-2, 2],
    "z_extent": [18, 35],
    "dx": 0.1,
    "dy": 0.1,
    "dz": 0.2,
}

sim = PyField(transducer, monochromatic=True, verbose=False)
p, coords = sim(field_points, method="auto")
p_norm = p / p.max()

# ============================================================================
# STEP 3: LOAD AND POSITION STL MESH
# ============================================================================
print(" Loading STL mesh...")

has_stl = STL_FILE.exists()
if has_stl:
    petri_dish = load_mesh_from_stl(
        STL_FILE,
        scale=1,
        translation=(0, -10, 25),
        rotation_axis=(1, 0, 0),
        rotation_angle=-45,
    )
    print(f"  STL file loaded: {STL_FILE.name}")
    print(f"  Mesh bounds: {petri_dish.bounds}")
else:
    print(f"  Warning: STL file not found at {STL_FILE}")
    print("  Continuing without STL mesh...")

# ============================================================================
# STEP 4: RENDER 3-D SCENE
# ============================================================================
print(" Creating 3-D visualisation...")

pressure_vol = create_3Dvol_mesh(
    coords["x"], coords["y"], coords["z"], p_norm, scalars="Pressure"
)
tx_mesh = transducer.get_mesh()

if SAVE_FIG:
    plotter = pv.Plotter(window_size=(WIN_W * SCALE, WIN_H * SCALE), off_screen=True)
else:
    plotter = pv.Plotter(window_size=(WIN_W, WIN_H))

plotter = add_transducer_mesh(
    tx_mesh,
    plotter=plotter,
    scalars="Apodization",
    show_scalar_bar=False,
)
plotter = add_pressure_vol(
    pressure_vol,
    plotter=plotter,
    colorbar_title="Pressure",
    contour_levels=15,
    vmin=0,
    vmax=1,
    scalar_bar_args={
        "title": "Pressure (a.u.)",
        "title_font_size": 20 * SCALE,
        "label_font_size": 18 * SCALE,
        "vertical": True,
        "position_x": 0.8,
        "position_y": 0.6,
        "height": 0.3,
    },
)

if has_stl:
    plotter = add_stl_mesh(
        petri_dish,
        plotter=plotter,
        color="lightgray",
        opacity=0.3,
        show_edges=True,
        label="Petri Dish",
    )

plotter.camera.up = (0, 0, -1)
plotter.camera_position = [
    (-59.01688106487966, 71.55213364857292, 49.54316793764734),
    (3.631094007909384, -0.9265847114160961, 18.48889861413802),
    (0.42343180238636613, 0.6405023120483719, -0.6406733153398253),
]

if SAVE_FIG:
    plotter.screenshot(str(FIG_FOLDER / "stl_simulation.png"))
else:
    plotter.show()

plotter.close()
del sim, pressure_vol, tx_mesh, plotter
if has_stl:
    del petri_dish

print("\nDone.")
