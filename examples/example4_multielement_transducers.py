"""
Example 4: Multi-element Transducers — 3-D Pressure Visualisation

Computes a monochromatic focused pressure field for a linear array and a
matrix array, then visualises the transducer + pressure volume in 3-D
using PyVista.

Steps
-----
1. Define common focus and field grid
2. Simulate and render the linear array (Domino)
3. Simulate and render the matrix array (Zeus_Matrix)

Run with:
    uv run examples/example4_multielement_transducers.py
"""

import numpy as np
import pyvista as pv

# ============================================================================
# CONFIGURATION
# ============================================================================
from config import FIG_FOLDER, SAVE_FIG, SCALE

from pyfield.plotting import add_pressure_vol, add_transducer_mesh, create_3Dvol_mesh
from pyfield.psimulation import PyField
from pyfield.transducers import Domino, Zeus_Matrix
from pyfield.utilities import reshape_to_mapped_points

# SAVE_FIG = True
RUN_LINEAR_ARRAY = True
RUN_MATRIX_ARRAY = True
THEME = "white"

# Focus and F/D
FOCUS_MM = np.array([-2, 0, 8])
FOVERD = 1

# Field grid: small volume centred on the focus
X_HALF_MM = 0.25
Y_HALF_MM = 0.5
Z_HALF_MM = 1.5
DX_MM = 0.015
DY_MM = 0.02
DZ_MM = 0.05

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
    pv.global_theme.anti_aliasing = "ssaa"
    COLOR = "black"
    AMBIENT_TX = 1
    AMBIENT_PR = 0.5

print("\n --- Example 4: Multi-element Transducers (3-D) --- \n")

if SAVE_FIG:
    FIG_FOLDER.mkdir(exist_ok=True)

# ============================================================================
# STEP 1: BUILD FIELD-POINT GRID
# ============================================================================
x_extent_mm = [-X_HALF_MM + FOCUS_MM[0], X_HALF_MM + FOCUS_MM[0]]
y_extent_mm = [-Y_HALF_MM + FOCUS_MM[1], Y_HALF_MM + FOCUS_MM[1]]
z_extent_mm = [-Z_HALF_MM + FOCUS_MM[2], Z_HALF_MM + FOCUS_MM[2]]

field_point_mm = {
    "x_extent_mm": x_extent_mm,
    "y_extent_mm": y_extent_mm,
    "z_extent_mm": z_extent_mm,
    "dx_mm": DX_MM,
    "dy_mm": DY_MM,
    "dz_mm": DZ_MM,
}


# ============================================================================
# STEP 2: LINEAR ARRAY (DOMINO)
# ============================================================================
if RUN_LINEAR_ARRAY:
    print("\n--- Linear Array Transducer ---\n")

    linear_probe = Domino()
    linear_probe.compute_delays(focus_mm=FOCUS_MM)
    linear_probe.compute_apodization(focus_mm=FOCUS_MM, FoverD=1)
    linear_probe.plot_delays_apodization()

    linear_sim = PyField(linear_probe)
    p_linear, coords_l = linear_sim(field_point_mm, method="auto")

    # Build PyVista meshes
    tx_mesh_l = linear_probe.get_mesh()
    pr_mesh_l = create_3Dvol_mesh(
        p_linear / p_linear.max(),
        coords_l["x"],
        coords_l["y"],
        coords_l["z"],
        scalars="Pressure ",
    )

    # --- TX + Pressure scene ---
    scale = SCALE if SAVE_FIG else 1
    if SAVE_FIG:
        pl = pv.Plotter(window_size=(500 * SCALE, 600 * SCALE), off_screen=True)
    else:
        pl = pv.Plotter(window_size=(500, 600))

    pl = add_pressure_vol(pr_mesh_l, plotter=pl, ambient=AMBIENT_PR, scale=scale)
    pl = add_transducer_mesh(tx_mesh_l, plotter=pl, ambient=AMBIENT_TX, scale=scale)
    pl.add_axes(label_size=(0.1, 0.1))
    pl.show_grid(
        grid="back",
        color=COLOR,
        font_size=12 * scale,
        location="outer",
        xtitle="X (mm)",
        ytitle="Y (mm)",
        ztitle="Z (mm)",
        n_xlabels=5,
        n_ylabels=3,
        n_zlabels=6,
        use_3d_text=False,
    )
    pl.camera.up = (0, 0, -1)
    pl.camera_position = [
        (-22.51, 14.44, -13.41),
        (-1.10, -0.81, 5.54),
        (0.50, -0.30, -0.81),
    ]

    if SAVE_FIG:
        pl.screenshot(str(FIG_FOLDER / f"linear_array_field_{THEME}.png"))
    else:
        pl.show()
    pl.close()

    # --- Pressure-only scene ---
    if SAVE_FIG:
        pl2 = pv.Plotter(window_size=(400 * SCALE, 600 * SCALE), off_screen=True)
    else:
        pl2 = pv.Plotter(window_size=(400, 600))

    pl2 = add_pressure_vol(pr_mesh_l, plotter=pl2, ambient=AMBIENT_PR, scale=scale)
    pl2.add_axes(label_size=(0.1, 0.1))
    pl2.show_grid(
        grid="back",
        color=COLOR,
        font_size=12 * scale,
        location="outer",
        xtitle="X (mm)",
        ytitle="Y (mm)",
        ztitle="Z (mm)",
        n_xlabels=3,
        n_ylabels=5,
        n_zlabels=6,
        use_3d_text=False,
    )
    pl2.camera.up = (0, 0, -1)
    pl2.camera_position = [
        (2.08, 5.18, 4.38),
        (-1.92, -0.04, 8.02),
        (-0.25, -0.42, -0.87),
    ]

    if SAVE_FIG:
        pl2.screenshot(str(FIG_FOLDER / f"linear_array_pressure_field_{THEME}.png"))
    else:
        pl2.show()
    pl2.close()

    del pl, pl2, tx_mesh_l, pr_mesh_l

# ============================================================================
# STEP 3: MATRIX ARRAY (ZEUS_MATRIX)
# ============================================================================
if RUN_MATRIX_ARRAY:
    print("\n--- Matrix Array Transducer ---\n")

    matrix_probe = Zeus_Matrix()
    matrix_probe.compute_delays(focus_mm=FOCUS_MM)
    matrix_probe.compute_apodization(focus_mm=FOCUS_MM, FoverD=FOVERD)
    matrix_probe.plot_delays_apodization()

    matrix_sim = PyField(matrix_probe)
    p_matrix, coords_m = matrix_sim(field_point_mm, method="auto")

    tx_mesh_m = matrix_probe.get_mesh()
    pr_mesh_m = create_3Dvol_mesh(
        p_matrix / p_matrix.max(),
        coords_m["x"],
        coords_m["y"],
        coords_m["z"],
        scalars="Pressure",
    )

    # --- TX + Pressure scene ---
    scale = SCALE if SAVE_FIG else 1
    if SAVE_FIG:
        pl = pv.Plotter(window_size=(600 * SCALE, 600 * SCALE), off_screen=True)
    else:
        pl = pv.Plotter(window_size=(600, 600))

    pl = add_pressure_vol(pr_mesh_m, plotter=pl, ambient=AMBIENT_PR, scale=scale)
    pl = add_transducer_mesh(tx_mesh_m, plotter=pl, ambient=AMBIENT_TX, scale=scale)
    pl.add_axes(label_size=(0.1, 0.1))
    pl.show_grid(
        grid="back",
        color=COLOR,
        font_size=12 * scale,
        location="outer",
        xtitle="X (mm)",
        ytitle="Y (mm)",
        ztitle="Z (mm)",
        n_xlabels=5,
        n_ylabels=5,
        n_zlabels=5,
        use_3d_text=False,
    )
    pl.camera.up = (0, 0, -1)
    pl.camera_position = [
        (-37.54, 25.79, -29.55),
        (0.56, 0.003, 4.80),
        (0.11, -0.73, -0.67),
    ]

    if SAVE_FIG:
        pl.screenshot(str(FIG_FOLDER / f"matrix_array_field_{THEME}.png"))
    else:
        pl.show()
    pl.close()

    # --- Pressure-only scene ---
    if SAVE_FIG:
        pl2 = pv.Plotter(window_size=(400 * SCALE, 600 * SCALE), off_screen=True)
    else:
        pl2 = pv.Plotter(window_size=(400, 600))

    pl2 = add_pressure_vol(pr_mesh_m, plotter=pl2, ambient=AMBIENT_PR, scale=scale)
    pl2.add_axes(label_size=(0.1, 0.1))
    pl2.show_grid(
        grid="back",
        color=COLOR,
        font_size=12 * scale,
        location="outer",
        xtitle="X (mm)",
        ytitle="Y (mm)",
        ztitle="Z (mm)",
        n_xlabels=3,
        n_ylabels=5,
        n_zlabels=6,
        use_3d_text=False,
    )
    pl2.camera.up = (0, 0, -1)
    pl2.camera_position = [
        (2.08, 5.18, 4.38),
        (-1.92, -0.04, 8.02),
        (-0.25, -0.42, -0.87),
    ]

    if SAVE_FIG:
        pl2.screenshot(str(FIG_FOLDER / f"matrix_array_pressure_field_{THEME}.png"))
    else:
        pl2.show()
    pl2.close()

    del pl, pl2, tx_mesh_m, pr_mesh_m
