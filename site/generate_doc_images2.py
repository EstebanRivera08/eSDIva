"""
Generate static images for the PyField documentation.

Saves all figures to docs/assets/.  Run with:

    uv run generate_doc_images.py

Requirements: PyVista must be able to render off-screen (Mesa / osmesa or a
real display).  On headless Linux set:  export DISPLAY=:99
"""

import gc
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend — must be before pyplot import
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

pv.OFF_SCREEN = True  # global off-screen flag for PyVista

from pyfield.brain_atlas import BG_Atlas
from pyfield.psimulation import PyField
from pyfield.transducers import (
    ConcaveCircularTransducer,
    Domino,
    Zeus_Matrix,
)
from pyfield.utilities import (
    add_pressure_vol,
    add_regions_mesh,
    add_transducer_mesh,
    create_vol_mesh,
    plot_pressure_field,
    plot_pressure_planes,
)

OUT = Path("docs/assets")
OUT.mkdir(parents=True, exist_ok=True)

FC_HZ = 1.5e6
FOCUS = [0, 0, 50]
WIN = (700, 500)


# ===========================================================================
# 8. Monochromatic pressure field — Domino linear array (2-D planes)
# ===========================================================================
print("8. Monochromatic pressure — Domino (2-D planes)")
sim_tx = Domino()
sim_tx.compute_delays(focus_mm=[0, 0, 8])
sim_tx.compute_apodization(focus_mm=[0, 0, 8], FoverD=1)

sim = PyField(sim_tx)
x, y, z, p = sim(
    {
        "x_extent": [-0.25, 0.25],
        "y_extent": [-0.5, 0.5],
        "z_extent": [6.5, 9.5],
        "dx": 0.015,
        "dy": 0.02,
        "dz": 0.05,
    },
    method="auto",
)
plot_pressure_planes(
    x,
    y,
    z,
    p,
    db_scale=False,
    p_max=p.max(),
    centered_to_max=False,
    figsize=(9, 5),
    vmin=0,
    vmax=1,
    label="Pressure (a.u.)",
    save_fig_name="pressure_linear_xz.png",
    save_dir=str(OUT),
)
plt.close("all")
print(f"  saved {OUT / 'pressure_linear_xz.png'}")


# ===========================================================================
# 9. Monochromatic pressure field — Zeus_Matrix (2-D planes)
# ===========================================================================
print("9. Monochromatic pressure — Zeus_Matrix (2-D planes)")
focus = (-2, 0, 8)
sim_tx2 = Zeus_Matrix()
sim_tx2.compute_delays(focus_mm=focus)
sim_tx2.compute_apodization(focus_mm=focus, FoverD=1)

sim2 = PyField(sim_tx2)
x2, y2, z2, p2 = sim2(
    {
        "x_extent": [-0.25 + focus[0], 0.25 + focus[0]],
        "y_extent": [-0.5 + focus[1], 0.5 + focus[1]],
        "z_extent": [-1.5 + focus[2], 1.5 + focus[2]],
        "dx": 0.015,
        "dy": 0.025,
        "dz": 0.05,
    },
    method="auto",
)
plot_pressure_planes(
    x2,
    y2,
    z2,
    p2,
    db_scale=False,
    p_max=p2.max(),
    figsize=(9, 5),
    centered_to_max=False,
    vmin=0,
    vmax=1,
    label="Pressure (a.u.)",
    save_fig_name="pressure_matrix_3d.png",
    save_dir=str(OUT),
)
plt.close("all")
print(f"  saved {OUT / 'pressure_matrix_3d.png'}")


# ===========================================================================
# 10. 3-D PyVista — Domino pressure field
# ===========================================================================
print("10. 3-D pressure (PyVista) — Domino")
pl_lin = plot_pressure_field(
    x,
    y,
    z,
    p,
    off_screen=True,
    contour_levels=11,
    colorbar_title="Pressure (a.u.)",
)
pl_lin.camera.up = (0, 0, -1)
pl_lin.reset_camera()
pl_lin.screenshot(str(OUT / "pressure_linear_3d.png"), return_img=False)
pl_lin.close()
del pl_lin, x, y, z, p, sim, sim_tx
gc.collect()
print(f"  saved {OUT / 'pressure_linear_3d.png'}")


# ===========================================================================
# 11. 3-D PyVista — Zeus_Matrix pressure field
# ===========================================================================
print("11. 3-D pressure (PyVista) — Zeus_Matrix")
pl_mat = plot_pressure_field(
    x2,
    y2,
    z2,
    p2,
    off_screen=True,
    contour_levels=11,
    colorbar_title="Pressure (a.u.)",
)
pl_mat.camera.up = (0, 0, -1)
pl_mat.reset_camera()
pl_mat.screenshot(str(OUT / "pressure_matrix_3d_pyvista.png"), return_img=False)
pl_mat.close()
del pl_mat, x2, y2, z2, p2, sim2, sim_tx2, focus
gc.collect()
print(f"  saved {OUT / 'pressure_matrix_3d_pyvista.png'}")


# ===========================================================================
# 12. Rat brain atlas + Domino TX + pressure field
# ===========================================================================
print("12. Rat brain atlas scene (whs_sd_rat)")
focus_rat = np.array([-1, 0, 8])
rat_tx = Domino()
rat_tx.compute_delays(focus_mm=focus_rat)
rat_tx.compute_apodization(focus_mm=focus_rat, FoverD=1, apodization_type="rect")

rat_sim = PyField(rat_tx, verbose=False)
x_rat, y_rat, z_rat, p_rat = rat_sim(
    {
        "x_extent": [-0.25 + focus_rat[0], 0.25 + focus_rat[0]],
        "y_extent": [-0.5 + focus_rat[1], 0.5 + focus_rat[1]],
        "z_extent": [-1.0 + focus_rat[2], 1.0 + focus_rat[2]],
        "dx": 0.0125,
        "dy": 0.025,
        "dz": 0.05,
    },
    method="auto",
)
pressure_vol_rat = create_vol_mesh(
    x_rat, y_rat, z_rat, p_rat / p_rat.max(), scalars="Pressure"
)
del x_rat, y_rat, z_rat, p_rat, rat_sim
gc.collect()

region_names_rat = ("root", "M1", "S1-hl")
rat_atlas = BG_Atlas("whs_sd_rat_39um", region_names=region_names_rat)

lambda_bregma_mm = 8.0
cortex2probe_mm = 4.5
scale_rat = np.eye(4)
scale_rat[:3, :3] *= lambda_bregma_mm
atlas_z_max_rat = rat_atlas.pv_mesh["root"].bounds[5]
trans_depth = np.eye(4)
trans_depth[2, 3] = -cortex2probe_mm - atlas_z_max_rat * lambda_bregma_mm
trans_xy = np.eye(4)
trans_xy[0, 3] = 2.0
trans_xy[1, 3] = -2.0
inv_z = np.diag([1.0, 1.0, -1.0, 1.0])
T_rat = inv_z @ trans_depth @ trans_xy @ scale_rat
rat_atlas.transform(T_matrix=T_rat, inplace=True)

pl_rat = pv.Plotter(window_size=(800, 600), off_screen=True)
pl_rat = add_regions_mesh(
    rat_atlas.pv_mesh,
    plotter=pl_rat,
    kwargs_dict={
        region_names_rat[0]: {"color": "lightgray", "opacity": 0.35},
        region_names_rat[1]: {"color": "permanentgreen", "opacity": 0.5},
        region_names_rat[2]: {"color": "cadmiumlemon", "opacity": 0.5},
    },
)
pl_rat = add_transducer_mesh(rat_tx.get_mesh(), plotter=pl_rat, scalars="Delays")
pl_rat = add_pressure_vol(pressure_vol_rat, plotter=pl_rat, plot_focal_spot=True)
pl_rat.show_bounds(
    grid="back",
    location="outer",
    xtitle="X (mm)",
    ytitle="Y (mm)",
    ztitle="Z (mm)",
    font_size=10,
    use_3d_text=False,
)
pl_rat.add_axes()
pl_rat.camera.up = (0, 0, -1)
pl_rat.camera_position = [(-15.2, 35.7, -13.2), (2.4, -6.8, 10.3), (0.20, -0.41, -0.89)]
pl_rat.screenshot(str(OUT / "brain_rat_scene.png"), return_img=False)
pl_rat.close()
del pl_rat, rat_atlas, pressure_vol_rat, rat_tx
del scale_rat, trans_depth, trans_xy, inv_z, T_rat, focus_rat
gc.collect()
print(f"  saved {OUT / 'brain_rat_scene.png'}")


# ===========================================================================
# 13. Mouse brain atlas + ConcaveCircularTransducer + pressure field
# ===========================================================================
print("13. Mouse brain atlas scene (allen_mouse_25um)")
focus_depth_mm = 10
mouse_tx = ConcaveCircularTransducer(
    diameter_mm=10.0,
    radius_of_curvature_mm=focus_depth_mm,
    no_sub=20,
    frequency_Hz=5e6,
)

mouse_sim = PyField(mouse_tx, verbose=False)
x_mus, y_mus, z_mus, p_mus = mouse_sim(
    {
        "x_extent": [-1, 1],
        "y_extent": [-1, 1],
        "z_extent": [-1 + focus_depth_mm, 1 + focus_depth_mm],
        "dx": 0.04,
        "dy": 0.04,
        "dz": 0.04,
    },
    method="auto",
)

# plot_pressure_planes(x_mus, y_mus, z_mus, p_mus)

pressure_vol_mouse = create_vol_mesh(
    x_mus, y_mus, z_mus, p_mus / p_mus.max(), scalars="Pressure"
)
del x_mus, y_mus, z_mus, p_mus, mouse_sim
gc.collect()

region_names_mouse = ("root", "Isocortex", "CA1")
mouse_atlas = BG_Atlas("allen_mouse_25um", region_names=region_names_mouse)

lambda_bregma_mm_m = 5.0
cortex2probe_mm_m = focus_depth_mm - 1.0
scale_mouse = np.eye(4)
scale_mouse[:3, :3] *= lambda_bregma_mm_m
atlas_z_max_mouse = mouse_atlas.pv_mesh["root"].bounds[5]
trans_depth_m = np.eye(4)
trans_depth_m[2, 3] = -cortex2probe_mm_m - atlas_z_max_mouse * lambda_bregma_mm_m
inv_z_m = np.diag([1.0, 1.0, -1.0, 1.0])
T_mouse = inv_z_m @ trans_depth_m @ scale_mouse
mouse_atlas.transform(T_matrix=T_mouse, inplace=True)

pl_mouse = pv.Plotter(window_size=(800, 600), off_screen=True)
pl_mouse = add_regions_mesh(
    mouse_atlas.pv_mesh,
    plotter=pl_mouse,
    kwargs_dict={
        region_names_mouse[0]: {"color": "lightgray", "opacity": 0.2},
        region_names_mouse[1]: {"color": "lightblue", "opacity": 0.2},
        region_names_mouse[2]: {"color": "salmon", "opacity": 0.2},
    },
)
pl_mouse = add_transducer_mesh(mouse_tx.get_mesh(), plotter=pl_mouse)
pl_mouse = add_pressure_vol(pressure_vol_mouse, plotter=pl_mouse, plot_focal_spot=False)
pl_mouse.show_bounds(
    grid="back",
    location="outer",
    xtitle="X (mm)",
    ytitle="Y (mm)",
    ztitle="Z (mm)",
    font_size=10,
    use_3d_text=False,
)
pl_mouse.add_axes()
pl_mouse.camera.up = (0, 0, -1)
pl_mouse.reset_camera()
# pl_mouse.show()
# pl_mouse.camera_position
pl_mouse.screenshot(str(OUT / "brain_mouse_scene.png"), return_img=False)
pl_mouse.close()
del pl_mouse, mouse_atlas, pressure_vol_mouse, mouse_tx
del scale_mouse, trans_depth_m, inv_z_m, T_mouse
gc.collect()
print(f"  saved {OUT / 'brain_mouse_scene.png'}")

# ===========================================================================


print("\nDone — all images saved to", OUT.resolve())
