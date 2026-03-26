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

# matplotlib.use("Agg")  # non-interactive backend — must be before pyplot import
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

from pyfield.brain_atlas import BG_Atlas
from pyfield.psimulation import PyField
from pyfield.transducers import (
    ConcaveCircularTransducer,
    ConvexArrayTransducer,
    ConvexCircularTransducer,
    CustomTransducer,
    Domino,
    FlatCircularTransducer,
    FocusedCircularTransducer,
    LinearArrayTransducer,
    MatrixArrayTransducer,
    Zeus_Matrix,
)
from pyfield.utilities import (
    add_pressure_vol,
    add_regions_mesh,
    add_transducer_mesh,
    create_vol_mesh,
    plot_pressure_field,
    plot_pressure_planes,
    plot_slices_2d,
)

OUT = Path("docs/assets")
OUT.mkdir(parents=True, exist_ok=True)

FC_HZ = 1.5e6
FOCUS = [0, 0, 50]
WIN = (700, 500)
OFF_SCREEN = True
#
# # ---------------------------------------------------------------------------
# # Helper — screenshot a transducer mesh
# # ---------------------------------------------------------------------------
# def _save_transducer(tx, path: Path, scalars="Apodization"):
#     mesh = tx.get_mesh()
#     pl = pv.Plotter(window_size=WIN, off_screen=True)
#
#     cmap = "cool" if scalars == "Apodization" else "rainbow"
#     title = "Apodization" if scalars == "Apodization" else "Delays (s)"
#     clim = [0, 1] if scalars == "Apodization" else None
#
#     pl.add_mesh(
#         mesh,
#         scalars=scalars,
#         cmap=cmap,
#         clim=clim,
#         show_scalar_bar=True,
#         scalar_bar_args={
#             "title": title,
#             "vertical": True,
#             "position_x": 0.80,
#             "position_y": 0.10,
#         },
#         show_edges=True,
#         ambient=1,
#     )
#     pl.add_axes()
#     pl.show_bounds(
#         grid="back",
#         location="outer",
#         xtitle="X (mm)",
#         ytitle="Y (mm)",
#         ztitle="Z (mm)",
#         n_xlabels=3,
#         n_ylabels=3,
#         n_zlabels=3,
#         font_size=10,
#     )
#     pl.camera.up = (0, 0, -1)
#     pl.reset_camera()
#     pl.screenshot(str(path), return_img=False)
#     pl.close()
#     del mesh, pl
#     print(f"  saved {path}")
#
#
# # ---------------------------------------------------------------------------
# # Helper — focal-law figure (delays + apodization side-by-side)
# # ---------------------------------------------------------------------------
# def _save_focal_law(tx, path: Path):
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
#     tx.plot_delays(ax=ax1)
#     tx.plot_apodization(ax=ax2)
#     plt.tight_layout()
#     fig.savefig(str(path), dpi=120, bbox_inches="tight")
#     plt.close(fig)
#     del fig, ax1, ax2
#     print(f"  saved {path}")
#
#
# ===========================================================================
# 4. ConvexCircularTransducer -- lenses
# ===========================================================================
print("\n--- 4. ConvexCircularTransducer — geometry screenshot ---")
conv = ConvexCircularTransducer(
    diameter_mm=30.0,
    radius_of_curvature_mm=15.0,
    no_sub=30,
    frequency_Hz=1.5e6,
    patch_fill=0.8,
)
mesh_conv = conv.get_mesh()
conv.show()
pl_conv = pv.Plotter(window_size=WIN, off_screen=OFF_SCREEN)
pl_conv.add_mesh(
    mesh_conv,
    scalars="Apodization",
    cmap="cool",
    clim=[0, 1],
    show_scalar_bar=True,
    scalar_bar_args={
        "title": "Apodization",
        "vertical": True,
        "position_x": 0.80,
        "position_y": 0.10,
    },
    show_edges=True,
    ambient=1,
)
pl_conv.add_axes()
pl_conv.show_bounds(
    grid="back",
    location="outer",
    xtitle="X (mm)",
    ytitle="Y (mm)",
    ztitle="Z (mm)",
    n_xlabels=3,
    n_ylabels=3,
    n_zlabels=3,
    font_size=10,
)
pl_conv.reset_camera()
pl_conv.screenshot(str(OUT / "transducer_convex_circular.png"), return_img=False)
pl_conv.close()
del mesh_conv, pl_conv
print(f"  saved {OUT / 'transducer_convex_circular.png'}")

# ===========================================================================
# Subdivision utility — ellipsoidal cap example figure
# ===========================================================================
print("\n--- Ellipsoidal cap subdivision figure ---")
from pyfield.utilities.surface_subdivision import subdivide_parametric_surface

a, b, c_ax = 30e-3, 20e-3, 15e-3  # semi-axes in metres
R_ap = 15e-3  # circular aperture radius


def ellipsoid_cap(x, y):
    arg = max(1.0 - (x / a) ** 2 - (y / b) ** 2, 0.0)
    return np.array([x, y, c_ax * np.sqrt(arg)])


frames = subdivide_parametric_surface(
    ellipsoid_cap,
    u_range=(-R_ap, R_ap),
    v_range=(-R_ap, R_ap),
    n_u=10,
    n_v=10,
    inside_fn=lambda x, y: x**2 / a**2 + y**2 / b**2 <= 1.0,
    normal_sign=1.0,
    patch_fill=0.9,
    max_patch_scale=1.5,
)

centers = frames["centers"]  # (M, 3)  in metres
normals = frames["normals"]  # (M, 3)
wu = frames["wu"]  # (M,)  half-widths
wv = frames["wv"]
coverage = frames["coverage"]
n_rejected = frames["n_rejected"]
M = len(centers)

# Build a matplotlib 3-D figure: surface wireframe + patch centres + normals
fig = plt.figure(figsize=(12, 5))

# Left panel — 3-D view of patches
ax3d = fig.add_subplot(121, projection="3d")
tu = frames["tangents_u"]  # (M, 3)
tv = frames["tangents_v"]  # (M, 3)

for i in range(M):
    c = centers[i]
    u_half = wu[i] * tu[i]
    v_half = wv[i] * tv[i]
    corners = (
        np.array(
            [
                c - u_half - v_half,
                c + u_half - v_half,
                c + u_half + v_half,
                c - u_half + v_half,
                c - u_half - v_half,  # close loop
            ]
        )
        * 1e3
    )  # → mm
    ax3d.plot(
        corners[:, 0],
        corners[:, 1],
        corners[:, 2],
        color="steelblue",
        linewidth=0.4,
        alpha=0.7,
    )

# Draw normals (subsampled for clarity)
step = max(1, M // 60)
scale_n = 1.5  # mm
for i in range(0, M, step):
    c_mm = centers[i] * 1e3
    n_mm = normals[i] * scale_n
    ax3d.quiver(
        *c_mm,
        *n_mm,
        length=1,
        normalize=False,
        color="tomato",
        linewidth=0.8,
        arrow_length_ratio=0.3,
    )

ax3d.set_xlabel("X (mm)", fontsize=8)
ax3d.set_ylabel("Y (mm)", fontsize=8)
ax3d.set_zlabel("Z (mm)", fontsize=8)
ax3d.set_title(
    f"Ellipsoidal cap — {M} patches\ncoverage {coverage * 100:.1f} %, {n_rejected} rejected",
    fontsize=9,
)
ax3d.view_init(elev=25, azim=-60)
ax3d.tick_params(labelsize=7)

# Right panel — top-down XY map of patch centres, coloured by patch area
ax2d = fig.add_subplot(122)
areas = (2 * wu) * (2 * wv) * 1e6  # mm²
sc = ax2d.scatter(
    centers[:, 0] * 1e3,
    centers[:, 1] * 1e3,
    c=areas,
    cmap="plasma",
    s=8,
    alpha=0.85,
)
cb = fig.colorbar(sc, ax=ax2d, pad=0.02)
cb.set_label("Patch area (mm²)", fontsize=8)
cb.ax.tick_params(labelsize=7)
# draw aperture boundary
theta_circ = np.linspace(0, 2 * np.pi, 300)
ax2d.plot(
    R_ap * 1e3 * np.cos(theta_circ),
    R_ap * 1e3 * np.sin(theta_circ),
    "k--",
    linewidth=0.8,
    label="Aperture",
)
ax2d.set_xlabel("X (mm)", fontsize=8)
ax2d.set_ylabel("Y (mm)", fontsize=8)
ax2d.set_title("Top view — patch centres (colour = area)", fontsize=9)
ax2d.set_aspect("equal")
ax2d.tick_params(labelsize=7)
ax2d.legend(fontsize=7)

plt.tight_layout()
fig.savefig(str(OUT / "subdivision_ellipsoid_cap.png"), dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"  saved {OUT / 'subdivision_ellipsoid_cap.png'}")

# ---------------------------------------------------------------------------
# PyVista comparison: theoretical curved surface vs flat patch mosaic
# ---------------------------------------------------------------------------
print("\n--- Ellipsoidal cap — PyVista comparison figure ---")

# Build the theoretical ellipsoidal cap surface as a dense mesh (mm units)
N_SURF = 120
u_lin = np.linspace(-R_ap, R_ap, N_SURF)
v_lin = np.linspace(-R_ap, R_ap, N_SURF)
UU, VV = np.meshgrid(u_lin, v_lin)
mask_surf = (UU / a) ** 2 + (VV / b) ** 2 <= 1.0
ZZ = np.where(
    mask_surf,
    c_ax * np.sqrt(np.maximum(1.0 - (UU / a) ** 2 - (VV / b) ** 2, 0.0)),
    np.nan,
)

# PyVista StructuredGrid — filter out NaN rows/cols at edges
pts_surf = np.column_stack([UU.ravel() * 1e3, VV.ravel() * 1e3, ZZ.ravel() * 1e3])
valid = ~np.isnan(pts_surf[:, 2])
pts_surf[~valid, 2] = 0.0  # pyvista needs no NaN; we'll colour them transparent

grid = pv.StructuredGrid()
grid.points = pts_surf
grid.dimensions = [N_SURF, N_SURF, 1]
grid["valid"] = valid.astype(float)

# Build the patch mosaic as a PolyData (one quad cell per patch, in mm)
tu_arr = frames["tangents_u"]
tv_arr = frames["tangents_v"]

all_pts = []
all_faces = []
offset = 0
for i in range(M):
    c_i = centers[i] * 1e3  # → mm
    u_h = wu[i] * tu_arr[i] * 1e3
    v_h = wv[i] * tv_arr[i] * 1e3
    quad = np.array(
        [
            c_i - u_h - v_h,
            c_i + u_h - v_h,
            c_i + u_h + v_h,
            c_i - u_h + v_h,
        ]
    )
    all_pts.append(quad)
    all_faces.append([4, offset, offset + 1, offset + 2, offset + 3])
    offset += 4

patch_pts = np.vstack(all_pts)
patch_faces = np.array(all_faces).ravel()
patch_mesh = pv.PolyData(patch_pts, patch_faces)
patch_mesh["area_mm2"] = (2 * wu * 2 * wv) * 1e6  # mm²

# Plot — two side-by-side viewports: theoretical | patches
pl2 = pv.Plotter(shape=(1, 2), window_size=(1100, 480), off_screen=OFF_SCREEN)

# Left — theoretical curved surface coloured by height (Z)
pl2.subplot(0, 0)
pl2.add_text("Theoretical surface", font_size=9, position="upper_edge")
pl2.add_mesh(
    grid.threshold(0.5, scalars="valid"),  # keep only valid cells
    scalars=grid.points[grid["valid"] > 0.5, 2],
    cmap="viridis",
    show_scalar_bar=False,
    opacity=1.0,
    ambient=0.4,
)
pl2.add_axes()
pl2.view_isometric()
pl2.reset_camera()

# Right — flat patch mosaic coloured by patch area
pl2.subplot(0, 1)
pl2.add_text(
    f"Flat patch mosaic  ({M} patches, coverage {coverage * 100:.0f} %)",
    font_size=9,
    position="upper_edge",
)
pl2.add_mesh(
    patch_mesh,
    scalars="area_mm2",
    cmap="plasma",
    show_scalar_bar=True,
    scalar_bar_args={
        "title": "Patch area (mm²)",
        "vertical": True,
        "position_x": 0.82,
        "position_y": 0.15,
        "title_font_size": 9,
        "label_font_size": 8,
    },
    show_edges=True,
    edge_color="white",
    line_width=0.3,
    ambient=0.6,
)
# overlay the theoretical surface as a semi-transparent wireframe for comparison
pl2.add_mesh(
    grid.threshold(0.5, scalars="valid"),
    color="cyan",
    opacity=0.18,
    show_scalar_bar=False,
)
pl2.add_axes()
pl2.view_isometric()
pl2.reset_camera()

pl2.screenshot(str(OUT / "subdivision_ellipsoid_cap_pyvista.png"), return_img=False)
pl2.close()
del grid, patch_mesh, all_pts, all_faces, patch_pts, patch_faces
print(f"  saved {OUT / 'subdivision_ellipsoid_cap_pyvista.png'}")
#
# # # ===========================================================================
# # # 13. Mouse brain atlas + ConcaveCircularTransducer + pressure field
# # # ===========================================================================
# # print("13. Mouse brain atlas scene (allen_mouse_25um)")
# # focus_depth_mm = 10
# # mouse_tx = ConcaveCircularTransducer(
# #     diameter_mm=10.0,
# #     radius_of_curvature_mm=focus_depth_mm,
# #     no_sub=20,
# #     frequency_Hz=5e6,
# # )
# #
# # mouse_sim = PyField(mouse_tx, verbose=False)
# # x_mus, y_mus, z_mus, p_mus = mouse_sim(
# #     {
# #         "x_extent": [-1, 1],
# #         "y_extent": [-1, 1],
# #         "z_extent": [-1 + focus_depth_mm, 1 + focus_depth_mm],
# #         "dx": 0.04,
# #         "dy": 0.04,
# #         "dz": 0.04,
# #     },
# #     method="auto",
# # )
# #
# # # plot_pressure_planes(x_mus, y_mus, z_mus, p_mus)
# #
# # pressure_vol_mouse = create_vol_mesh(
# #     x_mus, y_mus, z_mus, p_mus / p_mus.max(), scalars="Pressure"
# # )
# # del x_mus, y_mus, z_mus, p_mus, mouse_sim
# # gc.collect()
# #
# # region_names_mouse = ("root", "Isocortex", "CA1")
# # mouse_atlas = BG_Atlas("allen_mouse_25um", region_names=region_names_mouse)
# #
# # lambda_bregma_mm_m = 5.0
# # cortex2probe_mm_m = focus_depth_mm - 1.0
# # scale_mouse = np.eye(4)
# # scale_mouse[:3, :3] *= lambda_bregma_mm_m
# # atlas_z_max_mouse = mouse_atlas.pv_mesh["root"].bounds[5]
# # trans_depth_m = np.eye(4)
# # trans_depth_m[2, 3] = -cortex2probe_mm_m - atlas_z_max_mouse * lambda_bregma_mm_m
# # inv_z_m = np.diag([1.0, 1.0, -1.0, 1.0])
# # T_mouse = inv_z_m @ trans_depth_m @ scale_mouse
# # mouse_atlas.transform(T_matrix=T_mouse, inplace=True)
# #
# # pl_mouse = pv.Plotter(window_size=(800, 600), off_screen=OFF_SCREEN)
# # pl_mouse = add_regions_mesh(
# #     mouse_atlas.pv_mesh,
# #     plotter=pl_mouse,
# #     kwargs_dict={
# #         region_names_mouse[0]: {"color": "lightgray", "opacity": 0.2},
# #         region_names_mouse[1]: {"color": "lightblue", "opacity": 0.2},
# #         region_names_mouse[2]: {"color": "salmon", "opacity": 0.2},
# #     },
# # )
# # pl_mouse = add_transducer_mesh(mouse_tx.get_mesh(), plotter=pl_mouse)
# # pl_mouse = add_pressure_vol(pressure_vol_mouse, plotter=pl_mouse, plot_focal_spot=False)
# # pl_mouse.show_bounds(
# #     grid="back",
# #     location="outer",
# #     xtitle="X (mm)",
# #     ytitle="Y (mm)",
# #     ztitle="Z (mm)",
# #     font_size=10,
# #     use_3d_text=False,
# # )
# # pl_mouse.add_axes()
# # pl_mouse.camera.up = (0, 0, -1)
# # pl_mouse.reset_camera()
# # # pl_mouse.show()
# # # pl_mouse.camera_position
# # pl_mouse.screenshot(str(OUT / "brain_mouse_scene.png"), return_img=False)
# # pl_mouse.close()
# # del pl_mouse, mouse_atlas, pressure_vol_mouse, mouse_tx
# # del scale_mouse, trans_depth_m, inv_z_m, T_mouse
# # gc.collect()
# # print(f"  saved {OUT / 'brain_mouse_scene.png'}")
# #
# #
# # # ===========================================================================
# # # 12. Rat brain atlas + Domino TX + pressure field
# # # ===========================================================================
# # print("12. Rat brain atlas scene (whs_sd_rat)")
# # focus_rat = np.array([-1, 0, 8])
# # rat_tx = Domino()
# # rat_tx.compute_delays(focus_mm=focus_rat)
# # rat_tx.compute_apodization(focus_mm=focus_rat, FoverD=1, apodization_type="rect")
# #
# # rat_sim = PyField(rat_tx, verbose=False)
# # x_rat, y_rat, z_rat, p_rat = rat_sim(
# #     {
# #         "x_extent": [-0.25 + focus_rat[0], 0.25 + focus_rat[0]],
# #         "y_extent": [-0.5 + focus_rat[1], 0.5 + focus_rat[1]],
# #         "z_extent": [-1.0 + focus_rat[2], 1.0 + focus_rat[2]],
# #         "dx": 0.0125,
# #         "dy": 0.025,
# #         "dz": 0.05,
# #     },
# #     method="auto",
# # )
# # pressure_vol_rat = create_vol_mesh(
# #     x_rat, y_rat, z_rat, p_rat / p_rat.max(), scalars="Pressure"
# # )
# # del x_rat, y_rat, z_rat, p_rat, rat_sim
# # gc.collect()
# #
# # region_names_rat = ("root", "M1", "S1-hl")
# # rat_atlas = BG_Atlas("whs_sd_rat_39um", region_names=region_names_rat)
# #
# # lambda_bregma_mm = 8.0
# # cortex2probe_mm = 4.5
# # scale_rat = np.eye(4)
# # scale_rat[:3, :3] *= lambda_bregma_mm
# # atlas_z_max_rat = rat_atlas.pv_mesh["root"].bounds[5]
# # trans_depth = np.eye(4)
# # trans_depth[2, 3] = -cortex2probe_mm - atlas_z_max_rat * lambda_bregma_mm
# # trans_xy = np.eye(4)
# # trans_xy[0, 3] = 2.0
# # trans_xy[1, 3] = -2.0
# # inv_z = np.diag([1.0, 1.0, -1.0, 1.0])
# # T_rat = inv_z @ trans_depth @ trans_xy @ scale_rat
# # rat_atlas.transform(T_matrix=T_rat, inplace=True)
# #
# # pl_rat = pv.Plotter(window_size=(800, 600), off_screen=True)
# # pl_rat = add_regions_mesh(
# #     rat_atlas.pv_mesh,
# #     plotter=pl_rat,
# #     kwargs_dict={
# #         region_names_rat[0]: {"color": "lightgray", "opacity": 0.35},
# #         region_names_rat[1]: {"color": "permanentgreen", "opacity": 0.5},
# #         region_names_rat[2]: {"color": "cadmiumlemon", "opacity": 0.5},
# #     },
# # )
# # pl_rat = add_transducer_mesh(rat_tx.get_mesh(), plotter=pl_rat, scalars="Delays")
# # pl_rat = add_pressure_vol(pressure_vol_rat, plotter=pl_rat, plot_focal_spot=True)
# # pl_rat.show_bounds(
# #     grid="back",
# #     location="outer",
# #     xtitle="X (mm)",
# #     ytitle="Y (mm)",
# #     ztitle="Z (mm)",
# #     font_size=10,
# #     use_3d_text=False,
# # )
# # pl_rat.add_axes()
# # pl_rat.camera.up = (0, 0, -1)
# # pl_rat.camera_position = [(-15.2, 35.7, -13.2), (2.4, -6.8, 10.3), (0.20, -0.41, -0.89)]
# # pl_rat.screenshot(str(OUT / "brain_rat_scene.png"), return_img=False)
# # pl_rat.close()
# # del pl_rat, rat_atlas, pressure_vol_rat, rat_tx
# # del scale_rat, trans_depth, trans_xy, inv_z, T_rat, focus_rat
# # gc.collect()
# # print(f"  saved {OUT / 'brain_rat_scene.png'}")
# #
#
print("\nDone — all images saved to", OUT.resolve())
