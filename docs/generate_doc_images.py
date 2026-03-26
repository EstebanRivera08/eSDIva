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

from pyfield.transducers import (
    ConcaveCircularTransducer,
    ConvexArrayTransducer,
    ConvexCircularTransducer,
    CustomTransducer,
    FlatCircularTransducer,
    FocusedCircularTransducer,
    LinearArrayTransducer,
    MatrixArrayTransducer,
)
from pyfield.utilities import subdivide_parametric_surface

OUT = Path("assets")
OUT.mkdir(parents=True, exist_ok=True)

FC_HZ = 1.5e6
FOCUS = [0, 0, 50]
WIN = (700, 500)
OFF_SCREEN = True


# ---------------------------------------------------------------------------
# Helper — screenshot a transducer mesh
# ---------------------------------------------------------------------------
def _save_transducer(tx, path: Path, scalars="Apodization"):
    mesh = tx.get_mesh()
    pl = pv.Plotter(window_size=WIN, off_screen=True)

    cmap = "cool" if scalars == "Apodization" else "rainbow"
    title = "Apodization" if scalars == "Apodization" else "Delays (s)"
    clim = [0, 1] if scalars == "Apodization" else None

    pl.add_mesh(
        mesh,
        scalars=scalars,
        cmap=cmap,
        clim=clim,
        show_scalar_bar=True,
        scalar_bar_args={
            "title": title,
            "vertical": True,
            "position_x": 0.80,
            "position_y": 0.10,
        },
        show_edges=True,
        ambient=1,
    )
    pl.add_axes()
    pl.show_bounds(
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
    pl.camera.up = (0, 0, -1)
    pl.reset_camera()
    pl.screenshot(str(path), return_img=False)
    pl.close()
    del mesh, pl
    print(f"  saved {path}")


# ---------------------------------------------------------------------------
# Helper — focal-law figure (delays + apodization side-by-side)
# ---------------------------------------------------------------------------
def _save_focal_law(tx, path: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    tx.plot_delays(ax=ax1)
    tx.plot_apodization(ax=ax2)
    plt.tight_layout()
    fig.savefig(str(path), dpi=120, bbox_inches="tight")
    plt.close(fig)
    del fig, ax1, ax2
    print(f"  saved {path}")


# ===========================================================================
# 1. Linear array
# ===========================================================================
print("1. LinearArrayTransducer")
linear = LinearArrayTransducer(
    n_elements=128,
    element_width_mm=0.6,
    element_height_mm=14.0,
    kerf_mm=0.1,
    no_sub_x=2,
    no_sub_y=10,
    frequency_Hz=FC_HZ,
    elevation_focus_mm=80,
)
linear.compute_delays(focus_mm=[0, 0, 30])
linear.compute_apodization(focus_mm=[0, 0, 30], FoverD=2.0)
_save_transducer(linear, OUT / "transducer_linear.png")
_save_focal_law(linear, OUT / "transducer_linear_focal_law.png")
del linear
gc.collect()


# ===========================================================================
# 2. Matrix array
# ===========================================================================
print("2. MatrixArrayTransducer")
matrix = MatrixArrayTransducer(
    n_elements_x=32,
    n_elements_y=32,
    element_width_mm=0.3,
    element_height_mm=0.3,
    kerf_x_mm=0.05,
    kerf_y_mm=0.05,
    no_sub_x=2,
    no_sub_y=2,
    frequency_Hz=FC_HZ,
)
matrix.compute_delays(focus_mm=[-2, 0, 4])
matrix.compute_apodization(focus_mm=[-2, 0, 4], FoverD=1)
_save_transducer(matrix, OUT / "transducer_matrix.png")
_save_focal_law(matrix, OUT / "transducer_matrix_focal_law.png")
del matrix
gc.collect()


# ===========================================================================
# 3a. Convex array — flat elevation
# ===========================================================================
print("3a. ConvexArrayTransducer")
convex = ConvexArrayTransducer(
    n_elements=128,
    element_width_mm=0.6,
    element_height_mm=14.0,
    kerf_mm=0.1,
    radius_of_curvature_mm=60.0,
    no_sub_x=2,
    no_sub_y=4,
    frequency_Hz=3.5e6,
)
convex.compute_delays(focus_mm=[0, 0, 30])
convex.compute_apodization(focus_mm=[0, 0, 30], FoverD=1.5)
_save_transducer(convex, OUT / "transducer_convex.png")
_save_focal_law(convex, OUT / "transducer_convex_focal_law.png")
del convex
gc.collect()


# ===========================================================================
# 3b. Convex array — with elevation focus
# ===========================================================================
print("3b. ConvexArrayTransducer (elevation focused)")
convex_f = ConvexArrayTransducer(
    n_elements=128,
    element_width_mm=0.6,
    element_height_mm=14.0,
    kerf_mm=0.1,
    radius_of_curvature_mm=60.0,
    no_sub_x=2,
    no_sub_y=4,
    frequency_Hz=3.5e6,
    elevation_focus_mm=20,
)
convex_f.compute_delays(focus_mm=[0, 0, 30])
convex_f.compute_apodization(focus_mm=[0, 0, 30], FoverD=1.5)
_save_transducer(convex_f, OUT / "transducer_convex_focused.png")
_save_focal_law(convex_f, OUT / "transducer_convex_focused_focal_law.png")
del convex_f
gc.collect()


# ===========================================================================
# 4. Flat circular piston
# ===========================================================================
print("4. FlatCircularTransducer")
flat = FlatCircularTransducer(diameter_mm=25.0, no_sub=30, frequency_Hz=FC_HZ)
_save_transducer(flat, OUT / "transducer_flat_circular.png")
del flat
gc.collect()


# ===========================================================================
# 5. Concave bowl (HIFU / TUS)
# ===========================================================================
print("5. ConcaveCircularTransducer")
bowl = ConcaveCircularTransducer(
    diameter_mm=40.0, radius_of_curvature_mm=60.0, no_sub=30, frequency_Hz=0.5e6
)
_save_transducer(bowl, OUT / "transducer_concave_circular.png")
del bowl
gc.collect()


# ===========================================================================
# 6. Focused circular (cylindrical, line focus)
# ===========================================================================
print("6. FocusedCircularTransducer")
cyl = FocusedCircularTransducer(
    diameter_mm=20.0,
    radius_of_curvature_mm=40.0,
    no_sub=20,
    focus_axis="y",
    frequency_Hz=FC_HZ,
)
_save_transducer(cyl, OUT / "transducer_focused_circular.png")
del cyl
gc.collect()

# ===========================================================================
# 7. Convex circular (diffusive to simulate lenses)
# ===========================================================================
print("7. ConvexCircularTransducer")
conv = ConvexCircularTransducer(
    diameter_mm=30.0,
    radius_of_curvature_mm=15.0,
    no_sub=30,
    frequency_Hz=1.5e6,
    border_refine=3,
    patch_fill=1,
    filled_radius_with_big_patches=0.8,
)
_save_transducer(conv, OUT / "transducer_convex_circular.png")
del conv
gc.collect()


# ===========================================================================
# 8. Custom — TUS helmet
# ===========================================================================
print("8. CustomTransducer (helmet)")
bowl_elem = ConcaveCircularTransducer(
    diameter_mm=20.0, radius_of_curvature_mm=40.0, no_sub=20, frequency_Hz=0.5e6
)
R_mm = 50.0
n = 6
theta = np.deg2rad(45.0)
phi = np.linspace(0, 2 * np.pi, n, endpoint=False)
pos = R_mm * np.column_stack(
    [
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta) * np.ones(n),
    ]
)
norms = -pos / np.linalg.norm(pos, axis=1, keepdims=True)
helmet = CustomTransducer(
    elements=[bowl_elem] * n,
    positions_mm=pos,
    normals=norms,
    frequency_Hz=0.5e6,
)
helmet.compute_delays(focus_mm=[0.0, 0.0, 0.0])
_save_transducer(helmet, OUT / "transducer_helmet.png", scalars="Delays")
del helmet, bowl_elem, pos, norms, phi, theta, R_mm, n
gc.collect()

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
    patch_fill=0.5,
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

print("\nDone — all images saved to", OUT.resolve())
