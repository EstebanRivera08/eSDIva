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

# ===========================================================================
# Subdivision utility — ellipsoidal cap example figure
# ===========================================================================
print("\n--- Ellipsoidal cap subdivision figure ---")

a, b, c_ax = 30e-3, 20e-3, 15e-3  # semi-axes in metres
R_ap = 15e-3  # circular aperture radius


def ellipsoid_cap(x, y):
    arg = max(1.0 - (x / a) ** 2 - (y / b) ** 2, 0.0)
    return np.array([x, y, c_ax * np.sqrt(arg)])


# Parameter guide for this surface
# ----------------------------------
# This ellipsoidal cap has strong curvature (the metric ||dr/du|| exceeds the
# curvature_threshold), so the function switches to *high-curvature mode*:
# an arc-length adapted grid is built so that patch centres are uniformly
# spaced on the surface, and patch sizes are scaled by `patch_fill`.
#
# patch_fill controls physical overlap of flat patches on the curved surface.
#   patch_fill = 1.0  Each patch extends to half the arc-length spacing in
#                     every direction.  Centres are arc-spaced so patches
#                     "touch" along the arc, but because the surface curves
#                     between adjacent centres the *flat* rectangles tilt
#                     relative to each other and their corners physically
#                     protrude into the neighbouring patch — visible overlap.
#   patch_fill = 0.5  Each patch covers only half the arc-length cell.  The
#                     flat rectangles stay within their own "lane" on the
#                     curved surface: no physical intersection, uniform gaps.
#                     Coverage ≈ patch_fill² ≈ 25 % — intentional for this
#                     strongly curved surface at this coarse resolution.
#
# For clinical transducers (large radius-of-curvature / small aperture) the
# curvature measure stays below the threshold, so the function uses
# *low-curvature mode* (uniform grid, full arc-length) and patch_fill has no
# effect.  The right patch_fill for high-curvature surfaces depends on local
# curvature and grid resolution: coarser grids (smaller n_u/n_v) need a
# smaller patch_fill because each patch subtends a larger angle.
#
# max_patch_scale rejects patches where the local arc-length amplification
# exceeds this factor.  1.5 here is conservative — patches steeper than
# 1.5× the nominal cell size are discarded, leaving intentional holes rather
# than large, poorly-approximated flat rectangles near extreme rims.
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
