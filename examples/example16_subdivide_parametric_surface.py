"""
Example 10: Parametric Surface Subdivision

Demonstrates ``pyfield.utilities.surface_subdivision.subdivide_parametric_surface``,
the public utility that tiles any C1 parametric surface with flat tangent-plane
rectangles for use by the SIR kernel.

The example uses an **ellipsoidal cap** — a surface with strong curvature
variation from centre to rim — to show how the arc-length adapted grid keeps
patch centres equidistant across the aperture.

Steps
-----
1. Define the ellipsoidal cap parametric function
2. Subdivide with default parameters
3. Print the ``frames`` dict summary
4. Visualise with Matplotlib: 3-D patch mosaic + normals, top-down area map
5. Visualise with PyVista: theoretical surface vs flat patch mosaic

Run with:
    uv run examples/example10_subdivide_parametric_surface.py
"""

import matplotlib

# matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

# ============================================================================
# CONFIGURATION
# ============================================================================
from config import FIG_FOLDER, SAVE_FIG, SCALE
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from pyfield.utilities.surface_subdivision import subdivide_parametric_surface

WIN_W, WIN_H = 500, 500
THEME = "dark"

if SAVE_FIG:
    pv.OFF_SCREEN = True
else:
    pv.OFF_SCREEN = False

if THEME == "dark":
    pv.set_plot_theme("dark")
    pv.global_theme.background = "black"
    pv.global_theme.font.color = "white"
    pv.global_theme.anti_aliasing = "ssaa"
else:
    pv.set_plot_theme("default")

print("=" * 70)
print("Parametric Surface Subdivision Example")
print("=" * 70)

if SAVE_FIG:
    FIG_FOLDER.mkdir(exist_ok=True)

# ============================================================================
# STEP 1: DEFINE ELLIPSOIDAL CAP
# ============================================================================
print("\nStep 1: Define ellipsoidal cap surface")
print("-" * 40)

# Ellipsoidal cap: z(x, y) = c * sqrt(1 - x²/a² - y²/b²)
a, b, c = 30e-3, 20e-3, 15e-3  # semi-axes in metres
R_ap = 15e-3  # aperture radius (circular mask)

print(f"  Semi-axes: a={a * 1e3:.0f} mm, b={b * 1e3:.0f} mm, c={c * 1e3:.0f} mm")
print(f"  Aperture radius: R_ap={R_ap * 1e3:.0f} mm")
print(f"  Peak height at centre: z(0,0) = {c * 1e3:.0f} mm")


def ellipsoid_cap(x, y):
    arg = max(1.0 - (x / a) ** 2 - (y / b) ** 2, 0.0)
    return np.array([x, y, c * np.sqrt(arg)])


# ============================================================================
# STEP 2: SUBDIVIDE THE SURFACE
# ============================================================================
print("\nStep 2: Subdivide with arc-length adapted grid")
print("-" * 40)

# This cap has strong curvature → high-curvature mode is triggered.
#
# patch_fill = 0.5: each patch fills only half the arc-length cell in
# each direction (coverage ≈ 25 %).  Using patch_fill = 1.0 would cause
# the flat patches to physically intersect in 3-D — the surface curves
# enough between adjacent centres that full-width flat rectangles
# protrude into their neighbours.  With a coarse grid (n_u = n_v = 10),
# 0.5 is the empirically safe value for this geometry; a finer grid
# would allow a higher patch_fill.
#
# max_patch_scale = 1.5: conservatively rejects the steepest rim cells.
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

# ============================================================================
# STEP 3: PRINT FRAMES SUMMARY
# ============================================================================
print("\nStep 3: frames dict summary")
print("-" * 40)

n_patches = frames["centers"].shape[0]
areas = frames["wu"] * frames["wv"]

print(f"  Patches accepted : {n_patches}")
print(f"  Patches rejected : {frames['n_rejected']}")
print(f"  Coverage         : {frames['coverage']:.1%}")
print(f"  Mean patch area  : {areas.mean() * 1e6:.3f} mm²")
print(f"  Area range       : {areas.min() * 1e6:.3f} – {areas.max() * 1e6:.3f} mm²")
print(f"\n  frames keys: {list(frames.keys())}")

# ============================================================================
# STEP 4: MATPLOTLIB — 3-D MOSAIC + TOP-DOWN AREA MAP
# ============================================================================
print("\nStep 4: Matplotlib visualisation")
print("-" * 40)

fig = plt.figure(figsize=(12, 5))
fig.patch.set_facecolor("#1a1a2e" if THEME == "dark" else "white")
text_color = "white" if THEME == "dark" else "black"
ax_bg = "#1a1a2e" if THEME == "dark" else "white"

# --- Left: 3-D patch mosaic with normals ---
ax3d = fig.add_subplot(121, projection="3d")
ax3d.set_facecolor(ax_bg)

corners_list = frames["corners"]  # list of (4,3) arrays
centers = frames["centers"]  # (M, 3)
normals = frames["normals"]  # (M, 3)
areas_plot = frames["wu"] * frames["wv"]

# Colour patches by area
vmin_a, vmax_a = areas_plot.min(), areas_plot.max()
cmap = plt.cm.plasma

polys = []
face_colors = []
for i, corn in enumerate(corners_list):
    polys.append(corn)
    t = (areas_plot[i] - vmin_a) / (vmax_a - vmin_a + 1e-30)
    face_colors.append(cmap(t))

poly_col = Poly3DCollection(polys, alpha=0.6, linewidths=0.4, edgecolors="royalblue")
poly_col.set_facecolor(face_colors)
ax3d.add_collection3d(poly_col)

# Outward normals (red arrows, scaled)
scale = 3e-3
ax3d.quiver(
    centers[:, 0],
    centers[:, 1],
    centers[:, 2],
    normals[:, 0] * scale,
    normals[:, 1] * scale,
    normals[:, 2] * scale,
    color="red",
    linewidth=0.5,
    arrow_length_ratio=0.3,
)

ax3d.set_xlabel("x (m)", color=text_color, fontsize=8)
ax3d.set_ylabel("y (m)", color=text_color, fontsize=8)
ax3d.set_zlabel("z (m)", color=text_color, fontsize=8)
ax3d.set_title("3-D patch mosaic + normals", color=text_color, fontsize=10)
ax3d.tick_params(colors=text_color, labelsize=7)
for pane in (ax3d.xaxis.pane, ax3d.yaxis.pane, ax3d.zaxis.pane):
    pane.set_alpha(0.1)

# --- Right: top-down area map ---
ax2d = fig.add_subplot(122)
ax2d.set_facecolor(ax_bg)

sc = ax2d.scatter(
    centers[:, 0] * 1e3,
    centers[:, 1] * 1e3,
    c=areas_plot * 1e6,
    cmap="plasma",
    s=40,
    edgecolors="none",
)
cb = fig.colorbar(sc, ax=ax2d, label="Patch area (mm²)")
cb.ax.yaxis.label.set_color(text_color)
cb.ax.tick_params(colors=text_color)

# Aperture circle reference
theta = np.linspace(0, 2 * np.pi, 200)
ax2d.plot(
    R_ap * 1e3 * np.cos(theta),
    R_ap * 1e3 * np.sin(theta),
    "w--" if THEME == "dark" else "k--",
    lw=1,
    label="Aperture",
)
ax2d.set_xlabel("x (mm)", color=text_color, fontsize=9)
ax2d.set_ylabel("y (mm)", color=text_color, fontsize=9)
ax2d.set_title(
    f"Top-down patch area map  (coverage {frames['coverage']:.0%})",
    color=text_color,
    fontsize=10,
)
ax2d.set_aspect("equal")
ax2d.tick_params(colors=text_color)
ax2d.legend(fontsize=8, labelcolor=text_color, facecolor=ax_bg, edgecolor=text_color)
for spine in ax2d.spines.values():
    spine.set_edgecolor(text_color)

fig.tight_layout()

if SAVE_FIG:
    out = FIG_FOLDER / "subdivision_ellipsoid_cap.png"
    fig.savefig(
        out, dpi=100 * SCALE, bbox_inches="tight", facecolor=fig.get_facecolor()
    )
    print(f"  Saved: {out}")
else:
    plt.show()

plt.close(fig)

# ============================================================================
# STEP 5: PYVISTA — THEORETICAL SURFACE vs FLAT PATCH MOSAIC
# ============================================================================
print("\nStep 5: PyVista visualisation — theoretical vs approximated")
print("-" * 40)

# Build theoretical ellipsoidal cap surface — valid aperture points only
u_vals = np.linspace(-R_ap, R_ap, 80)
v_vals = np.linspace(-R_ap, R_ap, 80)
U, V = np.meshgrid(u_vals, v_vals)

mask = (U / a) ** 2 + (V / b) ** 2 <= 1.0
Z_surf = c * np.sqrt(np.maximum(1 - (U / a) ** 2 - (V / b) ** 2, 0))

# Extract only valid points and triangulate in the XY plane (Delaunay 2D)
x_valid = U[mask]
y_valid = V[mask]
z_valid = Z_surf[mask]
cloud = pv.PolyData(np.column_stack([x_valid, y_valid, z_valid]))
grid = cloud.delaunay_2d()
grid["height_mm"] = grid.points[:, 2] * 1e3

# Build patch mosaic PolyData
all_pts = []
all_faces = []
patch_areas = []
offset = 0
for i, corn in enumerate(corners_list):
    for pt in corn:
        all_pts.append(pt)
    all_faces.extend([4, offset, offset + 1, offset + 2, offset + 3])
    patch_areas.append(frames["wu"][i] * frames["wv"][i] * 1e6)
    offset += 4

mosaic_mesh = pv.PolyData(
    np.array(all_pts, dtype=float),
    np.array(all_faces, dtype=int),
)
mosaic_mesh["area_mm2"] = np.repeat(patch_areas, 4)

pl = pv.Plotter(window_size=(WIN_W * SCALE, WIN_H * SCALE))
pl.add_mesh(
    mosaic_mesh,
    scalars="area_mm2",
    cmap="plasma",
    show_edges=True,
    edge_color="white",
    line_width=0.5,
    scalar_bar_args={"title": "Area (mm²)"},
)
pl.add_mesh(grid, opacity=0.25, color="cyan", show_scalar_bar=False)
pl.view_isometric()
pl.reset_camera()

if SAVE_FIG:
    out = FIG_FOLDER / "subdivision_ellipsoid_cap_pyvista.png"
    pl.screenshot(str(out))
    print(f"  Saved: {out}")
else:
    pl.show()

pl.close()

# ============================================================================
# STEP 6: COMPARE patch_fill VALUES
# ============================================================================
print("\nStep 6: Comparing patch_fill = 0.5, 0.75, 1.0")
print("-" * 40)

fill_values = [0.5, 0.8, 1.1]
results = {}

for pf in fill_values:
    f = subdivide_parametric_surface(
        ellipsoid_cap,
        u_range=(-R_ap, R_ap),
        v_range=(-R_ap, R_ap),
        n_u=10,
        n_v=10,
        inside_fn=lambda x, y: x**2 / a**2 + y**2 / b**2 <= 1.0,
        normal_sign=1.0,
        patch_fill=pf,
        max_patch_scale=1.5,
    )
    results[pf] = f
    print(
        f"  patch_fill={pf:.2f}: {f['centers'].shape[0]:3d} patches, "
        f"coverage={f['coverage']:.1%}, rejected={f['n_rejected']}"
    )

fig2, axes = plt.subplots(1, 3, figsize=(13, 4))
fig2.patch.set_facecolor("#1a1a2e" if THEME == "dark" else "white")
fig2.suptitle(
    "Effect of patch_fill on ellipsoidal cap subdivision", color=text_color, fontsize=11
)

for ax, pf in zip(axes, fill_values):
    ax.set_facecolor(ax_bg)
    f = results[pf]
    ax.scatter(
        f["centers"][:, 0] * 1e3,
        f["centers"][:, 1] * 1e3,
        c=(f["wu"] * f["wv"]) * 1e6,
        cmap="plasma",
        s=30,
        edgecolors="none",
    )
    ax.plot(
        R_ap * 1e3 * np.cos(theta),
        R_ap * 1e3 * np.sin(theta),
        "w--" if THEME == "dark" else "k--",
        lw=0.8,
    )
    ax.set_title(
        f"patch_fill={pf}  coverage={f['coverage']:.0%}",
        color=text_color,
        fontsize=9,
    )
    ax.set_aspect("equal")
    ax.tick_params(colors=text_color, labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor(text_color)
    ax.set_xlabel("x (mm)", color=text_color, fontsize=8)
    ax.set_ylabel("y (mm)", color=text_color, fontsize=8)

fig2.tight_layout()

if SAVE_FIG:
    out = FIG_FOLDER / "subdivision_patch_fill_comparison.png"
    fig2.savefig(
        out, dpi=100 * SCALE, bbox_inches="tight", facecolor=fig2.get_facecolor()
    )
    print(f"  Saved: {out}")
else:
    plt.show()

plt.close(fig2)

print("\n" + "=" * 70)
print("Done.")
print("=" * 70)

del (
    pl,
    mosaic_mesh,
    grid,
    cloud,
)
