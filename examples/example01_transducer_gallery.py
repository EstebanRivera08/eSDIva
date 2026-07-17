"""
Example 1: Transducer Gallery

Demonstrates all transducer types available in pyfield.transducers, showing
their geometry (PyVista mesh) and, where applicable, their apodization and
delay patterns.

Transducers covered
-------------------
1. LinearArrayTransducer  — 1-D row of rectangular elements, elevation lens
2. ConvexArrayTransducer  — curvilinear row of elements
3. MatrixArrayTransducer  — 2-D grid of rectangular elements
4. FlatCircularTransducer     — flat piston disc (mono-element)
5. ConcaveCircularTransducer  — spherically curved bowl (TUS / HIFU)
6. FocusedCircularTransducer  — circular-disk aperture with single-axis curvature
7. CustomTransducer       — helmet assembled from several bowl elements

Steps
-----
1. Create each transducer with representative parameters
2. Compute delays and apodization for multi-element types
3. Show 3D geometry via PyVista

Run with:
    uv run examples/example01_transducer_gallery.py
"""

import numpy as np
import pyvista as pv

# ============================================================================
# CONFIGURATION
# ============================================================================
from config import FIG_FOLDER, SAVE_FIG, SCALE

import pyfield
from pyfield.transducers import (
    ConcaveCircularTransducer,
    ConvexArrayTransducer,
    CustomTransducer,
    FlatCircularTransducer,
    FocusedCircularTransducer,
    LinearArrayTransducer,
    MatrixArrayTransducer,
)

# Physics / transducer constants
FC_HZ = 1.5e6  # Centre frequency (1.5 MHz)
FOCUS_MM = [0, 0, 50]  # Electronic focus for array transducers
WIN_W, WIN_H = 800, 600  # Base window size


def _plot_beamforming(tx):
    """Delay/apodization curves — interactive sessions only (windows pile up in batch)."""
    if not SAVE_FIG:
        tx.plot_delays_apodization()


def _show_or_save(tx, filename, scalars="Apodization"):
    """Show transducer interactively or save screenshot."""
    if SAVE_FIG:
        FIG_FOLDER.mkdir(exist_ok=True)
        plotter = pv.Plotter(
            window_size=(WIN_W * SCALE, WIN_H * SCALE), off_screen=True
        )
        scale = SCALE
    else:
        plotter = pv.Plotter(window_size=(WIN_W, WIN_H))
        scale = 1

    mesh = tx.get_mesh()
    if scalars == "Apodization":
        cmap, title, clim = "cool", "Apodization", [0, 1]
    else:
        cmap, title, clim = "rainbow", "Delays (s)", None

    plotter.add_mesh(
        mesh,
        scalars=scalars,
        cmap=cmap,
        clim=clim,
        show_scalar_bar=True,
        scalar_bar_args={
            "title": title,
            "vertical": True,
            "title_font_size": 12 * scale,
            "label_font_size": 10 * scale,
        },
        show_edges=True,
        ambient=1,
    )
    plotter.add_axes()
    plotter.show_grid(
        font_size=10 * scale,
        xtitle="X (mm)",
        ytitle="Y (mm)",
        ztitle="Z (mm)",
    )

    if SAVE_FIG:
        plotter.screenshot(str(FIG_FOLDER / filename))
        plotter.close()
    else:
        plotter.show()


print("\n" + "=" * 60)
print("  PyField Transducer Gallery")
print("=" * 60 + "\n")
print(
    "Available transducers in pyfield.transducers:",
    pyfield.transducers.available_transducers(),
)

# ============================================================================
# STEP 1: LINEAR ARRAY
# ============================================================================
print("--- 1. LinearArrayTransducer ---")
linear = LinearArrayTransducer(
    n_elements=64,
    element_width_mm=1,
    element_height_mm=12.0,
    kerf_mm=0.05,
    no_sub_x=1,
    no_sub_y=10,
    frequency_Hz=FC_HZ,
    elevation_focus_mm=60.0,
)
linear.compute_delays(focus_mm=FOCUS_MM)
linear.compute_apodization(focus_mm=FOCUS_MM, FoverD=2.0)
_plot_beamforming(linear)
_show_or_save(linear, "ex01_gallery_linear.png")

# ============================================================================
# STEP 2: CONVEX (CURVILINEAR) ARRAY
# ============================================================================
print("\n--- 2. ConvexArrayTransducer ---")
convex = ConvexArrayTransducer(
    n_elements=128,
    element_width_mm=0.5,
    element_height_mm=10.0,
    kerf_mm=0.1,
    radius_of_curvature_mm=60.0,
    no_sub_x=1,
    no_sub_y=10,
    frequency_Hz=3.5e6,
)
convex.compute_delays(focus_mm=[0, 0, 60])
convex.compute_apodization(focus_mm=[0, 0, 60], FoverD=1.5)
_plot_beamforming(convex)
_show_or_save(convex, "ex01_gallery_convex.png")

# ============================================================================
# STEP 3: CONVEX ARRAY WITH ELEVATION FOCUS
# ============================================================================
print("\n--- 3. ConvexArrayTransducer (with elevation focus) ---")
convex_focused = ConvexArrayTransducer(
    n_elements=32,
    element_width_mm=1.9,
    element_height_mm=10,
    kerf_mm=1.9 / 2,
    radius_of_curvature_mm=30,
    no_sub_x=2,
    no_sub_y=10,
    elevation_focus_mm=5,
    frequency_Hz=3.5e6,
)
convex_focused.compute_delays(focus_mm=[0, 0, 60])
convex_focused.compute_apodization(focus_mm=[0, 0, 60], FoverD=1.5)
_plot_beamforming(convex_focused)
_show_or_save(convex_focused, "ex01_gallery_convex_focused.png")

# ============================================================================
# STEP 4: MATRIX ARRAY
# ============================================================================
print("\n--- 4. MatrixArrayTransducer ---")
matrix = MatrixArrayTransducer(
    n_elements_x=16,
    n_elements_y=16,
    element_width_mm=0.3,
    element_height_mm=0.3,
    kerf_x_mm=0.05,
    kerf_y_mm=0.05,
    no_sub_x=2,
    no_sub_y=2,
    frequency_Hz=FC_HZ,
)
matrix.compute_delays(focus_mm=FOCUS_MM)
matrix.compute_apodization(focus_mm=FOCUS_MM, FoverD=2.0)
_plot_beamforming(matrix)
_show_or_save(matrix, "ex01_gallery_matrix.png")

# ============================================================================
# STEP 5: CIRCULAR FLAT PISTON
# ============================================================================
print("\n--- 5. FlatCircularTransducer ---")
circ = FlatCircularTransducer(
    diameter_mm=25.0,
    no_sub_diameter=30,
    frequency_Hz=FC_HZ,
)
_show_or_save(circ, "ex01_gallery_flat_circular.png")

# ============================================================================
# STEP 6: CONCAVE BOWL (HIFU / TUS)
# ============================================================================
print("\n--- 6. ConcaveCircularTransducer ---")
bowl = ConcaveCircularTransducer(
    diameter_mm=40.0,
    focus_mm=60.0,
    no_sub_diameter=30,
    frequency_Hz=0.5e6,
)
_show_or_save(bowl, "ex01_gallery_concave.png")

# ============================================================================
# STEP 7: FOCUSED CIRCULAR (LINE FOCUS)
# ============================================================================
print("\n--- 7. FocusedCircularTransducer ---")
cyl = FocusedCircularTransducer(
    diameter_mm=20.0,
    focus_mm=40.0,
    no_sub_diameter=20,
    focus_axis="y",
    frequency_Hz=FC_HZ,
)
_show_or_save(cyl, "ex01_gallery_focused_circular.png")

# ============================================================================
# STEP 8: CUSTOM TRANSDUCER — TUS HELMET (6 BOWL ELEMENTS)
# ============================================================================
print("\n--- 8. CustomTransducer (TUS helmet, 6 elements) ---")

bowl_elem = ConcaveCircularTransducer(
    diameter_mm=20.0,
    focus_mm=40.0,
    no_sub_diameter=20,
    frequency_Hz=0.5e6,
)

# Place 6 elements on a hemisphere of radius 50 mm pointing toward the origin.
R_HELMET_MM = 50.0
N_ELEM = 6
theta = np.deg2rad(45.0)
phi = np.linspace(0, 2 * np.pi, N_ELEM, endpoint=False)

positions_mm = R_HELMET_MM * np.column_stack(
    [
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta) * np.ones(N_ELEM),
    ]
)
normals = -positions_mm / np.linalg.norm(positions_mm, axis=1, keepdims=True)

helmet = CustomTransducer(
    elements=[bowl_elem] * N_ELEM,
    positions_mm=positions_mm,
    normals=normals,
    frequency_Hz=0.5e6,
)
helmet.compute_delays(focus_mm=[0.0, 0.0, 0.0])
_show_or_save(helmet, "ex01_gallery_custom_helmet.png", scalars="Delays")

print("\nGallery complete.")
