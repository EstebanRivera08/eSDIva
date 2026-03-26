"""
Example 6: Transducer Gallery

Demonstrates all transducer types available in pyfield.transducers, showing
their geometry (PyVista mesh) and, where applicable, their apodization and
delay patterns.

Transducers covered
-------------------
1. LinearArrayTransducer  — 1-D row of rectangular elements, elevation lens
2. MatrixArrayTransducer  — 2-D grid of rectangular elements
3. FlatCircularTransducer     — flat piston disc (mono-element)
4. ConcaveCircularTransducer  — spherically curved bowl (TUS / HIFU)
5. FocusedCircularTransducer  — circular-disk aperture with single-axis curvature
6. CustomTransducer       — helmet assembled from several bowl elements

Run with:
    uv run example6_transducer_gallery.py
"""

import numpy as np

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

# ---------------------------------------------------------------------------
# Common parameters
# ---------------------------------------------------------------------------
FC_HZ = 1.5e6  # centre frequency (1.5 MHz)
FOCUS_MM = [0, 0, 50]  # electronic focus for array transducers

print("\n" + "=" * 60)
print("  PyField Transducer Gallery")
print("=" * 60 + "\n")
print(
    "Available transucers in pyfield.transducers:",
    pyfield.transducers.available_transducers(),
)

# ===========================================================================
# 1. Linear array
# ===========================================================================
print("--- 1. LinearArrayTransducer ---")
linear = LinearArrayTransducer(
    n_elements=64,
    element_width_mm=0.25,
    element_height_mm=12.0,
    kerf_mm=0.05,
    no_sub_x=2,
    no_sub_y=4,
    frequency_Hz=FC_HZ,
    elevation_focus_mm=60.0,  # fixed elevation focus (lens) at 60 mm depth
)
linear.compute_delays(focus_mm=FOCUS_MM)
linear.compute_apodization(focus_mm=FOCUS_MM, FoverD=2.0)
linear.plot_delays_apodization()
linear.show(scalars="Apodization")

# ===========================================================================
# 2. Matrix array
# ===========================================================================
print("\n--- 2. MatrixArrayTransducer ---")
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
matrix.plot_delays_apodization()
matrix.show(scalars="Apodization")

# ===========================================================================
# 2b. Convex (curvilinear) array — abdominal / obstetric probe geometry
# ===========================================================================
print("\n--- 2b. ConvexArrayTransducer ---")
convex = ConvexArrayTransducer(
    n_elements=128,
    element_width_mm=2,
    element_height_mm=12.0,
    kerf_mm=0.1,
    radius_of_curvature_mm=60.0,  # typical abdominal probe radius
    no_sub_x=2,
    no_sub_y=10,
    frequency_Hz=3.5e6,
)
convex.compute_delays(focus_mm=[0, 0, 60])
convex.compute_apodization(focus_mm=[0, 0, 60], FoverD=1.5)
convex.plot_delays_apodization()
convex.show(scalars="Apodization")


# ===========================================================================
# 2c. Convex (curvilinear) array WITH FOCUSING (same as xdc_focused_convex)
# ===========================================================================
print("\n--- 2c. ConvexArrayTransducer (with elevation focus)---")
convex = ConvexArrayTransducer(
    n_elements=32,
    element_width_mm=1.9,
    element_height_mm=10,
    kerf_mm=1.9 / 2,
    radius_of_curvature_mm=30,  # typical abdominal probe radius
    no_sub_x=2,
    no_sub_y=10,
    elevation_focus_mm=5,  # fixed elevation focus (lens) at 60 mm depth
    frequency_Hz=3.5e6,
)
convex.compute_delays(focus_mm=[0, 0, 60])
convex.compute_apodization(focus_mm=[0, 0, 60], FoverD=1.5)
convex.plot_delays_apodization()
convex.show(scalars="Apodization")

# ===========================================================================
# 3. Circular flat piston
# ===========================================================================
print("\n--- 3. FlatCircularTransducer ---")
circ = FlatCircularTransducer(
    diameter_mm=25.0,
    no_sub=30,
    frequency_Hz=FC_HZ,
)
circ.show()

# ===========================================================================
# 4. Focused bowl (spherical, HIFU / TUS)
# ===========================================================================
print("\n--- 4. ConcaveCircularTransducer ---")
bowl = ConcaveCircularTransducer(
    diameter_mm=40.0,
    radius_of_curvature_mm=60.0,  # geometric focus at 60 mm depth
    no_sub=30,
    frequency_Hz=0.5e6,
)
bowl.show()

# ===========================================================================
# 5. Cylindrical transducer (line focus)
# ===========================================================================
print("\n--- 5. FocusedCircularTransducer ---")
cyl = FocusedCircularTransducer(
    diameter_mm=20.0,
    radius_of_curvature_mm=40.0,
    no_sub=20,
    focus_axis="y",  # curvature along elevation axis
    frequency_Hz=FC_HZ,
)
cyl.show()

# ===========================================================================
# 6. CustomTransducer — mini TUS helmet (6 bowl elements on a hemisphere)
# ===========================================================================
print("\n--- 6. CustomTransducer (TUS helmet, 6 elements) ---")

# Build a single bowl element prototype
bowl_elem = ConcaveCircularTransducer(
    diameter_mm=20.0,
    radius_of_curvature_mm=40.0,
    no_sub=20,
    frequency_Hz=0.5e6,
)

# Place 6 elements on a hemisphere of radius 50 mm pointing toward the origin
R_helmet_mm = 50.0
n_elem = 6
# Evenly spaced azimuth angles; elevation fixed at 45 deg from the top
theta = np.deg2rad(45.0)
phi = np.linspace(0, 2 * np.pi, n_elem, endpoint=False)

positions_mm = R_helmet_mm * np.column_stack(
    [
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta) * np.ones(n_elem),
    ]
)
# Normals point from each element toward the origin
normals = -positions_mm / np.linalg.norm(positions_mm, axis=1, keepdims=True)

helmet = CustomTransducer(
    elements=[bowl_elem] * n_elem,
    positions_mm=positions_mm,
    normals=normals,
    frequency_Hz=0.5e6,
)

# Focus all elements electronically at the origin
helmet.compute_delays(focus_mm=[0.0, 0.0, 0.0])
helmet.show(scalars="Delays")

print("\nGallery complete.")
