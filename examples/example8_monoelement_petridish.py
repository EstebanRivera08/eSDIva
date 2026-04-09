"""
Integration example: STL mesh visualization with PyField transducer and pressure field.

This script demonstrates how to combine STL models (experimental setup) with
PyField acoustic simulations to visualize the complete experimental configuration.

Run with:
    uv run others/monoelement_setup/stl_with_simulation.py
"""

from pathlib import Path

import numpy as np
import pyvista as pv

from pyfield.psimulation import PyField
from pyfield.transducers import ConcaveCircularTransducer
from pyfield.utilities import (
    add_pressure_vol,
    add_stl_mesh,
    add_transducer_mesh,
    create_vol_mesh,
    load_stl_mesh,
    to_dB,
)

# Configuration
SCRIPT_DIR = Path(__file__).parent
STL_FILE = SCRIPT_DIR / "Petri_dish.stl"
THEME = "dark"  # or "light"

# Apply theme
if THEME == "dark":
    pv.set_plot_theme("dark")
    pv.global_theme.background = "black"
    pv.global_theme.font.color = "white"
    pv.global_theme.anti_aliasing = "ssaa"
else:
    pv.set_plot_theme("default")

print("=" * 70)
print("PyField + STL Mesh Visualization Example")
print("=" * 70)

# ============================================================================
# Step 1: Create a circular transducer
# ============================================================================
print("\n Creating transducer...")

# Create a flat circular transducer (typical for focused ultrasound)
transducer = ConcaveCircularTransducer(
    diameter_mm=0.5 * 25.4,  # 0.5 inch radius
    radius_of_curvature_mm=1 * 25.4,
    no_sub=20,
    frequency_Hz=5e6,  # 5 MHz
)
# transducer.show()

# ============================================================================
# Step 2: Define simulation field
# ============================================================================
print("\n Defining simulation field...")

# Define a field along the beam axis
field_points = {
    "x_extent": [-2, 2],  # mm
    "y_extent": [-2, 2],  # mm
    "z_extent": [18, 35],  # mm, starts at 5mm from transducer
    "dx": 0.1,
    "dy": 0.1,
    "dz": 0.2,
}

# ============================================================================
# Step 3: Run simulation
# ============================================================================
print("\n Running acoustic simulation...")

sim = PyField(transducer, monochromatic=True, verbose=False)
x, y, z, p = sim(field_points, method="auto")

# Convert to dB scale for better visualization
# p_db = to_dB(np.abs(p))
p_db = p / p.max()

# ============================================================================
# Step 4: Load and position STL mesh
# ============================================================================
print("\n Loading STL mesh...")

if STL_FILE.exists():
    # Load the petri dish STL
    # Adjust scale and position to match the simulation coordinate system
    petri_dish = load_stl_mesh(
        STL_FILE,
        scale=1,  # Adjust scale based on STL units
        translation=(0, -10, 25),  # Position at z=30mm (in the field)
        rotation_axis=(1, 0, 0),
        rotation_angle=-45,  # Flip if needed
    )
    print(f"  STL file loaded: {STL_FILE.name}")
    print(f"  Mesh bounds: {petri_dish.bounds}")
    has_stl = True
else:
    print(f"  Warning: STL file not found at {STL_FILE}")
    print(f"  Continuing without STL mesh...")
    has_stl = False

# ============================================================================
# Step 5: Create 3D visualization
# ============================================================================
print("\n Creating 3D visualization...")

# Create pressure volume mesh
pressure_vol = create_vol_mesh(x, y, z, p_db, scalars="Pressure")

# Create transducer mesh
tx_mesh = transducer.get_mesh()

# Create plotter
plotter = pv.Plotter(window_size=(500, 600))

# Add transducer
plotter = add_transducer_mesh(
    tx_mesh, plotter=plotter, scalars="Apodization", show_scalar_bar=False
)

# Add pressure field as isosurfaces
plotter = add_pressure_vol(
    pressure_vol,
    plotter=plotter,
    colorbar_title="Pressure",
    contour_levels=15,
    # vmin=-40,
    # vmax=0,
    vmin=0,
    vmax=1,
    scalar_bar_args={
        "title": "Pressure (a.u.)",
        "title_font_size": int(20),
        "label_font_size": int(18),
        "vertical": True,
        "position_x": 0.8,
        "position_y": 0.6,
        "height": 0.3,
    },
)

# Add STL mesh if available
if has_stl:
    plotter = add_stl_mesh(
        petri_dish,
        plotter=plotter,
        color="lightgray",
        opacity=0.3,
        show_edges=True,
        label="Petri Dish",
    )

# Configure camera and display
plotter.camera_position = "yz"
plotter.camera.up = (0, 0, -1)

# plotter.add_text("PyField Simulation + STL Setup", position="upper_edge", font_size=14)

# Optional: Add a legend if STL is present
# if has_stl:
#     plotter.add_legend(bcolor=(0.1, 0.1, 0.1) if THEME == "dark" else (0.9, 0.9, 0.9))

print("\nDisplaying visualization...")
print("  - Blue/Green mesh: Transducer (colored by apodization)")
print("  - Rainbow isosurfaces: Acoustic pressure field")
if has_stl:
    print("  - Gray mesh: Petri dish (experimental setup)")
print("\nClose the window to exit.")

plotter.show()

del sim, pressure_vol, tx_mesh, petri_dish, plotter
