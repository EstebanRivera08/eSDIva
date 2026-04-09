"""
Demonstration of STL mesh loading and visualization for PyField.

This script shows how to load and visualize STL files (e.g., experimental setup
components like petri dishes, chambers, or custom transducer holders) alongside
PyField transducers and pressure fields.

Run with:
    uv run others/monoelement_setup/visualize_stl.py
"""

from pathlib import Path

import numpy as np
import pyvista as pv

from pyfield.utilities import add_stl_mesh, load_stl_mesh

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent
STL_FILE = SCRIPT_DIR / "Petri_dish.stl"

# Check if STL file exists
if not STL_FILE.exists():
    raise FileNotFoundError(f"STL file not found: {STL_FILE}")

print(f"Loading STL file: {STL_FILE.name}")
print(f"File size: {STL_FILE.stat().st_size / 1024:.2f} KB\n")

# ============================================================================
# Example 1: Simple STL visualization
# ============================================================================
print("Example 1: Simple visualization with default settings")

plotter1 = add_stl_mesh(
    STL_FILE,
    color="lightblue",
    opacity=0.8,
    show_edges=True,
)

plotter1.camera_position = "iso"
plotter1.add_text("Example 1: Simple STL Visualization", position="upper_edge")
plotter1.show()


# ============================================================================
# Example 2: Load with transformations
# ============================================================================
print("\nExample 2: Load STL with transformations (scale, translate, rotate)")

# Load the mesh with transformations
mesh_transformed = load_stl_mesh(
    STL_FILE,
    scale=2.0,  # Scale up by 2x
    translation=(10, 5, 0),  # Shift in x, y, z
    rotation_axis=(0, 0, 1),  # Rotate around z-axis
    rotation_angle=45,  # 45 degrees
)

# Visualize the transformed mesh
plotter2 = add_stl_mesh(
    mesh_transformed,
    color="coral",
    opacity=0.9,
    show_edges=False,
    ambient=0.5,
)

plotter2.camera_position = "iso"
plotter2.add_text(
    "Example 2: Transformed STL (scaled 2x, translated, rotated 45°)",
    position="upper_edge",
)
plotter2.show()


# ============================================================================
# Example 3: Multiple STL objects in one scene
# ============================================================================
print("\nExample 3: Multiple STL objects in one scene")

# Load the original mesh
mesh1 = load_stl_mesh(STL_FILE)

# Load and position a second copy
mesh2 = load_stl_mesh(STL_FILE, translation=(20, 0, 0))

# Load and position a third copy with rotation
mesh3 = load_stl_mesh(
    STL_FILE, translation=(10, 15, 0), rotation_axis=(1, 0, 0), rotation_angle=90
)

# Create plotter and add all meshes
plotter3 = pv.Plotter()
plotter3 = add_stl_mesh(
    mesh1, plotter=plotter3, color="red", label="Original", opacity=0.7
)
plotter3 = add_stl_mesh(
    mesh2, plotter=plotter3, color="green", label="Translated", opacity=0.7
)
plotter3 = add_stl_mesh(
    mesh3, plotter=plotter3, color="blue", label="Rotated", opacity=0.7
)

plotter3.add_legend()
plotter3.camera_position = "iso"
plotter3.add_text("Example 3: Multiple STL Objects", position="upper_edge")
plotter3.show()


# ============================================================================
# Example 4: STL mesh analysis and properties
# ============================================================================
print("\nExample 4: Analyze STL mesh properties")

mesh = load_stl_mesh(STL_FILE)

print(f"Mesh Statistics:")
print(f"  Number of points: {mesh.n_points}")
print(f"  Number of cells: {mesh.n_cells}")
print(f"  Number of faces: {mesh.n_faces}")
print(f"  Bounds (x): [{mesh.bounds[0]:.2f}, {mesh.bounds[1]:.2f}]")
print(f"  Bounds (y): [{mesh.bounds[2]:.2f}, {mesh.bounds[3]:.2f}]")
print(f"  Bounds (z): [{mesh.bounds[4]:.2f}, {mesh.bounds[5]:.2f}]")
print(f"  Center: ({mesh.center[0]:.2f}, {mesh.center[1]:.2f}, {mesh.center[2]:.2f})")
print(f"  Volume: {mesh.volume:.4f} cubic units")
print(f"  Surface area: {mesh.area:.4f} square units")

# Visualize with different rendering options
plotter4 = pv.Plotter(shape=(1, 2))

# Left: Solid rendering
plotter4.subplot(0, 0)
plotter4.add_mesh(mesh, color="lightblue", show_edges=True)
plotter4.add_text("Solid Rendering", position="upper_edge", font_size=10)
plotter4.camera_position = "iso"

# Right: Wireframe rendering
plotter4.subplot(0, 1)
plotter4.add_mesh(mesh, style="wireframe", color="darkblue", line_width=2)
plotter4.add_text("Wireframe Rendering", position="upper_edge", font_size=10)
plotter4.camera_position = "iso"

plotter4.show()


# ============================================================================
# Example 5: STL with custom lighting and materials
# ============================================================================
print("\nExample 5: Custom lighting and material properties")

plotter5 = pv.Plotter()
plotter5.add_mesh(
    mesh,
    color="gold",
    metallic=0.8,
    roughness=0.2,
    show_edges=True,
    edge_color="darkgray",
    ambient=0.2,
    diffuse=0.8,
    specular=0.5,
    specular_power=30,
)

plotter5.add_light(pv.Light(position=(10, 10, 10), intensity=0.8))
plotter5.add_light(pv.Light(position=(-10, -10, 10), intensity=0.3))

plotter5.camera_position = "iso"
plotter5.add_text("Example 5: Custom Lighting & Materials", position="upper_edge")
plotter5.show()


print("\n" + "=" * 70)
print("All examples completed successfully!")
print("=" * 70)
print("\nUsage Tips:")
print("  - Use load_stl_mesh() to load and transform STL files")
print("  - Use add_stl_mesh() to add STL meshes to PyVista plotters")
print("  - Combine STL meshes with PyField transducers and pressure fields")
print("  - Adjust scale parameter if STL units don't match (e.g., mm vs m)")
print("  - Use translation to position experimental setup components")
print("  - Use rotation to orient components correctly")

del (
    plotter1,
    plotter2,
    plotter3,
    plotter4,
    plotter5,
)  # Clean up plotters to free resources
