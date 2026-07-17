"""
Example 14: STL Mesh Loading and Visualisation

Demonstrates how to load and visualise STL files (e.g., experimental setup
components like petri dishes, chambers, or custom holders) alongside PyField
transducers.  Three sub-examples illustrate different STL operations:

  14a. Simple STL visualisation
  14b. Multiple STL objects with transformations (translate, rotate)
  14c. Custom lighting and material properties

Steps
-----
1. Load an STL file with ``load_mesh_from_stl`` / ``add_stl_mesh``
2. Apply geometric transformations (translation, rotation)
3. Combine multiple meshes in one scene
4. Explore rendering and lighting options

Run with:
    uv run examples/example14_importstl_petri_dish.py
"""

from pathlib import Path

import pyvista as pv

from pyfield.plotting import add_stl_mesh, load_mesh_from_stl

# ============================================================================
# CONFIGURATION
# ============================================================================
from config import FIG_FOLDER, SAVE_FIG

WIN_W, WIN_H = 800, 600

# STL file location
SCRIPT_DIR = Path(__file__).parent
STL_FILE = SCRIPT_DIR / "Petri_dish.stl"

if not STL_FILE.exists():
    raise FileNotFoundError(f"STL file not found: {STL_FILE}")

print(f"Loading STL file: {STL_FILE.name}")
print(f"File size: {STL_FILE.stat().st_size / 1024:.2f} KB\n")

if SAVE_FIG:
    FIG_FOLDER.mkdir(exist_ok=True)
    off_screen = True
else:
    off_screen = False

# ============================================================================
# STEP 14a: SIMPLE STL VISUALISATION
# ============================================================================
print("14a: Simple visualisation with default settings")

plotter1 = add_stl_mesh(
    STL_FILE, color="lightblue", opacity=0.8, show_edges=True, off_screen=off_screen
)
plotter1.camera_position = "iso"
plotter1.add_text("14a: Simple STL Visualisation", position="upper_edge")

if SAVE_FIG:
    plotter1.screenshot(str(FIG_FOLDER / "ex14_stl_simple.png"))
else:
    plotter1.show()

# ============================================================================
# STEP 14b: MULTIPLE STL OBJECTS WITH TRANSFORMATIONS IN ONE SCENE
# ============================================================================
print("\n14b: Multiple STL objects in one scene")

mesh1 = load_mesh_from_stl(STL_FILE)
mesh2 = load_mesh_from_stl(STL_FILE, translation=(20, 0, 0))
mesh3 = load_mesh_from_stl(
    STL_FILE,
    translation=(10, 15, 0),
    rotation_axis=(1, 0, 0),
    rotation_angle=90,
)

plotter2 = pv.Plotter(off_screen=off_screen)
plotter2 = add_stl_mesh(
    mesh1, plotter=plotter2, color="red", label="Original", opacity=0.7
)
plotter2 = add_stl_mesh(
    mesh2, plotter=plotter2, color="green", label="Translated", opacity=0.7
)
plotter2 = add_stl_mesh(
    mesh3, plotter=plotter2, color="blue", label="Rotated", opacity=0.7
)
plotter2.add_legend()
plotter2.camera_position = "iso"
plotter2.add_text("14b: Multiple STL Objects", position="upper_edge")

if SAVE_FIG:
    plotter2.screenshot(str(FIG_FOLDER / "ex14_stl_multiple.png"))
else:
    plotter2.show()

# ============================================================================
# STEP 14c: CUSTOM LIGHTING AND MATERIALS
# ============================================================================
print("\n14c: Custom lighting and material properties")

plotter3 = pv.Plotter(off_screen=off_screen)
plotter3.add_mesh(
    mesh1,
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
plotter3.add_light(pv.Light(position=(10, 10, 10), intensity=0.8))
plotter3.add_light(pv.Light(position=(-10, -10, 10), intensity=0.3))
plotter3.camera_position = "iso"
plotter3.add_text("14c: Custom Lighting & Materials", position="upper_edge")

if SAVE_FIG:
    plotter3.screenshot(str(FIG_FOLDER / "ex14_stl_lighting.png"))
else:
    plotter3.show()

del plotter1, plotter2, plotter3, mesh1, mesh2, mesh3

print("\n" + "=" * 70)
print("All sub-examples completed successfully!")
print("=" * 70)
