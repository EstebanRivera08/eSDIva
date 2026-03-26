"""
Example 2: Rat Brain Zone Focusing

Demonstrates focused ultrasound simulation targeting specific anatomical
zones of a rat brain using a BrainGlobe atlas for anatomy and a linear
array transducer.  Shows:
  1. Atlas loading and brain region mesh extraction
  2. Focus computation toward a specific brain target
  3. Monochromatic pressure field simulation
  4. Joint visualisation of anatomy + pressure field

Run with:
    uv run example2_ratbrainzones_focus.py
"""

import numpy as np
import pyvista as pv

import pyfield.transducers as Transducers
from pyfield.brain_atlas import BG_Atlas
from pyfield.psimulation import PyField
from pyfield.utilities import (
    add_3D_vol,
    add_pressure_vol,
    add_regions_mesh,
    add_transducer_mesh,
    create_vol_mesh,
)

print("\n --- Example 2: Rat Brain Zone Focusing --- \n")

save_fig = False
fig_folder = r"./others/figures//"
fig_version = ""

theme = "dark"
add_pressure_sim = True

if theme == "dark":
    pv.set_plot_theme("dark")

    pv.global_theme.anti_aliasing = "ssaa"
    pv.global_theme.background = "black"
    color = "white"
    pv.global_theme.font.color = "white"
    ambient_tx = 0.1
    ambient_pr = 0.55
else:
    pv.set_plot_theme("default")
    pv.global_theme.anti_aliasing = "ssaa"
    color = "black"
    ambient_tx = 1
    ambient_pr = 0.5

# ----------------- Get the atlas object --------------------

print("\n --- Import Brain Atlas --- \n")

atlas_name = "whs_sd_rat_39um"

region_names = ("root", "M1", "S1-hl")


Brain_Atlas = BG_Atlas(atlas_name, region_names=region_names)

# The Brain_Atlas is on the whs space, so it is normalized by the lambda-bregma
# distance, and centered in a specific voxel.
# So we need to set our bregma-lambda distance for our animal
# For rat it is around 8mm, and in mouse is around 4mm


# ------------------- Get the transducer object --------------------
if add_pressure_sim:
    print("\n --- Import transducer --- \n")
    domino = Transducers.Domino()

    # Focalization spot
    focus_mm = np.array([-1, 0, 8])  # mm [x, y, z]

    # Define apodization and delay law
    delays = domino.compute_delays(focus_mm=focus_mm)
    apodization = domino.compute_apodization(
        focus_mm=focus_mm, FoverD=1, apodization_type="rect"
    )

    # Get the meshes and get them to the probe coordinate system
    TX_mesh = domino.get_mesh()

    # ------------------ Compute the pressue field --------------------
    # Use PyField to compute the pressure field
    print("\n --- Compute Pressure Field --- \n")
    Domino_field = PyField(domino)
    field_info_mm = {
        "x_extent": [-0.25 + focus_mm[0], 0.25 + focus_mm[0]],
        "y_extent": [-0.5 + focus_mm[1], 0.5 + focus_mm[1]],
        "z_extent": [-1 + focus_mm[2], 1 + focus_mm[2]],
        "dx": 0.0125,
        "dy": 0.025,
        "dz": 0.05,
    }
    x, y, z, pr = Domino_field(field_info_mm)

    # Compute the pressure volume mesh
    pressure_vol_mesh = create_vol_mesh(x, y, z, pr / pr.max(), scalars="Pressure")

# ------------------ Transform the meshes --------------------

# Transform bg_atlas to the Lab coordinate system
invertz = np.diag([1, 1, -1, 1])  # Invert z-axis for the transducer mesh

# Generic BrainToProbe transformation based on transducer XY-plane center
# and distance cortex to probe and the lambda-bregma distance

lambda_bregma_mm = 8  # mm
cortex2probe_mm = 4.5  # mm
x_translation = 2  # mm
y_translation = -2  # mm

# With the bregma-lambda distance scale the brain mesh
# BG_Atlas is normalized by a lambda-bregma reference.

scale2animalsize = np.eye(4, dtype=float)
scale2animalsize[0, 0] = lambda_bregma_mm
scale2animalsize[1, 1] = lambda_bregma_mm
scale2animalsize[2, 2] = lambda_bregma_mm

# Pick the upper bound z of the atlas (in atlas coordinates) and compute the
# translation in z so that the atlas cortex top sits cortex2probe_mm below the
# transducer top (TX_mesh.bounds[5]).
atlas_z_max = Brain_Atlas.pv_mesh["root"].bounds[
    5
]  # zmax in atlas units (before scaling)

# Optional xy-plane center translation

translate2xycenter = np.eye(4, dtype=float)
translate2xycenter[0, 3] = x_translation
translate2xycenter[1, 3] = y_translation
translate2xycenter[2, 3] = 0.0

# After scaling, atlas top will be atlas_z_max * scale_factor. We want:
# atlas_top_scaled + tz = tx_z_top - cortex2probe_mm  => tz = tx_z_top - cortex2probe_mm - atlas_top_scaled
translate2depth = np.eye(4, dtype=float)
translate2depth[2, 3] = -float(cortex2probe_mm) - atlas_z_max * lambda_bregma_mm

# create the final transformation matrix
# Order: scale -> translate in depth -> invert z for transducer frame

T_matrix = invertz @ translate2depth @ translate2xycenter @ scale2animalsize

# Apply transform to atlas (inplace)
Brain_Atlas.transform(T_matrix=T_matrix, inplace=True)


# ------------------ Code for plotting --------------------

scale = 1
off_screen = False

if save_fig:
    scale = 3
    off_screen = True

plotter = pv.Plotter(
    window_size=(int(800 * scale), int(600 * scale)), off_screen=off_screen
)
plotter = add_regions_mesh(
    Brain_Atlas.pv_mesh,
    plotter=plotter,
    kwargs_dict={
        region_names[0]: {"color": "lightgray", "opacity": 0.4},
        region_names[1]: {"color": "permanentgreen", "opacity": 0.3},
        region_names[2]: {"color": "cadmiumlemon", "opacity": 0.3},
    },
    label="Brain Atlas",
)
if add_pressure_sim:
    plotter = add_transducer_mesh(
        TX_mesh,
        plotter=plotter,
        show_edges=False,
        lighting=True,
        ambient=ambient_tx,
        scalars="Delays",
        scalar_bar_args={
            "title": "Delays (s)",
            "title_font_size": int(16 * scale),
            "label_font_size": int(12 * scale),
            "vertical": True,
            "position_x": 0.85,
            "position_y": 0.6,
            "height": 0.3,
            "color": color,
        },
    )

    plotter = add_pressure_vol(
        pressure_vol_mesh,
        plotter=plotter,
        plot_focal_spot=True,
        lighting=True,
        ambient=ambient_pr,
        scalar_bar_args={
            "title": "Pressure",
            "title_font_size": int(16 * scale),
            "label_font_size": int(12 * scale),
            "vertical": True,
            "position_x": 0.85,
            "position_y": 0.2,
            "height": 0.3,
            "color": color,
        },
    )

plotter.add_legend(loc="upper left")
plotter.show_grid(
    grid="back",
    color=color,
    font_size=int(12 * scale),
    location="outer",
    xtitle="X (mm)",
    ytitle="Y (mm)",
    ztitle="Z (mm)",
    n_xlabels=5,
    n_ylabels=7,
    n_zlabels=5,
    use_3d_text=False,
)
plotter.add_axes(color=color)
plotter.camera.up = (0, 0, -1)  # Set the camera up direction
if not add_pressure_sim:
    plotter.camera_position = [
        (20.614657986753393, 15.24621600858596, -21.08169089677548),
        (1.637153771588793, -3.3395843055651997, 12.319131317460506),
        (-0.5383012283264511, -0.568456131147254, -0.6221651023187735),
    ]
else:
    plotter.camera_position = [
        (-15.24900094616164, 35.67719428309785, -13.247200478173852),
        (2.3638032066875496, -6.75339371797477, 10.329108921636326),
        (0.19534195195135756, -0.41314237919923597, -0.8894688844009263),
    ]  # Set the camera position

if save_fig:
    fig_name = f"rat_brain_zones{fig_version}.png"
    print("\n saving figure...")
    plotter.screenshot(fig_folder + fig_name)

else:
    plotter.show()

# Print camera position while the plotter is still valid
# print(plotter.camera_position)

# Close the plotter and ensure PyVista releases resources before interpreter exit.
plotter.close()
try:
    pv.close_all()
except Exception:
    pass

# ------------------ Clean up --------------------


del TX_mesh, Brain_Atlas, pressure_vol_mesh, domino
# remove the last reference to the plotter so destructors run while pyvista is available
del plotter
