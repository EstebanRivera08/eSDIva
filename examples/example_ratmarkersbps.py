import h5py
import numpy as np

import pyfield.transducers as Transducers
from pyfield.cache import DopplerScan
from pyfield.psimulation import PyField
from pyfield.utilities import BG_Atlas, compute_affine_from_markers
from pyfield.plotting import (
    add_2D_image,
    add_3D_vol,
    add_markers,
    add_pressure_vol,
    add_regions_mesh,
    add_transducer_mesh,
    create_3Dvol_mesh,
)

# ----------------- Get the scan objects --------------------
print("\n --- Example 4: BPS with brain markers --- \n")

# Define the paths to the scan files and BPS file
# Make sure to change the paths according to your file structure

MAIN_FOLDER_PATH = r".\datatype\Felipe"

bps_PATH = MAIN_FOLDER_PATH + r"\3Dscan_angio3D.source.bps"
file_scan_3D_PATH = MAIN_FOLDER_PATH + r"\3Dscan_angio3D.source.scan"
file_scan_2D_PATH = MAIN_FOLDER_PATH + r"\2Dscan.source.scan"
markers_PATH = MAIN_FOLDER_PATH + r"\markers_coronal.mrk"
plot_tx_pr = False

# Create the scan objects
Doppler3D = DopplerScan(scan_PATH=file_scan_3D_PATH, bps_PATH=bps_PATH)
# Doppler3D.show()


Doppler2D = DopplerScan(scan_PATH=file_scan_2D_PATH)
# Doppler2D.show(interpolation = 'bilinear')

# ----------------- Get the atlas object --------------------
print("\n --- Import Brain Atlas --- \n")
# atlas_name = "allen_mouse_25um"
atlas_name = "whs_sd_rat_39um"

if atlas_name == "allen_mouse_25um":
    region_names = "root"
elif atlas_name == "whs_sd_rat_39um":
    region_names = ("root", "M1", "S1-hl")

Brain_Atlas = BG_Atlas(atlas_name, region_names=region_names)

# ------------------ Transform the meshes --------------------

# Transform bg_atlas to the Lab coordinate system
invertz = np.diag([1, 1, -1, 1])  # Invert z-axis for the transducer mesh
Brain_Atlas.reset_mesh()  # Reset the Brain Atlas mesh to the original mesh

Brain_Atlas.transform(T_matrix=invertz, inplace=True)

# 3) map the image-center to the *lab* coordinate system:
Doppler3D.transform(
    T_matrix=invertz @ np.linalg.inv(Doppler3D.BrainToLab) @ invertz, inplace=True
)  # Transform the scan mesh to the probe coordinate system
Doppler2D.transform(
    T_matrix=invertz @ np.linalg.inv(Doppler3D.BrainToLab) @ invertz, inplace=True
)  # Transform the scan mesh to the probe coordinate system

# -------- Translate the 2D Dopler scan to the marker plane coordinate system --------

# print("\n --- Import and Transform with brain markers --- \n")

# Load the markers file
file_scan = h5py.File(markers_PATH, "r")
point1 = np.array(file_scan["Plane"]["Plane_0"]["first"][()]).ravel()
point2 = np.array(file_scan["Plane"]["Plane_0"]["second"][()]).ravel()

# Let's invert the z axis to match the transducer coordinate system
point1[2] = -point1[2]
point2[2] = -point2[2]

center_plane, R = compute_affine_from_markers(
    point1, point2
)  # Compute the affine transformation matrix from the markers

center_2Ddoppler = np.array(Doppler2D.pv_mesh.center)

# NOTE: this data is from the special motor of Felipe, which does not have a translation
# along the x-axis. Thus the alingment is not guaranteed, and the translation should be just along y-axis.
# center_2Ddoppler[0] = 0
center_2Ddoppler[2] = 0

# assemble 4×4
t = center_plane - center_2Ddoppler  # Translation vector to the marker plane


T = np.eye(4)
T[:3, :3] = R
T[:3, 3] = t

Doppler2D.transform(
    T_matrix=T, inplace=True
)  # Transform the scan mesh to the marker plane coordinate system

if plot_tx_pr:
    print("\n --- Create Transducer and Compute Pressure Field --- \n")
    # ------------------- Get the transducer object --------------------
    domino = Transducers.Domino()

    # Focalization spot
    focus_mm = np.array([-1, 0, 4.5])  # mm [x, y, z]

    # Define apodization and delay law
    delays = domino.compute_delays(focus_mm=focus_mm)
    apodization = domino.compute_apodization(
        focus_mm=focus_mm, FoverD=1, apodization_type="rect"
    )

    # Get the meshes and get them to the probe coordinate system
    TX_mesh = domino.get_mesh()

    # ------------------ Compute the pressue field --------------------
    # Use PyField to compute the pressure field
    Domino_field = PyField(domino)
    field_info_mm = {
        "x_extent": [-0.25 + focus_mm[0], 0.25 + focus_mm[0]],
        "y_extent": [-0.5 + focus_mm[1], 0.5 + focus_mm[1]],
        "z_extent": [-1 + focus_mm[2], 1 + focus_mm[2]],
        "dx": 0.0125,
        "dy": 0.025,
        "dz": 0.05,
    }
    pressure_field, coords = Domino_field(field_info_mm)

    # # Compute the pressure volume mesh
    # Compute the pressure volume mesh
    pressure_vol_mesh = create_3Dvol_mesh(
        coords["x"], coords["y"], coords["z"], pressure_field,
        scalars="Pressure (PII)"
    )
# ------------------ Code for plotting --------------------

plotter = add_regions_mesh(
    Brain_Atlas.pv_mesh,
    notebook=False,
    window_size=[1000, 800],
    kwargs_dict={
        region_names[0]: {"color": "lightgray", "opacity": 0.4},
        region_names[1]: {"color": "vandykebrown", "opacity": 0.3},
        region_names[2]: {"color": "lawngreen", "opacity": 0.3},
    },
    label="Brain Atlas",
)

plotter = add_3D_vol(
    Doppler3D.pv_mesh,
    plotter=plotter,
    cmap="hot",
    opacity="sigmoid",
    opacity_unit_distance=1,
    scalar_bar_args={
        "title": "3D Doppler (dB)",
        "title_font_size": 16,
        "label_font_size": 12,
        "color": "white",
    },
)

plotter = add_2D_image(
    Doppler2D.pv_mesh,
    plotter=plotter,
    cmap="inferno",
    opacity=1.0,
    show_scalar_bar=True,
    lighting=True,
    ambient=1,
    scalar_bar_args={
        "title": "2D Doppler (dB)",
        "title_font_size": 16,
        "label_font_size": 12,
        "vertical": True,
        "position_x": 0.1,
        "position_y": 0.2,
        "height": 0.3,
        "color": "white",
    },
)

plotter = add_markers(
    [point1, point2],
    plotter=plotter,
    point_size=20,
    color="white",
    labels=["P1", "P2"],
    label_offset=(0, 0, 0.2),
    label_font_size=14,
    ambient=1,
)

if plot_tx_pr:
    plotter = add_transducer_mesh(
        TX_mesh,
        plotter=plotter,
        show_edges=False,
        lighting=True,
        ambient=1,
        scalar_bar_args={
            "title": "Apodization",
            "title_font_size": 16,
            "label_font_size": 12,
            "vertical": True,
            "position_x": 0.85,
            "position_y": 0.6,
            "height": 0.3,
            "color": "white",
        },
    )

    plotter = add_pressure_vol(
        pressure_vol_mesh,
        plotter=plotter,
        plot_focal_spot=False,
        lighting=True,
        ambient=1,
        scalar_bar_args={
            "title": "Pressure Field (PII)",
            "title_font_size": 16,
            "label_font_size": 12,
            "vertical": True,
            "position_x": 0.85,
            "position_y": 0.2,
            "height": 0.3,
            "color": "white",
        },
    )

plotter.set_background("black")
plotter.show_grid(
    color="white", font_size=10
)  # Show grid with white color and font size 10
plotter.add_axes(color="white")
plotter.camera_position = "zy"  # Set the camera position
plotter.camera.up = (0, 0, -1)  # Set the camera up direction
plotter.show()  # Show the plotter in Jupyter Notebook

plotter.close()  # for plotters

# ------------------ Clean up --------------------

Doppler3D.clean()  # for scans
Doppler2D.clean()  # for scans
Brain_Atlas.clean()  # for atlas

del plotter  # for plotters

if plot_tx_pr:
    del TX_mesh, pressure_vol_mesh  # for mesh objects
