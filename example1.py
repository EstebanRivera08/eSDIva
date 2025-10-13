import numpy as np

import pyfield.transducers as Transducers
from pyfield.brain_atlas import BG_Atlas
from pyfield.dopplerscan import DopplerScan
from pyfield.psimulation import PyField, TorchField
from pyfield.utilities import (
    add_2D_image,
    add_3D_vol,
    add_pressure_vol,
    add_regions_mesh,
    add_transducer_mesh,
    align_transducer_to_probe,
    create_vol_mesh,
)

# ----------------- Get the scan objects --------------------

# Define the paths to the scan files and BPS file
# Make sure to change the paths according to your file structure

MAIN_FOLDER_PATH = r".\src\pyfield\datatype\Silvia"

bps_PATH = MAIN_FOLDER_PATH + r"\3Dscan_angio3D.source.bps"
file_scan_3D_PATH = MAIN_FOLDER_PATH + r"\3Dscan_angio3D.source.scan"
file_scan_2D_PATH = MAIN_FOLDER_PATH + r"\2Dscan.source.scan"

save_fig = True  # Set to True to save the figures
fig_folder = r"tutorials\Draft/Figures/"  # Folder to save the figures

# Create the scan objects
Doppler3D = DopplerScan(scan_PATH=file_scan_3D_PATH, bps_PATH=bps_PATH)
# Doppler3D.show()

Doppler2D = DopplerScan(scan_PATH=file_scan_2D_PATH)
# Doppler2D.show(interpolation = 'bilinear')

# ----------------- Get the atlas object --------------------
atlas_name = "allen_mouse_25um"

region_names = "root"

Brain_Atlas = BG_Atlas(atlas_name, region_names=region_names)

# ------------------- Get the transducer object --------------------
domino = Transducers.Domino()

# Focalization spot
focus_mm = np.array([-1, 0, 4.5])  # mm [x, y, z]

# Define apodization and delay law
delays = domino.compute_delays(focus_mm=focus_mm)
apodization = domino.compute_apodization(
    focus_mm=focus_mm, FoverD=1, apodization_type="rect"
)

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
x, y, z, pressure_field = Domino_field(field_info_mm)
# The first time you call the function from one script there is
# an additional deadtime in the simulation for compiling and organize
# the parallel computing
x, y, z, pressure_field = Domino_field(field_info_mm)

# ------------------ Transform the meshes --------------------

# Compute the pressure volume mesh
pressure_vol_mesh = create_vol_mesh(x, y, z, pressure_field, scalars="Pressure (PII)")

# Get the meshes and get them to the probe coordinate system
TX_mesh = domino.get_mesh()

# Transform bg_atlas to the Lab coordinate system
invertz = np.diag([1, 1, -1, 1])  # Invert z-axis for the transducer mesh
Brain_Atlas.reset_mesh()  # Reset the Brain Atlas mesh to the original mesh
Brain_Atlas.transform(T_matrix=invertz @ Doppler3D.BrainToLab, inplace=True)

# Transform from the lab to the probe coordinate system
LabToProbe = align_transducer_to_probe(
    TX_mesh, Doppler2D
)  # Get the transformation matrix from the lab to the probe coordinate system
Doppler3D.transform(
    T_matrix=LabToProbe, inplace=True
)  # Transform the scan mesh to the probe coordinate system
Doppler2D.transform(
    T_matrix=LabToProbe, inplace=True
)  # Transform the scan mesh to the probe coordinate system
Brain_Atlas.transform(
    T_matrix=LabToProbe, inplace=True
)  # Transform the atlas mesh to the probe coordinate system

# ------------------ Code for plotting --------------------
scale = 1
off_screen = False
if save_fig:
    off_screen = True
    scale = 3
final_plotter = add_regions_mesh(
    Brain_Atlas.pv_mesh,
    notebook=False,
    window_size=[800 * scale, 600 * scale],
    off_screen=off_screen,
    kwargs_dict={
        region_names[0]: {"color": "lightgray", "opacity": 0.1},
        region_names[1]: {"color": "permanentgreen", "opacity": 0.2},
        region_names[2]: {"color": "cadmiumlemon", "opacity": 0.2},
    },
    label="Brain Atlas",
)

final_plotter = add_3D_vol(
    Doppler3D.pv_mesh,
    plotter=final_plotter,
    cmap="hot",
    opacity="sigmoid",
    opacity_unit_distance=1,
    scalar_bar_args={
        "title": "3D Doppler (dB)",
        "title_font_size": 16 * scale,
        "label_font_size": 12 * scale,
        "color": "white",
        "vertical": False,
        "position_x": 0.6,
        "position_y": 0.1,
        "width": 0.3,
    },
    ambient=1,
)

final_plotter = add_2D_image(
    Doppler2D.pv_mesh,
    plotter=final_plotter,
    cmap="gray",
    opacity=1.0,
    show_scalar_bar=True,
    lighting=True,
    ambient=1,
    scalar_bar_args={
        "title": "2D Doppler (dB)",
        "title_font_size": 16 * scale,
        "label_font_size": 12 * scale,
        "color": "white",
        "vertical": False,
        "position_x": 0.2,
        "position_y": 0.1,
        "width": 0.3,
    },
)

final_plotter = add_transducer_mesh(
    TX_mesh,
    plotter=final_plotter,
    show_edges=False,
    lighting=True,
    ambient=1,
    scalar_bar_args={
        "title": "Apodization",
        "title_font_size": 16 * scale,
        "label_font_size": 12 * scale,
        "vertical": True,
        "position_x": 0.85,
        "position_y": 0.6,
        "height": 0.3,
        "color": "white",
    },
)

final_plotter = add_pressure_vol(
    pressure_vol_mesh,
    plotter=final_plotter,
    plot_focal_spot=False,
    lighting=True,
    scalar_bar_args={
        "title": "Pressure",
        "title_font_size": 16 * scale,
        "label_font_size": 12 * scale,
        "vertical": True,
        "position_x": 0.85,
        "position_y": 0.2,
        "height": 0.3,
        "color": "white",
    },
)

final_plotter.set_background("black")
final_plotter.show_grid(
    grid="back",
    color="white",
    font_size=12 * scale,
    location="outer",
    xtitle="X (mm)",
    ytitle="Y (mm)",
    ztitle="Z (mm)",
    n_xlabels=3,
    n_ylabels=5,
    n_zlabels=6,
    use_3d_text=False,
)  # Show grid with white color and font size 10

final_plotter.add_axes(label_size=(0.1, 0.1), color="white")
final_plotter.camera_position = [
    (-35.82464339746701, -19.934674593533888, 6.906725165821046),
    (-2.316449860794587, -1.1753275901078455, 8.498045099318546),
    (0.03252715920946665, 0.026652630106333085, -0.9991154193696428),
]  # Set the camera position
final_plotter.camera.up = (0, 0, -1)  # Set the camera up direction

if save_fig:
    final_plotter.screenshot(fig_folder + "mouse_brain_doppler_figure.png")
else:
    final_plotter.show()  # Show the plotter in Jupyter Notebook

    final_plotter.close()  # for plotters
print(final_plotter.camera_position)

# ------------------ Clean up --------------------

Doppler3D.clean()  # for scans
Doppler2D.clean()  # for scans
domino.clean()  # for transducers
Brain_Atlas.clean()  # for atlas
# Domino_field.clean()  # for fields
del TX_mesh, pressure_vol_mesh  # for mesh objects
del final_plotter  # for plotters
