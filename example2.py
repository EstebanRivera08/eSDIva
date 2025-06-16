import numpy as np
import pysonogen.transducers as Transducers
from pysonogen.atlas import BG_Atlas
from pysonogen.scans import DopplerScan
from pysonogen.pyfield import PyField
from pysonogen.functions import align_transducer_to_probe, compute_affine_from_markers
from pysonogen.functions import (add_regions_mesh, add_3D_vol, add_2D_image,
                                add_transducer_mesh, add_pressure_vol, add_markers)  
import h5py

# ----------------- Get the scan objects --------------------

# Define the paths to the scan files and BPS file
# Make sure to change the paths according to your file structure

MAIN_FOLDER_PATH = r".\src\pysonogen\datatype\Felipe"

bps_PATH = MAIN_FOLDER_PATH + r"\3Dscan_angio3D.source.bps"
file_scan_3D_PATH = MAIN_FOLDER_PATH + r"\3Dscan_angio3D.source.scan"
file_scan_2D_PATH = MAIN_FOLDER_PATH + r"\2Dscan.source.scan"
markers_PATH = MAIN_FOLDER_PATH + r"\markers_coronal.mrk"


# Create the scan objects
Doppler3D = DopplerScan(scan_PATH=file_scan_3D_PATH, bps_PATH = bps_PATH)
# Doppler3D.show()


Doppler2D = DopplerScan(scan_PATH=file_scan_2D_PATH)
# Doppler2D.show(interpolation = 'bilinear')

# ----------------- Get the atlas object --------------------
# atlas_name = "allen_mouse_25um"
atlas_name = "whs_sd_rat_39um"

if atlas_name == "allen_mouse_25um":
    region_names = ("root")
elif atlas_name == "whs_sd_rat_39um":
    region_names = ("root", "M1", "S1-hl")

Brain_Atlas = BG_Atlas(atlas_name, region_names = region_names)

# ------------------- Get the transducer object --------------------
domino = Transducers.Domino()

# Focalization spot
focus_mm =  np.array([-1, 0, 4.5])  # mm [x, y, z]

# Define apodization and delay law
delays = domino.compute_delays(focus_mm = focus_mm)
apodization = domino.compute_apodization(focus_mm = focus_mm, F_over_D = 1, apodization_type='rect')

# Get the meshes and get them to the probe coordinate system
# TX_mesh = domino.get_mesh()

# ------------------ Compute the pressue field --------------------
# Use PyField to compute the pressure field
Domino_field = PyField(domino)
field_info_mm = {
    "x_extent" : [-0.25+focus_mm[0], 0.25+focus_mm[0]],
    "y_extent" : [-0.5+focus_mm[1], 0.5+focus_mm[1]],
    "z_extent" : [-1+focus_mm[2], 1+focus_mm[2]],
    "dx" : 0.0125,
    "dy" : 0.025,
    "dz" : 0.05,
}
# Domino_field.compute_pressure_field(field_info_mm)

# Compute the pressure volume mesh
# pressure_vol_mesh = Domino_field.get_mesh()

# ------------------ Transform the meshes --------------------

# Transform bg_atlas to the Lab coordinate system
invertz = np.diag([1, 1, -1, 1])  # Invert z-axis for the transducer mesh
Brain_Atlas.reset_mesh()  # Reset the Brain Atlas mesh to the original mesh

Brain_Atlas.transform(T_matrix = invertz , inplace=True) 

# 3) map the image-center to the *lab* coordinate system:
Doppler3D.transform(T_matrix = invertz @ np.linalg.inv(Doppler3D.BrainToLab) @ invertz , inplace=True)  # Transform the scan mesh to the probe coordinate system
Doppler2D.transform(T_matrix = invertz @ np.linalg.inv(Doppler3D.BrainToLab) @ invertz , inplace=True)  # Transform the scan mesh to the probe coordinate system

# -------- Translate the 2D Dopler scan to the marker plane coordinate system --------

# Load the markers file
file_scan = h5py.File(markers_PATH, "r") 
point1 = np.array(file_scan['Plane']['Plane_0']['first'][()]).ravel()
point2 = np.array(file_scan['Plane']['Plane_0']['second'][()]).ravel()

# Let's invert the z axis to match the transducer coordinate system
point1[2] = -point1[2]
point2[2] = -point2[2]

center_plane, R = compute_affine_from_markers(point1, point2)  # Compute the affine transformation matrix from the markers

center_2Ddoppler = np.array(Doppler2D.pv_mesh.center)

center_2Ddoppler[0] = 0 # Set the z coordinate to 0 because we just want translation in the xy plane
center_2Ddoppler[2] = 0 # Set the z coordinate to 0 because we just want translation in the xy plane

# assemble 4×4
t = center_plane - center_2Ddoppler  # Translation vector to the marker plane

# NOTE: this data is from the special motor of Felipe, which does not have a translation
# along the x-axis, so alingment is not guaranteed.

T = np.eye(4)
T[:3, :3] = R
T[:3,  3] = t

Doppler2D.transform(T_matrix = T, inplace=True)  # Transform the scan mesh to the marker plane coordinate system

# ------------------ Code for plotting --------------------

final_plotter = add_regions_mesh(Brain_Atlas.pv_mesh,
                            notebook=False, window_size=[1000, 800],
                            kwargs_dict={region_names[0]: {"color": "lightgray", "opacity": 0.1},	 
                                        region_names[1]: {"color": "permanentgreen", "opacity": 0.2}, 
                                        region_names[2]: {"color": "cadmiumlemon", "opacity": 0.2}}, label = "Brain Atlas")

final_plotter = add_3D_vol(Doppler3D.pv_mesh, plotter = final_plotter,
                                cmap="hot", opacity="sigmoid", opacity_unit_distance = 1,
                                scalar_bar_args={
                                    'title': '3D Doppler (dB)',
                                    "title_font_size": 16,
                                    "label_font_size": 12,
                                    'color' : 'white',
                                })

final_plotter = add_2D_image(Doppler2D.pv_mesh, plotter=final_plotter, cmap='gray', opacity=1.0, show_scalar_bar=True, lighting=True, ambient=1,
                             scalar_bar_args={
                                    'title': '2D Doppler (dB)',
                                    "title_font_size": 16,
                                    "label_font_size": 12,
                                    'vertical': True,
                                    'position_x': 0.1,
                                    'position_y': 0.2,
                                    "height": 0.3,
                                    'color' : 'white',
                                })

final_plotter = add_markers(
    [point1, point2],
    plotter=final_plotter,
    glyph='sphere',
    glyph_scale=0.05,
    color='red',
    labels=['P1', 'P2'],
    label_offset=(0, 0, 0.2),
    label_font_size=14
)

# final_plotter = add_transducer_mesh(TX_mesh,
#                         plotter=final_plotter, show_edges=False, lighting=True, ambient=1,
#                         scalar_bar_args={
#                             'title': 'Apodization',
#                             "title_font_size": 16,
#                             "label_font_size": 12,
#                             'vertical': True,
#                             'position_x': 0.85,
#                             'position_y': 0.6,
#                             "height": 0.3,
#                             'color' : 'white',
#                         })

# final_plotter = add_pressure_vol(pressure_vol_mesh,
#                         plotter=final_plotter,
#                         plot_focal_spot=False, lighting=True, ambient=1,
#                         scalar_bar_args={
#                             'title': 'Pressure Field (PII)',
#                             "title_font_size": 16,
#                             "label_font_size": 12,
#                             'vertical': True,
#                             'position_x': 0.85,
#                             'position_y': 0.2,
#                             "height": 0.3,
#                             'color' : 'white',
#                         })

final_plotter.set_background("black")
final_plotter.show_grid(color='white', font_size=10)  # Show grid with white color and font size 10
final_plotter.add_axes(color='white')
final_plotter.camera_position = 'zy'  # Set the camera position
final_plotter.camera.up = (0, 0, -1)  # Set the camera up direction
final_plotter.show()  # Show the plotter in Jupyter Notebook

final_plotter.close()  # for plotters

# ------------------ Clean up --------------------

Doppler3D.clean()  # for scans
Doppler2D.clean()  # for scans
Domino_field.clean()  # for fields
domino.clean()  # for transducers
Brain_Atlas.clean()  # for atlas
# del TX_mesh, pressure_vol_mesh       # for mesh objects
del final_plotter  # for plotters