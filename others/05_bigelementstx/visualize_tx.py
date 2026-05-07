import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvista as pv
from plotting_functions import create_2D_image_mesh, plot_volume_slices

from pyfield import PyField
from pyfield.transducers import (
    ConcaveCircularTransducer,
    ConvexCircularTransducer,
    CustomTransducer,
)
from pyfield.utilities import (
    add_2D_image,
    add_transducer_mesh,
    plot_pressure_2D,
    plot_pressure_planes,
)

POSITION_MAPPING = "hex_pad_curved_128_3_lambda_v6_corrected.xlsx"

CENTRAL_FREQUENCY_MHz = 1  # MHz
SPEED_OF_SOUND_MS = 1540  # m/s

lambda_mm = SPEED_OF_SOUND_MS / (CENTRAL_FREQUENCY_MHz * 1e3)  # mm
element_diameter_mm = 2.9 * lambda_mm
radius_curvature_mm = element_diameter_mm / 2

pd_dataframe = pd.read_excel(POSITION_MAPPING)
element_positions_m = pd_dataframe.to_numpy()

# ADD : base in the element position compute normal vectors. The elements are placed on
# a curved surface, so the normal vectors will not be the same for all elements.
normal_vectors = np.zeros_like(element_positions_m)

monoelement = ConvexCircularTransducer(
    diameter_mm=element_diameter_mm,
    focus_mm=2,
    frequency_Hz=CENTRAL_FREQUENCY_MHz * 1e6,
    no_sub_diameter=30,
    ratio_big_patches=0.80,
    refine_factor=3,
    method="spherical",
    normalize_patch_size=True,
)

monoelement.show()

# breakpoint()
# Check pressure field
dxyz = 0.4
plane_monoelement = {
    "x_extent_mm": (-20, 20),
    "y_extent_mm": (0, 0),
    "z_extent_mm": (0, 40),
    "dx_mm": dxyz,
    "dy_mm": dxyz,
    "dz_mm": dxyz,
}

simmono = PyField(monoelement, fs=100e6)
x, y, z, plane_mono = simmono(plane_monoelement)

plot_pressure_planes(x, y, z, plane_mono, db_scale=True)


breakpoint()
num_elements = element_positions_m.shape[0]

list_of_elements = []
for i in range(num_elements):
    list_of_elements.append(monoelement)

transducer = CustomTransducer(
    elements=list_of_elements, positions_mm=element_positions_m * 1e3
)

transducer.show(show_edges=False)

breakpoint()
# compute field
center_mm = (0, 0, 100)
x_extent_mm = (-40, 40)
z_extent_mm = (50, 150)
y_extent_mm = (-40, 40)
dx_mm = 1
dy_mm = 1
dz_mm = 1


VMIN = 0
VMAX = 1
FIGSIZE = (15, 5)

plane_xy_dict = {
    "x_extent_mm": x_extent_mm,
    "y_extent_mm": y_extent_mm,
    "z_extent_mm": (center_mm[2], center_mm[2]),
    "dx_mm": dx_mm,
    "dy_mm": dy_mm,
    "dz_mm": dz_mm,
}
plane_xz_dict = {
    "x_extent_mm": x_extent_mm,
    "y_extent_mm": (center_mm[1], center_mm[1]),
    "z_extent_mm": z_extent_mm,
    "dx_mm": dx_mm,
    "dy_mm": dy_mm,
    "dz_mm": dz_mm,
}
plane_yz_dict = {
    "x_extent_mm": (center_mm[0], center_mm[0]),
    "y_extent_mm": y_extent_mm,
    "z_extent_mm": z_extent_mm,
    "dx_mm": dx_mm,
    "dy_mm": dy_mm,
    "dz_mm": dz_mm,
}

sampling_frequency_Hz = 50e6
txsim = PyField(transducer, fs=sampling_frequency_Hz)
# Dont do a volume but planes
x, y, z, plane_xz = txsim(plane_xz_dict)

max_pr = plane_xz.max()

planes = {"plane_xz": plane_xz.squeeze() / max_pr}

_, y, z, plane_yz = txsim(plane_yz_dict)
x, y, _, plane_xy = txsim(plane_xy_dict)


planes = {
    "plane_xz": plane_xz.squeeze() / max_pr,
    "plane_yz": plane_yz.squeeze() / max_pr,
    "plane_xy": plane_xy.squeeze() / max_pr,
}
coords = {"x": x, "y": y, "z": z}

# turn planes into pyvista meshes
plotter = pv.Plotter(window_size=(800, 800))
transducer_mesh = transducer.get_mesh()

plane_y_offset_mm = {
    "plane": "y",
    "offset_mm": center_mm[1],
}
plane_x_offset_mm = {
    "plane": "x",
    "offset_mm": center_mm[0],
}
plane_z_offset_mm = {
    "plane": "z",
    "offset_mm": center_mm[2],
}
plane_xz_mesh = create_2D_image_mesh(
    planes["plane_xz"],
    extent=(x.min(), x.max(), z.min(), z.max()),
    plane_offset=plane_y_offset_mm,
    scalars="pressure (a.u.)",
)
plane_yz_mesh = create_2D_image_mesh(
    planes["plane_yz"],
    extent=(y.min(), y.max(), z.min(), z.max()),
    plane_offset=plane_x_offset_mm,
    scalars="pressure (a.u.)",
)
plane_xy_mesh = create_2D_image_mesh(
    planes["plane_xy"],
    extent=(x.min(), x.max(), y.min(), y.max()),
    plane_offset=plane_z_offset_mm,
    scalars="pressure (a.u.)",
)

plotter = add_2D_image(
    plane_xz_mesh,
    plotter=plotter,
    cmap="jet",
)
plotter = add_2D_image(
    plane_yz_mesh,
    plotter=plotter,
    cmap="jet",
)
plotter = add_2D_image(
    plane_xy_mesh,
    plotter=plotter,
    cmap="jet",
)


plotter = add_transducer_mesh(transducer_mesh, plotter=plotter, show_edges=False)

font_color = "black"
scale = 1
plotter.show_grid(
    grid="back",
    color=font_color,
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

plotter.show()

del plotter, plane_xz_mesh, plane_yz_mesh, plane_xy_mesh, transducer_mesh
