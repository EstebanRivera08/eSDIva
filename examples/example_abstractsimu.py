import numpy as np
import pyvista as pv

from pyfield.plotting import add_pressure_vol, add_transducer_mesh, create_3Dvol_mesh

# Import pyfield modules
from pyfield.psimulation import PyField
from pyfield.transducers import Domino, MatrixArrayTransducer, Zeus_Matrix

print("\n --- Example 0: Abstract simulation --- \n")

save_fig = False
fig_folder = r""

scale = 3
theme = "dark"
theme = ""
show_scalar_bar = True
if theme == "dark":
    pv.set_plot_theme("dark")

    pv.global_theme.anti_aliasing = "ssaa"
    pv.global_theme.background = "black"
    color = "white"
    pv.global_theme.font.color = "white"
    ambient_tx = 1
    ambient_pr = 0.7
else:
    pv.set_plot_theme("default")
    pv.global_theme.anti_aliasing = "ssaa"
    color = "black"
    ambient_tx = 1
    ambient_pr = 0.6

## -------Define the transducer focus and F/D and field points grid--------
focus_mm = np.array([0, 0, 3])
FoverD = 1

## Define the field points grid
x_half_size_mm = 0.5
y_half_size_mm = 0.5
z_half_size_mm = 1.5
x_extent_mm = [-x_half_size_mm + focus_mm[0], x_half_size_mm + focus_mm[0]]
y_extent_mm = [-y_half_size_mm + focus_mm[1], y_half_size_mm + focus_mm[1]]
z_extent_mm = [-z_half_size_mm + focus_mm[2], z_half_size_mm + focus_mm[2]]
dx_mm = 0.015
dy_mm = 0.015
dz_mm = 0.025


x = np.linspace(
    x_extent_mm[0], x_extent_mm[1], int((x_extent_mm[1] - x_extent_mm[0]) / dx_mm) + 1
)
y = np.linspace(
    y_extent_mm[0], y_extent_mm[1], int((y_extent_mm[1] - y_extent_mm[0]) / dy_mm) + 1
)
z = np.linspace(
    z_extent_mm[0], z_extent_mm[1], int((z_extent_mm[1] - z_extent_mm[0]) / dz_mm) + 1
)
field_point_mm = np.array(np.meshgrid(x, y, z)).T.reshape(-1, 3)
print(
    f"Number of field points: {field_point_mm.shape[0]}, and x.shape: {x.shape}, y.shape: {y.shape}, z.shape: {z.shape}"
)
print(
    f"Field points from {field_point_mm.min(axis=0)} mm to {field_point_mm.max(axis=0)} mm"
)

# Define transducer

matrix_array_probe = MatrixArrayTransducer(
    n_elements_x=17,
    n_elements_y=17,
    element_width_mm=0.2,
    element_height_mm=0.2,
    kerf_x_mm=0.05,
    kerf_y_mm=0.05,
    no_sub_x=2,
    no_sub_y=2,
    frequency_Hz=10e6,
)
# matrix_array_probe = Zeus_Matrix()
## ---------------- Matrix Transducer ------------------------

print("\n--- Matrix Array Transducer ---\n")
# Prepare transducer for simulation

delays = matrix_array_probe.compute_delays(focus_mm=focus_mm)
apodization = matrix_array_probe.compute_apodization(focus_mm=focus_mm, FoverD=FoverD)
_ = matrix_array_probe.plot_delays_apodization()
# matrix_array_probe.show(scalars="Delays")

# Perform simulation

matrix_field = PyField(matrix_array_probe)
p_matrixfield2, coords2 = matrix_field(field_point_mm, method="auto")

# Visualize the results

# TX + Pressure
transducer2_mesh = matrix_array_probe.get_mesh()
pressure2_mesh = create_3Dvol_mesh(
    p_matrixfield2 / p_matrixfield2.max(),
    coords2["x"],
    coords2["y"],
    coords2["z"],
    scalars="Pressure",
)

if save_fig:
    plotter_matrix = pv.Plotter(
        window_size=(600 * scale, 600 * scale), notebook=False, off_screen=True
    )
else:
    plotter_matrix = pv.Plotter(window_size=(600, 600), notebook=False)
    scale = 1
plotter_matrix = add_pressure_vol(
    pressure2_mesh,
    plotter=plotter_matrix,
    ambient=ambient_pr,
    scale=scale,
    show_scalar_bar=show_scalar_bar,
)
plotter_matrix = add_transducer_mesh(
    transducer2_mesh,
    plotter=plotter_matrix,
    ambient=ambient_tx,
    scale=scale,
    show_scalar_bar=show_scalar_bar,
)
plotter_matrix.add_axes(label_size=(0.1, 0.1))
# plotter_matrix.show_grid(
#     grid="back",
#     color=color,
#     font_size=12 * scale,
#     location="outer",
#     xtitle="X (mm)",
#     ytitle="Y (mm)",
#     ztitle="Z (mm)",
#     n_xlabels=5,
#     n_ylabels=5,
#     n_zlabels=5,
#     use_3d_text=False,
# )
plotter_matrix.camera.up = (0, 0, -1)
plotter_matrix.camera_position = [
    (8.337575802510562, 8.068681968876689, 9.846013835561196),
    (0.9644865337340529, -0.6479536811605671, 1.0768292142004936),
    (0.007140178925858137, 0.7062024310127115, -0.7079739714683323),
]


if save_fig:
    plotter_matrix.screenshot(fig_folder + "matrix_array_field.png")
else:
    plotter_matrix.show()

del plotter_matrix, pressure2_mesh, transducer2_mesh
