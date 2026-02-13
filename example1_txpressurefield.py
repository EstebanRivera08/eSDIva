import numpy as np
import pyvista as pv

from pyfield.psimulation import PyField
from pyfield.transducers import Domino, Zeus_Matrix
from pyfield.utilities import add_pressure_vol, add_transducer_mesh, create_vol_mesh

print("\n --- Example 0: Linear and Matrix Array Transducers --- \n")

save_fig = False
fig_folder = r""

scale = 3
run_linear_array = True
run_matrix_array = False
theme = "dark"


if theme == "dark":
    pv.set_plot_theme("dark")

    pv.global_theme.anti_aliasing = "ssaa"
    pv.global_theme.background = "black"
    color = "white"
    pv.global_theme.font.color = "white"
    ambient_tx = 0.1
    ambient_pr = 0.55
else:
    pv.global_theme.anti_aliasing = "ssaa"
    color = "black"
    ambient_tx = 1
    ambient_pr = 0.5

## -------Define the transducer focus and F/D and field points grid--------
focus_mm = np.array([-2, 0, 8])
FoverD = 1

## Define the field points grid
x_half_size_mm = 0.25
y_half_size_mm = 0.5
z_half_size_mm = 1.5
x_extent_mm = [-x_half_size_mm + focus_mm[0], x_half_size_mm + focus_mm[0]]
y_extent_mm = [-y_half_size_mm + focus_mm[1], y_half_size_mm + focus_mm[1]]
z_extent_mm = [-z_half_size_mm + focus_mm[2], z_half_size_mm + focus_mm[2]]
dx_mm = 0.015
dy_mm = 0.02
dz_mm = 0.05


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


## ------------------------ Linear Transducer ------------------------

if run_linear_array:
    print("\n--- Linear Array Transducer ---\n")
    # Prepare transducer for simulation

    linear_array_probe = Domino()

    delays = linear_array_probe.compute_delays(focus_mm=focus_mm)
    apodization = linear_array_probe.compute_apodization(focus_mm=focus_mm, FoverD=1)
    linear_array_probe.plot_delays_apodization()

    # Perform simulation

    linear_field = PyField(linear_array_probe)
    x, y, z, p_linearfield = linear_field(field_point_mm, method="auto")

    # Visualize the results

    # TX + Pressure
    transducer_mesh = linear_array_probe.get_mesh()
    pressure_mesh = create_vol_mesh(
        x, y, z, p_linearfield / p_linearfield.max(), scalars="Pressure "
    )

    if save_fig:
        plotter_linear = pv.Plotter(
            window_size=(500 * scale, 600 * scale), off_screen=True
        )
    else:
        plotter_linear = pv.Plotter(window_size=(500, 600))
        scale = 1
    plotter_linear = add_pressure_vol(
        pressure_mesh, plotter=plotter_linear, ambient=ambient_pr, scale=scale
    )
    plotter_linear = add_transducer_mesh(
        transducer_mesh, plotter=plotter_linear, ambient=ambient_tx, scale=scale
    )
    plotter_linear.add_axes(label_size=(0.1, 0.1))
    plotter_linear.show_grid(
        grid="back",
        color=color,
        font_size=12 * scale,
        location="outer",
        xtitle="X (mm)",
        ytitle="Y (mm)",
        ztitle="Z (mm)",
        n_xlabels=5,
        n_ylabels=3,
        n_zlabels=6,
        use_3d_text=False,
    )
    plotter_linear.camera.up = (0, 0, -1)
    plotter_linear.camera_position = [
        (-22.51213444439079, 14.439697300516023, -13.407465311152013),
        (-1.0952768832851267, -0.8088689986131998, 5.543738629581104),
        (0.5021268156034906, -0.30191187051551605, -0.8103813198079781),
    ]
    if save_fig:
        plotter_linear.screenshot(fig_folder + "linear_array_field.png")
    else:
        plotter_linear.show(jupyter_backend="static")

    plotter_linear.close()

    # Pressure

    if save_fig:
        plotter1_linear = pv.Plotter(
            window_size=(400 * scale, 600 * scale), notebook=False, off_screen=True
        )
    else:
        plotter1_linear = pv.Plotter(window_size=(400, 600), notebook=False)
        scale = 1

    plotter1_linear = add_pressure_vol(
        pressure_mesh, plotter=plotter1_linear, ambient=ambient_pr, scale=scale
    )
    plotter1_linear.add_axes(label_size=(0.1, 0.1))
    plotter1_linear.show_grid(
        grid="back",
        color=color,
        font_size=12 * scale,
        location="outer",
        xtitle="X (mm)",
        ytitle="Y (mm)",
        ztitle="Z (mm)",
        n_xlabels=3,
        n_ylabels=5,
        n_zlabels=6,
        use_3d_text=False,
    )
    plotter1_linear.camera.up = (0, 0, -1)
    plotter1_linear.camera_position = [
        (2.0820282989636683, 5.177797014981693, 4.3757579791624535),
        (-1.9244180557178105, -0.043042807059305105, 8.02139891723204),
        (-0.24872901027829428, -0.41893111811461886, -0.8732872366645557),
    ]
    if save_fig:
        plotter1_linear.screenshot(fig_folder + "linear_array_pressure_field.png")
    else:
        plotter1_linear.show()

    plotter1_linear.close()

    del (
        plotter_linear,
        plotter1_linear,
        transducer_mesh,
        pressure_mesh,
    )

## ---------------- Matrix Transducer ------------------------

if run_matrix_array:
    print("\n--- Matrix Array Transducer ---\n")
    # Prepare transducer for simulation

    matrix_array_probe = Zeus_Matrix()

    delays = matrix_array_probe.compute_delays(focus_mm=focus_mm)
    apodization = matrix_array_probe.compute_apodization(
        focus_mm=focus_mm, FoverD=FoverD
    )
    matrix_array_probe.plot_delays_apodization()
    matrix_array_probe.show(notebook=True, jupyter_backend="static", scalars="Delays")

    # Perform simulation

    matrix_field = PyField(matrix_array_probe)
    x2, y2, z2, p_matrixfield2 = matrix_field(field_point_mm, method="auto")

    # Visualize the results

    # TX + Pressure
    transducer2_mesh = matrix_array_probe.get_mesh()
    pressure2_mesh = create_vol_mesh(
        x2, y2, z2, p_matrixfield2 / p_matrixfield2.max(), scalars="Pressure"
    )

    if save_fig:
        plotter_matrix = pv.Plotter(
            window_size=(600 * scale, 600 * scale), notebook=False, off_screen=True
        )
    else:
        plotter_matrix = pv.Plotter(window_size=(600, 600), notebook=False)
        scale = 1
    plotter_matrix = add_pressure_vol(
        pressure2_mesh, plotter=plotter_matrix, ambient=ambient_pr, scale=scale
    )
    plotter_matrix = add_transducer_mesh(
        transducer2_mesh, plotter=plotter_matrix, ambient=ambient_tx, scale=scale
    )
    plotter_matrix.add_axes(label_size=(0.1, 0.1))
    plotter_matrix.show_grid(
        grid="back",
        color=color,
        font_size=12 * scale,
        location="outer",
        xtitle="X (mm)",
        ytitle="Y (mm)",
        ztitle="Z (mm)",
        n_xlabels=5,
        n_ylabels=5,
        n_zlabels=5,
        use_3d_text=False,
    )
    plotter_matrix.camera.up = (0, 0, -1)
    plotter_matrix.camera_position = [
        (-37.53844651353719, 25.79056886018332, -29.548514656452344),
        (0.5603746639260521, 0.0033895602062230523, 4.801817969236558),
        (0.10785893662586615, -0.7339343663829789, -0.6706018160070494),
    ]

    if save_fig:
        plotter_matrix.screenshot(fig_folder + "matrix_array_field.png")
    else:
        plotter_matrix.show(jupyter_backend="static")
    plotter_matrix.close()

    #  Pressure
    if save_fig:
        plotter2_matrix = pv.Plotter(
            window_size=(400 * scale, 600 * scale), notebook=False, off_screen=True
        )
    else:
        plotter2_matrix = pv.Plotter(window_size=(400, 600), notebook=False)
        scale = 1
    plotter2_matrix = add_pressure_vol(
        pressure2_mesh, plotter=plotter2_matrix, ambient=ambient_pr, scale=scale
    )
    plotter2_matrix.add_axes(label_size=(0.1, 0.1))
    plotter2_matrix.show_grid(
        grid="back",
        color=color,
        font_size=12 * scale,
        location="outer",
        xtitle="X (mm)",
        ytitle="Y (mm)",
        ztitle="Z (mm)",
        n_xlabels=3,
        n_ylabels=5,
        n_zlabels=6,
        use_3d_text=False,
    )
    plotter2_matrix.camera.up = (0, 0, -1)
    plotter2_matrix.camera_position = [
        (2.0820282989636683, 5.177797014981693, 4.3757579791624535),
        (-1.9244180557178105, -0.043042807059305105, 8.02139891723204),
        (-0.24872901027829428, -0.41893111811461886, -0.8732872366645557),
    ]
    if save_fig:
        plotter2_matrix.screenshot(fig_folder + "matrix_array_pressure_field.png")
    else:
        plotter2_matrix.show(jupyter_backend="static")

    plotter2_matrix.close()

    del plotter2_matrix, plotter_matrix, transducer2_mesh, pressure2_mesh
