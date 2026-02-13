import numpy as np
import pyvista as pv

import pyfield
from pyfield.psimulation import PyField
from pyfield.transducers import Domino, Zeus_Matrix
from pyfield.utilities import add_transducer_mesh

print("\n --- Example 0: Linear and Matrix Array Transducers --- \n")

save_fig = False
fig_folder = r".\others\figures\\"
fig_version = r"_tx"

run_linear_array = True
run_matrix_array = True
theme = None  # "dark"

scale = 1
off_screen = False

if save_fig:
    scale = 3
    off_screen = True


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

## -------Define the transducer focus and F/D and field points grid--------
focus_mm = np.array([-2, 0, 8])
FoverD = 1

# Create transducer with elevation focus
n_elements = 128
pitch_mm = 0.125
element_width_mm = 0.105
element_height_mm = 1.5
elevation_focus_mm = 3  # mm
tx_freq = 12.5  # MHz

linear_array_probe = pyfield.transducers.LinearArrayTransducer(
    n_elements=n_elements,
    element_width_mm=element_width_mm,
    element_height_mm=element_height_mm,
    elevation_focus_mm=elevation_focus_mm,
    kerf_mm=pitch_mm - element_width_mm,
    no_sub_x=1,
    no_sub_y=15,
    frequency_Hz=tx_freq * 1e6,  # MHz),
)


## ------------------------ Linear Transducer ------------------------

if run_linear_array:
    print("\n--- Linear Array Transducer ---\n")
    # Prepare transducer for simulation

    delays = linear_array_probe.compute_delays(focus_mm=focus_mm)
    apodization = linear_array_probe.compute_apodization(focus_mm=focus_mm, FoverD=1)

    # Perform simulation

    # linear_field = PyField(linear_array_probe)
    # x, y, z, p_linearfield = linear_field(field_point_mm, method="auto")

    # Visualize the results

    # TX + Pressure
    transducer_mesh = linear_array_probe.get_mesh()
    # pressure_mesh = create_vol_mesh(
    #     x, y, z, p_linearfield / p_linearfield.max(), scalars="Pressure "
    # )

    plotter_linear = pv.Plotter(
        window_size=(int(850 * scale), int(750 * scale)), off_screen=off_screen
    )
    # plotter_linear = add_pressure_vol(
    #     pressure_mesh, plotter=plotter_linear, ambient=ambient_pr, scale=scale
    # )
    plotter_linear = add_transducer_mesh(
        transducer_mesh, plotter=plotter_linear, ambient=ambient_tx, scale=scale
    )
    plotter_linear.add_axes(label_size=(0.1, 0.1))
    plotter_linear.show_grid(
        grid="back",
        color=color,
        font_size=int(12 * scale),
        location="outer",
        xtitle="X (mm)",
        ytitle="Y (mm)",
        ztitle="Z (mm)",
        n_xlabels=5,
        n_ylabels=3,
        n_zlabels=2,
        use_3d_text=False,
    )
    plotter_linear.camera.up = (0, 0, -1)

    # # Zoomed in view
    # plotter_linear.camera_position = [
    #     (-11.047333815555612, -0.47907480704120475, -1.6410793367116736),
    #     (-3.299557626274004, 0.748324925128876, 1.720490313371935),
    #     (0.3730955246553406, 0.15170613665836877, -0.9153059475292119),
    # ]

    # Full view

    plotter_linear.camera_position = [
        (-15.538531850881864, -1.7859023675049732, -6.738322261485786),
        (-2.714507591995717, 0.5245443200761408, 0.9293185881022636),
        (0.47773137293754436, 0.192788809749591, -0.8570911329295999),
    ]

    if save_fig:
        linear_fig_name = "linear" + fig_version + ".png"
        print("\n Saving linear tx figure...")
        plotter_linear.screenshot(fig_folder + linear_fig_name)
    else:
        plotter_linear.show()
    # plotter_linear.close()
    #
    # del (
    #     plotter_linear,
    #     transducer_mesh,
    #     # pressure_mesh,
    # )


## ---------------- Matrix Transducer ------------------------

if run_matrix_array:
    print("\n--- Matrix Array Transducer ---\n")
    # Prepare transducer for simulation

    matrix_array_probe = Zeus_Matrix()

    delays = matrix_array_probe.compute_delays(focus_mm=focus_mm)
    apodization = matrix_array_probe.compute_apodization(
        focus_mm=focus_mm, FoverD=FoverD
    )

    # Perform simulation

    matrix_field = PyField(matrix_array_probe)
    # x2, y2, z2, p_matrixfield2 = matrix_field(field_point_mm, method="auto")

    # Visualize the results

    # TX + Pressure
    transducer2_mesh = matrix_array_probe.get_mesh()
    # pressure2_mesh = create_vol_mesh(
    #     x2, y2, z2, p_matrixfield2 / p_matrixfield2.max(), scalars="Pressure"
    # )

    plotter_matrix = pv.Plotter(
        window_size=(int(600 * scale), int(600 * scale)), off_screen=off_screen
    )

    # plotter_matrix = add_pressure_vol(
    #     pressure2_mesh, plotter=plotter_matrix, ambient=ambient_pr, scale=scale
    # )
    plotter_matrix = add_transducer_mesh(
        transducer2_mesh, plotter=plotter_matrix, ambient=ambient_tx, scale=scale
    )
    plotter_matrix.add_axes(label_size=(0.1, 0.1))
    plotter_matrix.show_grid(
        grid="back",
        color=color,
        font_size=int(12 * scale),
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
        (-14.89496893666854, 16.6721215658849, -36.850531067595966),
        (0.6351523603255553, -0.29448608571276025, 4.652062366165944),
        (0.10314365853154672, -0.9065885750200096, -0.409216985654208),
    ]

    if save_fig:
        print("\n Saving matrix tx figure...")
        matrix_fig_name = "matrix" + fig_version + ".png"
        plotter_matrix.screenshot(fig_folder + matrix_fig_name)
    else:
        plotter_matrix.show()

    # del plotter_matrix
    # (transducer2_mesh,)
    # # pressure2_mesh
