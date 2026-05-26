import numpy as np
import pandas as pd

from pyfield import Emission, PyField
from pyfield.plotting import (
    add_transducer_mesh,
    plot2D_transient_slices,
    plot3D_transient_slices,
)
from pyfield.transducers import (
    ConvexCircularTransducer,
    CustomTransducer,
)
from pyfield.utilities import align_to_common_time

POSITION_MAPPING = "hex_pad_curved_128_3_lambda_v6_corrected.xlsx"
GEOMETRIC_FOCUS_MM = (0, 0, 100)  # mm
CENTRAL_FREQUENCY_MHz = 1  # MHz
SPEED_OF_SOUND_MS = 1540  # m/s
FIG_NAME = "zeus_pad_transient"
FOLDER_NAME = "./figures/"
DB_SCALE = True
PLANE_WAVE = True
PLOT_2D = False
PLOT_3D = True
SAVE_FIG = False

lambda_mm = SPEED_OF_SOUND_MS / (CENTRAL_FREQUENCY_MHz * 1e3)  # mm
element_diameter_mm = 2.9 * lambda_mm
radius_curvature_mm = element_diameter_mm / 2

pd_dataframe = pd.read_excel(POSITION_MAPPING)
element_positions_m = pd_dataframe.to_numpy()
geometric_focus_m = np.array(GEOMETRIC_FOCUS_MM).reshape((1, -1)) * 1e-3
element_normal_vectors_m = geometric_focus_m - element_positions_m


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

# monoelement.show()

num_elements = element_positions_m.shape[0]


list_of_elements = []
for i in range(num_elements):
    list_of_elements.append(monoelement)


transducer = CustomTransducer(
    elements=list_of_elements,
    positions_mm=element_positions_m * 1e3,
    normals=element_normal_vectors_m,
)


if PLANE_WAVE:
    # zmax_elements = np.max(element_positions_m[:, 2])
    # virtual_plane_array = np.concatenate(
    #     [element_positions_m[:, :2], np.ones((num_elements, 1)) * zmax_elements], axis=1
    # )
    # distance_to_virtual_plane_m = np.linalg.norm(
    #     geometric_focus_m - virtual_plane_array, axis=1
    # )
    # delay_s = distance_to_virtual_plane_m / (SPEED_OF_SOUND_MS)  # s
    # # delay_s = np.max(delay_s) - delay_s  # invert to get delays for plane wave
    # transducer.set_delays(delay_s)
    transducer.compute_delays(angle_steering_deg=(10, -10))

transducer.show(scalars="Delays", show_edges=False)
# breakpoint()
# compute field
center_mm = (0, 0, 100)
x_extent_mm = (-50, 50)
z_extent_mm = (10, 150)
y_extent_mm = (-50, 50)
dx_mm = 1
dy_mm = 1
dz_mm = 1

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
txsim = Emission(transducer, fs=sampling_frequency_Hz)
# Transient simulation — each plane returns (p, coords) with p=(Nt, Nx, Ny, Nz)
pxz, coords_xz = txsim(plane_xz_dict)
pyz, coords_yz = txsim(plane_yz_dict)
# pxy, coords_xy = txsim(plane_xy_dict, monochromatic=False)

common_t, [pxz_a, pyz_a] = align_to_common_time(
    [(pxz, coords_xz), (pyz, coords_yz)],
    align_to_shorter=True,
    # (pxy, coords_xy)]
)

coords = {
    "x": coords_xz["x"],
    "y": coords_yz["y"],
    "z": coords_xz["z"],
}

# Build planes list with offset tracking — squeeze singleton spatial dims → (Nt, N1, N2)
planes = [
    {"plane": "xz", "data": pxz_a.squeeze(), "translation": (0, center_mm[1], 0)},
    {"plane": "yz", "data": pyz_a.squeeze(), "translation": (center_mm[0], 0, 0)},
    # {"plane": "xy", "data": pxy_a.squeeze(), "translation": (0, 0, center_mm[2])},
]

# --- 2D transient animation (matplotlib) ---

if SAVE_FIG:
    save_path = FOLDER_NAME
    comment1 = "dB" if DB_SCALE else "linear"
    comment2 = "plane_wave" if PLANE_WAVE else "focused"
    file2D_name = FIG_NAME + f"_{comment1}_{comment2}_2D.gif"
    file3D_name = FIG_NAME + f"_{comment1}_{comment2}_3D.gif"
else:
    save_path = None
    file2D_name = None
    file3D_name = None


if PLOT_2D:
    plot2D_transient_slices(
        planes,
        coords=coords,
        time_array=common_t,
        db_scale=DB_SCALE,
        save_path=save_path,
        file_name=file2D_name,
        cmap="jet",
        # vmin=-20,
        # vmax=0,
        figsize=(12, 6),
    )

# --- 3D transient with transducer (PyVista) ---

if PLOT_3D:
    transducer_mesh = transducer.get_mesh()

    plotter = add_transducer_mesh(
        transducer_mesh, window_size=(800, 800), show_edges=False
    )

    plotter = plot3D_transient_slices(
        planes,
        coords=coords,
        db_scale=DB_SCALE,
        save_path=save_path,
        file_name=file3D_name,
        time_array=common_t,
        center_mm=center_mm,
        plotter=plotter,
        # vmin=-20,
        # vmax=0,
        cmap="jet",
    )

    del plotter, transducer_mesh
