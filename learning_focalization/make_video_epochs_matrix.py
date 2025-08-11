import numpy as np
import torch

# from TorchField import TorchField
from TorchFieldv2 import TorchFieldv2 as TorchField

import pysonogen
import pysonogen.transducers as Transducers
from learning_focalization.helper_function import pattern_from_pr_3Dto2D

# print(torch.__version__)
# print(torch.version.cuda)

use_cuda = True  # Set to False if you want to run on CPU
device_number = 0  # if you have multiple GPUs
if torch.cuda.is_available() and use_cuda:
    print(f"Using GPU: {torch.cuda.get_device_name(device_number)}")
    device = torch.device(f"cuda:{device_number}")
else:
    print("No GPU available, running on CPU. May be slow.")
    device = torch.device("cpu")

print(Transducers.available_transducers())

Zeus_Matrix = Transducers.Zeus_Matrix()

# ---------------------------- Functions ----------------------------


def take_slices(pressure_field, x, y, z, centered=False):
    """Take 2D slices of the 3D pressure field at the middle or centered indices.
    Parameters
    ----------
    pressure_field : ndarray
        3D pressure field data.
    x, y, z : ndarray
        Coordinate arrays.
    centered : bool, optional
        If True, slices are taken at the indices closest to the maximum value in the field.
    Returns
    -------
    tuple of ndarray
        Slices of the pressure field in the XZ, XY, and YZ planes.
    """
    if centered:
        # Look for the y, x, z indices that are closest to the max value
        max_idx = np.unravel_index(np.nanargmax(pressure_field), pressure_field.shape)
        y0, x0, z0 = max_idx[1], max_idx[0], max_idx[2]
    else:
        # Use the middle indices
        y0 = int(np.floor(y.shape[0] / 2))
        x0 = int(np.floor(x.shape[0] / 2))
        z0 = int(np.floor(z.shape[0] / 2))
    # print(
    #     f"Taking slice x_ind, y_ind, z_ind = {x0 + 1}/{x.shape[0]}, {y0 + 1}/{y.shape[0]}, {z0 + 1}/{z.shape[0]}"
    # )

    # Use nanmin and nanmax to ignore NaN values
    vmin = np.nanmin(pressure_field)
    vmax = np.nanmax(pressure_field)

    XZ_plane = pressure_field[:, y0, :].squeeze()
    XY_plane = pressure_field[:, :, z0].squeeze()
    YZ_plane = pressure_field[x0, :, :].squeeze()

    return {
        "x_slice": x[x0],
        "y_slice": y[y0],
        "z_slice": z[z0],
        "XZ": XZ_plane,
        "XY": XY_plane,
        "YZ": YZ_plane,
        "vmin": vmin,
        "vmax": vmax,
    }


def plot_results(data, pressure_field, x, y, z, figure_name=None):
    loss_vect = data["loss"]
    delays_vect = data["delays"]
    apodization_vect = data["apodization"]
    max_pr_vect = data["max_pr"]

    y_pred = pattern_from_pr_3Dto2D(pressure_field, max_pr_vect)


# ----------------------------

# Focalization spot
focus_mm = np.array([0, 0, 5])  # mm [x, y, z] #8
FoverD = 0.75
delays = Zeus_Matrix.compute_delays(focus_mm=focus_mm, plot=False)
apodization = Zeus_Matrix.compute_apodization(
    focus_mm=focus_mm, F_over_D=FoverD, apodization_type="circular", plot=False
)
target = "1lambda"  # Target pattern to use
name_model = f"opt_150epochs_3DloglossE_1planes_noprocess_delay_apod_{target}_v1"
state_name = f"Matrix_torch_state_{name_model}.pth"
state_folder = r".\test_models\matrix"
path = f"{state_folder}/{state_name}"
data_path = f"{state_folder}/{state_name}_data.npz"
figure_name = (
    state_folder + "/" + state_name + "small.png"  # "_unwrap08.png"  # + "_unwrap_"
)  # Add the correct extension

Delta_x = 2  #  0.8  # mm
Delta_y = 2  #  0.8  # mm
Delta_z = 3  #  1 mm
field_info_mm = {
    "x_extent": [-Delta_x + focus_mm[0], Delta_x + focus_mm[0]],
    "y_extent": [-Delta_y + focus_mm[1], Delta_y + focus_mm[1]],
    "z_extent": [-Delta_z + focus_mm[2], Delta_z + focus_mm[2]],
    "dx": 0.075,  # 0.02
    "dy": 0.075,  # 0.02
    "dz": 0.075,  # 0.02
}

# Zeus_Matrix.show()

torch.cuda.empty_cache()
Matrix_torch = TorchField(Zeus_Matrix, device=device)

# Load the model's state dictionary# Load the checkpoint
checkpoint = torch.load(path)
Matrix_torch.load_state_dict(checkpoint)
apodization = Matrix_torch.apodization.reshape(
    Zeus_Matrix.n_elem_x, Zeus_Matrix.n_elem_y
)
delays = Matrix_torch.delays.reshape(Zeus_Matrix.n_elem_x, Zeus_Matrix.n_elem_y)

Zeus_Matrix.plot_apodization(apodization.detach().cpu().numpy())
Zeus_Matrix.plot_delays(delays.detach().cpu().numpy())

# # Example usage
print("Model state dictionary loaded from: ", state_name)


pr2, x2, y2, z2 = Matrix_torch.examine_bottleneck(field_info_mm, batch_size=2048)


pysonogen.plot_field_planes(
    pr2.detach().cpu().numpy(),
    x2.detach().cpu().numpy(),
    y2.detach().cpu().numpy(),
    z2.detach().cpu().numpy(),
    interpolation=None,
    centered=False,
    save_fig_name=figure_name,
)

plotter = pysonogen.plot_pressure_field(
    pr2.detach().cpu().numpy(),
    x2.detach().cpu().numpy(),
    y2.detach().cpu().numpy(),
    z2.detach().cpu().numpy(),
)

plotter.show()
plotter.deep_clean()  # Explicitly clean up PyVista objects
plotter.close()
