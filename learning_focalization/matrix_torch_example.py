import time

import numpy as np
import torch
from helper_function import (
    pattern_from_pr_3Dto2D,
)

import pysonogen
import pysonogen.transducers as Transducers
from pysonogen.psimulation import PyField, TorchField

# print(torch.__version__)
# print(torch.version.cuda)

use_gpu = False  # Set to False if you want to run on CPU
device_number = 0  # if you have multiple GPUs
if torch.cuda.is_available() and use_gpu:
    print(f"Using GPU: {torch.cuda.get_device_name(device_number)}")
    device = torch.device(f"cuda:{device_number}")
else:
    print("No GPU available, running on CPU. May be slow.")
    device = torch.device("cpu")

print(Transducers.available_transducers())

Zeus_Matrix = Transducers.Zeus_Matrix()

# ----------- Main booleans ----------------
plot_result = False
save_figure = False


# ---------- Result, path and model info ---------------
delay_type = "z"  # 'f' for focus, 'z' for zeros, etc
apodization_type = "h"  # 'h' for half, '1' for ones, etc
z_plane_mm = 5
num_epoch = 150
target = "Point100um"
I_times = 1.5

version = f"v{z_plane_mm}mm2"
name_model = f"opt_{num_epoch}epochs_3DloglossE_1planes_delay{delay_type}_apod{apodization_type}_I{I_times}_{target}_{version}"
state_name = f"Matrix_torch_state_{name_model}"
state_folder = r".\test_models\matrix\Phase_tests"
path = f"{state_folder}/{state_name}.pth"

figure_name = (
    state_folder + "/" + state_name + ".png"  # "_unwrap08.png"  # + "_unwrap_"
)  # Add the correct extension


# ---------- Set up the transducer and field computation ---------------
focus_mm = np.array([0, 0, z_plane_mm])  # mm [x, y, z] #8
FoverD = 1
delays = Zeus_Matrix.compute_delays(focus_mm=focus_mm, plot=False)
apodization = Zeus_Matrix.compute_apodization(
    focus_mm=focus_mm, FoverD=FoverD, plot=False
)


Delta_x = 0.3  # 2  #0.8  # mm
Delta_y = 0.3  # 2  #  0.8  # mm
Delta_z = 1  # 3  #  1 mm
field_info_mm = {
    "x_extent": [-Delta_x + focus_mm[0], Delta_x + focus_mm[0]],
    "y_extent": [-Delta_y + focus_mm[1], Delta_y + focus_mm[1]],
    "z_extent": [-Delta_z + focus_mm[2], Delta_z + focus_mm[2]],
    "dx": 0.02,  # 0.075, # 0.02
    "dy": 0.02,  # 0.075, # 0.02
    "dz": 0.02,  # 0.075, # 0.02
}
# factor = 1
# Delta_x = 6 / factor  # 2  #0.8  # mm
# Delta_y = 6 / factor  # 2  #  0.8  # mm
# Delta_z = 0.2  # 3  #  1 mm
# field_info_mm = {
#     "x_extent": [-Delta_x + focus_mm[0], Delta_x + focus_mm[0]],
#     "y_extent": [-Delta_y + focus_mm[1], Delta_y + focus_mm[1]],
#     "z_extent": [-Delta_z + focus_mm[2], Delta_z + focus_mm[2]],
#     "dx": 0.1 / factor,  # 0.075, # 0.02
#     "dy": 0.1 / factor,  # 0.075, # 0.02
#     "dz": 0.1,  # 0.075, # 0.02
# }
# ----------- Create the field object and load the model ---------------

torch.cuda.empty_cache()
Matrix_torch = PyField(Zeus_Matrix)  # , device=device)

# Load the model's state dictionary# Load the checkpoint
path_found_flag = True
if plot_result:
    try:
        checkpoint = torch.load(path)
        Matrix_torch.load_state_dict(checkpoint)
        apodization = Matrix_torch.apodization
        apodization = (
            Matrix_torch._process_apodization(apodization).detach().cpu().numpy()
        )
        delays = Matrix_torch.delays.detach().cpu().numpy()
        print(apodization.shape, delays.shape)
        # # Example usage
        print("Model state dictionary loaded from: ", state_name)
        # Units in TorchField are in us
        delays = delays * 1e-6  # Convert to seconds

    except Exception as e:
        path_found_flag = False
        print("Error loading the model state dictionary:", e)
        # Countdown from 10 seconds

if not path_found_flag:
    countdown_time = 5

    print("Computing focalization example in:\n", end=" ")
    for t in range(countdown_time, 0, -1):
        print(f"{t} seconds...", end="\r", flush=True)
        time.sleep(1)

Zeus_Matrix.set_apodization(apodization)
Zeus_Matrix.set_delays(delays)
Zeus_Matrix.plot_apodization()
clim = None  # [-0.06, 0.06]
Zeus_Matrix.plot_delays(clim=clim)


# ----------- Compute and plot the field ---------------

# x2, y2, z2, pr2 = Matrix_torch.examine_bottleneck(field_info_mm, batch_size=2048)
x2, y2, z2, pr2 = Matrix_torch(field_info_mm)

pr2 = pr2.detach().cpu().numpy()
x2 = x2.detach().cpu().numpy()
y2 = y2.detach().cpu().numpy()
z2 = z2.detach().cpu().numpy()

if not save_figure:
    figure_name = None

pysonogen.plot_field_planes(
    pr2,
    x2,
    y2,
    z2,
    interpolation=None,
    centered=False,
    ratios=[0.2, 1, 0.2],
    save_fig_name=figure_name,
)

plotter = pysonogen.plot_pressure_field(
    pr2,
    x2,
    y2,
    z2,
)


plotter = pysonogen.functions.add_transducer_mesh(
    Zeus_Matrix.get_mesh(), plotter=plotter, lighting=True, ambient=1
)

plotter.show()
plotter.deep_clean()  # Explicitly clean up PyVista objects
plotter.close()
del plotter

# Matrix_field = pyfield.PyField(Zeus_Matrix)
# pr1, x1, y1, z1 = Matrix_field.compute_pressure_field(field_info_mm, inplace=False)
# pysonogen.plot_field_planes(pr1, x1, y1, z1, interpolation=None)
