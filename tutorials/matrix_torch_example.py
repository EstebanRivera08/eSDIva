import numpy as np
import torch
import torch.nn.functional as F
from helper_function import (
    apply_gaussian_filter,
    gaussian_kernel,  # Import the helper functions
)
from TorchField import TorchField
from TorchFieldv2 import TorchFieldv2 as TorchField

import pysonogen
import pysonogen.transducers as Transducers

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

# ----------------------------

# Focalization spot
focus_mm = np.array([-0.5, 0, 5])  # mm [x, y, z]
delays = Zeus_Matrix.compute_delays(focus_mm=focus_mm, plot=False)

folder = r"..\pressure_fields"
filename = r"/target_inverse.npz"
name_model = "opt_10epochs_init_focus_5mm_FoverD_1_v5"
state_name = f"Matrix_torch_state_{name_model}.pth"

Delta_x = 2  # 0.3  # 1  # mm
Delta_y = 2  # 0.3  # 1  # mm
Delta_z = 3  # 1  # 2  # mm`
field_info_mm = {
    "x_extent": [-Delta_x + focus_mm[0], Delta_x + focus_mm[0]],
    "y_extent": [-Delta_y + focus_mm[1], Delta_y + focus_mm[1]],
    "z_extent": [-Delta_z + focus_mm[2], Delta_z + focus_mm[2]],
    "dx": 0.075,
    "dy": 0.075,
    "dz": 0.075,
}

# Zeus_Matrix.show()

torch.cuda.empty_cache()
Matrix_torch = TorchField(Zeus_Matrix, device=device)

# Load the model's state dictionary# Load the checkpoint
checkpoint = torch.load(state_name)
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
    centered=True,
)

plotter, vol_mesh = pysonogen.plot_pressure_field(
    pr2.detach().cpu().numpy(),
    x2.detach().cpu().numpy(),
    y2.detach().cpu().numpy(),
    z2.detach().cpu().numpy(),
)

plotter.show()
plotter.deep_clean()  # Explicitly clean up PyVista objects
plotter.close()

# Matrix_field = pyfield.PyField(Zeus_Matrix)
# pr1, x1, y1, z1 = Matrix_field.compute_pressure_field(field_info_mm, inplace=False)
# pysonogen.plot_field_planes(pr1, x1, y1, z1, interpolation=None)
