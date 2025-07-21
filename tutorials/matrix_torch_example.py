import numpy as np
import torch
from TorchFieldv2 import TorchFieldv2 as TorchField

import pysonogen
import pysonogen.transducers as Transducers
from pysonogen import pyfield

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

# Focalization spot
focus_mm = np.array([0, 0, 5])  # mm [x, y, z]
# delays = Zeus_Matrix.compute_delays(focus_mm=focus_mm, plot=True)

folder = r"..\pressure_fields"
filename = r"/target_inverse.npz"
name_model = "opt_20epcochs_init_focus_5mm_FoverD_2.0"
state_name = f"Matrix_torch_state_{name_model}.pth"

Delta_x = 1  # mm
Delta_y = 1  # mm
Delta_z = 2  # mm`
field_info_mm = {
    "x_extent": [-Delta_x + focus_mm[0], Delta_x + focus_mm[0]],
    "y_extent": [-Delta_y + focus_mm[1], Delta_y + focus_mm[1]],
    "z_extent": [-Delta_z + focus_mm[2], Delta_z + focus_mm[2]],
    "dx": 0.03,
    "dy": 0.03,
    "dz": 0.05,
}

# Zeus_Matrix.show()

torch.cuda.empty_cache()
Matrix_torch = TorchField(Zeus_Matrix, device=device)

# Load the model's state dictionary
Matrix_torch.load_state_dict(torch.load(state_name))
print("Model state dictionary loaded from: ", state_name)


pr2, x2, y2, z2 = Matrix_torch(field_info_mm, batch_size=2048)

pysonogen.plot_field_planes(
    pr2.detach().cpu().numpy(),
    x2.detach().cpu().numpy(),
    y2.detach().cpu().numpy(),
    z2.detach().cpu().numpy(),
    interpolation="bilinear",
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
