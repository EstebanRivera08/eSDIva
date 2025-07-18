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
focus_mm = np.array([0, 0, 8])  # mm [x, y, z]

field_info_mm = {
    "x_extent": [-0.3 + focus_mm[0], 0.3 + focus_mm[0]],
    "y_extent": [-0.3 + focus_mm[1], 0.3 + focus_mm[1]],
    "z_extent": [-1 + focus_mm[2], 1 + focus_mm[2]],
    "dx": 0.02,
    "dy": 0.02,
    "dz": 0.02,
}

delays = Zeus_Matrix.compute_delays(focus_mm=focus_mm, plot=True)
# Zeus_Matrix.show()

torch.cuda.empty_cache()
Matrix_torch = TorchField(Zeus_Matrix, device=device)

pr2, x2, y2, z2 = Matrix_torch(field_info_mm)

pysonogen.plot_field_planes(
    pr2.detach().cpu().numpy(),
    x2.detach().cpu().numpy(),
    y2.detach().cpu().numpy(),
    z2.detach().cpu().numpy(),
    interpolation=None,
)

# Matrix_field = pyfield.PyField(Zeus_Matrix)
# pr1, x1, y1, z1 = Matrix_field.compute_pressure_field(field_info_mm, inplace=False)
# pysonogen.plot_field_planes(pr1, x1, y1, z1, interpolation=None)
