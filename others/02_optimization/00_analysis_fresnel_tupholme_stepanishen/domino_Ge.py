import time

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

# from torch_test import TorchField, create_simulation_grid
import torch
from scipy.io import savemat

import pyfield
from pyfield.psimulation import PyField, TorchField

print(torch.__version__)
torch.cuda.empty_cache()  # Clear CUDA cache if using GPU
use_cuda = False  # Set to False if you want to run on CPU
device_cpu = torch.device("cpu")
device_number = 0  # if you have multiple GPUs
if torch.cuda.is_available():
    print(f"GPU is available: cuda:{torch.cuda.get_device_name(device_number)}")
    device_cuda = torch.device(f"cuda:{device_number}")
else:
    print("Not GPU available.")
    device_cuda = device_cpu

device = device_cuda if use_cuda else device_cpu

folder = "./data_ge/"
# ------------------- Transducer Matrix -------------------

field_matrix_mm = {
    "x_extent": [-8, 8],  # mm (16,5 mm)
    "y_extent": [0, 0],  # mm(16,5 mm)
    "z_extent": [0, 10],  # mm (2 mm)
    "dx": 0.02,
    "dy": 0.02,
    "dz": 0.02,
}

apod = 1
linear_array_tx = pyfield.transducers.Domino()
linear_array_tx.__setattr__("fc", 10e6)  # 10 MHz
linear_array_tx.set_apodization(np.ones(linear_array_tx.n_elements) * apod)
linear_array_tx.plot_apodization()
linear_array_tx.plot_delays()
linear_array_tx.show()

# ------------------- Create TorchField -------------------

linear_array_torch = TorchField(linear_array_tx)
x, y, z, pr = linear_array_torch(field_matrix_mm, normalize=False)
# linear_array_torch = PyField(linear_array_tx)
# pr, x, y, z = linear_array_torch(field_matrix_mm, normalize=False)


# ------------------- Plotting -------------------
save = False

print(f"Pressure field shape: {pr.shape}")
# plot pressure field
if isinstance(pr, torch.Tensor):
    plane = pr.squeeze().cpu().numpy()  # Convert to numpy array if it's a tensor
    y = y.cpu().numpy()
    x = x.cpu().numpy()
    z = z.cpu().numpy()
else:
    # If pr is already a numpy array, no need to convert
    plane = pr.squeeze()

print(f"Pressure field shape: {plane.shape}")

fig, ax = plt.subplots(figsize=(10, 6))

# flip z = z[::-1]  # Reverse the z-axis for correct orientation
plane = plane[:, ::-1]  # Reverse the z-axis for correct orientation
dz2 = z[2] - z[1]  # Assuming uniform spacing
dx2 = x[2] - x[1]  # Assuming uniform spacing

aspect_ratio = dz2 / dx2  # Calculate aspect ratio based on z and y spacing

# Plot the pressure field
im = ax.imshow(
    plane.T,
    extent=[x.min(), x.max(), z.max(), z.min()],
    aspect=aspect_ratio,  # Set aspect to "image"
    origin="lower",
    cmap="jet",  # Use "jet" colormap
)

# Add labels and title
ax.set_xlabel("x (mm)")
ax.set_ylabel("z (mm)")
ax.set_title("Pressure Field")

# Add colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label("Pressure (units)")

# Adjust layout and show the plot
if save:
    plt.savefig(
        folder + f"pressure_field_apod_{apod}.png", dpi=300, bbox_inches="tight"
    )

    # Prepare data for export
    data_to_export = {
        "plane": plane,  # Pressure field
        "y": y,  # x-coordinates
        "z": z,  # y-coordinates
    }

    # Save data to a .mat file
    savemat(folder + f"pressure_field_apod_{apod}.mat", data_to_export)
    print(f"Data saved to {folder}/pressure_field_apod_{apod}.mat")

plt.show()
