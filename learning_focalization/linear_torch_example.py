import matplotlib.pyplot as plt
import numpy as np
import torch

# from TorchField import TorchField
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


Domino = Transducers.Domino()

# ----------------------------
version = "v1"
num_epoch = 30

target_folder = r".\target_masks"
target_filename = r"/linear_4lambda.npz"
destination = r".\test_models\linear"
name_model = f"opt_{num_epoch}epochs_3DloglossE_5planes_delays_random_{version}"
state_name = f"Linear_torch_state_{name_model}"
path = destination + "/" + state_name + ".pth"  # Add the correct extension
data = destination + "/" + name_model + ".npz"  # Add the correct extension


# Focalization spot
focus_mm = np.array([0, 0, 8])  # mm [x, y, z]
delays = Domino.compute_delays(focus_mm=focus_mm, plot=False)


Delta_x = 1.2  # 2  # mm
Delta_y = 0.5  # 2  # mm
Delta_z = 2  # 3  # mm`
field_info_mm = {
    "x_extent": [-Delta_x + focus_mm[0], Delta_x + focus_mm[0]],
    "y_extent": [-Delta_y + focus_mm[1], Delta_y + focus_mm[1]],
    "z_extent": [-Delta_z + focus_mm[2], Delta_z + focus_mm[2]],
    "dx": 0.03,  # 0.075
    "dy": 0.03,  # 0.075
    "dz": 0.03,  # 0.075
}

# Zeus_Matrix.show()

torch.cuda.empty_cache()
domino_torch = TorchField(Domino, device=device)

# Load the model's state dictionary# Load the checkpoint
checkpoint = torch.load(path)
process = True
domino_torch.load_state_dict(checkpoint)
apodization = domino_torch.apodization
delays = domino_torch.delays
if process:
    apodization = domino_torch._process_apodization(apodization)
    delays = domino_torch._process_delays(delays)
apodization = apodization.detach().cpu().numpy()
delays = delays.detach().cpu().numpy()
print(f"Apodization shape: {apodization.shape}")
print(f"Delays shape: {delays.shape}")


x_test = torch.linspace(-2, 2, 100)
y_test = domino_torch.softplus(x_test)
plt.plot(x_test.detach().cpu().numpy(), x_test.detach().cpu().numpy(), "--b")
plt.plot(x_test.detach().cpu().numpy(), y_test.detach().cpu().numpy(), "r")
plt.grid()
plt.show()

Domino.set_apodization(apodization)
Domino.set_delays(delays * 1e-6)
Domino.plot_apodization()
Domino.plot_delays()


domino_torch = TorchField(Domino, device=device)

# # Example usage
print("Model state dictionary loaded from: ", state_name)


pr2, x2, y2, z2 = domino_torch.examine_bottleneck(field_info_mm, batch_size=2048)

pysonogen.plot_field_planes(
    pr2.detach().cpu().numpy(),
    x2.detach().cpu().numpy(),
    y2.detach().cpu().numpy(),
    z2.detach().cpu().numpy(),
    interpolation=None,
    centered=False,
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

# Matrix_field = pyfield.PyField(Zeus_Matrix)
# pr1, x1, y1, z1 = Matrix_field.compute_pressure_field(field_info_mm, inplace=False)
# pysonogen.plot_field_planes(pr1, x1, y1, z1, interpolation=None)
