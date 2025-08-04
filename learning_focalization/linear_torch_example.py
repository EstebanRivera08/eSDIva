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
# Domino.show()

# ----------------------------
version = "v3"
num_epoch = 200

target_folder = r".\target_masks"
target_filename = r"/linear_4lambda.npz"
destination = r".\test_models\linear"
name_model = (
    f"opt_{num_epoch}epochs_3DloglossE_1planes_noprocess_half_target2_{version}"
)
state_name = f"Linear_torch_state_{name_model}"
path = destination + "/" + state_name + ".pth"  # Add the correct extension
data = destination + "/" + name_model + ".npz"  # Add the correct extension
figure_name = (
    destination + "/" + state_name + ".png"  # + "_unwrap_"
)  # Add the correct extension


# Focalization spot
focus_mm = np.array([0, 0, 8])  # mm [x, y, z]
delays_expected = Domino.compute_delays(focus_mm=focus_mm, plot=False)


Delta_x = 1.2  # 2  # mm
Delta_y = 0.5  # 2  # mm
Delta_z = 3  # 3  # mm
field_info_mm = {
    "x_extent": [-Delta_x + focus_mm[0], Delta_x + focus_mm[0]],
    "y_extent": [-Delta_y + focus_mm[1], Delta_y + focus_mm[1]],
    "z_extent": [-Delta_z + focus_mm[2], Delta_z + focus_mm[2]],
    "dx": 0.03,  # 0.075
    "dy": 0.03,  # 0.075
    "dz": 0.06,  # 0.075
}

# Zeus_Matrix.show()

torch.cuda.empty_cache()
domino_torch = TorchField(Domino, device=device)

# Load the model's state dictionary# Load the checkpoint
checkpoint = torch.load(path)
domino_torch.load_state_dict(checkpoint)
process = True
apodization = domino_torch.apodization
delays_before = domino_torch.delays
if process:
    apodization = domino_torch._process_apodization(apodization)
    delays = domino_torch._process_delays(delays_before)


apodization = apodization.detach().cpu().numpy()
delays_before = delays_before.detach().cpu().numpy()
delays = delays.detach().cpu().numpy()
print(f"Apodization shape: {apodization.shape}")
print(f"Delays shape: {delays.shape}")


# x_test = torch.linspace(-2, 2, 100)
# y_test = domino_torch.softplus(x_test)
# plt.plot(x_test.detach().cpu().numpy(), x_test.detach().cpu().numpy(), "--b")
# plt.plot(x_test.detach().cpu().numpy(), y_test.detach().cpu().numpy(), "r")
# plt.grid()
# plt.show()

# Domino.plot_apodization(apodization)
# Domino.plot_delays(delays * 1e-6)
# Compute jumps
d_delays = np.abs(np.diff(delays))


period = 2 * np.pi / Domino.fc * 1e6  # Convert to microseconds


def compute_unwrap_threshold(wrapped_delays, expected_delays, unwrap_threshold):
    """
    Compute the unwrapped delays based on a threshold.
    """

    unwrapped_delays = np.unwrap(wrapped_delays, period=unwrap_threshold)
    return np.abs(
        (unwrapped_delays - unwrapped_delays.max())
        - (expected_delays - expected_delays.max())
    ).sum()


thresholds = np.linspace(0.01, 0.15, 100)
errors = np.zeros_like(thresholds)
for i, threshold in enumerate(thresholds):
    error = compute_unwrap_threshold(delays, delays_expected, threshold)
    errors[i] = error
# Domino.plot_apodization()
# Domino.plot_delays()
argmin_error = np.argmin(errors)
unwrap_threshold = thresholds[argmin_error]
print(f"Optimal unwrap threshold: {unwrap_threshold:.4f} µs")
unwrapped_delays = np.unwrap(delays, period=0.08)
# plt.plot(thresholds, errors, label="Error vs Threshold")
# plt.xlabel("Unwrap Threshold (µs)")


print(f"period: {period} µs")
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax = ax.flatten()

k = 0
ax[k].plot(delays_before, "ro", label="unprocessed Delays (µs)")
ax[k].plot(delays_before, "-k", label="unprocessed Delays (µs)")
ax[k].set_xlabel("Element Index")
ax[k].set_ylabel("Delay (µs)")
ax[k].set_title("unprocessed Delays")
ax[k].grid()
k += 1

ax[k].plot(delays, "ro", label="wrapped Delays (µs)")
ax[k].plot(delays, "-k", label="wrapped Delays (µs)")
ax[k].set_xlabel("Element Index")
ax[k].set_ylabel("Delay (µs)")
ax[k].set_title("process Delays")
ax[k].grid()
k += 1

ax[k].plot(unwrapped_delays, "bo", label="Unwrapped Delays (µs)")
ax[k].plot(unwrapped_delays, "-k", label="Unwrapped Delays (µs)")
ax[k].hlines(
    0, 0, len(unwrapped_delays), colors="k", linestyles="--", label="Zero Line"
)
ax[k].set_xlabel("Element Index")
ax[k].set_ylabel("Delay (µs)")
ax[k].set_title("Unwrapped Delays")
ax[k].grid()

plt.tight_layout()
plt.show()

# ----------------- Compute the pressure field -----------------
Domino.set_apodization(apodization)
Domino.set_delays(delays * 1e-6)
# Domino.set_delays(unwrapped_delays * 1e-6)
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
    save_fig_name=figure_name,
)

# plotter = pysonogen.plot_pressure_field(
#     pr2.detach().cpu().numpy(),
#     x2.detach().cpu().numpy(),
#     y2.detach().cpu().numpy(),
#     z2.detach().cpu().numpy(),
# )

# plotter.show()
# plotter.deep_clean()  # Explicitly clean up PyVista objects
# plotter.close()

# # Matrix_field = pyfield.PyField(Zeus_Matrix)
# # pr1, x1, y1, z1 = Matrix_field.compute_pressure_field(field_info_mm, inplace=False)
# # pysonogen.plot_field_planes(pr1, x1, y1, z1, interpolation=None)
# del plotter
