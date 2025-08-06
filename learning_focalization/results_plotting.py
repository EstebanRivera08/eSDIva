import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from TorchFieldv2 import TorchFieldv2 as TorchField

import pysonogen.transducers as Transducers

use_cuda = True  # Set to False if you want to run on CPU
device_number = 0  # if you have multiple GPUs
if torch.cuda.is_available() and use_cuda:
    print(f"Using GPU: {torch.cuda.get_device_name(device_number)}")
    device = torch.device(f"cuda:{device_number}")
else:
    print("No GPU available, running on CPU. May be slow.")
    device = torch.device("cpu")

print(Transducers.available_transducers())


# Domino.show()

# ----------------------------
version = "v3"
num_epoch = 200

target_folder = r".\target_masks"
target_filename = r"/linear_4lambda.npz"
destination = r".\test_models\linear"

name_model1 = (
    f"opt_{num_epoch}epochs_3DloglossE_1planes_noprocess_half_target2_{version}"
)

name_model2 = (
    f"opt_{num_epoch}epochs_3DloglossE_1planes_noprocess_zeros_target2_{version}"
)


torch.cuda.empty_cache()
Domino = Transducers.Domino()
focus_mm = np.array([0, 0, 8])  # mm [x, y, z]

delays0 = Domino.compute_delays(focus_mm=focus_mm, plot=True)

delays0 = (delays0 - delays0.min()) * 1e6  # Normalize delays to start from zero

domino_torch = TorchField(Domino, device=device)


# Zeus_Matrix.show()
def get_delays_from_file(name_model):
    state_name = f"Linear_torch_state_{name_model}"
    path = destination + "/" + state_name + ".pth"  # Add the correct extension
    checkpoint = torch.load(path)
    domino_torch.load_state_dict(checkpoint)
    apodization = domino_torch.apodization.detach().cpu().numpy()
    delays = domino_torch.delays.detach().cpu().numpy()
    return delays, apodization


delays1, apodization1 = get_delays_from_file(name_model1)
delays2, apodization2 = get_delays_from_file(name_model2)

delays1 = delays1 - delays1.min()
delays2 = delays2 - delays2.min()

# -------------- Plotting the delays for comparison --------------

fig, ax = plt.subplots(1, 2, figsize=(12, 6))

ax[0].plot(delays1, "r-", label="Model 1 Delays")
ax[0].plot(delays2, "b-.", label="Model 2 Delays")
ax[0].set_title("Delays Comparison")
ax[0].set_xlabel("Element Index")
ax[0].set_ylabel("Delay (us)")
ax[0].grid()
ax[0].legend()
ax[1].plot(delays2 - delays1, "k", label="difference (Model 2 - Model 1)")
# ax[1].plot(delays2 - delays1, "ks", label="difference (Model 2 - Model 1)")
ax[1].set_title("Delays Comparison")
ax[1].set_xlabel("Element Index")
ax[1].set_ylabel("Delay (us)")
ax[1].grid()
ax[1].legend()

plt.tight_layout()
plt.show()

# ------------- Compare to parabolic delays --------------

dif_delays1 = np.diff(delays1)
dif_delays0 = np.diff(delays0)

delays10 = delays0 - delays1
dif_delays10 = np.diff(delays10)

fig, ax = plt.subplots(1, 2, figsize=(8, 4))
ax[0].plot(delays0, "k-", label="Expected Delays")
ax[0].plot(delays1, "r-", label="Model 1 Delays")
ax[0].plot(delays10, "b-", label="Expected - Model 1 Delays")
ax[0].set_title("a) Delays")
ax[0].set_xlabel("Element Index")
ax[0].set_ylabel("Delay (us)")
ax[0].grid()
ax[0].legend()


linewidth = 1
ax[1].plot(dif_delays0, "k-", label="Expected Delay derivative")
ax[1].plot(dif_delays1, "r-", linewidth=linewidth, label="Model 1 Delay derivative")
ax[1].plot(dif_delays10, "b-", linewidth=linewidth, label="Diff derivative")
ax[1].set_title("b) Delay Derivative")
ax[1].set_xlabel("Element Index")
ax[1].set_ylabel("Delay derivative (us)")
ax[1].grid()
# ax[1].legend()
plt.tight_layout()
plt.show()


# ---------- unwrap delays ----------

unwrapped_delays = np.unwrap(delays1, period=0.08)
unwrapped_delays = np.unwrap(unwrapped_delays, period=0.071)
unwrapped_delays = np.unwrap(unwrapped_delays, period=0.065)
delays3 = unwrapped_delays
delays30 = delays0 - delays3
dif_delays3 = np.diff(delays3)
dif_delays30 = np.diff(delays30)

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(delays0, "k-", label="Expected Delays")
ax[0].plot(delays3, "r-", label="unwrapped Model 1 Delays")
ax[0].plot(delays0 - delays3, "b-", label="Expected - unwrapped Model 1 Delays")
ax[0].set_title("a) Delays")
ax[0].set_xlabel("Element Index")
ax[0].set_ylabel("Delay (us)")
ax[0].grid()
ax[0].legend()


linewidth = 1
ax[1].plot(dif_delays0, "k-", label="Expected Delay derivative")
ax[1].plot(
    dif_delays3, "r-", linewidth=linewidth, label="unwrapped Model 1 Delay derivative"
)
ax[1].plot(dif_delays30, "b-", linewidth=linewidth, label="Diff derivative")
ax[1].set_title("b) Delay Derivative")
ax[1].set_xlabel("Element Index")
ax[1].set_ylabel("Delay derivative (us)")
ax[1].grid()
# ax[1].legend()
plt.tight_layout()
plt.show()


pitch = Domino.kerf + Domino.el_w  # in m
zf = 8 * 1e-3  # focal depth in meters

x_width = 300
element = np.arange(-x_width, x_width)  # element indices
r0 = np.sqrt(zf**2 + (element * pitch) ** 2)
r1 = np.sqrt(zf**2 + ((element + 1) * pitch) ** 2)
max_delta_tau = pitch / 1540 * 1e6  # in microseconds
delta_tau = (r0 - r1) / 1540 * 1e6  # in microseconds

element = element + Domino.n_elements / 2  # shift to positive indices
print(f"max_delta_tau: {max_delta_tau} us")

unwrapped_delays = np.unwrap(delays1, period=max_delta_tau)
delays4 = unwrapped_delays
delays40 = delays0 - delays4
dif_delays4 = np.diff(delays4)
dif_delays40 = np.diff(delays40)

fig, ax = plt.subplots(1, 3, figsize=(15, 5))

ax[0].plot(delays0, "k-", label="Expected Delays")
ax[0].plot(delays4, "r-", label="unwrapped Delays")
ax[0].plot(delays40, "b-", label="Expected - unwrapped Delays")
ax[0].set_title("a) Delays")
ax[0].set_xlabel("Element Index")
ax[0].set_ylabel("Delay (us)")
ax[0].grid()
ax[0].legend()

linewidth = 1

ax[1].plot(
    element,
    delta_tau * 1e3,
    "k--",
    label="theoretical $\\Delta \\tau$",
)
ax[1].plot(dif_delays0 * 1e3, "k-", label="_nolegend_")
ax[1].plot(dif_delays4 * 1e3, "r-", linewidth=linewidth, label="_nolegend_")
ax[1].plot(
    dif_delays40 * 1e3,
    "b-",
    linewidth=linewidth,
    label="_nolegend_",
)
ax[1].axhline(max_delta_tau * 1e3, color="g", linestyle="-", label="$p_x/c$")
ax[1].axhline(
    -max_delta_tau * 1e3,
    color="g",
    linestyle="-",
    label="$-p_x/c$",
)
ax[1].set_title("b) Delay Derivative")
ax[1].set_xlabel("Element Index")
ax[1].set_ylabel("Delay derivative (ns)")
ax[1].grid()
ax[1].set_xlim([-10, 138])  # Adjusted to fit the element range

linewidth = 0.3
ax[2].plot(
    element,
    delta_tau * 1e3,
    "k--",
    label="theoretical $\\Delta \\tau$",
)
ax[2].plot(dif_delays0 * 1e3, "k-", label="_nolegend_")
ax[2].plot(dif_delays4 * 1e3, "r-", linewidth=linewidth, label="_nolegend_")
ax[2].plot(
    dif_delays40 * 1e3,
    "b-",
    linewidth=linewidth,
    label="_nolegend_",
)
ax[2].axhline(max_delta_tau * 1e3, color="g", linestyle="-", label="$p_x/c$")
ax[2].axhline(
    -max_delta_tau * 1e3,
    color="g",
    linestyle="-",
    label="$-p_x/c$",
)
ax[2].set_title("c) Delay Derivative (Zoomed out)")
ax[2].set_xlabel("Element Index")
ax[2].set_ylabel("Delay derivative (ns)")
ax[2].grid()
ax[2].legend()
ax[2].set_xlim([element.min(), element.max()])
plt.tight_layout()
plt.show()

# ---------- see target mask ----------

# target_folder = r".\target_masks"
# target_filename = r"/linear_lambda2.npz"


# target_dic = np.load(target_folder + target_filename)
# target_matrix = target_dic["target"]
# wavelength = target_dic["wavelength"]
# x_length_mm = target_dic["x_length_mm"]
# y_length_mm = target_dic["y_length_mm"]
# dx = target_dic["dx"]
# dy = target_dic["dy"]

# extent = [-x_length_mm / 2, x_length_mm / 2, -y_length_mm / 2, y_length_mm / 2]
# plt.imshow(target_matrix.T, cmap="gray", interpolation=None, extent=extent)
# plt.title("Target Mask")
# plt.colorbar()
# plt.xlabel("X-axis (mm)")
# plt.ylabel("Y-axis (mm)")
# plt.show()
