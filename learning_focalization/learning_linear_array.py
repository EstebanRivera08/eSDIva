import time

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

# from torch_test import TorchField, create_simulation_grid
import torch
from TorchFieldv2 import TorchFieldv2 as TorchField

import pysonogen

print(torch.__version__)

use_cuda = False  # Set to False if you want to run on CPU
device_number = 0  # if you have multiple GPUs
if torch.cuda.is_available():
    print(f"GPU is available: cuda:{torch.cuda.get_device_name(device_number)}")
    device_cuda = torch.device(f"cuda:{device_number}")
    device_cpu = torch.device("cpu")
else:
    print("Not GPU available.")
    device_cpu = torch.device("cpu")

device = device_cuda if use_cuda else device_cpu

# ----------------- compute pattern from pressure field -----------------
train = True  # Set to True to enable training mode
version = "v1"  # Version of the model
num_epoch = 10
FoverD = 2  # Focalization over Diameter ratio

batch_size = 4096  # Batch size for training
target_folder = r".\target_masks"
target_filename = r"/linear_4lambda.npz"
destination = r".\test_models\linear"
name_model = f"opt_{num_epoch}epochs_{version}"
state_name = f"Linear_torch_state_{name_model}.pth"
path = destination + "/" + state_name

z_len = 5  # Length of the z-axis
z_weights = torch.linspace(-1, 1, z_len, device=device)  # Linear range from -1 to 1
z_weights = torch.exp(-(z_weights**2) / (2 * 0.5**2))  # Gaussian weights (sigma = 0.5)
z_weights = z_weights / z_weights.max()  # Normalize weights to sum to 1

plt.plot(z_weights.cpu().numpy(), "k-s", label="z_weights")
plt.xlabel("z index")
plt.ylabel("Weight")
plt.grid()
plt.show()
# z_weights = z_weights / z_weights.max()  # Normalize weights to sum to 1


def pattern_from_pressure_field(pressure, max):
    # Apply the weights to the pressure tensor along the z-axis
    pressure_disk = (pressure * z_weights).sum(dim=-1)  # Weighted sum along the z-axis
    pressure_disk = pressure_disk / max  # Normalize

    # Use a differentiable thresholding operation
    focal_mask = torch.sigmoid(10 * (pressure_disk - 0.5))
    return focal_mask


# ----------------- Load target pattern -----------------

target_dic = np.load(target_folder + target_filename)
target_matrix = target_dic["target"]
wavelength = target_dic["wavelength"]
x_length_mm = target_dic["x_length_mm"]
y_length_mm = target_dic["y_length_mm"]
dx = target_dic["dx"]
dy = target_dic["dy"]

y_target = torch.tensor(target_matrix, dtype=torch.float32, device=device)


# ------------------- Transducer Matrix -------------------
focus_mm = np.array([0, 0, 8])  # mm [x, y, z]

field_matrix_mm = {
    "x_extent": [-x_length_mm / 2, x_length_mm / 2],  # mm (16,5 mm)
    "y_extent": [-y_length_mm / 2, y_length_mm / 2],  # mm(16,5 mm)
    "z_extent": [focus_mm[2] - 1, focus_mm[2] + 1],  # mm (2 mm)
    "dx": dx,
    "dy": dy,
    "dz": 0.5,
}

linear_array_tx = pysonogen.transducers.Domino()

linear_array_tx.compute_apodization(focus_mm=focus_mm, F_over_D=FoverD)
linear_array_tx.compute_delays(focus_mm=focus_mm)

# We first create how one focalization pattern would look like to have a reference
# of the wanted amplitude
linear_array_torch = TorchField(linear_array_tx, device=device_cuda)
pr, x, y, z = linear_array_torch(field_matrix_mm, batch_size=batch_size)
max_pr0 = (
    (pr * z_weights.to(device_cuda)).sum(dim=-1).max().item()
)  # Sum along z-axis, and we take the max of the disk
print(f"Max pressure: {max_pr0:.2f} units")

# We then create random delays and apodization to test the learning
# We will use these random values to initialize the TorchField object
# and then optimize them to match the target pattern

## We create it on the same scale as the computed delays
# np.random.seed(42)  # For reproducibility
# torch.manual_seed(42)  # For reproducibility
# delays = np.random.rand(linear_array_tx.n_elements) * np.max(delays)
# apodization = np.random.rand(linear_array_tx.n_elements)  # Random apodization for testing
delays = np.zeros(linear_array_tx.n_elements)
apodization = np.ones(linear_array_tx.n_elements)  # Random apodization for testing
linear_array_tx.set_delays(delays)
linear_array_tx.set_apodization(apodization)
linear_array_tx.plot_apodization()
linear_array_tx.plot_delays()

# ------------------- Create Torch Field object -------------------
del linear_array_torch, x, y, z  # Clear previous instance if any
torch.cuda.empty_cache()  # Clear CUDA cache if using GPU
linear_array_torch = TorchField(linear_array_tx, device=device)
# Load the model's state dictionary
# linear_array_torch.load_state_dict(torch.load("linear_array_torch_state.pth"))
# print("Model state dictionary loaded from 'linear_array_torch_state.pth'")

# ----------------- Define loss function and optimizer -----------------

# MSE loss function
loss_fn = torch.nn.MSELoss()

# Initialize the optimizer
learning_rate_delays = 2e-2
learning_rate_apods = 2e-1

optimizer = torch.optim.Adam(
    [
        {"params": linear_array_torch.delays, "lr": learning_rate_delays},
        {"params": linear_array_torch.apodization, "lr": learning_rate_apods},
    ]
)

# ----------------- First check of the functions -----------------

# 2) Forward pass

for name, param in linear_array_torch.named_parameters():
    if param.requires_grad:
        print(name, param.shape, param[:10])
pr, x, y, z = linear_array_torch(field_matrix_mm, batch_size=2048, training=True)
print(f"Max pressure: {(pr.to(device) * z_weights).sum(dim=-1).max().item():.2f} units")
y_pred = pattern_from_pressure_field(pr.to(device), max_pr0)

first_loss = loss_fn(y_target, y_pred).item()
first_prediction = y_pred.detach().cpu().numpy()
first_apod = linear_array_torch.apodization.detach().cpu().numpy()
first_delays = (
    linear_array_torch.delays.detach().cpu().numpy()
)  # Convert to microseconds for better visualization

# Plot First Prediction, Last Prediction, and Target Pattern
fig, ax = plt.subplots(1, 2, figsize=(18, 6))
vmin = min(first_prediction.min(), target_matrix.min())
vmax = max(first_prediction.max(), target_matrix.max())
# First Prediction
im0 = ax[0].imshow(first_prediction, cmap="gray", vmin=vmin, vmax=vmax)
ax[0].set_title("First Prediction")
ax[0].axis("off")

# Target Pattern
im2 = ax[1].imshow(target_matrix, cmap="gray", vmin=vmin, vmax=vmax)
ax[1].set_title("Target Pattern")
ax[1].axis("off")

# Add colorbar to the last subplot
plt.colorbar(im2, ax=ax[1], orientation="vertical", fraction=0.046, pad=0.04)

# Adjust layout
plt.tight_layout()
plt.suptitle("Predictions and Target Pattern")
plt.show()
plt.close()

# ----------------- Training loop -----------------
t0 = time.time()
loss_vec = np.zeros(num_epoch + 1)
torch.autograd.set_detect_anomaly(True)

if train:
    for epoch in range(num_epoch + 1):
        print(f"Epoch {epoch + 1}/{num_epoch + 1}")
        # 1) Zero the gradients
        optimizer.zero_grad()

        # 2) Forward pass
        pr, x, y, z = linear_array_torch(
            field_matrix_mm, batch_size=batch_size, training=True
        )
        max_pr = pr.max().item()
        if max_pr > max_pr0:
            max_pr0 = max_pr

        y_pred = pattern_from_pressure_field(pr, max_pr0)

        # 3) Compute the loss
        loss = loss_fn(y_target, y_pred)

        # 4) Backward pass
        loss.backward()

        # 5) Update the parameters
        optimizer.step()

        # 6) Store loss
        loss_vec[epoch] = loss.item()
        print(
            f"loss: {loss_vec[epoch] / first_loss * 100:.4f} % relative to first loss."
        )

    # Save the model's state dictionary
    torch.save(linear_array_torch.state_dict(), path)
    print("Model state dictionary saved to: ", state_name)

last_prediction = y_pred.detach().cpu().numpy()
last_apod = linear_array_torch.apodization.detach().cpu().numpy()
last_delays = linear_array_torch.delays.detach().cpu().numpy()
print(f"Apodization is {last_apod.min():.2f} to {last_apod.max():.2f} units.")
print(f"Delays are {last_delays.min():.2f} to {last_delays.max():.2f} microseconds.")

t1 = time.time()
print(f"Computation took {t1 - t0:.1f} seconds.")

# ----------------- Plot results -----------------
diff_apod = last_apod - first_apod
diff_delays = last_delays - first_delays

extent = [x.min(), x.max(), y.min(), y.max()]


plt.rcParams.update({"axes.titlesize": 12, "axes.labelsize": 10})
fig = plt.figure(figsize=(17, 12), constrained_layout=True)
gs = gridspec.GridSpec(3, 4, figure=fig)

# Loss Plot spanning first row (3 columns)
ax_loss = fig.add_subplot(gs[0, :3])
ax_loss.plot(loss_vec / first_loss * 100, label="Loss")
ax_loss.set_xlabel("Epoch")
ax_loss.set_ylabel("Loss (%)")
ax_loss.set_title("a) Training Loss")
ax_loss.grid()
ax_loss.legend()

# Start placing other subplots from here
subplot_index = 1  # Tracking from 1 now because loss is at position 0
axes = []

# First Prediction
ax = fig.add_subplot(gs[0, 3])
im6 = ax.imshow(first_prediction.T, cmap="gray", extent=extent, vmin=0, vmax=1)
ax.set_title("h) First Prediction")
ax.set_xlabel("X (mm)")
ax.set_ylabel("Y (mm)")
axes.append(ax)

# Apodization Before
ax = fig.add_subplot(gs[1, 0])

im1 = ax.plot(
    np.arange(linear_array_tx.n_elements),
    first_apod,
    "k-",
    marker="o",
    markerfacecolor="r",
)
ax.grid(True)
ax.set_title("b) Apodization (Before)")
ax.set_xlabel("Element #")
ax.set_ylabel("Apodization")
axes.append(ax)

# Apodization After
ax = fig.add_subplot(gs[1, 1])

im1 = ax.plot(
    np.arange(linear_array_tx.n_elements),
    last_apod,
    "k-",
    marker="o",
    markerfacecolor="r",
)
ax.grid(True)
ax.set_title("b) Apodization (Before)")
ax.set_xlabel("Element #")
ax.set_ylabel("Apodization")
axes.append(ax)

# Apodization Difference
ax = fig.add_subplot(gs[1, 2])

im1 = ax.plot(
    np.arange(linear_array_tx.n_elements),
    diff_apod,
    "k-",
    marker="o",
    markerfacecolor="b",
)
ax.grid(True)
ax.set_title("b) Apodization (Before)")
ax.set_xlabel("Element #")
ax.set_ylabel("Apodization")
axes.append(ax)


# Last Prediction
ax = fig.add_subplot(gs[1, 3])
im7 = ax.imshow(last_prediction.T, cmap="gray", extent=extent, vmin=0, vmax=1)
ax.set_title("i) Last Prediction")
ax.set_xlabel("X (mm)")
ax.set_ylabel("Y (mm)")
axes.append(ax)

# Delays Before
ax = fig.add_subplot(gs[2, 0])
im3 = ax.plot(
    np.arange(linear_array_tx.n_elements),
    first_delays,
    "k-",
    marker="o",
    markerfacecolor="r",
)
ax.set_title("e) Delays (Before)")
ax.set_xlabel("Element #")
ax.set_ylabel("Delay (us)")
ax.grid(True)
axes.append(ax)

# Delays After
ax = fig.add_subplot(gs[2, 1])
im4 = ax.plot(
    np.arange(linear_array_tx.n_elements),
    last_delays,
    "k-",
    marker="o",
    markerfacecolor="r",
)
ax.set_title("e) Delays (Before)")
ax.set_xlabel("Element #")
ax.set_ylabel("Delay (us)")
ax.grid(True)
axes.append(ax)

# Delays Difference
ax = fig.add_subplot(gs[2, 2])
im4 = ax.plot(
    np.arange(linear_array_tx.n_elements),
    diff_delays,
    "k-",
    marker="o",
    markerfacecolor="r",
)
ax.set_title("e) Delays difference")
ax.set_xlabel("Element #")
ax.set_ylabel("Delay (us)")
ax.grid(True)
axes.append(ax)

# Target Pattern
ax = fig.add_subplot(gs[2, 3])
ax.imshow(target_matrix.T, cmap="gray", vmin=0, vmax=1, extent=extent)
ax.set_title("j) Target Pattern")
ax.set_xlabel("X (mm)")
ax.set_ylabel("Y (mm)")
axes.append(ax)

# plt.tight_layout()
# fig.subplots_adjust(top=0.92, hspace=0.6, wspace=0.4)
plt.savefig(f"summary_{state_name}.png", dpi=300, bbox_inches="tight")
plt.show()

plotter = pysonogen.plot_pressure_field(
    pr.detach().cpu().numpy(),
    x.detach().cpu().numpy(),
    y.detach().cpu().numpy(),
    z.detach().cpu().numpy(),
)

plotter.show()
