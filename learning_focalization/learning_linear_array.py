import time

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

# from torch_test import TorchField, create_simulation_grid
import torch
from helper_function import (
    gaussian_1d,
    pattern_from_pr_3Dto2D,
    pattern_from_pr_3Dto3D,
    stack_2D_to_3D,
)
from TorchFieldv2 import TorchFieldv2 as TorchField

import pysonogen

print(torch.__version__)

use_cuda = True  # Set to False if you want to run on CPU
device_cpu = torch.device("cpu")
device_number = 0  # if you have multiple GPUs
if torch.cuda.is_available():
    print(f"GPU is available: cuda:{torch.cuda.get_device_name(device_number)}")
    device_cuda = torch.device(f"cuda:{device_number}")
else:
    print("Not GPU available.")
    device_cuda = device_cpu

device = device_cuda if use_cuda else device_cpu

# ----------------- compute pattern from pressure field -----------------
train = True  # Set to True to enable training mode
save_fig = True  # Set to True to save the model's state dictionary
version = "v1"  # Version of the model
num_epoch = 200
FoverD = 1  # Focalization over Diameter ratio
sigma = 0.7
batch_size = 2048  # Batch size for training
target = "lambda2"  # Target pattern to use
target_folder = r".\target_masks"
target_filename = f"/linear_{target}.npz"
destination = r".\test_models\linear"
name_model = (
    f"opt_{num_epoch}epochs_3DloglossE_1planes_noprocess_delay_apod_{target}_{version}"
)
state_name = f"Linear_torch_state_{name_model}"
path = destination + "/" + state_name


# ----------------- Load target pattern -----------------

target_dic = np.load(target_folder + target_filename)
target_matrix = target_dic["target"]
wavelength = target_dic["wavelength"]
x_length_mm = target_dic["x_length_mm"]
y_length_mm = target_dic["y_length_mm"]
dx = target_dic["dx"]
dy = target_dic["dy"]

print(f"Extent: {x_length_mm} mm x {y_length_mm} mm, dx={dx} mm, dy={dy} mm")


y_target2D = torch.tensor(target_matrix, dtype=torch.float32, device=device)

# ------------------- Transducer Matrix -------------------
z_plane_mm = 8  # mm
focus_mm = np.array([0, 0, z_plane_mm])  # mm [x, y, z]

field_matrix_mm = {
    "x_extent": [-x_length_mm / 2, x_length_mm / 2],  # mm (16,5 mm)
    "y_extent": [-y_length_mm / 2, y_length_mm / 2],  # mm(16,5 mm)
    "z_extent": [focus_mm[2], focus_mm[2]],  # mm (2 mm)
    "dx": dx,
    "dy": dy,
    "dz": 1,
}

linear_array_tx = pysonogen.transducers.Domino()
linear_array_tx.compute_apodization(focus_mm=focus_mm, F_over_D=FoverD)
delays = linear_array_tx.compute_delays(focus_mm=focus_mm)

# ------------------- Reference (focalization at depth) -------------------
# We first create how one focalization pattern would look like to have a reference
# of the wanted amplitude
linear_array_torch = TorchField(linear_array_tx, device=device_cuda)
pr, x, y, z = linear_array_torch(field_matrix_mm, batch_size=batch_size)

# We compute the Gaussian weights for the z-axis
nz = pr.shape[-1]  # Number of z points
z_weights = (
    gaussian_1d(nz, sigma=sigma, device=device, plot=False).unsqueeze(0).unsqueeze(0)
)


max_pr_plane0 = (
    (pr.to(device) * z_weights).sum(dim=-1).max().item()
)  # Sum along z-axis, and we take the max of the disk

# y_target2D = pattern_from_pr_3Dto2D(pr.to(device), max_pr_plane0)
max_pr0 = pr.max().item()  # Sum along z-axis, and we take the max of the disk

# ------------- Set initial delays and apodization -------------------

## Random apodization for testing
np.random.seed(42)  # For reproducibility
torch.manual_seed(42)  # For reproducibility
# delays = np.random.rand(linear_array_tx.n_elements) * np.max(delays)
# apodization = np.random.rand(
#     linear_array_tx.n_elements
# )

## Zeros delays and ones apodization for testing
delays = (
    np.zeros(linear_array_tx.n_elements) + np.max(delays) * 0.5
)  # Initial delays set to half the max delay
apodization = np.ones(linear_array_tx.n_elements)  # Random apodization for testing
linear_array_tx.set_delays(delays)
linear_array_tx.set_apodization(apodization)

# ------------------- Create Torch Field object -------------------

del linear_array_torch, x, y, z  # Clear previous instance if any
torch.cuda.empty_cache()  # Clear CUDA cache if using GPU
linear_array_torch = TorchField(linear_array_tx, device=device)

# ----------------- Define loss function and optimizer -----------------

# MSE loss function
loss_MSE = torch.nn.MSELoss()


def loss_energy(y_target_3D, PII, min_error=1e-6):
    """
    Custom loss function that computes the mean squared error between the target and predicted patterns.
    """
    # Compute the MSE loss
    E_focus = y_target_3D * PII

    E_sides = (1 - y_target_3D) * PII

    log_loss = (
        torch.log(E_sides.mean() + min_error)  # Minimize mean value outside the focus
        - torch.log(E_focus.mean() + min_error)  # Maximize Mean value inside the focus
    )
    return log_loss


# Initialize the optimizer
learning_rate_delays = 1e-3
learning_rate_apods = 1e-2

optimizer = torch.optim.Adam(
    [
        {"params": linear_array_torch.delays, "lr": learning_rate_delays},
        {"params": linear_array_torch.apodization, "lr": learning_rate_apods},
    ]
)

# ----------------- check forward of the model -----------------

# 2) Forward pass

pr, x, y, z = linear_array_torch(field_matrix_mm, batch_size=batch_size, training=False)

y_pred = pattern_from_pr_3Dto2D(pr.to(device), max_pr_plane0)

y_target3D = stack_2D_to_3D(y_target2D, nz=nz, sigma=0)

## ---------------- Compute first loss -----------------

print(f"Target3D shape: {y_target3D.shape}, Pressure shape: {pr.shape}")
print(f"Target2D shape: {y_target2D.shape}, Predicted pattern shape: {y_pred.shape}")

if y_pred.ndim > 2:
    first_prediction = pattern_from_pr_3Dto2D(pr.to(device), max_pr_plane0)
else:
    first_prediction = y_pred

apodization = linear_array_torch.apodization
delays = linear_array_torch.delays

for name, param in linear_array_torch.named_parameters():
    if param.requires_grad:
        print(name, param.shape, param[:10])

loss_physic = loss_energy(y_target3D, pr).item()
loss_comparison = loss_MSE(y_target2D, y_pred).item()

# Compute the ratio of the losses
# This ratio can be used to balance the contributions of the two losses
alpha = loss_physic / loss_comparison
print(
    f"Loss physic: {loss_physic:.4f}, Loss comparison: {loss_comparison:.4f}, ratio: {alpha:.4f}"
)

first_loss = loss_physic + alpha * loss_comparison  # + loss_delays + loss_apodization
first_prediction_np = first_prediction.detach().cpu().clone().numpy()
first_apod = apodization.detach().cpu().clone().numpy()
first_delays = delays.detach().cpu().clone().numpy()

linear_array_tx.plot_apodization(first_apod)
linear_array_tx.plot_delays(first_delays * 1e-6)


# ------------- Print and plot first prediction -------------

print(f"Max pressure focal: {max_pr0:.2f} units")

print(f"Max pressure init: {pr.max().item():.2f} units")
print(f"x shape: {x.shape}, y shape: {y.shape}, z shape: {z.shape}")
print(f"first loss: {first_loss:.4f} units")

plotter = pysonogen.plot_pressure_field(
    pr.detach().cpu().numpy(),
    x.detach().cpu().numpy(),
    y.detach().cpu().numpy(),
    z.detach().cpu().numpy(),
)
plotter.show()

plotter = pysonogen.plot_pressure_field(
    y_target3D.detach().cpu().numpy(),
    x.detach().cpu().numpy(),
    y.detach().cpu().numpy(),
    z.detach().cpu().numpy(),
)
plotter.show()

del plotter  # Clear the plotter to free memory

# Plot First Prediction, Last Prediction, and Target Pattern
fig, ax = plt.subplots(1, 2, figsize=(18, 6))
vmin = min(first_prediction_np.min(), target_matrix.min())
vmax = max(first_prediction_np.max(), target_matrix.max())
# First Prediction
im0 = ax[0].imshow(first_prediction_np.T, cmap="gray", vmin=vmin, vmax=vmax)
ax[0].set_title("First Prediction")
ax[0].axis("off")

# Target Pattern
im2 = ax[1].imshow(
    y_target2D.detach().cpu().numpy().T, cmap="gray", vmin=vmin, vmax=vmax
)
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
loss_energies_vec = np.zeros(num_epoch + 1)
target_loss_vec = np.zeros(num_epoch + 1)

pred_vect = np.zeros((num_epoch + 1, y_target2D.shape[0], y_target2D.shape[1]))
max_pr_vec = np.zeros(num_epoch + 1)
apod_vect = np.zeros((num_epoch + 1, linear_array_tx.n_elements))
delays_vect = np.zeros((num_epoch + 1, linear_array_tx.n_elements))

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
        print(f"Max pressure: {max_pr:.2f} units")
        if max_pr > max_pr0:
            print(f"Found a new max pressure at epoch {epoch}")
            max_pr0 = max_pr
            max_pr_plane0 = (pr * z_weights).sum(dim=-1).max().item()

        y_pred = pattern_from_pr_3Dto2D(pr, max_pr_plane0)

        # 3) Compute the loss
        loss_physic = loss_energy(y_target3D, pr)
        loss_comparison = loss_MSE(y_target2D, y_pred)
        # loss_delays = loss_smoothness_delays(linear_array_torch.delays)
        # loss_apodization = loss_smoothness_apodization(linear_array_torch.apodization)
        loss = alpha * loss_comparison + loss_physic  # + loss_delays + loss_apodization

        # 4) Backward pass
        loss.backward()

        # 5) Update the parameters
        optimizer.step()

        # 6) Store loss
        apod_vect[epoch] = linear_array_torch.apodization.detach().cpu().numpy()
        delays_vect[epoch] = linear_array_torch.delays.detach().cpu().numpy()
        loss_vec[epoch] = loss.item()
        loss_energies_vec[epoch] = loss_physic.item()
        target_loss_vec[epoch] = loss_comparison.item()
        pred_vect[epoch] = y_pred.detach().cpu().numpy()
        max_pr_vec[epoch] = max_pr
        print(
            f"loss: {loss_vec[epoch] / first_loss * 100:.4f} % relative to first loss."
        )
        print(
            f"loss = energy + alpha*target : {loss.item():.4f} = {loss_physic.item():.4f} + alpha*{loss_comparison.item():.4f}"
        )

    # Save the model's state dictionary
    torch.save(linear_array_torch.state_dict(), path + ".pth")
    print("Model state dictionary saved to: ", state_name)
    # Save other data
    np.savez(
        path + "_data.npz",
        apodization=apod_vect,
        delays=delays_vect,
        loss=loss_vec,
        first_loss=first_loss,
        max_pr0=max_pr0,
        max_pr=max_pr_vec,
    )


if y_pred.ndim > 2:
    last_prediction = pattern_from_pr_3Dto2D(pr.to(device), max_pr0)
else:
    last_prediction = y_pred


last_prediction_np = last_prediction.detach().cpu().numpy()
pr = pr.detach().cpu().numpy()
x = x.detach().cpu().numpy()
y = y.detach().cpu().numpy()
z = z.detach().cpu().numpy()

last_apod = linear_array_torch.apodization
last_apod = linear_array_torch._process_apodization(last_apod).detach().cpu().numpy()
last_delays = linear_array_torch.delays
last_delays = linear_array_torch._process_delays(last_delays).detach().cpu().numpy()
print(f"Apodization is {last_apod.min():.2f} to {last_apod.max():.2f} units.")
print(f"Delays are {last_delays.min():.2f} to {last_delays.max():.2f} microseconds.")

t1 = time.time()
print(f"Computation took {t1 - t0:.1f} seconds.")

# ----------------- Plot results -----------------


# ---------------- MAIN RESULTS PLOT -----------------
diff_apod = last_apod - first_apod
diff_delays = last_delays - first_delays

extent = [x.min(), x.max(), y.min(), y.max()]


plt.rcParams.update({"axes.titlesize": 12, "axes.labelsize": 10})
fig = plt.figure(figsize=(16, 10))
gs = gridspec.GridSpec(3, 4, figure=fig)

# Loss Plot spanning first row (3 columns)
ax_loss = fig.add_subplot(gs[0, :3])
ax_loss.plot(loss_vec, label="Loss")
ax_loss.plot(target_loss_vec * alpha, label="Target Loss")
ax_loss.plot(loss_energies_vec, label="Energy Loss")
ax_loss.set_xlabel("Epoch")
ax_loss.set_ylabel("Loss")
ax_loss.set_title("a) Training Loss")
ax_loss.grid()
ax_loss.legend()

# Start placing other subplots from here
subplot_index = 1  # Tracking from 1 now because loss is at position 0
axes = []

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

im2 = ax.plot(
    np.arange(linear_array_tx.n_elements),
    last_apod,
    "k-",
    marker="o",
    markerfacecolor="r",
)
ax.grid(True)
ax.set_title("c) Apodization (After)")
ax.set_xlabel("Element #")
ax.set_ylabel("Apodization")
axes.append(ax)

# Apodization Difference
ax = fig.add_subplot(gs[1, 2])

im3 = ax.plot(
    np.arange(linear_array_tx.n_elements),
    diff_apod,
    "k-",
    marker="o",
    markerfacecolor="b",
)
ax.grid(True)
ax.set_title("d) Apodization (Before)")
ax.set_xlabel("Element #")
ax.set_ylabel("Apodization")
axes.append(ax)

# Delays Before
ax = fig.add_subplot(gs[2, 0])
im4 = ax.plot(
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
im5 = ax.plot(
    np.arange(linear_array_tx.n_elements),
    last_delays,
    "k-",
    marker="o",
    markerfacecolor="r",
)
ax.set_title("f) Delays (After)")
ax.set_xlabel("Element #")
ax.set_ylabel("Delay (us)")
ax.grid(True)
axes.append(ax)

# Delays Difference
ax = fig.add_subplot(gs[2, 2])
im6 = ax.plot(
    np.arange(linear_array_tx.n_elements),
    diff_delays,
    "k-",
    marker="o",
    markerfacecolor="b",
)
ax.set_title("g) Delays difference")
ax.set_xlabel("Element #")
ax.set_ylabel("Delay (us)")
ax.grid(True)
axes.append(ax)

# First Prediction
ax = fig.add_subplot(gs[0, 3])
im7 = ax.imshow(first_prediction_np.T, cmap="gray", extent=extent, vmin=0, vmax=1)
ax.set_title("h) First Prediction")
ax.set_xlabel("X (mm)")
ax.set_ylabel("Y (mm)")
axes.append(ax)

# Last Prediction
ax = fig.add_subplot(gs[1, 3])
im8 = ax.imshow(last_prediction_np.T, cmap="gray", extent=extent, vmin=0, vmax=1)
ax.set_title("i) Last Prediction")
ax.set_xlabel("X (mm)")
ax.set_ylabel("Y (mm)")
axes.append(ax)

# Target Pattern
ax = fig.add_subplot(gs[2, 3])
im9 = ax.imshow(
    y_target2D.detach().cpu().numpy().T, cmap="gray", vmin=0, vmax=1, extent=extent
)
ax.set_title("j) Target Pattern")
ax.set_xlabel("X (mm)")
ax.set_ylabel("Y (mm)")
axes.append(ax)

# Add a single colorbar for all imshow subplots
cbar_ax = fig.add_axes([0.92, 0.3, 0.02, 0.4])  # [left, bottom, width, height]
cbar = fig.colorbar(im9, cax=cbar_ax)
cbar.set_label("Intensity")
# Final manual layout adjustments
fig.subplots_adjust(left=0.05, right=0.9, top=0.95, bottom=0.05, hspace=0.6, wspace=0.4)


if save_fig:
    plt.savefig(
        destination + "/" + f"summary_{state_name}.png", dpi=300, bbox_inches="tight"
    )
plt.show()


# ---------------- PARAMS VS EPOCHS -----------------

fig, axes = plt.subplots(1, 2, figsize=(18, 6), constrained_layout=True)

# Colormap for epochs
cmap = plt.cm.jet
norm = plt.Normalize(vmin=0, vmax=num_epoch)

# Plot apodization (left)
for epoch in range(num_epoch + 1):
    color = cmap(norm(epoch))
    label = f"Epoch {epoch}" if epoch % max(1, (num_epoch // 10)) == 0 else None
    axes[0].plot(
        np.arange(linear_array_tx.n_elements),
        apod_vect[epoch],
        color=color,
        label=label,
        linewidth=1,
    )

axes[0].set_title("Apodization Across Epochs")
axes[0].set_xlabel("Element #")
axes[0].set_ylabel("Apodization")
axes[0].grid(True)
axes[0].set_xlim([0, linear_array_tx.n_elements - 1])

# Plot delays (right)
for epoch in range(num_epoch + 1):
    color = cmap(norm(epoch))
    label = f"Epoch {epoch}" if epoch % max(1, (num_epoch // 10)) == 0 else None
    axes[1].plot(
        np.arange(linear_array_tx.n_elements),
        delays_vect[epoch],
        color=color,
        label=label,
        linewidth=1,
    )

axes[1].set_title("Delays Across Epochs")
axes[1].set_xlabel("Element #")
axes[1].set_ylabel("Delay (μs)")
axes[1].grid(True)
axes[1].set_xlim([0, linear_array_tx.n_elements - 1])

# Add colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=axes, orientation="horizontal", pad=0.08, aspect=40)
cbar.set_label("Epochs")

if save_fig:
    # Save the figur
    plt.savefig(
        destination + "/" + f"params_vs_epochs_{state_name}.png",
        dpi=300,
        bbox_inches="tight",
    )
plt.show()
plt.close()

# ----------------- Plot the final pressure field -----------------

# plotter = pysonogen.plot_pressure_field(
#     pr,
#     x,
#     y,
#     z,
# )

# plotter.show()

# del plotter  # Delete the plotter object
