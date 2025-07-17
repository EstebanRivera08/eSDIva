import time

import matplotlib.pyplot as plt
import numpy as np

# from torch_test import TorchField, create_simulation_grid
import torch
from PyFieldTorch import PyFieldTorch
from TorchField import TorchField
from tqdm import tqdm

import pysonogen
from pysonogen import pyfield

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
def pattern_from_pressure_field(pressure, max):
    # Normalize pressure to ensure differentiability
    pressure_disk = pressure.sum(dim=-1)  # Sum along the last dimension
    pressure_disk = pressure_disk / max  # Normalize

    # Use a differentiable thresholding operation
    focal_mask = torch.sigmoid(10 * (pressure_disk - 0.5))
    return focal_mask


# ------------------- Transducer Matrix -------------------
focus_mm = np.array([0, 0, 5])  # mm [x, y, z]
F_over_D = 1

field_matrix_mm = {
    "x_extent": [-55 * 0.3 / 2, 55 * 0.3 / 2],
    "y_extent": [-55 * 0.3 / 2, 55 * 0.3 / 2],
    "z_extent": [focus_mm[2] - 1, focus_mm[2] + 1],
    "dx": 0.3,
    "dy": 0.3,
    "dz": 0.5,
}

plot_figures = False
Zeus_Matrix = pysonogen.transducers.Zeus_Matrix()
FoverD = 2.0
delays = Zeus_Matrix.compute_delays(focus_mm=focus_mm, plot=plot_figures)
apodization = Zeus_Matrix.compute_apodization(
    focus_mm=focus_mm, F_over_D=FoverD, apodization_type="circular", plot=plot_figures
)

# We first create how one focalization pattern would look like to have a reference
# of the wanted amplitude
Matrix_torch = TorchField(Zeus_Matrix, device=device_cuda)
pr, x, y, z = Matrix_torch(field_matrix_mm, batch_size=1024)
max_pr0 = (
    pr.sum(dim=-1).max().item()
)  # Sum along z-axis, and we take the max of the disk
print(f"Max pressure: {max_pr0:.2f} units")

# We then create random delays and apodization to test the learning
# We will use these random values to initialize the TorchField object
# and then optimize them to match the target pattern
np.random.seed(42)  # For reproducibility
torch.manual_seed(42)  # For reproducibility

# delays = np.random.rand(Zeus_Matrix.n_elements) * np.max(
#     delays
# )  # We create it on the same scale as the computed delays
# # apodization = np.random.rand(Zeus_Matrix.n_elements)  # Random apodization for testing
apodization = np.ones(Zeus_Matrix.n_elements)  # Random apodization for testing
Zeus_Matrix.set_delays(delays)
Zeus_Matrix.set_apodization(apodization)
Zeus_Matrix.plot_apodization()
Zeus_Matrix.plot_delays()

# ------------------- Create Torch Field object -------------------
del Matrix_torch, pr, x, y, z  # Clear previous instance if any
torch.cuda.empty_cache()  # Clear CUDA cache if using GPU
Matrix_torch = TorchField(Zeus_Matrix, device=device)

# ----------------- Load target pattern -----------------
folder = r"..\pressure_fields"
filename = r"/target_inverse.npz"

target_matrix = np.load(folder + filename)["target"]
y_target = torch.tensor(target_matrix, dtype=torch.float32, device=device)

# ----------------- Define loss function and optimizer -----------------
# Decide number of iterations and learning rate
num_epoch = 3

# MSE loss function
loss_fn = torch.nn.MSELoss()

# Initialize the optimizer
learning_rate_delays = 1e-5

for name, param in Matrix_torch.named_parameters():
    if param.requires_grad:
        print(name, param.shape)

optimizer = torch.optim.Adam(
    [
        {"params": Matrix_torch.delays, "lr": learning_rate_delays},
        # {'params': model_index_distortion.dn_param, 'lr': learning_rate_coeff}
    ]
)

# ----------------- First cheack of the functions -----------------

# 2) Forward pass
pr, x, y, z = Matrix_torch(field_matrix_mm, batch_size=512, training=True)
print(f"Max pressure: {pr.sum(dim=-1).max().item():.2f} units")
y_pred = pattern_from_pressure_field(pr, max_pr0)

first_loss = loss_fn(y_target, y_pred).item()
first_prediction = y_pred.detach().cpu().numpy()
first_apod = apodization
first_delays = delays


fig, ax = plt.subplots(1, 2, figsize=(15, 5))
ax[0].imshow(first_prediction, cmap="gray", vmin=0, vmax=1)
ax[0].set_title("First Prediction")
ax[0].axis("off")
ax[1].imshow(target_matrix, cmap="gray", vmin=0, vmax=1)
ax[1].set_title("Target Pattern")
ax[1].axis("off")
plt.tight_layout()
plt.colorbar(
    ax[1].images[0], ax=ax[1], orientation="vertical", fraction=0.046, pad=0.04
)
plt.suptitle("Initial Predictions vs Target Pattern")
plt.show()
plt.close()

# ----------------- Training loop -----------------
t0 = time.time()
loss_vec = np.zeros(num_epoch + 1)

for epoch in range(num_epoch + 1):
    # 1) Zero the gradients
    optimizer.zero_grad()

    # 2) Forward pass
    pr, x, y, z = Matrix_torch(field_matrix_mm, batch_size=512, training=True)
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
        f"Epoch {epoch + 1}/{num_epoch + 1}, with loss: {loss_vec[epoch] / first_loss * 100:.4f} % relative to first loss."
    )


last_prediction = y_pred.detach().cpu().numpy()
last_apod = Matrix_torch.apods.detach().cpu().numpy()
last_delays = Matrix_torch.delays.detach().cpu().numpy()

t1 = time.time()
print(f"Computation took {t1 - t0:.1f} seconds.")

# ----------------- Plot results -----------------

plt.figure(figsize=(10, 5))
plt.plot(loss_vec, label="Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.grid()
plt.legend()
plt.show()

diff_apod = last_apod - first_apod
diff_delays = last_delays - first_delays
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Before Apodization
im0 = axes[0, 0].imshow(
    first_apod.reshape((Zeus_Matrix.n_elem_x, Zeus_Matrix.n_elem_y)),
    cmap="cool",
    vmin=0,
    vmax=1,
)
axes[0, 0].set_title("Apodization (Before)")
axes[0, 0].set_xlabel("Element X")
axes[0, 0].set_ylabel("Element Y")
plt.colorbar(im0, ax=axes[0, 0], orientation="vertical", fraction=0.046, pad=0.04)

# After Apodization
im1 = axes[0, 1].imshow(
    last_apod.reshape((Zeus_Matrix.n_elem_x, Zeus_Matrix.n_elem_y)),
    cmap="cool",
    vmin=0,
    vmax=1,
)
axes[0, 1].set_title("Apodization (After)")
axes[0, 1].set_xlabel("Element X")
axes[0, 1].set_ylabel("Element Y")
plt.colorbar(im1, ax=axes[0, 1], orientation="vertical", fraction=0.046, pad=0.04)

# Difference in Apodization
im2 = axes[0, 2].imshow(
    diff_apod.reshape((Zeus_Matrix.n_elem_x, Zeus_Matrix.n_elem_y)),
    cmap="cool",
)
axes[0, 2].set_title("Apodization (Difference)")
axes[0, 2].set_xlabel("Element X")
axes[0, 2].set_ylabel("Element Y")
plt.colorbar(im2, ax=axes[0, 2], orientation="vertical", fraction=0.046, pad=0.04)

# Before Delays
im3 = axes[1, 0].imshow(
    first_delays.reshape((Zeus_Matrix.n_elem_x, Zeus_Matrix.n_elem_y)),
    cmap="jet",
)
axes[1, 0].set_title("Delays (Before)")
axes[1, 0].set_xlabel("Element X")
axes[1, 0].set_ylabel("Element Y")
plt.colorbar(im3, ax=axes[1, 0], orientation="vertical", fraction=0.046, pad=0.04)

# After Delays
im4 = axes[1, 1].imshow(
    last_delays.reshape((Zeus_Matrix.n_elem_x, Zeus_Matrix.n_elem_y)),
    cmap="jet",
)
axes[1, 1].set_title("Delays (After)")
axes[1, 1].set_xlabel("Element X")
axes[1, 1].set_ylabel("Element Y")
plt.colorbar(im4, ax=axes[1, 1], orientation="vertical", fraction=0.046, pad=0.04)

# Difference in Delays
im5 = axes[1, 2].imshow(
    diff_delays.reshape((Zeus_Matrix.n_elem_x, Zeus_Matrix.n_elem_y)),
    cmap="jet",
)
axes[1, 2].set_title("Delays (Difference)")
axes[1, 2].set_xlabel("Element X")
axes[1, 2].set_ylabel("Element Y")
plt.colorbar(im5, ax=axes[1, 2], orientation="vertical", fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()

fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].imshow(first_prediction, cmap="gray")
ax[0].set_title("First Prediction")
ax[0].axis("off")
ax[1].imshow(last_prediction, cmap="gray")
ax[1].set_title("Last Prediction")
ax[1].axis("off")
ax[2].imshow(target_matrix, cmap="gray")
ax[2].set_title("Target Pattern")
ax[2].axis("off")
plt.tight_layout()
plt.show()
plt.close()

plotter, vol_mesh = pysonogen.plot_pressure_field(
    pr.detach().cpu().numpy(),
    x.detach().cpu().numpy(),
    y.detach().cpu().numpy(),
    z.detach().cpu().numpy(),
)

plotter.show()
plotter.deep_clean()  # Explicitly clean up PyVista objects
plotter.close()
