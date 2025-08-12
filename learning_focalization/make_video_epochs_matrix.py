import imageio.v2 as imageio
import matplotlib.gridspec as gridspec
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

Zeus_Matrix = Transducers.Zeus_Matrix()

# ---------------------------- Functions ----------------------------


def take_slices(pressure_field, x, y, z, centered=False):
    """Take 2D slices of the 3D pressure field at the middle or centered indices.
    Parameters
    ----------
    pressure_field : ndarray
        3D pressure field data.
    x, y, z : ndarray
        Coordinate arrays.
    centered : bool, optional
        If True, slices are taken at the indices closest to the maximum value in the field.
    Returns
    -------
    tuple of ndarray
        Slices of the pressure field in the XZ, XY, and YZ planes.
    """
    if centered:
        # Look for the y, x, z indices that are closest to the max value
        max_idx = np.unravel_index(np.nanargmax(pressure_field), pressure_field.shape)
        y0, x0, z0 = max_idx[1], max_idx[0], max_idx[2]
    else:
        # Use the middle indices
        y0 = int(np.floor(y.shape[0] / 2))
        x0 = int(np.floor(x.shape[0] / 2))
        z0 = int(np.floor(z.shape[0] / 2))
    # print(
    #     f"Taking slice x_ind, y_ind, z_ind = {x0 + 1}/{x.shape[0]}, {y0 + 1}/{y.shape[0]}, {z0 + 1}/{z.shape[0]}"
    # )

    # Use nanmin and nanmax to ignore NaN values
    vmin = np.nanmin(pressure_field)
    vmax = np.nanmax(pressure_field)

    XZ_plane = pressure_field[:, y0, :].squeeze()
    XY_plane = pressure_field[:, :, z0].squeeze()
    YZ_plane = pressure_field[x0, :, :].squeeze()

    return {
        "x_slice": x[x0],
        "y_slice": y[y0],
        "z_slice": z[z0],
        "XZ": XZ_plane,
        "XY": XY_plane,
        "YZ": YZ_plane,
        "vmin": vmin,
        "vmax": vmax,
    }


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


# ----------------------------

# Focalization spot
focus_mm = np.array([0, 0, 5])  # mm [x, y, z] #8
FoverD = 0.75
delays = Zeus_Matrix.compute_delays(focus_mm=focus_mm, plot=False)
apodization = Zeus_Matrix.compute_apodization(
    focus_mm=focus_mm, F_over_D=FoverD, apodization_type="circular", plot=False
)
target = "1lambda"  # Target pattern to use
target_folder = r".\target_masks"
target_filename = f"/matrix_{target}_10MHz.npz"
name_model = f"opt_150epochs_3DloglossE_1planes_noprocess_delay_apod0_{target}_v1"
state_name = f"Matrix_torch_state_{name_model}"
state_folder = r".\test_models\matrix"
state_path = f"{state_folder}/{state_name}.pth"
data_path = f"{state_folder}/{state_name}_data.npz"
figure_name = (
    state_folder + "/" + state_name + "small.png"  # "_unwrap08.png"  # + "_unwrap_"
)  # Add the correct extension

# -------------------- load the target pattern --------------------
print(f"Loading target pattern {target_filename}")
target_dic = np.load(target_folder + target_filename)
target_matrix = target_dic["target"]

y_target2D = torch.tensor(target_matrix, dtype=torch.float32, device=device)
z_plane_mm = 5  # mm
focus_mm = np.array([0, 0, z_plane_mm])  # mm [x, y, z]


# ----------------- Loading data --------------------
data = np.load(data_path)
loss_vect = data["loss"]
delays_vect = data["delays"]
apodization_vect = data["apodization"]
max_pr_vect = data["max_pr"]


# # ----------------- Add initial delays and apodization --------------------
# loss0 = data["first_loss"]
# apodization0 = np.ones(Zeus_Matrix.n_elem_x * Zeus_Matrix.n_elem_y) * 0.5
# delays0 = np.zeros(Zeus_Matrix.n_elem_x * Zeus_Matrix.n_elem_y)

# apodization_vect.insert(0, apodization0)
# delays_vect.insert(0, delays0)

# ----------------- Define field info --------------------
x_slice = 0
y_slice = 0
z_slice = focus_mm[2]
dx = 0.075
dy = 0.075
dz = 0.075

Delta_x = 2.5  #  0.8  # mm
Delta_y = 2.5  #  0.8  # mm
Delta_z = 0  #  1 mm

XY_plane_info_mm = {
    "x_extent": [-Delta_x + focus_mm[0], Delta_x + focus_mm[0]],
    "y_extent": [-Delta_y + focus_mm[1], Delta_y + focus_mm[1]],
    "z_extent": [-Delta_z + focus_mm[2], Delta_z + focus_mm[2]],
    "dx": dx,  # 0.02
    "dy": dy,  # 0.02
    "dz": dz,  # 0.02
}
Delta_y = 0  #  0.8  # mm
Delta_z = 2.5  #  1 mm
XZ_plane_info_mm = {
    "x_extent": [-Delta_x + focus_mm[0], Delta_x + focus_mm[0]],
    "y_extent": [-Delta_y + focus_mm[1], Delta_y + focus_mm[1]],
    "z_extent": [-Delta_z + focus_mm[2], Delta_z + focus_mm[2]],
    "dx": dx,  # 0.02
    "dy": dy,  # 0.02
    "dz": dz,  # 0.02
}
Delta_x = 0
Delta_y = 2.5  #  0.8  # mm
YZ_plane_info_mm = {
    "x_extent": [-Delta_x + focus_mm[0], Delta_x + focus_mm[0]],
    "y_extent": [-Delta_y + focus_mm[1], Delta_y + focus_mm[1]],
    "z_extent": [-Delta_z + focus_mm[2], Delta_z + focus_mm[2]],
    "dx": dx,  # 0.02
    "dy": dy,  # 0.02
    "dz": dz,  # 0.02
}


plt.rcParams.update({"axes.titlesize": 12, "axes.labelsize": 10})

period = 1 / Zeus_Matrix.fc * 1e6  # us


# ----------------- Prepare the field for plotting --------------------
name_video = "training_video2.mp4"
n_elem_x = Zeus_Matrix.n_elem_x
n_elem_y = Zeus_Matrix.n_elem_y
epochs = len(loss_vect) - 1
interpolation = "bilinear"  # "nearest", "bilinear", "bicubic"

# Create video writer
with imageio.get_writer(name_video, fps=3) as writer:
    # if 1:
    for epoch in range(epochs + 1):
        # if 1:
        Zeus_Matrix.set_apodization(apodization_vect[epoch])
        Zeus_Matrix.set_delays(delays_vect[epoch] * 1e-6)
        Matrix_torch = TorchField(Zeus_Matrix, device=device)
        XZ_plane, x1, y1, z1 = Matrix_torch(XZ_plane_info_mm, batch_size=2048)
        XY_plane, x2, y2, z2 = Matrix_torch(XY_plane_info_mm, batch_size=2048)
        YZ_plane, x3, y3, z3 = Matrix_torch(YZ_plane_info_mm, batch_size=2048)

        # Take slices for plotting
        XZ_plane = np.flip(XZ_plane.squeeze().detach().cpu().numpy(), axis=1)
        XY_plane = XY_plane.squeeze().detach().cpu().numpy()
        YZ_plane = np.flip(YZ_plane.squeeze().detach().cpu().numpy(), axis=1)
        x = x1.detach().cpu().numpy()
        y = y2.detach().cpu().numpy()
        z = z3.detach().cpu().numpy()
        vmin = min(XZ_plane.min(), XY_plane.min(), YZ_plane.min())
        vmax = max(XZ_plane.max(), XY_plane.max(), YZ_plane.max())

        # ---------- Suggested GridSpec layout ----------
        # We'll use 6 rows x 7 columns:
        #  row 0 : loss (spans cols 0..5)
        #  row 1 : small imshows (apod columns 0-1, delays 2-3, target 4-5)
        #  rows 2-4 : planes (each plane spans rows 2..4 to give same height)
        #  row 5 : bottom row reserved for horizontal colorbars (aligned with columns 0-1,2-3,4-5)
        #  column 6 : reserved for vertical colorbar for planes (spans rows 2..4)
        #
        # Note: change nrows to 7 if you prefer two rows for horizontal colorbars.
        nrows, ncols = 4, 4
        height_ratios = [
            0.5,
            1,
            0.1,
            1.0,
        ]  # tweak to make top/plane rows equal; last row controls horizontal cbar height
        width_ratios = [
            1,
            1,
            1,
            0.1,
        ]  # last column controls vertical cbar width (increase to thicken it)

        fig = plt.figure(figsize=(10, 8), constrained_layout=True)
        gs = gridspec.GridSpec(
            nrows,
            ncols,
            figure=fig,
            height_ratios=height_ratios,
            width_ratios=width_ratios,
        )
        # nrows, ncols = 6, 7
        # fig = plt.figure(figsize=(17, 12), constrained_layout=True)
        # gs = gridspec.GridSpec(nrows, ncols, figure=fig)

        # --- LOSS (top row across cols 0..5) ---
        ax_loss = fig.add_subplot(gs[0, :3])
        ax_loss.plot(
            loss_vect[:epoch],
            "k-",
            marker="^",
            markerfacecolor="red",
            label="Loss",
        )
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Loss")
        ax_loss.set_title("a) Training Loss")
        ax_loss.grid()
        ax_loss.legend()

        # --- Row 1: Apodization, Delays, Target (cols grouped as 0:2, 2:4, 4:6) ---
        ax_apo = fig.add_subplot(gs[1, 0])
        im_apo = ax_apo.imshow(
            apodization_vect[epoch].reshape((n_elem_x, n_elem_y)),
            cmap="cool",
            vmin=0,
            vmax=1,
            origin="lower",
        )
        ax_apo.set_title(f"b) Apodization epoch: {epoch}")
        ax_apo.set_xlabel("Element X")
        ax_apo.set_ylabel("Element Y")

        ax_del = fig.add_subplot(gs[1, 1])
        im_del = ax_del.imshow(
            delays_vect[epoch].reshape((n_elem_x, n_elem_y)),
            cmap="hsv",
            vmin=-period,
            vmax=period,
            origin="lower",
        )
        ax_del.set_title(f"c) Delays epoch: {epoch}")
        ax_del.set_xlabel("Element X")
        ax_del.set_ylabel("Element Y")

        ax_targ = fig.add_subplot(gs[1, 2])
        im_targ = ax_targ.imshow(
            target_matrix.T, cmap="gray", vmin=0, vmax=1, origin="lower"
        )
        ax_targ.set_title("d) Target Pattern")
        ax_targ.set_xlabel("Element X")
        ax_targ.set_ylabel("Element Y")

        # ----------- colorbars for apodization, delays, target in bottom row (row 5) -----------
        # --- Bottom row (row 5) holds horizontal colorbars aligned under 0:2, 2:4, 4:6 respectively ---
        cax_apo = fig.add_subplot(gs[2, 0])
        cb_apo = fig.colorbar(im_apo, cax=cax_apo, orientation="horizontal")
        cb_apo.set_label("Apodization")
        # clean the axis (remove ticks/labels on the small cax if desired)
        cax_apo.xaxis.set_ticks_position("bottom")

        cax_del = fig.add_subplot(gs[2, 1])
        cb_del = fig.colorbar(im_del, cax=cax_del, orientation="horizontal")
        cb_del.set_label("Delay (us)")
        cax_del.xaxis.set_ticks_position("bottom")

        cax_targ = fig.add_subplot(gs[2, 2])
        cb_targ = fig.colorbar(im_targ, cax=cax_targ, orientation="horizontal")
        cb_targ.set_label("Intensity")
        cax_targ.xaxis.set_ticks_position("bottom")

        # --- Rows 2..5: Planes (each plane spans rows 2..4 to create tall equal-sized axes) ---
        ax_xz = fig.add_subplot(gs[3, 0])
        im_xz = ax_xz.imshow(
            XZ_plane.T,
            cmap="jet",
            extent=[x.min(), x.max(), z.max(), z.min()],
            vmin=vmin,
            vmax=vmax,
            interpolation=interpolation,
            origin="lower",
        )
        ax_xz.set_title(f"e) XZ Plane at y = {y_slice:.2f} mm")
        ax_xz.set_xlabel("X (mm)")
        ax_xz.set_ylabel("Z (mm)")

        ax_yz = fig.add_subplot(gs[3, 1])
        im_yz = ax_yz.imshow(
            YZ_plane.T,
            cmap="jet",
            extent=[y.min(), y.max(), z.max(), z.min()],
            vmin=vmin,
            vmax=vmax,
            interpolation=interpolation,
            origin="lower",
        )
        ax_yz.set_title(f"f) YZ Plane at x = {x_slice:.2f} mm")
        ax_yz.set_xlabel("Y (mm)")
        ax_yz.set_ylabel("Z (mm)")

        ax_xy = fig.add_subplot(gs[3, 2])
        im_xy = ax_xy.imshow(
            XY_plane.T,
            cmap="jet",
            extent=[x.min(), x.max(), y.min(), y.max()],
            vmin=vmin,
            vmax=vmax,
            interpolation=interpolation,
            origin="lower",
        )
        ax_xy.set_title(f"g) XY Plane at z = {z_slice:.2f} mm")
        ax_xy.set_xlabel("X (mm)")
        ax_xy.set_ylabel("Y (mm)")

        # --- Vertical colorbar for planes: place in column 6, rows 2..4 (same vertical span) ---
        cax_planes = fig.add_subplot(gs[3, 3])
        cb_planes = fig.colorbar(im_xz, cax=cax_planes, orientation="vertical")
        cb_planes.set_label("Pressure (normalized)")

        # --- Final layout tweaks (constrained_layout manages most spacing) ---
        # But ensure the figure edges are fine:
        fig.subplots_adjust(top=0.95, bottom=0.08, left=0.05, right=0.92, hspace=0.35)

        # plt.show()
        # Convert current figure to image array and append to video
        fig.canvas.draw()
        frame = np.array(fig.canvas.buffer_rgba())
        writer.append_data(frame)
        plt.close(fig)
        print(f"Epoch {epoch + 1}/{epochs + 1} processed.")

print(f"Video saved as {name_video}")


# Zeus_Matrix.show()
