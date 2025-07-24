import matplotlib.pyplot as plt
import numpy as np

folder = r".\target_masks"
filename = r"/linear_lambda4.npz"

f_c = 12.5e6  # center frequency in Hz
c = 1540  # speed of sound in m/s
lambda_ = c / f_c * 1e3  # wavelength in mm

times_lambda = 1 / 4  # number of wavelengths in the target

x_length_mm = 5
y_length_mm = 5
dx = lambda_ / 4  # half-wavelength in mm
dy = lambda_ / 4  # half-wavelength in mm
Nx = int(x_length_mm / dx) + 2
Ny = int(y_length_mm / dy) + 2
Nx = Nx + 1 if Nx % 2 == 0 else Nx  # ensure odd number of elements
Ny = Ny + 1 if Ny % 2 == 0 else Ny  # ensure odd number of elements

dx = x_length_mm / (Nx - 1)  # pixel size in mm
dy = y_length_mm / (Ny - 1)  # pixel size in mm


diameter_x_pix = times_lambda * lambda_ / dx  # diameter in pixels
diameter_y_pix = times_lambda * lambda_ / dy  # diameter in pixels

# ellipse mask
Y, X = np.ogrid[:Nx, :Ny]
cx, cy = (Nx - 1) / 2, (Ny - 1) / 2
a, b = diameter_x_pix / 2, diameter_y_pix / 2
mask = ((X - cx) ** 2 / a**2 + (Y - cy) ** 2 / b**2) <= 1

# mask[Nx // 2, :] = 0  # horizontal line
# mask[:, Ny // 2] = 0  # vertical line


print(mask.shape)
fig, ax = plt.subplots(figsize=(8, 8))
im = ax.imshow(mask, cmap="gray_r", vmin=0, vmax=1)
ax.set_title("Click to toggle elements (1=target, 0=off)")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
np.savez_compressed(
    folder + filename,
    target=mask,
    x_length_mm=x_length_mm,
    y_length_mm=y_length_mm,
    dx=dx,
    dy=dy,
    wavelength=lambda_,
)

# print(f"Using {dxy:.2f} mm wavelength")
# print(f"Using {x_length_mm} mm x {y_length_mm} mm field size")
# print(f"Using {Nx} x {Ny} grid points")

# target_matrix = np.zeros((Nx, Ny), dtype=int)

# fig, ax = plt.subplots(figsize=(8, 8))
# im = ax.imshow(target_matrix, cmap="gray_r", vmin=0, vmax=1)
# ax.set_title("Click to toggle elements (1=target, 0=off)")
# plt.xlabel("X")
# plt.ylabel("Y")


# def onclick(event):
#     if event.inaxes == ax:
#         x, y = int(round(event.xdata)), int(round(event.ydata))
#         if 0 <= x < Nx and 0 <= y < Ny:
#             target_matrix[y, x] = 1 - target_matrix[y, x]
#             im.set_data(target_matrix)
#             fig.canvas.draw_idle()


# cid = fig.canvas.mpl_connect("button_press_event", onclick)
# plt.show()

# plt.close(fig)
# np.savez_compressed(folder + filename, target=target_matrix)
# print(f"Saved field to {folder + filename}")
