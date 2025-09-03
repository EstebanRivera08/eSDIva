import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import fftconvolve

# ----- transducer / physics params -----
save_figure = True

z0 = 3  # focus depth in mm
name_figure = f"fresnel_zones_matrix_{z0}mm"
fc = 10.0  # MHz
c = 1540.0  # m/s
lambda_mm = c / (fc * 1e3)  # mm
k = 2 * np.pi / lambda_mm

num_elem_x = 55
num_elem_y = 55
pitch_x = 0.30  # mm
pitch_y = 0.30  # mm
w_elem = 0.29  # mm  (element active width)
h_elem = w_elem  # assume square element for now

# --- geometry ---
Nx_e, Ny_e = num_elem_x, num_elem_y
px, py = pitch_x, pitch_y
wx, wy = w_elem, h_elem

# element centers (exact)
x_elem_c = (np.arange(Nx_e) - (Nx_e - 1) / 2.0) * px
y_elem_c = (np.arange(Ny_e) - (Ny_e - 1) / 2.0) * py

# active aperture half extents (cover full active face)
Hx = (Nx_e - 1) / 2.0 * px
Hy = (Ny_e - 1) / 2.0 * py

# fine grid aligned to pitch/oversample so centers fall on nodes
oversample = 100
dx = px / oversample
dy = py / oversample
Nx = np.round((Hx * 2) / dx).astype(int)
Ny = np.round((Hy * 2) / dy).astype(int)
if Nx % 2 == 0:
    Nx += 1
if Ny % 2 == 0:
    Ny += 1

x = np.linspace(-Hx, Hx, Nx)  # +0.5*dx to ensure inclusion of +Hx
y = np.linspace(-Hy, Hy, Ny)
X, Y = np.meshgrid(x, y, indexing="xy")


# ----- continuous Fresnel index map -----
r = np.sqrt(X**2 + Y**2)
n = k * (np.sqrt(z0**2 + r**2) - z0) / np.pi  # radians, this is k*deltaR
correlation = np.cos(n * np.pi)  # parity field (1 if  n = odd, -1 if n = even)

# ----- convolve with rect element shape (box blur) -----
# rect kernel in continuous grid units:
grid_dx = x[1] - x[0]
grid_dy = y[1] - y[0]
kx = int(np.round(w_elem / grid_dx))
ky = int(np.round(h_elem / grid_dy))
if kx < 1:
    kx = 1
if ky < 1:
    ky = 1
rect_kernel = np.ones((ky, kx), dtype=float)
rect_kernel /= rect_kernel.sum()

# For a binary (odd/even) map: smooth with rect to get per-element effective value
correlation_element1 = correlation  # Continuous
correlation_element2 = fftconvolve(correlation, rect_kernel, mode="same")  # Convolved

# ----- sample at element centers (comb) -----
# element centers positions (mm)

# --- indices of element centers (exactly on nodes when oversample is integer) ---
ix_centers = np.clip(np.round((x_elem_c - x[0]) / dx).astype(int), 0, len(x) - 1)
iy_centers = np.clip(np.round((y_elem_c - y[0]) / dy).astype(int), 0, len(y) - 1)
position_error = np.mean(np.abs(x[ix_centers] - x_elem_c))  # mm


def bilinear_sample(A, x_mm, y_mm, x0, y0, dx, dy):
    # convert mm -> fractional indices
    fx = (x_mm - x0) / dx
    fy = (y_mm - y0) / dy
    x0i = np.floor(fx).astype(int)
    y0i = np.floor(fy).astype(int)
    x1i = np.clip(x0i + 1, 0, A.shape[1] - 1)
    y1i = np.clip(y0i + 1, 0, A.shape[0] - 1)
    wx1 = fx - x0i
    wx0 = 1 - wx1
    wy1 = fy - y0i
    wy0 = 1 - wy1
    v00 = A[y0i, x0i]
    v10 = A[y0i, x1i]
    v01 = A[y1i, x0i]
    v11 = A[y1i, x1i]
    return wy0 * (wx0 * v00 + wx1 * v10) + wy1 * (wx0 * v01 + wx1 * v11)


# vectorized sample of all centers
Xc, Yc = np.meshgrid(x_elem_c, y_elem_c, indexing="xy")
n_tx = bilinear_sample(n, Xc, Yc, x[0], y[0], dx, dy)
correlation_tx1 = bilinear_sample(correlation_element1, Xc, Yc, x[0], y[0], dx, dy)
correlation_tx2 = bilinear_sample(correlation_element2, Xc, Yc, x[0], y[0], dx, dy)

# n_tx = np.zeros((num_elem_x, num_elem_y), dtype=int)
# correlation_tx1 = np.zeros((num_elem_x, num_elem_y))
# correlation_tx2 = np.zeros((num_elem_x, num_elem_y))
# for i, ix in enumerate(ix_centers):
#     for j, iy in enumerate(iy_centers):
#         n_tx[j, i] = n[iy, ix]
#         correlation_tx1[j, i] = correlation_element1[
#             iy, ix
#         ]  # note ordering (row,col) vs x,y
#         correlation_tx2[j, i] = correlation_element2[
#             iy, ix
#         ]  # note ordering (row,col) vs x,y

# Create masks
mask_tx1 = np.zeros_like(correlation_tx1)
mask_tx2 = np.zeros_like(correlation_tx2)

mask_tx1[correlation_tx1 > 0] = 1
mask_tx2[correlation_tx2 > 0] = 1
print("Position error at element centers (mm): ", position_error)

# ----- apply final aperture window if desired (rectangle) -----
# here we already sampled only inside the N elements, so it's naturally windowed


# normalize parity to be between 0 and 1
def normalize(matrix):
    matrix_new = (matrix - matrix.min()) / (matrix.max() - matrix.min())
    return matrix_new


# ----- plotting -----
fig, axes = plt.subplots(2, 4, figsize=(17, 7))
axes = axes.flatten()

i = 0
im1 = axes[i].imshow(
    n.T * np.pi,
    origin="lower",
    extent=(x[0], x[-1], y[0], y[-1]),
    cmap="jet",
    interpolation="bilinear",
)
axes[i].set_title(r"a) $\Delta \phi = n\times\pi$")
axes[i].set_xlabel("x (mm)")
axes[i].set_ylabel("y (mm)")
plt.colorbar(im1, ax=axes[i], label="Phase delay, rad")

i += 1
im2 = axes[i].imshow(
    correlation_element1.T,
    origin="lower",
    extent=(x[0], x[-1], y[0], y[-1]),
    cmap="gray",
    interpolation="bilinear",
)
axes[i].set_title(r"b) $C^{continue}(\Delta \phi) = cos(\Delta \phi)$")
axes[i].set_xlabel("x (mm)")
axes[i].set_ylabel("y (mm)")
plt.colorbar(
    im2,
    ax=axes[i],
    label="Correlation \n (1 = constructive, -1 = destructive)",
    shrink=0.7,
)


i += 1
im4 = axes[i].imshow(
    correlation_tx1.T,
    origin="lower",
    extent=(
        x_elem_c[0] - pitch_x / 2,
        x_elem_c[-1] + pitch_x / 2,
        y_elem_c[-1] - pitch_y / 2,
        y_elem_c[0] + pitch_y / 2,
    ),
    cmap="gray",
)
axes[i].set_title(r"c) $C^{continue}(\Delta \phi_{(x_e,y_e)})$ (sampled per-element)")
axes[i].set_xlabel("x (mm)")
axes[i].set_ylabel("y (mm)")
plt.colorbar(
    im4,
    ax=axes[i],
    label="Correlation \n (1 = constructive, -1 = destructive)",
    shrink=0.7,
)

i += 1
im5 = axes[i].imshow(
    mask_tx1.T,
    origin="lower",
    extent=(
        x_elem_c[0] - pitch_x / 2,
        x_elem_c[-1] + pitch_x / 2,
        y_elem_c[-1] - pitch_y / 2,
        y_elem_c[0] + pitch_y / 2,
    ),
    cmap="gray",
)
axes[i].set_title(r"$mask(C^{continue}_{>0})$")
axes[i].set_xlabel("x (mm)")
axes[i].set_ylabel("y (mm)")

i += 1
im3 = axes[i].imshow(
    n_tx.T * np.pi,
    origin="lower",
    extent=(
        x_elem_c[0] - pitch_x / 2,
        x_elem_c[-1] + pitch_x / 2,
        y_elem_c[-1] - pitch_y / 2,
        y_elem_c[0] + pitch_y / 2,
    ),
    cmap="jet",
)
axes[i].set_title(r"d) $\Delta \phi_{(x_e,y_e)} = n_{(x_e,y_e)}\times\pi$")
axes[i].set_xlabel("x (mm)")
axes[i].set_ylabel("y (mm)")
plt.colorbar(im3, ax=axes[i], label="Phase delay, rad")

i += 1
im2 = axes[i].imshow(
    correlation_element2.T,
    origin="lower",
    extent=(x[0], x[-1], y[0], y[-1]),
    cmap="gray",
    interpolation="bilinear",
)
axes[i].set_title(r"e) $C^{elements} = C^{continue} * rect_{\{w_x,w_y\}}$")
axes[i].set_xlabel("x (mm)")
axes[i].set_ylabel("y (mm)")
plt.colorbar(
    im2,
    ax=axes[i],
    label="Correlation \n (1 = constructive, -1 = destructive)",
    shrink=0.7,
)

i += 1
im4 = axes[i].imshow(
    correlation_tx2.T,
    origin="lower",
    extent=(
        x_elem_c[0] - pitch_x / 2,
        x_elem_c[-1] + pitch_x / 2,
        y_elem_c[-1] - pitch_y / 2,
        y_elem_c[0] + pitch_y / 2,
    ),
    cmap="gray",
)
axes[i].set_title(r"f) $C^{elements}(\Delta \phi_{(x_e,y_e)})$ (sampled per-element)")
axes[i].set_xlabel("x (mm)")
axes[i].set_ylabel("y (mm)")
plt.colorbar(
    im4,
    ax=axes[i],
    label="Correlation \n (1 = constructive, -1 = destructive)",
    shrink=0.7,
)

i += 1
im5 = axes[i].imshow(
    mask_tx2.T,
    origin="lower",
    extent=(
        x_elem_c[0] - pitch_x / 2,
        x_elem_c[-1] + pitch_x / 2,
        y_elem_c[-1] - pitch_y / 2,
        y_elem_c[0] + pitch_y / 2,
    ),
    cmap="gray",
)
axes[i].set_title(r"g) $mask(C^{elements}_{>0})$")
axes[i].set_xlabel("x (mm)")
axes[i].set_ylabel("y (mm)")

plt.tight_layout()

plt.show()

if save_figure:
    fig.savefig(name_figure + ".png", dpi=300, bbox_inches="tight")
