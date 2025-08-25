import matplotlib.pyplot as plt
import numpy as np

z0 = 5  # mm
fc = 10  # MHz
c = 1540  # m/s
lambda_mm = c / (fc * 1e3)  # mm


num_elem = 55
pitch = 0.30  # mm
w_elem = 0.29  # mm


x_half_extent = (num_elem) * pitch / 2
x_extent = [-x_half_extent, x_half_extent]
y_extent = x_extent


# fresnel zones radii


def r_n(n):
    """
    Fresnel zone radius
    k*(sqrt(z_0^2+r_n^2)-z_0)=n*pi
    -> sqrt(z_0^2+r_n^2) = z_0 + n*pi/k
    -> z_0^2+r_n^2 = (z_0 + n*pi/k)^2
    -> r_n^2 = (z_0 + n*pi/k)^2 - z_0^2
    """
    R_n = z0 + n * lambda_mm / 2
    return np.sqrt(R_n**2 - z0**2)  # mm


Nx = int((x_extent[1] - x_extent[0]) / (pitch))
Ny = Nx
k = 2 * np.pi / lambda_mm  # 1/mm
x = np.linspace(x_extent[0], x_extent[1], Nx)  # mm
y = np.linspace(y_extent[0], y_extent[1], Ny)  # mm

Fresnel_zones = np.zeros((Nx, Ny), dtype=int)
Odd_fresnel_zones = np.zeros((Nx, Ny), dtype=int)

for ix in range(Nx):
    for iy in range(Ny):
        r = np.sqrt(x[ix] ** 2 + y[iy] ** 2)
        n = int(np.floor(k * (np.sqrt(z0**2 + r**2) - z0) / np.pi))
        Fresnel_zones[ix, iy] = n
        if n % 2 == 1:
            Odd_fresnel_zones[ix, iy] = 1

plt.figure(figsize=(6, 5))
plt.imshow(
    Fresnel_zones.T,
    extent=(x_extent[0], x_extent[1], y_extent[0], y_extent[1]),
    origin="lower",
    cmap="jet",
)
plt.colorbar(label="Fresnel Zone Index")
plt.xlabel("X Position (mm)")
plt.ylabel("Y Position (mm)")
plt.title("Fresnel Zones")
plt.grid()
plt.show()

plt.figure(figsize=(6, 5))
plt.imshow(
    Odd_fresnel_zones.T,
    extent=(x_extent[0], x_extent[1], y_extent[0], y_extent[1]),
    origin="lower",
    cmap="gray",
)
plt.colorbar(label="Fresnel Zone. Odd = 1")
plt.xlabel("X Position (mm)")
plt.ylabel("Y Position (mm)")
plt.title(f"Fresnel Zones at z0={z0}")
plt.grid()
plt.show()
