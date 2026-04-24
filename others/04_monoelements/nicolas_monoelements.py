import matplotlib.pyplot as plt
import numpy as np

from pyfield.psimulation import PyField
from pyfield.transducers import ConcaveCircularTransducer

# Define the transducer parameters
Diameter_mm = 25.4 / 2  # 0.5 inch diameter
Frequency_MHz = [5, 10, 15]  # 1 MHz frequency
Focal_mm = 25.4  # 1 inch focal length

# Create the transducer
tx_5MHz = ConcaveCircularTransducer(
    diameter_mm=Diameter_mm,
    radius_of_curvature_mm=Focal_mm,
    frequency_Hz=Frequency_MHz[0] * 1e6,
    no_sub=30,
)
tx_10MHz = ConcaveCircularTransducer(
    diameter_mm=Diameter_mm,
    radius_of_curvature_mm=Focal_mm,
    frequency_Hz=Frequency_MHz[1] * 1e6,
    no_sub=50,
)
tx_15MHz = ConcaveCircularTransducer(
    diameter_mm=Diameter_mm,
    radius_of_curvature_mm=Focal_mm,
    frequency_Hz=Frequency_MHz[2] * 1e6,
    no_sub=75,
)

tx_5MHz.show()
tx_10MHz.show()
tx_15MHz.show()

# Define field of view
field_dict = {
    "x_extent_mm": [-2, 2],
    "z_extent_mm": [15, 40],
    "y_extent_mm": [0, 0],
    "dx_mm": 0.025,
    "dz_mm": 0.125,
    "dy_mm": 0.1,
}
field_dict = {
    "x_extent_mm": [-8, 8],
    "z_extent_mm": [5, 50],
    "y_extent_mm": [0, 0],
    "dx_mm": 0.125,
    "dz_mm": 0.125,
    "dy_mm": 0.1,
}
# Create the PyField instance for simulation
pf_5MHz = PyField(transducer=tx_5MHz)
pf_10MHz = PyField(transducer=tx_10MHz)
pf_15MHz = PyField(transducer=tx_15MHz)


# Simulate the acoustic field
x, y, z, p_5MHz = pf_5MHz(field_dict)
x, y, z, p_10MHz = pf_10MHz(field_dict)
x, y, z, p_15MHz = pf_15MHz(field_dict)

# Plot the results
plt.figure(figsize=(12, 8))
plt.subplot(1, 3, 1)
extent = (
    field_dict["x_extent_mm"][0],
    field_dict["x_extent_mm"][1],
    field_dict["z_extent_mm"][1],
    field_dict["z_extent_mm"][0],
)

plt.imshow(
    np.squeeze(p_5MHz).T, extent=extent, origin="upper", aspect="auto", cmap="jet"
)
plt.title("5 MHz")
plt.xlabel("x (mm)")
plt.ylabel("z (mm)")

plt.subplot(1, 3, 2)
plt.imshow(
    np.squeeze(p_10MHz).T, extent=extent, origin="upper", aspect="auto", cmap="jet"
)
plt.title("10 MHz")
plt.xlabel("x (mm)")
plt.ylabel("z (mm)")

plt.subplot(1, 3, 3)
plt.imshow(
    np.squeeze(p_15MHz).T, extent=extent, origin="upper", aspect="auto", cmap="jet"
)
plt.title("15 MHz")
plt.xlabel("x (mm)")
plt.ylabel("z (mm)")
plt.tight_layout()
plt.show()
