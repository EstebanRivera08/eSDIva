from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata

FOVERD = 1.5
FOLDER = r"./focal_spot_analysis_results/"
FILENAME = f"focal_spot_analysis_data_FD{FOVERD}"

PATH = Path(FOLDER + FILENAME + ".npz")
SAVE_FIG = True

# Import npz file
data = np.load(PATH)

# Extract data
x_focals = data["x_focals"]
z_focals = data["z_focals"]
x_max_vec = data["x_max_vec"]
z_max_vec = data["z_max_vec"]
p_max_array = data["p_max_array"]


# Contour colored map
X, Z = np.meshgrid(x_focals, z_focals)
focal_points = np.array([X.flatten(), Z.flatten()]).T

# Create a finer grid
x_fine = np.linspace(x_focals.min(), x_focals.max(), 100)
z_fine = np.linspace(z_focals.min(), z_focals.max(), 100)
X_fine, Z_fine = np.meshgrid(x_fine, z_fine)

# Interpolate the max pressure values onto the finer grid
p_max_norm = p_max_array.T.flatten() / np.max(
    p_max_array
)  # Normalize for better visualization
p_max_fine = griddata(focal_points, p_max_norm, (X_fine, Z_fine), method="cubic")


def contour_focal_spot(x, z, p_max):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    contour = ax.contourf(x, z, p_max, levels=14, cmap="jet", vmin=0, vmax=1)

    plt.colorbar(contour, label="Max Pressure (Pa)", ax=ax)
    ax.invert_yaxis()  # Invert y-axis to match the original orientation
    ax.set_xlabel("X Focal Position (m)")
    ax.set_ylabel("Z Focal Position (m)")
    ax.set_title(f"Max Pressure at focal spot map (FD={FOVERD})")
    ax.grid()

    if SAVE_FIG:
        fig.savefig(PATH.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.show()


contour_focal_spot(x_fine, z_fine, p_max_fine)
