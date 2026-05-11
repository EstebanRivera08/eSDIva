import matplotlib.pyplot as plt
import numpy as np

import pyfield as pf
from pyfield.psimulation import PyField
from pyfield.transducers import Domino

# Parameters
FOVERD = 1.5
SAVE_RESULTS = True
FOLDER_NAME = "focal_spot_analysis_results"
FILE_NAME = f"focal_spot_analysis_data_FD{FOVERD}.npz"

# Create transducers

tx = Domino()

# List of focal spots to be tested

x_focals = np.linspace(0, 10, 21)
z_focals = np.linspace(0, 30, 31)

# Create a list to store the results
x_max_vec = []
z_max_vec = []
p_max_vec = []

for x_f in x_focals:
    for z_f in z_focals:
        print(f"\nSimulating for focal spot at (x={x_f}, z={z_f})")
        # Set the focal point of the transducer
        tx.compute_delays((x_f, 0, z_f))
        tx.compute_apodization((x_f, 0, z_f), FoverD=FOVERD)

        # Create the simulation grid
        field_dict = {
            "x_extent": (-2 + x_f, 2 + x_f),
            "z_extent": (-2 + z_f, 2 + z_f),
            "y_extent": (0, 0),
            "dx": 0.1,
            "dz": 0.1,
            "dy": 0.1,
        }

        # Create the simulator instance
        sim = PyField(tx, verbose=False)

        # Run the simulation
        p, coords = sim(field_dict)

        # Pick the highest pressure point
        max_idx = np.unravel_index(np.argmax(p), p.shape)
        x_idx, z_idx = max_idx[0], max_idx[2]
        x_max = coords["x"][x_idx]
        z_max = coords["z"][z_idx]
        p_max = p[max_idx]

        # Store the results
        x_max_vec.append(x_max)
        z_max_vec.append(z_max)
        p_max_vec.append(p_max)

# Save the results to a file
p_max_array = np.array(p_max_vec).reshape(len(x_focals), len(z_focals))

results_dict = {
    "x_focals": x_focals,
    "z_focals": z_focals,
    "x_max_vec": x_max_vec,
    "z_max_vec": z_max_vec,
    "p_max_array": p_max_array,
}

from pathlib import Path

File_Path = Path(f"{FOLDER_NAME}/{FILE_NAME}")
File_Path.parent.mkdir(parents=True, exist_ok=True)

if SAVE_RESULTS:
    np.savez(File_Path, **results_dict)


def plot_pmax_map(x_focals, z_focals, p_max_array):
    plt.figure(figsize=(8, 6))
    plt.imshow(
        p_max_array.T,
        extent=(x_focals[0], x_focals[-1], z_focals[-1], z_focals[0]),
        origin="upper",
        aspect="auto",
        cmap="jet",
    )
    plt.colorbar(label="Max Pressure")
    plt.xlabel("Focal Depth (z)")
    plt.ylabel("Focal Lateral Position (x)")
    plt.title("Max Pressure Map for Different Focal Spots")
    if SAVE_RESULTS:
        plt.savefig(File_Path.with_suffix(".png"))
    plt.show()


if __name__ == "__main__":
    plot_pmax_map(x_focals, z_focals, p_max_array)
