import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import io as sio

import pyfield
import pyfield.transducers as transducers
from pyfield.psimulation import PyField

# ------------------ Find .mat files -----------------------
BASE = Path(r"c:\Users\INSERM\Documents\Esteban\PyField\comparison_FIELDII\data")
version = "_pulsed"

# --------------- Permanent variables ----------------------

# ------------------ Saving / options -----------------------
frequency_MHz = 12.5
c = 1540.0  # m/s
sampling_frequency_MHz = 200  # MHz

# ------- focus and simulation window (input parameters) -------
x_extent_mm = [-8, 8]
y_extent_mm = [0, 0]
z_extent_mm = [1, 15]
dxyz_mm = 0.05  # mm
focus_mm = [0.0, 0.0, 8]

#  ------------------- transducer characteristics --------------------
tx_N_elements = 128
tx_element_height_mm = 1.5
tx_width_mm = 0.108
tx_pitch_mm = 0.11
tx_kerf_mm = tx_pitch_mm - tx_width_mm
tx_elevationFocus_mm = 8.0
tx_frequency_Hz = frequency_MHz * 1e6
no_sub_x = 1
no_sub_y = 10


# ------- Create transducer (Linear array equivalent) -------
tx = transducers.LinearArrayTransducer(
    n_elements=tx_N_elements,
    element_width_mm=tx_width_mm,
    element_height_mm=tx_element_height_mm,
    elevation_focus_mm=tx_elevationFocus_mm,
    kerf_mm=tx_kerf_mm,
    no_sub_x=no_sub_x,
    no_sub_y=no_sub_y,
    frequency_Hz=tx_frequency_Hz,
)

# Optional: compute/aply delays / apodization
delays = tx.compute_delays(focus_mm=focus_mm)

# -------- DEFINE GRID (keep odd-number-of-points logic) --------

# Use PyField API which accepts extents + spacing (same style as notebook)
field_info_mm = {
    "x_extent": x_extent_mm,
    "y_extent": y_extent_mm,
    "z_extent": z_extent_mm,
    "dx": dxyz_mm,
    "dy": dxyz_mm,
    "dz": dxyz_mm,
}

# ------------------ Compute the field --------------------
field_solver = PyField(tx, fs=sampling_frequency_MHz * 1e6, c=c, monochromatic=False)

# first computation includes setup time for compiling and
# optimizing kernels. Subsequent it wont be counted.\

x, y, z, pr_auto = field_solver(field_info_mm, method="auto")
x, y, z, pr_sdi = field_solver(field_info_mm, method="sdi")
x, y, z, pr_naive = field_solver(field_info_mm, method="naive")

repetition = 1
for rep in range(repetition):
    x, y, z, pr_naive = field_solver(field_info_mm, method="naive")

for rep in range(repetition):
    x, y, z, pr_sdi = field_solver(field_info_mm, method="sdi")

for rep in range(repetition):
    x, y, z, pr_auto = field_solver(field_info_mm, method="auto")

# # From the timing logs compute average times and std deviations

start = 3
timelogs = field_solver.sir_running_time_log
print(f"t_log ({len(timelogs[start:])}) : {timelogs[start:]}")

# sometimes some seconds are added due to OS processes
# to avoid biasing the mean, we will perform 7 repetitions and takeout
# min and max values and average the 5 middle values.
times_naive = np.sort(timelogs[start : start + repetition])
times_sdi = np.sort(timelogs[start + repetition : start + 2 * repetition])
times_auto = np.sort(timelogs[start + 2 * repetition : start + 3 * repetition])
time_naive = np.mean(times_naive)
time_sdi = np.mean(times_sdi)
time_auto = np.mean(times_auto)
std_naive = np.std(times_naive)
std_sdi = np.std(times_sdi)
std_auto = np.std(times_auto)

print(
    f"Average computation times over {repetition} repetitions:\n"
    f" Naive: {time_naive:.4f} s (std: {std_naive:.4f} s)\n"
    f" SDI:   {time_sdi:.4f} s (std: {std_sdi:.4f} s)\n"
    f" Auto:  {time_auto:.4f} s (std: {std_auto:.4f} s)\n"
)

data = {
    "c": c,
    "sampling_frequency": sampling_frequency_MHz * 1e6,
    "f0": tx_frequency_Hz,
    "lamda": c / tx_frequency_Hz,
    "focus_mm": focus_mm,
    "tx_N_elements": tx_N_elements,
    "tx_element_height_mm": tx_element_height_mm,
    "tx_width_mm": tx_width_mm,
    "tx_pitch_mm": tx_pitch_mm,
    "tx_kerf_mm": tx_kerf_mm,
    "tx_elevationFocus_mm": tx_elevationFocus_mm,
    "tx_frequency_Hz": tx_frequency_Hz,
    "no_sub_x": no_sub_x,
    "no_sub_y": no_sub_y,
    "M": field_solver.M,
    "P": field_solver.P_log[-1],
    "T": field_solver.T_log[-1],
    "deltak": field_solver.mean_sub_elem_delta_k_log[-1],
    "h_calc_time": timelogs[start:],
    "time_naive": time_naive,
    "time_sdi": time_sdi,
    "time_auto": time_auto,
    "std_naive": std_naive,
    "std_sdi": std_sdi,
    "std_auto": std_auto,
    "x": x,
    "y": y,
    "z": z,
    "pr_naive": pr_naive,
    "pr_sdi": pr_sdi,
    "pr_auto": pr_auto,
}

# print
filename = f"Linear_nsubx{data['no_sub_x']}_nsuby{data['no_sub_y']}_fs{sampling_frequency_MHz}_P{data['P']}_M_{data['M']}_T{data['T']}{version}.npz"
print(f"Saving filename : {filename}")
# save data with numpy
np.savez_compressed(BASE / filename, **data)

# Optional: visualize
max_pr = np.max(np.abs(pr_auto))

print(pr_auto.shape)

fig = plt.figure(figsize=(10, 7))

plt.ion()  # Enable interactive mode

for i in range(1, pr_auto.shape[0] - 1, 20):  # Loop through time dimension
    plt.clf()
    _ = plt.title(f"Time step {i + 1} / {pr_auto.shape[0]}")
    img = plt.imshow(
        np.squeeze(pr_auto[i, :, 0, :]).T,
        extent=(x[0], x[-1], z[-1], z[0]),
        cmap="jet",
        # vmin=0,
        # vmax=max_pr,
    )
    _ = plt.colorbar(img)
    plt.draw()
    fig.canvas.flush_events()
    plt.pause(0.1)

plt.ioff()  # Turn off interactive mode at the end
plt.show()
