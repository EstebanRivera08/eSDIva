import re
from pathlib import Path

import numpy as np
from scipy import io as sio

import pyfield
import pyfield.transducers as transducers
from pyfield.psimulation import PyField

# ------------------ Find .mat files -----------------------
BASE = Path(r"c:\Users\INSERM\Documents\Esteban\PyField\comparison_FIELDII\data\Matrix")

# find .mat file(s)
pattern = re.compile(
    r"nsubx(?P<nsubx>\d+)_nsuby(?P<nsuby>\d+)_fs(?P<fs>\d+)_nxyz(?P<nxyz>\d+)",
    re.IGNORECASE,
)

mats = list(BASE.glob("*.mat"))
print(f"Found {len(mats)}.mat files. \n ")
repetition = 7

# mats = [mats[0]]  # for testing pick the first one only

# --------------- Permanent variables ----------------------

# ------------------ Saving / options -----------------------
frequency_MHz = 10
c = 1540.0  # m/s

# ------- focus and simulation window (input parameters) -------
x_extent_mm = [-2, 2]
y_extent_mm = [-2, 2]
z_extent_mm = [3, 13]
focus_mm = [0.0, 0.0, 8]

#  ------------------- transducer characteristics --------------------
# Create transducer with elevation focus
tx_N_elem_x = 55
tx_N_elem_y = 55
tx_elem_width_mm = 0.29
tx_elem_height_mm = 0.29
tx_pitch_mm = 0.3
tx_kerf_x_mm = tx_pitch_mm - tx_elem_width_mm
tx_kerf_y_mm = tx_pitch_mm - tx_elem_height_mm
tx_frequency_Hz = frequency_MHz * 1e6  # MHz

# ------------------ Loop over .mat files -----------------------
for i, mat in enumerate(mats):
    print("--------------------------------------------------")
    print(f"idx {i + 1}/{len(mats)}: {mat.name}")
    dict_w_vars = pattern.search(mat.name).groupdict()

    # ------ P (points) dependent variables from .mat file -------
    nxyz = int(dict_w_vars["nxyz"])
    dx_mm = np.diff(x_extent_mm) / (nxyz - 1)  # mm
    dy_mm = dx_mm * np.diff(y_extent_mm) / np.diff(x_extent_mm)
    dz_mm = dx_mm * np.diff(z_extent_mm) / np.diff(x_extent_mm)

    # ------- T (time) dependent variables from .mat file -------
    sampling_frequency_MHz = int(dict_w_vars["fs"])

    # ------- M (patches) dependent variables from .mat file --------
    no_sub_x = int(dict_w_vars["nsubx"])
    no_sub_y = int(dict_w_vars["nsuby"])

    # ------- Create transducer (Linear array equivalent) -------
    tx = transducers.MatrixArrayTransducer(
        N_elem_x=tx_N_elem_x,
        N_elem_y=tx_N_elem_y,
        elem_width_mm=tx_elem_width_mm,
        elem_height_mm=tx_elem_height_mm,
        kerf_x_mm=tx_kerf_x_mm,
        kerf_y_mm=tx_kerf_y_mm,
        no_sub_x=no_sub_x,
        no_sub_y=no_sub_x,
        frequency_Hz=tx_frequency_Hz,  # MHz),
    )

    # Optional: compute/aply delays / apodization
    tx.compute_delays(focus_mm=focus_mm)

    # -------- DEFINE GRID (keep odd-number-of-points logic) --------

    # Use PyField API which accepts extents + spacing (same style as notebook)
    field_info_mm = {
        "x_extent": x_extent_mm,
        "y_extent": y_extent_mm,
        "z_extent": z_extent_mm,
        "dx": dx_mm,
        "dy": dy_mm,
        "dz": dz_mm,
    }

    # ------------------ Compute the field --------------------
    field_solver = PyField(tx, fs=sampling_frequency_MHz * 1e6, c=c)

    # first computation includes setup time for compiling and
    # optimizing kernels. Subsequent it wont be counted.
    start = 0
    if i == 0:
        start = 3
        x, y, z, pr_naive = field_solver(field_info_mm, method="naive")
        # returns grids and field (pressure)
        x, y, z, pr_sdi = field_solver(field_info_mm, method="sdi")

        x, y, z, pr_auto = field_solver(field_info_mm, method="auto")

    for rep in range(repetition):
        x, y, z, pr_naive = field_solver(field_info_mm, method="naive")

    for rep in range(repetition):
        x, y, z, pr_sdi = field_solver(field_info_mm, method="sdi")

    for rep in range(repetition):
        x, y, z, pr_auto = field_solver(field_info_mm, method="auto")

    # From the timing logs compute average times and std deviations
    timelogs = field_solver.sir_running_time_log
    print(f"t_log ({len(timelogs[start:])}) : {timelogs[start:]}")
    times_naive = np.sort(timelogs[start : start + repetition])
    times_sdi = np.sort(timelogs[start + repetition : start + 2 * repetition])
    times_auto = np.sort(timelogs[start + 2 * repetition : start + 3 * repetition])

    # sometimes some seconds are added due to OS processes
    # to avoid biasing the mean, we will perform 7 repetitions and takeout
    # and average the 5 middle values

    time_naive = np.mean(times_naive[1:-1])
    time_sdi = np.mean(times_sdi[1:-1])
    time_auto = np.mean(times_auto[1:-1])
    std_naive = np.std(times_naive[1:-1])
    std_sdi = np.std(times_sdi[1:-1])
    std_auto = np.std(times_auto[1:-1])

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
        "tx_N_element_x": tx_N_elem_x,
        "tx_N_element_y": tx_N_elem_y,
        "tx_element_height_mm": tx_elem_height_mm,
        "tx_element_width_mm": tx_elem_width_mm,
        "tx_pitch_mm": tx_pitch_mm,
        "tx_kerf_x_mm": tx_kerf_x_mm,
        "tx_kerf_y_mm": tx_kerf_y_mm,
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
    # print(f"Saving {mat.stem}_pyfield.npz with keys: {list(data.keys())}")
    # save data with numpy
    save_name = mat.stem + "_pyfield.npz"
    np.savez_compressed(BASE / save_name, **data)

    # Optional: visualize (requires interactive backend)
    # pyfield.plot_field_planes(x, y, z, pr_naive)
