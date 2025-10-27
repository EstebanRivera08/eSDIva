import time

import numpy as np

import pyfield
import pyfield.transducers as transducers
from pyfield.psimulation import PyField


def main():
    # ------------------ Saving / options -----------------------
    frequency_MHz = 12.5
    c = 1540.0  # m/s

    # ------- focus and simulation window (input parameters) -------
    x_extent_mm = [-5, 5]
    y_extent_mm = [-5, 5]
    z_extent_mm = [1, 16]
    dxyz = 0.5  # mm
    dx_mm = dxyz
    dy_mm = dxyz
    dz_mm = dxyz
    focus_mm = [0.0, 0.0, 8]
    sampling_frequency_MHz = 100.0

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

    # Create transducer (Linear array equivalent)
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
    tx.compute_delays(focus_mm=focus_mm)
    # tx.compute_apodization(focus_mm=focus_mm, FoverD=1, apodization_type='rect')

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

    # for comparison compute twice (first computation includes setup time
    # for compiling and optimizing kernels)
    x, y, z, pressure_field = field_solver(
        field_info_mm
    )  # returns grids and field (pressure)

    for i in range(5):  # warmup runs
        x, y, z, pressure_field = field_solver(
            field_info_mm
        )  # returns grids and field (pressure)
    print(field_solver.sir_running_time_log)
    print(
        "x/y/z shapes:",
        x.shape,
        y.shape,
        z.shape,
        "pressure shape:",
        pressure_field.shape,
    )

    # Optional: visualize (requires interactive backend)
    pyfield.plot_field_planes(x, y, z, pressure_field)


if __name__ == "__main__":
    main()
