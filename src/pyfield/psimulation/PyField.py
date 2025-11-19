import time

import numpy as np

from pyfield.h_sir.farfield_rect_patch import compute_h_sir
from pyfield.utilities.helper_functions import (
    check_field_points,
    compute_sub_elem_attributes,
    compute_time_grid,
)

from .sir_to_pressure import (
    from_sir_to_monochromatic_pressure,
    from_sir_to_pressure,
)

inv_2pi = 1 / (2 * np.pi)


class PyField:
    def __init__(
        self,
        transducer,
        *,
        rho=1.0,
        c=1540.0,
        fs=200e6,
        alpha0=0,
        freq_power=1.0,
    ):
        self.tx = transducer
        self.fc = transducer.fc  # Hz
        (
            self.centers_sub_elem,
            self.apodization_sub_elem,
            self.delays_sub_elem,
            self.M,
            self.sub_elem_delta_k,
        ) = compute_sub_elem_attributes(transducer)

        # compute patch centers_sub_elem/apodization/delays once
        elem_height = transducer.elem_height / transducer.no_sub_y
        elem_width = transducer.elem_width / transducer.no_sub_x
        self.wx = elem_width
        self.wy = elem_height
        self.delays = transducer.delays
        self.apodization = transducer.apodization

        # Medium parameters
        self.rho = rho  # kg/m^3
        self.c = c  # m/s
        self.alpha0 = alpha0  # dB/(MHz^y cm)
        self.freq_power = freq_power  # freq_power law exponent
        self.fs = fs  # Hz
        lambda_m = c / self.fc  # m
        print(
            f"Min distance must be >> w^2/(4*lambda): {max(self.wx, self.wy) ** 2 / 4 / lambda_m * 1e3:.4f} mm"
        )

        # Initialize logs
        self.mean_sub_elem_delta_k_log = []
        self.T_log = []
        self.P_log = []
        self.sir_running_time_log = []

    def compute_sir(self, points, *, method="auto"):
        x, y, z, points = check_field_points(points)

        if method not in ["auto", "naive", "sdi", None]:
            raise ValueError("method must be None or 'auto', 'naive', or 'sdi'.")
        if method == "naive":
            method_flag = 0
        elif method == "sdi":
            method_flag = 1
        else:
            method_flag = None

        P, M = points.shape[0], self.M

        print(f"\nComputing SIR for {P} points and {M} patches with method {method}...")
        startSIR = time.time()

        time_grid, t0, dt, T = compute_time_grid(
            P,
            M,
            points,
            self.centers_sub_elem,
            self.wx / self.tx.no_sub_x,
            self.wy / self.tx.no_sub_y,
            self.c,
            self.fs,
            self.delays,
        )

        h_sir, self.sub_elem_delta_k = compute_h_sir(
            P,
            M,
            T,
            dt,
            time_grid,
            points,
            self.centers_sub_elem,
            self.wx,
            self.wy,
            1 / self.c,
            self.fs,
            self.apodization_sub_elem,
            self.delays_sub_elem,
            method_flag,
        )
        # h_sir shape (P, T)

        runtime_sir = time.time() - startSIR

        # The t0 is understimated due to the way time_grid is computed
        # If we want to reduce time_grid size and adjust t0
        # to where h_sir entries are not zero, we can compute it as follows:
        try:
            s = h_sir.sum(axis=0)  # shape (T,)
            diff = np.diff(s)  # shape (T-1,)
            nz = np.nonzero(diff > 0)[0]
            idx = int(nz[0]) if nz.size else None
            tbefore = t0
            t0 = tbefore + idx * dt
            if idx / T > 0.5:
                print(
                    f"Warning: Adjusted t0 from {tbefore:.2e} s to {t0:.2e} s. corresponding to {idx}/{T}={idx / T * 100:.4f}% idx of time grid."
                )
            h_sir = h_sir[:, idx:]
            T = h_sir.shape[1]
        except Exception as e:
            print(f"Could not adjust t0 due to error: {e}")

        # Store information
        self.P_log.append(P)
        self.T_log.append(T)
        self.mean_sub_elem_delta_k_log.append(np.mean(self.sub_elem_delta_k))
        self.sir_running_time_log.append(runtime_sir)

        print(f"Transducer SIR computed in {runtime_sir:.3f} seconds...")
        return h_sir.T, t0, x, y, z

    def __call__(
        self,
        field_points_mm,
        *,
        method="auto",
        normalize=False,
        monochromatic=True,
        excitation=None,
    ):
        """
        Compute the pressure field at specified points.
        Parameters
        ----------
        field_points_mm : array-like or dict
            Points where the pressure field is to be computed. Can be a dict with grid parameters or an array of points.
        method : str, optional
            Method for SIR computation. Options are 'auto', 'naive', or 'sdi'. Default is 'auto'.
        normalize : bool, optional
            If True, normalize the pressure field to its maximum value. Default is False.
        sort_meshgrid : bool, optional
            If True, organize the field points into a sorted meshgrid. Default is True.
        Returns
        -------
        x, y, z : 1D arrays
            Coordinates of the grid points in mm.
        pressure_field : 3D array
            Computed pressure field at the specified points.
        """
        if excitation is not None:
            monochromatic = False

        start = time.time()
        h_sir, self.t0, x, y, z = self.compute_sir(field_points_mm, method=method)
        if monochromatic:
            pressure_field = from_sir_to_monochromatic_pressure(
                h_sir, x, y, z, self.fc, self.fs
            )
        else:
            pressure_field = from_sir_to_pressure(
                h_sir, x, y, z, self.fs, rho=self.rho, excitation=excitation
            )
        print(f"Pressure field computed in {time.time() - start:.2f} seconds... \n")

        if normalize:
            pressure_field = pressure_field / pressure_field.max()

        return x, y, z, pressure_field

    def set_field(self, attribute_name, value):
        if not hasattr(self, attribute_name):
            self.__repr__()
            raise AttributeError(
                f"{attribute_name} is not a valid attribute of PyField."
            )
        setattr(self, attribute_name, value)
        print(f"Attribute '{attribute_name}' updated.")

    def compute_delays(self, focus_mm):
        self.delays = self.tx.compute_delays(focus_mm=focus_mm, c=self.c)
        self.compute_sub_elem_attributes()

    def compute_apodization(self, focus_mm, FoverD=1, apodization_type="rect"):
        self.apodization = self.tx.compute_apodization(
            focus_mm=focus_mm, FoverD=FoverD, apodization_type=apodization_type
        )
        self.compute_sub_elem_attributes()

    def summary(self):
        """
        Print a summary of the PyField object.
        """
        print("----------PyField Summary:----------")
        for key, value in self.__dict__.items():
            if key == "field":
                if value is not None:
                    print(f"{key}: pressure field with shape {value.shape}")
                else:
                    print(f"{key}: None")
            elif key in ["x", "y", "z"]:
                if value is not None:
                    print(f"{key}: grid with shape {value.shape}")
                else:
                    print(f"{key}: None")
            else:
                print(f"{key}: {value}")

    def __repr__(self):
        """
        String representation of the PyField object.

        Returns
        -------
        str
            A string representation of the PyField object.
        """
        return f"PyField(transducer={self.tx}, c={self.c} m/s, fs={self.fs} Hz, fc={self.fc} Hz, alpha0={self.alpha0} dB/(MHz^y cm), freq_power={self.freq_power})"
