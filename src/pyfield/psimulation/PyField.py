import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from pyfield.h_sir.farfield_rect_patch import compute_h_sir
from pyfield.utilities.helper_functions import (
    check_field_points,
    compute_sub_elem_attributes,
    compute_time_grid,
    reshape_to_mapped_points,
)

inv_2pi = 1 / (2 * np.pi)


class PyField:
    def __init__(self, transducer, *, c=1540.0, fs=200e6, alpha0=0, freq_power=1.0):
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
        if isinstance(points, (np.ndarray, list, tuple)):
            if isinstance(points, (list, tuple)):
                points = np.array(points, dtype=np.float32)
            # check shape
            if points.ndim < 2:
                if points.shape[0] == 3:
                    points = points.reshape(1, 3)
                else:
                    raise ValueError("points must 1D (3,) or 2D (N,3).")
            elif points.ndim == 2:
                pass
            else:
                raise ValueError("points must 1D (3,) or 2D (N,3).")

        if method not in ["auto", "naive", "sdi", None]:
            raise ValueError("method must be None or 'auto', 'naive', or 'sdi'.")
        if method == "naive":
            method_flag = 0
        elif method == "sdi":
            method_flag = 1
        else:
            method_flag = None

        P, M = points.shape[0], self.M

        print(f"\nComputing SIR for {P} points and {M} patches...")
        startSIR = time.time()

        time_grid, t0, dt, T = compute_time_grid(
            P,
            M,
            points,
            self.centers_sub_elem,
            self.wx,
            self.wy,
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

        runtime_sir = time.time() - startSIR
        # Store information
        self.P_log.append(P)
        self.T_log.append(T)
        self.mean_sub_elem_delta_k_log.append(np.mean(self.sub_elem_delta_k))
        self.sir_running_time_log.append(runtime_sir)

        print(f"Transducer SIR computed in {runtime_sir:.3f} seconds...")
        return t0, h_sir.T

    def from_sir_to_pressure(self, h_sir, x, y, z, batch_size=2048, max_workers=None):
        """
        Compute the pressure field from the Spatial Impulse Response (SIR) in parallel.
        """
        start_time = time.time()
        n_points = h_sir.shape[1]
        # Frequency vector
        freq_vect = np.linspace(0, self.fs, h_sir.shape[0])
        idx = np.argmin((freq_vect - self.fc) ** 2)

        fft_results = np.zeros(n_points, dtype=np.float32)

        def process_batch(start):
            end = min(start + batch_size, n_points)
            fft_batch = np.fft.fft(h_sir[:, start:end], axis=0)
            return start, end, np.abs(fft_batch[idx, :])

        # Parallel loop
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for start, end, vals in executor.map(
                process_batch, range(0, n_points, batch_size)
            ):
                fft_results[start:end] = vals

        # Reshape back to 3D grid
        amp_sir_at_tx_freq = reshape_to_mapped_points(x, y, z, fft_results)
        print(
            f"Pressure computed from SIR in {time.time() - start_time:.2f} seconds..."
        )
        return amp_sir_at_tx_freq

    def __call__(self, field_points_mm, *, method="auto", normalize=False):
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
        start = time.time()
        x, y, z, points = check_field_points(field_points_mm)
        t0, h_sir = self.compute_sir(points, method=method)
        pressure_field = self.from_sir_to_pressure(h_sir, x, y, z)
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
