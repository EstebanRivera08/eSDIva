from time import time as TIME

import numpy as np
import pyvista as pv
from numba import njit, prange

import pysonogen


def create_simulation_grid(simulation_struct):
    """
    Create a simulation mesh for the ultrasound field.

    Parameters
    ----------
    simulation_grid_dict : dict
        Dictionary containing the simulation parameters:
        - x_extent : list
            The extent of the simulation in the x direction (in mm).
        - y_extent : list
            The extent of the simulation in the y direction (in mm).
        - z_extent : list
            The extent of the simulation in the z direction (in mm).
        - dx : float
            The grid spacing in the x direction (in mm).
        - dy : float
            The grid spacing in the y direction (in mm).
        - dz : float
            The grid spacing in the z direction (in mm).

    Returns
    -------
    grid_points : ndarray
        Array of points in the simulation space.
    """
    # Create a grid of points in the simulation space
    [x0, xf], [y0, yf], [z0, zf] = (
        simulation_struct["x_extent"],
        simulation_struct["y_extent"],
        simulation_struct["z_extent"],
    )
    dx, dy, dz = (
        simulation_struct["dx"],
        simulation_struct["dy"],
        simulation_struct["dz"],
    )

    Nx = int((xf - x0) / dx) if (dx != 0 and abs(xf - x0) > 1e-10) else 1
    Ny = int((yf - y0) / dy) if (dy != 0 and abs(yf - y0) > 1e-10) else 1
    Nz = int((zf - z0) / dz) if (dz != 0 and abs(zf - z0) > 1e-10) else 1
    if Nx % 2 == 0:
        Nx += 1
    if Ny % 2 == 0:
        Ny += 1
    if Nz % 2 == 0:
        Nz += 1

    # print(
    #     f"Creating grid with {Nx} x {Ny} x {Nz} points in x, y, z directions respectively."
    # )
    # print(f"Grid extents: x: [{x0}, {xf}], y: [{y0}, {yf}], z: [{z0}, {zf}]")
    x = np.linspace(x0, xf, Nx)
    y = np.linspace(y0, yf, Ny)
    z = np.linspace(z0, zf, Nz)
    # Create a meshgrid of points
    grid_points = np.array(np.meshgrid(x, y, z)).T.reshape(-1, 3)

    return x, y, z, grid_points * 1e-3


pi = np.pi
tolerance_apod = 1e-3


@njit
def compute_patch_sir(wx, wy, xp, yp, l, c0, apod, delay, sampling_rate_Hz):
    # Common sampling rate is 100 MHz
    # Then minimum time step is 0.01 us,
    # if apod < tolerance_apod:
    #     return 0, 0, 0, 0, 0
    epsilon = 1 / (sampling_rate_Hz)  # 1 ns
    xp_abs = abs(xp) * wx / c0  # us
    yp_abs = abs(yp) * wy / c0
    Dt1 = min(xp_abs, yp_abs)
    Dt2 = max(xp_abs, yp_abs)
    area = wx * wy / (2 * pi * l)

    t1 = l / c0 - 0.5 * (Dt1 + Dt2) + delay
    t2 = t1 + Dt1
    t3 = t1 + Dt2
    t4 = t1 + Dt1 + Dt2

    if 2 * Dt2 < epsilon:
        Dt2 = epsilon
    # trapezoid area
    h_max = area * apod / Dt2
    return t1, t2, t3, t4, h_max


@njit(parallel=True)
def compute_all_events(
    P, M, pts, centers, wx, wy, c, apods, delays, events, sampling_rate_Hz
):
    for p in prange(P):
        for i in range(M):
            dx = pts[p, 0] - centers[i, 0]
            dy = pts[p, 1] - centers[i, 1]
            dz = pts[p, 2] - centers[i, 2]
            dist = np.sqrt(dx * dx + dy * dy + dz * dz)

            xp, yp = dx / (dist), dy / (dist)
            t1, t2, t3, t4, h_max = compute_patch_sir(
                wx,
                wy,
                xp,
                yp,
                dist,
                c,
                apods[i],
                delays[i],
                sampling_rate_Hz,
            )
            events[p, i, 0] = t1
            events[p, i, 1] = t2
            events[p, i, 2] = t3
            events[p, i, 3] = t4
            events[p, i, 4] = h_max


@njit(parallel=True)
def accumulate_from_events(P, M, events, fs, t0, h_out):
    """
    Parallel accumulation of trapezoidal SIR contributions for all patches.
    events shape: (P, M, 5) storing t1, t2, t3, t4, h_max.
    h_out: (P, n2) output array, t0: start time, fs: sampling rate.
    """
    dt = 1.0 / fs
    n2 = h_out.shape[1]
    for p in prange(P):
        for i in range(M):
            t1, t2, t3, t4, h_max = (
                events[p, i, 0],
                events[p, i, 1],
                events[p, i, 2],
                events[p, i, 3],
                events[p, i, 4],
            )

            # find the first/last sample indices that could possibly overlap
            k_start = int(np.floor((t1 - t0) * fs) + 1)
            k_end = int(np.ceil((t4 - t0) * fs) + 1)

            # clamp to valid range
            if k_end < 0 or k_start >= n2:
                continue
            if k_start < 0:
                k_start = 0
            if k_end > n2:
                k_end = n2

            # loop over every sample that might see part of this trapezoid
            for k in range(k_start, k_end):
                t = t0 + k * dt
                # evaluate continuous trapezoid h(t)
                if t < t1 or t >= t4:
                    continue
                elif t < t2:
                    h = h_max * ((t - t1) / (t2 - t1))
                elif t < t3:
                    h = h_max
                else:
                    h = h_max * ((t4 - t) / (t4 - t3))

                # accumulate
                h_out[p, k] += h


class PyField:
    def __init__(self, transducer):
        self.tx = transducer
        self.c = 1540.0
        self.fs = 300e6  # Hz
        self.fc = transducer.fc  # Hz
        self.lambda_mm = self.c / self.fc
        # compute patch centers/apods/delays once
        el_h = self.tx.el_h / self.tx.no_sub_y
        el_w = self.tx.el_w / self.tx.no_sub_x
        centers, apods, delays = [], [], []
        for elem in range(self.tx.n_elements):
            for sub_elem in range(self.tx.no_sub_x * self.tx.no_sub_y):
                verts = self.tx.sub_quad_verts[
                    elem * (self.tx.no_sub_x * self.tx.no_sub_y) + sub_elem
                ]
                centers.append(verts.mean(axis=0))
                apods.append(self.tx.apodization[elem])
                delays.append(self.tx.delays[elem])
        self.centers = np.array(centers, dtype=np.float32)
        self.apods = np.array(apods, dtype=np.float32)
        self.delays = np.array(delays, dtype=np.float32)
        self.wx = el_w
        self.wy = el_h

        self.field = None
        self.x = self.y = self.z = None

    def spatial_impulse_response(self, field_points, return_all=False):
        start_comput_time = TIME()
        if not isinstance(field_points, np.ndarray):
            try:
                # Only use the grid_points (last element of the tuple)
                *_, field_points = create_simulation_grid(field_points)
            except Exception as e:
                raise ValueError(
                    "Invalid field_points input. It should be a numpy array or a dictionary with simulation parameters."
                ) from e

        pts = np.atleast_2d(field_points).astype(np.float32)
        P, M = pts.shape[0], self.centers.shape[0]

        print(f"Computing SIR for {P} points and {M} patches...")
        # allocate events
        events = np.zeros((P, M, 5), dtype=np.float32)
        # tqdm.write("Computing all patch events...")
        compute_all_events(
            P,
            M,
            pts,
            self.centers,
            self.wx,
            self.wy,
            self.c,
            self.apods,
            self.delays,
            events,
            self.fs,
        )
        events_time = TIME()
        print(
            f"Events patch - field points computed in: {events_time - start_comput_time:.4f} seconds."
        )
        # build global time vector from real event times
        all_times = np.unique(events[:, :, 0:4].ravel())
        all_times.sort()
        t0, tN = all_times[0], all_times[-1]
        # create sampling grid
        dt = 1.0 / self.fs
        num_samples = int(np.ceil((tN - t0) * self.fs)) if tN > t0 else 1
        # next power of two
        n2 = 2 ** max(int(np.ceil(np.log2(num_samples))), 5)
        t_global = t0 + np.arange(n2, dtype=np.float32) * dt
        h_out = np.zeros((P, n2), dtype=np.float32)
        # tqdm.write("Accumulating SIR from events...")
        accumulate_from_events(P, M, events, self.fs, t0, h_out)
        print(f"Accumulation of events elapsed in: {TIME() - events_time:.4f} seconds.")

        if return_all:
            return t_global, h_out.T, events
        return t0, h_out.T

    def compute_pr_from_sir(self, h_sir, x, y, z):
        """
        Compute the pressure field from the Spatial Impulse Response (SIR).

        Parameters
        ----------
        field_points : ndarray
            Array of points in the simulation space.

        Returns
        -------
        pressure : ndarray
            The computed pressure field.
        """
        start_time = TIME()
        # Reshape the SIR to match the grid dimensions
        # print(f"Original h shape: {h_sir.shape}")
        spatial_impulse_response_field = h_sir.reshape(
            -1, z.shape[0], x.shape[0], y.shape[0]
        ).transpose(0, 2, 3, 1)
        # print(f"Reshaped h shape: {spatial_impulse_response_field.shape}")

        # Perform FFT along the first axis
        spatial_impulse_response_field_FT = np.fft.fft(
            spatial_impulse_response_field, axis=0
        )
        # Generate the frequency vector
        freq_vect = np.linspace(0, self.fs, spatial_impulse_response_field_FT.shape[0])

        # Find the index of the desired frequency
        idx_freq = np.argmin((freq_vect - self.fc) ** 2)
        print(
            f"Looking for fc: {self.fc} Hz, found : {freq_vect[idx_freq]} Hz, in {TIME() - start_time:.2f} seconds."
        )

        # Amplitude for the given frequency
        amp_response_tx_freq = np.abs(
            spatial_impulse_response_field_FT[idx_freq, :, :, :]
        )

        return amp_response_tx_freq

    def compute_pressure_field(self, field_info, *, normalize=True, inplace=False):
        """
        Compute the pressure field from the Spatial Impulse Response (SIR).

        Parameters
        ----------
        field_info : dict
        - x_extent : list
            The extent in mm of the simulation in the x direction.
        - y_extent : list
            The extent in mm of the simulation in the y direction.
        - z_extent : list
            The extent in mm of the simulation in the z direction.
        - dx : float
            The grid spacing in mm along the x direction.
        - dy : float
            The grid spacing in mm along the y direction.
        - dz : float
            The grid spacing in mm along the z direction.

        normalize : bool, optional
            If True, normalize the pressure field to the maximum value. Default is True.

        inplace : bool, optional
            If True, store the pressure field in the instance variables. If False, return the pressure field and grid points. Default is True.

        Returns
        -------
        pressure : ndarray
            The computed pressure field.
        """
        start_time = TIME()
        # print("Creating simulation grid...")
        x, y, z, grid_points = create_simulation_grid(field_info)
        # print("Computing spatial impulse response...")
        start_time, h_sir = self.spatial_impulse_response(grid_points)
        pressure_field = self.compute_pr_from_sir(h_sir, x, y, z)

        print(f"Pressure field computed in: {TIME() - start_time:.4f} seconds.")
        # print(f"Pressure field shape: {pressure_field.shape}")
        if normalize:
            pressure_field = pressure_field / np.max(pressure_field)

        if inplace:
            self.field = pressure_field
            self.x = x
            self.y = y
            self.z = z
        else:
            return pressure_field, x, y, z

    def set_field(self, name_struct_str, value_float):
        """
        Dynamically modifies a class property if it exists.

        Parameters
        ----------
        name_struct_str : str
            The name of the property to modify.
        value_float : float
            The new value to assign to the property.

        Raises
        ------
        AttributeError
            If the property does not exist in the class.
        TypeError
            If the value is not a float.
        """
        if not isinstance(value_float, (float, int)):  # Allow integers as well
            raise TypeError(
                f"The value must be a float or int, got {type(value_float).__name__}."
            )

        if hasattr(self, name_struct_str):
            setattr(self, name_struct_str, value_float)
            print(f"Property '{name_struct_str}' updated to {value_float}.")
        else:
            raise AttributeError(
                f"Property '{name_struct_str}' does not exist in the class."
            )

    def get_mesh(self):
        """
        Get the mesh of the pressure field.

        Returns
        -------
        pv_mesh : pyvista.PolyData
            The mesh of the pressure field.
        """
        if self.field is None or self.x is None or self.y is None or self.z is None:
            raise ValueError(
                "Pressure field has not been computed yet. Call compute_pressure_field() first."
            )

        return pysonogen.compute_pressure_vol_mesh(self.field, self.x, self.y, self.z)

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

    def clean(self):
        """
        Clean the PyField object by removing the pressure field and grid points.
        """
        self.field = None
        self.x = None
        self.y = None
        self.z = None
        print("PyField object cleaned.")

    def __call__(self, field_info, *, normalize=True, inplace=False):
        """
        Make the class callable. Calls the compute_pressure_field method.

        Parameters
        ----------
        field_info : dict
            The input field information for the pressure computation.
        normalize : bool, optional
            If True, normalize the pressure field. Default is True.
        inplace : bool, optional
            If True, store the pressure field in the instance variables. Default is False.

        Returns
        -------
        pressure : ndarray
            The computed pressure field.
        """

        return self.compute_pressure_field(
            field_info, normalize=normalize, inplace=inplace
        )

    def __repr__(self):
        """
        String representation of the PyField object.

        Returns
        -------
        str
            A string representation of the PyField object.
        """
        return f"PyField(transducer={self.tx}, c={self.c}, fs={self.fs}, fc={self.fc}, lambda_mm={self.lambda_mm})"
