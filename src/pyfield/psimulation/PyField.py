"""PyField pressure field simulation class."""

import time

import numpy as np

from pyfield.h_sir.farfield_rect_patch import compute_h_sir
from pyfield.utilities.helper_functions import (
    check_valid_field_points,
    compute_sub_elem_attributes,
    compute_time_grid,
    create_3D_spatial_grid_from_points,
)

from .sir_to_pressure import (
    from_sir_to_monochromatic_pressure,
    from_sir_to_pressure,
)

inv_2pi = 1 / (2 * np.pi)


class PyField:
    """Compute the acoustic pressure field for a given transducer.

    Parameters
    ----------
    transducer : Transducer
        The transducer object containing the transducer parameters and attributes.
    rho : float, optional
        Density of the medium in kg/m^3. Default is 1.0 kg/m^3.
    c : float, optional
        Speed of sound in the medium in m/s. Default is 1540 m/s.
    fs : float, optional
        Sampling frequency in Hz. Default is 200 MHz.
    alpha0 : float, optional
        Attenuation coefficient at 1 MHz in dB/(MHz^y cm). Default is 0 dB/(MHz^y cm).
    freq_power : float, optional
        Exponent for frequency power law in attenuation. Default is 1.0.
    verbose : bool, optional
        If True, print verbose output during computations. Default is True.
    monochromatic : bool, optional
        If True, compute monochromatic pressure field. If False, compute broadband
        pressure field. Default is True.


    Attributes
    ----------
    tx : Transducer
        The transducer object containing the transducer parameters and attributes.
    fc : float
        Center frequency of the transducer in Hz.
    centers_sub_elem : ndarray
        Centers of the sub-elements in meters.
    apodization_sub_elem : ndarray
        Apodization values for the sub-elements.
    delays_sub_elem : ndarray
        Delays for the sub-elements in seconds.
    M : int
        Total number of sub-elements (patches).
    sub_elem_delta_k : ndarray
        Delta k values for the sub-elements in 1/m.
    wx : float
        Width of the sub-elements in meters.
    wy : float
        Height of the sub-elements in meters.
    delays : ndarray
        Delays for the main elements in seconds.
    apodization : ndarray
        Apodization values for the main elements.
    rho : float
        Density of the medium in kg/m^3.
    fs : float
        Sampling frequency in Hz.
    c : float (Temporarily unused)
        Speed of sound in the medium in m/s.
    alpha0 : float (Temporarily unused)
        Attenuation coefficient at 1 MHz in dB/(MHz^y cm).
    freq_power : float (Temporarily unused)
        Exponent for frequency power law in attenuation.
    """

    def __init__(
        self,
        transducer,
        *,
        rho=1.0,
        c=1540.0,
        fs=200e6,
        alpha0=0,
        freq_power=1.0,
        verbose=True,
        monochromatic=True,
    ):
        self.tx = transducer
        self.fc = transducer.fc  # Hz
        (
            self.centers_sub_elem,
            self.apodization_sub_elem,
            self.delays_sub_elem,
            self.M,
            self.sub_elem_delta_k,
            self.wx_arr,
            self.wy_arr,
        ) = compute_sub_elem_attributes(transducer)

        # Scalar max patch size — used for time-grid bounds (conservative)
        self.wx = float(self.wx_arr.max())
        self.wy = float(self.wy_arr.max())
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
        self.pressure_calculation_time_log = []
        self.verbose = verbose
        self.monochromatic = monochromatic

    def compute_sir(self, points, *, method="auto", adjust_t0=True, verbose=False):
        """
        Compute the Spatial Impulse Response (SIR) for the given field points.

        Parameters
        ----------
        points : array-like
            An array of shape (P, 3) containing the coordinates of the field points in meters.
        method : str, optional
            Method for SIR computation. Options are 'auto', 'naive', or 'sdi'. Default
            is 'auto'.
        adjust_t0 : bool, optional
            If True, adjust the time grid to start from the first non-zero entry in h_sir.
            Default is True.
        verbose : bool, optional
            If True, print verbose output during computations. Default is None, which
            means it will use the instance's verbose attribute.

        Returns
        -------
        h_sir : ndarray
            The computed Spatial Impulse Response (SIR) with shape (T, P).
        t0 : float
            Start time of the time grid in seconds.

        Examples
        --------
        >>> import numpy as np
        >>> from pyfield import PyField
        >>> # Create a transducer (example parameters)
        >>> transducer = LinearArrayTransducer(
        ...     n_elements=128,
        ...     element_width_mm=0.108,
        ...     element_height_mm=1.5,
        ...     elevation_focus_mm=8,
        ...     kerf_mm=0.002,
        ...     no_sub_x=1,
        ...     no_sub_y=10,
        ...     frequency_Hz=12.5e6,
        ... )
        >>> # Initialize PyField with the transducer
        >>> field = PyField(transducer)
        >>> # Define field points (example with meshgrid, can also be a dict for grid
        parameters)
        >>> x = np.linspace(-0.5, 0.5, 10)  # mm
        >>> y = np.linspace(-0.5, 0.5, 10)  # mm
        >>> z = np.linspace(0, 1.5, 10)     # mm
        >>> field_points_mm = np.array(np.meshgrid(x, y, z)).T.reshape(-1, 3)  # (P, 3)
        >>> # Compute SIR
        >>> h_sir, t0 = field.compute_sir(field_points_mm, method='auto',
        ...     adjust_t0=True, verbose=True)
        """

        if verbose is None:
            verbose = self.verbose

        points = check_valid_field_points(points)

        if method not in ["auto", "naive", "sdi", "SDI", None]:
            raise ValueError("method must be None or 'auto', 'naive', or 'sdi'.")
        if method == "naive":
            method_flag = 0
        elif method == "sdi" or method == "SDI":
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
            self.wx,
            self.wy,
            self.c,
            self.fs,
            self.delays,
            verbose=verbose,
        )

        h_sir, info_struct = compute_h_sir(
            P,
            M,
            T,
            dt,
            time_grid,
            points,
            self.centers_sub_elem,
            self.wx_arr,
            self.wy_arr,
            1 / self.c,
            self.fs,
            self.apodization_sub_elem,
            self.delays_sub_elem,
            method_flag,
        )
        # h_sir shape (P, T)
        self.sub_elem_delta_k = info_struct["range_k_matrix"]

        runtime_sir = time.time() - startSIR

        # The t0 is understimated due to the way time_grid is computed
        # If we want to reduce time_grid size and adjust t0
        # to where h_sir entries are not zero, we can compute it as follows:
        if adjust_t0:
            try:
                t_start = info_struct["min_time"]
                t_end = info_struct["max_time"]
                idx_start = max(0, int(np.floor((t_start - t0) / dt)))
                idx_end = min(
                    T, int(np.ceil((t_end - t0) / dt)) + 1
                )  # +1 to include last index
                h_sir = h_sir[:, idx_start:idx_end]
                if verbose:
                    print(
                        f"Adjusted t0 : {t0: .2e} -> {t0 + idx_start * dt:.2e} s, and tN : {time_grid[-1]:.2e} -> {t0 + (idx_end - 1) * dt:.2e} s, \n h_sir size: {P} x {h_sir.shape[1]} (was {P} x {T})"
                    )
            except Exception as e:
                print(f"Could not adjust t0 due to error: {e}")

        # Store information
        self.P_log.append(P)
        self.T_log.append(T)
        self.mean_sub_elem_delta_k_log.append(np.mean(self.sub_elem_delta_k))
        self.sir_running_time_log.append(runtime_sir)

        if verbose:
            print(f"Transducer SIR computed in {runtime_sir:.3f} seconds...")
        return h_sir.T, t0

    def __call__(
        self,
        field_points_mm,
        *,
        method="auto",
        normalize=False,
        monochromatic=None,
        excitation=None,
        create_meshgrid=False,
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
            If True, organize the field points into a sorted meshgrid with no repetition
            of points. Default is True.

        Returns
        -------
        x, y, z : 1D arrays
            Coordinates of the grid points in mm.
        pressure_field : 3D array
            Computed pressure field at the specified points.

        Examples
        --------
        >>> import numpy as np
        >>> from pyfield import PyField
        >>> # Create a transducer (example parameters)
        >>> transducer = LinearArrayTransducer(
        ...     n_elements=128,
        ...     element_width_mm=0.108,
        ...     element_height_mm=1.5,
        ...     elevation_focus_mm=8,
        ...     kerf_mm=0.002,
        ...     no_sub_x=1,
        ...     no_sub_y=10,
        ...     frequency_Hz=12.5e6,
        ... )
        >>> # Initialize PyField with the transducer
        >>> field = PyField(transducer)
        >>> # Define field points (as a dict for grid parameters)
        >>> field_points_mm = {
        ...     'x_extent_mm': [-0.5, 0.5],
        ...     'y_extent_mm': [-0.5, 0.5],
        ...     'z_extent_mm': [0, 1.5],
        ...     'dx_mm': 0.1,
        ...     'dy_mm': 0.1,
        ...     'dz_mm': 0.1,
        ... }
        >>> # Compute pressure field
        >>> x, y, z, pressure_field = field(
        ...     field_points_mm,
        ...     method='auto',
        ...     normalize=True,
        ...     monochromatic=True)
        >>> # Pulsed excitation example (plane xz wave with 2 cycles at fc)
        >>> time_excitation = np.arange(0, 2 / field.fc, 1 / field.fs)
        >>> excitation = np.sin(2 * np.pi * field.fc * time_excitation)
        >>> plane_points_mm - {
        ...     'x_extent_mm': [-0.5, 0.5],
        ...     'y_extent_mm': [0, 0],
        ...     'z_extent_mm': [0, 1.5],
        ...     'dx_mm': 0.1,
        ...     'dy_mm': 0,
        ...     'dz_mm': 0.1,
        ... }
        >>> x_plane, y_plane, z_plane, pressure_field_plane = field(
        ...     plane_points_mm, excitation=excitation)
        """
        if monochromatic is None:
            monochromatic = self.monochromatic
        if excitation is not None:
            monochromatic = False

        x, y, z, field_points_mm = create_3D_spatial_grid_from_points(
            field_points_mm, create_meshgrid=create_meshgrid
        )

        start = time.time()
        h_sir, self.t0 = self.compute_sir(
            field_points_mm, method=method, verbose=self.verbose
        )
        if monochromatic:
            pressure_field = from_sir_to_monochromatic_pressure(
                h_sir, x, y, z, self.fc, self.fs
            )
        else:
            pressure_field = from_sir_to_pressure(
                h_sir, x, y, z, self.fs, rho=self.rho, excitation=excitation
            )

        self.pressure_calculation_time_log.append(time.time() - start)
        print(
            f"Pressure field computed in {self.pressure_calculation_time_log[-1]:.2f} seconds... \n"
        )

        # Clean numerical errors
        # pr_max = np.max(np.abs(pressure_field))
        # pressure_field = np.clip(pressure_field, a_min=pr_max * 0.0001, a_max=None)

        if normalize:
            pressure_field = pressure_field / pressure_field.max()

        return x, y, z, pressure_field

    def set_field(self, attribute_name, value):
        """Set the value of a specified attribute of the PyField object.

        Parameters
        ----------
        attribute_name : str
            The name of the attribute to be updated.
        value : object
            The new value to be assigned to the specified attribute.
        """
        if not hasattr(self, attribute_name):
            self.__repr__()
            raise AttributeError(
                f"{attribute_name} is not a valid attribute of PyField."
            )
        setattr(self, attribute_name, value)
        print(f"Attribute '{attribute_name}' updated.")

    def compute_delays(self, focus_mm):
        """Recompute element delays and refresh sub-element attributes.

        Parameters
        ----------
        focus_mm : array-like
            Focal point in mm.
        """
        self.delays = self.tx.compute_delays(focus_mm=focus_mm, c=self.c)
        (
            self.centers_sub_elem,
            self.apodization_sub_elem,
            self.delays_sub_elem,
            self.M,
            self.sub_elem_delta_k,
            self.wx_arr,
            self.wy_arr,
        ) = compute_sub_elem_attributes(self.tx)
        self.wx = float(self.wx_arr.max())
        self.wy = float(self.wy_arr.max())

    def compute_apodization(self, focus_mm, FoverD=1, apodization_type="rect"):
        """Recompute element apodization and refresh sub-element attributes.

        Parameters
        ----------
        focus_mm : array-like
            Focal point in mm.
        FoverD : float, optional
            F-number for aperture sizing. Default 1.
        apodization_type : str, optional
            Window type. Default ``'rect'``.
        """
        self.apodization = self.tx.compute_apodization(
            focus_mm=focus_mm, FoverD=FoverD, apodization_type=apodization_type
        )
        (
            self.centers_sub_elem,
            self.apodization_sub_elem,
            self.delays_sub_elem,
            self.M,
            self.sub_elem_delta_k,
            self.wx_arr,
            self.wy_arr,
        ) = compute_sub_elem_attributes(self.tx)
        self.wx = float(self.wx_arr.max())
        self.wy = float(self.wy_arr.max())

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
