"""Spatial Impulse Response wrapper class."""

import time

import numpy as np

from pyfield.utilities.helper_functions import (
    check_valid_field_points as check_field_points,
    compute_sub_elem_attributes,
    compute_time_grid,
)

from .farfield_rect_patch import compute_h_sir


class h_sir:
    """Compute the Spatial Impulse Response for a given transducer.

    Parameters
    ----------
    transducer : TransducerBase
        Transducer instance with geometry and beamforming state.
    c : float, optional
        Speed of sound in m/s. Default 1540.
    fs : float, optional
        Sampling frequency in Hz. Default 200 MHz.
    alpha0 : float, optional
        Attenuation coefficient in dB/(MHz^y cm). Default 0.5.
    freq_power : float, optional
        Frequency power law exponent. Default 1.0.
    """

    def __init__(self, transducer, *, c=1540.0, fs=200e6, alpha0=0.5, freq_power=1.0):
        self.tx = transducer
        self.fc = transducer.fc  # Hz
        (
            self.centers_sub_elem,
            self.apodization_sub_elem,
            self.delays_sub_elem,
            self.M,
            self.range_k,
            self.wx_arr,
            self.wy_arr,
        ) = compute_sub_elem_attributes(transducer)

        self.wx = float(self.wx_arr.max())
        self.wy = float(self.wy_arr.max())
        self.delays = transducer.delays
        self.apodization = transducer.apodization

        # Initialize logs
        self.mean_range_k_log = []
        self.T_log = []
        self.P_log = []
        self.sir_running_time_log = []

        # Medium parameters
        self.c = c  # m/s
        self.alpha0 = alpha0  # dB/(MHz^y cm)
        self.freq_power = freq_power  # freq_power law exponent
        self.fs = fs  # Hz

        # Initialize stored params to None
        self.x = None
        self.y = None
        self.z = None

    def __call__(self, field_points_mm, *, method="auto"):
        self.x, self.y, self.z, points = check_field_points(field_points_mm)

        if method not in ["auto", "naive", "sdi", None]:
            raise ValueError("method must be None or 'auto', 'naive', or 'sdi'.")
        if method == "naive":
            method_flag = 0
        elif method == "sdi":
            method_flag = 1
        else:
            method_flag = None

        P, M = points.shape[0], self.M

        print(f"Computing SIR for {P} points and {M} patches...")
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

        self.range_k = info_struct["range_k_matrix"]

        runtime_sir = time.time() - startSIR
        # Store information
        self.P_log.append(P)
        self.T_log.append(T)
        self.mean_range_k_log.append(np.mean(self.range_k))
        self.sir_running_time_log.append(runtime_sir)

        print(f"Transducer SIR computed in {runtime_sir:.2f} seconds...")
        return t0, h_sir.T

    def set_field(self, attribute_name, value):
        """Set an attribute value by name.

        Parameters
        ----------
        attribute_name : str
            Name of the attribute to set.
        value : object
            New value for the attribute.
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
            self.range_k,
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
            self.range_k,
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
        return f"h_sir(transducer={self.tx}, c={self.c} m/s, fs={self.fs} Hz, fc={self.fc} Hz, alpha0={self.alpha0} dB/(MHz^y cm), freq_power={self.freq_power})"
