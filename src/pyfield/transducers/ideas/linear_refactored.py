"""
Refactored LinearArrayTransducer that inherits from TransducerBase.

This version demonstrates how to adapt existing transducers to the new
architecture. The public API is unchanged for backward compatibility.
"""

import warnings
from time import time as TIME
from typing import Optional, Tuple, List

import matplotlib.pyplot as plt
import numpy as np

from pyfield.transducers.base import TransducerBase
from pyfield.transducers import validators, geometry_utils


class LinearArrayTransducer(TransducerBase):
    """
    Linear array transducer with rectangular patch discretization.
    
    This is a refactored version that inherits from TransducerBase while
    maintaining full backward compatibility with the original API.

    Parameters
    ----------
    n_elements : int
        Number of elements in the array.
    element_width_mm : float
        Width of each element in millimeters.
    element_height_mm : float
        Height of each element in millimeters.
    kerf_mm : float
        Gap between elements in millimeters.
    no_sub_x : int
        Number of subdivisions in x-direction (element width).
    no_sub_y : int
        Number of subdivisions in y-direction (element height).
    elevation_focus_mm : float, optional
        Elevation focus distance in millimeters. If None, flat surface.
    frequency_Hz : float, optional
        Operating frequency in Hz.
    """

    def __init__(
        self,
        *,
        n_elements,
        element_width_mm,
        element_height_mm,
        kerf_mm,
        no_sub_x,
        no_sub_y,
        elevation_focus_mm=None,
        frequency_Hz=None,
    ):
        super().__init__()
        start_time = TIME()

        self.type = "linear"
        self.name = "LinearArrayTransducer"

        if kerf_mm < 0:
            raise ValueError("Kerf must be non-negative.")
        if no_sub_x <= 0 or no_sub_y <= 0:
            raise ValueError("Number of subdivisions must be positive.")
        if not isinstance(no_sub_x, int) or not isinstance(no_sub_y, int):
            raise ValueError("Number of subdivisions must be positive integers.")
        if element_height_mm <= 0 or element_width_mm <= 0:
            raise ValueError("Element dimensions must be positive.")

        if elevation_focus_mm is None:
            elevation_focus_mm = 0

        if elevation_focus_mm < 0:
            raise ValueError("Elevation focus must be non-negative or None.")

        element_height = element_height_mm * 1e-3
        element_width = element_width_mm * 1e-3

        kerf = kerf_mm * 1e-3
        elevation_focus = elevation_focus_mm * 1e-3
        
        self.n_elements = n_elements
        self.elem_width = element_width
        self.elem_height = element_height
        self.kerf = kerf
        self.pitch = element_width + kerf
        self.elev_focus = elevation_focus
        self.no_sub_x = no_sub_x
        self.no_sub_y = no_sub_y

        if elevation_focus is not None and elevation_focus > 0 and no_sub_y < 2:
            raise ValueError(
                "Elevation focus requires at least 2 subdivisions in y-dir."
            )

        if frequency_Hz is not None:
            self.frequency_Hz = frequency_Hz
        else:
            self.frequency_Hz = 1e6
            print("Warning: No central frequency provided. Defaulting to 1 MHz.")

        end_time = TIME()
        print(f"\nLinearArrayTransducer initialized in {end_time - start_time:.4f} seconds.")

    def _compute_element_centers(self) -> np.ndarray:
        """
        Compute element center positions for linear array.
        """
        total_width = self.n_elements * self.elem_width + (self.n_elements - 1) * self.kerf
        start_x = -total_width / 2 + self.elem_width / 2
        
        centers = np.array([
            [start_x + i * (self.elem_width + self.kerf), 0.0, 0.0]
            for i in range(self.n_elements)
        ])
        return centers

    def _build_subdivisions(
        self,
    ) -> Tuple[List[np.ndarray], float, List[int]]:
        """
        Build rectangular subdivision patches for all elements.
        """
        return geometry_utils.build_all_subdivisions(
            self.element_centers,
            self.elem_width,
            self.elem_height,
            self.no_sub_x,
            self.no_sub_y,
            self.elev_focus,
        )

    def compute_apodization(
        self,
        focus_mm,
        *,
        FoverD=None,
        apodization_type=None,
        plot=False,
        equiv_energy=False,
    ) -> np.ndarray:
        """
        Compute per-element apodization for focusing.

        Parameters
        ----------
        focus_mm : sequence of float
            Focus coordinates in mm [x, z] or [x, y, z].
        FoverD : float, optional
            F/D ratio for aperture sizing.
        apodization_type : str, optional
            Window type: 'none', 'rect', 'hanning', 'hamming'.
        plot : bool, optional
            Show apodization plot.
        equiv_energy : bool, optional
            For Hanning/Hamming, use more elements to maintain energy.

        Returns
        -------
        ndarray, shape (n_elements,)
            Normalized apodization weights.
        """
        defined_types = {None, "none", "rect", "hanning", "hamming"}
        if apodization_type not in defined_types:
            raise ValueError(
                f"Unknown apodization_type '{apodization_type}'. Must be one of {defined_types}"
            )

        focus = np.array(focus_mm) * 1e-3

        if focus.shape == (3,):
            x_foc, y_foc, z_foc = focus[0], focus[1], focus[2]
        elif focus.shape == (2,):
            x_foc, z_foc = focus[0], focus[1]
            y_foc = 0
        else:
            raise ValueError("Focus must be [x, z] or [x, y, z]")

        if z_foc <= 0:
            print("z_foc is negative. Diverging wave apodization will be computed.")

        N = self.n_elements
        pitch = self.elem_width + self.kerf
        total_ap = N * pitch

        if apodization_type is None:
            print("Warning: No apodization type provided. Using 'rect'.")
            apodization_type = "rect"

        if apodization_type == "none":
            apod = np.ones(N, dtype=float)
        else:
            if FoverD is not None:
                self.FoverD = FoverD

            if self.FoverD is None:
                print("Warning: F/D ratio not set. Using default value of 1.0.")
                self.FoverD = 1.0

            D = abs(z_foc) / self.FoverD
            if self.n_elements % 2 == 1:
                N_virt = int(round((D / total_ap) * N / 2) * 2 + 1)
            else:
                N_virt = int(round((D / total_ap) * N / 2) * 2)

            if equiv_energy:
                factor = {"rect": 1.0, "hanning": 0.5, "hamming": 0.54}[
                    apodization_type
                ]
            else:
                factor = 1

            N_ext = int(np.round(N_virt / factor))

            if N_ext > N:
                warnings.warn("Focus outside imaging window: using full aperture")
                N_ext = N

            if apodization_type == "rect":
                wins = np.ones(N_ext)
            elif apodization_type == "hanning":
                wins = np.hanning(N_ext)
            elif apodization_type == "hamming":
                wins = np.hamming(N_ext)
            else:
                raise ValueError(f"Unknown apodization_type '{apodization_type}'")

            shift_elems = int(np.round(x_foc / pitch)) - 1
            if shift_elems < -(N - 1) // 2:
                shift_elems = -(N - 1) // 2
            if shift_elems > (N - 1) // 2:
                shift_elems = (N - 1) // 2 + 1

            center = (N_ext - 1) // 2 - shift_elems
            idxs = np.arange(N_ext) - center + N // 2

            valid = (idxs >= 0) & (idxs < N)
            apod = np.zeros(N)
            apod[idxs[valid]] = wins[valid]

        if plot:
            self.plot_apodization()

        self.apodization = apod
        self.apodization_type = apodization_type
        return apod

    def compute_delays(
        self, focus_mm, *, c=None, inline=True, plot=False
    ) -> np.ndarray:
        """
        Compute per-element delays for focusing.

        Parameters
        ----------
        focus_mm : sequence of float
            Focus coordinates in mm [x, z] or [x, y, z].
        c : float, optional
            Speed of sound in m/s. Default 1540.
        inline : bool, optional
            Store result in self.delays.
        plot : bool, optional
            Show delays plot.

        Returns
        -------
        ndarray, shape (n_elements,)
            Delays in seconds.
        """
        if c is None:
            c = 1540.0
            print("Warning: No speed of sound provided. Defaulting to 1540 m/s.")

        focus = np.array(focus_mm) * 1e-3

        delays = np.linalg.norm(self.element_centers - focus, axis=1) / c

        if focus[2] <= 0:
            delays = delays - delays.min()
        else:
            delays = -delays + delays.max()

        if inline:
            self.delays = delays
        if plot:
            self.plot_delays()

        return delays
