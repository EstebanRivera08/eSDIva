"""
2-D matrix array transducer.

A matrix array is a rectangular grid of N_x × N_y elements.  All elements
are square-ish rectangular patches arranged in a 2-D plane.  Electronic
focusing and steering in both lateral directions are controlled by
``compute_delays`` / ``compute_apodization``.
"""

from time import time as TIME
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import windows

from . import validators
from .base import TransducerBase


class MatrixArrayTransducer(TransducerBase):
    """
    2-D matrix (multi-row) array transducer.

    Parameters
    ----------
    n_elements_x : int
        Number of elements in the x-direction (lateral).
    n_elements_y : int
        Number of elements in the y-direction (elevation).
    element_width_mm : float or array-like of length n_elements_x
        Element width(s) in x, in mm.  A scalar applies the same width to
        every column; an array allows per-column width variation.
    element_height_mm : float or array-like of length n_elements_y
        Element height(s) in y, in mm.  Scalar or per-row array.
    kerf_x_mm : float
        Inter-element gap in x, in mm (≥ 0).
    kerf_y_mm : float
        Inter-element gap in y, in mm (≥ 0).
    no_sub_x : int
        Subdivisions per element in x (≥ 1).
    no_sub_y : int
        Subdivisions per element in y (≥ 1).
    frequency_Hz : float, optional
        Centre frequency in Hz.
    dir_angle_deg : float, optional
        Half-angle directivity cone used when computing the active aperture
        for a given F/D.  Default is 30°.
    """

    def __init__(
        self,
        *,
        n_elements_x: int,
        n_elements_y: int,
        element_width_mm,
        element_height_mm,
        kerf_x_mm: float,
        kerf_y_mm: float,
        no_sub_x: int,
        no_sub_y: int,
        frequency_Hz: Optional[float] = None,
        dir_angle_deg: float = 30.0,
    ) -> None:
        super().__init__()
        t0 = TIME()

        self.type = "matrix"
        self.name = "MatrixArrayTransducer"

        # --- expand scalar/array element sizes ---
        widths_mm = np.atleast_1d(np.asarray(element_width_mm, dtype=float))
        heights_mm = np.atleast_1d(np.asarray(element_height_mm, dtype=float))
        if widths_mm.size == 1:
            widths_mm = np.full(n_elements_x, widths_mm[0])
        if heights_mm.size == 1:
            heights_mm = np.full(n_elements_y, heights_mm[0])
        if len(widths_mm) != n_elements_x:
            raise ValueError(
                f"element_width_mm has {len(widths_mm)} values but n_elements_x={n_elements_x}."
            )
        if len(heights_mm) != n_elements_y:
            raise ValueError(
                f"element_height_mm has {len(heights_mm)} values but n_elements_y={n_elements_y}."
            )

        validators.validate_kerf(kerf_x_mm, widths_mm.min(), name="kerf_x_mm")
        validators.validate_kerf(kerf_y_mm, heights_mm.min(), name="kerf_y_mm")
        no_sub_x, no_sub_y = validators.validate_subdivisions(no_sub_x, no_sub_y)

        self.n_elem_x = n_elements_x
        self.n_elem_y = n_elements_y
        self.n_elements = n_elements_x * n_elements_y

        # Per-element size arrays (metres)
        self._widths_m = widths_mm * 1e-3  # shape (n_elements_x,)
        self._heights_m = heights_mm * 1e-3  # shape (n_elements_y,)

        # Representative scalars for PyField compatibility
        self.elem_width = float(self._widths_m.mean())
        self.elem_height = float(self._heights_m.mean())

        self.kerf_x = kerf_x_mm * 1e-3
        self.kerf_y = kerf_y_mm * 1e-3
        self.pitch_x = self.elem_width + self.kerf_x  # representative pitch
        self.pitch_y = self.elem_height + self.kerf_y
        self.no_sub_x = no_sub_x
        self.no_sub_y = no_sub_y
        self.dir_angle_deg = dir_angle_deg

        self.fc = float(frequency_Hz) if frequency_Hz is not None else 1e6

        n_patches = self.n_elements * no_sub_x * no_sub_y
        print(
            f"MatrixArrayTransducer initialised in {TIME() - t0:.4f} s  "
            f"({self.n_elements} elements, {n_patches} patches)."
        )

    # ------------------------------------------------------------------
    # Abstract method implementations
    # ------------------------------------------------------------------

    def _compute_element_centers(self) -> np.ndarray:
        """Rectangular grid of element centres in the z=0 plane.

        Handles non-uniform element sizes by accumulating widths/heights.
        """
        # x positions: cumulative sum of widths + kerfs, centred at 0
        total_w = self._widths_m.sum() + (self.n_elem_x - 1) * self.kerf_x
        x_edges = np.concatenate([[0.0], np.cumsum(self._widths_m[:-1] + self.kerf_x)])
        x_centers = x_edges + self._widths_m / 2 - total_w / 2

        # y positions
        total_h = self._heights_m.sum() + (self.n_elem_y - 1) * self.kerf_y
        y_edges = np.concatenate([[0.0], np.cumsum(self._heights_m[:-1] + self.kerf_y)])
        y_centers = y_edges + self._heights_m / 2 - total_h / 2

        centers = []
        for y in y_centers:
            for x in x_centers:
                centers.append([x, y, 0.0])
        return np.array(centers)

    def _build_subdivisions(
        self,
    ) -> Tuple[List[np.ndarray], float, List[int]]:
        """Flat rectangular patches tiling every element.

        Each element uses its own width/height so variable-size arrays are
        supported.  The representative patch area returned is the mean area.
        """
        quads, el_indices = [], []
        total_area = 0.0

        for iy in range(self.n_elem_y):
            h = self._heights_m[iy]
            ys_local = np.linspace(-h / 2, h / 2, self.no_sub_y + 1)
            for ix in range(self.n_elem_x):
                w = self._widths_m[ix]
                xs_local = np.linspace(-w / 2, w / 2, self.no_sub_x + 1)
                patch_area = (w / self.no_sub_x) * (h / self.no_sub_y)
                total_area += patch_area * self.no_sub_x * self.no_sub_y

                idx = iy * self.n_elem_x + ix
                center = self.element_centers[idx]
                for i in range(self.no_sub_x):
                    for j in range(self.no_sub_y):
                        corners = (
                            np.array(
                                [
                                    [xs_local[i], ys_local[j], 0.0],
                                    [xs_local[i + 1], ys_local[j], 0.0],
                                    [xs_local[i + 1], ys_local[j + 1], 0.0],
                                    [xs_local[i], ys_local[j + 1], 0.0],
                                ]
                            )
                            + center
                        )
                        quads.append(corners)
                        el_indices.append(idx)

        mean_area = total_area / len(quads) if quads else 0.0
        return quads, mean_area, el_indices

    # ------------------------------------------------------------------
    # 2-D apodization — override with windowed aperture selection
    # ------------------------------------------------------------------

    def compute_apodization(
        self,
        focus_mm=None,
        *,
        FoverD: Optional[float] = None,
        apodization_type: Optional[str] = "circular",
        plot: bool = False,
        inline: bool = True,
    ) -> np.ndarray:
        """
        Compute per-element 2-D apodization for focusing at ``focus_mm``.

        The active aperture diameter is computed from the directivity angle and
        F/D ratio.  Supported window shapes are circular (elliptical mask),
        rectangular, Hanning, and Hamming.

        Parameters
        ----------
        focus_mm : array-like, shape (3,)
            Focal point ``[x, y, z]`` in mm.  Must be 3-D.
        FoverD : float, optional
            F-number.
        apodization_type : {'none', 'rect', 'circular', 'hanning', 'hamming'}
            Window shape.  Default is ``'circular'``.
        plot : bool
            Display the 2-D apodization map after computation.
        inline : bool
            Store result in ``self.apodization`` (default True).

        Returns
        -------
        ndarray
            Apodization weights, shape ``(n_elements,)``, flattened in
            row-major order (y-first).
        """
        if focus_mm is None:
            raise ValueError("focus_mm is required for multi-element transducers.")

        allowed = {"none", "rect", "circular", "hanning", "hamming"}
        if apodization_type not in allowed:
            raise ValueError(f"apodization_type must be one of {allowed}.")

        focus_m = np.array(focus_mm, dtype=float) * 1e-3
        if focus_m.shape != (3,):
            raise ValueError("focus_mm must be a 3-D coordinate [x, y, z].")

        x_foc, y_foc, z_foc = focus_m
        N_x, N_y = self.n_elem_x, self.n_elem_y

        if z_foc <= 0:
            print("z_foc <= 0: computing diverging-wave apodization.")

        if apodization_type == "none":
            apod_2d = np.ones((N_x, N_y))
        else:
            if FoverD is not None:
                self.FoverD = float(FoverD)
            if self.FoverD is None:
                print("F/D not set — defaulting to 1.0.")
                self.FoverD = 1.0

            # Active aperture diameter from directivity + F/D
            d_tx = 2 * abs(z_foc) * np.tan(np.radians(self.dir_angle_deg)) / self.FoverD
            Nvx = int(round(d_tx / self.pitch_x)) | 1  # force odd
            Nvy = int(round(d_tx / self.pitch_y)) | 1

            if apodization_type == "rect":
                profile = np.ones((Nvx, Nvy))
            elif apodization_type == "circular":
                Y, X = np.ogrid[:Nvx, :Nvy]
                cx, cy = (Nvx - 1) / 2, (Nvy - 1) / 2
                profile = (
                    (X - cx) ** 2 / (Nvx / 2) ** 2 + (Y - cy) ** 2 / (Nvy / 2) ** 2
                ) <= 1
                profile = profile.astype(float)
            elif apodization_type == "hanning":
                profile = np.outer(windows.hann(Nvx), windows.hann(Nvy))
            else:  # hamming
                profile = np.outer(windows.hamming(Nvx), windows.hamming(Nvy))

            sx = int(np.round(x_foc / self.elem_width))
            sy = int(np.round(y_foc / self.elem_height))
            ix = np.arange(Nvx) - (Nvx - 1) // 2 + N_x // 2 + sx
            iy = np.arange(Nvy) - (Nvy - 1) // 2 + N_y // 2 + sy
            valid_x = (ix >= 0) & (ix < N_x)
            valid_y = (iy >= 0) & (iy < N_y)

            apod_2d = np.zeros((N_x, N_y))
            apod_2d[np.ix_(ix[valid_x], iy[valid_y])] = profile[
                np.ix_(valid_x, valid_y)
            ]

        # Flatten in y-first (row-major) order to match element_centers ordering
        apod = apod_2d.T.flatten()

        if inline:
            self.apodization = apod
            self.apodization_type = apodization_type
        if plot:
            self.plot_apodization(apod)
        return apod

    # ------------------------------------------------------------------
    # 2-D plot overrides
    # ------------------------------------------------------------------

    def plot_apodization(
        self,
        apodization: Optional[np.ndarray] = None,
        *,
        figsize: Tuple = (6, 5),
        ax=None,
        **kwargs,
    ):
        """Display the 2-D apodization map as an image.

        Parameters
        ----------
        apodization : ndarray, optional
            Weights to plot. Defaults to ``self.apodization``.
        figsize : tuple of int
            Figure size in inches ``(width, height)``.
        ax : matplotlib.axes.Axes, optional
            Axes to draw on. If *None*, a new figure is created.
        **kwargs
            Forwarded to ``ax.imshow()``.

        Returns
        -------
        matplotlib.axes.Axes or None
            The axes object if ``ax`` was provided, otherwise *None*.
        """
        standalone = ax is None
        if apodization is None:
            apodization = self.apodization
        if standalone:
            _, ax = plt.subplots(figsize=figsize)

        assert ax is not None
        im = ax.imshow(
            apodization.reshape((self.n_elem_x, self.n_elem_y)),
            cmap="cool",
            vmin=0,
            vmax=1,
            **kwargs,
        )
        ax.set_title(f"Apodization: {self.apodization_type}")
        ax.set_xlabel("Element x #")
        ax.set_ylabel("Element y #")
        ax._im = im  # store for colorbar if needed

        if standalone:
            plt.tight_layout()
            plt.show()
            plt.close()
        else:
            return ax

    def plot_delays(
        self,
        delays: Optional[np.ndarray] = None,
        *,
        figsize: Tuple = (6, 5),
        ax=None,
        **kwargs,
    ):
        """Display the 2-D delay map as an image.

        Parameters
        ----------
        delays : ndarray, optional
            Delays to plot. Defaults to ``self.delays``.
        figsize : tuple of int
            Figure size in inches ``(width, height)``.
        ax : matplotlib.axes.Axes, optional
            Axes to draw on. If *None*, a new figure is created.
        **kwargs
            Forwarded to ``ax.imshow()``.

        Returns
        -------
        matplotlib.axes.Axes or None
            The axes object if ``ax`` was provided, otherwise *None*.
        """
        standalone = ax is None
        if delays is None:
            delays = self.delays
        if standalone:
            _, ax = plt.subplots(figsize=figsize)

        assert ax is not None
        im = ax.imshow(
            delays.reshape((self.n_elem_x, self.n_elem_y)) * 1e6,
            cmap="jet",
            **kwargs,
        )
        ax.set_title("Delays (µs)")
        ax.set_xlabel("Element x #")
        ax.set_ylabel("Element y #")
        ax._im = im  # store for colorbar if needed
        if standalone:
            plt.tight_layout()
            plt.show()
            plt.close()
        else:
            return ax

    def __repr__(self) -> str:
        return (
            f"MatrixArrayTransducer("
            f"n_elem=({self.n_elem_x},{self.n_elem_y}), "
            f"elem_width={self.elem_width * 1e3:.3f} mm, "
            f"elem_height={self.elem_height * 1e3:.3f} mm, "
            f"kerf=({self.kerf_x * 1e3:.3f},{self.kerf_y * 1e3:.3f}) mm, "
            f"no_sub=({self.no_sub_x},{self.no_sub_y}), "
            f"fc={self.fc / 1e6:.2f} MHz)"
        )
