"""
Linear and convex array transducers.

LinearArrayTransducer
    N rectangular elements in a flat row along x.  Optional elevation
    (y-axis) lens via cylindrical curvature.

ConvexArrayTransducer
    N rectangular elements arranged on a convex cylindrical arc in the
    XZ plane — the standard geometry for abdominal / obstetric probes.
    The centre of curvature is behind the probe face (at z = -R), so the
    outer elements are angled outward producing a widening field of view.
"""

import warnings
from time import time as TIME
from typing import List, Optional, Tuple

import numpy as np

from . import validators
from .base import TransducerBase


class LinearArrayTransducer(TransducerBase):
    """
    1-D linear array transducer.

    Elements are laid out along x.  Electronic beam steering and focusing are
    controlled via ``compute_delays`` / ``compute_apodization``.  Elevation
    focusing (y-direction) is achieved by curving the element surface into a
    cylindrical arc.

    Parameters
    ----------
    n_elements : int
        Number of active elements.
    element_width_mm : float
        Element dimension along the steering axis (x), in mm.
    element_height_mm : float
        Element dimension in the elevation axis (y), in mm.
    kerf_mm : float
        Gap between adjacent elements in mm (≥ 0).
    no_sub_x : int
        Subdivisions per element in x (lateral, ≥ 1).
    no_sub_y : int
        Subdivisions per element in y (elevation, ≥ 1).
        Must be ≥ 2 when ``elevation_focus_mm`` is set.
    elevation_focus_mm : float, optional
        Radius of curvature for the cylindrical lens in mm.
        ``None`` (default) means a flat aperture.
    frequency_Hz : float, optional
        Centre frequency in Hz.  Defaults to 1 MHz with a warning.
    """

    def __init__(
        self,
        *,
        n_elements: int,
        element_width_mm: float,
        element_height_mm: float,
        kerf_mm: float,
        no_sub_x: int,
        no_sub_y: int,
        elevation_focus_mm: Optional[float] = None,
        frequency_Hz: Optional[float] = None,
    ) -> None:
        super().__init__()
        t0 = TIME()

        self.type = "linear"
        self.name = "LinearArrayTransducer"

        # --- validate inputs ---
        validators.validate_kerf(kerf_mm, element_width_mm)
        no_sub_x, no_sub_y = validators.validate_subdivisions(no_sub_x, no_sub_y)
        validators.validate_positive(element_width_mm, "element_width_mm", strict=True)
        validators.validate_positive(
            element_height_mm, "element_height_mm", strict=True
        )

        if elevation_focus_mm is not None:
            validators.validate_positive(elevation_focus_mm, "elevation_focus_mm")
            if elevation_focus_mm > 0 and no_sub_y < 2:
                raise ValueError(
                    "elevation_focus_mm requires no_sub_y ≥ 2 to model the curved surface."
                )

        # --- store parameters in SI units ---
        self.n_elements = n_elements
        self.elem_width = element_width_mm * 1e-3
        self.elem_height = element_height_mm * 1e-3
        self.kerf = kerf_mm * 1e-3
        self.pitch = self.elem_width + self.kerf
        self.elev_focus = (elevation_focus_mm * 1e-3) if elevation_focus_mm else 0.0
        self.no_sub_x = no_sub_x
        self.no_sub_y = no_sub_y

        if frequency_Hz is not None:
            self.fc = float(frequency_Hz)
        else:
            self.fc = 1e6
            print("Warning: No frequency provided. Defaulting to 1 MHz.")

        print(
            f"LinearArrayTransducer initialised in {TIME() - t0:.4f} s  "
            f"({n_elements} elements, {n_elements * no_sub_x * no_sub_y} patches)."
        )

    # ------------------------------------------------------------------
    # Abstract method implementations
    # ------------------------------------------------------------------

    def _compute_element_centers(self) -> np.ndarray:
        """Evenly spaced element centres along x at z=0."""
        total_w = self.n_elements * self.elem_width + (self.n_elements - 1) * self.kerf
        start_x = -total_w / 2 + self.elem_width / 2
        return np.array(
            [[start_x + i * self.pitch, 0.0, 0.0] for i in range(self.n_elements)]
        )

    def _build_subdivisions(
        self,
    ) -> Tuple[List[np.ndarray], float, List[int]]:
        """
        Build rectangular patches for every element.

        Each element is subdivided into ``no_sub_x × no_sub_y`` patches.
        When ``elev_focus > 0`` the y-edges of each patch are lifted onto a
        cylindrical arc so that all patches lie on the curved lens surface.
        """
        xs = np.linspace(-self.elem_width / 2, self.elem_width / 2, self.no_sub_x + 1)
        ys = np.linspace(-self.elem_height / 2, self.elem_height / 2, self.no_sub_y + 1)
        patch_area = (self.elem_width / self.no_sub_x) * (
            self.elem_height / self.no_sub_y
        )

        quads, el_indices = [], []
        for idx, center in enumerate(self.element_centers):
            for i in range(self.no_sub_x):
                for j in range(self.no_sub_y):
                    corners = np.array(
                        [
                            [xs[i], ys[j], 0.0],
                            [xs[i + 1], ys[j], 0.0],
                            [xs[i + 1], ys[j + 1], 0.0],
                            [xs[i], ys[j + 1], 0.0],
                        ]
                    )
                    corners[:, 0] += center[0]
                    corners[:, 1] += center[1]

                    if self.elev_focus > 0:
                        # Cylindrical curvature: z offset along y
                        y_vals = corners[:, 1]
                        corners[:, 2] += self.elev_focus - np.sqrt(
                            np.clip(self.elev_focus**2 - y_vals**2, 0, None)
                        )
                    else:
                        corners[:, 2] += center[2]

                    quads.append(corners)
                    el_indices.append(idx)

        return quads, patch_area, el_indices

    # ------------------------------------------------------------------
    # Apodization — override with windowed aperture selection
    # ------------------------------------------------------------------

    def compute_apodization(
        self,
        focus_mm,
        *,
        FoverD: Optional[float] = None,
        apodization_type: Optional[str] = None,
        plot: bool = False,
        equiv_energy: bool = False,
        inline: bool = True,
    ) -> np.ndarray:
        """
        Compute per-element apodization for focusing at ``focus_mm``.

        The active sub-aperture is sized by the F/D ratio: only elements
        within ``D = |z_focus| / FoverD`` of the focus lateral position are
        assigned non-zero weights.

        Parameters
        ----------
        focus_mm : array-like, shape (2,) or (3,)
            Focus in mm. 2-D ``[x, z]`` is accepted (y=0 assumed).
        FoverD : float, optional
            F-number.  Ignored when ``apodization_type='none'``.
        apodization_type : {'none', 'rect', 'hanning', 'hamming'}, optional
            Window shape. ``None`` defaults to ``'rect'`` with a warning.
        plot : bool
            Display the result after computation.
        equiv_energy : bool
            Scale Hanning/Hamming windows to maintain the same total energy
            as a rectangular window of the same F/D.
        inline : bool
            Store result in ``self.apodization`` (default True).

        Returns
        -------
        apod : ndarray, shape (n_elements,)
        """
        allowed = {None, "none", "rect", "hanning", "hamming"}
        if apodization_type not in allowed:
            raise ValueError(
                f"apodization_type must be one of {allowed}, got '{apodization_type}'."
            )

        focus_m = validators.validate_focus_coordinates(focus_mm)
        x_foc, z_foc = focus_m[0], focus_m[2]

        if z_foc <= 0:
            print("z_foc ≤ 0: computing diverging-wave apodization.")

        N = self.n_elements

        if apodization_type is None:
            print("No apodization_type given — defaulting to 'rect'.")
            apodization_type = "rect"

        if apodization_type == "none":
            apod = np.ones(N, dtype=float)
        else:
            if FoverD is not None:
                self.FoverD = float(FoverD)
            if self.FoverD is None:
                print("F/D not set — defaulting to 1.0.")
                self.FoverD = 1.0

            D = abs(z_foc) / self.FoverD
            # Number of elements spanning aperture D (must match parity of N)
            N_virt = int(round((D / (N * self.pitch)) * N / 2) * 2 + (N % 2))
            N_virt = max(1, N_virt)

            factor = 1.0
            if equiv_energy:
                factor = {"rect": 1.0, "hanning": 0.5, "hamming": 0.54}[
                    apodization_type
                ]

            N_ext = int(np.round(N_virt / factor))
            if N_ext > N:
                warnings.warn("Focus outside imaging window: using full aperture.")
                N_ext = N

            if apodization_type == "rect":
                wins = np.ones(N_ext)
            elif apodization_type == "hanning":
                wins = np.hanning(N_ext)
            else:
                wins = np.hamming(N_ext)

            # Slide window so its centre aligns with x_foc
            shift_elems = int(np.round(x_foc / self.pitch)) - 1
            shift_elems = np.clip(shift_elems, -(N - 1) // 2, (N - 1) // 2 + 1)

            center_idx = (N_ext - 1) // 2 - shift_elems
            idxs = np.arange(N_ext) - center_idx + N // 2
            valid = (idxs >= 0) & (idxs < N)
            apod = np.zeros(N)
            apod[idxs[valid]] = wins[valid]

        if inline:
            self.apodization = apod
            self.apodization_type = apodization_type

        if plot:
            self.plot_apodization(apod)

        return apod

    # ------------------------------------------------------------------
    # 2-D plot override for delays (keeps 1-D line chart, consistent with base)
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"LinearArrayTransducer("
            f"n_elements={self.n_elements}, "
            f"elem_width={self.elem_width * 1e3:.3f} mm, "
            f"elem_height={self.elem_height * 1e3:.3f} mm, "
            f"kerf={self.kerf * 1e3:.3f} mm, "
            f"elev_focus={self.elev_focus * 1e3:.1f} mm, "
            f"no_sub=({self.no_sub_x},{self.no_sub_y}), "
            f"fc={self.fc / 1e6:.2f} MHz)"
        )


# ============================================================================
# Convex (curvilinear) linear array
# ============================================================================


class ConvexArrayTransducer(TransducerBase):
    """
    Convex (curvilinear) linear array transducer.

    Elements are arranged on a convex cylindrical arc in the XZ plane — the
    standard geometry for abdominal, cardiac, and obstetric probes.  The
    centre of curvature sits *behind* the probe face at ``z = -R``, so outer
    elements are tilted outward and the field of view widens with depth.

    The centre element is positioned at the origin with its normal pointing
    in ``+z`` (depth direction).  Electronic beam steering and focusing are
    controlled by ``compute_delays`` / ``compute_apodization``.

    Parameters
    ----------
    n_elements : int
        Number of active elements.
    element_width_mm : float
        Arc-length dimension of each element (azimuth), in mm.
    element_height_mm : float
        Element dimension in the elevation axis (y), in mm.
    kerf_mm : float
        Arc-length gap between adjacent elements, in mm (≥ 0).
    radius_of_curvature_mm : float
        Radius of the convex arc in mm.  Larger values give a flatter probe.
        Typical clinical values: 40 – 80 mm.
    no_sub_x : int
        Patch subdivisions per element along the arc (azimuth, ≥ 1).
    no_sub_y : int
        Patch subdivisions per element in elevation (y, ≥ 1).
        Must be ≥ 2 when ``elevation_focus_mm`` is set.
    elevation_focus_mm : float, optional
        Cylindrical elevation-lens focus depth in mm.  When provided, each
        element surface is curved in the y-direction so that
        ``z(y) = R_elev − √(R_elev² − y²)``, producing a geometric line
        focus at ``elevation_focus_mm`` depth in elevation.  Equivalent to
        the acoustic lens of a focused convex probe (FIELD II
        ``xdc_focused_convex``).  Must be ≥ ``element_height_mm / 2``.
    frequency_Hz : float, optional
        Centre frequency in Hz.  Defaults to 1 MHz with a warning.
    """

    def __init__(
        self,
        *,
        n_elements: int,
        element_width_mm: float,
        element_height_mm: float,
        kerf_mm: float,
        radius_of_curvature_mm: float,
        no_sub_x: int,
        no_sub_y: int,
        elevation_focus_mm: Optional[float] = None,
        frequency_Hz: Optional[float] = None,
    ) -> None:
        super().__init__()
        t0 = TIME()

        self.type = "convex"
        self.name = "ConvexArrayTransducer"

        validators.validate_kerf(kerf_mm, element_width_mm)
        validators.validate_positive(element_width_mm, "element_width_mm", strict=True)
        validators.validate_positive(
            element_height_mm, "element_height_mm", strict=True
        )
        validators.validate_positive(
            radius_of_curvature_mm, "radius_of_curvature_mm", strict=True
        )
        no_sub_x, no_sub_y = validators.validate_subdivisions(no_sub_x, no_sub_y)

        if elevation_focus_mm is not None:
            validators.validate_positive(
                elevation_focus_mm, "elevation_focus_mm", strict=True
            )
            if elevation_focus_mm < element_height_mm / 2:
                raise ValueError(
                    f"elevation_focus_mm ({elevation_focus_mm:.2f}) must be ≥ "
                    f"element_height_mm/2 ({element_height_mm / 2:.2f} mm)."
                )
            if no_sub_y < 2:
                raise ValueError("no_sub_y must be ≥ 2 when elevation_focus_mm is set.")

        self.n_elements = n_elements
        self.elem_width = element_width_mm * 1e-3
        self.elem_height = element_height_mm * 1e-3
        self.kerf = kerf_mm * 1e-3
        self.pitch = self.elem_width + self.kerf
        self.R = radius_of_curvature_mm * 1e-3  # radius of curvature (metres)
        self.no_sub_x = no_sub_x
        self.no_sub_y = no_sub_y
        self._elev_R = (
            float(elevation_focus_mm) * 1e-3 if elevation_focus_mm is not None else None
        )

        self.fc = float(frequency_Hz) if frequency_Hz is not None else 1e6
        if frequency_Hz is None:
            print("Warning: No frequency provided. Defaulting to 1 MHz.")

        # Pre-compute per-element angles
        # Arc angle between adjacent element centres
        d_theta = self.pitch / self.R
        self._thetas = (np.arange(n_elements) - (n_elements - 1) / 2) * d_theta

        elev_str = (
            f", elev_focus={elevation_focus_mm:.1f} mm"
            if elevation_focus_mm is not None
            else ""
        )
        print(
            f"ConvexArrayTransducer initialised in {TIME() - t0:.4f} s  "
            f"({n_elements} elements, R={radius_of_curvature_mm:.1f} mm, "
            f"arc span={np.degrees(self._thetas[-1] - self._thetas[0]):.1f}°"
            f"{elev_str})."
        )

    # ------------------------------------------------------------------
    # Abstract method implementations
    # ------------------------------------------------------------------

    def _compute_element_centers(self) -> np.ndarray:
        """
        Element centres on the convex arc.

        The arc is defined with the centre of curvature at (0, 0, -R).
        Element i at angle θ_i is at (R·sin θ_i, 0, R·(cos θ_i − 1)).
        """
        x = self.R * np.sin(self._thetas)
        y = np.zeros(self.n_elements)
        z = self.R * (np.cos(self._thetas) - 1.0)
        return np.column_stack([x, y, z])

    def _build_subdivisions(self) -> Tuple[List[np.ndarray], float, List[int]]:
        """
        Build rectangular patches for every element, each rotated to sit
        tangent to the arc.

        The local patch grid is flat and centred at the origin; it is rotated
        around the y-axis by θ_i to match the element tilt, then translated
        to the element centre position.
        """
        # Local (un-rotated) patch grid: flat rectangle at origin
        xs = np.linspace(-self.elem_width / 2, self.elem_width / 2, self.no_sub_x + 1)
        ys = np.linspace(-self.elem_height / 2, self.elem_height / 2, self.no_sub_y + 1)
        patch_area = (self.elem_width / self.no_sub_x) * (
            self.elem_height / self.no_sub_y
        )

        quads, el_indices = [], []

        for idx, (theta, center) in enumerate(zip(self._thetas, self.element_centers)):
            # Rotation matrix around y-axis by theta (aligns local +z with element normal)
            c, s = np.cos(theta), np.sin(theta)
            Ry = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])

            for i in range(self.no_sub_x):
                for j in range(self.no_sub_y):
                    corners = np.array(
                        [
                            [xs[i], ys[j], 0.0],
                            [xs[i + 1], ys[j], 0.0],
                            [xs[i + 1], ys[j + 1], 0.0],
                            [xs[i], ys[j + 1], 0.0],
                        ]
                    )
                    # Apply elevation curvature in local frame (before arc rotation)
                    if self._elev_R is not None:
                        y_vals = corners[:, 1]
                        corners[:, 2] += self._elev_R - np.sqrt(
                            np.clip(self._elev_R**2 - y_vals**2, 0, None)
                        )
                    # Rotate to element tilt, then translate to arc position
                    corners = corners @ Ry.T + center
                    quads.append(corners)
                    el_indices.append(idx)

        return quads, patch_area, el_indices

    # ------------------------------------------------------------------
    # Apodization — same windowed aperture logic as LinearArrayTransducer
    # ------------------------------------------------------------------

    def compute_apodization(
        self,
        focus_mm,
        *,
        FoverD: Optional[float] = None,
        apodization_type: Optional[str] = None,
        plot: bool = False,
        inline: bool = True,
    ) -> np.ndarray:
        """
        Compute per-element apodization.  Delegates to
        :class:`LinearArrayTransducer` logic (F/D aperture + window).
        """
        # Reuse linear apodization — the arc curvature only matters for delays
        focus_m = validators.validate_focus_coordinates(focus_mm)
        x_foc, z_foc = focus_m[0], focus_m[2]

        if apodization_type is None:
            apodization_type = "rect"

        N = self.n_elements
        if apodization_type == "none":
            apod = np.ones(N, dtype=float)
        else:
            if FoverD is not None:
                self.FoverD = float(FoverD)
            if self.FoverD is None:
                self.FoverD = 1.0
            D = abs(z_foc) / self.FoverD
            N_virt = int(round((D / (N * self.pitch)) * N / 2) * 2 + (N % 2))
            N_virt = max(1, N_virt)

            if apodization_type == "rect":
                wins = np.ones(N_virt)
            elif apodization_type == "hanning":
                wins = np.hanning(N_virt)
            else:
                wins = np.hamming(N_virt)

            if N_virt > N:
                N_virt = N
                wins = wins[:N]

            shift_elems = int(np.round(x_foc / self.pitch))
            center_idx = (N_virt - 1) // 2 - shift_elems
            idxs = np.arange(N_virt) - center_idx + N // 2
            valid = (idxs >= 0) & (idxs < N)
            apod = np.zeros(N)
            apod[idxs[valid]] = wins[valid]

        if inline:
            self.apodization = apod
            self.apodization_type = apodization_type
        if plot:
            self.plot_apodization(apod)
        return apod

    def __repr__(self) -> str:
        elev_str = (
            f", elev_focus={self._elev_R * 1e3:.1f} mm"
            if self._elev_R is not None
            else ""
        )
        return (
            f"ConvexArrayTransducer("
            f"n_elements={self.n_elements}, "
            f"elem_width={self.elem_width * 1e3:.3f} mm, "
            f"R={self.R * 1e3:.1f} mm, "
            f"arc={np.degrees(self._thetas[-1] - self._thetas[0]):.1f}°"
            f"{elev_str}, "
            f"no_sub=({self.no_sub_x},{self.no_sub_y}), "
            f"fc={self.fc / 1e6:.2f} MHz)"
        )
