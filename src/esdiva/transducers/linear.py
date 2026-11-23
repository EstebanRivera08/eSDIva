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

from time import time as TIME
from typing import Callable, List, Optional, Tuple

import numpy as np

from . import geometry_utils, validators
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
        Cylindrical elevation lens, given as its RADIUS OF CURVATURE in mm.
        ``None`` or 0 (default) means a flat aperture. POSITIVE curves the
        surface away from the medium (concave, converging — an ordinary
        elevation lens); NEGATIVE makes it bulge into the medium (convex,
        diverging), which spreads the elevation beam instead of focusing it.
        ``|radius|`` must be at least ``element_height_mm / 2``, and `no_sub_y`
        must be ≥ 2 so the curved surface is actually resolved by the patches.

        The line focus does NOT land at ``elevation_focus_mm``: the surface is
        referenced with its RIM at ``z = 0`` (the Field II ``xdc_focused_array``
        datum), so its centre of curvature — the depth where every patch is
        equidistant, i.e. the true focus — sits one sagitta shallower, at
        `elevation_focus_depth_mm`. The two differ by ``h²/(8R)``: 0.17 mm for a
        4 mm aperture at R = 12 mm, 0.03 mm for a 1.5 mm aperture at R = 8 mm.
    elevation_profile_mm : callable, optional
        Any elevation lens shape, as the element surface ``y_mm -> z_mm`` (``+z``
        toward the medium), e.g. ``lambda y: y**2 / 24`` for a parabolic concave
        lens. The rims are placed at ``z = 0`` and the face is tiled in equal
        arc-length strips. Exclusive with `elevation_focus_mm`, which is the
        circular special case. Needs ``no_sub_y >= 2``.
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
        elevation_profile_mm: Optional[Callable[[np.ndarray], np.ndarray]] = None,
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

        self._elev_sag = geometry_utils.elevation_sag_from_inputs(
            elevation_focus_mm, elevation_profile_mm, element_height_mm, no_sub_y
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

    def _default_elevation_lens_sag(self) -> float:
        """SIGNED centre offset (m) of the cylindrical elevation lens; 0 if flat.

        The sagitta ``|R| − √(R² − (height/2)²)``, carrying the sign of the
        curvature: POSITIVE when the surface dishes BACK at the element centre
        (concave/converging, the centre sits at ``z = −sag`` behind the rim), and
        NEGATIVE when it bulges FORWARD into the medium (convex/diverging).

        The sign is what makes the reception time origin correct for both. A
        recessed centre lengthens the path to a target by one sagitta relative to
        the ``z = 0`` plane a beamformer measures depth from; a protruding centre
        shortens it. Reception therefore subtracts this signed value (per
        aperture) from the beamforming reference, which lands the echo at its
        true depth in either case and is a no-op for a flat aperture.
        """
        return self._elevation_tile()[2]

    def _elevation_tile(self) -> Tuple[List[np.ndarray], float, float]:
        """One element's lens-shaped patches (own frame), patch area, signed sag."""
        return geometry_utils.elevation_profile_quads(
            self.elem_width,
            self.elem_height,
            self.no_sub_x,
            self.no_sub_y,
            self._elev_sag,
        )

    @property
    def elevation_focus_depth_mm(self) -> float:
        """Depth (mm) of the elevation line focus — the arc's centre of curvature.

        Every patch is exactly ``|R|`` from this point, which is what focusing
        means, so this is where the elevation beam is narrowest (width ~
        ``λ·z/height``). Because the lens is referenced with its rim at ``z = 0``,
        it is one sagitta SHALLOWER than ``|elevation_focus_mm|``.

        For a convex (diverging) lens the focus is virtual: the value is negative,
        marking the point BEHIND the array the elevation beam appears to spread
        from.

        Returns
        -------
        float
            Elevation focal depth in mm; ``inf`` for a flat aperture (no
            elevation focus at all), negative for a convex lens's virtual focus,
            ``nan`` for a free `elevation_profile_mm` (no single focus defined).
        """
        if self._elev_sag is not None and not self.elev_focus:
            return float("nan")
        if not self.elev_focus:
            return float("inf")
        return float((abs(self.elev_focus) - abs(self.elevation_lens_sag)) * 1e3) * (
            1.0 if self.elev_focus > 0 else -1.0
        )

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
        """Tile every element with ``no_sub_x × no_sub_y`` patches on its lens surface.

        All elements share one shape, so a single element is tiled (flat, circular
        arc, or free elevation profile — rims at ``z = 0``) and copied to each
        element centre along x.
        """
        local, patch_area, _ = self._elevation_tile()
        quads = [q + center for center in self.element_centers for q in local]
        el_indices = [i for i in range(self.n_elements) for _ in local]
        return quads, patch_area, el_indices

    # ------------------------------------------------------------------
    # Apodization — override with windowed aperture selection
    # ------------------------------------------------------------------

    def compute_apodization(
        self,
        focus_mm=None,
        *,
        FoverD: Optional[float] = None,
        apodization_type: Optional[str] = None,
        plot: bool = False,
        inplace: bool = True,
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
        inplace : bool
            Store result in ``self.apodization`` (default True).

        Returns
        -------
        ndarray
            Apodization weights, shape ``(n_elements,)``.
        """
        if focus_mm is None:
            raise ValueError("focus_mm is required for multi-element transducers.")

        allowed = {None, "none", "rect", "hanning", "hamming"}
        if apodization_type not in allowed:
            raise ValueError(
                f"apodization_type must be one of {allowed}, got '{apodization_type}'."
            )

        focus_m = validators.validate_focus_coordinates(focus_mm)
        x_foc, z_foc = focus_m[0], focus_m[2]

        if z_foc <= 0:
            print("z_foc <= 0: computing diverging-wave apodization.")

        if apodization_type is None:
            print("No apodization_type given — defaulting to 'rect'.")
            apodization_type = "rect"

        if apodization_type != "none":
            if FoverD is not None:
                self.FoverD = float(FoverD)
            if self.FoverD is None:
                print("F/D not set — defaulting to 1.0.")
                self.FoverD = 1.0

        apod = geometry_utils.windowed_apodization_1d(
            self.n_elements,
            self.pitch,
            x_foc,
            z_foc,
            self.FoverD if self.FoverD is not None else 1.0,
            apodization_type,
        )

        if inplace:
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
        Cylindrical elevation lens, given as its RADIUS OF CURVATURE in mm
        (positive = concave/converging, negative = convex/diverging). Each
        element surface is curved in y onto that arc, with the RIM held at
        ``z = 0``. The resulting line focus is the arc's centre of curvature,
        one sagitta shallower than ``|elevation_focus_mm|`` — read it from
        `elevation_focus_depth_mm`. ``|value|`` must be ≥
        ``element_height_mm / 2``.
    elevation_profile_mm : callable, optional
        Free elevation lens shape ``y_mm -> z_mm`` (``+z`` toward the medium),
        rims at ``z = 0``. Exclusive with `elevation_focus_mm`.
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
        elevation_profile_mm: Optional[Callable[[np.ndarray], np.ndarray]] = None,
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

        self._elev_sag = geometry_utils.elevation_sag_from_inputs(
            elevation_focus_mm, elevation_profile_mm, element_height_mm, no_sub_y
        )

        self.n_elements = n_elements
        self.elem_width = element_width_mm * 1e-3
        self.elem_height = element_height_mm * 1e-3
        self.kerf = kerf_mm * 1e-3
        self.pitch = self.elem_width + self.kerf
        self.R = radius_of_curvature_mm * 1e-3  # radius of curvature (metres)
        self.no_sub_x = no_sub_x
        self.no_sub_y = no_sub_y
        self._elev_R = float(elevation_focus_mm) * 1e-3 if elevation_focus_mm else None

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

    def _default_elevation_lens_sag(self) -> float:
        """SIGNED centre offset (m) of the elevation lens: + recessed, − protruding, 0 flat.

        Reception subtracts it from the beamforming time origin so echoes land at
        their true depth (same convention as `LinearArrayTransducer`).
        """
        return geometry_utils.elevation_profile_quads(
            self.elem_width, self.elem_height, 1, self.no_sub_y, self._elev_sag
        )[2]

    def _compute_element_centers(self) -> np.ndarray:
        """
        Element centres on the convex arc.

        The arc is defined with the centre of curvature at (0, 0, -R).
        Element i at angle θ_i is at (R·sin θ_i, 0, R·(cos θ_i - 1)).
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
        # One element tiled in its own frame (flat or lens-shaped, rims at z = 0).
        local, patch_area, _ = geometry_utils.elevation_profile_quads(
            self.elem_width,
            self.elem_height,
            self.no_sub_x,
            self.no_sub_y,
            self._elev_sag,
        )
        quads, el_indices = [], []
        for idx, (theta, center) in enumerate(zip(self._thetas, self.element_centers)):
            # Rotate about y by θ (local +z onto the element normal), then move onto the arc.
            c, s = np.cos(theta), np.sin(theta)
            Ry = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
            quads.extend(q @ Ry.T + center for q in local)
            el_indices.extend([idx] * len(local))

        return quads, patch_area, el_indices

    # ------------------------------------------------------------------
    # Apodization — same windowed aperture logic as LinearArrayTransducer
    # ------------------------------------------------------------------

    def compute_apodization(
        self,
        focus_mm=None,
        *,
        FoverD: Optional[float] = None,
        apodization_type: Optional[str] = None,
        plot: bool = False,
        inplace: bool = True,
    ) -> np.ndarray:
        """Compute per-element apodization for the convex array.

        Parameters
        ----------
        focus_mm : array-like, shape (2,) or (3,)
            Focus in mm. 2-D ``[x, z]`` is accepted (y=0 assumed).
        FoverD : float, optional
            F-number. Ignored when ``apodization_type='none'``.
        apodization_type : {'none', 'rect', 'hanning', 'hamming'}, optional
            Window shape. ``None`` defaults to ``'rect'``.
        plot : bool
            Display the result after computation.
        inplace : bool
            Store result in ``self.apodization`` (default True).

        Returns
        -------
        ndarray
            Apodization weights, shape ``(n_elements,)``.
        """
        if focus_mm is None:
            raise ValueError("focus_mm is required for multi-element transducers.")

        # The arc curvature only affects delays; lateral apodization is the same
        # windowed sub-aperture as the flat linear array.
        focus_m = validators.validate_focus_coordinates(focus_mm)
        x_foc, z_foc = focus_m[0], focus_m[2]

        if apodization_type is None:
            apodization_type = "rect"

        if apodization_type != "none":
            if FoverD is not None:
                self.FoverD = float(FoverD)
            if self.FoverD is None:
                self.FoverD = 1.0

        apod = geometry_utils.windowed_apodization_1d(
            self.n_elements,
            self.pitch,
            x_foc,
            z_foc,
            self.FoverD if self.FoverD is not None else 1.0,
            apodization_type,
        )

        if inplace:
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
