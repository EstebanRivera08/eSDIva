"""Mono-element circular transducer types.

All four classes model single-element transducers (no electronic steering).
The aperture is always used at full amplitude --- ``compute_apodization`` is
inherited from ``TransducerBase`` and returns ``[1.0]``.  Use them directly
for a single-focus experiment, or assemble multiple instances into a
``CustomTransducer`` to build multi-element arrays such as a TUS helmet.

Notes
-----
Available classes:

- ``FlatCircularTransducer`` -- Flat piston, simplest unfocused circular source.
- ``ConcaveCircularTransducer`` -- Spherically curved (bowl-shaped) surface.
  Achieves geometric focus without electronic delays.
- ``ConvexCircularTransducer`` -- Spherically convex (dome-shaped) surface.
  The virtual focus is behind the transducer.
- ``FocusedCircularTransducer`` -- Circular-disk aperture curved in one axis
  only (cylindrical arc), producing a line focus.
"""

from time import time as TIME
from typing import Dict, List, Optional, Tuple

import numpy as np

from pyfield.utilities.surface_subdivision import subdivide_parametric_surface

from . import validators
from .base import TransducerBase

# ---------------------------------------------------------------------------
# Helper: circular mask for patch tiling
# ---------------------------------------------------------------------------


def _tile_disk(
    radius_m: float,
    no_sub: int = 25,
    border_refine: int = 3,
    filled_radius_with_big_patches: float = 0.95,
) -> Tuple[List[np.ndarray], float, List[int]]:
    """
    Tile a flat disk with adaptive rectangular patches.

    The bounding box ``[-radius, +radius]²`` is divided into a base grid of
    ``no_sub × no_sub`` coarse patches.  Each patch is classified by its four
    corner distances from the origin:

    * **Interior** — all four corners inside the disk: kept at coarse size
      ``dx = 2·radius / no_sub``.
    * **Exterior** — all four corners outside the disk: discarded.
    * **Boundary** — any corner straddles the edge: subdivided into
      ``border_refine × border_refine`` fine patches of size
      ``dx / border_refine``; only fine patches whose centre is inside the
      disk are kept.  This smooths the jagged circular edge.

    Parameters
    ----------
    radius_m : float
        Disc radius in metres.
    no_sub : int
        Number of coarse patches across the diameter (controls interior
        resolution).
    border_refine : int
        Subdivision factor applied to boundary patches.  ``border_refine=3``
        means each boundary patch is replaced by up to 16 smaller patches
        at 1/4 the coarse size.  Default 4.
    filled_radius_with_big_patches : float, optional
        The radius within which big patches are used without refinement.  Defined
        as a fraction of the disc radius (e.g. 0.9 means big patches are used up to 90%
        of the radius, and only the outer 10% is refined).  Default 1.0 (no inner
        cutoff).

    Returns
    -------
    quads : list of ndarray (4, 3)
        Patch corner vertices in metres (z = 0 for all).
    mean_area : float
        Mean patch area across all patches in m² (representative scalar).
    el_idx : list of int
        All zeros — every patch belongs to element 0.
    """
    dx = 2.0 * radius_m / no_sub
    R2 = radius_m**2
    half = no_sub // 2
    base_coords = (np.arange(no_sub) - half + 0.5) * dx
    filled_radius = filled_radius_with_big_patches

    quads: List[np.ndarray] = []
    total_area = 0.0

    def _add_patch(x0: float, y0: float, size: float) -> None:
        """Append one quad and accumulate its area."""
        x1, y1 = x0 + size, y0 + size
        quads.append(
            np.array(
                [
                    [x0, y0, 0.0],
                    [x1, y0, 0.0],
                    [x1, y1, 0.0],
                    [x0, y1, 0.0],
                ]
            )
        )
        nonlocal total_area
        total_area += size * size

    for cx in base_coords:
        x0c = cx - dx / 2
        for cy in base_coords:
            y0c = cy - dx / 2
            x1c, y1c = x0c + dx, y0c + dx

            # Classify by corners only — no center heuristic
            c_r2 = [
                x0c**2 + y0c**2,
                x1c**2 + y0c**2,
                x1c**2 + y1c**2,
                x0c**2 + y1c**2,
            ]
            all_inside = all(r2 <= filled_radius * R2 for r2 in c_r2)
            all_outside = all(r2 > filled_radius * R2 for r2 in c_r2)

            if all_inside:
                # Interior — keep at coarse resolution
                _add_patch(x0c, y0c, dx)
            elif all_outside:
                # Exterior — entirely outside the disk, skip
                continue
            else:
                # Boundary — straddles the edge, fill with fine patches
                sdx = dx / border_refine
                for i in range(border_refine):
                    for j in range(border_refine):
                        scx = x0c + (i + 0.5) * sdx
                        scy = y0c + (j + 0.5) * sdx
                        if scx**2 + scy**2 <= R2:
                            _add_patch(x0c + i * sdx, y0c + j * sdx, sdx)

    mean_area = total_area / len(quads) if quads else dx * dx
    el_idx = [0] * len(quads)
    return quads, mean_area, el_idx


# ---------------------------------------------------------------------------
# FlatCircularTransducer — flat piston
# ---------------------------------------------------------------------------


class FlatCircularTransducer(TransducerBase):
    """
    Flat circular piston transducer (mono-element).

    The aperture is approximated by a square grid of rectangular patches; only
    patches whose centre falls within the circle are included.  Increasing
    ``no_sub`` improves the circular approximation and the spatial accuracy of
    the SIR simulation.

    Parameters
    ----------
    diameter_mm : float
        Outer diameter of the active aperture in mm.
    no_sub : int
        Number of coarse patches across the diameter.  A value of 20–40 is
        typically sufficient for far-field calculations.
    border_refine : int, optional
        Subdivision factor for boundary patches.  Each patch that straddles
        the circular edge is replaced by ``border_refine²`` smaller patches,
        smoothing the jagged border.  Default 3.
    frequency_Hz : float, optional
        Centre frequency in Hz.  Defaults to 1 MHz.
    filled_radius_with_big_patches : float, optional
        Fraction of the radius to fill with coarse patches. Defaults to 0.99.
        The area to be filled is ``filled_radius_with_big_patches * diameter_mm / 2``.

    Notes
    -----
    Because this is a mono-element transducer, ``compute_delays`` returns
    ``[0.0]`` and ``compute_apodization`` returns ``[1.0]``.  All physical
    focusing is achieved through the excitation pulse shape and SIR
    convolution, not electronic delays.
    """

    def __init__(
        self,
        *,
        diameter_mm: float,
        no_sub: int = 25,
        border_refine: int = 3,
        frequency_Hz: Optional[float] = None,
        filled_radius_with_big_patches: float = 0.99,
    ) -> None:
        super().__init__()
        t0 = TIME()

        self.type = "circular"
        self.name = "FlatCircularTransducer"

        validators.validate_positive(diameter_mm, "diameter_mm", strict=True)
        validators.validate_integer(no_sub, "no_sub", min_val=4)
        validators.validate_integer(border_refine, "border_refine", min_val=1)
        validators.validate_positive(
            filled_radius_with_big_patches,
            "filled_radius_with_big_patches",
            strict=True,
        )

        self.diameter = diameter_mm * 1e-3
        self.radius = self.diameter / 2
        self.no_sub = no_sub
        self.border_refine = border_refine
        self.n_elements = 1
        self.filled_radius = filled_radius_with_big_patches

        self.elem_width = self.diameter
        self.elem_height = self.diameter
        self.no_sub_x = no_sub
        self.no_sub_y = no_sub

        self.fc = float(frequency_Hz) if frequency_Hz is not None else 1e6

        _ = self.sub_quad_verts
        print(
            f"FlatCircularTransducer initialised in {TIME() - t0:.4f} s  "
            f"(diameter={diameter_mm:.2f} mm, {self.n_sub_patches} patches, "
            f"coarse={self.diameter / no_sub * 1e3:.3f} mm, "
            f"border={self.diameter / no_sub / border_refine * 1e3:.3f} mm)."
        )

    def _compute_element_centers(self) -> np.ndarray:
        """Single element centred at the origin."""
        return np.array([[0.0, 0.0, 0.0]])

    def _build_subdivisions(
        self,
    ) -> Tuple[List[np.ndarray], float, List[int]]:
        return _tile_disk(
            self.radius, self.no_sub, self.border_refine, self.filled_radius
        )

    def __repr__(self) -> str:
        return (
            f"FlatCircularTransducer("
            f"diameter={self.diameter * 1e3:.2f} mm, "
            f"no_sub={self.no_sub}, "
            f"fc={self.fc / 1e6:.2f} MHz, "
            f"patches={self.n_sub_patches})"
        )


# ---------------------------------------------------------------------------
# ConcaveCircularTransducer — spherically curved bowl
# ---------------------------------------------------------------------------


class ConcaveCircularTransducer(TransducerBase):
    """
    Spherically focused single-element transducer (bowl / concave disc).

    The transducer surface is a spherical cap.  All points on the surface are
    equidistant (distance = ``radius_of_curvature``) from the geometric focus,
    so the acoustic wave converges at that point without any electronic delays.
    This geometry is common in HIFU therapy and transcranial ultrasound
    stimulation (TUS).

    The coordinate convention follows PyField: the transducer sits at z ≈ 0,
    and the focus is at z = ``radius_of_curvature`` along the beam axis.
    Patch vertices are lifted toward positive z according to the spherical cap
    equation:

        z(r) = R - √(R² - r²),    r = √(x² + y²) ≤ D/2

    so the centre patch is at z = 0 and the rim patches are at z > 0.

    Parameters
    ----------
    diameter_mm : float
        Outer diameter of the bowl aperture in mm.
    radius_of_curvature_mm : float
        Radius of the spherical cap in mm.  Must satisfy
        ``radius_of_curvature_mm ≥ diameter_mm / 2`` (the bowl cannot curve
        more than a hemisphere).
    no_sub : int
        Number of coarse patches across the diameter.
    border_refine : int, optional
        Subdivision factor for boundary patches.  Default 3.
    patch_fill : float, optional
        Fraction of the nominal patch size used in high-curvature mode.
        1.0 = patches touch; 0.75 gives a small safety gap that prevents
        physical overlap between tilted flat patches on the curved surface.
        Default 1.
    max_patch_scale : float, optional
        Rejection threshold: patches whose arc-length extent exceeds
        ``max_patch_scale × du_nominal`` are discarded, leaving intentional
        holes near the rim rather than overlapping oversized patches.
        Default 3.0.
    curvature_threshold : float, optional
        Ratio ``R / (D/2)`` below which the high-curvature grid strategy is
        activated.  Default 1.1.
    frequency_Hz : float, optional
        Centre frequency in Hz.  Defaults to 1 MHz.
    filled_radius_with_big_patches : float, optional
        Fraction of the radius to fill with coarse patches. Defaults to 0.95.
        The area to be filled is ``filled_radius_with_big_patches * diameter_mm / 2``.

    Raises
    ------
    ValueError
        If the radius of curvature is smaller than the aperture radius (the
        focal point would be inside the aperture --- physically impossible).
    """

    def __init__(
        self,
        *,
        diameter_mm: float,
        radius_of_curvature_mm: float,
        no_sub: int = 25,
        border_refine: int = 3,
        patch_fill: float = 1,
        max_patch_scale: float = 3.0,
        curvature_threshold: float = 1.1,
        frequency_Hz: Optional[float] = None,
        filled_radius_with_big_patches: float = 0.95,
    ) -> None:
        super().__init__()
        t0 = TIME()

        self.type = "focused_bowl"
        self.name = "ConcaveCircularTransducer"

        validators.validate_positive(diameter_mm, "diameter_mm", strict=True)
        validators.validate_positive(
            radius_of_curvature_mm, "radius_of_curvature_mm", strict=True
        )
        validators.validate_integer(no_sub, "no_sub", min_val=4)
        validators.validate_integer(border_refine, "border_refine", min_val=1)

        if radius_of_curvature_mm < diameter_mm / 2:
            raise ValueError(
                f"radius_of_curvature_mm ({radius_of_curvature_mm:.2f}) must be >= "
                f"diameter_mm/2 ({diameter_mm / 2:.2f}).  The focus cannot be inside "
                "the aperture."
            )

        self.diameter = diameter_mm * 1e-3
        self.radius = self.diameter / 2
        self.radius_of_curvature = radius_of_curvature_mm * 1e-3
        self.no_sub = no_sub
        self.border_refine = border_refine
        self.patch_fill = patch_fill
        self.max_patch_scale = max_patch_scale
        self.curvature_threshold = curvature_threshold
        self.n_elements = 1
        self.filled_radius = filled_radius_with_big_patches

        self.elem_width = self.diameter
        self.elem_height = self.diameter
        self.no_sub_x = no_sub
        self.no_sub_y = no_sub

        self.fc = float(frequency_Hz) if frequency_Hz is not None else 1e6

        self._sag = self.radius_of_curvature - np.sqrt(
            self.radius_of_curvature**2 - self.radius**2
        )

        _ = self.sub_quad_verts
        print(
            f"ConcaveCircularTransducer initialised in {TIME() - t0:.4f} s  "
            f"(diameter={diameter_mm:.2f} mm, "
            f"ROC={radius_of_curvature_mm:.2f} mm, "
            f"sag={self._sag * 1e3:.3f} mm, "
            f"{self.n_sub_patches} patches)."
        )

    def _compute_element_centers(self) -> np.ndarray:
        """Single element; centre is placed at the bowl's deepest point (origin)."""
        return np.array([[0.0, 0.0, 0.0]])

    def _build_subdivisions(
        self,
    ) -> Tuple[List[np.ndarray], float, List[int]]:
        """
        Subdivide the spherical cap into correctly-framed flat patches.

        Uses :func:`subdivide_parametric_surface` with the surface equation

            z(x, y) = R - √(R² - x² - y²)

        so that each patch has an **arc-length** extent (wu, wv) and a local
        frame whose tangent axes lie tangent to the sphere.  This replaces the
        old flat-warp approach (which produced non-rectangular, sheared patches).

        Frames are stored in ``_sub_patch_frames`` as a side-effect so that
        ``sub_patch_frames`` returns them without recomputation.
        """
        R = self.radius_of_curvature
        R_ap = self.radius

        # Warn when the rim half-angle θ_max > 30° (sin θ = R_ap/R).
        # At θ_max = 45° the arc-length amplification ||∂r/∂u|| = 1/cos(θ) = √2
        # and grows rapidly beyond that, causing many border patches to be rejected.
        rim_factor = R_ap / np.sqrt(max(R * R - R_ap * R_ap, 1e-30))
        if rim_factor > np.tan(np.radians(45)):
            print(
                f"  WARNING: ConcaveCircularTransducer — rim half-angle "
                f"{np.degrees(np.arctan(rim_factor)):.1f}° > 45°. "
                f"Arc-length amplification = {np.sqrt(1 + rim_factor**2):.2f}×. "
                f"Expect significant holes near the rim; increase no_sub for better coverage."
            )
        elif rim_factor > np.tan(np.radians(30)):
            print(
                f"  INFO: ConcaveCircularTransducer — rim half-angle "
                f"{np.degrees(np.arctan(rim_factor)):.1f}° (arc-length factor "
                f"{np.sqrt(1 + rim_factor**2):.2f}×). Some border patches may be rejected."
            )

        def surface_fn(x: float, y: float) -> np.ndarray:
            z = R - np.sqrt(max(R * R - x * x - y * y, 0.0))
            return np.array([x, y, z])

        frames = subdivide_parametric_surface(
            surface_fn,
            u_range=(-R_ap, R_ap),
            v_range=(-R_ap, R_ap),
            n_u=self.no_sub,
            n_v=self.no_sub,
            inside_fn=lambda x, y: x * x + y * y <= self.filled_radius**2 * R_ap * R_ap,
            border_refine=self.border_refine,
            normal_sign=1.0,  # ∂r/∂x × ∂r/∂y gives +z component -> toward medium
            patch_fill=self.patch_fill,
            max_patch_scale=self.max_patch_scale,
            curvature_threshold=self.curvature_threshold,
        )

        # Cache frames so sub_patch_frames finds them without recomputing
        self._sub_patch_frames = frames

        mean_area = float(np.mean(frames["wu"] * frames["wv"]))
        return frames["corners"], mean_area, frames["el_idx"]

    def _build_patch_frames(self) -> Dict:
        """Frames already built inside _build_subdivisions; just return them."""
        _ = self.sub_quad_verts  # ensures _build_subdivisions has run
        assert self._sub_patch_frames is not None
        return self._sub_patch_frames

    def __repr__(self) -> str:
        return (
            f"ConcaveCircularTransducer("
            f"diameter={self.diameter * 1e3:.2f} mm, "
            f"ROC={self.radius_of_curvature * 1e3:.2f} mm, "
            f"no_sub={self.no_sub}, "
            f"fc={self.fc / 1e6:.2f} MHz)"
        )


# ---------------------------------------------------------------------------
# ConvexCircularTransducer — spherically convex dome
# ---------------------------------------------------------------------------


class ConvexCircularTransducer(TransducerBase):
    """
    Spherically convex single-element transducer (dome / convex disc).

    The surface is a spherical dome that **bulges toward the propagation
    medium** (positive-z direction).  In contrast to
    :class:`ConcaveCircularTransducer`, which curves inward and converges to a
    geometric focus at depth ``R``, the convex surface diverges — its virtual
    focus is at ``z = -R`` (behind the transducer).

    This geometry models transducers that use an **acoustic refractive lens**
    to achieve focusing: the convex surface widens the natural directivity
    pattern and the lens refracts the wave to the desired focal depth.  It is
    also useful when a broad, diverging beam is desired (e.g. wide-field
    illumination or tissue characterisation).

    Surface z-profile (rim at z = 0, apex at z = sag):

        sag = R - √(R² - (D/2)²)
        z(r) = sag - (R - √(R² - r²))   r = √(x² + y²) ≤ D/2

    Parameters
    ----------
    diameter_mm : float
        Outer diameter of the dome aperture in mm.
    radius_of_curvature_mm : float
        Radius of the spherical surface in mm.  Must satisfy
        ``radius_of_curvature_mm ≥ diameter_mm / 2``.
    no_sub : int
        Number of coarse patches across the diameter.
    border_refine : int, optional
        Subdivision factor for boundary patches.  Default 3.
    patch_fill : float, optional
        Fraction of the nominal patch size used in high-curvature mode.
        Default 1.  See :class:`ConcaveCircularTransducer` for details.
    max_patch_scale : float, optional
        Rejection threshold for oversized patches at the rim.  Default 3.0.
        See :class:`ConcaveCircularTransducer` for details.
    curvature_threshold : float, optional
        Ratio ``R / (D/2)`` below which the high-curvature grid strategy is
        activated.  Default 1.1.
    frequency_Hz : float, optional
        Centre frequency in Hz.  Defaults to 1 MHz.
    filled_radius_with_big_patches : float, optional
        Fraction of the radius to fill with coarse patches. Defaults to 0.95.
    """

    def __init__(
        self,
        *,
        diameter_mm: float,
        radius_of_curvature_mm: float,
        no_sub: int = 25,
        border_refine: int = 3,
        patch_fill: float = 1,
        max_patch_scale: float = 3.0,
        curvature_threshold: float = 1.1,
        frequency_Hz: Optional[float] = None,
        filled_radius_with_big_patches: float = 0.95,
    ) -> None:
        super().__init__()
        t0 = TIME()

        self.type = "convex_bowl"
        self.name = "ConvexCircularTransducer"

        validators.validate_positive(diameter_mm, "diameter_mm", strict=True)
        validators.validate_positive(
            radius_of_curvature_mm, "radius_of_curvature_mm", strict=True
        )
        validators.validate_integer(no_sub, "no_sub", min_val=4)
        validators.validate_integer(border_refine, "border_refine", min_val=1)

        if radius_of_curvature_mm < diameter_mm / 2:
            raise ValueError(
                f"radius_of_curvature_mm ({radius_of_curvature_mm:.2f}) must be >= "
                f"diameter_mm/2 ({diameter_mm / 2:.2f}).  The dome cannot curve more "
                "than a hemisphere."
            )

        self.diameter = diameter_mm * 1e-3
        self.radius = self.diameter / 2
        self.radius_of_curvature = radius_of_curvature_mm * 1e-3
        self.no_sub = no_sub
        self.border_refine = border_refine
        self.patch_fill = patch_fill
        self.max_patch_scale = max_patch_scale
        self.curvature_threshold = curvature_threshold
        self.n_elements = 1
        self.filled_radius = filled_radius_with_big_patches

        self.elem_width = self.diameter
        self.elem_height = self.diameter
        self.no_sub_x = no_sub
        self.no_sub_y = no_sub

        self.fc = float(frequency_Hz) if frequency_Hz is not None else 1e6

        self._sag = self.radius_of_curvature - np.sqrt(
            self.radius_of_curvature**2 - self.radius**2
        )

        _ = self.sub_quad_verts
        print(
            f"ConvexCircularTransducer initialised in {TIME() - t0:.4f} s  "
            f"(diameter={diameter_mm:.2f} mm, "
            f"ROC={radius_of_curvature_mm:.2f} mm, "
            f"sag={self._sag * 1e3:.3f} mm, "
            f"{self.n_sub_patches} patches)."
        )

    def _compute_element_centers(self) -> np.ndarray:
        """Single element; centre placed at the aperture plane origin."""
        return np.array([[0.0, 0.0, 0.0]])

    def _build_subdivisions(
        self,
    ) -> Tuple[List[np.ndarray], float, List[int]]:
        """
        Subdivide the convex spherical dome into correctly-framed flat patches.

        Surface equation (apex at z = sag, rim at z = 0):

            z(x, y) = sag - (R - √(R² - x² - y²))

        Uses :func:`subdivide_parametric_surface` so patches have arc-length
        extents and tangent axes that lie tangent to the dome surface.
        Frames are stored in ``_sub_patch_frames`` as a side-effect.
        """
        R = self.radius_of_curvature
        R_ap = self.radius
        sag = self._sag

        rim_factor = R_ap / np.sqrt(max(R * R - R_ap * R_ap, 1e-30))
        if rim_factor > np.tan(np.radians(45)):
            print(
                f"  WARNING: ConvexCircularTransducer — rim half-angle "
                f"{np.degrees(np.arctan(rim_factor)):.1f}° > 45°. "
                f"Arc-length amplification = {np.sqrt(1 + rim_factor**2):.2f}×. "
                f"Expect significant holes near the rim; increase no_sub for better coverage."
            )
        elif rim_factor > np.tan(np.radians(30)):
            print(
                f"  INFO: ConvexCircularTransducer — rim half-angle "
                f"{np.degrees(np.arctan(rim_factor)):.1f}° (arc-length factor "
                f"{np.sqrt(1 + rim_factor**2):.2f}×). Some border patches may be rejected."
            )

        def surface_fn(x: float, y: float) -> np.ndarray:
            z = sag - (R - np.sqrt(max(R * R - x * x - y * y, 0.0)))
            return np.array([x, y, z])

        frames = subdivide_parametric_surface(
            surface_fn,
            u_range=(-R_ap, R_ap),
            v_range=(-R_ap, R_ap),
            n_u=self.no_sub,
            n_v=self.no_sub,
            inside_fn=lambda x, y: x * x + y * y <= self.filled_radius**2 * R_ap * R_ap,
            border_refine=self.border_refine,
            normal_sign=1.0,
            patch_fill=self.patch_fill,
            max_patch_scale=self.max_patch_scale,
            curvature_threshold=self.curvature_threshold,
        )

        self._sub_patch_frames = frames
        mean_area = float(np.mean(frames["wu"] * frames["wv"]))
        return frames["corners"], mean_area, frames["el_idx"]

    def _build_patch_frames(self) -> Dict:
        """Frames already built inside _build_subdivisions; just return them."""
        _ = self.sub_quad_verts
        assert self._sub_patch_frames is not None
        return self._sub_patch_frames

    def __repr__(self) -> str:
        return (
            f"ConvexCircularTransducer("
            f"diameter={self.diameter * 1e3:.2f} mm, "
            f"ROC={self.radius_of_curvature * 1e3:.2f} mm, "
            f"no_sub={self.no_sub}, "
            f"fc={self.fc / 1e6:.2f} MHz)"
        )


# ---------------------------------------------------------------------------
# FocusedCircularTransducer — curved in one axis only (line focus)
# ---------------------------------------------------------------------------


class FocusedCircularTransducer(TransducerBase):
    """
    Cylindrically focused single-element transducer (line focus).

    The aperture is a **circular disk** (not rectangular), curved along one
    axis only — either y (elevation, default) or x (lateral) — creating a
    cylindrical surface.  The resulting pressure field is focused along a
    line perpendicular to the curved axis (a geometric line focus at
    depth ``radius_of_curvature_mm``).

    Compared to :class:`ConcaveCircularTransducer` (which curves in *both* axes
    to produce a point focus), this class curves in only *one* axis, which
    produces a line focus or a tight focus in one plane only.

    Typical use cases:

    * 2-D cross-sectional imaging with a fixed elevation focus.
    * Line-focused therapeutic ultrasound along a tissue region.
    * Single-element stand-in for the elevation lens of a linear array.

    The curvature follows:

        z(val) = R - √(R² - val²)

    where ``val`` is the x- or y-coordinate of each patch corner (depending on
    ``focus_axis``).  The centre is at z = 0 and the outer edges are lifted
    toward z > 0.

    Parameters
    ----------
    diameter_mm : float
        Outer diameter of the circular aperture in mm.
    radius_of_curvature_mm : float
        Cylindrical radius of curvature in mm.  Must be ≥ ``diameter_mm / 2``.
        The geometric line focus is at this distance from the aperture centre.
    no_sub : int
        Number of coarse patches across the diameter.
    border_refine : int, optional
        Subdivision factor for boundary patches.  Default 3.
    focus_axis : {'y', 'x'}
        Which axis carries the curvature.  Default is ``'y'`` (elevation).
    patch_fill : float, optional
        Fraction of the nominal patch size used in high-curvature mode.
        Default 1.  See :class:`ConcaveCircularTransducer` for details.
    max_patch_scale : float, optional
        Rejection threshold for oversized patches at the rim.  Default 3.0.
        See :class:`ConcaveCircularTransducer` for details.
    curvature_threshold : float, optional
        Ratio ``R / (D/2)`` below which the high-curvature grid strategy is
        activated.  Default 1.1.
    frequency_Hz : float, optional
        Centre frequency in Hz.  Defaults to 1 MHz.
    filled_radius_with_big_patches : float, optional
        Fraction of the radius to fill with coarse patches. Defaults to 0.95.
        The area to be filled is ``filled_radius_with_big_patches * diameter_mm / 2``.
    """

    def __init__(
        self,
        *,
        diameter_mm: float,
        radius_of_curvature_mm: float,
        no_sub: int = 25,
        border_refine: int = 3,
        focus_axis: str = "y",
        patch_fill: float = 1,
        max_patch_scale: float = 3.0,
        curvature_threshold: float = 1.1,
        frequency_Hz: Optional[float] = None,
        filled_radius_with_big_patches: float = 0.95,
    ) -> None:
        super().__init__()
        t0 = TIME()

        self.type = "cylindrical"
        self.name = "FocusedCircularTransducer"

        validators.validate_positive(diameter_mm, "diameter_mm", strict=True)
        validators.validate_positive(
            radius_of_curvature_mm, "radius_of_curvature_mm", strict=True
        )
        validators.validate_integer(no_sub, "no_sub", min_val=4)
        validators.validate_integer(border_refine, "border_refine", min_val=1)

        if focus_axis not in ("x", "y"):
            raise ValueError("focus_axis must be 'x' or 'y'.")

        if radius_of_curvature_mm < diameter_mm / 2:
            raise ValueError(
                f"radius_of_curvature_mm ({radius_of_curvature_mm:.2f}) must be >= "
                f"diameter_mm/2 ({diameter_mm / 2:.2f} mm)."
            )

        self.diameter = diameter_mm * 1e-3
        self.radius = self.diameter / 2
        self.radius_of_curvature = radius_of_curvature_mm * 1e-3
        self.focus_axis = focus_axis
        self.no_sub = no_sub
        self.border_refine = border_refine
        self.patch_fill = patch_fill
        self.max_patch_scale = max_patch_scale
        self.curvature_threshold = curvature_threshold
        self.n_elements = 1
        self.filled_radius = filled_radius_with_big_patches

        self.elem_width = self.diameter
        self.elem_height = self.diameter
        self.no_sub_x = no_sub
        self.no_sub_y = no_sub

        self.fc = float(frequency_Hz) if frequency_Hz is not None else 1e6

        _ = self.sub_quad_verts
        print(
            f"FocusedCircularTransducer initialised in {TIME() - t0:.4f} s  "
            f"(diameter={diameter_mm:.2f} mm, "
            f"ROC={radius_of_curvature_mm:.2f} mm, axis={focus_axis}, "
            f"{self.n_sub_patches} patches)."
        )

    def _compute_element_centers(self) -> np.ndarray:
        """Single element centred at the origin."""
        return np.array([[0.0, 0.0, 0.0]])

    def _build_subdivisions(
        self,
    ) -> Tuple[List[np.ndarray], float, List[int]]:
        """
        Subdivide the cylindrical-cap aperture into correctly-framed flat patches.

        Surface equation (curved along ``focus_axis``):

            z(x, y) = R - √(R² - val²)   where val = y (or x) for focus_axis

        Uses :func:`subdivide_parametric_surface` so each patch has arc-length
        extents and tangents tangent to the cylindrical surface.
        Frames are stored in ``_sub_patch_frames`` as a side-effect.
        """
        R = self.radius_of_curvature
        R_ap = self.radius
        axis = self.focus_axis

        # For cylindrical curvature: amplification only along the curved axis
        rim_factor = R_ap / np.sqrt(max(R * R - R_ap * R_ap, 1e-30))
        if rim_factor > np.tan(np.radians(45)):
            print(
                f"  WARNING: FocusedCircularTransducer — rim half-angle along {axis}-axis "
                f"{np.degrees(np.arctan(rim_factor)):.1f}° > 45°. "
                f"Arc-length amplification = {np.sqrt(1 + rim_factor**2):.2f}×. "
                f"Expect holes near the rim; increase no_sub for better coverage."
            )
        elif rim_factor > np.tan(np.radians(30)):
            print(
                f"  INFO: FocusedCircularTransducer — rim half-angle along {axis}-axis "
                f"{np.degrees(np.arctan(rim_factor)):.1f}° (arc-length factor "
                f"{np.sqrt(1 + rim_factor**2):.2f}×). Some border patches may be rejected."
            )

        def surface_fn(x: float, y: float) -> np.ndarray:
            val = y if axis == "y" else x
            z = R - np.sqrt(max(R * R - val * val, 0.0))
            return np.array([x, y, z])

        frames = subdivide_parametric_surface(
            surface_fn,
            u_range=(-R_ap, R_ap),
            v_range=(-R_ap, R_ap),
            n_u=self.no_sub,
            n_v=self.no_sub,
            inside_fn=lambda x, y: x * x + y * y <= self.filled_radius**2 * R_ap * R_ap,
            border_refine=self.border_refine,
            normal_sign=1.0,
            patch_fill=self.patch_fill,
            max_patch_scale=self.max_patch_scale,
            curvature_threshold=self.curvature_threshold,
        )

        self._sub_patch_frames = frames
        mean_area = float(np.mean(frames["wu"] * frames["wv"]))
        return frames["corners"], mean_area, frames["el_idx"]

    def _build_patch_frames(self) -> Dict:
        """Frames already built inside _build_subdivisions; just return them."""
        _ = self.sub_quad_verts
        assert self._sub_patch_frames is not None
        return self._sub_patch_frames

    def __repr__(self) -> str:
        return (
            f"FocusedCircularTransducer("
            f"diameter={self.diameter * 1e3:.2f} mm, "
            f"ROC={self.radius_of_curvature * 1e3:.2f} mm, "
            f"axis={self.focus_axis}, "
            f"no_sub={self.no_sub}, "
            f"fc={self.fc / 1e6:.2f} MHz)"
        )
