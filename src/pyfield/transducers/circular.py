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

from pyfield.utilities.surface_subdivision import (
    subdivide_parametric_surface,
    subdivide_spherical_cap,
)

from . import validators
from .base import TransducerBase

# ---------------------------------------------------------------------------
# Helper: circular mask for patch tiling
# ---------------------------------------------------------------------------


def _tile_disk(
    radius_m: float,
    no_sub_diameter: int = 25,
    refine_factor: int = 3,
    ratio_big_patches: float = 0.85,
) -> Tuple[List[np.ndarray], float, List[int]]:
    """
    Tile a flat disk with adaptive rectangular patches.

    The bounding box ``[-radius, +radius]²`` is divided into a base grid of
    ``no_sub_diameter × no_sub_diameter`` coarse patches.  Each patch is
    classified by its four corner distances from the origin:

    * **Interior** — all four corners inside the coarse region: kept at coarse
      size ``dx = 2·radius / no_sub_diameter``.
    * **Exterior** — all four corners outside the coarse region: discarded or
      refined.
    * **Boundary** — any corner straddles the edge: subdivided into
      ``refine_factor × refine_factor`` fine patches; only fine patches whose
      centre is inside ``1.005 × radius`` are kept.

    Parameters
    ----------
    radius_m : float
        Disc radius in metres.
    no_sub_diameter : int
        Number of coarse patches across the diameter.
    refine_factor : int
        Subdivision factor applied to boundary patches.  Default ``3``.
    ratio_big_patches : float
        Fraction of the radius filled with coarse patches (0–1).  The outer
        ``1 - ratio_big_patches`` fraction is refined.  Default ``0.85``.

    Returns
    -------
    quads : list of ndarray (4, 3)
        Patch corner vertices in metres (z = 0 for all).
    mean_area : float
        Mean patch area across all patches in m² (representative scalar).
    el_idx : list of int
        All zeros — every patch belongs to element 0.
    """
    dx = 2.0 * radius_m / no_sub_diameter
    R2 = radius_m**2
    half = no_sub_diameter // 2
    base_coords = (np.arange(no_sub_diameter) - half + 0.5) * dx
    filled_radius = ratio_big_patches

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
                sdx = dx / refine_factor
                for i in range(refine_factor):
                    for j in range(refine_factor):
                        scx = x0c + (i + 0.5) * sdx
                        scy = y0c + (j + 0.5) * sdx
                        if scx**2 + scy**2 <= (1.005 * radius_m) ** 2:
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
    ``no_sub_diameter`` improves the circular approximation and the spatial
    accuracy of the SIR simulation.

    Parameters
    ----------
    diameter_mm : float
        Outer diameter of the active aperture in mm.
    no_sub_diameter : int
        Number of coarse patches across the diameter.  A value of 20–40 is
        typically sufficient for far-field calculations.
    ratio_big_patches : float
        Fraction of the radius filled with coarse patches (0–1).  The outer
        ``1 - ratio_big_patches`` fraction is refined.  Default ``0.85``.
    refine_factor : int
        Subdivision factor for boundary patches.  Each boundary patch is
        replaced by ``refine_factor²`` smaller patches.  Default ``3``.
    frequency_Hz : float, optional
        Centre frequency in Hz.  Defaults to 1 MHz.

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
        no_sub_diameter: int = 25,
        ratio_big_patches: float = 0.85,
        refine_factor: int = 3,
        frequency_Hz: Optional[float] = None,
    ) -> None:
        super().__init__()
        t0 = TIME()

        self.type = "circular"
        self.name = "FlatCircularTransducer"

        validators.validate_positive(diameter_mm, "diameter_mm", strict=True)
        validators.validate_integer(no_sub_diameter, "no_sub_diameter", min_val=4)
        validators.validate_integer(refine_factor, "refine_factor", min_val=1)

        self.diameter = diameter_mm * 1e-3
        self.radius = self.diameter / 2
        self.no_sub_diameter = no_sub_diameter
        self.ratio_big_patches = ratio_big_patches
        self.refine_factor = refine_factor
        self.n_elements = 1

        self.elem_width = self.diameter
        self.elem_height = self.diameter
        self.no_sub_x = no_sub_diameter
        self.no_sub_y = no_sub_diameter

        self.fc = float(frequency_Hz) if frequency_Hz is not None else 1e6

        _ = self.sub_quad_verts
        print(
            f"FlatCircularTransducer initialised in {TIME() - t0:.4f} s  "
            f"(diameter={diameter_mm:.2f} mm, {self.n_sub_patches} patches, "
            f"coarse={self.diameter / no_sub_diameter * 1e3:.3f} mm, "
            f"border={self.diameter / no_sub_diameter / refine_factor * 1e3:.3f} mm)."
        )

    def _compute_element_centers(self) -> np.ndarray:
        """Single element centred at the origin."""
        return np.array([[0.0, 0.0, 0.0]])

    def _build_subdivisions(
        self,
    ) -> Tuple[List[np.ndarray], float, List[int]]:
        return _tile_disk(
            self.radius, self.no_sub_diameter, self.refine_factor, self.ratio_big_patches
        )

    def __repr__(self) -> str:
        return (
            f"FlatCircularTransducer("
            f"diameter={self.diameter * 1e3:.2f} mm, "
            f"no_sub_diameter={self.no_sub_diameter}, "
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
    equidistant from the geometric focus, so the acoustic wave converges at
    that point without any electronic delays.  Common in HIFU therapy and TUS.

    ``focus_mm`` is the **axial depth** from the rim plane (z = 0) to the
    geometric focus.  The radius of curvature is derived as
    ``R = sqrt(focus_mm² + (D/2)²)``.  ``focus_mm = 0`` gives a hemisphere
    (R = D/2).

    Parameters
    ----------
    diameter_mm : float
        Outer diameter of the bowl aperture in mm.
    focus_mm : float
        Axial distance from the rim to the geometric focus in mm.
        Must be ``>= 0``.  ``0`` = hemisphere.
    no_sub_diameter : int
        Target number of patches across the diameter.
    method : {'spherical', 'cartesian'}
        ``'spherical'`` (default) uses ring-based spherical-coordinate tiling
        via :func:`subdivide_spherical_cap` — works at any curvature including
        hemispheres.  ``'cartesian'`` uses the Cartesian parameter-space grid
        via :func:`subdivide_parametric_surface`.
    ratio_big_patches : float
        Fraction of the surface covered by coarse patches (0–1).  The
        remaining region is refined.  For spherical method, controls inner
        ring refinement; for cartesian, controls border refinement.
        Default ``0.85``.
    refine_factor : int
        Subdivision factor in the refined region.  Default ``3``.
    frequency_Hz : float, optional
        Centre frequency in Hz.  Defaults to 1 MHz.
    """

    def __init__(
        self,
        *,
        diameter_mm: float,
        focus_mm: float,
        no_sub_diameter: int = 25,
        method: str = "spherical",
        ratio_big_patches: float = 0.85,
        refine_factor: int = 3,
        normalize_patch_size: bool = False,
        frequency_Hz: Optional[float] = None,
    ) -> None:
        super().__init__()
        t0 = TIME()

        self.type = "focused_bowl"
        self.name = "ConcaveCircularTransducer"

        validators.validate_positive(diameter_mm, "diameter_mm", strict=True)
        validators.validate_integer(no_sub_diameter, "no_sub_diameter", min_val=4)

        if focus_mm < 0:
            raise ValueError(
                f"focus_mm ({focus_mm:.2f}) must be >= 0.  "
                "Use focus_mm=0 for a hemisphere."
            )
        if method not in ("spherical", "cartesian"):
            raise ValueError("method must be 'spherical' or 'cartesian'.")

        self.diameter = diameter_mm * 1e-3
        self.radius = self.diameter / 2
        self.no_sub_diameter = no_sub_diameter
        self.method = method
        self.ratio_big_patches = ratio_big_patches
        self.refine_factor = refine_factor
        self.normalize_patch_size = normalize_patch_size
        self.n_elements = 1

        # focus_mm = z_depth → R = sqrt(f² + r_ap²)
        r_ap = self.radius
        f = focus_mm * 1e-3
        R = np.sqrt(f**2 + r_ap**2)
        self.radius_of_curvature = R
        self._sag = R - f  # since sqrt(R² - r_ap²) = f

        # Spherical half-angle from pole to rim
        self._theta_max = np.arcsin(r_ap / R)

        # Ring count from target patch size
        target_size = self.diameter / no_sub_diameter
        dtheta = target_size / R
        self._n_rings = max(3, round(self._theta_max / dtheta))

        self.elem_width = self.diameter
        self.elem_height = self.diameter
        self.no_sub_x = no_sub_diameter
        self.no_sub_y = no_sub_diameter

        self.fc = float(frequency_Hz) if frequency_Hz is not None else 1e6

        _ = self.sub_quad_verts
        print(
            f"ConcaveCircularTransducer initialised in {TIME() - t0:.4f} s  "
            f"(diameter={diameter_mm:.2f} mm, "
            f"focus_mm={focus_mm:.2f} mm (z-depth), "
            f"R={R * 1e3:.2f} mm, "
            f"sag={self._sag * 1e3:.3f} mm, "
            f"method={method}, "
            f"{self.n_sub_patches} patches)."
        )

    def _compute_element_centers(self) -> np.ndarray:
        """Single element; centre is placed at the bowl's deepest point (origin)."""
        return np.array([[0.0, 0.0, 0.0]])

    def _build_subdivisions(
        self,
    ) -> Tuple[List[np.ndarray], float, List[int]]:
        """Subdivide the spherical cap using the chosen method."""
        if self.method == "spherical":
            frames = subdivide_spherical_cap(
                self.radius_of_curvature,
                self._theta_max,
                self._n_rings,
                concave=True,
                normal_sign=1.0,
                ratio_big_patches=self.ratio_big_patches,
                refine_factor=self.refine_factor,
            )
        else:
            R = self.radius_of_curvature
            R_ap = self.radius
            rbp = self.ratio_big_patches
            rf = self.refine_factor

            # Arc-length reparameterization: uniform grid in (sx, sy)
            # where x = R*sin(sx/R). This yields uniform patch sizes
            # even at high curvature (e.g. hemisphere).
            s_max = R * np.arcsin(R_ap / R)

            R_inner = rbp * R_ap
            R_accept = 1.005 * R_ap

            D_flat = np.sqrt(max(R * R - R_ap * R_ap, 0.0))  # = R*cos(theta_max)

            def surface_fn(sx: float, sy: float) -> np.ndarray:
                x = R * np.sin(sx / R)
                y = R * np.sin(sy / R)
                z = D_flat - np.sqrt(max(R * R - x * x - y * y, 0.0))  # rim at z=0
                return np.array([x, y, z])

            def inside_fn(sx: float, sy: float) -> bool:
                x = R * np.sin(sx / R)
                y = R * np.sin(sy / R)
                return x * x + y * y <= R_inner * R_inner

            def accept_fn(sx: float, sy: float) -> bool:
                x = R * np.sin(sx / R)
                y = R * np.sin(sy / R)
                return x * x + y * y <= R_accept * R_accept

            frames = subdivide_parametric_surface(
                surface_fn,
                u_range=(-s_max, s_max),
                v_range=(-s_max, s_max),
                n_u=self.no_sub_diameter,
                n_v=self.no_sub_diameter,
                inside_fn=inside_fn,
                accept_fn=accept_fn,
                border_refine=rf,
                normal_sign=1.0,
                normalize_patch_size=self.normalize_patch_size,
            )

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
            f"R={self.radius_of_curvature * 1e3:.2f} mm, "
            f"no_sub_diameter={self.no_sub_diameter}, "
            f"fc={self.fc / 1e6:.2f} MHz)"
        )


# ---------------------------------------------------------------------------
# ConvexCircularTransducer — spherically convex dome
# ---------------------------------------------------------------------------


class ConvexCircularTransducer(TransducerBase):
    """
    Spherically convex single-element transducer (dome / convex disc).

    The surface is a spherical dome that **bulges toward the propagation
    medium** (positive-z direction).  The convex surface diverges — its
    virtual focus is at ``z = -R`` (behind the transducer).

    ``focus_mm`` is the **axial depth** from the rim plane to the virtual
    focus (same convention as :class:`ConcaveCircularTransducer`).
    ``R = sqrt(focus_mm² + (D/2)²)``.  ``focus_mm = 0`` gives a hemisphere.

    Surface z-profile (rim at z = 0, apex at z = sag):

        sag = R - √(R² - (D/2)²) = R - focus_mm
        z(r) = sag - (R - √(R² - r²))   r = √(x² + y²) ≤ D/2

    Parameters
    ----------
    diameter_mm : float
        Outer diameter of the dome aperture in mm.
    focus_mm : float
        Axial distance from the rim to the virtual focus in mm.
        Must be ``>= 0``.  ``0`` = hemisphere.
    no_sub_diameter : int
        Target number of patches across the diameter.
    method : {'spherical', 'cartesian'}
        ``'spherical'`` (default) or ``'cartesian'``.
    ratio_big_patches : float
        Fraction of surface with coarse patches.  Default ``0.85``.
    refine_factor : int
        Subdivision factor in the refined region.  Default ``3``.
    frequency_Hz : float, optional
        Centre frequency in Hz.  Defaults to 1 MHz.
    """

    def __init__(
        self,
        *,
        diameter_mm: float,
        focus_mm: float,
        no_sub_diameter: int = 25,
        method: str = "spherical",
        ratio_big_patches: float = 0.85,
        refine_factor: int = 3,
        normalize_patch_size: bool = False,
        frequency_Hz: Optional[float] = None,
    ) -> None:
        super().__init__()
        t0 = TIME()

        self.type = "convex_bowl"
        self.name = "ConvexCircularTransducer"

        validators.validate_positive(diameter_mm, "diameter_mm", strict=True)
        validators.validate_integer(no_sub_diameter, "no_sub_diameter", min_val=4)

        if focus_mm < 0:
            raise ValueError(
                f"focus_mm ({focus_mm:.2f}) must be >= 0.  "
                "Use focus_mm=0 for a hemisphere."
            )
        if method not in ("spherical", "cartesian"):
            raise ValueError("method must be 'spherical' or 'cartesian'.")

        self.diameter = diameter_mm * 1e-3
        self.radius = self.diameter / 2
        self.no_sub_diameter = no_sub_diameter
        self.method = method
        self.ratio_big_patches = ratio_big_patches
        self.refine_factor = refine_factor
        self.normalize_patch_size = normalize_patch_size
        self.n_elements = 1

        # focus_mm = z_depth → R = sqrt(f² + r_ap²)
        r_ap = self.radius
        f = focus_mm * 1e-3
        R = np.sqrt(f**2 + r_ap**2)
        self.radius_of_curvature = R
        self._sag = R - f

        self._theta_max = np.arcsin(r_ap / R)
        target_size = self.diameter / no_sub_diameter
        dtheta = target_size / R
        self._n_rings = max(3, round(self._theta_max / dtheta))

        self.elem_width = self.diameter
        self.elem_height = self.diameter
        self.no_sub_x = no_sub_diameter
        self.no_sub_y = no_sub_diameter

        self.fc = float(frequency_Hz) if frequency_Hz is not None else 1e6

        _ = self.sub_quad_verts
        print(
            f"ConvexCircularTransducer initialised in {TIME() - t0:.4f} s  "
            f"(diameter={diameter_mm:.2f} mm, "
            f"focus_mm={focus_mm:.2f} mm (z-depth), "
            f"R={R * 1e3:.2f} mm, "
            f"sag={self._sag * 1e3:.3f} mm, "
            f"method={method}, "
            f"{self.n_sub_patches} patches)."
        )

    def _compute_element_centers(self) -> np.ndarray:
        """Single element; centre placed at the aperture plane origin."""
        return np.array([[0.0, 0.0, 0.0]])

    def _build_subdivisions(
        self,
    ) -> Tuple[List[np.ndarray], float, List[int]]:
        """Subdivide the convex dome using the chosen method."""
        if self.method == "spherical":
            frames = subdivide_spherical_cap(
                self.radius_of_curvature,
                self._theta_max,
                self._n_rings,
                concave=False,
                normal_sign=1.0,
                ratio_big_patches=self.ratio_big_patches,
                refine_factor=self.refine_factor,
            )
        else:
            R = self.radius_of_curvature
            R_ap = self.radius
            sag = self._sag
            rbp = self.ratio_big_patches
            rf = self.refine_factor

            # Arc-length reparameterization: uniform grid in (sx, sy) where
            # x = R*sin(sx/R). Keeps Jacobian near 1 on-axis and avoids the
            # singularity (Jacobian → ∞) that the direct (x,y) grid produces
            # near the rim of high-curvature / hemisphere surfaces.
            s_max = R * np.arcsin(R_ap / R)
            R_inner = rbp * R_ap
            R_accept = 1.005 * R_ap

            def surface_fn(sx: float, sy: float) -> np.ndarray:
                x = R * np.sin(sx / R)
                y = R * np.sin(sy / R)
                z = sag - (R - np.sqrt(max(R * R - x * x - y * y, 0.0)))
                return np.array([x, y, z])

            def inside_fn(sx: float, sy: float) -> bool:
                x = R * np.sin(sx / R)
                y = R * np.sin(sy / R)
                return x * x + y * y <= R_inner * R_inner

            def accept_fn(sx: float, sy: float) -> bool:
                x = R * np.sin(sx / R)
                y = R * np.sin(sy / R)
                return x * x + y * y <= R_accept * R_accept

            frames = subdivide_parametric_surface(
                surface_fn,
                u_range=(-s_max, s_max),
                v_range=(-s_max, s_max),
                n_u=self.no_sub_diameter,
                n_v=self.no_sub_diameter,
                inside_fn=inside_fn,
                accept_fn=accept_fn,
                border_refine=rf,
                normal_sign=1.0,
                normalize_patch_size=self.normalize_patch_size,
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
            f"R={self.radius_of_curvature * 1e3:.2f} mm, "
            f"no_sub_diameter={self.no_sub_diameter}, "
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
    line perpendicular to the curved axis.

    ``focus_mm`` is the **axial depth** from the rim to the line focus
    (same convention as :class:`ConcaveCircularTransducer`).
    ``R = sqrt(focus_mm² + (D/2)²)``.  Must be ``>= 0``.

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
    focus_mm : float
        Axial distance from the rim to the line focus in mm.  Must be ``>= 0``.
    no_sub_diameter : int
        Number of coarse patches across the diameter.
    ratio_big_patches : float
        Fraction of the radius filled with coarse patches.  Default ``0.85``.
    refine_factor : int
        Subdivision factor for boundary patches.  Default ``3``.
    focus_axis : {'y', 'x'}
        Which axis carries the curvature.  Default is ``'y'`` (elevation).
    frequency_Hz : float, optional
        Centre frequency in Hz.  Defaults to 1 MHz.
    """

    def __init__(
        self,
        *,
        diameter_mm: float,
        focus_mm: float,
        no_sub_diameter: int = 25,
        ratio_big_patches: float = 0.85,
        refine_factor: int = 3,
        focus_axis: str = "y",
        frequency_Hz: Optional[float] = None,
    ) -> None:
        super().__init__()
        t0 = TIME()

        self.type = "cylindrical"
        self.name = "FocusedCircularTransducer"

        validators.validate_positive(diameter_mm, "diameter_mm", strict=True)
        validators.validate_integer(no_sub_diameter, "no_sub_diameter", min_val=4)
        validators.validate_integer(refine_factor, "refine_factor", min_val=1)

        if focus_axis not in ("x", "y"):
            raise ValueError("focus_axis must be 'x' or 'y'.")

        if focus_mm < 0:
            raise ValueError(
                f"focus_mm ({focus_mm:.2f}) must be >= 0.  "
                "Use focus_mm=0 for a semicircular cylinder."
            )

        self.diameter = diameter_mm * 1e-3
        self.radius = self.diameter / 2
        self.focus_axis = focus_axis
        self.no_sub_diameter = no_sub_diameter
        self.ratio_big_patches = ratio_big_patches
        self.refine_factor = refine_factor
        self.n_elements = 1

        # focus_mm = z_depth → R = sqrt(f² + r_ap²)
        r_ap = self.radius
        f = focus_mm * 1e-3
        R = np.sqrt(f**2 + r_ap**2)
        self.radius_of_curvature = R

        self.elem_width = self.diameter
        self.elem_height = self.diameter
        self.no_sub_x = no_sub_diameter
        self.no_sub_y = no_sub_diameter

        self.fc = float(frequency_Hz) if frequency_Hz is not None else 1e6

        _ = self.sub_quad_verts
        print(
            f"FocusedCircularTransducer initialised in {TIME() - t0:.4f} s  "
            f"(diameter={diameter_mm:.2f} mm, "
            f"focus_mm={focus_mm:.2f} mm (z-depth), "
            f"R={R * 1e3:.2f} mm, axis={focus_axis}, "
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

            z(x, y) = √(R² - R_ap²) - √(R² - val²)   where val = y (or x)

        Rim at z = 0 (at val = ±R_ap), centre at z = -sag.

        Uses :func:`subdivide_parametric_surface` so each patch has arc-length
        extents and tangents tangent to the cylindrical surface.
        Frames are stored in ``_sub_patch_frames`` as a side-effect.
        """
        R = self.radius_of_curvature
        R_ap = self.radius
        axis = self.focus_axis
        rbp = self.ratio_big_patches
        rf = self.refine_factor

        # D_flat: z-coordinate of the sphere at val=±R_ap (the curved-axis rim).
        # Setting z = D_flat - sqrt(R²-val²) places the curved-axis rim at z=0
        # and the center at z = D_flat - R = -sag, consistent with focus_mm
        # measured from the rim plane.
        D_flat = np.sqrt(max(R * R - R_ap * R_ap, 0.0))

        def surface_fn(x: float, y: float) -> np.ndarray:
            val = y if axis == "y" else x
            z = D_flat - np.sqrt(max(R * R - val * val, 0.0))  # rim at z=0
            return np.array([x, y, z])

        R_inner = rbp * R_ap
        R_accept = 1.005 * R_ap

        frames = subdivide_parametric_surface(
            surface_fn,
            u_range=(-R_ap, R_ap),
            v_range=(-R_ap, R_ap),
            n_u=self.no_sub_diameter,
            n_v=self.no_sub_diameter,
            inside_fn=lambda x, y: x * x + y * y <= R_inner * R_inner,
            accept_fn=lambda x, y: x * x + y * y <= R_accept * R_accept,
            border_refine=rf,
            normal_sign=1.0,
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
            f"R={self.radius_of_curvature * 1e3:.2f} mm, "
            f"axis={self.focus_axis}, "
            f"no_sub_diameter={self.no_sub_diameter}, "
            f"fc={self.fc / 1e6:.2f} MHz)"
        )
