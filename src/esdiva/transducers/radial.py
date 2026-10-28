"""Single-element transducer with an arbitrary radial surface profile ``z(r)``."""

from typing import Callable, Dict, List, Tuple

import numpy as np

from esdiva.utilities.surface_subdivision import subdivide_cartesian, subdivide_radial

from . import validators
from .base import TransducerBase


class RadialProfileTransducer(TransducerBase):
    """Rigid single element whose surface is any radial sag ``z(r)``.

    Covers every axisymmetric cap — flat piston, convex dome, concave bowl,
    aspheric lens. The highest point of the profile is pinned to ``z = 0`` and the
    rest recedes to ``z < 0`` (for a concave bowl that apex is the rim, for a
    convex/divergent cap it is the centre); a pure z-translation leaves the
    far-field directivity unchanged.

    Parameters
    ----------
    diameter_mm : float
        Aperture diameter in millimetres.
    surface_sag_m : callable
        ``r_m (ndarray) -> z_m (ndarray)`` radial profile in metres.
    frequency_Hz : float, default: 1e6
        Centre frequency in Hz.
    no_sub_diameter : int, default: 40
        Subdivision resolution across the diameter (``n_rings = no_sub_diameter // 2``).
    method : {"radial", "cartesian"}, default: "radial"
        ``"radial"`` tiles by concentric rings — no shear, exact circular rim,
        right for any curved cap. ``"cartesian"`` tiles an ``(x, y)`` grid — fine
        for a flat / mildly curved piston, shears a strongly curved cap.
    ratio_big_patches : float, default: 0.8
        Fraction of outer rings kept coarse (radial method).
    refine_factor : int, default: 3
        Sub-ring count for the refined inner rings (radial method).
    """

    def __init__(
        self,
        *,
        diameter_mm: float,
        surface_sag_m: Callable[[np.ndarray], np.ndarray],
        frequency_Hz: float = 1e6,
        no_sub_diameter: int = 40,
        method: str = "radial",
        ratio_big_patches: float = 0.8,
        refine_factor: int = 3,
    ) -> None:
        super().__init__()
        validators.validate_positive(diameter_mm, "diameter_mm", strict=True)
        validators.validate_integer(no_sub_diameter, "no_sub_diameter", min_val=4)
        if method not in ("radial", "cartesian"):
            raise ValueError(f"method must be 'radial' or 'cartesian', got {method!r}")

        self.type, self.name = "convex_bowl", "RadialProfileTransducer"
        self.diameter = diameter_mm * 1e-3
        self.radius = self.diameter / 2

        # Pin the highest point of the profile to z=0; the rest recedes to z<0.
        z_peak = float(np.max(surface_sag_m(np.linspace(0.0, self.radius, 512))))
        self._sag = lambda r, _s=surface_sag_m, _z0=z_peak: _s(r) - _z0

        self.no_sub_diameter = self.no_sub_x = self.no_sub_y = no_sub_diameter
        self.method = method
        self.ratio_big_patches = ratio_big_patches
        self.refine_factor = refine_factor
        self.n_elements = 1
        self.elem_width = self.elem_height = self.diameter
        self.fc = float(frequency_Hz)
        _ = self.sub_quad_verts

    def _compute_element_centers(self) -> np.ndarray:
        return np.array([[0.0, 0.0, 0.0]])

    def _build_subdivisions(self) -> Tuple[List[np.ndarray], float, List[int]]:
        if self.method == "radial":
            frames = subdivide_radial(
                self._sag,
                self.radius,
                self.no_sub_diameter // 2,
                ratio_big_patches=self.ratio_big_patches,
                refine_factor=self.refine_factor,
            )
        else:
            r = self.radius

            def fn(x, y):
                return np.array([x, y, float(self._sag(np.array([np.hypot(x, y)]))[0])])

            frames = subdivide_cartesian(
                fn,
                u_range=(-r, r),
                v_range=(-r, r),
                n_u=self.no_sub_diameter,
                n_v=self.no_sub_diameter,
                inside_fn=lambda x, y: x * x + y * y
                <= (self.ratio_big_patches * r) ** 2,
                accept_fn=lambda x, y: x * x + y * y <= (1.005 * r) ** 2,
                border_refine=self.refine_factor,
                normalize_patch_size=True,
            )
        self._sub_patch_frames = frames
        return (
            frames["corners"],
            float(np.mean(frames["wu"] * frames["wv"])),
            frames["el_idx"],
        )

    def _build_patch_frames(self) -> Dict:
        _ = self.sub_quad_verts
        assert self._sub_patch_frames is not None
        return self._sub_patch_frames

    def __repr__(self) -> str:
        return (
            f"RadialProfileTransducer(diameter={self.diameter * 1e3:.2f} mm, "
            f"method={self.method}, no_sub_diameter={self.no_sub_diameter}, "
            f"fc={self.fc / 1e6:.2f} MHz, patches={self.n_sub_patches})"
        )
