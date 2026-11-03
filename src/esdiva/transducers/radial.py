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
    surface_sag_mm : callable
        ``r_mm (ndarray) -> z_mm (ndarray)`` radial profile, in millimetres
        (same mm convention as ``diameter_mm``). Converted to SI internally.
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
        surface_sag_mm: Callable[[np.ndarray], np.ndarray],
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

        # The user profile is mm->mm (package convention); wrap it into the internal
        # SI callable r_m->z_m the tilers and virtual_source_m expect.
        def _sag_m(r_m, _f=surface_sag_mm):
            return np.asarray(_f(np.asarray(r_m, float) * 1e3), float) * 1e-3

        # Pin the highest point of the profile to z=0; the rest recedes to z<0.
        z_peak = float(np.max(_sag_m(np.linspace(0.0, self.radius, 512))))
        self._sag = lambda r, _s=_sag_m, _z0=z_peak: _s(r) - _z0

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

    @property
    def virtual_source_m(self) -> np.ndarray:
        """Acoustic centre the cap radiates from — where a beamformer references delays.

        **No field is simulated.** A rigid cap moves as one surface, so by Huygens every
        point emits in phase at ``t = 0`` and the initial wavefront *is* the surface. A
        spherical surface thus radiates from its own centre of curvature — the "virtual
        source" a delay-and-sum must place the element at, not the aperture plane. Diverging
        cap (apex at ``z = 0``, surface receding) → source BEHIND at ``z < 0``; converging
        bowl → focus in front at ``z > 0``. Using ``z = 0`` mis-times every echo by the
        cap's curvature and defocuses a compounded reconstruction.

        Found by least-squares fitting the stored ``z(r)`` to an on-axis circle: expanding
        ``r² + (z − z_c)² = R_c²`` to ``r² + z² = 2·z_c·z + const`` is linear in ``z_c``,
        so one line fit recovers the centre (exact for a sphere, best-fit for an aspheric).
        On-axis by symmetry (``x = y = 0``); a flat piston collapses to ``(0, 0, 0)``.

        Returns
        -------
        (3,) numpy.ndarray
            Virtual-source position ``[0, 0, z_vs]`` in metres. ``z_vs`` is signed:
            negative behind the aperture (diverging cap), positive in front
            (converging bowl), zero for a flat piston.
        """
        r = np.linspace(0.0, self.radius, 256)
        z = np.asarray(self._sag(r), dtype=float)
        if float(np.ptp(z)) < 1e-12 * self.radius:  # flat piston: no curvature
            return np.array([0.0, 0.0, 0.0])
        # Geometric centre; that it is the acoustically right datum is confirmed by
        # sweeping it in a pulse-echo PSF (the lobe tightens vs the aperture plane).
        # r²+z² = 2·z_c·z + const: slope against z (col 0) is 2·z_c.
        coeff, *_ = np.linalg.lstsq(
            np.column_stack([z, np.ones_like(z)]), r * r + z * z, rcond=None
        )
        return np.array([0.0, 0.0, float(coeff[0]) / 2.0])

    def __repr__(self) -> str:
        return (
            f"RadialProfileTransducer(diameter={self.diameter * 1e3:.2f} mm, "
            f"method={self.method}, no_sub_diameter={self.no_sub_diameter}, "
            f"fc={self.fc / 1e6:.2f} MHz, patches={self.n_sub_patches})"
        )
