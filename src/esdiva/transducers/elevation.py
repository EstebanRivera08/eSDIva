"""Single rectangular element with an arbitrary elevation surface profile ``z(y)``."""

from typing import Callable, List, Tuple

import numpy as np

from . import geometry_utils, validators
from .base import TransducerBase


class ElevationProfileTransducer(TransducerBase):
    """Rigid rectangular element whose face is curved along elevation only, ``z = f(y)``.

    The building block of every elevation lens: straight along its width (x),
    shaped along its height (y) by any profile — flat, circular (cylindrical
    lens), parabolic, aspheric, or one found by optimisation. Each element of a
    `LinearArrayTransducer`, `ConvexArrayTransducer` or
    `RowColumnAddressedTransducer` given the same ``elevation_profile_mm`` is this
    element, copied along the array.

    The face is tiled in ``no_sub_y`` strips of equal arc length so each flat patch
    follows the curve, and the two rims (``y = ±height/2``) sit at ``z = 0``: a
    concave profile dishes back (centre at ``z < 0``, converging, focuses the
    elevation beam), a convex one bulges into the medium (diverging).

    Parameters
    ----------
    element_width_mm : float
        Element width along x, in mm.
    element_height_mm : float
        Element height along y (the curved direction), in mm.
    elevation_profile_mm : callable
        Surface shape ``y_mm (ndarray) -> z_mm (ndarray)``, ``+z`` toward the
        medium. E.g. ``lambda y: y**2 / (2 * 12)`` is a parabolic concave lens of
        12 mm apex radius.
    no_sub_x : int
        Patches across the width (≥ 1).
    no_sub_y : int
        Patches along the curved height (≥ 2).
    frequency_Hz : float, default: 1e6
        Centre frequency in Hz.
    """

    def __init__(
        self,
        *,
        element_width_mm: float,
        element_height_mm: float,
        elevation_profile_mm: Callable[[np.ndarray], np.ndarray],
        no_sub_x: int,
        no_sub_y: int,
        frequency_Hz: float = 1e6,
    ) -> None:
        super().__init__()
        validators.validate_positive(element_width_mm, "element_width_mm", strict=True)
        validators.validate_positive(
            element_height_mm, "element_height_mm", strict=True
        )
        no_sub_x, no_sub_y = validators.validate_subdivisions(no_sub_x, no_sub_y)
        self._elev_sag = geometry_utils.elevation_sag_from_inputs(
            None, elevation_profile_mm, element_height_mm, no_sub_y
        )
        self.type, self.name = "elevation_profile", "ElevationProfileTransducer"
        self.n_elements = 1
        self.elem_width = element_width_mm * 1e-3
        self.elem_height = element_height_mm * 1e-3
        self.no_sub_x, self.no_sub_y = no_sub_x, no_sub_y
        self.fc = float(frequency_Hz)

    def _tile(self) -> Tuple[List[np.ndarray], float, float]:
        return geometry_utils.elevation_profile_quads(
            self.elem_width,
            self.elem_height,
            self.no_sub_x,
            self.no_sub_y,
            self._elev_sag,
        )

    def _default_elevation_lens_sag(self) -> float:
        """SIGNED centre offset (m): + recessed (concave), − protruding (convex).

        Reception subtracts it from the beamforming time origin, because the
        radiating centre sits that far behind (or ahead of) the ``z = 0`` plane a
        beamformer measures depth from.
        """
        return self._tile()[2]

    def _compute_element_centers(self) -> np.ndarray:
        return np.array([[0.0, 0.0, 0.0]])

    def _build_subdivisions(self) -> Tuple[List[np.ndarray], float, List[int]]:
        quads, area, _ = self._tile()
        return quads, area, [0] * len(quads)

    def __repr__(self) -> str:
        return (
            f"ElevationProfileTransducer(width={self.elem_width * 1e3:.3f} mm, "
            f"height={self.elem_height * 1e3:.3f} mm, "
            f"lens_sag={self.elevation_lens_sag * 1e6:.1f} um, "
            f"no_sub=({self.no_sub_x},{self.no_sub_y}), fc={self.fc / 1e6:.2f} MHz)"
        )
