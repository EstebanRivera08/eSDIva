"""
CustomTransducer — composite array from arbitrary mono-element transducers.

Researchers often need to model multi-element arrays whose geometry cannot be
described by a simple rectangular grid (linear or matrix).  A TUS helmet, for
example, is built from dozens of spherical-bowl transducers aimed at a common
target from different directions.  ``CustomTransducer`` handles these cases by
assembling any collection of mono-element transducer objects into a single
array that the eSDIva simulator can process normally.

Usage example — TUS helmet::

    from esdiva.transducers import ConcaveCircularTransducer, CustomTransducer
    import numpy as np

    # Ten identical bowl elements, positions and normals defined by the user
    elem = ConcaveCircularTransducer(diameter_mm=30, focus_mm=60,
                                     no_sub_diameter=20, frequency_Hz=0.5e6)

    positions = np.array([...])  # shape (10, 3), in mm
    normals   = np.array([...])  # shape (10, 3), unit vectors toward target

    helmet = CustomTransducer(
        elements=[elem] * 10,
        positions_mm=positions,
        normals=normals,
    )
    helmet.compute_delays(focus_mm=[0, 0, 0])
    helmet.show()
"""

import warnings
from time import time as TIME
from typing import List, Optional, Tuple

import numpy as np

from . import geometry_utils
from .base import TransducerBase


class CustomTransducer(TransducerBase):
    """
    Multi-element array assembled from individual mono-element transducers.

    Each element can be any ``TransducerBase`` subclass that represents a
    single physical source (``FlatCircularTransducer``, ``ConcaveCircularTransducer``,
    ``FocusedCircularTransducer``, or any custom subclass with
    ``n_elements == 1``).  The assembled array supports electronic delays and
    per-element apodization, enabling beam steering and focusing.

    The patches of each element are rigidly transformed — rotated to align
    their normal axis with the provided direction, then translated to the
    given position.  By default, all elements point in the +z direction
    (the eSDIva propagation axis).

    Parameters
    ----------
    elements : list of TransducerBase
        Individual mono-element transducer objects.  All must have
        ``n_elements == 1``.  They may be of different types or sizes,
        though sharing the same type is most common.
    positions_mm : array-like, shape (N, 3)
        3-D centre position of each element in mm.
    normals : array-like, shape (N, 3), optional
        Unit vectors pointing from each element toward the target medium
        (i.e. in the direction of wave propagation).  Defaults to
        ``[0, 0, 1]`` for all elements (all elements flat, facing +z).
    frequency_Hz : float, optional
        Override the centre frequency reported to the simulator.  If
        ``None`` (default) the frequency of the first element is used.

    Raises
    ------
    ValueError
        If any element has ``n_elements != 1``.
    """

    def __init__(
        self,
        elements: List[TransducerBase],
        positions_mm,
        normals=None,
        *,
        frequency_Hz: Optional[float] = None,
    ) -> None:
        super().__init__()
        t0 = TIME()

        self.type = "custom"
        self.name = "CustomTransducer"

        if not elements:
            raise ValueError("elements list must not be empty.")

        for i, el in enumerate(elements):
            if el.n_elements != 1:
                raise ValueError(
                    f"Element {i} ({el.name}) has n_elements={el.n_elements}. "
                    "Only mono-element transducers (n_elements=1) can be assembled "
                    "into a CustomTransducer."
                )

        positions_mm = np.asarray(positions_mm, dtype=float)
        if positions_mm.ndim != 2 or positions_mm.shape != (len(elements), 3):
            raise ValueError(
                f"positions_mm must have shape (N, 3), got {positions_mm.shape}."
            )

        if normals is None:
            normals = np.tile([0.0, 0.0, 1.0], (len(elements), 1))
        else:
            normals = np.asarray(normals, dtype=float)
            if normals.shape != (len(elements), 3):
                raise ValueError(
                    f"normals must have shape (N, 3), got {normals.shape}."
                )
            norms = np.linalg.norm(normals, axis=1, keepdims=True)
            if np.any(norms < 1e-12):
                raise ValueError("One or more normal vectors have zero magnitude.")
            normals = normals / norms

        # Overlap check — warn if any two element centres are closer than the
        # sum of their approximate radii (elem_width / 2 for each).
        N = len(elements)
        for i in range(N):
            r_i = elements[i].elem_width / 2  # metres
            for j in range(i + 1, N):
                r_j = elements[j].elem_width / 2
                dist_mm = np.linalg.norm(positions_mm[i] - positions_mm[j])
                min_dist_mm = (r_i + r_j) * 1e3
                if dist_mm < min_dist_mm:
                    raise ValueError(
                        f"Elements {i} and {j} overlap: centres are {dist_mm:.2f} mm apart "
                        f"but combined radius is {min_dist_mm:.2f} mm.  "
                        "Adjust positions_mm or use smaller elements."
                    )

        self._elements = elements
        self._positions_m = positions_mm * 1e-3
        self._normals = normals

        self.n_elements = len(elements)
        self.fc = float(frequency_Hz) if frequency_Hz is not None else elements[0].fc

        # Derive patch-size attributes from the first element (used by eSDIva
        # to compute the far-field condition warning).
        ref = elements[0]
        self.elem_width = ref.elem_width
        self.elem_height = ref.elem_height
        self.no_sub_x = ref.no_sub_x
        self.no_sub_y = ref.no_sub_y

        if len({type(e) for e in elements}) > 1:
            warnings.warn(
                "CustomTransducer contains elements of different types.  "
                "Patch-size attributes (elem_width, elem_height, no_sub_x/y) "
                "are taken from the first element and may not represent all patches.",
                UserWarning,
            )

        # Build geometry immediately so n_sub_patches is available
        _ = self.sub_quad_verts

        total_patches = sum(len(e.sub_quad_verts) for e in elements)
        print(
            f"CustomTransducer initialised in {TIME() - t0:.4f} s  "
            f"({self.n_elements} elements, {total_patches} total patches)."
        )

    # ------------------------------------------------------------------
    # Abstract method implementations
    # ------------------------------------------------------------------

    def _compute_element_centers(self) -> np.ndarray:
        """
        Element centres are the user-supplied positions (in metres).

        These are the points from which the electronic delay law is computed.
        """
        return self._positions_m.copy()

    def _build_subdivisions(
        self,
    ) -> Tuple[List[np.ndarray], float, List[int]]:
        """
        Assemble patches from all elements, each rigidly transformed.

        Each element's patches are rotated to align with its normal vector,
        then translated to its position.  The element index in the assembled
        array corresponds to the order in ``elements``.
        """
        all_quads: List[np.ndarray] = []
        all_el_idx: List[int] = []
        representative_area: float = 0.0

        for elem_idx, (element, pos_m, normal) in enumerate(
            zip(self._elements, self._positions_m, self._normals)
        ):
            R = geometry_utils.rotation_matrix_z_to_normal(normal)
            transformed = geometry_utils.transform_patches(
                element.sub_quad_verts, R, pos_m
            )
            all_quads.extend(transformed)
            all_el_idx.extend([elem_idx] * len(transformed))
            if elem_idx == 0:
                representative_area = element.sub_area

        return all_quads, representative_area, all_el_idx

    # ------------------------------------------------------------------
    # Apodization — override to support 'none', 'uniform' keywords
    # ------------------------------------------------------------------

    def compute_apodization(
        self,
        focus_mm=None,
        *,
        FoverD=None,
        apodization_type: Optional[str] = None,
        plot: bool = False,
        inline: bool = True,
    ) -> np.ndarray:
        """
        Set per-element apodization weights.

        For a ``CustomTransducer`` the aperture geometry is entirely determined
        by the element positions and normals supplied at construction, so only
        simple weight patterns are supported here.

        Parameters
        ----------
        focus_mm : array-like, optional
            Accepted for API consistency.
        FoverD : float, optional
            Accepted for API consistency.
        apodization_type : {'none', 'uniform'} or None
            ``'none'`` / ``'uniform'`` / ``None`` all give all-ones weights.
            For custom weight patterns, use ``set_apodization()`` directly.
        plot : bool
            Accepted for API consistency.
        inline : bool
            If True (default), store result in ``self.apodization``.

        Returns
        -------
        ndarray
            Uniform apodization weights, shape ``(n_elements,)``.
        """
        apod = np.ones(self.n_elements, dtype=float)
        if inline:
            self.apodization = apod
            self.apodization_type = apodization_type or "uniform"
        if plot:
            self.plot_apodization()
        return apod

    # ------------------------------------------------------------------
    # Helpers for introspection
    # ------------------------------------------------------------------

    @property
    def elements(self) -> List[TransducerBase]:
        """The individual mono-element transducers that make up this array.

        Returns
        -------
        list of TransducerBase
            Constituent mono-element transducers.
        """
        return self._elements

    @property
    def positions_mm(self) -> np.ndarray:
        """Element centre positions in mm, shape ``(n_elements, 3)``.

        Returns
        -------
        ndarray
            Positions in millimetres.
        """
        return self._positions_m * 1e3

    @property
    def normals(self) -> np.ndarray:
        """Unit normal vectors for each element, shape ``(n_elements, 3)``.

        Returns
        -------
        ndarray
            Unit normals for each element.
        """
        return self._normals.copy()

    def __repr__(self) -> str:
        types = list({type(e).__name__ for e in self._elements})
        return (
            f"CustomTransducer("
            f"n_elements={self.n_elements}, "
            f"element_types={types}, "
            f"fc={self.fc / 1e6:.2f} MHz)"
        )
