"""Import Field II transducer geometry into SonDI.

Converts the output of MATLAB ``xdc_get(Th, 'all')`` to a
:class:`FieldIITransducer`, which is a standard SonDI transducer whose
patches are taken directly from the Field II internal representation.

Typical MATLAB export workflow::

    Th = xdc_concave(R, Rfocus, ele_size);
    xdc_impulse(Th, impulse_response);
    xdc_excitation(Th, excitation);
    all_data = xdc_get(Th, 'all');
    save('tx_fieldii.mat', 'all_data');

Python import::

    import scipy.io
    from sondi.transducers import from_fieldii_xdc_data

    raw = scipy.io.loadmat('tx_fieldii.mat', simplify_cells=True)
    tx = from_fieldii_xdc_data(raw['all_data'], frequency_hz=3e6)

The function parses the ``geometri`` field (N × ≥17 matrix) where each row
encodes one rectangular mathematical element (patch):

    col 0   : element number (1-indexed, unused)
    col 1-3 : patch centre (x, y, z) in metres
    col 4-12: rotation matrix R stored column-major
              R[:,0]=cols 4-6, R[:,1]=cols 7-9, R[:,2]=cols 10-12
    col 13  : time delay in seconds
    col 14  : patch half-width  (metres, along R[:,0])
    col 15  : patch half-height (metres, along R[:,1])
    col 16  : apodization weight

Corner vertices are reconstructed as::

    c0 = centre - hw * R[:,0] - hh * R[:,1]
    c1 = centre + hw * R[:,0] - hh * R[:,1]
    c2 = centre + hw * R[:,0] + hh * R[:,1]
    c3 = centre - hw * R[:,0] + hh * R[:,1]

matching SonDI's quad-vertex ordering (edge v[1]-v[0] = u-direction,
edge v[3]-v[0] = v-direction).

.. note::
   Scale convention: SonDI uses ``rho / (2 * c²)`` while Field II uses
   approximately ``rho / 2``.  For a unit-amplitude scatterer the raw RF
   amplitudes therefore differ by a factor of ``c²`` (~2.37e6 at c=1540 m/s).
   Normalised PSF comparisons (envelope / peak) are unaffected.
"""

from typing import List, Optional, Tuple

import numpy as np

from .base import TransducerBase


class FieldIITransducer(TransducerBase):
    """Transducer built from Field II ``xdc_get`` patch data.

    Each patch is treated as an independent element so that per-patch
    apodization and delay values from Field II are preserved exactly.
    The transducer behaves identically to any other SonDI transducer —
    it can be passed to :class:`~sondi.emission.Emission`,
    :class:`~sondi.reception.Reception`, or
    :class:`~sondi.reception.ReceptionConventional`.

    Parameters
    ----------
    patch_quads : list of ndarray, each (4, 3)
        Corner vertices of every patch in metres, ordered
        ``[c0, c1, c2, c3]`` so that ``c1-c0`` is the u-tangent and
        ``c3-c0`` is the v-tangent.
    patch_apod : (N,) array-like
        Apodization weight for each patch.
    patch_delays : (N,) array-like
        Time delay for each patch in seconds.
    frequency_hz : float, default 1e6
        Transducer centre frequency in Hz.
    elevation_focus_mm : float, optional
        Elevation-lens focal length (= lens radius of curvature, Field II
        ``Rfocus``) in mm, for probes exported from ``xdc_focused_array`` /
        ``xdc_convex_focused_array``. The lens *curvature* is already carried
        by the imported patch geometry, but reception's RF time origin also
        needs the lens transit ``sag/c`` (the echo of a lens-focused aperture
        peaks one lens transit after the first-arriving rim). This computes
        ``sag = R − √(R² − (H/2)²)`` with ``H`` the full elevation (y) extent
        of the imported patches and stores it in ``elevation_lens_sag``.
        Leave ``None`` for unlensed probes (sag 0), or set
        ``elevation_lens_sag`` directly (metres) for exotic geometries.
    """

    def __init__(
        self,
        patch_quads: List[np.ndarray],
        patch_apod,
        patch_delays,
        *,
        frequency_hz: float = 1e6,
        elevation_focus_mm: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.type = "fieldii"
        self.name = "FieldIITransducer"

        self._quads: List[np.ndarray] = [
            np.asarray(q, dtype=np.float64) for q in patch_quads
        ]
        N = len(self._quads)
        if N == 0:
            raise ValueError("patch_quads must not be empty.")

        self.n_elements = N
        self.fc = float(frequency_hz)

        # Per-element (one element per Field II patch)
        self._apodization = np.asarray(patch_apod, dtype=np.float64).ravel()
        self._delays = np.asarray(patch_delays, dtype=np.float64).ravel()
        if len(self._apodization) != N or len(self._delays) != N:
            raise ValueError(
                f"patch_apod and patch_delays must each have length {N} "
                f"(number of patches)."
            )

        q0 = self._quads[0]
        self.elem_width = float(np.linalg.norm(q0[1] - q0[0]))
        self.elem_height = float(np.linalg.norm(q0[3] - q0[0]))
        self.no_sub_x = 1
        self.no_sub_y = 1

        if elevation_focus_mm is not None:
            # Lens transit for reception's RF origin: sag = R − √(R² − (H/2)²),
            # with H the full elevation (y) extent of the imported aperture
            # (Field II lenses curve along y).
            R = float(elevation_focus_mm) * 1e-3
            all_y = np.concatenate([q[:, 1] for q in self._quads])
            half_h = float(all_y.max() - all_y.min()) / 2.0
            if R < half_h:
                raise ValueError(
                    f"elevation_focus_mm ({elevation_focus_mm:.2f}) must be >= half "
                    f"the aperture elevation extent ({half_h * 1e3:.2f} mm)."
                )
            self.elevation_lens_sag = R - np.sqrt(R * R - half_h * half_h)

    # ------------------------------------------------------------------
    # TransducerBase abstract methods
    # ------------------------------------------------------------------

    def _compute_element_centers(self) -> np.ndarray:
        return np.array([q.mean(axis=0) for q in self._quads])

    def _build_subdivisions(
        self,
    ) -> Tuple[List[np.ndarray], float, List[int]]:
        areas = [
            float(np.linalg.norm(np.cross(q[1] - q[0], q[3] - q[0])))
            for q in self._quads
        ]
        mean_area = float(np.mean(areas)) if areas else 0.0
        el_idx = list(range(len(self._quads)))
        return list(self._quads), mean_area, el_idx

    def __repr__(self) -> str:
        return (
            f"FieldIITransducer(n_patches={self.n_elements}, "
            f"fc={self.fc / 1e6:.2f} MHz)"
        )


def from_fieldii_xdc_data(
    data,
    *,
    frequency_hz: Optional[float] = None,
    elevation_focus_mm: Optional[float] = None,
) -> FieldIITransducer:
    """Create a :class:`FieldIITransducer` from ``xdc_get(Th, 'all')`` output.

    Parameters
    ----------
    data : dict or structured ndarray
        Python representation of the MATLAB struct returned by
        ``xdc_get(Th, 'all')``, typically loaded with
        ``scipy.io.loadmat(..., simplify_cells=True)['all_data']``.
    frequency_hz : float, optional
        Transducer centre frequency in Hz.  If None, the function tries
        to read ``data['f0']``; falls back to 1 MHz if not present.
    elevation_focus_mm : float, optional
        Elevation-lens focal length in mm (Field II ``Rfocus`` of
        ``xdc_focused_array``); sets ``elevation_lens_sag`` so reception's
        RF time origin includes the lens transit. None for unlensed probes.

    Returns
    -------
    FieldIITransducer
        SonDI transducer with one element per Field II mathematical element.

    Raises
    ------
    KeyError
        If ``data`` has no ``geometri`` field.
    ValueError
        If the ``geometri`` matrix does not have at least 17 columns.

    Notes
    -----
    The ``geometri`` column layout assumed here is the Field II v3.x format::

        col 0   : element number (1-indexed, skipped)
        col 1-3 : centre (x, y, z) [m]
        col 4-6 : R[:,0] — global u-tangent direction
        col 7-9 : R[:,1] — global v-tangent direction
        col 10-12: R[:,2] — patch normal (unused directly)
        col 13  : delay [s]
        col 14  : half-width  [m] (along u-tangent)
        col 15  : half-height [m] (along v-tangent)
        col 16  : apodization weight

    If the Field II version stores **full** widths in cols 14–15 instead of
    half-widths, pass the result through :func:`from_fieldii_patch_arrays`
    with ``half_widths=False``.
    """
    # ---- Extract geometri ----
    if isinstance(data, dict):
        if "geometri" not in data:
            raise KeyError(
                "'geometri' field not found in data. "
                "Available keys: " + str(list(data.keys()))
            )
        geom = np.asarray(data["geometri"], dtype=np.float64)
        fc_raw = data.get("f0", None)
    else:
        # Numpy structured array (scipy.io.loadmat without simplify_cells)
        geom = np.asarray(data["geometri"].squeeze(), dtype=np.float64)
        try:
            fc_raw = float(data["f0"].squeeze())
        except Exception:
            fc_raw = None

    if geom.ndim == 1:
        geom = geom.reshape(1, -1)
    if geom.shape[1] < 17:
        raise ValueError(
            f"geometri has {geom.shape[1]} columns; expected at least 17. "
            "Check Field II version or export format."
        )

    # ---- Resolve frequency ----
    if frequency_hz is None:
        if fc_raw is not None:
            try:
                frequency_hz = float(fc_raw)
            except (TypeError, ValueError):
                frequency_hz = 1e6
        else:
            frequency_hz = 1e6

    # ---- Parse patch geometry ----
    centres = geom[:, 1:4]  # (N, 3) patch centres in metres
    u_dir = geom[:, 4:7]  # (N, 3) global u-tangent (R[:,0])
    v_dir = geom[:, 7:10]  # (N, 3) global v-tangent (R[:,1])
    delays = geom[:, 13]  # (N,)   delay [s]
    half_wx = geom[:, 14]  # (N,)   half-width in u-direction [m]
    half_wy = geom[:, 15]  # (N,)   half-height in v-direction [m]
    apod = geom[:, 16]  # (N,)   apodization

    # Normalise direction vectors in case Field II stores non-unit vectors
    u_norm = np.linalg.norm(u_dir, axis=1, keepdims=True)
    v_norm = np.linalg.norm(v_dir, axis=1, keepdims=True)
    u_dir = np.where(u_norm > 1e-12, u_dir / u_norm, u_dir)
    v_dir = np.where(v_norm > 1e-12, v_dir / v_norm, v_dir)

    # ---- Reconstruct 4 corner vertices per patch ----
    # SonDI quad ordering: c0 = -u -v, c1 = +u -v, c2 = +u +v, c3 = -u +v
    hw = half_wx[:, None]  # (N, 1)
    hh = half_wy[:, None]  # (N, 1)
    c0 = centres - hw * u_dir - hh * v_dir
    c1 = centres + hw * u_dir - hh * v_dir
    c2 = centres + hw * u_dir + hh * v_dir
    c3 = centres - hw * u_dir + hh * v_dir

    # Stack into list of (4, 3) arrays
    patch_quads = [
        np.stack([c0[i], c1[i], c2[i], c3[i]], axis=0) for i in range(len(centres))
    ]

    return FieldIITransducer(
        patch_quads,
        apod,
        delays,
        frequency_hz=frequency_hz,
        elevation_focus_mm=elevation_focus_mm,
    )


def from_fieldii_rect_data(
    rect,
    *,
    frequency_hz: float = 1e6,
    elevation_focus_mm: Optional[float] = None,
) -> FieldIITransducer:
    """Create a :class:`FieldIITransducer` from ``xdc_get(Th, 'rect')`` output.

    ``xdc_get(Th, 'rect')`` returns a 26 × M matrix with one column per
    mathematical element (patch).  Rows used here (0-indexed):

        row 0     : physical element number
        row 4     : apodization weight
        rows 10-21: four corner vertices, each (x, y, z) in metres
        row 22    : time delay [s]

    Field II lists the corners walking the rectangle perimeter, which is
    NOT SonDI's quad ordering (``c1-c0`` must be the u-tangent and
    ``c3-c0`` the v-tangent).  Each quad is therefore re-ordered from the
    corner positions themselves: the corner farthest from ``c0`` is the
    diagonal, the two remaining corners give the u and v edges.  This makes
    the import robust to any corner ordering Field II may produce.

    Typical MATLAB export::

        rect = xdc_get(Th, 'rect');
        save('tx_rect.mat', 'rect');

    Parameters
    ----------
    rect : (26, M) array-like
        The ``xdc_get(Th, 'rect')`` matrix (also accepted transposed).
    frequency_hz : float, default 1e6
        Transducer centre frequency in Hz.
    elevation_focus_mm : float, optional
        Elevation-lens focal length in mm (Field II ``Rfocus``); sets
        ``elevation_lens_sag`` so reception's RF time origin includes the
        lens transit.  None for unlensed probes.

    Returns
    -------
    FieldIITransducer
        SonDI transducer with one element per Field II mathematical element.
    """
    geom = np.asarray(rect, dtype=np.float64)
    if geom.ndim != 2:
        raise ValueError("rect must be a 2-D matrix from xdc_get(Th, 'rect').")
    if geom.shape[0] != 26 and geom.shape[1] == 26:
        geom = geom.T
    if geom.shape[0] < 23:
        raise ValueError(
            f"rect has {geom.shape[0]} rows; expected the 26-row "
            "xdc_get(Th, 'rect') format."
        )

    corners = geom[10:22].T.reshape(-1, 4, 3)  # (M, 4, 3) metres, order unknown
    apod = geom[4]
    delays = geom[22]

    patch_quads = []
    for c4 in corners:
        # Re-order to SonDI's quad convention from geometry alone: the
        # farthest corner from c0 is the diagonal; the other two corners are
        # the u- and v-edge neighbours.
        d = np.linalg.norm(c4[1:] - c4[0], axis=1)
        i_diag = 1 + int(np.argmax(d))
        i_u, i_v = [i for i in (1, 2, 3) if i != i_diag]
        quad = np.stack([c4[0], c4[i_u], c4[i_diag], c4[i_v]], axis=0)
        u = quad[1] - quad[0]
        v = quad[3] - quad[0]
        cos_uv = abs(u @ v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-30)
        if cos_uv > 1e-3:
            raise ValueError(
                "Patch corners do not form a rectangle (u·v ≠ 0). "
                "Check the export format."
            )
        patch_quads.append(quad)

    return FieldIITransducer(
        patch_quads,
        apod,
        delays,
        frequency_hz=frequency_hz,
        elevation_focus_mm=elevation_focus_mm,
    )


def from_fieldii_patch_arrays(
    centres,
    u_tangents,
    v_tangents,
    half_widths,
    half_heights,
    *,
    delays=None,
    apodization=None,
    frequency_hz: float = 1e6,
    half_widths_input: bool = True,
    elevation_focus_mm: Optional[float] = None,
) -> FieldIITransducer:
    """Build a :class:`FieldIITransducer` from explicit patch arrays.

    Use this when you have exported the geometric data from MATLAB into
    separate arrays rather than the raw ``xdc_get`` struct.  Example MATLAB
    export::

        data = xdc_get(Th, 'all');
        g = data.geometri;
        centres   = g(:, 2:4);      % Nx3
        u_tang    = g(:, 5:7);      % Nx3  (R[:,0])
        v_tang    = g(:, 8:10);     % Nx3  (R[:,1])
        half_wx   = g(:, 15);       % Nx1
        half_wy   = g(:, 16);       % Nx1
        delays    = g(:, 14);       % Nx1
        apods     = g(:, 17);       % Nx1
        save('patches.mat', 'centres','u_tang','v_tang','half_wx','half_wy','delays','apods');

    Parameters
    ----------
    centres : (N, 3) array-like
        Patch centres in metres.
    u_tangents : (N, 3) array-like
        Global u-tangent unit vectors.
    v_tangents : (N, 3) array-like
        Global v-tangent unit vectors.
    half_widths : (N,) array-like
        Patch half-widths in metres (or full widths if
        ``half_widths_input=False``).
    half_heights : (N,) array-like
        Patch half-heights in metres (or full heights if
        ``half_widths_input=False``).
    delays : (N,) array-like or None
        Per-patch delay in seconds.  None defaults to zeros.
    apodization : (N,) array-like or None
        Per-patch apodization weight.  None defaults to ones.
    frequency_hz : float, default 1e6
        Transducer centre frequency in Hz.
    half_widths_input : bool, default True
        If True, ``half_widths`` and ``half_heights`` are half-widths.
        If False, they are full widths (will be halved internally).
    elevation_focus_mm : float, optional
        Elevation-lens focal length in mm (Field II ``Rfocus``); sets
        ``elevation_lens_sag`` for reception's RF time origin.

    Returns
    -------
    FieldIITransducer
        SonDI transducer with one patch per supplied centre/tangent row.
    """
    centres = np.asarray(centres, dtype=np.float64)
    u_dir = np.asarray(u_tangents, dtype=np.float64)
    v_dir = np.asarray(v_tangents, dtype=np.float64)
    hw = np.asarray(half_widths, dtype=np.float64).ravel()
    hh = np.asarray(half_heights, dtype=np.float64).ravel()
    N = len(centres)

    if not half_widths_input:
        hw = hw / 2.0
        hh = hh / 2.0

    if delays is None:
        delays = np.zeros(N, dtype=np.float64)
    if apodization is None:
        apodization = np.ones(N, dtype=np.float64)

    # Normalise tangents
    u_norm = np.linalg.norm(u_dir, axis=1, keepdims=True)
    v_norm = np.linalg.norm(v_dir, axis=1, keepdims=True)
    u_dir = np.where(u_norm > 1e-12, u_dir / u_norm, u_dir)
    v_dir = np.where(v_norm > 1e-12, v_dir / v_norm, v_dir)

    hw2 = hw[:, None]
    hh2 = hh[:, None]
    c0 = centres - hw2 * u_dir - hh2 * v_dir
    c1 = centres + hw2 * u_dir - hh2 * v_dir
    c2 = centres + hw2 * u_dir + hh2 * v_dir
    c3 = centres - hw2 * u_dir + hh2 * v_dir

    patch_quads = [np.stack([c0[i], c1[i], c2[i], c3[i]], axis=0) for i in range(N)]

    return FieldIITransducer(
        patch_quads,
        apodization,
        delays,
        frequency_hz=frequency_hz,
        elevation_focus_mm=elevation_focus_mm,
    )
