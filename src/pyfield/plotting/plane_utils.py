"""Unified plane handling utilities for 2D and 3D pressure field plotting.

Single source of truth for plane metadata, validation, and coordinate inference.
"""

from dataclasses import dataclass
from typing import TypedDict

import numpy as np


class _PlaneMeta(TypedDict):
    axes: tuple[str, str]
    normal: str
    ci: int
    xlabel: str
    ylabel: str


# Axis name → index in a (x, y, z) tuple
AXIS_IDX: dict[str, int] = {"x": 0, "y": 1, "z": 2}

# Plane metadata: axis pair, normal axis, index into (x,y,z) center tuple, labels
PLANE_META: dict[str, _PlaneMeta] = {
    "xz": {
        "axes": ("x", "z"),
        "normal": "y",
        "ci": 1,
        "xlabel": "X (mm)",
        "ylabel": "Z (mm)",
    },
    "xy": {
        "axes": ("x", "y"),
        "normal": "z",
        "ci": 2,
        "xlabel": "X (mm)",
        "ylabel": "Y (mm)",
    },
    "yz": {
        "axes": ("y", "z"),
        "normal": "x",
        "ci": 0,
        "xlabel": "Y (mm)",
        "ylabel": "Z (mm)",
    },
}


@dataclass
class PlaneSpec:
    """Specification of a single 2D pressure plane.

    Attributes
    ----------
    name : str
        Plane identifier: ``"xz"``, ``"xy"``, or ``"yz"``.
    data : numpy.ndarray
        2D ``(N1, N2)`` for static or 3D ``(Nt, N1, N2)`` for transient.
    translation : tuple
        ``(tx, ty, tz)`` shift in mm.  In-plane components shift the extent;
        the normal component positions the plane along its normal axis (used
        for titles in 2D and mesh offset in 3D).  Default ``(0.0, 0.0, 0.0)``.
    extent : tuple of float or None
        ``(c1_min, c1_max, c2_min, c2_max)`` for imshow extent.
    c1 : numpy.ndarray or None
        First in-plane coordinate array. When planes come from different
        simulations with different grids, each plane carries its own coords.
    c2 : numpy.ndarray or None
        Second in-plane coordinate array.
    """

    name: str
    data: np.ndarray
    translation: tuple = (0.0, 0.0, 0.0)
    extent: tuple | None = None
    c1: np.ndarray | None = None
    c2: np.ndarray | None = None


def parse_planes(
    pressure_field, *, expected_ndim=3, coords=None, center_mm=None, center_to_max=False
):
    """Parse pressure input into a list of PlaneSpec objects.

    Accepts three input formats:

    1. **Old dict**: ``{"xz": arr, "xy": arr}`` — backward compat.
    2. **List of dicts**: ``[{"plane": "xz", "data": arr, "offset": 5.0}]``
       — new primary format.
    3. **4D ndarray**: ``(Nt, Nx, Ny, Nz)`` volume — slices extracted at center.

    Parameters
    ----------
    pressure_field : dict, list, or numpy.ndarray
        Input pressure data in any of the three formats above.
    expected_ndim : int
        Expected ndim of each plane's data: 2 for static, 3 for transient.
    coords : dict, optional
        ``{"x": array, "y": array, "z": array}`` for coordinate inference
        when slicing a volume.
    center_mm : tuple of float, optional
        ``(x0, y0, z0)`` in mm for slice position (volume input) or title
        annotation (plane inputs).
    center_to_max : bool, optional
        If True and input is a volume, slice through the global pressure
        maximum. Default False.

    Returns
    -------
    planes : list of PlaneSpec
        Parsed plane specifications.
    center_mm : tuple of float
        The center point used (computed if not provided).
    coords : dict
        Coordinate arrays ``{"x": ..., "y": ..., "z": ...}``.
    """
    if coords is None:
        coords = {}

    x = coords.get("x")
    y = coords.get("y")
    z = coords.get("z")

    # --- Format 2: list of dicts ---
    if (
        isinstance(pressure_field, list)
        and len(pressure_field) > 0
        and isinstance(pressure_field[0], dict)
    ):
        planes = []
        for item in pressure_field:
            name = item["plane"]
            _validate_plane_name(name)
            data = np.asarray(item["data"])
            if data.ndim != expected_ndim:
                raise ValueError(
                    f"Plane '{name}' expected {expected_ndim}D data, "
                    f"got shape {data.shape}"
                )
            # Accept "translation" (3D) or legacy "offset" (scalar → 3D)
            if "translation" in item:
                translation = tuple(item["translation"])
            elif "offset" in item:
                meta = PLANE_META[name]
                t = [0.0, 0.0, 0.0]
                t[AXIS_IDX[meta["normal"]]] = float(item["offset"])
                translation = tuple(t)
            else:
                translation = (0.0, 0.0, 0.0)
            # Per-plane extent or coordinates (for planes from different grids)
            extent = item.get("extent", None)
            if extent is not None:
                extent = tuple(extent)
            c1 = item.get("c1", None)
            c2 = item.get("c2", None)
            if c1 is not None:
                c1 = np.asarray(c1, dtype=float)
            if c2 is not None:
                c2 = np.asarray(c2, dtype=float)
            planes.append(
                PlaneSpec(
                    name=name,
                    data=data,
                    translation=translation,
                    extent=extent,
                    c1=c1,
                    c2=c2,
                )
            )

        x, y, z = _infer_coords_from_planes(planes, x, y, z)
        if center_mm is None:
            center_mm = (
                float(x[len(x) // 2]),
                float(y[len(y) // 2]),
                float(z[len(z) // 2]),
            )
        coords_out = {"x": x, "y": y, "z": z}
        return planes, center_mm, coords_out

    # --- Format 1: old dict ---
    if isinstance(pressure_field, dict):
        valid = {"xz", "xy", "yz"}
        bad = set(pressure_field.keys()) - valid
        if bad or not pressure_field:
            raise ValueError(
                f"Plane keys must be a non-empty subset of {valid}, "
                f"got {set(pressure_field.keys())}"
            )
        planes = []
        for name, data in pressure_field.items():
            data = np.asarray(data)
            if data.ndim != expected_ndim:
                raise ValueError(
                    f"Plane '{name}' must be {expected_ndim}D "
                    f"(expected_ndim={expected_ndim}), got shape {data.shape}"
                )
            planes.append(PlaneSpec(name=name, data=data))

        x, y, z = _infer_coords_from_planes(planes, x, y, z)
        if center_mm is None:
            center_mm = (
                float(x[len(x) // 2]),
                float(y[len(y) // 2]),
                float(z[len(z) // 2]),
            )
        # Set translation: only normal component from center_mm
        for ps in planes:
            meta = PLANE_META[ps.name]
            t = [0.0, 0.0, 0.0]
            t[meta["ci"]] = center_mm[meta["ci"]]
            ps.translation = tuple(t)

        coords_out = {"x": x, "y": y, "z": z}
        return planes, center_mm, coords_out

    # --- Format 3: 4D ndarray ---
    if isinstance(pressure_field, np.ndarray) and pressure_field.ndim == 4:
        nt, nx, ny, nz = pressure_field.shape
        if x is None:
            x = np.arange(nx, dtype=float)
        if y is None:
            y = np.arange(ny, dtype=float)
        if z is None:
            z = np.arange(nz, dtype=float)

        if center_to_max:
            idx = np.unravel_index(
                np.nanargmax(np.abs(pressure_field)), pressure_field.shape
            )
            _, xi, yi, zi = idx
        elif center_mm is not None:
            xi = int(np.argmin(np.abs(x - center_mm[0])))
            yi = int(np.argmin(np.abs(y - center_mm[1])))
            zi = int(np.argmin(np.abs(z - center_mm[2])))
        else:
            xi, yi, zi = nx // 2, ny // 2, nz // 2

        center_mm = (float(x[xi]), float(y[yi]), float(z[zi]))

        planes = []
        if nx > 1 and nz > 1:
            planes.append(
                PlaneSpec(
                    name="xz",
                    data=pressure_field[:, :, yi, :],
                    translation=(0.0, float(y[yi]), 0.0),
                )
            )
        if nx > 1 and ny > 1:
            planes.append(
                PlaneSpec(
                    name="xy",
                    data=pressure_field[:, :, :, zi],
                    translation=(0.0, 0.0, float(z[zi])),
                )
            )
        if ny > 1 and nz > 1:
            planes.append(
                PlaneSpec(
                    name="yz",
                    data=pressure_field[:, xi, :, :],
                    translation=(float(x[xi]), 0.0, 0.0),
                )
            )
        if not planes:
            raise ValueError("No non-degenerate 2D slices in the given 4D field.")

        coords_out = {"x": x, "y": y, "z": z}
        return planes, center_mm, coords_out

    raise ValueError(
        "pressure_field must be a 4D ndarray (Nt,Nx,Ny,Nz), a dict of planes "
        "with keys from {'xz','xy','yz'}, or a list of plane dicts."
    )


def infer_coords(planes, x=None, y=None, z=None):
    """Infer missing coordinate arrays from plane shapes.

    For each axis, finds a plane that uses it and creates an index array
    matching the plane's shape along that axis.

    Parameters
    ----------
    planes : list of PlaneSpec
        Parsed planes.
    x, y, z : numpy.ndarray, optional
        Known coordinate arrays. If None, inferred from plane shapes.

    Returns
    -------
    x : numpy.ndarray
        Lateral coordinate array.
    y : numpy.ndarray
        Elevation coordinate array.
    z : numpy.ndarray
        Axial coordinate array.
    """
    return _infer_coords_from_planes(planes, x, y, z)


def compute_plane_extents(planes, coords):
    """Compute imshow extents for each PlaneSpec from coordinate arrays.

    Sets ``plane.extent = (c1_min, c1_max, c2_min, c2_max)`` on each plane.
    Skips planes that already have an explicit extent set. Uses per-plane
    ``c1``/``c2`` arrays when available, falling back to the shared ``coords``.

    Parameters
    ----------
    planes : list of PlaneSpec
        Planes to annotate with extents.
    coords : dict
        ``{"x": array, "y": array, "z": array}`` — shared fallback coords.
    """
    for ps in planes:
        if ps.extent is not None:
            continue
        meta = PLANE_META[ps.name]
        c1 = ps.c1 if ps.c1 is not None else coords[meta["axes"][0]]
        c2 = ps.c2 if ps.c2 is not None else coords[meta["axes"][1]]
        t1 = ps.translation[AXIS_IDX[meta["axes"][0]]]
        t2 = ps.translation[AXIS_IDX[meta["axes"][1]]]
        ps.extent = (
            float(c1.min()) + t1,
            float(c1.max()) + t1,
            float(c2.min()) + t2,
            float(c2.max()) + t2,
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_plane_name(name):
    valid = {"xz", "xy", "yz"}
    if name not in valid:
        raise ValueError(f"Plane name must be one of {valid}, got '{name}'")


# Mapping: axis -> [(plane_name, shape_index), ...]
_AXIS_SOURCES = {
    "x": [("xz", 1), ("xy", 1)],
    "y": [("xy", 2), ("yz", 1)],
    "z": [("xz", 2), ("yz", 2)],
}

# For expected_ndim=3 (transient), shape is (Nt, N1, N2) so axis indices are 1,2
# For expected_ndim=2 (static), shape is (N1, N2) so axis indices are 0,1
_AXIS_SOURCES_STATIC = {
    "x": [("xz", 0), ("xy", 0)],
    "y": [("xy", 1), ("yz", 0)],
    "z": [("xz", 1), ("yz", 1)],
}


def _infer_coords_from_planes(planes, x, y, z):
    """Infer missing coordinate arrays from plane data shapes."""
    plane_dict = {ps.name: ps.data for ps in planes}
    coords_local = {"x": x, "y": y, "z": z}

    # Determine if data is transient (3D) or static (2D)
    sample = planes[0].data if planes else None
    if sample is not None and sample.ndim == 3:
        sources = _AXIS_SOURCES
    else:
        sources = _AXIS_SOURCES_STATIC

    for cname, src_list in sources.items():
        if coords_local[cname] is None:
            for pk, ax_idx in src_list:
                if pk in plane_dict:
                    coords_local[cname] = np.arange(
                        plane_dict[pk].shape[ax_idx], dtype=float
                    )
                    break
            else:
                coords_local[cname] = np.array([0.0])

    return coords_local["x"], coords_local["y"], coords_local["z"]
