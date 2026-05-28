"""Validation helpers for pressure field plotting inputs."""


def _check_match(value1, value2, message=None):
    """Raise ValueError if two values differ."""
    if message is None:
        message = f"Values do not match: {value1} != {value2}"
    if value1 != value2:
        raise ValueError(message)


def check_planes_shape(planes):
    """Validate that three orthogonal planes have consistent shapes.

    Parameters
    ----------
    planes : list of numpy.ndarray
        Three 2D arrays ``[plane_xz, plane_xy, plane_yz]``.

    Returns
    -------
    list of int
        Consistent dimensions ``[nx, ny, nz]``.
    """
    plane_xz, plane_xy, plane_yz = planes
    nx1, nz1 = plane_xz.shape
    nx2, ny2 = plane_xy.shape
    ny3, nz3 = plane_yz.shape

    # Check x length
    message = f"Plane xz and xy must have the same x length, but got {nx1} and {nx2}"
    _check_match(nx1, nx2, message=message)
    # Check y length
    message = f"Plane xy and yz must have the same y length, but got {ny2} and {ny3}"
    _check_match(ny2, ny3, message=message)
    # Check z length
    message = f"Plane xz and yz must have the same z length, but got {nz1} and {nz3}"
    _check_match(nz1, nz3, message=message)

    return [nx1, ny2, nz1]


def check_coords(coords, shape):
    """Validate that coordinate arrays match the expected field shape.

    Parameters
    ----------
    coords : dict
        Coordinate dict with keys ``"x"``, ``"y"``, ``"z"``.
    shape : tuple of int
        Expected ``(nx, ny, nz)`` dimensions.
    """
    nx, ny, nz = shape
    if not isinstance(coords, dict):
        raise ValueError(
            f"Coords must be a dict with keys 'x', 'y', 'z', but got {type(coords)}"
        )
    x, y, z = coords["x"], coords["y"], coords["z"]
    _check_match(len(x), nx, message=f"x coord length {len(x)} != field nx {nx}")
    _check_match(len(y), ny, message=f"y coord length {len(y)} != field ny {ny}")
    _check_match(len(z), nz, message=f"z coord length {len(z)} != field nz {nz}")
