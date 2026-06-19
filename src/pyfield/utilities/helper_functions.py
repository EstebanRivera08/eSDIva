"""Helper functions for spatial grids, time grids, and unit conversions."""

import time

import numpy as np
from numba import njit, prange


# -------------------- Functions for pre calculations ---------------------------
@njit(parallel=True)
def compute_minmax_distance_patch_to_point(P, M, pts, center):
    """
    Compute the global min and max distance between field points and patch centres.

    Runs in parallel over field points via Numba's ``prange``.

    Parameters
    ----------
    P : int
        Number of field points.
    M : int
        Number of patches.
    pts : float32 array (P, 3)
        Field point coordinates (metres).
    center : float32 array (M, 3)
        Patch centre coordinates (metres).

    Returns
    -------
    global_max : float
        Maximum distance across all (point, patch) pairs (metres).
    global_min : float
        Minimum distance across all (point, patch) pairs (metres).
    """
    # Compute per-point min/max in parallel to avoid race conditions on shared scalars
    local_max = np.empty(P, dtype=np.float32)
    local_min = np.empty(P, dtype=np.float32)

    for p in prange(P):  # ty: ignore[not-iterable]
        # initialize per-point values
        max_d_p = 0.0
        min_d_p = 1e30
        for m in range(M):
            dx = pts[p, 0] - center[m, 0]
            dy = pts[p, 1] - center[m, 1]
            dz = pts[p, 2] - center[m, 2]

            dist_value = np.sqrt(dx * dx + dy * dy + dz * dz)
            if dist_value > max_d_p:
                max_d_p = dist_value
            if dist_value < min_d_p:
                min_d_p = dist_value

        local_max[p] = max_d_p
        local_min[p] = min_d_p

    # Reduce serially to get global min/max
    global_max = local_max[0]
    global_min = local_min[0]
    for p in range(1, P):
        if local_max[p] > global_max:
            global_max = local_max[p]
        if local_min[p] < global_min:
            global_min = local_min[p]

    return global_max, global_min


# ------------------ Functions for initialization -------------------------------
def compute_sub_elem_attributes(transducer):
    """
    Flatten transducer patch geometry into arrays required by the SIR kernel.

    Iterates over every patch using ``sub_quad_verts`` and ``sub_el_idx`` so
    that the patch-to-element mapping is correct even when the number of
    patches per element is not uniform (e.g. circular transducers).

    For curved transducers the vertices are flat rectangles in the local
    tangent plane (built by ``subdivide_parametric_surface``), so edge lengths
    give correct arc-length dimensions without any special treatment here.

    Parameters
    ----------
    transducer : TransducerBase
        Any configured transducer.  Must have ``sub_quad_verts``,
        ``sub_el_idx``, ``apodization``, and ``delays`` populated.

    Returns
    -------
    centers_sub_elem : float32 array (M, 3)
        Centroid of each patch (metres) — equals the surface point for patches
        built by ``subdivide_parametric_surface``.
    apodization_sub_elem : float32 array (M,)
        Apodization weight of the element that owns each patch.
    delays_sub_elem : float32 array (M,)
        Transmit delay (seconds) of the element that owns each patch.
    M : int
        Total number of patches.
    range_k : None
        Reserved for future use (Δk diagnostic storage).
    wx_arr : float32 array (M,)
        Width of each patch (metres) — ``‖v[1] - v[0]‖``.
    wy_arr : float32 array (M,)
        Height of each patch (metres) — ``‖v[3] - v[0]‖``.
    sub_el_idx_arr : int32 array (M,)
        Element index for each patch — maps patch m to parent element e.
    """
    centers_sub_elem, apodization_sub_elem, delays_sub_elem = [], [], []
    wx_list, wy_list, sub_el_idx_list = [], [], []
    for verts, el_idx in zip(transducer.sub_quad_verts, transducer.sub_el_idx):
        centers_sub_elem.append(verts.mean(axis=0))
        apodization_sub_elem.append(transducer.apodization[el_idx])
        delays_sub_elem.append(transducer.delays[el_idx])
        wx_list.append(np.linalg.norm(verts[1] - verts[0]))
        wy_list.append(np.linalg.norm(verts[3] - verts[0]))
        sub_el_idx_list.append(el_idx)

    centers_sub_elem = np.array(centers_sub_elem, dtype=np.float32)
    apodization_sub_elem = np.array(apodization_sub_elem, dtype=np.float32)
    delays_sub_elem = np.array(delays_sub_elem, dtype=np.float32)
    wx_arr = np.array(wx_list, dtype=np.float32)
    wy_arr = np.array(wy_list, dtype=np.float32)
    sub_el_idx_arr = np.array(sub_el_idx_list, dtype=np.int32)
    M = len(centers_sub_elem)
    range_k = None
    return (
        centers_sub_elem,
        apodization_sub_elem,
        delays_sub_elem,
        M,
        range_k,
        wx_arr,
        wy_arr,
        sub_el_idx_arr,
    )


# ------------------ Functions to create spatial and temporal grid -----------------


# Spatial grid
def create_spatial_grid_from_dict(simulation_struct, *, fs=200e6, c=1540.0):
    """Create a simulation mesh for the ultrasound field.

    Parameters
    ----------
    simulation_struct : dict
        Dictionary containing the simulation parameters:
        - x_extent (or x_extent_mm) : list
            The extent of the simulation in the x direction (in mm).
        - y_extent (or y_extent_mm) : list
            The extent of the simulation in the y direction (in mm).
        - z_extent (or z_extent_mm) : list
            The extent of the simulation in the z direction (in mm).
        - dx (or dx_mm) : float
            The grid spacing in the x direction (in mm).
        - dy (or dy_mm) : float
            The grid spacing in the y direction (in mm).
        - dz (or dz_mm) : float
            The grid spacing in the z direction (in mm).
    fs : float, default: 200e6
        Sampling frequency in Hz. Used to compute far-field condition warnings.
    c : float, default: 1540.0
        Speed of sound in m/s. Used for patch-size validation.

    Returns
    -------
    x : ndarray
        1-D array of x coordinates in mm.
    y : ndarray
        1-D array of y coordinates in mm.
    z : ndarray
        1-D array of z coordinates in mm.
    grid_points : ndarray
        Array of shape ``(N, 3)`` with all grid points in mm.
    """
    # Create a grid of points in the simulation space
    [x0, xf], [y0, yf], [z0, zf] = (
        simulation_struct["x_extent"],
        simulation_struct["y_extent"],
        simulation_struct["z_extent"],
    )
    dx, dy, dz = (
        simulation_struct["dx"],
        simulation_struct["dy"],
        simulation_struct["dz"],
    )

    Nx = int((xf - x0) / dx) if (dx != 0 and abs(xf - x0) > 1e-10) else 1
    Ny = int((yf - y0) / dy) if (dy != 0 and abs(yf - y0) > 1e-10) else 1
    Nz = int((zf - z0) / dz) if (dz != 0 and abs(zf - z0) > 1e-10) else 1
    if Nx % 2 == 0:
        Nx += 1
    if Ny % 2 == 0:
        Ny += 1
    if Nz % 2 == 0:
        Nz += 1

    x = np.linspace(x0, xf, Nx)
    y = np.linspace(y0, yf, Ny)
    z = np.linspace(z0, zf, Nz)
    # Create a meshgrid of points
    grid_points = np.array(np.meshgrid(x, y, z)).T.reshape(-1, 3)

    return x, y, z, grid_points


def create_3D_spatial_grid_from_points(field_points_mm):
    """Extract or build a 3-D coordinate grid from user-supplied field points.

    Accepts either a dict of grid parameters (delegated to
    ``create_spatial_grid_from_dict``) or a raw ``(N, 3)`` point array from
    which unique x, y, z axes are extracted.

    Parameters
    ----------
    field_points_mm : dict or ndarray
        Grid-parameter dict or ``(N, 3)`` point array in mm.

    Returns
    -------
    x : ndarray
        Unique x coordinates in mm.
    y : ndarray
        Unique y coordinates in mm.
    z : ndarray
        Unique z coordinates in mm.
    spatial_grid : ndarray
        Point array in metres, shape ``(N, 3)``.
    """
    field_points_mm = check_valid_field_points(field_points_mm)

    if isinstance(field_points_mm, dict):
        x, y, z, spatial_grid = create_spatial_grid_from_dict(field_points_mm)
    else:
        x = np.sort(np.unique(field_points_mm[:, 0]))
        y = np.sort(np.unique(field_points_mm[:, 1]))
        z = np.sort(np.unique(field_points_mm[:, 2]))

        if len(x) * len(y) * len(z) != field_points_mm.shape[0]:
            print(
                f"Warning: unique(x)*unique(y)*unique(z) = {len(x)}x{len(y)}x{len(z)} is different from the number of points provided (points.shape[0]={field_points_mm.shape[0]}). \n"
                "If you intended to provide a grid, please check the points or set `create_meshgrid=True` to automatically recompute the grid."
            )

        spatial_grid = np.array(np.meshgrid(x, y, z)).T.reshape(-1, 3)

    return x, y, z, spatial_grid * 1e-3  # convert to meters


def check_valid_field_points(field_points_mm):
    """Validate and normalise ``field_points_mm`` to a standard form.

    Accepts a grid-parameter dict (with ``x_extent``/``dx`` keys, or their
    ``_mm``-suffixed variants) or a numeric array/list of shape ``(N, 3)``
    or ``(3,)``.  Returns the input unchanged if it is already valid, or a
    normalised dict/array otherwise.  Raises ``ValueError`` on invalid input.

    Parameters
    ----------
    field_points_mm : dict or array-like
        Field points in mm as a dict, ndarray, list, or tuple.

    Returns
    -------
    dict or ndarray
        Validated field points.
    """
    if isinstance(field_points_mm, dict):
        # Detect which key convention is used
        if "x_extent" in field_points_mm:
            suffix = ""
        elif "x_extent_mm" in field_points_mm:
            suffix = "_mm"
        else:
            raise ValueError(
                "Dict must contain 'x_extent'/'y_extent'/'z_extent'/'dx'/'dy'/'dz' "
                "or their '_mm'-suffixed equivalents."
            )

        try:
            [x0, xf], [y0, yf], [z0, zf] = (
                field_points_mm[f"x_extent{suffix}"],
                field_points_mm[f"y_extent{suffix}"],
                field_points_mm[f"z_extent{suffix}"],
            )
            dx, dy, dz = (
                field_points_mm[f"dx{suffix}"],
                field_points_mm[f"dy{suffix}"],
                field_points_mm[f"dz{suffix}"],
            )
        except Exception as e:
            raise ValueError(
                f"Could not retrieve grid parameters from dict (suffix='{suffix}'): {e}"
            ) from e

        # Normalize to standard keys (no suffix) so downstream functions always work
        if suffix == "_mm":
            field_points_mm = {
                "x_extent": [x0, xf],
                "y_extent": [y0, yf],
                "z_extent": [z0, zf],
                "dx": dx,
                "dy": dy,
                "dz": dz,
            }

    elif isinstance(field_points_mm, (np.ndarray, list, tuple)):
        if isinstance(field_points_mm, (list, tuple)):
            field_points_mm = np.array(field_points_mm, dtype=np.float32)
        # check shape
        if field_points_mm.ndim < 2:
            if field_points_mm.shape[0] == 3:
                field_points_mm = field_points_mm.reshape(1, 3)
            else:
                raise ValueError("points must 1D (3,) or 2D (N,3).")
        elif field_points_mm.ndim == 2:
            pass
        else:
            raise ValueError("points must 1D (3,) or 2D (N,3).")

    else:
        raise ValueError("field_points_mm must be a dict or a numpy/list/tuple array.")

    return field_points_mm


def reshape_to_mapped_points(x, y, z, flattened_volume):
    """Reshape the flat SIR output to ``(Nt_or_1, Nx, Ny, Nz)`` layout.

    The SIR kernel returns a 2-D array ``(Nt, P)`` where ``P = Nx*Ny*Nz``
    field points were flattened with ``meshgrid`` order ``(z, x, y)`` (i.e.
    the loop order used during grid construction).  This function reverses
    that flattening and transposes the axes to the standard ``(Nt, Nx, Ny, Nz)``
    convention expected by the rest of the library.

    Parameters
    ----------
    x : ndarray
        1-D array of unique x coordinates.
    y : ndarray
        1-D array of unique y coordinates.
    z : ndarray
        1-D array of unique z coordinates.
    flattened_volume : ndarray or list
        Flat SIR output, shape ``(Nt, P)`` or ``(P,)``.

    Returns
    -------
    ndarray
        Reshaped array of shape ``(Nt, Nx, Ny, Nz)``.
    """
    if isinstance(flattened_volume, (list, tuple)):
        flattened_volume = np.array(flattened_volume)
    elif isinstance(flattened_volume, np.ndarray):
        pass
    else:
        raise ValueError("input must be a list or numpy array")

    if flattened_volume.ndim == 1:
        flattened_volume = flattened_volume.reshape(1, -1)

    return flattened_volume.reshape(
        flattened_volume.shape[0], len(z), len(x), len(y)
    ).transpose(0, 2, 3, 1)


def compute_time_grid(P, M, points, centers, wx, wy, c, fs, delays, verbose=True):
    """
    Compute the time axis needed to capture the full SIR response.

    The earliest possible arrival is ``(min_distance - 0.5*(wx+wy)) / c``
    and the latest is ``(max_distance + (wx+wy)) / c + max_delay``.  The
    returned grid spans this range at sampling interval ``1/fs``.

    Parameters
    ----------
    P, M : int
        Number of field points and patches.
    points : float32 (P, 3)
        Field point coordinates (metres).
    centers : float32 (M, 3)
        Patch centre coordinates (metres).
    wx, wy : float
        Patch dimensions (metres) — used to bound the SIR kernel width.
    c : float
        Speed of sound (m/s).
    fs : float
        Sampling frequency (Hz).
    delays : float32 (n_elements,)
        Per-element transmit delays (seconds).
    verbose : bool, optional
        Print diagnostic messages. Default True.

    Returns
    -------
    t_grid : ndarray
        Time samples in seconds, shape ``(T,)``.
    min_time : float
        Start of the time window (seconds).
    dt : float
        Sampling interval ``1/fs`` (seconds).
    T : int
        Number of time samples.
    """
    start = time.time()

    max_dist, min_dist = compute_minmax_distance_patch_to_point(P, M, points, centers)
    if verbose:
        print(
            f"Min distance: {min_dist * 1e3:.2f} mm, Max distance: {max_dist * 1e3:.2f} mm"
        )

    max_delay = delays.max()
    size_patch = wx + wy
    # Compute min and max time
    # t1 = min_l/c - 0.5*(max_Dt1 + max_Dt2) + min_delay
    # t4 = t1 + Dt1 + Dt2 = min_/c + 0.5*(max_Dt1 + max_Dt2) + max_delay
    # Dt1 and Dt2 max are wx/c and wy/c respectively
    # So:
    min_time = (min_dist - 0.5 * size_patch) / c  # us (or unit)
    max_time = (max_dist + size_patch) / c + max_delay  # us (or unit)

    dt = 1.0 / fs
    T = int(np.ceil((max_time - min_time) * fs))

    # Memory estimate for h_sir (P × T × 4 bytes float32).
    h_sir_gb = P * T * 4 / 1e9
    if h_sir_gb >= 2.0:
        print(
            f"\nWARNING: grid Nx×Ny×Nz = {P:,} points and T={T}— "
            f"estimated h_sir ~= {h_sir_gb:.1f} GB "
            "This will likely cause a memory error.\n"
            "  -> Reduce dx/dy/dz or, shrink the extent, or compute a 2-D plane. "
        )
    elif h_sir_gb >= 0.5:
        print(
            f"INFO: grid Nx×Ny×Nz = {P:,} points and T = {T} — "
            f"estimated h_sir ~= {h_sir_gb * 1e3:.0f} MB "
            "Consider a coarser grid if memory is limited.\n"
        )

    t_grid = min_time + np.arange(T, dtype=np.float32) * dt
    if verbose:
        print(
            f"Computed time grid from {min_time * 1e6:.2f} us to {max_time * 1e6:.2f} us, with {T} samples in {time.time() - start:.2f} seconds."
        )
    return t_grid, min_time, dt, T


def to_dB(matrix, *, vmin=None, vmax=None):
    """Convert a matrix to decibel (dB) scale.

    Parameters
    ----------
    matrix : array-like
        Input data (magnitude or complex).
    vmin : float, optional
        Floor value for the dB conversion (linear scale).
    vmax : float, optional
        Reference maximum for normalisation. Defaults to ``abs(matrix).max()``.

    Returns
    -------
    ndarray
        Values in decibels, normalised so that the peak is 0 dB.
    """

    mat = np.asarray(matrix, dtype=float)
    mag = np.abs(mat)

    # handle empty input
    if mag.size == 0:
        return np.array([])

    # normalize safely
    if vmax is None:
        vmax = mag.max()
        if vmax == 0:
            vmax = 1.0
    mag = mag / vmax

    if vmin is None:
        vmin = 1e-20  # default minimum magnitude to avoid log(0)
    # avoid log(0) without in-place assignment

    return 20 * np.log10(mag + vmin)


def align_to_common_time(fields_and_coords, *, align_to_shorter=False):
    """Interpolate multiple transient fields to a common time grid.

    When simulating separate planes, each call to ``PyField`` may produce a
    different ``t0`` and number of time samples ``Nt``.  This function
    reconstructs the individual time vectors, computes a shared time axis,
    and interpolates every field onto it.

    Parameters
    ----------
    fields_and_coords : list of (pressure, coords) tuples
        Each element is a ``(pressure, coords)`` pair as returned by
        ``PyField.__call__`` in transient mode.  ``coords`` must contain
        ``"t0"`` (float) and ``"dt"`` (float).
    align_to_shorter : bool, optional
        If ``False`` (default), the common time grid spans the full range
        from the earliest start to the latest end across all fields.
        Regions where a field has no data are zero-padded.

        If ``True``, the common time grid is restricted to the overlapping
        interval (latest start to earliest end).  This preserves the old
        behavior and raises ``ValueError`` if there is no overlap.

    Returns
    -------
    common_time : ndarray
        Shared time vector in seconds.
    aligned_fields : list of ndarray
        Each field interpolated along axis 0 (time) to *common_time*.

    Raises
    ------
    ValueError
        If any ``coords`` dict is missing ``"t0"`` or ``"dt"``, or if
        ``align_to_shorter=True`` and there is no overlapping time interval.

    Examples
    --------
    >>> pxz, cxz = sim(plane_xz, monochromatic=False)
    >>> pyz, cyz = sim(plane_yz, monochromatic=False)
    >>> common_t, [pxz_a, pyz_a] = align_to_common_time(
    ...     [(pxz, cxz), (pyz, cyz)]
    ... )
    """
    from scipy.interpolate import interp1d

    if not fields_and_coords:
        raise ValueError("fields_and_coords must be a non-empty list.")

    # Reconstruct individual time vectors
    time_vectors = []
    for p, c in fields_and_coords:
        if "t0" not in c or "dt" not in c:
            raise ValueError(
                "Each coords dict must contain 't0' and 'dt' keys "
                "(transient mode output)."
            )
        nt = p.shape[0]
        t = c["t0"] + np.arange(nt) * c["dt"]
        time_vectors.append(t)

    if align_to_shorter:
        # Old behavior: overlapping interval only
        t_start = max(t[0] for t in time_vectors)
        t_end = min(t[-1] for t in time_vectors)
        if t_start >= t_end:
            raise ValueError(
                f"No overlapping time interval: max(t_start)={t_start:.3e}, "
                f"min(t_end)={t_end:.3e}."
            )
    else:
        # New default: full range, zero-padded
        t_start = min(t[0] for t in time_vectors)
        t_end = max(t[-1] for t in time_vectors)

    # Use finest resolution
    dt_common = min(c["dt"] for _, c in fields_and_coords)
    common_time = np.arange(t_start, t_end, dt_common)

    # Interpolate each field along time axis (axis 0)
    aligned_fields = []
    for (p, _), t in zip(fields_and_coords, time_vectors):
        # Flatten spatial dims for interpolation, then reshape back
        orig_shape = p.shape  # (Nt, ...)
        spatial_shape = orig_shape[1:]
        p_flat = p.reshape(orig_shape[0], -1)  # (Nt, N_spatial)

        interp_fn = interp1d(
            t,
            p_flat,
            axis=0,
            kind="linear",
            bounds_error=False,
            fill_value=0.0,
        )
        p_interp = interp_fn(common_time)  # (Nt_common, N_spatial)
        aligned_fields.append(p_interp.reshape(len(common_time), *spatial_shape))

    return common_time, aligned_fields
