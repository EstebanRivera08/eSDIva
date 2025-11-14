import time

import numpy as np
from numba import njit, prange


# -------------------- Functions for pre calculations ---------------------------
@njit(parallel=True)
def compute_minmax_distance_patch_to_point(P, M, pts, center):
    # Compute per-point min/max in parallel to avoid race conditions on shared scalars
    local_max = np.empty(P, dtype=np.float32)
    local_min = np.empty(P, dtype=np.float32)

    for p in prange(P):
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
    centers_sub_elem, apodization_sub_elem, delays_sub_elem = [], [], []
    for elem in range(transducer.n_elements):
        for sub_elem in range(transducer.no_sub_x * transducer.no_sub_y):
            verts = transducer.sub_quad_verts[
                elem * (transducer.no_sub_x * transducer.no_sub_y) + sub_elem
            ]
            centers_sub_elem.append(verts.mean(axis=0))
            apodization_sub_elem.append(transducer.apodization[elem])
            delays_sub_elem.append(transducer.delays[elem])

    centers_sub_elem = np.array(centers_sub_elem, dtype=np.float32)  # mm
    apodization_sub_elem = np.array(apodization_sub_elem, dtype=np.float32)
    delays_sub_elem = np.array(delays_sub_elem, dtype=np.float32)
    M = len(centers_sub_elem)
    range_k = None
    return centers_sub_elem, apodization_sub_elem, delays_sub_elem, M, range_k


# ------------------ Functions to create spatial and temporal grid -----------------


# Spatial grid
def create_spatial_grid_from_dict(simulation_struct):
    """
    Create a simulation mesh for the ultrasound field.

    Parameters
    ----------
    simulation_grid_dict : dict
        Dictionary containing the simulation parameters:
        - x_extent : list
            The extent of the simulation in the x direction (in mm).
        - y_extent : list
            The extent of the simulation in the y direction (in mm).
        - z_extent : list
            The extent of the simulation in the z direction (in mm).
        - dx : float
            The grid spacing in the x direction (in mm).
        - dy : float
            The grid spacing in the y direction (in mm).
        - dz : float
            The grid spacing in the z direction (in mm).

    Returns
    -------
    grid_points : ndarray
        Array of points in the simulation space.
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

    if z0 <= 0.1:
        z0 = 0.1  # avoid z=0 plane

    Nx = int((xf - x0) / dx) if (dx != 0 and abs(xf - x0) > 1e-10) else 1
    Ny = int((yf - y0) / dy) if (dy != 0 and abs(yf - y0) > 1e-10) else 1
    Nz = int((zf - z0) / dz) if (dz != 0 and abs(zf - z0) > 1e-10) else 1
    if Nx % 2 == 0:
        Nx += 1
    if Ny % 2 == 0:
        Ny += 1
    if Nz % 2 == 0:
        Nz += 1

    # print(
    #     f"Creating grid with {Nx} x {Ny} x {Nz} points in x, y, z directions respectively."
    # )
    # print(f"Grid extents: x: [{x0}, {xf}], y: [{y0}, {yf}], z: [{z0}, {zf}]")
    x = np.linspace(x0, xf, Nx)
    y = np.linspace(y0, yf, Ny)
    z = np.linspace(z0, zf, Nz)
    # Create a meshgrid of points
    grid_points = np.array(np.meshgrid(x, y, z)).T.reshape(-1, 3)

    return x, y, z, grid_points


def check_field_points(field_points_mm):
    if isinstance(field_points_mm, dict):
        x, y, z, spatial_grid = create_spatial_grid_from_dict(field_points_mm)

    if isinstance(field_points_mm, (np.ndarray, list, tuple)):
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

        # Check
        x = np.sort(np.unique(field_points_mm[:, 0]))
        y = np.sort(np.unique(field_points_mm[:, 1]))
        z = np.sort(np.unique(field_points_mm[:, 2]))
        spatial_grid = np.array(np.meshgrid(x, y, z)).T.reshape(-1, 3)
    return x, y, z, spatial_grid * 1e-3  # convert to meters


# Reshape flattened volume to mapped points
def reshape_to_mapped_points(x, y, z, flattened_volume):
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


def compute_time_grid(P, M, points, centers, wx, wy, c, fs, delays):
    start = time.time()

    max_dist, min_dist = compute_minmax_distance_patch_to_point(P, M, points, centers)
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
    min_time = max(min_time, 0.0)

    max_time = (max_dist + 0.5 * size_patch) / c + max_delay  # us (or unit)

    dt = 1.0 / fs
    T = int(np.ceil((max_time - min_time) * fs))
    # next power of two
    t_grid = min_time + np.arange(T, dtype=np.float32) * dt
    print(
        f"Computed time grid from {min_time * 1e6:.2f} us to {max_time * 1e6:.2f} us, with {T} samples in {time.time() - start:.2f} seconds."
    )
    return t_grid, min_time, dt, T


def to_dB(matrix):
    """Convert a matrix to decibel (dB) scale."""

    mat = np.asarray(matrix, dtype=float)
    mag = np.abs(mat)

    # handle empty input
    if mag.size == 0:
        return np.array([])

    # normalize safely
    maxv = mag.max()
    if maxv == 0:
        maxv = 1.0
    mag = mag / maxv

    # avoid log(0) without in-place assignment
    mag = np.where(mag == 0, 1e-20, mag)

    return 20 * np.log10(mag)
