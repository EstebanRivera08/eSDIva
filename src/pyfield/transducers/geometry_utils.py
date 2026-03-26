"""
Shared geometry computation utilities for transducers.

This module provides common geometric operations such as element center
computation, subdivision generation, and mesh creation that are reused
across all transducer types.
"""

import numpy as np
from typing import List, Tuple, Optional
import pyvista as pv


def compute_1d_element_centers(
    n_elements: int,
    element_size_m: float,
    kerf_m: float,
) -> np.ndarray:
    """
    Compute element center positions for a 1D linear arrangement.

    Parameters
    ----------
    n_elements : int
        Number of elements.
    element_size_m : float
        Size of each element in meters (width or diameter).
    kerf_m : float
        Gap between elements in meters.

    Returns
    -------
    ndarray, shape (n_elements,)
        Element positions along the axis (x-coordinates for linear arrays).
    """
    pitch = element_size_m + kerf_m
    total_width = n_elements * element_size_m + (n_elements - 1) * kerf_m
    start_pos = -total_width / 2 + element_size_m / 2

    return np.array([start_pos + i * pitch for i in range(n_elements)])


def compute_2d_element_centers(
    n_elem_x: int,
    n_elem_y: int,
    element_size_x_m: float,
    element_size_y_m: float,
    kerf_x_m: float,
    kerf_y_m: float,
) -> np.ndarray:
    """
    Compute element center positions for a 2D rectangular grid arrangement.

    Parameters
    ----------
    n_elem_x : int
        Number of elements in x-direction.
    n_elem_y : int
        Number of elements in y-direction.
    element_size_x_m : float
        Element size in x-direction (meters).
    element_size_y_m : float
        Element size in y-direction (meters).
    kerf_x_m : float
        Kerf (gap) in x-direction (meters).
    kerf_y_m : float
        Kerf (gap) in y-direction (meters).

    Returns
    -------
    ndarray, shape (n_elem_x * n_elem_y, 2)
        Element centers as (x, y) coordinates.
    """
    pitch_x = element_size_x_m + kerf_x_m
    pitch_y = element_size_y_m + kerf_y_m
    
    total_w = n_elem_x * element_size_x_m + (n_elem_x - 1) * kerf_x_m
    total_h = n_elem_y * element_size_y_m + (n_elem_y - 1) * kerf_y_m
    
    start_x = -total_w / 2 + element_size_x_m / 2
    start_y = -total_h / 2 + element_size_y_m / 2

    centers_2d = []
    for iy in range(n_elem_y):
        y = start_y + iy * pitch_y
        for ix in range(n_elem_x):
            x = start_x + ix * pitch_x
            centers_2d.append([x, y])

    return np.array(centers_2d)


def build_rectangular_subdivisions(
    element_center: np.ndarray,
    element_width_m: float,
    element_height_m: float,
    no_sub_x: int,
    no_sub_y: int,
    elevation_curvature_m: Optional[float] = None,
) -> Tuple[List[np.ndarray], float]:
    """
    Build rectangular subdivision patches for a single element.

    Parameters
    ----------
    element_center : ndarray, shape (3,)
        3D center position of the element.
    element_width_m : float
        Width of the element in x-direction (meters).
    element_height_m : float
        Height of the element in y-direction (meters).
    no_sub_x : int
        Number of subdivisions in x-direction.
    no_sub_y : int
        Number of subdivisions in y-direction.
    elevation_curvature_m : float, optional
        Elevation focus radius for curved surface. If None, flat surface.

    Returns
    -------
    quads : list of ndarray
        List of quad vertices. Each quad is shape (4, 3) with vertices in order:
        [bottom-left, bottom-right, top-right, top-left].
    patch_area : float
        Area of each patch in square meters.
    """
    # Local grid edges in element coordinates
    xs = np.linspace(-element_width_m / 2, element_width_m / 2, no_sub_x + 1)
    ys = np.linspace(-element_height_m / 2, element_height_m / 2, no_sub_y + 1)

    patch_area = (element_width_m / no_sub_x) * (element_height_m / no_sub_y)
    quads = []

    for i in range(no_sub_x):
        for j in range(no_sub_y):
            # Four corners of the patch in local coordinates
            corners_local = np.array(
                [
                    [xs[i], ys[j], 0.0],
                    [xs[i + 1], ys[j], 0.0],
                    [xs[i + 1], ys[j + 1], 0.0],
                    [xs[i], ys[j + 1], 0.0],
                ],
                dtype=float,
            )

            # Translate to global coordinates
            corners = corners_local.copy()
            corners[:, 0] += element_center[0]
            corners[:, 1] += element_center[1]

            # Apply elevation curvature (curved surface in z)
            if elevation_curvature_m is not None and elevation_curvature_m > 0:
                y_vals = corners[:, 1]
                z_offset = elevation_curvature_m - np.sqrt(
                    np.clip(elevation_curvature_m**2 - y_vals**2, 0, None)
                )
                corners[:, 2] += z_offset
            else:
                corners[:, 2] += element_center[2]

            quads.append(corners)

    return quads, patch_area


def build_all_subdivisions(
    element_centers: np.ndarray,
    element_width_m: float,
    element_height_m: float,
    no_sub_x: int,
    no_sub_y: int,
    elevation_curvature_m: Optional[float] = None,
) -> Tuple[List[np.ndarray], float, List[int]]:
    """
    Build all subdivision patches for all elements.

    Parameters
    ----------
    element_centers : ndarray, shape (n_elements, 3)
        3D center positions of all elements.
    element_width_m : float
        Width of each element (meters).
    element_height_m : float
        Height of each element (meters).
    no_sub_x : int
        Number of subdivisions per element in x-direction.
    no_sub_y : int
        Number of subdivisions per element in y-direction.
    elevation_curvature_m : float, optional
        Elevation focus radius. If None, flat surfaces.

    Returns
    -------
    sub_quad_verts : list of ndarray
        List of all subdivision quad vertices.
    sub_area : float
        Area of each patch (same for all).
    sub_el_idx : list of int
        Element index for each patch (maps patch to its element).
    """
    sub_quad_verts = []
    sub_el_idx = []

    for idx, center in enumerate(element_centers):
        quads, patch_area = build_rectangular_subdivisions(
            center,
            element_width_m,
            element_height_m,
            no_sub_x,
            no_sub_y,
            elevation_curvature_m,
        )
        sub_quad_verts.extend(quads)
        sub_el_idx.extend([idx] * len(quads))

    return sub_quad_verts, patch_area, sub_el_idx


def create_mesh_from_quads(
    sub_quad_verts: List[np.ndarray],
    sub_el_idx: List[int],
    scalars_apodization: np.ndarray,
    scalars_delays: np.ndarray,
    scale_to_mm: bool = True,
) -> pv.PolyData:
    """
    Create a PyVista PolyData mesh from subdivision quads.

    Parameters
    ----------
    sub_quad_verts : list of ndarray
        List of quad vertices (each shape (4, 3)).
    sub_el_idx : list of int
        Element index for each quad.
    scalars_apodization : ndarray, shape (n_elements,)
        Apodization weights per element.
    scalars_delays : ndarray, shape (n_elements,)
        Delays per element (in seconds).
    scale_to_mm : bool, optional
        If True, scale coordinates from meters to millimeters. Default is True.

    Returns
    -------
    mesh : pyvista.PolyData
        Mesh with apodization and delays as cell data.
    """
    verts = []
    faces = []
    cell_apodization = []
    cell_delays = []

    pt_index = 0
    for quad, el_idx in zip(sub_quad_verts, sub_el_idx):
        # Add quad vertices
        verts.extend(quad.tolist())

        # Create face: [4, p0, p1, p2, p3] (VTK format)
        face = [4, pt_index, pt_index + 1, pt_index + 2, pt_index + 3]
        faces.append(face)

        # Add cell data
        cell_apodization.append(scalars_apodization[el_idx])
        cell_delays.append(scalars_delays[el_idx])

        pt_index += 4

    # Create mesh
    verts_array = np.array(verts)
    if scale_to_mm:
        verts_array *= 1e3  # Convert m to mm

    faces_flat = np.hstack(faces)
    mesh = pv.PolyData(verts_array, faces_flat)

    # Attach scalar data
    mesh.cell_data["Apodization"] = np.array(cell_apodization)
    mesh.cell_data["Delays"] = np.array(cell_delays)

    return mesh


def normalize_delays(delays: np.ndarray) -> np.ndarray:
    """
    Normalize delays so minimum is zero.

    Parameters
    ----------
    delays : ndarray
        Delay values in seconds.

    Returns
    -------
    ndarray
        Normalized delays.
    """
    return delays - np.min(delays)


def compute_distances_to_point(
    element_centers: np.ndarray,
    focus_point_m: np.ndarray,
) -> np.ndarray:
    """
    Compute distances from element centers to a focal point.

    Parameters
    ----------
    element_centers : ndarray, shape (n_elements, 3)
        3D positions of element centers in meters.
    focus_point_m : ndarray, shape (3,)
        3D focal point coordinates in meters.

    Returns
    -------
    ndarray, shape (n_elements,)
        Euclidean distances from each element to focus point.
    """
    return np.linalg.norm(element_centers - focus_point_m, axis=1)


def rotation_matrix_z_to_normal(normal: np.ndarray) -> np.ndarray:
    """
    Compute the 3×3 rotation matrix that rotates (0, 0, 1) onto ``normal``.

    Used when placing mono-element transducers at arbitrary orientations inside
    a ``CustomTransducer``.  The rotation is the minimal-angle rotation about
    the axis perpendicular to both vectors (Rodrigues' formula).

    Parameters
    ----------
    normal : ndarray, shape (3,)
        Target direction vector (need not be normalised).

    Returns
    -------
    ndarray, shape (3, 3)
        Orthogonal rotation matrix R such that R @ [0,0,1] ≈ normal/|normal|.
    """
    n = np.asarray(normal, dtype=float)
    n = n / np.linalg.norm(n)
    z = np.array([0.0, 0.0, 1.0])

    cross = np.cross(z, n)
    sin_theta = np.linalg.norm(cross)
    cos_theta = np.dot(z, n)

    if sin_theta < 1e-10:
        # Vectors are (anti-)parallel
        if cos_theta > 0:
            return np.eye(3)
        # Flip: rotate 180° about x-axis
        return np.diag([1.0, -1.0, -1.0])

    k = cross / sin_theta  # unit rotation axis
    K = np.array(
        [[0.0, -k[2], k[1]],
         [k[2],  0.0, -k[0]],
         [-k[1], k[0],  0.0]]
    )
    # Rodrigues: R = I + sin(θ)·K + (1 - cos(θ))·K²
    return np.eye(3) + sin_theta * K + (1.0 - cos_theta) * (K @ K)


def transform_patches(
    quads: List[np.ndarray],
    rotation: np.ndarray,
    translation_m: np.ndarray,
) -> List[np.ndarray]:
    """
    Apply a rigid-body transform (rotation then translation) to a list of quads.

    Parameters
    ----------
    quads : list of ndarray (4, 3)
        Patch vertices in metres.
    rotation : ndarray (3, 3)
        Rotation matrix.
    translation_m : ndarray (3,)
        Translation vector in metres.

    Returns
    -------
    list of ndarray (4, 3)
        Transformed patch vertices.
    """
    transformed = []
    for quad in quads:
        rotated = (rotation @ quad.T).T  # shape (4, 3)
        transformed.append(rotated + translation_m)
    return transformed
