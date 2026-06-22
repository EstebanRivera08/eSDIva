"""
Shared geometry computation utilities for transducers.

This module provides common geometric operations such as element center
computation, subdivision generation, and mesh creation that are reused
across all transducer types.
"""

import numpy as np
from typing import List
import pyvista as pv


def windowed_apodization_1d(
    n_elements: int,
    pitch_m: float,
    x_foc_m: float,
    z_foc_m: float,
    f_over_d: float,
    apodization_type: str,
) -> np.ndarray:
    """Windowed sub-aperture apodization for a 1-D array (linear / convex).

    The active aperture spans the F-number footprint ``D = |z_foc| / (F/D)``; the
    window (``rect`` / ``hanning`` / ``hamming``) is sized to the nearest number of
    elements covering ``D`` (parity matched to ``n_elements`` so an odd array keeps a
    centre element) and slid so its centre lands on the element nearest the lateral
    focus ``x_foc``.

    Centring is parity-symmetric: ``offset = (N - N_virt)//2 + round(x_foc/pitch)``
    puts the window middle on the array middle for both even and odd ``N`` (an on-axis
    focus therefore yields a strictly symmetric profile — the physical requirement).

    Parameters
    ----------
    n_elements : int
        Number of array elements ``N``.
    pitch_m : float
        Element-to-element spacing in metres.
    x_foc_m, z_foc_m : float
        Lateral and axial focus coordinates in metres.
    f_over_d : float
        F-number (focal length / aperture). Sets the active-aperture width.
    apodization_type : {"none", "rect", "hanning", "hamming"}
        Window shape. ``"none"`` returns uniform full-aperture weights.

    Returns
    -------
    (n_elements,) numpy.ndarray
        Per-element apodization weights (zero outside the active sub-aperture).
    """
    N = int(n_elements)
    if apodization_type == "none":
        return np.ones(N, dtype=float)

    # Active-aperture width D → element count, parity-matched to N (keeps a centre
    # element for odd arrays), clamped to the physical aperture.
    D = abs(z_foc_m) / f_over_d
    n_virt = int(round((D / (N * pitch_m)) * N / 2) * 2 + (N % 2))
    n_virt = max(1, min(n_virt, N))

    if apodization_type == "rect":
        wins = np.ones(n_virt)
    elif apodization_type == "hanning":
        wins = np.hanning(n_virt)
    else:  # hamming
        wins = np.hamming(n_virt)

    # Slide the window so its centre sits on the element nearest x_foc.
    offset = (N - n_virt) // 2 + int(np.round(x_foc_m / pitch_m))
    idxs = np.arange(n_virt) + offset
    valid = (idxs >= 0) & (idxs < N)
    apod = np.zeros(N, dtype=float)
    apod[idxs[valid]] = wins[valid]
    return apod


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
    pyvista.PolyData
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
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
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
