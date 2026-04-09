"""Coordinate transformation utilities for probe alignment."""

import numpy as np


def get_LabToTransducer(TX_mesh, Doppler2D):
    """Compute the lab-to-transducer transformation matrix.

    Parameters
    ----------
    TX_mesh : pyvista.PolyData
        The transducer mesh.
    Doppler2D : DopplerScan
        The DopplerScan object containing the probe mesh.

    Returns
    -------
    ndarray
        4x4 homogeneous transformation matrix (lab to transducer).
    """
    # Invert the
    invertz = np.diag([1, 1, -1, 1])  # Invert
    # Set the TX mesh origin

    Transducer_center = np.array(TX_mesh.center)

    # Get the Doppler2D center in the probe coordinate system

    pv_mesh = Doppler2D.transform(
        T_matrix=np.linalg.inv(Doppler2D.probeToLab) @ invertz, inplace=False
    )
    Doppler2D_inProbeSpace_center = np.array(pv_mesh.center)

    # print("Transducer center (m): ", Transducer_center)
    # print("2D Doppler center (m): ", Doppler2D_inProbeSpace_center)

    # Translation just along x and y axis
    Trans_vector_ProbeToTransducer = (
        Transducer_center[:2] - Doppler2D_inProbeSpace_center[:2]
    )
    # print(
    #     "Translation vector from `Probe` to `Transducer` (m): ",
    #     Trans_vector_ProbeToTransducer,
    # )

    set_TX_origin = np.eye(4)  # Create a 4x4 identity matrix for translation
    set_TX_origin[:2, 3] = Trans_vector_ProbeToTransducer

    # rescale units from m to mm
    rescale_mToMm = np.diag(
        [1000, 1000, 1000, 1]
    )  # Scale factors for x, y, z, and homogeneous coordinate

    LabToProbe = (
        invertz
        @ rescale_mToMm
        @ set_TX_origin
        @ np.linalg.inv(Doppler2D.probeToLab)
        @ invertz
    )  # Invert the probe to lab transformation matrix
    return LabToProbe


def compute_affine_from_markers(
    p1,
    p2,
    source_origin=np.zeros(3),
    source_normal=np.array([0, 1, 0]),
    up_axis=np.array([0, 0, -1]),
):
    """Compute a rigid-body transform from two marker points.

    Parameters
    ----------
    p1 : ndarray, shape (3,)
        First marker point (becomes the mapped origin).
    p2 : ndarray, shape (3,)
        Second marker point (defines target plane with ``up_axis``).
    source_origin : ndarray, shape (3,), optional
        Origin in the source frame. Default ``[0, 0, 0]``.
    source_normal : ndarray, shape (3,), optional
        Normal in the source frame. Default ``[0, 1, 0]``.
    up_axis : ndarray, shape (3,), optional
        Up direction used to define the target plane. Default ``[0, 0, -1]``.

    Returns
    -------
    t : ndarray, shape (3,)
        Translation vector.
    R : ndarray, shape (3, 3)
        Rotation matrix.
    """
    # 1) target normal n_t
    d = p2 - p1
    n_t = np.cross(d, up_axis)
    n_t /= np.linalg.norm(n_t)

    # 2) rotation axis & angle
    n_s = source_normal / np.linalg.norm(source_normal)
    axis = np.cross(n_s, n_t)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-8:
        # Normals already aligned or opposite:
        if np.dot(n_s, n_t) > 0:
            R = np.eye(3)
        else:
            # 180° rotation about any axis orthogonal to n_s
            # e.g. pick x-axis if n_s not collinear:
            v = np.array([1, 0, 0])
            if abs(np.dot(v, n_s)) > 0.9:
                v = np.array([0, 1, 0])
            axis = np.cross(n_s, v)
            axis /= np.linalg.norm(axis)
            theta = np.pi
            # build Rodrigues for 180°:
            K = np.array(
                [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
            )
            R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
        t = p1 + d / 2
    else:
        axis /= axis_norm
        theta = np.arccos(np.clip(np.dot(n_s, n_t), -1, 1))
        # Rodrigues' formula
        K = np.array(
            [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
        )
        R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
        # translation
        # New center
        t = p1 + d / 2

    t[2] = 0  # No translation in z-axis
    t = t
    return t, R
