import numpy as np
import pyvista as pv

def compute_pressure_vol_mesh(pressure_field, x, y, z) :
    """
    Compute the pressure volume mesh for the given pressure field and coordinates.
    
    Parameters
    ----------
    pressure_field : ndarray
        Pressure field data.
    x, y, z : ndarray
        Coordinate arrays.
    
    Returns
    -------
    pressure_vol : pyvista.UniformGrid
        The pressure volume mesh.
    """
    # If x, y, z are 1D (common case)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    dz = z[1] - z[0]

    nx, ny, nz = pressure_field.shape

    # Create the 3D UniformGrid
    pressure_vol = pv.ImageData(dimensions=(nx, ny, nz),
                        spacing=(dx, dy, dz),
                        origin=(x.min(), y.min(), z.min()))

    # Attach pressure data to the grid
    pressure_vol.point_data["Pressure"] = pressure_field.ravel(order="F")  # VERY important: Fortran order
    return pressure_vol


def align_transducer_to_probe(TX_mesh, Doppler2D):
    """
    Align the transducer mesh to the probe mesh.
    Args:
        TX_mesh (pv.PolyData): The transducer mesh.
        Doppler2D (DopplerScan): The DopplerScan object containing the probe mesh.
    Returns:
        pv.PolyData: The aligned transducer mesh.
    """
    # Invert the
    invertz = np.diag([1, 1, -1, 1])  # Invert
    # Set the TX mesh origin
    x_min, x_max, y_min, y_max, z_min, z_max = TX_mesh.bounds
    TX_origin = np.array([x_min, y_min])
    Probe_origin = Doppler2D.voxelsToProbe[:2, 3]   # Set the origin to the mesh bounds

    set_TX_origin = np.eye(4)  # Create a 4x4 identity matrix for translation
    set_TX_origin[:2, 3] =  Probe_origin - TX_origin*1e-3  # Compute the translation vector to the transducer origin

    # rescale units from m to mm
    rescale_mToMm = np.diag([1000, 1000, 1000, 1])  # Scale factors for x, y, z, and homogeneous coordinate

    LabToProbe =  invertz @ rescale_mToMm @ set_TX_origin @ np.linalg.inv(Doppler2D.probeToLab) @ invertz  # Invert the probe to lab transformation matrix
    return LabToProbe

def compute_affine_from_markers(p1, p2, 
                               source_origin=np.zeros(3), 
                               source_normal=np.array([0,1,0]), 
                               up_axis=np.array([0,0,-1])):
    """
    Returns a 4×4 rigid‐body transform T that maps:
      - source_origin → p1
      - source_normal     → target plane normal
    where the target plane is defined by points p1, p2, and the up_axis.
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
            v = np.array([1,0,0])
            if abs(np.dot(v, n_s)) > 0.9:
                v = np.array([0,1,0])
            axis = np.cross(n_s, v)
            axis /= np.linalg.norm(axis)
            theta = np.pi
            # build Rodrigues for 180°:
            K = np.array([[    0, -axis[2],  axis[1]],
                          [ axis[2],     0, -axis[0]],
                          [-axis[1], axis[0],     0]])
            R = np.eye(3) + np.sin(theta)*K + (1-np.cos(theta))*(K@K)
        t = p1 + d/2
    else:
        axis /= axis_norm
        theta = np.arccos(np.clip(np.dot(n_s, n_t), -1, 1))
        # Rodrigues' formula
        K = np.array([[    0, -axis[2],  axis[1]],
                      [ axis[2],     0, -axis[0]],
                      [-axis[1], axis[0],     0]])
        R = np.eye(3) + np.sin(theta)*K + (1-np.cos(theta))*(K@K)
        # translation
        # New center
        t = p1 + d/2

    t[2] = 0 # No translation in z-axis
    t = t 
    return t, R
