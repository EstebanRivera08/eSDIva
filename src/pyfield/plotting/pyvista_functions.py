"""Helper functions for PyVista mesh creation and styling."""

import numpy as np
import pyvista as pv


def _recompute_bounds(plotter):
    """Recompute the bounds of a PyVista plotter based on its current meshes.

    Parameters
    ----------
    plotter : pyvista.Plotter
        The plotter whose bounds should be recomputed.

    Returns
    -------
    tuple of float
        ``(x_min, x_max, y_min, y_max, z_min, z_max)``.
    """
    if not plotter.meshes:
        raise ValueError("The plotter has no meshes to compute bounds from.")

    all_bounds = np.array([mesh.bounds for mesh in plotter.meshes])
    x_min = all_bounds[:, 0].min()
    x_max = all_bounds[:, 1].max()
    y_min = all_bounds[:, 2].min()
    y_max = all_bounds[:, 3].max()
    z_min = all_bounds[:, 4].min()
    z_max = all_bounds[:, 5].max()

    return (x_min, x_max, y_min, y_max, z_min, z_max)


def _normalize_window_size(window_size, scale=1.0):
    """
    Return ``window_size`` scaled and rounded to a valid ``(width, height)`` tuple.

    PyVista requires integer pixel dimensions >= 1. Applying a ``scale`` factor
    (> 1 for high-res screenshots, < 1 for thumbnails) before rounding keeps the
    aspect ratio exact.
    """
    ws = np.array(window_size, dtype=float) * float(scale)
    if ws.ndim == 0:
        ws = np.array([ws, ws])
    # round and ensure at least 1 pixel
    ws = np.maximum(np.round(ws).astype(int), 1)
    return (int(ws[0]), int(ws[1]))


def _set_custom_style(plotter, *, scale=1.0):
    """
    Apply a consistent axis-label and gridline style to a PyVista plotter.

    Draws labelled outer bounding-box axes (X/Y/Z in mm) with white gridlines,
    and sets the camera "up" direction to ``(0, 0, -1)`` so that z points down
    (matching the acoustic convention where depth increases along z).
    """
    cube_actor = plotter.show_bounds(
        grid="back",
        color="black",
        font_size=int(12 * scale),
        location="outer",
        xtitle="X (mm)",
        ytitle="Y (mm)",
        ztitle="Z (mm)",
    )
    # turn on outer gridlines and inner gridlines if you want them
    cube_actor.DrawXGridlinesOn()
    cube_actor.DrawYGridlinesOn()
    cube_actor.DrawZGridlinesOn()

    # set gridline colors to white (RGB in 0..1)
    cube_actor.GetXAxesGridlinesProperty().SetColor(1.0, 1.0, 1.0)
    cube_actor.GetYAxesGridlinesProperty().SetColor(1.0, 1.0, 1.0)
    cube_actor.GetZAxesGridlinesProperty().SetColor(1.0, 1.0, 1.0)

    # (optional) tweak gridline width
    cube_actor.GetXAxesGridlinesProperty().SetLineWidth(2.0)
    cube_actor.GetYAxesGridlinesProperty().SetLineWidth(2.0)
    cube_actor.GetZAxesGridlinesProperty().SetLineWidth(2.0)

    plotter.camera.up = (0, 0, -1)


# -------------------- Plotting Functions --------------------
def create_3Dvol_mesh(vol_matrix, x=None, y=None, z=None, *, scalars="Values"):
    """Create a PyVista volume mesh from coordinate arrays and data.

    Parameters
    ----------
    vol_matrix : ndarray
        Volume data (dim = 3).
    x, y, z : ndarray or None
        1D coordinate arrays for the x, y, and z axes (in mm). If None, pixel
        indices are used with a default spacing of 1 mm.
    scalars : str, optional
        Name for the scalar data array. Default ``'Values'``.

    Returns
    -------
    pyvista.ImageData
        The volume mesh with attached scalar data.
    """

    if vol_matrix.ndim == 3:
        nx, ny, nz = vol_matrix.shape
    elif vol_matrix.ndim == 4:
        nt, nx, ny, nz = vol_matrix.shape

        if nt != 1:
            raise ValueError(
                f"Expected vol_matrix to have shape (1, nx, ny, nz) or (nx, ny, nz), but got {vol_matrix.shape}"
            )
        else:
            vol_matrix = vol_matrix[0]  # Remove the time dimension
    else:
        raise ValueError(
            f"Expected vol_matrix to have 3 dimensions (1, nx, ny, nz) or (nx, ny, nz), but got {vol_matrix.ndim}"
        )

    if x is None or y is None or z is None:
        # Create default coordinate arrays based on the shape of vol_matrix
        dx, dy, dz = 1.0, 1.0, 1.0
        xmin, ymin, zmin = 0.0, 0.0, 0.0
    else:
        dx = x[1] - x[0] if len(x) > 1 else 1e-6
        dy = y[1] - y[0] if len(y) > 1 else 1e-6
        dz = z[1] - z[0] if len(z) > 1 else 1e-6
        xmin, ymin, zmin = x.min(), y.min(), z.min()

    # Create the 3D UniformGrid
    pressure_vol = pv.ImageData(
        dimensions=(nx, ny, nz),
        spacing=(dx, dy, dz),
        origin=(xmin, ymin, zmin),
    )

    # Attach pressure data to the grid
    pressure_vol.point_data[scalars] = vol_matrix.ravel(
        order="F"
    )  # VERY important: Fortran order
    return pressure_vol


def create_2Dimage_mesh(matrix, *, extent=None, plane_offset=None, scalars="a.u."):
    """Create a PyVista structured grid from a 2D image matrix.

    Parameters
    ----------
    matrix : (H, W) numpy.ndarray
        2D array representing the image data.
    extent : tuple of float, optional
        Spatial extent ``(x_min, x_max, y_min, y_max)`` in mm. If None,
        pixel indices are used.
    plane_offset : dict, optional
        Single-key dict specifying the out-of-plane offset, e.g.
        ``{"y": 10.0}``. Must have exactly one key (``"x"``, ``"y"``, or
        ``"z"``). Defaults to ``{"y": 0}``.
    scalars : str, optional
        Name for the scalar data array. Default ``"a.u."``.

    Returns
    -------
    pyvista.StructuredGrid
        Structured grid representing the 2D image in world coordinates.
    """
    H, W = matrix.shape

    # Create grid indices
    if extent is not None:
        y_min, y_max, x_min, x_max = extent
        # switch x and y because the image is in row-column format but we want to plot
        # in x-y format
        x = np.linspace(x_min, x_max, W)
        y = np.linspace(y_min, y_max, H)
    else:
        x = np.arange(W)
        y = np.arange(H)

    i_grid, j_grid = np.meshgrid(x, y)  # Shape (H, W)

    if plane_offset is None:
        plane_offset = {"y": 0}
    else:
        # must be dict with plane keys, e.g. {"x": 10} to offset x-plane by 10 mm
        if not isinstance(plane_offset, dict):
            raise ValueError(
                "plane_offset must be a dict with plane keys, e.g. {'x': 10}"
            )
        else:
            # must have exactly one plane key
            if len(plane_offset) != 1:
                raise ValueError(
                    "plane_offset must have exactly one plane key, e.g. {'x': 10}"
                )

    plane, offset_mm = next(iter(plane_offset.items()))  # Get the plane and offset

    # Construct homogeneous voxel coordinates based in plane
    if plane == "x":
        # points_voxel shape (H, W, 4) with (x, y, z, 1) coordinates in homogeneous form
        # Construct homogeneous voxel coordinates [j, 0, i, 1]
        points_voxel = np.stack(
            [np.zeros_like(j_grid) + offset_mm, j_grid, i_grid, np.ones_like(j_grid)],
            axis=-1,
        )
    elif plane == "y":
        points_voxel = np.stack(
            [j_grid, np.zeros_like(j_grid) + offset_mm, i_grid, np.ones_like(j_grid)],
            axis=-1,
        )
    elif plane == "z":
        points_voxel = np.stack(
            [j_grid, i_grid, np.zeros_like(j_grid) + offset_mm, np.ones_like(j_grid)],
            axis=-1,
        )
    else:
        raise ValueError(f"Invalid plane {plane}, must be 'x', 'y' or 'z'")
    points_voxel_flat = points_voxel.reshape(-1, 4).T  # Shape (4, H*W)

    # Create structured grid
    xx = points_voxel_flat[0].reshape(H, W)
    yy = points_voxel_flat[1].reshape(H, W)
    zz = points_voxel_flat[2].reshape(H, W)
    grid = pv.StructuredGrid(xx, yy, zz)
    grid[scalars] = matrix.ravel(order="F").astype(
        np.float32
    )  # Add scalar matrix # FORTRAN ORDER IMPORTANT
    return grid


def load_mesh_from_stl(
    file_path,
    *,
    scale=1.0,
    translation=(0.0, 0.0, 0.0),
    rotation_axis=None,
    rotation_angle=0.0,
):
    """Load an STL file and return a PyVista mesh with optional transformations.

    Parameters
    ----------
    file_path : str or Path
        Path to the STL file.
    scale : float, optional
        Uniform scale factor. Default 1.0 (no scaling).
    translation : tuple of float, optional
        Translation vector ``(dx, dy, dz)``. Default ``(0, 0, 0)``.
    rotation_axis : tuple of float, optional
        Rotation axis ``(x, y, z)``. If None, no rotation is applied.
    rotation_angle : float, optional
        Rotation angle in degrees. Default 0.0.

    Returns
    -------
    pyvista.PolyData
        Loaded mesh with transformations applied.
    """
    from pathlib import Path

    import pyvista as pv

    # Load the STL file
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"STL file not found: {file_path}")

    mesh = pv.read(str(file_path))

    # Apply transformations
    if scale != 1.0:
        mesh = mesh.scale(scale)

    if rotation_axis is not None and rotation_angle != 0.0:
        mesh = mesh.rotate_vector(
            vector=rotation_axis, angle=rotation_angle, point=mesh.center
        )

    if translation != (0.0, 0.0, 0.0):
        mesh = mesh.translate(translation)

    return mesh
