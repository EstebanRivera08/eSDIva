import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

from pyfield.plotting import plot2D_pressure_plane


def plot_volume_slices(
    planes,
    *,
    coords=None,
    center=None,
    width_ratios=[1, 1, 1, 0.05],
    figsize=(15, 5),
    **kwargs,
):

    plane_xz, plane_xy, plane_yz = (
        planes["plane_xz"],
        planes["plane_xy"],
        planes["plane_yz"],
    )
    nx1, nz1 = plane_xz.shape
    nx2, ny2 = plane_xy.shape
    ny3, nz3 = plane_yz.shape

    # lenght between planes with shared extents must be the same
    # Check x length
    if not (nx1 == nx2):
        raise ValueError(
            f"Plane xz and xy must have the same x length, but got {nx1} and {nx2}"
        )
    # Check y length
    if not (ny2 == ny3):
        raise ValueError(
            f"Plane xy and yz must have the same y length, but got {ny2} and {ny3}"
        )
    # Check z length
    if not (nz1 == nz3):
        raise ValueError(
            f"Plane xz and yz must have the same z length, but got {nz1} and {nz3}"
        )

    if coords is not None:
        if isinstance(coords, dict):
            x, y, z = coords["x"], coords["y"], coords["z"]
            # check that the coords have the same length as the planes
            if not (len(x) == nx1):
                raise ValueError(
                    f"Coords x must have the same length as plane xz, but got {len(x)} and {nx1}"
                )
            if not (len(y) == ny2):
                raise ValueError(
                    f"Coords y must have the same length as plane xy, but got {len(y)} and {ny2}"
                )
            if not (len(z) == nz1):
                raise ValueError(
                    f"Coords z must have the same length as plane xz, but got {len(z)} and {nz1}"
                )
        else:
            raise ValueError(
                f"Coords must be a dict with keys 'x', 'y', 'z', but got {type(coords)}"
            )
    else:
        x = np.arange(nx1)
        y = np.arange(ny2)
        z = np.arange(nz1)

    if center is None:
        # supose the center is the middle of the grid
        nx, ny, nz = len(x), len(y), len(z)
        center = (x[nx // 2], y[ny // 2], z[nz // 2])

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, 4, width_ratios=width_ratios, wspace=0.3)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0 = plot2D_pressure_plane(
        x,
        z,
        plane_xz,
        title="XZ Plane",
        plane_axis="y",
        ax=ax0,
        **kwargs,
    )
    ax1 = fig.add_subplot(gs[0, 1])
    ax1 = plot2D_pressure_plane(
        x,
        y,
        plane_xy,
        title="XY Plane",
        plane_axis="z",
        ax=ax1,
        **kwargs,
    )
    ax2 = fig.add_subplot(gs[0, 2])
    ax2 = plot2D_pressure_plane(
        y,
        z,
        plane_yz,
        title="YZ Plane",
        plane_axis="x",
        ax=ax2,
        **kwargs,
    )

    cbar_ax = fig.add_subplot(gs[0, 3])
    cbar = fig.colorbar(ax2._image, cax=cbar_ax)
    cbar.ax.yaxis.set_label_position("left")

    plt.show()


def create_2D_image_mesh(matrix, *, extent=None, plane_offset=None, scalars="a.u."):
    """
    Compute a PyVista structured grid from a 2D image matrix.
    Arguments:
        matrix (np.ndarray):
            A 2D array representing and image.
        extent (tuple, optional):
            A tuple of (x_min, x_max, y_min, y_max) defining
            the spatial extent of the image. If None, the extent will be determined by
            the shape of the matrix.


    Returns:
        pv.StructuredGrid: A structured grid representing the 2D image in world coordinates.
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
        plane_offset = {"plane": "y", "offset_mm": 0}

    plane = plane_offset["plane"]
    offset_mm = plane_offset["offset_mm"]
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
