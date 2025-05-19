import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pyvista as pv

def plot_pressure_field(pressure_field, x, y, z, *,    
        off_screen=None,
        notebook=None,):
    """
    Plot the pressure field in 3D.
    
    Parameters
    ----------
    pressure_field : ndarray
        Pressure field data.
    x, y, z : ndarray
        Coordinate arrays.
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
    plotter = pv.Plotter(notebook=notebook, off_screen=off_screen)# ,off_screen=True) # Need to add this parameter to save the screenshot

    n_contours = 10
    min_val = 0
    max_val = pressure_field.max()
    levels = np.linspace(min_val, max_val, n_contours)
    iso_mesh = pressure_vol.contour(isosurfaces=levels, scalars="Pressure")  # Create isosurface at threshold
    plotter.add_mesh(
        iso_mesh,
        scalars="Pressure",  # use the scalar to color surfaces
        cmap="jet",                   # color map
        opacity='linear',                  # solid surfaces
        show_scalar_bar=True,
        scalar_bar_args={"title": "Pressure",
            "vertical": True,
            "title_font_size": 16,
            "label_font_size": 12,
            "position_x": 0.9,
            "position_y": 0.2,
            "height": 0.3,
        },
        label="Pressure PII",
        color = "r", # color of the mesh
    )

    plotter.add_axes()              # show XYZ axes
    plotter.show_grid()             # show grid
    plotter.show()



def plot_field_planes(pressure_field, x, y, z, *
                      ,figsize =(10, 5),  interpolation=None):
    """
    Plot the pressure field in 2D slices with a properly placed colorbar.
    
    Parameters
    ----------
    pressure_field : ndarray
        Pressure field data.
    x, y, z : ndarray
        Coordinate arrays.
    """ 
    y0 = int(np.floor(y.shape[0] / 2))
    x0 = int(np.floor(x.shape[0] / 2))
    z0 = int(np.floor(z.shape[0] / 2))
    print(f"Taking slice x_ind, y_ind, z_ind = {x0+1}/{x.shape[0]}, {y0+1}/{y.shape[0]}, {z0+1}/{z.shape[0]}")

    vmin = pressure_field.min()
    vmax = pressure_field.max()

    XZ_plane = pressure_field[:, y0, :].squeeze()
    XY_plane = pressure_field[:, :, z0].squeeze()
    YZ_plane = pressure_field[x0, :, :].squeeze()

    Dx, Dy, Dz = x.max()-x.min(), y.max()-y.min(), z.max()-z.min()
    ratios = [Dx/Dz, Dx/Dy, Dy/Dz]
    ratios = ratios / np.sum(ratios)

    # Create a GridSpec layout
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(1, 4, width_ratios=[ratios[0], ratios[1], ratios[2], 0.05*ratios.max()])  # Last column for the colorbar

    ax0 = fig.add_subplot(gs[0, 0])
    im0 = ax0.imshow(XZ_plane.T, cmap='jet', extent=[x.min(), x.max(), z.min(), z.max()], vmin=vmin, vmax=vmax, interpolation=interpolation)
    ax0.set_xlabel("X (mm)")
    ax0.set_ylabel("Z (mm)")
    ax0.set_title("XZ Plane")

    ax1 = fig.add_subplot(gs[0, 1])
    im1 = ax1.imshow(XY_plane.T, cmap='jet', extent=[x.min(), x.max(), y.min(), y.max()], vmin=vmin, vmax=vmax, interpolation=interpolation)
    ax1.set_xlabel("X (mm)")
    ax1.set_ylabel("Y (mm)")
    ax1.set_title("XY Plane")

    ax2 = fig.add_subplot(gs[0, 2])
    im2 = ax2.imshow(YZ_plane.T, cmap='jet', extent=[y.min(), y.max(), z.min(), z.max()], vmin=vmin, vmax=vmax, interpolation=interpolation)
    ax2.set_xlabel("Y (mm)")
    ax2.set_ylabel("Z (mm)")
    ax2.set_title("YZ Plane")

    # Add a colorbar to the last column
    cbar_ax = fig.add_subplot(gs[0, 3])
    cbar = fig.colorbar(im2, cax=cbar_ax)
    cbar.set_label("Pressure (normalized)")

    plt.tight_layout()
    plt.show()