import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from matplotlib.gridspec import GridSpec

from .processing import compute_pressure_vol_mesh


def plot_pressure_field(
    pressure_field,
    x,
    y,
    z,
    *,
    plotter=None,
    off_screen=False,
    window_size=[520, 720],
    notebook=False,
    return_mesh=False,
):
    """
    Plot the pressure field in 3D.

    Parameters
    ----------
    pressure_field : ndarray
        Pressure field data.
    x, y, z : ndarray
        Coordinate arrays.
    """
    # Create the pressure volume mesh
    pressure_vol = compute_pressure_vol_mesh(pressure_field, x, y, z)

    # Create a PyVista plotter
    if plotter is None:
        plotter = pv.Plotter(
            window_size=window_size, notebook=notebook, off_screen=off_screen
        )  # ,off_screen=True) # Need to add this parameter to save the screenshot

    n_contours = 10
    min_val = 0
    max_val = pressure_field.max()
    levels = np.linspace(min_val, max_val, n_contours)
    iso_mesh = pressure_vol.contour(
        isosurfaces=levels, scalars="Pressure"
    )  # Create isosurface at threshold
    plotter.add_mesh(
        iso_mesh,
        scalars="Pressure",  # use the scalar to color surfaces
        cmap="jet",  # color map
        opacity="linear",  # solid surfaces
        show_scalar_bar=True,
        scalar_bar_args={
            "title": "Pressure",
            "vertical": True,
            "title_font_size": 16,
            "label_font_size": 12,
            "position_x": 0.85,
            "position_y": 0.1,
            "height": 0.3,
        },
        label="Pressure PII",
        color="r",  # color of the mesh
    )

    plotter.add_axes()  # show XYZ axes
    plotter.show_grid()  # show grid
    if return_mesh:
        return plotter, pressure_vol
    return plotter


def plot_field_planes(
    pressure_field,
    x,
    y,
    z,
    *,
    figsize=(10, 5),
    interpolation=None,
    centered_to_max=False,
    save_fig_name=None,
    ratios=None,
    vmin=None,
    vmax=None,
    label="Pressure (a.u.)",
):
    """
    Plot the pressure field in 2D slices with a properly placed colorbar.

    Parameters
    ----------
    pressure_field : ndarray
        Pressure field data.
    x, y, z : ndarray
        Coordinate arrays.
    """
    if centered_to_max:
        # Look for the y, x, z indices that are closest to the max value
        max_idx = np.unravel_index(np.nanargmax(pressure_field), pressure_field.shape)
        y0, x0, z0 = max_idx[1], max_idx[0], max_idx[2]
    else:
        # Use the middle indices
        y0 = int(np.floor(y.shape[0] / 2))
        x0 = int(np.floor(x.shape[0] / 2))
        z0 = int(np.floor(z.shape[0] / 2))
    print(
        f"Taking slice ({x[x0]},{y[y0]},{z[z0]}) => x_ind, y_ind, z_ind = {x0 + 1}/{x.shape[0]}, {y0 + 1}/{y.shape[0]}, {z0 + 1}/{z.shape[0]}"
    )

    # Use nanmin and nanmax to ignore NaN values
    if vmin is None:
        vmin = np.nanmin(pressure_field)
    if vmax is None:
        vmax = np.nanmax(pressure_field)

    XZ_plane = pressure_field[:, y0, :].squeeze()
    XY_plane = pressure_field[:, :, z0].squeeze()
    YZ_plane = pressure_field[x0, :, :].squeeze()

    Dx, Dy, Dz = x.max() - x.min(), y.max() - y.min(), z.max() - z.min()

    if ratios is not None:
        # check if ratios has length 3
        if len(ratios) != 3:
            raise ValueError("Ratios must have length 3.")
        else:
            ratios = ratios / np.sum(ratios)
    else:
        ratios = [Dx / Dz, Dx / Dy, Dy / Dz]
        ratios = ratios / np.sum(ratios)

    # Create a GridSpec layout
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(
        1, 4, width_ratios=[ratios[0], ratios[1], ratios[2], 0.05 * ratios.max()]
    )  # Last column for the colorbar

    ax0 = fig.add_subplot(gs[0, 0])
    im0 = ax0.imshow(
        XZ_plane.T,
        cmap="jet",
        extent=[x.min(), x.max(), z.max(), z.min()],
        vmin=vmin,
        vmax=vmax,
        interpolation=interpolation,
    )
    ax0.set_xlabel("X (mm)")
    ax0.set_ylabel("Z (mm)")
    ax0.set_title("XZ Plane (Y={:.2f} mm)".format(y[y0]))

    ax1 = fig.add_subplot(gs[0, 1])
    im1 = ax1.imshow(
        XY_plane.T,
        cmap="jet",
        extent=[x.min(), x.max(), y.max(), y.min()],
        vmin=vmin,
        vmax=vmax,
        interpolation=interpolation,
    )
    ax1.set_xlabel("X (mm)")
    ax1.set_ylabel("Y (mm)")
    ax1.set_title("XY Plane (Z={:.2f} mm)".format(z[z0]))

    ax2 = fig.add_subplot(gs[0, 2])
    im2 = ax2.imshow(
        YZ_plane.T,
        cmap="jet",
        extent=[y.min(), y.max(), z.max(), z.min()],
        vmin=vmin,
        vmax=vmax,
        interpolation=interpolation,
    )
    ax2.set_xlabel("Y (mm)")
    ax2.set_ylabel("Z (mm)")
    ax2.set_title("YZ Plane (X={:.2f} mm)".format(x[x0]))

    # Add a colorbar to the last column
    cbar_ax = fig.add_subplot(gs[0, 3])
    cbar = fig.colorbar(im2, cax=cbar_ax)
    cbar.set_label(label)
    cbar.ax.yaxis.set_label_position("left")

    plt.tight_layout()
    if save_fig_name:
        plt.savefig(save_fig_name, dpi=300)
    plt.show()
    plt.close(fig)  # Close the figure to free memory


def add_transducer_to_plotter(TX_mesh, plotter, **kwargs):
    # Add the transducer to the plotter# 2) Add your TX_mesh with Apodization
    plotter.add_mesh(
        TX_mesh,
        scalars="Apodization",  # use the attached scalar
        cmap="cool",  # color map (you can change to "plasma", "coolwarm", etc.)
        show_scalar_bar=True,
        scalar_bar_args={
            "title": "Pressure (u.a.)",
            "title_font_size": 16,
            "label_font_size": 12,
            "vertical": True,
            "position_x": 0.85,
            "position_y": 0.2,
            "height": 0.3,
        },
        label="Transducer",  # label for the legend
        color="purple",  # color of the mesh
    )
    return plotter


def add_pressure_to_plotter(plotter, pressure_vol, plot_focal_spot=True):
    # 3) Add the pressure volume
    if plot_focal_spot:
        # pick a threshold, e.g. halfway to the max
        threshold = 0.7 * pressure_vol["Pressure"].max()
        iso_mesh = pressure_vol.contour(
            [threshold], scalars="Pressure"
        )  # Create isosurface at threshold# add that instead of (or in addition to) the volume
        plotter.add_mesh(
            iso_mesh,
            opacity=1.0,
            name="PressureIso",
            show_scalar_bar=False,
            label="Focal Spot",
            color="r",  # color of the mesh
        )
    else:
        n_contours = 10
        min_val = 0
        max_val = pressure_vol["Pressure"].max()
        levels = np.linspace(min_val, max_val, n_contours)
        iso_mesh = pressure_vol.contour(
            isosurfaces=levels, scalars="Pressure"
        )  # Create isosurface at threshold
        plotter.add_mesh(
            iso_mesh,
            scalars="Pressure",  # use the scalar to color surfaces
            cmap="jet",  # color map
            opacity="linear",  # solid surfaces
            show_scalar_bar=True,
            scalar_bar_args={
                "title": "Pressure",
                "vertical": True,
                "title_font_size": 16,
                "label_font_size": 12,
                "position_x": 0.9,
                "position_y": 0.2,
                "height": 0.3,
            },
            label="Pressure PII",
            color="r",  # color of the mesh
        )
    return plotter
