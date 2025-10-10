import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from .plotting_pyvista import add_pressure_vol, create_vol_mesh


def plot_pressure_field(
    x,
    y,
    z,
    pressure_field,
    *,
    scalars="Pressure (u.a.)",
    plotter=None,
    off_screen=False,
    window_size=[520, 720],
    notebook=False,
    return_mesh=False,
    plot_focal_spot=False,
    **kwargs,
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
    pressure_vol = create_vol_mesh(x, y, z, pressure_field, scalars=scalars)

    # Create a PyVista plotter
    plotter = add_pressure_vol(
        pressure_vol,
        plotter=plotter,
        window_size=window_size,
        notebook=notebook,
        plot_focal_spot=plot_focal_spot,
        off_screen=off_screen,
        title=None,
        **kwargs,
    )

    plotter.show_grid()  # show grid
    if return_mesh:
        return plotter, pressure_vol
    return plotter


def plot_field_planes(
    x,
    y,
    z,
    pressure_field,
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
