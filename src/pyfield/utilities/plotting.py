import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from matplotlib.gridspec import GridSpec

from .plotting_pyvista import add_pressure_vol, create_vol_mesh


def _set_custom_style(plotter, *, scale=1.0):
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


def plot_pressure_field(
    x,
    y,
    z,
    pressure_field,
    *,
    scalars="Pressure",
    plotter=None,
    off_screen=False,
    window_size=[520, 720],
    notebook=False,
    return_mesh=False,
    plot_focal_spot=False,
    scale=1.0,
    anti_aliasing="ssaa",
    colorbar_title="Pressure",
    box_color="#b0b0b0",
    box_opacity=0.2,
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
    pv.global_theme.anti_aliasing = anti_aliasing
    # Create the pressure volume mesh
    pressure_vol = create_vol_mesh(x, y, z, pressure_field, scalars=scalars)
    box = pressure_vol.bounding_box()

    # Create a PyVista plotter
    plotter = add_pressure_vol(
        pressure_vol,
        plotter=plotter,
        window_size=np.array(window_size) * scale,
        notebook=notebook,
        plot_focal_spot=plot_focal_spot,
        off_screen=off_screen,
        scale=scale,
        colorbar_title=colorbar_title,
        **kwargs,
    )

    _ = plotter.add_mesh(box, opacity=box_opacity, color=box_color)

    _set_custom_style(plotter, scale=scale)
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
    title=None,
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

    if title is not None:
        fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    if save_fig_name:
        plt.savefig(save_fig_name, dpi=300)
    plt.show()
    plt.close(fig)  # Close the figure to free memory


def plot_deltak_distribution(
    pyfield,
    *,
    figsize=(11, 4),
    per_element=True,
    cmap="turbo",
    hist_color="#AB0000E8",
    xlim=None,
    ylim=None,
):
    sub_elem_delta_k = pyfield.sub_elem_delta_k
    transducer = pyfield.tx

    M = pyfield.M
    T = pyfield.T_log[-1]
    P = pyfield.P_log[-1]

    condition = 8 + 2 * T / M
    mean_k = np.mean(sub_elem_delta_k)
    print(f"last simulation characteristics: M = {M}, P = {P}, T = {T}")
    print(f"mean_delta_k : {mean_k}, and 8+2T/M : {condition}")

    # If transducer is given we'll mean the values patch per element
    # and show results per element
    xlabel = "Patch index"
    if per_element is False:
        range_k = sub_elem_delta_k
    else:
        n_elements = transducer.n_elements
        no_sub_x = transducer.no_sub_x
        no_sub_y = transducer.no_sub_y
        xlabel = "Element index"

        range_k = np.zeros((P, n_elements), dtype=np.float32)
        for i in range(n_elements):
            range_k[:, i] = sub_elem_delta_k[
                :, i * no_sub_x * no_sub_y : (i + 1) * no_sub_x * no_sub_y
            ].mean(axis=1)

    # Create figure and choose 2D or 3D axes for the left subplot depending on transducer
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(1, 2, width_ratios=[1, 0.6])
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])

    im = ax0.imshow(range_k, aspect="auto", cmap=cmap)
    # ax0.set_title("a)")
    ax0.set_xlabel(xlabel)
    ax0.set_ylabel("Point index")
    if xlim is not None:
        ax0.set_xlim(xlim)
    if ylim is not None:
        ax0.set_ylim(ylim)
    fig.colorbar(im, ax=ax0, label=r"$\Delta k_{m,p}$ ")

    # plot Histogram of the krange
    ax1.hist(
        sub_elem_delta_k.flatten(),
        bins=10,
        color=hist_color,
        edgecolor="black",
        alpha=0.85,
    )
    ax1.set_xlabel(r"$\Delta k$")
    ax1.set_ylabel("Count")
    # ax1.set_title(r"b)")
    ax1.grid(axis="y", color="gray", linestyle="--", alpha=0.6)
    ax1.spines["right"].set_visible(False)
    ax1.spines["top"].set_visible(False)
    ax1.set_facecolor("#FFFFFFC5")
    plt.tight_layout()

    return fig
