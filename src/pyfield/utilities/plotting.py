import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from matplotlib.gridspec import GridSpec

from .helper_functions import to_dB
from .plotting_pyvista import add_pressure_vol, create_vol_mesh


def _normalize_window_size(window_size, scale=1.0):
    """
    Ensure window_size is a tuple of two positive integers suitable for pv.Plotter.
    """
    ws = np.array(window_size, dtype=float) * float(scale)
    if ws.ndim == 0:
        ws = np.array([ws, ws])
    # round and ensure at least 1 pixel
    ws = np.maximum(np.round(ws).astype(int), 1)
    return (int(ws[0]), int(ws[1]))


def _set_custom_style(plotter, *, scale=1.0):
    cube_actor = plotter.show_bounds(
        grid="back",
        color="black",
        font_size=int(18 * scale),
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
    colorbar_title=None,
    box_color="#b0b0b0",
    box_opacity=0.2,
    contour_levels=11,
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
        window_size=_normalize_window_size(window_size, scale=scale),
        notebook=notebook,
        plot_focal_spot=plot_focal_spot,
        off_screen=off_screen,
        scale=scale,
        colorbar_title=colorbar_title,
        contour_levels=contour_levels,
        **kwargs,
    )

    _ = plotter.add_mesh(box, opacity=box_opacity, color=box_color)

    _set_custom_style(plotter, scale=scale)
    if return_mesh:
        return plotter, pressure_vol
    return plotter


def plot_pressure_2D(
    x,
    z,
    pressure_plane,
    *,
    figsize=None,
    title=None,
    plane_axis="y",
    ax=None,
    **kwargs,
):
    """
    Plot a 2D pressure plane.
    Parameters
    ----------
    pressure_plane : ndarray
        2D pressure plane data.
    x, z (or y) : ndarray
        Coordinate arrays for the plane.
    plane_axis : str, optional
        Axis along which the plane is taken ("x", "y", or "z"). Default is "y".
    """
    cmap = kwargs.pop("cmap", "jet")

    if figsize is None:
        width = 6
        height = width * (z.max() - z.min()) / (x.max() - x.min())
        figsize = (width, height)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    elif not isinstance(ax, plt.Axes):
        raise ValueError("ax must be a matplotlib Axes object.")

    extent = [x.min(), x.max(), z.max(), z.min()]

    if plane_axis == "y":
        xlabel = "X (mm)"
        ylabel = "Z (mm)"
    elif plane_axis == "x":
        xlabel = "Y (mm)"
        ylabel = "Z (mm)"
    elif plane_axis == "z":
        xlabel = "X (mm)"
        ylabel = "Y (mm)"
    else:
        raise ValueError("plane_axis must be 'x', 'y', or 'z'.")

    im = ax.imshow(
        pressure_plane.T,
        extent=extent,
        origin="upper",
        cmap=cmap,
        **kwargs,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax._image = im
    if title is not None:
        ax.set_title(title)

    return ax


def plot_pressure_planes(
    x,
    y,
    z,
    pressure_field,
    *,
    db_scale=False,
    figsize=None,
    title=None,
    centered_to_max=False,
    save_fig_name=None,
    save_dir=None,
    ratios=None,
    label=None,
    fps=30,
    video_duration=5,
    p_max=None,
    **kwargs,
):
    """
    Plot the pressure field in 2D slices

    Parameters
    ----------
    x, y, z : ndarray
        Coordinate arrays.

    pressure_field : ndarray
        Pressure field datas.
    """
    import pathlib

    # Setup save directory if specified
    if save_dir is not None:
        save_path = pathlib.Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        print(f"Output will be saved to: {save_path.resolve()}")
    else:
        save_path = None

    if pressure_field.ndim == 4:
        is_transient = True
        nt, nx, ny, nz = pressure_field.shape
    elif pressure_field.ndim == 3:
        is_transient = False
        nx, ny, nz = pressure_field.shape
    else:
        raise ValueError("pressure_field must be either 3D or 4D array.")

    if nx == 1:
        plane_axis = "x"
        x1 = y
        x2 = z
    elif ny == 1:
        plane_axis = "y"
        x1 = x
        x2 = z
    elif nz == 1:
        plane_axis = "z"
        x1 = x
        x2 = y
    else:
        plane_axis = None

        # Convert to dB if requested
    if db_scale:
        pressure_field = to_dB(pressure_field, vmax=p_max)
        cb_label = label if label else "Pressure (dB)"
        vmin = kwargs.pop("vmin", -40)
        vmax = kwargs.pop("vmax", 0)
    else:
        cb_label = label if label else "Pressure (a.u.)"
        vmin = kwargs.pop("vmin", np.nanmin(pressure_field))
        vmax = kwargs.pop("vmax", np.nanmax(pressure_field))

    if plane_axis is not None:
        print("Plotting 2D slice along {}-axis".format(plane_axis))
        fig, ax = plt.subplots(figsize=figsize)

        ax = plot_pressure_2D(
            x1,
            x2,
            pressure_field.squeeze(),
            ax=ax,
            figsize=figsize,
            title=title,
            plane_axis=plane_axis,
            vmin=vmin,
            vmax=vmax,
            **kwargs,
        )
        cbar = fig.colorbar(ax._image, ax=ax)
        cbar.set_label(cb_label)

    else:
        if centered_to_max:
            # Look for the y, x, z indices that are closest to the max value
            max_idx = np.unravel_index(
                np.nanargmax(pressure_field), pressure_field.shape
            )
            y0, x0, z0 = max_idx[1], max_idx[0], max_idx[2]
        else:
            # Use the middle indices
            y0 = int(np.floor(y.shape[0] / 2))
            x0 = int(np.floor(x.shape[0] / 2))
            z0 = int(np.floor(z.shape[0] / 2))
        print(
            f"Taking slice ({x[x0]:.1e},{y[y0]:.1e},{z[z0]:.1e}) mm => x_ind, y_ind, z_ind = {x0 + 1}/{x.shape[0]}, {y0 + 1}/{y.shape[0]}, {z0 + 1}/{z.shape[0]}"
        )
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

        if not is_transient:
            ax0 = fig.add_subplot(gs[0, 0])
            ax0 = plot_pressure_2D(
                x,
                z,
                pressure_field[:, y0, :].squeeze(),
                figsize=figsize,
                title="XZ Plane (Y={:.2f} mm)".format(y[y0]),
                plane_axis="y",
                ax=ax0,
                vmin=vmin,
                vmax=vmax,
                **kwargs,
            )
            ax1 = fig.add_subplot(gs[0, 1])
            ax1 = plot_pressure_2D(
                x,
                y,
                pressure_field[:, :, z0].squeeze(),
                figsize=figsize,
                title="XY Plane (Z={:.2f} mm)".format(z[z0]),
                plane_axis="z",
                ax=ax1,
                vmin=vmin,
                vmax=vmax,
                **kwargs,
            )
            ax2 = fig.add_subplot(gs[0, 2])
            ax2 = plot_pressure_2D(
                y,
                z,
                pressure_field[x0, :, :].squeeze(),
                figsize=figsize,
                title="YZ Plane (X={:.2f} mm)".format(x[x0]),
                plane_axis="x",
                ax=ax2,
                vmin=vmin,
                vmax=vmax,
                **kwargs,
            )

            cbar_ax = fig.add_subplot(gs[0, 3])
            cbar = fig.colorbar(ax2._image, cax=cbar_ax)
            cbar.set_label(label)
            cbar.ax.yaxis.set_label_position("left")

    if title is not None:
        fig.suptitle(title)
    plt.tight_layout()
    if save_fig_name is not None:
        image_path = save_path / save_fig_name if save_path else save_fig_name
        plt.savefig(image_path, dpi=300)
    plt.show()


def plot_slices_2d(
    x,
    y,
    z,
    pressure_field,
    *,
    time_array=None,
    db_scale=True,
    figsize=(10, 5),
    save_dir=None,
    save_format="png",
    save_fps=10,
    vmin=None,
    vmax=None,
    centered_to_max=False,
    cmap="jet",
    title=None,
    label=None,
    interpolation=None,
    **kwargs,
):
    """
    Plot 2D slices of pressure field for both 3D (monochromatic) and 4D (transient) data.

    This function automatically detects whether the input is monochromatic (3D) or
    transient (4D) data and handles visualization accordingly:
    - **3D data**: Creates a single static figure with three orthogonal slices (XZ, XY, YZ)
    - **4D data**: Creates frame-by-frame visualization, optionally saving as image sequence or video

    Parameters
    ----------
    x : ndarray
        X coordinate array (mm).
    y : ndarray
        Y coordinate array (mm).
    z : ndarray
        Z coordinate array (mm).
    pressure_field : ndarray
        Pressure field data. Can be:
        - 3D array (Nx, Ny, Nz): Monochromatic field
        - 4D array (Nx, Ny, Nz, Nt): Transient field with time steps
    time_array : ndarray, optional
        Time values for each time step (only used for 4D data).
        If None, time step indices are used. Shape: (Nt,)
    db_scale : bool, optional
        If True, convert pressure values to dB scale. Default is True.
    figsize : tuple, optional
        Figure size (width, height) in inches. Default is (10, 5).
    save_dir : str or Path, optional
        Directory to save output. If None, no files are saved.
        - For 3D data: Saves single PNG/PDF file
        - For 4D data: Saves frame sequence or video (if ffmpeg available)
        Default is None.
    save_format : str, optional
        Output format for 3D plots or frame images. Options: "png", "pdf", "jpg", "svg".
        Default is "png".
    save_fps : int, optional
        Frames per second for video output (4D data). Default is 10.
    vmin : float, optional
        Minimum value for colorbar. If None, computed from data. Default is None.
    vmax : float, optional
        Maximum value for colorbar. If None, computed from data. Default is None.
    centered_to_max : bool, optional
        If True, center slice positions on the maximum pressure value.
        If False, use the geometric center of the field. Default is False.
    cmap : str, optional
        Colormap name. Default is "inferno".
    title : str, optional
        Title for the plot. If None, auto-generated. Default is None.
    label : str, optional
        Colorbar label. If None, automatically set based on db_scale parameter.
        Default is None (auto-generated).
    interpolation : str, optional
        Interpolation method for imshow. Options: None, "nearest", "bilinear", etc.
        Default is None.
    **kwargs :
        Additional keyword arguments (reserved for future extensions).

    Returns
    -------
    None
        Displays plot(s) and optionally saves to disk.

    Notes
    -----
    - For 4D transient data, displays frames sequentially with a brief pause between them.
    - Time values are converted to microseconds (µs) for display.
    - Uses `to_dB()` function for dB conversion if db_scale=True.

    Examples
    --------
    **Monochromatic (3D) visualization:**

    >>> x, y, z, p_mono = simulator(plane_config)
    >>> plot_slices_2d(x, y, z, p_mono, db_scale=True, save_dir="./results")

    **Transient (4D) visualization with video:**

    >>> x, y, z, p_transient = simulator(plane_config, excitation=excitation)
    >>> plot_slices_2d(
    ...     x, y, z, p_transient,
    ...     time_array=time_array,
    ...     db_scale=True,
    ...     save_dir="./results",
    ...     save_fps=15,
    ... )

    **Custom visualization:**

    >>> plot_slices_2d(
    ...     x, y, z, p_field,
    ...     figsize=(12, 6),
    ...     cmap="jet",
    ...     centered_to_max=True,
    ...     vmin=-60,
    ...     vmax=0,
    ... )
    """
    import pathlib

    # Set default label based on db_scale if not provided
    if label is None:
        label = "Pressure (dB)" if db_scale else "Pressure (a.u.)"

    # Auto-detect monochromatic (3D) vs transient (4D)
    is_transient = pressure_field.ndim == 4
    num_time_steps = pressure_field.shape[0] if is_transient else 1

    if is_transient and time_array is None:
        time_array = np.arange(num_time_steps)

    # Setup save directory if specified
    if save_dir is not None:
        save_path = pathlib.Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        print(f"Output will be saved to: {save_path.resolve()}")
    else:
        save_path = None

    # ========================================================================
    # HANDLE MONOCHROMATIC (3D) CASE
    # ========================================================================
    if not is_transient:
        # Check if is a plane
        nx, ny, nz = pressure_field.shape
        if nx == 1:
            plane_axis = "x"
        elif ny == 1:
            plane_axis = "y"
        elif nz == 1:
            plane_axis = "z"
        else:
            plane_axis = None
        pressure_field = pressure_field.squeeze()

        print(f"Field shape: {pressure_field.shape}")

        # Convert to dB if requested
        if db_scale:
            pressure_plot = to_dB(pressure_field)
            cb_label = label if label else "Pressure (dB)"
        else:
            pressure_plot = pressure_field
            cb_label = label if label else "Pressure (a.u.)"

        # Use existing plot_field_planes function for consistent styling
        auto_title = title if title else "Monochromatic Pressure Field"

        if plane_axis is not None:
            print(
                f"Note: Detected plane along {plane_axis}-axis. Plotting 2D slice instead of 3D planes."
            )

        if save_path:
            save_file = save_path / f"pressure_field.{save_format}"
            plot_field_planes(
                x,
                y,
                z,
                pressure_plot,
                figsize=figsize,
                title=auto_title,
                interpolation=interpolation,
                centered_to_max=centered_to_max,
                vmin=vmin,
                vmax=vmax,
                label=cb_label,
                save_fig_name=str(save_file),
            )
        else:
            plot_field_planes(
                x,
                y,
                z,
                pressure_plot,
                figsize=figsize,
                title=auto_title,
                interpolation=interpolation,
                centered_to_max=centered_to_max,
                vmin=vmin,
                vmax=vmax,
                label=cb_label,
            )

        return

    # ========================================================================
    # HANDLE TRANSIENT (4D) CASE
    # ========================================================================
    print(f"\n Plotting 4D (transient) pressure field")
    print(f"  Field shape: {pressure_field.shape}")
    print(f"  Time steps: {num_time_steps}")
    print(f"  Close figure window to complete visualization")

    # Normalize time array if provided (assume it's in seconds, convert to µs)
    if time_array is not None and time_array[0] < 1e-3:
        time_display = time_array * 1e6  # Convert seconds to microseconds
        time_unit = "µs"
    else:
        time_display = time_array
        time_unit = "s" if time_array is not None else ""

    # Compute global vmin/vmax across all time steps if not provided
    if vmin is None or vmax is None:
        if db_scale:
            all_data_db = to_dB(pressure_field.reshape(-1, num_time_steps))
            if vmin is None:
                vmin = np.nanmin(all_data_db)
            if vmax is None:
                vmax = np.nanmax(all_data_db)
        else:
            if vmin is None:
                vmin = np.nanmin(pressure_field)
            if vmax is None:
                vmax = np.nanmax(pressure_field)

    fig = plt.figure(figsize=figsize)

    for time_idx in range(num_time_steps):
        fig.clear()

        # Extract pressure field at this time step (squeeze to 3D)
        pressure_at_t = pressure_field[:, :, :, time_idx].squeeze()

        # Convert to dB if requested
        if db_scale:
            pressure_plot = to_dB(pressure_at_t)
            cb_label = label if label else "Pressure (dB)"
        else:
            pressure_plot = pressure_at_t
            cb_label = label if label else "Pressure (a.u.)"

        # Determine slice positions
        if centered_to_max:
            max_idx = np.unravel_index(np.nanargmax(pressure_plot), pressure_plot.shape)
            x_slice_idx = max_idx[0]
            y_slice_idx = max_idx[1]
            z_slice_idx = max_idx[2]
        else:
            x_slice_idx = int(np.floor(len(x) / 2))
            y_slice_idx = int(np.floor(len(y) / 2))
            z_slice_idx = int(np.floor(len(z) / 2))

        # Extract the three orthogonal slices
        xz_plane = pressure_plot[:, y_slice_idx, :].T
        xy_plane = pressure_plot[:, :, z_slice_idx].T
        yz_plane = pressure_plot[x_slice_idx, :, :].T

        # Create gridspec layout with proper aspect ratios
        Dx = x.max() - x.min()
        Dy = y.max() - y.min()
        Dz = z.max() - z.min()
        ratios = [Dx / Dz, Dx / Dy, Dy / Dz]
        ratios = np.array(ratios) / np.sum(ratios)

        gs = GridSpec(1, 4, width_ratios=[ratios[0], ratios[1], ratios[2], 0.05])

        # Plot XZ plane
        ax0 = fig.add_subplot(gs[0, 0])
        im0 = ax0.imshow(
            xz_plane,
            cmap=cmap,
            extent=[x.min(), x.max(), z.max(), z.min()],
            vmin=vmin,
            vmax=vmax,
            interpolation=interpolation,
            aspect="auto",
        )
        ax0.set_xlabel("X (mm)")
        ax0.set_ylabel("Z (mm)")
        ax0.set_title(f"XZ Plane (Y={y[y_slice_idx]:.2f} mm)")

        # Plot XY plane
        ax1 = fig.add_subplot(gs[0, 1])
        im1 = ax1.imshow(
            xy_plane,
            cmap=cmap,
            extent=[x.min(), x.max(), y.max(), y.min()],
            vmin=vmin,
            vmax=vmax,
            interpolation=interpolation,
            aspect="auto",
        )
        ax1.set_xlabel("X (mm)")
        ax1.set_ylabel("Y (mm)")
        ax1.set_title(f"XY Plane (Z={z[z_slice_idx]:.2f} mm)")

        # Plot YZ plane
        ax2 = fig.add_subplot(gs[0, 2])
        im2 = ax2.imshow(
            yz_plane,
            cmap=cmap,
            extent=[y.min(), y.max(), z.max(), z.min()],
            vmin=vmin,
            vmax=vmax,
            interpolation=interpolation,
            aspect="auto",
        )
        ax2.set_xlabel("Y (mm)")
        ax2.set_ylabel("Z (mm)")
        ax2.set_title(f"YZ Plane (X={x[x_slice_idx]:.2f} mm)")

        # Add colorbar
        cbar_ax = fig.add_subplot(gs[0, 3])
        cbar = fig.colorbar(im2, cax=cbar_ax)
        cbar.set_label(cb_label)

        # Add title with time information
        time_val = time_display[time_idx]
        auto_title = (
            title
            if title
            else f"Transient Pressure Field - t={time_val:.3f} {time_unit} (frame {time_idx + 1}/{num_time_steps})"
        )
        fig.suptitle(auto_title, fontsize=14, fontweight="bold")

        plt.tight_layout()

        # Save frame if requested
        if save_path:
            frame_file = save_path / f"frame_{time_idx:05d}.{save_format}"
            plt.savefig(frame_file, dpi=150, bbox_inches="tight")
            if (
                time_idx % max(1, num_time_steps // 5) == 0
                or time_idx == num_time_steps - 1
            ):
                print(f"  Saved frame {time_idx + 1}/{num_time_steps}")

        # Display frame
        plt.pause(0.05)

    plt.close(fig)
    print(f" Transient visualization complete ({num_time_steps} frames)")

    # Attempt to create video from frames if save_dir was specified
    if save_path:
        try:
            import imageio

            print(f"✓ Creating video from frames...")
            frame_files = sorted(save_path.glob(f"frame_*.{save_format}"))
            if frame_files:
                video_path = save_path / f"pressure_field_video.mp4"
                reader = imageio.get_reader("pillow")
                frames = [imageio.imread(str(f)) for f in frame_files]
                imageio.mimsave(str(video_path), frames, fps=save_fps)
                print(f"✓ Video saved: {video_path.resolve()}")
        except ImportError:
            print(
                f"⚠ imageio not installed. Frame images saved but video creation skipped."
            )
        except Exception as e:
            print(f"⚠ Video creation failed: {e}")


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

    counts, bins, _ = ax1.hist(
        sub_elem_delta_k.flatten(),
        bins=10,
        color=hist_color,
        edgecolor="black",
        alpha=0.85,
    )
    max_count = counts.max()
    ax1.axvline(mean_k, color="blue", linestyle="dashed", linewidth=2)
    ax1.text(
        mean_k,
        max_count * 1.05,
        r"$\Delta  \overline{k}$" + f": {mean_k:.2f}",
        color="blue",
        ha="center",
    )

    ax1.axvline(condition, color="black", linestyle="dotted", linewidth=2)
    ax1.text(
        condition,
        max_count * 1.15,
        r"$8+2T/M$" + f": {condition:.2f}",
        color="black",
        ha="center",
    )

    ax1.set_xlabel(r"$\Delta k$")
    ax1.set_ylabel("Counts")
    # ax1.set_title(r"b)")
    ax1.grid(axis="y", color="gray", linestyle="--", alpha=0.6)

    ax1.spines["right"].set_visible(False)
    ax1.spines["top"].set_visible(False)
    ax1.set_facecolor("#FFFFFFC5")
    plt.tight_layout()

    return fig
