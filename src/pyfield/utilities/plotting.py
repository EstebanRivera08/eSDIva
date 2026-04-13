"""Matplotlib and PyVista plotting helpers for pressure fields."""

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from matplotlib.gridspec import GridSpec

from .helper_functions import to_dB
from .plotting_pyvista import add_pressure_vol, create_vol_mesh


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
    Plot a 3D pressure field as a PyVista volume with bounding box and axes.

    Parameters
    ----------
    x : (Nx,) numpy.ndarray
        Lateral coordinates (mm).
    y : (Ny,) numpy.ndarray
        Elevation coordinates (mm).
    z : (Nz,) numpy.ndarray
        Axial coordinates (mm).
    pressure_field : (Nx, Ny, Nz) numpy.ndarray
        Pressure field samples on the grid.
    scalars : str, optional
        Name of the scalar array attached to the volume. Default is
        ``"Pressure"``.
    plotter : pyvista.Plotter, optional
        Existing plotter to draw into. If None, a new plotter is created.
        Default is None.
    off_screen : bool, optional
        Render off-screen (no window). Default is False.
    window_size : list[int], optional
        Window size as ``[width, height]`` in pixels before ``scale``.
        Default is ``[520, 720]``.
    notebook : bool, optional
        Use notebook-mode rendering. Default is False.
    return_mesh : bool, optional
        If True, also return the pressure volume mesh. Default is False.
    plot_focal_spot : bool, optional
        Draw the focal spot as an isosurface. Default is False.
    scale : float, optional
        Resolution scale factor applied to the window size and fonts.
        Default is 1.0.
    anti_aliasing : str, optional
        PyVista anti-aliasing mode. Default is ``"ssaa"``.
    colorbar_title : str, optional
        Colorbar title. If None, use the scalar name. Default is None.
    box_color : str, optional
        Bounding-box color. Default is ``"#b0b0b0"``.
    box_opacity : float, optional
        Bounding-box opacity. Default is 0.2.
    contour_levels : int, optional
        Number of isosurface levels. Default is 11.
    **kwargs
        Forwarded to ``plotter.add_mesh()``.

    Returns
    -------
    pyvista.Plotter or tuple[pyvista.Plotter, pyvista.ImageData]
        The configured plotter, or ``(plotter, pressure_vol)`` when
        ``return_mesh=True``.
    """
    pv.global_theme.anti_aliasing = anti_aliasing
    # Create the pressure volume mesh
    pressure_vol = create_vol_mesh(x, y, z, pressure_field, scalars=scalars)
    box = pressure_vol.bounding_box()

    kwargs.pop("ambient", 0.7)  # set default

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
    Plot a 2D pressure plane using ``matplotlib.pyplot.imshow``.

    Parameters
    ----------
    x : numpy.ndarray
        First in-plane coordinate array (lateral axis of the slice, mm).
    z : numpy.ndarray
        Second in-plane coordinate array (axial axis of the slice, mm).
    pressure_plane : (len(x), len(z)) numpy.ndarray
        2D pressure plane data.
    figsize : tuple[float, float], optional
        Figure size in inches. If None, computed from the grid aspect
        ratio. Default is None.
    title : str, optional
        Figure title. Default is None.
    plane_axis : str, optional
        Axis along which the plane is taken (``"x"``, ``"y"``, or ``"z"``).
        Default is ``"y"``.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw into. If None, a new figure and axes are
        created. Default is None.
    **kwargs
        Forwarded to ``matplotlib.axes.Axes.imshow``.

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the pressure image.
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
    ax._image = im  # ty: ignore[unresolved-attribute]
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
    save_path=None,
    ratios=None,
    label=None,
    p_max=None,
    **kwargs,
):
    """
    Plot orthogonal 2D slices of a monochromatic (3-D) pressure field.

    If the field has exactly one singleton dimension it is treated as a
    single plane and plotted as one image.  Otherwise three orthogonal
    planes (XZ at y0, XY at z0, YZ at x0) are shown side-by-side.

    For transient (4-D) data use :func:`plot_slices_2d` instead.

    Parameters
    ----------
    x, y, z : ndarray
        Coordinate arrays in mm.
    pressure_field : ndarray, shape (Nx, Ny, Nz)
        Monochromatic pressure values.
    db_scale : bool, optional
        Convert to dB before plotting.  Default False.
    figsize : tuple, optional
        Matplotlib figure size ``(width, height)``.
    title : str, optional
        Figure suptitle.
    centered_to_max : bool, optional
        If True the slice planes pass through the pressure maximum.
        Default False (geometric centre).
    save_path : str or Path, optional
        Directory for output file.
    ratios : array-like of length 3, optional
        Manual column width ratios ``[XZ, XY, YZ]``.  Auto-computed from
        physical extents when None.
    label : str, optional
        Colorbar label.
    p_max : float, optional
        Reference maximum for normalization. Default is None.
    **kwargs
        Passed through to :func:`plot_pressure_2D` (e.g. ``vmin``, ``vmax``,
        ``interpolation``).
    """

    if pressure_field.ndim != 3:
        raise ValueError(
            "plot_pressure_planes expects a 3-D array (Nx, Ny, Nz). "
            "For transient (4-D) data use plot_slices_2d."
        )
    nx, ny, nz = pressure_field.shape
    print(f"Pressure field shape: [{nx}, {ny}, {nz}]")

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
        if p_max is None:
            p_max = 1
        pressure_field = pressure_field / p_max
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

        # GridSpec: three data columns with aspect ratios matching field extents,
        # plus a narrow fourth column for the shared colorbar.
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(
            1, 4, width_ratios=[ratios[0], ratios[1], ratios[2], 0.05 * ratios.max()]
        )

        ax0 = fig.add_subplot(gs[0, 0])
        ax0 = plot_pressure_2D(
            x,
            z,
            pressure_field[:, y0, :].squeeze(),
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

    if save_path is not None:
        from pathlib import Path

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        print(f"\nPlot saved to: {save_path}")
    plt.show()
    plt.close()


def plot_slices_2d(
    x,
    y,
    z,
    pressure_field,
    *,
    time_array=None,
    db_scale=True,
    figsize=None,
    save_path=None,
    file_name="pressure_field",
    video_duration_s=5,
    fps=30,
    vmin=None,
    vmax=None,
    centered_to_max=False,
    cmap="jet",
    title=None,
    label=None,
    interpolation=None,
):
    """
    Plot orthogonal 2D slices of a pressure field.

    Handles both monochromatic (3D) and transient (4D) data. For 3D data a
    single static figure is produced; for 4D data frames are displayed
    sequentially and optionally saved to disk.

    Parameters
    ----------
    x, y, z : ndarray
        Coordinate arrays in mm.
    pressure_field : ndarray
        Pressure field data:
        - 3D array ``(Nx, Ny, Nz)``: monochromatic field.
        - 4D array ``(Nt, Nx, Ny, Nz)``: transient field (time along axis 0).
    time_array : ndarray, optional
        Physical time values for each frame (seconds).  If None, frame indices
        are used as labels.  Only used for 4D data.
    db_scale : bool, optional
        Convert pressures to dB before plotting.  Default True.
    figsize : tuple, optional
        Figure size ``(width, height)`` in inches.  Default ``(10, 5)``.
    save_path : str or Path, optional
        Directory for output files.  None means no saving.
        - 3D: saves one image named ``pressure_field.<fmt>``.
        - 4D: saves one image per frame (``frame_NNNNN.<fmt>``) and attempts
          to assemble an mp4 video via imageio (skipped if not installed).
    file_name : str, optional
        Base name for saved files (without extension).  Default ``"pressure_field"``.
    save_format : str, optional
        Image format for saved frames.  Default ``"png"``.
    video_duration_s : float, optional
        Total display and save duration in seconds (4D only).  All ``nt``
        frames are spread evenly over this duration.  Default 5 seconds.
    fps : int, optional
        Reserved parameter (kept for API compatibility).  The actual display
        and save frame rate is computed as ``nt / video_duration_s`` so that
        all frames are visible within the requested duration.
    vmin, vmax : float, optional
        Colorbar limits.  If None, computed globally from all frames so the
        colorbar is consistent across the animation.
    centered_to_max : bool, optional
        If True the three slice planes pass through the pressure maximum;
        otherwise they pass through the geometric centre.  Default False.
    cmap : str, optional
        Matplotlib colormap name.  Default ``"jet"``.
    title : str, optional
        Figure suptitle.  For 4D data the time stamp is appended unless a
        custom title is provided.
    label : str, optional
        Colorbar label.  Auto-set from ``db_scale`` when None.
    interpolation : str, optional
        ``imshow`` interpolation method (e.g. ``"bilinear"``).  Default None.

    Returns
    -------
    None
        Nothing is returned; the figure is shown via ``matplotlib.pyplot``
        and, if ``save_path`` is given, frames/videos are written to disk.
    """
    import pathlib

    if label is None:
        label = "Pressure (dB)" if db_scale else "Pressure (a.u.)"

    # --- dimensionality check ---
    if pressure_field.ndim == 3:
        is_transient = False
    elif pressure_field.ndim == 4:
        is_transient = True
        nt, nx, ny, nz = pressure_field.shape
    else:
        raise ValueError("pressure_field must be 3D (Nx,Ny,Nz) or 4D (Nt,Nx,Ny,Nz).")

    # --- save directory ---
    if save_path is not None:
        save_path = pathlib.Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        print(f"Output will be saved to: {save_path.resolve()}")
    else:
        save_path = None

    # === MONOCHROMATIC (3D) -- delegate to plot_pressure_planes ===
    if not is_transient:
        pressure_plot = to_dB(pressure_field) if db_scale else pressure_field.copy()
        auto_title = title or "Monochromatic Pressure Field"
        plot_pressure_planes(
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
            label=label,
            save_path=save_path / f"{file_name}.png" if save_path else None,
        )
        return

    # === TRANSIENT (4D) -- frame-by-frame animation ===
    print(
        f"Plotting transient pressure field: shape={pressure_field.shape}, frames={nt}"
    )

    # Default time labels
    if time_array is None:
        time_display = np.arange(nt).astype(float)
        time_unit = "frame"
    elif time_array[0] < 1e-3:
        time_display = time_array * 1e6  # seconds -> µs
        time_unit = "µs"
    else:
        time_display = np.asarray(time_array, dtype=float)
        time_unit = "s"

    from matplotlib.animation import FuncAnimation

    # Subsample first: only process the frames that will actually be displayed.
    # step = nt / (duration * fps)  ->  n_display ~= duration * fps frames
    step = max(1.0, nt / (video_duration_s * fps))
    frame_indices = np.unique(np.arange(0, nt, step).astype(int))
    n_display = len(frame_indices)
    interval_ms = 1000.0 / fps

    # Convert only the display subset — avoids processing the full nt-frame array
    display_frames = pressure_field[frame_indices]  # (n_display, Nx, Ny, Nz)
    if db_scale:
        display_frames = to_dB(display_frames)

    # Global vmin/vmax from the display subset (consistent colorbar)
    if vmin is None:
        vmin = float(np.nanmin(display_frames))
    if vmax is None:
        vmax = float(np.nanmax(display_frames))

    # Detect planar fields: if one spatial dimension is 1, show a single panel
    is_plane_xz = ny == 1  # most common: single XZ plane
    is_plane_xy = nz == 1
    is_plane_yz = nx == 1
    is_plane = is_plane_xz or is_plane_xy or is_plane_yz

    # Fixed slice indices (centred or max of first display frame)
    if centered_to_max:
        xi, yi, zi = np.unravel_index(
            np.nanargmax(np.abs(display_frames[0])), display_frames[0].shape
        )
    else:
        xi, yi, zi = len(x) // 2, len(y) // 2, len(z) // 2

    imshow_kw = dict(
        vmin=vmin, vmax=vmax, cmap=cmap, interpolation=interpolation, aspect="auto"
    )

    # --- Build figure (single panel for planar fields, three panels for volumes)
    if is_plane:
        fig, ax_main = plt.subplots(1, 1, figsize=figsize)

        if is_plane_xz:
            init_data = display_frames[0][:, 0, :].T
            extent = [x.min(), x.max(), z.max(), z.min()]
            xlabel, ylabel = "X (mm)", "Z (mm)"
        elif is_plane_xy:
            init_data = display_frames[0][:, :, 0].T
            extent = [x.min(), x.max(), y.max(), y.min()]
            xlabel, ylabel = "X (mm)", "Y (mm)"
        else:
            init_data = display_frames[0][0, :, :].T
            extent = [y.min(), y.max(), z.max(), z.min()]
            xlabel, ylabel = "Y (mm)", "Z (mm)"

        im_main = ax_main.imshow(init_data, origin="upper", extent=extent, **imshow_kw)
        ax_main.set_xlabel(xlabel)
        ax_main.set_ylabel(ylabel)
        fig.colorbar(im_main, ax=ax_main, label=label)
        plt.tight_layout()
        # Axes-level text inside the axes so tight_layout cannot clip it
        time_text = ax_main.text(
            0.5,
            0.97,
            "t = 0",
            transform=ax_main.transAxes,
            ha="center",
            va="top",
            fontsize=11,
            color="white",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.5),
        )

        def _update(i):
            p = display_frames[i]
            if is_plane_xz:
                im_main.set_data(p[:, 0, :].T)
            elif is_plane_xy:
                im_main.set_data(p[:, :, 0].T)
            else:
                im_main.set_data(p[0, :, :].T)
            t_val = time_display[frame_indices[i]]
            time_text.set_text(
                title
                if title
                else f"t = {t_val:.3f} {time_unit}  ({frame_indices[i] + 1}/{nt})"
            )
            return [im_main, time_text]

    else:
        # Three-panel layout for full 3D volume fields
        Dx = x.max() - x.min() or 1.0
        Dy = y.max() - y.min() or 1.0
        Dz = z.max() - z.min() or 1.0
        ratios = np.array([Dx / Dz, Dx / Dy, Dy / Dz])
        ratios /= ratios.sum()

        fig = plt.figure(figsize=figsize)
        gs = GridSpec(1, 4, width_ratios=[ratios[0], ratios[1], ratios[2], 0.05])

        ax0 = fig.add_subplot(gs[0, 0])
        ax0.set_xlabel("X (mm)")
        ax0.set_ylabel("Z (mm)")
        ax1 = fig.add_subplot(gs[0, 1])
        ax1.set_xlabel("X (mm)")
        ax1.set_ylabel("Y (mm)")
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.set_xlabel("Y (mm)")
        ax2.set_ylabel("Z (mm)")

        f0 = display_frames[0]
        im0 = ax0.imshow(
            f0[:, yi, :].T,
            origin="upper",
            extent=[x.min(), x.max(), z.max(), z.min()],
            **imshow_kw,
        )
        im1 = ax1.imshow(
            f0[:, :, zi].T,
            origin="upper",
            extent=[x.min(), x.max(), y.max(), y.min()],
            **imshow_kw,
        )
        im2 = ax2.imshow(
            f0[xi, :, :].T,
            origin="upper",
            extent=[y.min(), y.max(), z.max(), z.min()],
            **imshow_kw,
        )

        ax0.set_title(f"XZ  (Y={y[yi]:.2f} mm)")
        ax1.set_title(f"XY  (Z={z[zi]:.2f} mm)")
        ax2.set_title(f"YZ  (X={x[xi]:.2f} mm)")

        cbar_ax = fig.add_subplot(gs[0, 3])
        fig.colorbar(im2, cax=cbar_ax, label=label)
        plt.tight_layout()
        # Axes-level text inside centre panel so tight_layout cannot clip it
        time_text = ax1.text(
            0.5,
            0.97,
            "t = 0",
            transform=ax1.transAxes,
            ha="center",
            va="top",
            fontsize=11,
            color="white",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.5),
        )

        def _update(i):
            p = display_frames[i]
            im0.set_data(p[:, yi, :].T)
            im1.set_data(p[:, :, zi].T)
            im2.set_data(p[xi, :, :].T)
            t_val = time_display[frame_indices[i]]
            time_text.set_text(
                title
                if title
                else f"t = {t_val:.3f} {time_unit}  ({frame_indices[i] + 1}/{nt})"
            )
            return [im0, im1, im2, time_text]

    ani = FuncAnimation(
        fig, _update, frames=n_display, interval=interval_ms, blit=True, repeat=False
    )

    if save_path:
        # n_display frames at fps gives exactly video_duration_s seconds
        save_fps = fps
        video_path = save_path / f"{file_name}_video.mp4"
        try:
            ani.save(str(video_path), writer="ffmpeg", fps=save_fps, dpi=150)
            print(f"Video saved: {video_path.resolve()}")
        except Exception as e:
            # ffmpeg not available — fall back to pillow gif
            gif_path = save_path / f"{file_name}_video.gif"
            try:
                ani.save(str(gif_path), writer="pillow", fps=save_fps)
                print(f"GIF saved (ffmpeg unavailable): {gif_path.resolve()}")
            except Exception as e2:
                print(f"Video/GIF export failed: {e} | {e2}")

    plt.show()
    plt.close(fig)
    print(
        f"Done — {n_display}/{nt} frames displayed at {fps} fps ({video_duration_s:.1f} s)."
    )


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
    """
    Plot the distribution of the SDI band-limiting factor ``delta_k``.

    Draws a side-by-side range plot and histogram to inspect the
    ``delta_k`` values used by the sparse-delta-integration path.

    Parameters
    ----------
    pyfield : pyfield.psimulation.PyField
        Simulation instance whose last-run ``sub_elem_delta_k``, ``T_log``
        and ``P_log`` are plotted.
    figsize : tuple[float, float], optional
        Figure size in inches. Default is ``(11, 4)``.
    per_element : bool, optional
        If True, average ``delta_k`` across sub-elements belonging to the
        same physical element; otherwise plot per-patch values. Default is
        True.
    cmap : str, optional
        Colormap for the range plot. Default is ``"turbo"``.
    hist_color : str, optional
        Fill color for the histogram bars. Default is ``"#AB0000E8"``.
    xlim : tuple[float, float], optional
        X-axis limits for the histogram. Default is None (auto).
    ylim : tuple[float, float], optional
        Y-axis limits for the range plot. Default is None (auto).

    Returns
    -------
    matplotlib.figure.Figure
        The matplotlib figure containing the two subplots.
    """
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
    max_count = np.max(counts)
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
