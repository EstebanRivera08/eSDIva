"""Matplotlib plotting helpers for 2D pressure field visualization."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from pyfield.utilities import to_dB

from .export_utils import _resolve_export_path, save_matplotlib_animation
from .plane_utils import (
    AXIS_IDX,
    PLANE_META,
    compute_plane_extents,
    parse_planes,
)
from .validators import check_coords, check_planes_shape


def plot2D_pressure_plane(
    pressure_plane,
    x=None,
    z=None,
    *,
    figsize=None,
    title=None,
    plane_axis="y",
    ax=None,
    **kwargs,
):
    """Plot a 2D pressure plane using ``matplotlib.pyplot.imshow``.

    Parameters
    ----------
    pressure_plane : (len(x), len(z)) numpy.ndarray
        2D pressure plane data.
    x : numpy.ndarray, optional
        First in-plane coordinate array (lateral axis of the slice, mm).
        If None, default indices are used.
    z : numpy.ndarray, optional
        Second in-plane coordinate array (axial axis of the slice, mm).
        If None, default indices are used.
    figsize : tuple of float, optional
        Figure size in inches. If None, computed from the grid aspect ratio.
    title : str, optional
        Figure title. If None, no title is shown.
    plane_axis : {"y", "x", "z"}, optional
        Axis normal to the plane being plotted. Controls axis labels.
        Default is ``"y"``.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw into. If None, a new figure and axes are created.
    **kwargs
        Forwarded to ``matplotlib.axes.Axes.imshow``.

    Returns
    -------
    matplotlib.axes.Axes
        Axes containing the pressure image.
    """
    kwargs.setdefault("cmap", "jet")

    if x is None or z is None:
        print(
            "Warning: x and z coordinate arrays not provided. Using default indices"
            " as coordinates."
        )
        if x is None:
            x = np.arange(pressure_plane.shape[0])
        if z is None:
            z = np.arange(pressure_plane.shape[1])
    x = np.asarray(x)
    z = np.asarray(z)

    if figsize is None:
        # Match the figure aspect to the physical extent so 1 mm renders
        # equally in both directions (clamped to avoid degenerate figures).
        width = 6
        span_x = (x.max() - x.min()) or 1.0
        span_z = (z.max() - z.min()) or 1.0
        height = width * np.clip(span_z / span_x, 0.3, 3.0)
        figsize = (width, height)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    elif not isinstance(ax, plt.Axes):
        raise ValueError("ax must be a matplotlib Axes object.")

    extent = [x.min(), x.max(), z.max(), z.min()]

    if plane_axis == "y":
        xlabel, ylabel = "X (mm)", "Z (mm)"
    elif plane_axis == "x":
        xlabel, ylabel = "Y (mm)", "Z (mm)"
    elif plane_axis == "z":
        xlabel, ylabel = "X (mm)", "Y (mm)"
    else:
        raise ValueError("plane_axis must be 'x', 'y', or 'z'.")

    im = ax.imshow(
        pressure_plane.T,
        extent=extent,
        origin="upper",
        **kwargs,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax._image = im  # ty: ignore[unresolved-attribute]
    if title is not None:
        ax.set_title(title)

    return ax


def plot2D_planes(
    planes,
    coords=None,
    *,
    center=None,
    width_ratios=[1, 1, 1, 0.05],
    figsize=(15, 5),
    **kwargs,
):
    """Plot three orthogonal 2D planes (XZ, XY, YZ) side-by-side with a shared colorbar.

    Parameters
    ----------
    planes : list of numpy.ndarray or dict
        Three 2D planes to plot. Either a list ``[plane_xz, plane_xy, plane_yz]``
        or a dict with keys ``"xz"``, ``"xy"``, ``"yz"``.
    coords : dict, optional
        Coordinate arrays as a dict with keys ``"x"``, ``"y"``, ``"z"``.
        If None, default indices are used.
    center : tuple of float, optional
        Center point ``(x0, y0, z0)`` for the planes. If None, the middle
        of the grid is used.
    width_ratios : list of float, optional
        Column width ratios for the four panels (three planes + colorbar).
        Default is ``[1, 1, 1, 0.05]``.
    figsize : tuple of float, optional
        Figure size in inches. Default is ``(15, 5)``.
    **kwargs
        Forwarded to :func:`plot2D_pressure_plane`.

    Returns
    -------
    None
        This function displays the figure and returns nothing.
    """
    if isinstance(planes, dict):
        plane_xz = planes["xz"]
        plane_xy = planes["xy"]
        plane_yz = planes["yz"]
    elif isinstance(planes, list) and len(planes) == 3:
        plane_xz, plane_xy, plane_yz = planes
    else:
        raise ValueError(
            "planes must be a list of three 2D arrays or a dict with keys"
            "'xz', 'xy', 'yz'."
        )

    nx, ny, nz = check_planes_shape([plane_xz, plane_xy, plane_yz])

    if coords is not None:
        check_coords(coords, shape=(nx, ny, nz))
        x, y, z = coords["x"], coords["y"], coords["z"]
    else:
        x = np.arange(nx)
        y = np.arange(ny)
        z = np.arange(nz)

    if center is None:
        center = (x[nx // 2], y[ny // 2], z[nz // 2])

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, 4, width_ratios=width_ratios, wspace=0.3)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0 = plot2D_pressure_plane(
        plane_xz, x, z, title="XZ Plane", plane_axis="y", ax=ax0, **kwargs
    )
    ax1 = fig.add_subplot(gs[0, 1])
    ax1 = plot2D_pressure_plane(
        plane_xy, x, y, title="XY Plane", plane_axis="z", ax=ax1, **kwargs
    )
    ax2 = fig.add_subplot(gs[0, 2])
    ax2 = plot2D_pressure_plane(
        plane_yz, y, z, title="YZ Plane", plane_axis="x", ax=ax2, **kwargs
    )

    cbar_ax = fig.add_subplot(gs[0, 3])
    cbar = fig.colorbar(ax2._image, cax=cbar_ax)
    cbar.ax.yaxis.set_label_position("left")

    plt.show()


def plot2D_pressure_slices(
    pressure_field,
    x=None,
    y=None,
    z=None,
    *,
    coords=None,
    time_array=None,
    db_scale=True,
    figsize=None,
    save_path=None,
    file_name="pressure_field.png",
    video_duration_s=5,
    fps=30,
    centered_to_max=False,
    title=None,
    label=None,
    ratios=None,
    p_max=None,
    **kwargs,
):
    """Plot orthogonal 2D slices of a pressure field.

    Handles both monochromatic (3D) and transient (4D) data. For 3D data a
    single static figure is produced; for 4D data frames are displayed
    sequentially and optionally saved to disk.

    If the 3D field has exactly one singleton dimension it is treated as a
    single plane and plotted as one image. Otherwise three orthogonal
    planes (XZ at y0, XY at z0, YZ at x0) are shown side-by-side.

    Parameters
    ----------
    pressure_field : numpy.ndarray
        Pressure field data:

        - Shape ``(Nx, Ny, Nz)``: monochromatic (static) field.
        - Shape ``(Nt, Nx, Ny, Nz)``: transient field, time along axis 0.
    x, y, z : numpy.ndarray, optional
        Coordinate arrays in mm. If None, default indices are used.
    coords : dict, optional
        Coordinate dict with keys ``"x"``, ``"y"``, ``"z"`` (and optionally
        ``"t0"``, ``"dt"``).  Overrides individual *x*, *y*, *z* when provided.
    time_array : numpy.ndarray, optional
        Physical time values in seconds for each frame. Only used for 4D data.
        If None, frame indices are used as labels.
    db_scale : bool, optional
        Convert pressures to dB before plotting. Default True.
    figsize : tuple of float, optional
        Figure size ``(width, height)`` in inches.
    save_path : str or Path, optional
        Output directory. If None, nothing is saved.

        - 3D: saves one image named ``<file_name>.png``.
        - 4D: saves one image per frame and attempts to assemble an mp4 via
          ffmpeg (falls back to GIF via pillow).
    file_name : str, optional
        File name with extension. Default ``"pressure_field.png"``.
        For 4D data, image extensions are auto-swapped to ``.mp4``.
    video_duration_s : float, optional
        Total display duration in seconds for 4D data. All frames are spread
        evenly over this duration. Default 5.
    fps : int, optional
        Frame rate for 4D display and export. Default 30.
    centered_to_max : bool, optional
        If True, slice planes pass through the pressure maximum; otherwise
        through the geometric centre. Default False.
    title : str, optional
        Figure suptitle. For 4D data the time stamp is appended when None.
    label : str, optional
        Colorbar label. Auto-set from ``db_scale`` when None.
    ratios : array-like of length 3, optional
        Manual column width ratios ``[XZ, XY, YZ]``. Auto-computed from
        physical extents when None. Only used for 3D full-volume data.
    p_max : float, optional
        Reference peak for normalisation. Used as ``vmax`` in dB conversion
        or as divisor in linear mode. Defaults to ``abs(field).max()``.
    **kwargs
        Forwarded to :func:`plot2D_pressure_plane` (e.g. ``vmin``, ``vmax``,
        ``interpolation``).

    Returns
    -------
    None
        This function displays the figure and returns nothing.
    """
    import pathlib

    # --- unpack coords dict ---
    if coords is not None:
        x = coords.get("x", x)
        y = coords.get("y", y)
        z = coords.get("z", z)
        if time_array is None and "t0" in coords:
            nt = pressure_field.shape[0]
            time_array = coords["t0"] + np.arange(nt) * coords["dt"]

    if label is None:
        label = "Pressure (dB)" if db_scale else "Pressure (a.u.)"

    kwargs.setdefault("cmap", "jet")

    # --- dimensionality check ---
    if pressure_field.ndim == 3:
        nx, ny, nz = pressure_field.shape
        is_transient = False
    elif pressure_field.ndim == 4:
        is_transient = True
        nt, nx, ny, nz = pressure_field.shape
    else:
        raise ValueError("pressure_field must be 3D (Nx,Ny,Nz) or 4D (Nt,Nx,Ny,Nz).")

    if x is None or y is None or z is None:
        print(
            "Warning: x, y, z coordinate arrays not provided. Using default indices"
            " as coordinates."
        )
        x = np.arange(nx)
        y = np.arange(ny)
        z = np.arange(nz)

    # --- save directory ---
    if save_path is not None:
        save_path = pathlib.Path(save_path)
        print(f"Output will be saved to: {save_path.resolve()}")

    # ==========================================================================
    # MONOCHROMATIC (3D)
    # ==========================================================================
    if not is_transient:
        print(f"Pressure field shape: [{nx}, {ny}, {nz}]")

        if db_scale:
            pressure_plot = to_dB(pressure_field, vmax=p_max)
            cb_label = label
            vmin = kwargs.pop("vmin", -40)
            vmax = kwargs.pop("vmax", 0)
        else:
            ref = p_max if p_max is not None else 1.0
            pressure_plot = pressure_field / ref
            cb_label = label
            vmin = kwargs.pop("vmin", float(np.nanmin(pressure_plot)))
            vmax = kwargs.pop("vmax", float(np.nanmax(pressure_plot)))

        # --- single-plane case (one singleton spatial dimension) ---
        if nx == 1:
            plane_axis, x1, x2 = "x", y, z
        elif ny == 1:
            plane_axis, x1, x2 = "y", x, z
        elif nz == 1:
            plane_axis, x1, x2 = "z", x, y
        else:
            plane_axis = None

        if plane_axis is not None:
            print(f"Plotting 2D slice along {plane_axis}-axis")
            if figsize is None:
                # Aspect from physical extents: axial span sets the height.
                span_1 = (np.max(x1) - np.min(x1)) or 1.0
                span_2 = (np.max(x2) - np.min(x2)) or 1.0
                width = 6
                figsize = (width, width * np.clip(span_2 / span_1, 0.3, 3.0) + 0.5)
            fig, ax = plt.subplots(figsize=figsize)
            ax = plot2D_pressure_plane(
                pressure_plot.squeeze(),
                x=x1,
                z=x2,
                ax=ax,
                plane_axis=plane_axis,
                vmin=vmin,
                vmax=vmax,
                **kwargs,
            )
            cbar = fig.colorbar(ax._image, ax=ax)
            cbar.set_label(cb_label)

        else:
            # --- three-panel layout for full 3D volumes ---
            if centered_to_max:
                max_idx = np.unravel_index(
                    np.nanargmax(pressure_plot), pressure_plot.shape
                )
                x0, y0, z0 = max_idx[0], max_idx[1], max_idx[2]
            else:
                x0, y0, z0 = len(x) // 2, len(y) // 2, len(z) // 2

            print(
                f"Taking slice ({x[x0]:.1e},{y[y0]:.1e},{z[z0]:.1e}) mm"
                f" => x_ind, y_ind, z_ind = {x0 + 1}/{nx}, {y0 + 1}/{ny}, {z0 + 1}/{nz}"
            )

            Dx = x.max() - x.min() or 1.0
            Dy = y.max() - y.min() or 1.0
            Dz = z.max() - z.min() or 1.0

            if ratios is not None:
                if len(ratios) != 3:
                    raise ValueError("ratios must have length 3.")
                r = np.asarray(ratios, dtype=float)
                r = r / r.sum()
            else:
                r = np.array([Dx / Dz, Dx / Dy, Dy / Dz])
                r = r / r.sum()

            if figsize is None:
                # Total width from the three panel aspect ratios at a common
                # height (clamped so extreme extents stay printable).
                height = 5.0
                panel_w = height * np.clip(
                    np.array([Dx / Dz, Dx / Dy, Dy / Dz]), 0.3, 3.0
                )
                figsize = (float(np.clip(panel_w.sum() + 1.0, 6.0, 18.0)), height)

            fig = plt.figure(figsize=figsize)
            gs = GridSpec(1, 4, width_ratios=[r[0], r[1], r[2], 0.05 * r.max()])

            ax0 = fig.add_subplot(gs[0, 0])
            ax0 = plot2D_pressure_plane(
                pressure_plot[:, y0, :].squeeze(),
                x=x,
                z=z,
                title=f"XZ Plane (Y={y[y0]:.2f} mm)",
                plane_axis="y",
                ax=ax0,
                vmin=vmin,
                vmax=vmax,
                **kwargs,
            )
            ax1 = fig.add_subplot(gs[0, 1])
            ax1 = plot2D_pressure_plane(
                pressure_plot[:, :, z0].squeeze(),
                x=x,
                z=y,
                title=f"XY Plane (Z={z[z0]:.2f} mm)",
                plane_axis="z",
                ax=ax1,
                vmin=vmin,
                vmax=vmax,
                **kwargs,
            )
            ax2 = fig.add_subplot(gs[0, 2])
            ax2 = plot2D_pressure_plane(
                pressure_plot[x0, :, :].squeeze(),
                x=y,
                z=z,
                title=f"YZ Plane (X={x[x0]:.2f} mm)",
                plane_axis="x",
                ax=ax2,
                vmin=vmin,
                vmax=vmax,
                **kwargs,
            )
            cbar_ax = fig.add_subplot(gs[0, 3])
            cbar = fig.colorbar(ax2._image, cax=cbar_ax)
            cbar.set_label(cb_label)
            cbar.ax.yaxis.set_label_position("left")

        if title is not None:
            fig.suptitle(title)
        plt.tight_layout()

        if save_path is not None:
            sp = _resolve_export_path(save_path, file_name)
            plt.savefig(sp)
            print(f"\nPlot saved to: {sp}")

        plt.show()
        plt.close()
        return

    # ==========================================================================
    # TRANSIENT (4D) — frame-by-frame animation
    # ==========================================================================
    print(
        f"Plotting transient pressure field: shape={pressure_field.shape}, frames={nt}"
    )

    if time_array is None:
        time_display = np.arange(nt).astype(float)
        time_unit = "frame"
    elif time_array[0] < 1e-3:
        time_display = time_array * 1e6  # s → µs
        time_unit = "µs"
    else:
        time_display = np.asarray(time_array, dtype=float)
        time_unit = "s"

    from matplotlib.animation import FuncAnimation

    step = max(1.0, nt / (video_duration_s * fps))
    frame_indices = np.unique(np.arange(0, nt, step).astype(int))
    n_display = len(frame_indices)
    interval_ms = 1000.0 / fps

    # Global reference across all frames so the colour scale is consistent.
    if p_max is None:
        p_max = float(np.nanmax(np.abs(pressure_field)))
        if p_max == 0:
            p_max = 1.0

    display_frames = pressure_field[frame_indices]  # (n_display, Nx, Ny, Nz)
    if db_scale:
        display_frames = to_dB(display_frames, vmax=p_max)
        vmin = kwargs.pop("vmin", -40)
        vmax_plot = kwargs.pop("vmax", 0)
    else:
        display_frames = display_frames / p_max
        vmin = kwargs.pop("vmin", float(np.nanmin(display_frames)))
        vmax_plot = kwargs.pop("vmax", float(np.nanmax(display_frames)))

    is_plane_xz = ny == 1
    is_plane_xy = nz == 1
    is_plane_yz = nx == 1
    is_plane = is_plane_xz or is_plane_xy or is_plane_yz

    if centered_to_max:
        xi, yi, zi = np.unravel_index(
            np.nanargmax(np.abs(pressure_field[frame_indices[0]])),
            pressure_field.shape[1:],
        )
    else:
        xi, yi, zi = len(x) // 2, len(y) // 2, len(z) // 2

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

        im_main = ax_main.imshow(
            init_data,
            origin="upper",
            extent=extent,
            vmin=vmin,
            vmax=vmax_plot,
            **kwargs,
        )
        ax_main.set_xlabel(xlabel)
        ax_main.set_ylabel(ylabel)
        fig.colorbar(im_main, ax=ax_main, label=label)
        plt.tight_layout()
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
        Dx = x.max() - x.min() or 1.0
        Dy = y.max() - y.min() or 1.0
        Dz = z.max() - z.min() or 1.0
        r = np.array([Dx / Dz, Dx / Dy, Dy / Dz])
        r /= r.sum()

        fig = plt.figure(figsize=figsize)
        gs = GridSpec(1, 4, width_ratios=[r[0], r[1], r[2], 0.05])

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
            vmin=vmin,
            vmax=vmax_plot,
            **kwargs,
        )
        im1 = ax1.imshow(
            f0[:, :, zi].T,
            origin="upper",
            extent=[x.min(), x.max(), y.max(), y.min()],
            vmin=vmin,
            vmax=vmax_plot,
            **kwargs,
        )
        im2 = ax2.imshow(
            f0[xi, :, :].T,
            origin="upper",
            extent=[y.min(), y.max(), z.max(), z.min()],
            vmin=vmin,
            vmax=vmax_plot,
            **kwargs,
        )

        ax0.set_title(f"XZ  (Y={y[yi]:.2f} mm)")
        ax1.set_title(f"XY  (Z={z[zi]:.2f} mm)")
        ax2.set_title(f"YZ  (X={x[xi]:.2f} mm)")

        cbar_ax = fig.add_subplot(gs[0, 3])
        fig.colorbar(im2, cax=cbar_ax, label=label)
        plt.tight_layout()
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
        # Derive video filename: swap image extensions to .mp4
        vname = file_name
        if pathlib.Path(vname).suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}:
            vname = str(pathlib.Path(vname).with_suffix(".mp4"))
        save_matplotlib_animation(ani, save_path, vname, fps=fps)

    plt.show()
    plt.close(fig)
    print(
        f"Done — {n_display}/{nt} frames displayed at {fps} fps ({video_duration_s:.1f} s)."
    )


def plot2D_transient_slices(
    pressure_field,
    x=None,
    y=None,
    z=None,
    *,
    coords=None,
    time_array=None,
    center_mm=None,
    center_to_max=False,
    db_scale=True,
    figsize=None,
    save_path=None,
    file_name="transient_2Dslices.gif",
    video_duration_s=5,
    fps=30,
    title=None,
    label=None,
    p_max=None,
    **kwargs,
):
    """Animate orthogonal pressure slices of transient data with Matplotlib.

    Accepts either a full 4D volume (slices computed internally) or a dict of
    pre-computed 3D planes (up to 3).

    Parameters
    ----------
    pressure_field : numpy.ndarray or dict
        - ``(Nt, Nx, Ny, Nz)`` ndarray: full transient volume. Orthogonal
          slices are extracted automatically.
        - ``dict`` with keys from ``{"xz", "xy", "yz"}``: pre-computed planes,
          each ``(Nt, N1, N2)``.
    x, y, z : numpy.ndarray, optional
        Coordinate arrays in mm. Default: index arrays.
    coords : dict, optional
        Coordinate dict with keys ``"x"``, ``"y"``, ``"z"`` (and optionally
        ``"t0"``, ``"dt"``).  Overrides individual *x*, *y*, *z* when provided.
    time_array : numpy.ndarray, optional
        Physical time values (length Nt). If None, frame indices are used.
    center_mm : tuple of float, optional
        ``(x0, y0, z0)`` in mm.  For volume input this selects the slice
        position; for planes input it sets the subplot titles.
        Default: geometric centre (volume) or coordinate midpoints (planes).
    center_to_max : bool, optional
        If True and input is a volume, slice through the global pressure
        maximum instead of *center_mm*. Default False.
    db_scale : bool, optional
        Convert to dB before display. Default True.
    figsize : tuple of float, optional
        Figure size in inches.
    save_path : str or Path, optional
        Output directory. Saves MP4 (or GIF fallback).
    file_name : str, optional
        Base file name for saved video. Default ``"transient_slices"``.
    video_duration_s : float, optional
        Target video duration in seconds. Default 5.
    fps : int, optional
        Frame rate. Default 30.
    title : str, optional
        Override time-stamp text in each frame.
    label : str, optional
        Colorbar label.
    p_max : float, optional
        Reference peak for normalisation / dB conversion.
    **kwargs
        Forwarded to ``imshow`` (e.g. ``cmap``, ``vmin``, ``vmax``,
        ``interpolation``).
    """

    from matplotlib.animation import FuncAnimation

    # --- unpack coords dict ---
    if coords is not None:
        x = coords.get("x", x)
        y = coords.get("y", y)
        z = coords.get("z", z)
        if time_array is None and "t0" in coords:
            if isinstance(pressure_field, np.ndarray):
                nt = pressure_field.shape[0]
            else:
                nt = next(iter(pressure_field.values())).shape[0]
            time_array = coords["t0"] + np.arange(nt) * coords["dt"]

    if label is None:
        label = "Pressure (dB)" if db_scale else "Pressure (a.u.)"
    kwargs.setdefault("cmap", "jet")

    # ------------------------------------------------------------------
    # Resolve input: 4D volume / dict / list-of-dicts → PlaneSpec list
    # ------------------------------------------------------------------
    plane_specs, center_mm, coords = parse_planes(
        pressure_field,
        expected_ndim=3,
        coords={"x": x, "y": y, "z": z},
        center_mm=center_mm,
        center_to_max=center_to_max,
    )
    compute_plane_extents(plane_specs, coords)
    x, y, z = coords["x"], coords["y"], coords["z"]

    # Build ordered dict for frame access
    plane_order = [ps.name for ps in plane_specs]
    planes = {ps.name: ps.data for ps in plane_specs}
    nt = next(iter(planes.values())).shape[0]

    # Truncate to the minimum common frame count across planes
    min_nt = min(v.shape[0] for v in planes.values())
    if min_nt < nt:
        planes = {k: v[:min_nt] for k, v in planes.items()}
        nt = min_nt

    # ------------------------------------------------------------------
    # Global reference for dB / normalisation (computed before decimation)
    # ------------------------------------------------------------------
    if p_max is None:
        p_max = max(float(np.nanmax(np.abs(v))) for v in planes.values())
    if p_max == 0:
        p_max = 1.0

    # ------------------------------------------------------------------
    # Frame decimation
    # ------------------------------------------------------------------
    step = max(1.0, nt / (video_duration_s * fps))
    frame_indices = np.unique(np.arange(0, nt, step).astype(int))
    n_display = len(frame_indices)
    interval_ms = 1000.0 / fps

    disp = {k: v[frame_indices] for k, v in planes.items()}

    if db_scale:
        disp = {k: to_dB(v, vmax=p_max) for k, v in disp.items()}
        vmin = kwargs.pop("vmin", -40)
        vmax = kwargs.pop("vmax", 0)
    else:
        disp = {k: v / p_max for k, v in disp.items()}
        vmin = kwargs.pop("vmin", min(float(np.nanmin(v)) for v in disp.values()))
        vmax = kwargs.pop("vmax", max(float(np.nanmax(v)) for v in disp.values()))

    # ------------------------------------------------------------------
    # Time display
    # ------------------------------------------------------------------
    if time_array is None:
        time_display = np.arange(nt, dtype=float)
        time_unit = "frame"
    elif np.max(time_array) < 1e-3:
        time_display = np.asarray(time_array) * 1e6
        time_unit = "µs"
    else:
        time_display = np.asarray(time_array, dtype=float)
        time_unit = "s"

    # ------------------------------------------------------------------
    # Layout: adaptive to number of planes (1-3)
    # ------------------------------------------------------------------
    n_planes = len(plane_order)

    ratios = []
    plane_spec_map = {ps.name: ps for ps in plane_specs}
    for key in plane_order:
        ps = plane_spec_map[key]
        # Use per-plane extent (c1_min, c1_max, c2_min, c2_max)
        ext = ps.extent
        D1 = (ext[1] - ext[0]) or 1.0
        D2 = (ext[3] - ext[2]) or 1.0
        ratios.append(D1 / D2)
    ratios = np.array(ratios)
    ratios /= ratios.sum()

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(
        1,
        n_planes + 1,
        width_ratios=[*ratios, 0.05 * ratios.max()],
    )

    ims = []
    axes = []
    for i, key in enumerate(plane_order):
        meta = PLANE_META[key]
        ps = plane_spec_map[key]
        ext = ps.extent  # (c1_min, c1_max, c2_min, c2_max)
        extent = [ext[0], ext[1], ext[3], ext[2]]  # imshow: [left, right, bottom, top]
        normal_val = ps.translation[AXIS_IDX[meta["normal"]]]

        ax = fig.add_subplot(gs[0, i])
        im = ax.imshow(
            disp[key][0].T,
            origin="upper",
            extent=extent,
            vmin=vmin,
            vmax=vmax,
            **kwargs,
        )
        ax.set_xlabel(meta["xlabel"])
        ax.set_ylabel(meta["ylabel"])
        ax.set_title(f"{key.upper()} ({meta['normal'].upper()}={normal_val:.2f} mm)")
        ims.append(im)
        axes.append(ax)

    cbar_ax = fig.add_subplot(gs[0, n_planes])
    fig.colorbar(ims[-1], cax=cbar_ax, label=label)
    plt.tight_layout()

    # Time text on the middle axis
    mid_ax = axes[len(axes) // 2]
    time_text = mid_ax.text(
        0.5,
        0.97,
        "t = 0",
        transform=mid_ax.transAxes,
        ha="center",
        va="top",
        fontsize=11,
        color="white",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.5),
    )

    def _update(i):
        for j, key in enumerate(plane_order):
            ims[j].set_data(disp[key][i].T)
        t_val = time_display[frame_indices[i]]
        time_text.set_text(
            title
            if title
            else f"t = {t_val:.3f} {time_unit}  ({frame_indices[i] + 1}/{nt})"
        )
        return [*ims, time_text]

    ani = FuncAnimation(
        fig,
        _update,
        frames=n_display,
        interval=interval_ms,
        blit=True,
        repeat=False,
    )

    if save_path:
        save_matplotlib_animation(ani, save_path, file_name, fps=fps)

    plt.show()
    plt.close(fig)
    print(f"Done — {n_display}/{nt} frames at {fps} fps ({video_duration_s:.1f} s).")
