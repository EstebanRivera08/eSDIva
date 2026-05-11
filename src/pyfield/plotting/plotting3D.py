from pathlib import Path

import numpy as np
import pyvista as pv

import pyfield as pf

from .plotting_pyvista import add_2D_image, add_pressure_vol
from .pyvista_functions import (
    _normalize_window_size,
    _set_custom_style,
    create_2Dimage_mesh,
    create_3Dvol_mesh,
)


def plot3D_pressure_vol(
    pressure_field,
    x=None,
    y=None,
    z=None,
    *,
    db_scale=False,
    coords=None,
    show_fig=True,
    save_path=None,
    file_name="3D_pressure_slices.mp4",
    scalars="Pressure",
    plotter=None,
    off_screen=False,
    window_size=[520, 720],
    notebook=False,
    plot_focal_spot=False,
    scale=1.0,
    anti_aliasing="ssaa",
    colorbar_title=None,
    box_color="#b0b0b0",
    box_opacity=0.2,
    contour_levels=11,
    camera_position=None,
    camera_elevation=None,
    camera_azimuth=None,
    **kwargs,
):
    """Plot a 3D pressure field as a PyVista volume with bounding box and axes.

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
    show_fig : bool, optional
        If True, call ``plotter.show()`` to display the figure. Default True.
    save_path : str or Path, optional
        Path to save a screenshot. If None, no file is written. Default None.
    scalars : str, optional
        Name of the scalar array attached to the volume. Default ``"Pressure"``.
    plotter : pyvista.Plotter, optional
        Existing plotter to draw into. If None, a new plotter is created.
    off_screen : bool, optional
        Render off-screen (no window). Default False.
    window_size : list of int, optional
        Window size ``[width, height]`` in pixels before ``scale``.
        Default ``[520, 720]``.
    notebook : bool, optional
        Use notebook-mode rendering. Default False.
    plot_focal_spot : bool, optional
        Draw the focal spot as an isosurface. Default False.
    scale : float, optional
        Resolution scale factor applied to window size and fonts. Default 1.0.
    anti_aliasing : str, optional
        PyVista anti-aliasing mode. Default ``"ssaa"``.
    colorbar_title : str, optional
        Colorbar title. If None, the scalar name is used.
    box_color : str, optional
        Bounding-box colour. Default ``"#b0b0b0"``.
    box_opacity : float, optional
        Bounding-box opacity. Default 0.2.
    contour_levels : int, optional
        Number of isosurface contour levels. Default 11.
    camera_position : str or list, optional
        PyVista camera position. If None, the default view is used.
    camera_elevation : float, optional
        Camera elevation angle in degrees. If None, the default is used.
    camera_azimuth : float, optional
        Camera azimuth angle in degrees. If None, the default is used.
    **kwargs
        Forwarded to ``plotter.add_mesh()``.

    Returns
    -------
    pyvista.Plotter
        The configured plotter.
    """
    # --- unpack coords dict ---
    if coords is not None:
        x = coords.get("x", x)
        y = coords.get("y", y)
        z = coords.get("z", z)

    pv.global_theme.anti_aliasing = anti_aliasing

    if save_path is not None:
        from pathlib import Path

        file_path = Path(save_path) / file_name
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        scale = 3  # override default scale for high-res screenshots
        off_screen = True  # render off-screen for saving

    # Create the pressure volume mesh
    if db_scale:
        pressure_field = pf.to_dB(pressure_field)
    pressure_vol = create_3Dvol_mesh(x, y, z, pressure_field, scalars=scalars)
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

    plotter.camera_position = camera_position
    plotter.camera.elevation = camera_elevation
    plotter.camera.azimuth = camera_azimuth

    if save_path is not None:
        from pathlib import Path

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plotter.screenshot(str(save_path), transparent_background=True)
        print(f"\nPlot saved to: {save_path}")

    elif show_fig:
        plotter.show()

    return plotter


def plot3D_pressure_slices(
    pressure_field,
    x=None,
    y=None,
    z=None,
    *,
    db_scale=False,
    coords=None,
    show_fig=True,
    save_path=None,
    scalars="Pressure",
    plotter=None,
    off_screen=False,
    window_size=[520, 720],
    notebook=False,
    center_to_max=False,
    scale=1.0,
    anti_aliasing="ssaa",
    colorbar_title=None,
    camera_position=None,
    camera_elevation=None,
    camera_azimuth=None,
    **kwargs,
):
    """Plot orthogonal XZ/XY/YZ slices of a 3D pressure field using PyVista.

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
    show_fig : bool, optional
        If True, call ``plotter.show()`` to display the figure. Default True.
    save_path : str or Path, optional
        Path to save a screenshot. If None, no file is written. Default None.
    scalars : str, optional
        Name of the scalar array attached to the volume. Default ``"Pressure"``.
    plotter : pyvista.Plotter, optional
        Existing plotter to draw into. If None, a new plotter is created.
    off_screen : bool, optional
        Render off-screen (no window). Default False.
    window_size : list of int, optional
        Window size ``[width, height]`` in pixels before ``scale``.
        Default ``[520, 720]``.
    notebook : bool, optional
        Use notebook-mode rendering. Default False.
    center_to_max : bool, optional
        If True, centre slices on the pressure maximum; otherwise on the
        geometric centre. Default False.
    scale : float, optional
        Resolution scale factor applied to window size and fonts. Default 1.0.
    anti_aliasing : str, optional
        PyVista anti-aliasing mode. Default ``"ssaa"``.
    colorbar_title : str, optional
        Colorbar title. If None, the scalar name is used.
    camera_position : str or list, optional
        PyVista camera position. If None, the default view is used.
    camera_elevation : float, optional
        Camera elevation angle in degrees. If None, the default is used.
    camera_azimuth : float, optional
        Camera azimuth angle in degrees. If None, the default is used.
    **kwargs
        Forwarded to ``plotter.add_mesh()``.

    Returns
    -------
    pyvista.Plotter
        The configured plotter.
    """

    # --- unpack coords dict ---
    if coords is not None:
        x = coords.get("x", x)
        y = coords.get("y", y)
        z = coords.get("z", z)

    pv.global_theme.anti_aliasing = anti_aliasing

    if save_path is not None:
        from pathlib import Path

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        scale = 3  # override default scale for high-res screenshots
        off_screen = True  # render off-screen for saving

    nx, ny, nz = pressure_field.shape

    if x is None or y is None or z is None:
        x = np.arange(nx)
        y = np.arange(ny)
        z = np.arange(nz)

    if center_to_max:
        # Look for the y, x, z indices that are closest to the max value
        max_idx = np.unravel_index(np.nanargmax(pressure_field), pressure_field.shape)
        y0, x0, z0 = max_idx[1], max_idx[0], max_idx[2]
    else:
        # Use the middle indices
        y0 = int(np.floor(y.shape[0] / 2))
        x0 = int(np.floor(x.shape[0] / 2))
        z0 = int(np.floor(z.shape[0] / 2))
    print(
        f"Taking slice ({x[x0]:.1e},{y[y0]:.1e},{z[z0]:.1e}) mm => x_ind, y_ind, z_ind = {x0 + 1}/{x.shape[0]}, {y0 + 1}/{y.shape[0]}, {z0 + 1}/{z.shape[0]}"
    )

    extent_xz = [x[0], x[-1], z[0], z[-1]]
    extent_xy = [x[0], x[-1], y[0], y[-1]]
    extent_yz = [y[0], y[-1], z[0], z[-1]]
    slice_xz = pressure_field[:, y0, :]
    slice_xy = pressure_field[:, :, z0]
    slice_yz = pressure_field[x0, :, :]
    yz_offset = {"x": x[x0]}
    xy_offset = {"z": z[z0]}
    xz_offset = {"y": y[y0]}

    slice_xz = pf.to_dB(slice_xz) if db_scale else slice_xz
    slice_xy = pf.to_dB(slice_xy) if db_scale else slice_xy
    slice_yz = pf.to_dB(slice_yz) if db_scale else slice_yz

    slice_xz_mesh = create_2Dimage_mesh(
        slice_xz,
        extent=extent_xz,
        plane_offset=xz_offset,
        scalars=scalars,
    )
    slice_xy_mesh = create_2Dimage_mesh(
        slice_xy,
        extent=extent_xy,
        plane_offset=xy_offset,
        scalars=scalars,
    )
    slice_yz_mesh = create_2Dimage_mesh(
        slice_yz,
        extent=extent_yz,
        plane_offset=yz_offset,
        scalars=scalars,
    )

    plotter = add_2D_image(
        slice_xz_mesh,
        plotter=plotter,
        window_size=_normalize_window_size(window_size, scale=scale),
        notebook=notebook,
        off_screen=off_screen,
        scale=scale,
        colorbar_title=colorbar_title,
        **kwargs,
    )
    plotter = add_2D_image(
        slice_xy_mesh,
        plotter=plotter,
        scale=scale,
        **kwargs,
    )
    plotter = add_2D_image(
        slice_yz_mesh,
        plotter=plotter,
        scale=scale,
        **kwargs,
    )

    kwargs.pop("ambient", 0.7)  # set default
    plotter.camera_position = camera_position
    plotter.camera.elevation = camera_elevation
    plotter.camera.azimuth = camera_azimuth

    if save_path is not None:
        from pathlib import Path

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plotter.screenshot(str(save_path), transparent_background=True)
        print(f"\nPlot saved to: {save_path}")

    elif show_fig:
        plotter.show()

    return plotter


def plot3D_transient_slices(
    pressure_field,
    x=None,
    y=None,
    z=None,
    *,
    coords=None,
    time_array=None,
    center_mm=None,
    center_to_max=False,
    show_fig=True,
    db_scale=False,
    save_path=None,
    file_name="3D_pressure_slices.mp4",
    video_duration_s=5,
    fps=30,
    scalars="Pressure",
    plotter=None,
    off_screen=False,
    window_size=[520, 720],
    notebook=False,
    scale=1.0,
    anti_aliasing="ssaa",
    colorbar_title=None,
    camera_position=None,
    camera_elevation=None,
    camera_azimuth=None,
    vmin=None,
    vmax=None,
    **kwargs,
):
    """Plot orthogonal pressure slices of transient data with a PyVista time
    slider or video export.

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
    time_array : numpy.ndarray, optional
        Physical time values (length Nt). If None, frame indices are used.
    center_mm : tuple of float, optional
        ``(x0, y0, z0)`` in mm.  For volume input this selects the slice
        position; for planes input it sets the 3-D plane offsets.
        Default: geometric centre (volume) or coordinate midpoints (planes).
    center_to_max : bool, optional
        If True and input is a volume, slice through the global pressure
        maximum instead of `center_mm`. Default False.
    show_fig : bool, optional
        Call ``plotter.show()``. Default True.
    save_path : str or Path, optional
        ``.gif`` or ``.mp4`` path for video export. If None, interactive
        slider is shown.
    video_duration_s : float, optional
        Target video duration in seconds. Default 5.
    fps : int, optional
        Video frame rate. Default 30.
    scalars : str, optional
        Scalar array name on each mesh. Default ``"Pressure"``.
    plotter : pyvista.Plotter, optional
        Existing plotter (e.g. with a transducer mesh already added).
    off_screen : bool, optional
        Render off-screen. Default False.
    window_size : list of int, optional
        ``[width, height]`` before *scale*. Default ``[520, 720]``.
    notebook : bool, optional
        Notebook rendering mode. Default False.
    scale : float, optional
        Resolution scale factor. Default 1.0.
    anti_aliasing : str, optional
        PyVista anti-aliasing mode. Default ``"ssaa"``.
    colorbar_title : str, optional
        Colorbar title. Default: *scalars* name.
    camera_position, camera_elevation, camera_azimuth
        Camera settings forwarded to PyVista.
    **kwargs
        Forwarded to ``plotter.add_mesh()``.

    Returns
    -------
    pyvista.Plotter
        The configured plotter.
    """
    # --- unpack coords dict ---
    if coords is not None:
        x = coords.get("x", x)
        y = coords.get("y", y)
        z = coords.get("z", z)
        if time_array is None and "t0" in coords:
            if isinstance(pressure_field, np.ndarray):
                nt_hint = pressure_field.shape[0]
            else:
                nt_hint = next(iter(pressure_field.values())).shape[0]
            time_array = coords["t0"] + np.arange(nt_hint) * coords["dt"]

    # Plane metadata: axes pair, offset key, index into center_mm
    _PLANE_META = {
        "xz": {"axes": ("x", "z"), "offset_key": "y", "ci": 1},
        "xy": {"axes": ("x", "y"), "offset_key": "z", "ci": 2},
        "yz": {"axes": ("y", "z"), "offset_key": "x", "ci": 0},
    }

    # ------------------------------------------------------------------
    # Resolve input: 4D volume → planes dict
    # ------------------------------------------------------------------
    if isinstance(pressure_field, dict):
        planes = dict(pressure_field)
        valid = {"xz", "xy", "yz"}
        bad = set(planes.keys()) - valid
        if bad or not planes:
            raise ValueError(
                f"Plane keys must be a non-empty subset of {valid}, got {set(planes.keys())}"
            )
        for k, v in planes.items():
            if v.ndim != 3:
                raise ValueError(
                    f"Plane '{k}' must be 3D (Nt, N1, N2), got shape {v.shape}"
                )
        nt = next(iter(planes.values())).shape[0]

        # Default coords from plane shapes
        _src = {
            "x": [("xz", 1), ("xy", 1)],
            "y": [("xy", 2), ("yz", 1)],
            "z": [("xz", 2), ("yz", 2)],
        }
        _coords = {"x": x, "y": y, "z": z}
        for cname, sources in _src.items():
            if _coords[cname] is None:
                for pk, ax in sources:
                    if pk in planes:
                        _coords[cname] = np.arange(planes[pk].shape[ax], dtype=float)
                        break
                else:
                    _coords[cname] = np.array([0.0])
        x, y, z = _coords["x"], _coords["y"], _coords["z"]

        if center_mm is None:
            center_mm = (
                float(x[len(x) // 2]),
                float(y[len(y) // 2]),
                float(z[len(z) // 2]),
            )

    elif isinstance(pressure_field, np.ndarray) and pressure_field.ndim == 4:
        nt, nx, ny, nz = pressure_field.shape
        if x is None:
            x = np.arange(nx, dtype=float)
        if y is None:
            y = np.arange(ny, dtype=float)
        if z is None:
            z = np.arange(nz, dtype=float)

        # Determine slice indices
        if center_to_max:
            idx = np.unravel_index(
                np.nanargmax(np.abs(pressure_field)), pressure_field.shape
            )
            _, xi, yi, zi = idx
        elif center_mm is not None:
            xi = int(np.argmin(np.abs(x - center_mm[0])))
            yi = int(np.argmin(np.abs(y - center_mm[1])))
            zi = int(np.argmin(np.abs(z - center_mm[2])))
        else:
            xi, yi, zi = nx // 2, ny // 2, nz // 2

        center_mm = (float(x[xi]), float(y[yi]), float(z[zi]))

        # Only create planes for non-degenerate dimensions
        planes = {}
        if nx > 1 and nz > 1:
            planes["xz"] = pressure_field[:, :, yi, :]
        if nx > 1 and ny > 1:
            planes["xy"] = pressure_field[:, :, :, zi]
        if ny > 1 and nz > 1:
            planes["yz"] = pressure_field[:, xi, :, :]
        if not planes:
            raise ValueError("No non-degenerate 2D slices in the given 4D field.")
    else:
        raise ValueError(
            "pressure_field must be a 4D ndarray (Nt,Nx,Ny,Nz) or a dict of "
            "planes with keys from {'xz', 'xy', 'yz'}."
        )

    coords = {"x": x, "y": y, "z": z}
    plane_order = [k for k in ("xz", "xy", "yz") if k in planes]

    # Truncate to the minimum common frame count across planes
    min_nt = min(v.shape[0] for v in planes.values())
    if min_nt < nt:
        planes = {k: v[:min_nt] for k, v in planes.items()}
        nt = min_nt

    print(
        f"Transient 3D slices: {len(plane_order)} planes, {nt} frames, "
        f"center=({center_mm[0]:.2f}, {center_mm[1]:.2f}, {center_mm[2]:.2f}) mm"
    )

    # ------------------------------------------------------------------
    # Global color range (fixed across all frames)
    # ------------------------------------------------------------------
    if db_scale:
        for k, v in planes.items():
            planes[k] = pf.to_dB(v)
        vmax = 0 if vmax is None else vmax
        vmin = -40 if vmin is None else vmin
    else:
        vmax = (
            max(float(np.nanmax(np.abs(v))) for v in planes.values())
            if vmax is None
            else vmax
        )
        vmin = 0 if vmin is None else vmin

    clim = [vmin, vmax]

    # ------------------------------------------------------------------
    # Time label
    # ------------------------------------------------------------------
    if time_array is None:
        time_label = "Frame"
        time_values = np.arange(nt, dtype=float)
    elif np.max(time_array) < 1e-3:
        time_label = "Time (µs)"
        time_values = np.asarray(time_array) * 1e6
    else:
        time_label = "Time (s)"
        time_values = np.asarray(time_array, dtype=float)

    def _format_time(t_idx):
        return f"{time_label}: {time_values[t_idx]:.2f}"

    # ------------------------------------------------------------------
    # Create PyVista meshes (frame 0)
    # ------------------------------------------------------------------
    meshes = {}
    for key in plane_order:
        meta = _PLANE_META[key]
        c1, c2 = coords[meta["axes"][0]], coords[meta["axes"][1]]
        extent = [c1[0], c1[-1], c2[0], c2[-1]]
        offset = {meta["offset_key"]: center_mm[meta["ci"]]}
        meshes[key] = create_2Dimage_mesh(
            planes[key][0], extent=extent, plane_offset=offset, scalars=scalars
        )

    # ------------------------------------------------------------------
    # Plotter setup
    # ------------------------------------------------------------------
    pv.global_theme.anti_aliasing = anti_aliasing

    if save_path is not None:
        save_path = str(save_path + file_name)
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        off_screen = True

    kwargs.setdefault("clim", clim)

    for i, key in enumerate(plane_order):
        if i == 0:
            plotter = add_2D_image(
                meshes[key],
                plotter=plotter,
                window_size=_normalize_window_size(window_size, scale=scale),
                notebook=notebook,
                off_screen=off_screen,
                scale=scale,
                colorbar_title=colorbar_title,
                **kwargs,
            )
        else:
            plotter = add_2D_image(
                meshes[key],
                plotter=plotter,
                scale=scale,
                show_scalar_bar=False,
                **kwargs,
            )

    _set_custom_style(plotter, scale=scale)

    if camera_position is not None:
        plotter.camera_position = camera_position
    if camera_elevation is not None:
        plotter.camera.elevation = camera_elevation
    if camera_azimuth is not None:
        plotter.camera.azimuth = camera_azimuth

    # ------------------------------------------------------------------
    # Time overlay + update callback
    # ------------------------------------------------------------------
    text_actor = plotter.add_text(
        _format_time(0), position="upper_right", font_size=int(12 * scale)
    )

    def _update_time(value):
        t_idx = int(np.clip(round(value), 0, nt - 1))
        for key in plane_order:
            meshes[key][scalars] = (
                planes[key][t_idx].ravel(order="F").astype(np.float32)
            )
        text_actor.SetText(3, _format_time(t_idx))

    # ------------------------------------------------------------------
    # Interactive slider or video export
    # ------------------------------------------------------------------
    if save_path is None:
        plotter.add_slider_widget(
            _update_time,
            rng=[0, nt - 1],
            value=0,
            title=time_label,
            pointa=(0.2, 0.93),
            pointb=(0.8, 0.93),
            style="modern",
        )
        if show_fig:
            plotter.show()
    else:
        total_frames = int(video_duration_s * fps)
        step = max(1.0, nt / total_frames)
        frame_indices = np.unique(np.arange(0, nt, step).astype(int))

        if save_path.endswith(".gif"):
            plotter.open_gif(save_path)
        else:
            plotter.open_movie(save_path, framerate=fps)

        for t_idx in frame_indices:
            _update_time(t_idx)
            plotter.write_frame()

        plotter.close()
        print(f"\nVideo saved to: {save_path}")

    return plotter
