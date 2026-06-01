"""3D pressure field visualization using PyVista."""

from pathlib import Path

import numpy as np
import pyvista as pv

import pyfield as pf

from .plane_utils import AXIS_IDX, PLANE_META, compute_plane_extents, parse_planes
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
    pressure_field : (Nx, Ny, Nz) numpy.ndarray
        Pressure field samples on the grid.
    x : (Nx,) numpy.ndarray, optional
        Lateral coordinates (mm).
    y : (Ny,) numpy.ndarray, optional
        Elevation coordinates (mm).
    z : (Nz,) numpy.ndarray, optional
        Axial coordinates (mm).
    db_scale : bool, optional
        Convert to dB before display. Default False.
    coords : dict, optional
        Coordinate dict with keys ``"x"``, ``"y"``, ``"z"``.  Overrides
        individual *x*, *y*, *z* when provided.
    show_fig : bool, optional
        If True, call ``plotter.show()`` to display the figure. Default True.
    save_path : str or Path, optional
        Path to save a screenshot. If None, no file is written. Default None.
    file_name : str, optional
        File name for saved video/screenshot. Default
        ``"3D_pressure_slices.mp4"``.
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
    pressure_field : (Nx, Ny, Nz) numpy.ndarray
        Pressure field samples on the grid.
    x : (Nx,) numpy.ndarray, optional
        Lateral coordinates (mm).
    y : (Ny,) numpy.ndarray, optional
        Elevation coordinates (mm).
    z : (Nz,) numpy.ndarray, optional
        Axial coordinates (mm).
    db_scale : bool, optional
        Convert to dB before display. Default False.
    coords : dict, optional
        Coordinate dict with keys ``"x"``, ``"y"``, ``"z"``.  Overrides
        individual *x*, *y*, *z* when provided.
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
    file_name=None,
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
    theme="white",
    show_grid_kwargs=None,
    **kwargs,
):
    """Plot transient pressure slices with PyVista time slider or video.

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
        ``"t0"``, ``"dt"``).  Overrides *x*, *y*, *z* when provided.
    time_array : numpy.ndarray, optional
        Physical time values (length Nt). If None, frame indices are used.
    center_mm : tuple of float, optional
        ``(x0, y0, z0)`` in mm.  For volume input this selects the slice
        position; for planes input it sets the 3-D plane offsets.
        Default: geometric centre (volume) or coordinate midpoints (planes).
    center_to_max : bool, optional
        If True and input is a volume, slice through the global pressure
        maximum instead of *center_mm*. Default False.
    show_fig : bool, optional
        Call ``plotter.show()``. Default True.
    db_scale : bool, optional
        Convert to dB before display. Default False.
    save_path : str or Path, optional
        Directory for video export. If None, interactive slider is shown.
    file_name : str, optional
        Video file name with extension (e.g. ``"slices.mp4"``, ``"slices.gif"``).
        Default: ``"3D_pressure_slices.mp4"`` if *save_path* is not None.
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
    camera_position : str or list, optional
        PyVista camera position. Default: automatic.
    camera_elevation : float, optional
        Camera elevation angle in degrees. Default: automatic.
    camera_azimuth : float, optional
        Camera azimuth angle in degrees. Default: automatic.
    vmin : float, optional
        Minimum scalar value for the colour map.
    vmax : float, optional
        Maximum scalar value for the colour map.
    theme : str, optional
        PyVista colour theme. Default ``"white"``.
    show_grid_kwargs : dict, optional
        Keyword arguments forwarded to ``plotter.show_grid()``.
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
    if file_name is None:
        file_name = "3D_pressure_slices.mp4" if save_path is not None else None
    # -----------------------------------------------------------------
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

    plane_order = [ps.name for ps in plane_specs]
    planes = {ps.name: ps.data for ps in plane_specs}
    nt = next(iter(planes.values())).shape[0]

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
    plane_spec_map = {ps.name: ps for ps in plane_specs}
    for key in plane_order:
        meta = PLANE_META[key]
        ps = plane_spec_map[key]
        ext = ps.extent  # (c1_min, c1_max, c2_min, c2_max)
        extent = [ext[0], ext[1], ext[2], ext[3]]
        normal_val = ps.translation[AXIS_IDX[meta["normal"]]]
        offset = {meta["normal"]: normal_val}
        meshes[key] = create_2Dimage_mesh(
            planes[key][0], extent=extent, plane_offset=offset, scalars=scalars
        )

    # ------------------------------------------------------------------
    # Plotter setup
    # ------------------------------------------------------------------
    pv.global_theme.anti_aliasing = anti_aliasing
    if theme == "dark":
        font_color = "white"
        background_color = "black"
    elif theme == "white":
        font_color = "black"
        background_color = "white"
    else:
        raise ValueError(f"Unsupported theme: {theme}")

    pv.global_theme.background = background_color

    if save_path is not None:
        video_path = str(Path(save_path) / file_name)  # ty: ignore[unsupported-operator]
        Path(video_path).parent.mkdir(parents=True, exist_ok=True)
        off_screen = True

    kwargs.setdefault("clim", clim)
    kwargs.setdefault("cmap", "jet")

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

    plotter.camera.up = (0, 0, -1)  # ty: ignore[unresolved-attribute]
    if camera_position is not None:
        plotter.camera_position = camera_position  # ty: ignore[invalid-assignment]
    if camera_elevation is not None:
        plotter.camera.elevation = camera_elevation  # ty: ignore[unresolved-attribute]
    if camera_azimuth is not None:
        plotter.camera.azimuth = camera_azimuth  # ty: ignore[unresolved-attribute]

    default_show_grid_kwargs = {
        "grid": "back",
        "font_size": 12 * scale,
        "color": font_color,
        "location": "outer",
        "xtitle": "X (mm)",
        "ytitle": "Y (mm)",
        "ztitle": "Z (mm)",
        "n_xlabels": 5,
        "n_ylabels": 5,
        "n_zlabels": 6,
        "use_3d_text": False,
    }

    if show_grid_kwargs is not None:
        default_show_grid_kwargs.update(show_grid_kwargs)

    plotter.show_grid(**default_show_grid_kwargs)  # ty: ignore[unresolved-attribute]

    # ------------------------------------------------------------------
    # Time overlay + update callback
    # ------------------------------------------------------------------
    text_actor = plotter.add_text(  # ty: ignore[unresolved-attribute]
        _format_time(0),
        position="upper_right",
        font_size=int(12 * scale),
        color=font_color,
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
    assert plotter is not None
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

        if video_path.endswith(".gif"):
            plotter.open_gif(video_path)
        else:
            plotter.open_movie(video_path, framerate=fps)

        for t_idx in frame_indices:
            _update_time(t_idx)
            plotter.write_frame()

        plotter.close()
        print(f"\nVideo saved to: {video_path}")

    return plotter
