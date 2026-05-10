import numpy as np
import pyvista as pv

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
    show_fig=True,
    save_path=None,
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
    pv.global_theme.anti_aliasing = anti_aliasing

    if save_path is not None:
        from pathlib import Path

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        scale = 3  # override default scale for high-res screenshots
        off_screen = True  # render off-screen for saving

    # Create the pressure volume mesh
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
