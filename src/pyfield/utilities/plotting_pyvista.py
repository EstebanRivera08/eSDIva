import numpy as np
import pyvista as pv


# -------------------- Plotting Functions --------------------
def create_vol_mesh(x, y, z, vol_matrix, *, scalars="Values"):
    """
    Compute the volume mesh for the given vol_matrix and coordinates.

    Parameters
    ----------
    x, y, z : ndarray
        Coordinate arrays.
    vol_matrix : ndarray
        Volume data (dim = 3).

    Returns
    -------
    pressure_vol : pyvista.UniformGrid
        The pressure volume mesh.
    """
    dx = x[1] - x[0] if len(x) > 1 else 1e-6
    dy = y[1] - y[0] if len(y) > 1 else 1e-6
    dz = z[1] - z[0] if len(z) > 1 else 1e-6

    nx, ny, nz = vol_matrix.shape

    # Create the 3D UniformGrid
    pressure_vol = pv.ImageData(
        dimensions=(nx, ny, nz),
        spacing=(dx, dy, dz),
        origin=(x.min(), y.min(), z.min()),
    )

    # Attach pressure data to the grid
    pressure_vol.point_data[scalars] = vol_matrix.ravel(
        order="F"
    )  # VERY important: Fortran order
    return pressure_vol


# ------------- Brain Regions Mesh -------------
def add_regions_mesh(
    pv_regions_dict,
    *,
    plotter=None,
    window_size=[800, 800],
    notebook=False,
    off_screen=False,
    kwargs_dict=None,
    **kwargs,
):
    """
    Plot the PyVista mesh of a specified BrainAtlas structure.

    Parameters:
    -----------
        pv_regions_dict (dict): A dictionary of PyVista meshes for different brain regions.
        window_size (list, optional): Size of the plot window. Default is [800, 800].
        notebook (bool, optional): Whether to use notebook mode for the plotter. Default is True.
        off_screen (bool, optional): Whether to render the plot off-screen. Default is False.
        kwargs_dict (dict, optional): Additional keyword arguments for the mesh rendering.
    Returns:
    --------
        pv.Plotter: The PyVista plotter with the mesh added.
    """
    if plotter is None:
        plotter = pv.Plotter(
            notebook=notebook, window_size=window_size, off_screen=off_screen
        )
    default_kwargs = {
        "color": "lightgrey",  # Default color for the mesh
        "opacity": 0.4,  # Default opacity for the mesh
    }

    if isinstance(pv_regions_dict, dict):
        for region in pv_regions_dict.keys():
            if kwargs_dict is not None:
                if region not in kwargs_dict.keys():
                    kwargs = default_kwargs
                else:
                    kwargs = kwargs_dict[region]
                kwargs["label"] = region  # if region != "root" else "Brain"
            elif isinstance(kwargs_dict, dict) and "default" in kwargs_dict.keys():
                kwargs = default_kwargs
            else:
                kwargs = default_kwargs
            plotter.add_mesh(pv_regions_dict[region], **kwargs)
    else:
        try:
            for key, value in default_kwargs.items():
                if key not in kwargs:
                    kwargs[key] = value
            plotter.add_mesh(pv_regions_dict, **kwargs)
        except KeyError as e:
            raise ValueError(
                f"Error: {e}. pv_regions_dict should be a dictionary with brain region meshes or the mesh."
            )

    plotter.add_axes()
    return plotter


# ------------- doppler3D_vol Mesh -------------


def add_3D_vol(
    vol_3D,
    *,
    plotter=None,
    notebook=False,
    window_size=[700, 700],
    off_screen=False,
    scale=1,
    colorbar_title=None,
    **kwargs,
):
    """
    Add PyVista objects to a plotter for visualizing.
    Args:
        vol_3D (pv.ImageData): The volumen data (e.g. 3D ultrasound scan).
        plotter (pv.Plotter, optional): An existing PyVista plotter to add the
            volume to. If None, a new plotter will be created. Default is None.
        notebook (bool, optional): Whether to use notebook mode for the plotter. Default
        is True.
        window_size (list, optional): Size of the plot window. Default is [700, 700].
        off_screen (bool, optional): Whether to render the plot off-screen. Default is False
            scale (float, optional): Scaling factor for font sizes in the scalar bar.
            Default is 1.
        kwargs: Additional keyword arguments to pass to the add_volume method of the
        plotter.
    Returns:
        pv.Plotter: The PyVista plotter with the volume added.
    """
    if plotter is None:
        plotter = pv.Plotter(
            window_size=window_size, notebook=notebook, off_screen=off_screen
        )

    scalars = vol_3D.point_data.keys()[0]  # Get the name of the first scalar

    cb_title = colorbar_title if colorbar_title is not None else scalars
    default_kwargs = {
        "scalars": scalars,
        "cmap": "hot",
        "opacity": "sigmoid",
        "mapper": "smart",
        "show_scalar_bar": True,
        "ambient": 0.3,
        "scalar_bar_args": {
            "title": cb_title,
            "title_font_size": int(20 * scale),
            "label_font_size": int(18 * scale),
            "vertical": False,
            "position_x": 0.1,
            "position_y": 0.1,
            "height": 0.3,
        },
    }
    for key, value in default_kwargs.items():
        if key not in kwargs:
            kwargs[key] = value

    vol = plotter.add_volume(vol_3D, **kwargs)
    vol.prop.interpolation_type = "linear"
    plotter.add_axes()
    return plotter


# ------------- doppler2D_image Mesh -------------


def add_2D_image(
    image_grid,
    *,
    plotter=None,
    notebook=False,
    window_size=[700, 700],
    off_screen=False,
    scale=1,
    colorbar_title=None,
    **kwargs,
):
    """
    Add a 2D image as a mesh to a PyVista plotter.

    Parameters
    ----------
    image_grid : pv.ImageData
        The 2D image data to be added as a mesh (e.g. 2D ultrasound image).
    plotter : pv.Plotter, optional
        An existing PyVista plotter to which the image will be added. If None, a new
        plotter will be created. Default is None.
    notebook : bool, optional
        Whether to use notebook mode for the plotter. Default is False.
    window_size : list, optional
        Size of the plot window. Default is [700, 700].
    off_screen : bool, optional
        Whether to render the plot off-screen. Default is False.
    scale : float, optional
        Scaling factor for font sizes in the scalar bar. Default is 1.
    kwargs: Additional keyword arguments to pass to the add_mesh method of the plotter.

    Returns
    -------
    pv.Plotter
        The PyVista plotter with the 2D image mesh added.

    """
    if plotter is None:
        plotter = pv.Plotter(
            window_size=window_size, notebook=notebook, off_screen=off_screen
        )

    scalars = image_grid.point_data.keys()[0]
    cb_title = colorbar_title if colorbar_title is not None else scalars
    default_kwargs = {
        "show_edges": False,
        "cmap": "gray",
        "opacity": 1.0,
        "name": "2D doppler",  # Name for the volume
        "show_scalar_bar": True,
        "scalar_bar_args": {
            "title": cb_title,
            "title_font_size": int(20 * scale),
            "label_font_size": int(18 * scale),
            "vertical": True,
            "position_x": 0.1,
            "position_y": 0.2,
            "height": 0.3,
        },
    }
    for key, value in default_kwargs.items():
        if key not in kwargs:
            kwargs[key] = value

    vol = plotter.add_mesh(image_grid, **kwargs)
    plotter.add_axes()
    return plotter


# ------------- Pressure field Mesh -------------


def add_pressure_vol(
    pressure_vol,
    *,
    plotter=None,
    window_size=[800, 800],
    notebook=False,
    plot_focal_spot=False,
    off_screen=False,
    colorbar_title=None,
    contour_levels=11,
    scale=1,
    vmin=None,
    vmax=None,
    **kwargs,
):
    """
    Add a pressure volume mesh to a PyVista plotter.

    Parameters
    ----------
    pressure_vol : pv.ImageData
        The pressure volume data to be added as a mesh.
    plotter : pv.Plotter, optional
        An existing PyVista plotter to which the pressure volume will be added. If None,
        a new plotter will be created. Default is None.
    notebook : bool, optional
        Whether to use notebook mode for the plotter. Default is False.
    window_size : list, optional
        Size of the plot window. Default is [800, 800].
    plot_focal_spot : bool, optional
        Whether to plot the focal spot as an isosurface. Default is False.
    off_screen : bool, optional
        Whether to render the plot off-screen. Default is False.
    colorbar_title : str, optional
        Title for the colorbar. If None, it will use the name of the scalar field in
        pressure_vol. Default is None.
    contour_levels : int, optional
        Number of contour levels to use when plotting the pressure volume. Default is
        11.
    scale : float, optional
        Scaling factor for font sizes in the scalar bar. Default is 1.
    vmin : float, optional
        Minimum value for the contour levels. If None, it will use the minimum value in
        the pressure volume. Default is None.
    vmax : float, optional
        Maximum value for the contour levels. If None, it will use the maximum value in
        the pressure volume. Default is None.
    kwargs: Additional keyword arguments to pass to the add_mesh method of the plotter.

    Returns
    -------
    pv.Plotter
        The PyVista plotter with the pressure volume mesh added.

    """

    if plotter is None:
        plotter = pv.Plotter(
            notebook=notebook, window_size=window_size, off_screen=off_screen
        )

    scalars = pressure_vol.point_data.keys()[0]  # Get the name of the first scalar

    if scalars != "Pressure":
        print(
            f"Warning: The scalar field in the pressure volume is named '{scalars}' instead of 'Pressure'. Proceeding with '{scalars}'."
        )

    if colorbar_title is None:
        colorbar_title = f"{scalars}"

    # 3) Add the pressure volume
    if plot_focal_spot:
        default_kwargs = {
            "opacity": 1,
            "name": colorbar_title,
            "show_scalar_bar": False,
            "label": "Focal Spot",  # label for the legend
            "color": "r",  # color of the mesh
            "ambient": 0.3,
        }
        for key, value in default_kwargs.items():
            if key not in kwargs:
                kwargs[key] = value

        # pick a threshold, e.g. halfway to the max
        threshold = 0.7 * pressure_vol[scalars].max()
        iso_mesh = pressure_vol.contour(
            [threshold], scalars=scalars
        )  # Create isosurface at threshold# add that instead of (or in addition to) the volume
        plotter.add_mesh(iso_mesh, **kwargs)

    else:
        n_contours = contour_levels
        if vmin is not None:
            min_val = vmin
            # print(f"Using provided vmin: {vmin}")
        else:
            min_val = pressure_vol[scalars].min()
        if vmax is not None:
            max_val = vmax
            # print(f"Using provided vmax: {vmax}")
        else:
            max_val = pressure_vol[scalars].max()
        levels = np.linspace(min_val, max_val, n_contours)
        iso_mesh = pressure_vol.contour(
            isosurfaces=levels, scalars=scalars
        )  # Create isosurface at threshold
        default_kwargs = {
            "scalars": scalars,  # use the scalar to color surfaces
            "opacity": "linear",
            "cmap": "jet",
            "show_scalar_bar": True,
            "scalar_bar_args": {
                "title": colorbar_title,
                "title_font_size": int(20 * scale),
                "label_font_size": int(18 * scale),
                "vertical": True,
                "position_x": 0.8,
                "position_y": 0.1,
                "height": 0.3,
            },
            "label": scalars,  # label for the legend
            "color": "r",  # color of the mesh,
            "ambient": 0.3,
        }
        for key, value in default_kwargs.items():
            if key not in kwargs:
                kwargs[key] = value

        plotter.add_mesh(iso_mesh, **kwargs)

    plotter.add_axes()
    return plotter


# ------------- Transducer Mesh -------------


def add_transducer_mesh(
    TX_mesh,
    *,
    plotter=None,
    window_size=[800, 800],
    notebook=False,
    off_screen=False,
    scale=1,
    scalars="Apodization",
    colorbar_title=None,
    **kwargs,
):
    """
    Add a transducer mesh to a PyVista plotter, colored by either apodization or delays.

    Parameters
    ----------
    TX_mesh : pv.PolyData
        The mesh representing the transducer, with point data for apodization and delays.
    plotter : pv.Plotter, optional
        An existing PyVista plotter to which the transducer mesh will be added. If None
        a new plotter will be created. Default is None.
    window_size : list, optional
        Size of the plot window. Default is [800, 800].
    notebook : bool, optional
        Whether to use notebook mode for the plotter. Default is False.
    off_screen : bool, optional
        Whether to render the plot off-screen. Default is False.
    scale : float, optional
        Scaling factor for font sizes in the scalar bar. Default is 1.
    scalars : str, optional
        Which scalar field to use for coloring the transducer mesh. Must be either
        "Apodization" or "Delays". Default is "Apodization".
    colorbar_title : str, optional
        Title for the colorbar. If None, it will use "Apodization" or "Delays" based on
        the scalars parameter. Default is None.
    kwargs: Additional keyword arguments to pass to the add_mesh method of the plotter.

    Returns
    -------
    pv.Plotter
        The PyVista plotter with the transducer mesh added and colored by the specified
        scalar field.
    """

    if plotter is None:
        plotter = pv.Plotter(
            notebook=notebook, window_size=window_size, off_screen=off_screen
        )
    if scalars == "Apodization":
        title_name = "Apodization"
        cmap = "cool"
    elif scalars == "Delays":
        title_name = "Delays (s)"
        cmap = "rainbow"
    else:
        raise ValueError("Scalars must be 'Apodization' or 'Delays'")

    if colorbar_title is not None:
        title_name = colorbar_title

    default_kwargs = {
        "scalars": scalars,
        "cmap": cmap,
        "clim": [0, 1] if scalars == "Apodization" else None,
        "show_scalar_bar": True,
        "scalar_bar_args": {
            "title": title_name,
            "title_font_size": int(20 * scale),
            "label_font_size": int(18 * scale),
            "vertical": True,
            "position_x": 0.8,
            "position_y": 0.5,
            "height": 0.3,
        },
        "opacity": 1.0,
        "show_edges": True,
        "ambient": 1,
    }

    for key, value in default_kwargs.items():
        if key not in kwargs:
            kwargs[key] = value

    plotter.add_mesh(
        TX_mesh,  # Convert to mm for visualization
        **kwargs,
    )
    plotter.add_axes()
    return plotter


# ------------- Plot markers spheres ------


def add_markers(
    points,
    *,
    plotter=None,
    notebook=False,
    window_size=(700, 700),
    off_screen=False,
    point_size=1,
    labels=None,
    label_offset=(0, 0, 0),
    label_font_size=12,
    **kwargs,
):
    """
    Add marker points (and optional labels) to a pyvista Plotter.

    - Consumes 'color' and 'glyph_scale' explicitly.
    - Does not forward arbitrary kwargs that pyvista.Property will treat as color.
    """
    import pyvista as pv

    # prepare plotter
    if plotter is None:
        plotter = pv.Plotter(
            window_size=window_size, notebook=notebook, off_screen=off_screen
        )

    pts = pv.PolyData(np.asarray(points))

    # extract and remove plotting-specific keys from kwargs
    color = kwargs.pop("color", "red")

    # prepare mesh kwargs (do not forward color as ambiguous tuple/name)
    mesh_kwargs = dict(kwargs)
    # ensure we don't accidentally pass other high-level plotting dicts as color
    mesh_kwargs.pop("scalar_bar_args", None)

    # add points as spheres (render_points_as_spheres) with a size derived from glyph_scale
    # compute a sensible point_size (integer)

    plotter.add_mesh(
        pts,
        color=color,
        render_points_as_spheres=True,
        point_size=point_size,
        **mesh_kwargs,
    )

    # add optional labels (filter kwargs forwarded to add_point_labels)
    if labels is not None:
        labels = list(labels)
        if len(labels) != pts.n_points:
            raise ValueError(
                f"labels length {len(labels)} != number of points {pts.n_points}"
            )
        # allowed keys for add_point_labels (keep conservative)
        allowed_label_keys = {
            "always_visible",
            "background_color",
            "use_2d",
            "point_size",
            "fill_shape",
            "show_points",
            "name",
            "opacity",
            "fmt",
            "italic",
            "bold",
            "shadow",
        }
        label_kwargs = {k: v for k, v in kwargs.items() if k in allowed_label_keys}
        for idx, txt in enumerate(labels):
            pos = np.array(points[idx]) + np.array(label_offset)
            plotter.add_point_labels(
                np.array([pos]),
                [txt],
                font_size=label_font_size,
                text_color=color,
                **label_kwargs,
            )

    return plotter


# ...existing code...


# ------------ helper functions --------------


def recompute_bounds(plotter):
    """Recompute the bounds of a PyVista plotter based on its current meshes."""
    if not plotter.meshes:
        raise ValueError("The plotter has no meshes to compute bounds from.")

    all_bounds = np.array([mesh.bounds for mesh in plotter.meshes])
    x_min = all_bounds[:, 0].min()
    x_max = all_bounds[:, 1].max()
    y_min = all_bounds[:, 2].min()
    y_max = all_bounds[:, 3].max()
    z_min = all_bounds[:, 4].min()
    z_max = all_bounds[:, 5].max()

    return (x_min, x_max, y_min, y_max, z_min, z_max)
