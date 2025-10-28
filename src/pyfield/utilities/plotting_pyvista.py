import numpy as np
import pyvista as pv


# -------------------- Plotting Functions --------------------
def create_vol_mesh(x, y, z, vol_matrix, *, scalars="Values"):
    """
    Compute the pressure volume mesh for the given pressure field and coordinates.

    Parameters
    ----------
    pressure_field : ndarray
        Pressure field data.
    x, y, z : ndarray
        Coordinate arrays.

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
    Plot the PyVista mesh of a specified structure.
    Args:
        pv_regions_dict (dict): A dictionary of PyVista meshes for different brain regions.
        window_size (list, optional): Size of the plot window. Default is [800, 800].
        notebook (bool, optional): Whether to use notebook mode for the plotter. Default is True.
        off_screen (bool, optional): Whether to render the plot off-screen. Default is False.
        kwargs_dict (dict, optional): Additional keyword arguments for the mesh rendering.
    Returns:
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
    **kwargs,
):
    """
    Plot the ultrasound volume using PyVista.
    Args:
        doppler3D_vol (pv.ImageData): The ultrasound volume data.
        cmap (str): Colormap to use for the volume rendering.
        opacity (str or tuple): Opacity function for the volume rendering.
        clim (list): Color limits for the scalars.
    """
    if plotter is None:
        plotter = pv.Plotter(
            window_size=window_size, notebook=notebook, off_screen=off_screen
        )

    scalars = vol_3D.point_data.keys()[0]  # Get the name of the first scalar

    default_kwargs = {
        "scalars": scalars,
        "cmap": "hot",
        "opacity": "sigmoid",
        "mapper": "smart",
        "show_scalar_bar": True,
        "scalar_bar_args": {
            "title": "Doppler (dB)",
            "title_font_size": 16 * scale,
            "label_font_size": 12 * scale,
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
    **kwargs,
):
    """
    Plot the ultrasound volume using PyVista.
    Args:
        doppler3D_vol (pv.ImageData): The ultrasound volume data.
        cmap (str): Colormap to use for the volume rendering.
        opacity (str or tuple): Opacity function for the volume rendering.
        clim (list): Color limits for the scalars.
    """
    if plotter is None:
        plotter = pv.Plotter(
            window_size=window_size, notebook=notebook, off_screen=off_screen
        )

    default_kwargs = {
        "show_edges": False,
        "cmap": "gray",
        "opacity": 1.0,
        "name": "2D doppler",  # Name for the volume
        "show_scalar_bar": True,
        "scalar_bar_args": {
            "title": "2D Doppler (dB)",
            "title_font_size": 16 * scale,
            "label_font_size": 12 * scale,
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
    scale=1,
    vmin=None,
    vmax=None,
    **kwargs,
):
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
            "name": "PressureIso",
            "show_scalar_bar": False,
            "label": "Focal Spot",  # label for the legend
            "color": "r",  # color of the mesh
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
        n_contours = 10
        if vmin is not None:
            min_val = vmin
        else:
            min_val = pressure_vol[scalars].min()
        if vmax is not None:
            max_val = vmax
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
                "title_font_size": 16 * scale,
                "label_font_size": 12 * scale,
                "vertical": True,
                "position_x": 0.8,
                "position_y": 0.1,
                "height": 0.3,
            },
            "label": scalars,  # label for the legend
            "color": "r",  # color of the mesh
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
            "title_font_size": 16 * scale,
            "label_font_size": 12 * scale,
            "vertical": True,
            "position_x": 0.8,
            "position_y": 0.5,
            "height": 0.3,
        },
        "opacity": 1.0,
        "show_edges": True,
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
    glyph="sphere",
    glyph_scale=1.0,
    color="red",
    labels=None,
    label_offset=(0, 0, 0),
    label_font_size=12,
    **kwargs,
):
    """
    Add 3D point markers (and optional labels) to a PyVista scene.
    Args:
        points (array-like of shape (N,3)): XYZ coords.
        plotter (pv.Plotter, optional): existing plotter.
        notebook, window_size, off_screen: passed to Plotter if created.
        glyph (str or pv.PolyData): 'sphere', 'cone', 'cube', or your own mesh.
        glyph_scale (float): uniform scale of each glyph.
        color: marker color.
        labels (list of str, optional): one label per point.
        label_offset (tuple): xyz offset added to each label position.
        label_font_size (int): label font size.
        **kwargs: passed to plotter.add_mesh().
    Returns:
        pv.Plotter
    """
    if plotter is None:
        plotter = pv.Plotter(
            window_size=window_size, notebook=notebook, off_screen=off_screen
        )
    pts = pv.PolyData(np.asarray(points))

    # build glyph source
    if isinstance(glyph, str):
        name = glyph.lower()
        if name == "sphere":
            source = pv.Sphere(radius=1.0)
        elif name == "cone":
            source = pv.Cone(radius=0.5, height=2.0)
        elif name == "cube":
            source = pv.Cube()
        else:
            raise ValueError(
                f"Unsupported glyph '{glyph}'. Use 'sphere','cone','cube', or pass your own mesh."
            )
    else:
        source = glyph  # assume it's a pv.PolyData or mesh

    glyphs = pts.glyph(scale=False, geom=source, factor=glyph_scale)
    plotter.add_mesh(glyphs, color=color, **kwargs)

    # add optional labels
    if labels is not None:
        labels = list(labels)
        if len(labels) != pts.n_points:
            raise ValueError(
                f"labels length {len(labels)} != number of points {pts.n_points}"
            )
        for idx, txt in enumerate(labels):
            pos = np.array(points[idx]) + np.array(label_offset)
            plotter.add_point_labels(
                np.array([pos]),
                [txt],
                font_size=label_font_size,
                text_color=color,
                **kwargs,
            )

    return plotter


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
