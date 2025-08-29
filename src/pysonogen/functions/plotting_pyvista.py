import numpy as np
import pyvista as pv

# -------------------- Plotting Functions --------------------


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
                kwargs["label"] = region if region != "root" else "Brain"
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
    doppler3D_vol,
    *,
    plotter=None,
    notebook=False,
    window_size=[700, 700],
    off_screen=False,
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
        "scalars": "doppler",  # Assuming 'doppler' is a scalar field in doppler3D_vol
        "cmap": "hot",
        "opacity": "sigmoid",
        "mapper": "smart",
        "show_scalar_bar": True,
        "scalar_bar_args": {
            "title": "Doppler (dB)",
            "title_font_size": 16,
            "label_font_size": 12,
        },
    }
    for key, value in default_kwargs.items():
        if key not in kwargs:
            kwargs[key] = value

    vol = plotter.add_volume(doppler3D_vol, **kwargs)
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
            "title_font_size": 16,
            "label_font_size": 12,
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
    plot_focal_spot=True,
    off_screen=False,
    **kwargs,
):
    if plotter is None:
        plotter = pv.Plotter(
            notebook=notebook, window_size=window_size, off_screen=off_screen
        )

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
        threshold = 0.7 * pressure_vol["Pressure"].max()
        iso_mesh = pressure_vol.contour(
            [threshold], scalars="Pressure"
        )  # Create isosurface at threshold# add that instead of (or in addition to) the volume
        plotter.add_mesh(iso_mesh, **kwargs)

    else:
        n_contours = 10
        min_val = 0
        max_val = pressure_vol["Pressure"].max()
        levels = np.linspace(min_val, max_val, n_contours)
        iso_mesh = pressure_vol.contour(
            isosurfaces=levels, scalars="Pressure"
        )  # Create isosurface at threshold
        default_kwargs = {
            "scalars": "Pressure",  # use the scalar to color surfaces
            "opacity": "linear",
            "cmap": "jet",
            "show_scalar_bar": True,
            "scalar_bar_args": {
                "title": "Pressure (u.a.)",
                "title_font_size": 16,
                "label_font_size": 12,
                "vertical": True,
                "position_x": 0.85,
                "position_y": 0.2,
                "height": 0.3,
            },
            "label": "Pressure PII",  # label for the legend
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
    **kwargs,
):
    if plotter is None:
        plotter = pv.Plotter(
            notebook=notebook, window_size=window_size, off_screen=off_screen
        )

    default_kwargs = {
        "scalars": "Apodization",  # Assuming 'Apodization' is a scalar field in TX_mesh
        "cmap": "cool",
        "clim": [0, 1],
        "opacity": 1.0,
        "show_edges": True,
        "show_scalar_bar": True,
        "scalar_bar_args": {
            "title": "Apodization",
            "title_font_size": 16,
            "label_font_size": 12,
            "vertical": True,
            "position_x": 0.85,
            "position_y": 0.6,
            "height": 0.3,
        },
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
