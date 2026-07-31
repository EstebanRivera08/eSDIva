"""PyVista 3-D visualisation utilities for pressure fields and transducers."""

import numpy as np
import pyvista as pv

from .pyvista_functions import _normalize_window_size


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
    """Plot the PyVista mesh of specified BrainAtlas structures.

    Parameters
    ----------
    pv_regions_dict : dict or pyvista.PolyData
        Dictionary of PyVista meshes keyed by region name, or a single mesh.
    plotter : pyvista.Plotter, optional
        Existing plotter. If *None*, a new one is created.
    window_size : list of int, optional
        Render window size in pixels. Default ``[800, 800]``.
    notebook : bool, optional
        Enable Jupyter notebook rendering. Default False.
    off_screen : bool, optional
        Render off-screen. Default False.
    kwargs_dict : dict, optional
        Per-region keyword arguments for ``add_mesh``.
    **kwargs
        Forwarded to ``plotter.add_mesh()`` as defaults.

    Returns
    -------
    pyvista.Plotter
        The plotter with brain region meshes added.
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
    """Add a 3-D volume to a PyVista plotter.

    Parameters
    ----------
    vol_3D : pyvista.ImageData
        The volume data (e.g. 3-D ultrasound scan).
    plotter : pyvista.Plotter, optional
        Existing plotter. If *None*, a new one is created.
    notebook : bool, optional
        Enable Jupyter notebook rendering. Default False.
    window_size : list of int, optional
        Render window size. Default ``[700, 700]``.
    off_screen : bool, optional
        Render off-screen. Default False.
    scale : float, optional
        Scaling factor for scalar bar font sizes. Default 1.
    colorbar_title : str, optional
        Title for the colour bar. Defaults to the scalar name.
    **kwargs
        Forwarded to ``plotter.add_volume()``.

    Returns
    -------
    pyvista.Plotter
        The plotter with the volume added.
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
            "height": 0.1,
        },
    }
    for key, value in default_kwargs.items():
        if key not in kwargs:
            kwargs[key] = value

    plotter.add_volume(vol_3D, **kwargs)
    plotter.add_axes()
    return plotter


# Add image


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
    colorbar_title : str, optional
        Title for the colour bar. Defaults to the scalar name.
    **kwargs
        Forwarded to ``plotter.add_mesh()``.

    Returns
    -------
    pyvista.Plotter
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
        "cmap": "inferno",
        "opacity": 1.0,
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

    plotter.add_mesh(image_grid, **kwargs)
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
    window_size : list, optional
        Size of the plot window. Default is [800, 800].
    notebook : bool, optional
        Whether to use notebook mode for the plotter. Default is False.
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
    **kwargs
        Forwarded to ``plotter.add_mesh()``.

    Returns
    -------
    pyvista.Plotter
        The PyVista plotter with the pressure volume mesh added.
    """

    if plotter is None:
        plotter = pv.Plotter(
            notebook=notebook, window_size=window_size, off_screen=off_screen
        )

    scalars = pressure_vol.point_data.keys()[0]  # Get the name of the first scalar

    if scalars != "Pressure":
        print(f"Warning: The scalar field in the pressure volume is named '{scalars}'.")

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
            "ambient": 0.7,
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
                "position_x": 0.75,
                "position_y": 0.1,
                "height": 0.3,
            },
            "label": scalars,  # label for the legend
            "color": "r",  # color of the mesh,
            "ambient": 0.7,
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
    color=None,
    colorbar_title=None,
    **kwargs,
):
    """
    Add a transducer mesh to a PyVista plotter, colored by apodization, delays or a flat color.

    Parameters
    ----------
    TX_mesh : pv.PolyData
        The mesh representing the transducer, with point data for apodization and delays.
    plotter : pv.Plotter, optional
        An existing PyVista plotter to which the transducer mesh will be added. If None
        a new plotter will be created. Default is None.
    window_size : list, optional
        Base size of the plot window before ``scale``. Default is [800, 800].
    notebook : bool, optional
        Whether to use notebook mode for the plotter. Default is False.
    off_screen : bool, optional
        Whether to render the plot off-screen. Default is False.
    scale : float, optional
        Resolution scale factor applied to window size and scalar-bar fonts
        (only when this call creates the plotter). Default is 1.
    scalars : str, optional
        Which scalar field to use for coloring the transducer mesh. Must be either
        "Apodization" or "Delays". Default is "Apodization".
    color : str or tuple, optional
        Uniform PyVista color (name, hex string or RGB tuple). When given it
        overrides ``scalars``: the whole mesh is painted this color and no
        scalar bar is shown. Default is None (color by ``scalars``).
    colorbar_title : str, optional
        Title for the colorbar. If None, it will use "Apodization" or "Delays" based on
        the scalars parameter. Default is None.
    **kwargs
        Forwarded to ``plotter.add_mesh()``.

    Returns
    -------
    pyvista.Plotter
        The plotter with the transducer mesh added.
    """

    if plotter is None:
        plotter = pv.Plotter(
            notebook=notebook,
            window_size=_normalize_window_size(window_size, scale=scale),
            off_screen=off_screen,
        )

    if color is not None:
        # Uniform color: no scalar field, no color bar (lighting defaults shared below).
        default_kwargs = {"color": color, "show_scalar_bar": False}
    else:
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
                "position_x": 0.75,
                "position_y": 0.5,
                "height": 0.3,
            },
        }

    default_kwargs.update({"opacity": 1.0, "show_edges": True, "ambient": 1})
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
    """Add marker points and optional labels to a PyVista plotter.

    Parameters
    ----------
    points : array-like, shape (N, 3)
        Marker coordinates.
    plotter : pyvista.Plotter, optional
        Existing plotter. If *None*, a new one is created.
    notebook : bool, optional
        Enable Jupyter notebook rendering. Default False.
    window_size : tuple of int, optional
        Render window size. Default ``(700, 700)``.
    off_screen : bool, optional
        Render off-screen. Default False.
    point_size : float, optional
        Size of marker points. Default 1.
    labels : list of str, optional
        Text labels for each point.
    label_offset : tuple of float, optional
        Offset for label placement. Default ``(0, 0, 0)``.
    label_font_size : int, optional
        Font size for labels. Default 12.
    **kwargs
        Forwarded to ``plotter.add_mesh()``.

    Returns
    -------
    pyvista.Plotter
        The plotter with markers added.
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


def add_stl_mesh(
    stl_mesh,
    *,
    plotter=None,
    window_size=(800, 800),
    notebook=False,
    off_screen=False,
    color="lightblue",
    opacity=1.0,
    show_edges=True,
    edge_color="black",
    ambient=0.3,
    label=None,
    **kwargs,
):
    """Add an STL mesh to a PyVista plotter.

    Parameters
    ----------
    stl_mesh : pyvista.PolyData or str or Path
        Mesh object or path to an STL file. If a path is given, the mesh is
        loaded automatically.
    plotter : pyvista.Plotter, optional
        Existing plotter. If None, a new one is created.
    window_size : tuple of int, optional
        Render window size ``(width, height)``. Default ``(800, 800)``.
    notebook : bool, optional
        Enable Jupyter notebook rendering. Default False.
    off_screen : bool, optional
        Render off-screen. Default False.
    color : str or tuple, optional
        Mesh colour. Default ``"lightblue"``.
    opacity : float, optional
        Mesh opacity from 0 (transparent) to 1 (opaque). Default 1.0.
    show_edges : bool, optional
        Show mesh edges. Default True.
    edge_color : str or tuple, optional
        Edge colour when ``show_edges=True``. Default ``"black"``.
    ambient : float, optional
        Ambient lighting coefficient. Default 0.3.
    label : str, optional
        Legend label. If None, no label is added.
    **kwargs
        Forwarded to ``plotter.add_mesh()``.

    Returns
    -------
    pyvista.Plotter
        The plotter with the STL mesh added.
    """
    from pathlib import Path

    import pyvista as pv

    # Create plotter if not provided
    if plotter is None:
        plotter = pv.Plotter(
            notebook=notebook, window_size=window_size, off_screen=off_screen
        )

    # Load mesh if path is provided
    if isinstance(stl_mesh, (str, Path)):
        from .pyvista_functions import load_mesh_from_stl

        stl_mesh = load_mesh_from_stl(stl_mesh)

    # Set up default keyword arguments
    default_kwargs = {
        "color": color,
        "opacity": opacity,
        "show_edges": show_edges,
        "edge_color": edge_color,
        "ambient": ambient,
    }

    if label is not None:
        default_kwargs["label"] = label

    # Merge with user-provided kwargs (user kwargs take precedence)
    for key, value in default_kwargs.items():
        if key not in kwargs:
            kwargs[key] = value

    # Add mesh to plotter
    plotter.add_mesh(stl_mesh, **kwargs)
    plotter.add_axes()

    return plotter
