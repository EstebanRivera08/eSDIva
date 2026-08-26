---
icon: lucide/square-activity
---

# Plotting

2-D Matplotlib and 3-D PyVista visualization. All PyVista helpers accept
`plotter=` to compose several objects in one scene. See the
[Visualization user guide](../user-guide/visualization.md).

## 2-D (Matplotlib)

::: sondi.plotting.plot2D_pressure_slices

::: sondi.plotting.plot2D_transient_slices

::: sondi.plotting.plot2D_pressure_plane

## 3-D (PyVista)

High-level renderers that open a ready-to-show `pv.Plotter`.

::: sondi.plotting.plot3D_pressure_vol

::: sondi.plotting.plot3D_pressure_slices

::: sondi.plotting.plot3D_transient_slices

## PyVista scene helpers

Composable `add_*` helpers — each takes a `plotter=` and returns it, so scenes are
built up one object at a time (pressure, transducer, STL, atlas regions, markers).

::: sondi.plotting.add_pressure_vol

::: sondi.plotting.add_transducer_mesh

::: sondi.plotting.add_3D_vol

::: sondi.plotting.add_2D_image

::: sondi.plotting.add_regions_mesh

::: sondi.plotting.add_stl_mesh

::: sondi.plotting.add_markers

## PyVista mesh builders

Turn raw arrays or files into PyVista meshes for use with the helpers above.

::: sondi.plotting.create_3Dvol_mesh

::: sondi.plotting.create_2Dimage_mesh

::: sondi.plotting.load_mesh_from_stl

## Export

::: sondi.plotting.save_pyvista_screenshot

::: sondi.plotting.save_pyvista_movie

::: sondi.plotting.save_matplotlib_animation
