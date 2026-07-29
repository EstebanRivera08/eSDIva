---
icon: lucide/square-activity
---

# Plotting

2-D Matplotlib and 3-D PyVista visualization. All PyVista helpers accept
`plotter=` to compose several objects in one scene. See the
[Visualization user guide](../user-guide/visualization.md).

## 2-D (Matplotlib)

::: pyfield.plotting.plot2D_pressure_slices

::: pyfield.plotting.plot2D_transient_slices

::: pyfield.plotting.plot2D_pressure_plane

## 3-D (PyVista)

High-level renderers that open a ready-to-show `pv.Plotter`.

::: pyfield.plotting.plot3D_pressure_vol

::: pyfield.plotting.plot3D_pressure_slices

::: pyfield.plotting.plot3D_transient_slices

## PyVista scene helpers

Composable `add_*` helpers — each takes a `plotter=` and returns it, so scenes are
built up one object at a time (pressure, transducer, STL, atlas regions, markers).

::: pyfield.plotting.add_pressure_vol

::: pyfield.plotting.add_transducer_mesh

::: pyfield.plotting.add_3D_vol

::: pyfield.plotting.add_2D_image

::: pyfield.plotting.add_regions_mesh

::: pyfield.plotting.add_stl_mesh

::: pyfield.plotting.add_markers

## PyVista mesh builders

Turn raw arrays or files into PyVista meshes for use with the helpers above.

::: pyfield.plotting.create_3Dvol_mesh

::: pyfield.plotting.create_2Dimage_mesh

::: pyfield.plotting.load_mesh_from_stl

## Export

::: pyfield.plotting.save_pyvista_screenshot

::: pyfield.plotting.save_pyvista_movie

::: pyfield.plotting.save_matplotlib_animation
