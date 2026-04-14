---
icon: lucide/square-activity
---

# Visualization Guide

PyField provides two visualization backends for different use cases.

## Matplotlib (2D static plots)

Best for publications, quick inspection, and notebook workflows.

### Monochromatic fields

```python
from pyfield.utilities import plot_pressure_planes

plot_pressure_planes(x, y, z, p, db_scale=True, vmin=-40)
```

This produces three orthogonal slice views (XZ, XY, YZ) through the pressure
volume. If one spatial dimension has size 1, a single 2D image is shown instead.

![Monochromatic pressure planes — linear array](../examples/assets/lineartx_monochromatic.png)

### Transient fields

```python
from pyfield.utilities import plot_slices_2d

plot_slices_2d(x, y, z, p_transient,
               time_array=t,
               db_scale=True,
               video_duration_s=5)
```

This creates an animated display of time frames. All frames are spread evenly
over the specified duration.

![Transient wavefront animation](../examples/assets/pressure_field_video.gif)

### Tips

- Use `db_scale=True` with `vmin=-40` for a standard 40 dB dynamic range
- Set `centered_to_max=True` to center slice planes on the pressure maximum
- Pass `save_path="path/to/dir"` to export figures

## PyVista (3D interactive)

Best for exploring pressure fields in 3D, composing transducer + anatomy scenes.

### Basic 3D pressure view

```python
from pyfield.utilities import plot_pressure_field

pl = plot_pressure_field(x, y, z, p, contour_levels=11)
pl.show()
```

![3-D pressure field — linear array](../examples/assets/linear_array_pressure_field.png)
![3-D pressure field — matrix array](../examples/assets/matrix_array_pressure_field.png)

### Transducer visualization

```python
tx.show(scalars="Apodization")  # Interactive 3D view
tx.show(scalars="Delays")       # Colour by delay values
```

### Composing scenes

Multiple PyVista helpers can be chained on the same plotter:

```python
import pyvista as pv
from pyfield.utilities import (
    add_pressure_vol, add_transducer_mesh, create_vol_mesh
)

pl = pv.Plotter()
mesh = create_vol_mesh(x, y, z, p)
pl = add_pressure_vol(mesh, plotter=pl)
pl = add_transducer_mesh(tx.get_mesh(), plotter=pl)
pl.show()
```

For brain atlas overlays, see the [Brain Atlas API docs](../api/brain_atlas.md).
