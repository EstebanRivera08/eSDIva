---
icon: lucide/chart-area
---

# 2D Planes

The Matplotlib backend produces static 2-D pressure plots suitable for publications and quick inspection. See [Mono-element Fields](../examples/example02_monoelements_monochromatic_CW.md) for working code.

### Monochromatic fields

```python
from esdiva.plotting import plot2D_pressure_slices

plot2D_pressure_slices(p, x=x, y=y, z=z, db_scale=True, vmin=-40)
```

Renders three orthogonal slice views (XZ, XY, YZ). If one spatial dimension has size 1, a single 2-D image is shown.

![Monochromatic pressure slices — linear array](../examples/assets/ex03_linear_array_field.png)

### Transient fields

```python
from esdiva.plotting import plot2D_pressure_slices

plot2D_pressure_slices(p_transient, x=x, y=y, z=z, time_array=t, db_scale=True, video_duration_s=5)
```

Creates an animated display of time frames spread evenly over the specified duration.

![Diverging-wave transient wavefront](../examples/assets/ex04_dw_transient.gif)

Full parameters: [API → Plotting](../api/plotting.md).
