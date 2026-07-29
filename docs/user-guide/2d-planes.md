---
icon: lucide/chart-area
---

# 2D Planes

!!! warning "Coming soon"
    This section is under active development. Content will be added in a future release.

    See the [Visualization Guide](visualization.md) for a quick reference, and [Mono-element Fields](../examples/example02_monoelements_monochromatic_CW.md) for working plot examples.

## Overview

The Matplotlib backend produces static 2-D pressure plots suitable for publications and quick inspection.

### Monochromatic fields

```python
from pyfield.plotting import plot2D_pressure_slices

plot2D_pressure_slices(p, x=x, y=y, z=z, db_scale=True, vmin=-40)
```

Renders three orthogonal slice views (XZ, XY, YZ). If one spatial dimension has size 1, a single 2-D image is shown.

### Transient fields

```python
from pyfield.plotting import plot2D_pressure_slices

plot2D_pressure_slices(p_transient, x=x, y=y, z=z, time_array=t, db_scale=True, video_duration_s=5)
```

Creates an animated display of time frames spread evenly over the specified duration.
