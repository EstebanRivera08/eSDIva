---
icon: lucide/chart-area
---

# 2D Planes

!!! warning "Coming soon"
    This section is under active development. Content will be added in a future release.

    See the [Visualization Guide](visualization.md) for a quick reference, and [Mono-element Fields](../examples/example2_monoelement_transducers.md) for working plot examples.

## Overview

The Matplotlib backend produces static 2-D pressure plots suitable for publications and quick inspection.

### Monochromatic fields

```python
from pyfield.utilities import plot_pressure_planes

plot_pressure_planes(x, y, z, p, db_scale=True, vmin=-40)
```

Renders three orthogonal slice views (XZ, XY, YZ). If one spatial dimension has size 1, a single 2-D image is shown.

### Transient fields

```python
from pyfield.utilities import plot_slices_2d

plot_slices_2d(x, y, z, p_transient, time_array=t, db_scale=True, video_duration_s=5)
```

Creates an animated display of time frames spread evenly over the specified duration.
