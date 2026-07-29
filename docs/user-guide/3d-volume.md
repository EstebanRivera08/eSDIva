---
icon: lucide/box
---

# 3D Volume

!!! warning "Coming soon"
    This section is under active development. Content will be added in a future release.

    See the [Multi-element 3-D example](../examples/example03_multielements_monochromatic_CW.md) for working 3-D volume rendering.

## Overview

The PyVista backend renders the full 3-D pressure volume as isosurfaces, enabling interactive exploration of the focal region.

```python
from pyfield.plotting import plot3D_pressure_vol

pl = plot3D_pressure_vol(p, x=x, y=y, z=z, contour_levels=11)
pl.show()
```

### Composing scenes

```python
import pyvista as pv
from pyfield.plotting import add_pressure_vol, add_transducer_mesh, create_3Dvol_mesh

pl = pv.Plotter()
mesh = create_3Dvol_mesh(coords["x"], coords["y"], coords["z"], p)
pl = add_pressure_vol(mesh, plotter=pl)
pl = add_transducer_mesh(tx.get_mesh(), plotter=pl)
pl.show()
```
