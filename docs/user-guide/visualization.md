---
icon: lucide/square-activity
---

# Visualization

PyField provides two visualization backends suited for different workflows.

<div class="grid cards" markdown>

-   :lucide-chart-area: **[2D Planes](2d-planes.md)**

    ---

    Matplotlib-based static plots. Orthogonal slice views (XZ, XY, YZ) for monochromatic fields and animated frame sequences for transient simulations.

-   :lucide-box: **[3D Volume](3d-volume.md)**

    ---

    Interactive PyVista isosurface rendering of the full pressure volume. Compose transducer geometry, pressure, and anatomy in one 3-D scene.

-   :lucide-layers: **[3D Planes](3d-planes.md)**

    ---

    Flat cross-section planes rendered in 3-D with PyVista. Useful for locating focal spots and visualising field cross-sections in anatomical context.

</div>

---

## Quick reference

| Function | Backend | Use case |
|----------|---------|---------|
| `plot2D_pressure_slices` | Matplotlib | Monochromatic/transient 2-D orthogonal slices |
| `plot3D_pressure_vol` | PyVista | Interactive 3-D isosurfaces |
| `add_pressure_vol` | PyVista | Composable scene helper |
| `add_transducer_mesh` | PyVista | Transducer geometry overlay |
| `create_3Dvol_mesh` | PyVista | Build pressure mesh for composing |

## dB scale

Both backends support logarithmic display:

```python
from pyfield.plotting import plot2D_pressure_slices
plot2D_pressure_slices(p, x=x, y=y, z=z, db_scale=True, vmin=-40)
```

`vmin=-40` gives a standard 40 dB dynamic range, typical for beam pattern analysis.

## Composing 3-D scenes

Multiple PyVista helpers chain on the same plotter:

```python
import pyvista as pv
from pyfield.plotting import add_pressure_vol, add_transducer_mesh, create_3Dvol_mesh

pl = pv.Plotter()
mesh = create_3Dvol_mesh(coords["x"], coords["y"], coords["z"], p)
pl = add_pressure_vol(mesh, plotter=pl)
pl = add_transducer_mesh(tx.get_mesh(), plotter=pl)
pl.show()
```

See [Brain Atlas](brain-atlas.md) for adding anatomical overlays to these scenes.
