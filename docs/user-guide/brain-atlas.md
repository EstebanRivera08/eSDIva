---
icon: lucide/brain
---

# Brain Atlas Integration

eSDIva integrates with the [BrainGlobe Atlas API](https://brainglobe.info/documentation/brainglobe-atlasapi/index.html) to map acoustic pressure fields onto anatomical brain structures.

The `BG_Atlas` class in `esdiva.utilities` wraps the BrainGlobe API and provides:

- Loading and querying rat and mouse brain atlases
- Coordinate registration between the transducer frame and atlas space
- Overlaying pressure volumes onto brain anatomy in PyVista scenes
- Targeting specific anatomical regions by name

## Quick example

Load anatomy, register it into the transducer frame, then compose brain regions,
the transducer, and a simulated pressure volume in a single PyVista scene using
eSDIva's own helpers:

```python
import pyvista as pv
from esdiva.utilities import BG_Atlas
from esdiva.plotting import add_regions_mesh, add_transducer_mesh, add_pressure_vol

# 1. Load named structures (downloads the atlas on first use).
atlas = BG_Atlas("allen_mouse_25um", region_names=["root", "CTX", "TH"])

# 2. Register atlas space → transducer frame (4×4 homogeneous, translation in mm).
atlas.transform(T_matrix=T_matrix, inplace=True)

# 3. One scene: anatomy + probe geometry + focused pressure field.
pl = pv.Plotter()
pl = add_regions_mesh(
    atlas.pv_mesh,                         # dict of region meshes keyed by name
    plotter=pl,
    kwargs_dict={
        "root": {"color": "lightgray", "opacity": 0.2},
        "CTX":  {"color": "lightblue", "opacity": 0.3},
        "TH":   {"color": "salmon",    "opacity": 0.3},
    },
)
pl = add_transducer_mesh(tx.get_mesh(), plotter=pl)
pl = add_pressure_vol(pressure_vol, plotter=pl, contour_levels=8)
pl.show()
```

Full working simulations: [Mouse Brain Atlas](../examples/example12_txconcave_mousebrain.md)
(a concave transducer focused into a mouse brain) and
[Rat Brain Targeting](../examples/example13_txlinear_ratbrainzones.md)
(a linear array steered onto named rat-brain zones).

| | |
|---|---|
| ![Mouse brain scene](../examples/assets/ex12_brain_mouse_scene.png) | ![Rat brain zones](../examples/assets/ex13_rat_brain_zones.png) |

Every public method is documented in [API → Brain Atlas](../api/brain_atlas.md).

## Supported atlases

| Atlas | Species | Resolution |
|-------|---------|-----------|
| `allen_mouse_25um` | Mouse | 25 µm |
| `whs_sd_rat_39um` | Rat (SD) | 39 µm |

Atlas data is downloaded automatically on first use via BrainGlobe.
