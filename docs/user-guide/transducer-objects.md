---
icon: lucide/box
---

# Transducer Objects

!!! warning "Coming soon"
    This section is under active development. Content will be added in a future release.

    For now, see the [Transducers API reference](../api/transducers.md) for a full listing of properties and methods.

## Object structure

Every transducer in SonDI inherits from `TransducerBase` and exposes:

| Attribute / Method | Description |
|--------------------|-------------|
| `patch_frames` | Array of patch positions and orientations |
| `delays` | Per-element delay vector (seconds) |
| `apodization` | Per-element weight vector |
| `get_mesh()` | PyVista mesh for 3-D rendering |
| `show()` | Interactive 3-D preview |
| `compute_delays(focus_mm)` | Set electronic focus delays |
| `compute_apodization(focus_mm, FoverD)` | Set apodization weights |
| `set_apodization(weights)` | Manually set per-patch weights |
| `transform(T_matrix)` | Rigidly move the aperture (4×4 homogeneous matrix, translation in mm) |

## Moving a transducer in space

`transform(T_matrix)` applies a rigid-body rotation + translation to **all**
computed geometry — patch vertices, patch frames (centres, normals, tangents)
and element centres — so both the simulation and the 3-D preview see the moved
aperture:

```python
import numpy as np

theta = np.deg2rad(20)
T = np.eye(4)
T[:3, :3] = [[np.cos(theta), 0, np.sin(theta)],
             [0, 1, 0],
             [-np.sin(theta), 0, np.cos(theta)]]  # tilt about y
T[:3, 3] = [0, 0, 10]                             # then shift 10 mm in z
tx.transform(T)
```

Two caveats:

- Delays and apodization are untouched — a focus computed *before* the move
  still aims at the old global-frame target; call `compute_delays` /
  `compute_apodization` again if the beam must follow.
- Simulators snapshot geometry at construction — after transforming, refresh
  with `sim.set("transducer", tx)` (Emission) or `sim.set("tx", tx)` /
  `sim.set("rx", rx)` (Reception).

## 3-D preview

```python
tx.show(scalars="Apodization")  # colour patches by apodization value
tx.show(scalars="Delays")       # colour patches by delay value
```

Geometry properties are computed lazily — they are only evaluated on first access.
