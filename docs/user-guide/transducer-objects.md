---
icon: lucide/box
---

# Transducer Objects

!!! warning "Coming soon"
    This section is under active development. Content will be added in a future release.

    For now, see the [Transducers API reference](../api/transducers.md) for a full listing of properties and methods.

## Object structure

Every transducer in PyField inherits from `TransducerBase` and exposes:

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

## 3-D preview

```python
tx.show(scalars="Apodization")  # colour patches by apodization value
tx.show(scalars="Delays")       # colour patches by delay value
```

Geometry properties are computed lazily — they are only evaluated on first access.
