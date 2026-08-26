---
icon: lucide/container
---

# Transducers

SonDI models every transducer geometry as a collection of small flat rectangular patches. The Spatial Impulse Response (SIR) is computed for each patch independently, then summed with per-element delays and apodization weights to produce the total field.

| | |
|---|---|
| ![Linear array](../examples/assets/ex01_gallery_linear.png) | ![Matrix array](../examples/assets/ex01_gallery_matrix.png) |
| ![Concave circular](../examples/assets/ex01_gallery_concave.png) | ![Custom helmet](../examples/assets/ex01_gallery_custom_helmet.png) |

<div class="grid cards" markdown>

-   :lucide-circle: **[Mono-elements](mono-elements.md)**

    ---

    Flat, concave, convex, and focused circular single-element transducers. Focusing achieved through physical curvature, not electronic delays.

-   :lucide-layout-grid: **[Multi-elements](multi-elements.md)**

    ---

    Linear, convex, and matrix arrays with independently controlled elements. Electronic beam steering and focusing via delays and apodization.

-   :lucide-box: **[Transducer Objects](transducer-objects.md)**

    ---

    How SonDI represents transducers as Python objects: geometry properties, patch frames, delays, apodization, and 3-D visualization.

</div>

---

## The patch model

Every surface — no matter the shape — is approximated by a mosaic of small flat rectangles. The `no_sub_x` and `no_sub_y` parameters control subdivision density:

| Parameter | Effect |
|-----------|--------|
| Higher values | Better SIR accuracy, longer computation |
| Lower values | Faster, sufficient for low-frequency or simple geometries |

Good starting points: `no_sub_x=2, no_sub_y=4` for arrays; `no_sub_diameter=20–30` for circular transducers.

## Coordinate system

| Axis | Direction |
|------|-----------|
| X | Lateral — across array elements |
| Y | Elevation — perpendicular to imaging plane |
| Z | Axial — beam propagation direction, depth |

For curved transducers the **rim is always at z = 0**. The medium and geometric focus lie at z > 0.

## Delays and apodization

Both are applied at the element level for multi-element arrays and can be recomputed for any focal point without recreating the transducer:

```python
tx.compute_delays(focus_mm=[0, 0, 30])
tx.compute_apodization(focus_mm=[0, 0, 30], FoverD=2.0)
```

For mono-element transducers, `compute_delays()` is ignored — focusing is purely geometric. Patch-wise apodization via `set_apodization()` is still physically meaningful (e.g., apodizing aperture edges).

## Transducer types at a glance

| Type | Category | Focusing |
|------|----------|---------|
| `LinearArrayTransducer` | Multi-element | Electronic |
| `ConvexArrayTransducer` | Multi-element | Electronic + geometry |
| `MatrixArrayTransducer` | Multi-element | Electronic (2-D steering) |
| `FlatCircularTransducer` | Mono-element | None (diverging) |
| `ConcaveCircularTransducer` | Mono-element | Geometric (bowl) |
| `ConvexCircularTransducer` | Mono-element | Geometric (dome, diverging) |
| `FocusedCircularTransducer` | Mono-element | Geometric (line-focus) |
| `CustomTransducer` | Multi mono-element | Per-element positions |
