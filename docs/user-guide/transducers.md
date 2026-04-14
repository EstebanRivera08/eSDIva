---
icon: lucide/container
---

# Understanding Transducers

This guide explains how transducers work in PyField at a conceptual level.
For parameter details and code examples, see the
[API reference](../api/transducers.md).

## The patch-based model

Every transducer in PyField -- regardless of shape -- is decomposed into small
flat rectangular patches. The Spatial Impulse Response (SIR) is computed for
each patch independently, then summed with per-element delays and apodization
weights to produce the total transducer response.

This approach means:

- **Any geometry** can be modelled as long as it can be tiled with rectangles
- **Accuracy** is controlled by subdivision density (`no_sub_x`, `no_sub_y`)
- **Speed** depends on the total number of patches times field points

## Transducer categories

### Multi-element arrays

These transducers have multiple independently controlled elements. Each element
can have its own delay and apodization weight, enabling electronic beam
steering and focusing.

| Type | Description | Typical use |
|------|-------------|-------------|
| `LinearArrayTransducer` | 1D row of elements along x | B-mode imaging |
| `ConvexArrayTransducer` | Elements on a convex arc in XZ | Abdominal imaging |
| `MatrixArrayTransducer` | 2D grid of elements | 3D volumetric imaging |

![LinearArrayTransducer](../examples/assets/gallery_linear.png)
![ConvexArrayTransducer](../examples/assets/gallery_convex.png)
![MatrixArrayTransducer](../examples/assets/gallery_matrix.png)

### Mono-element transducers

Single-element transducers with `n_elements = 1`. Focusing is purely
geometric (curved surface). Electronic steering is not available.

| Type | Description | Typical use |
|------|-------------|-------------|
| `FlatCircularTransducer` | Flat circular piston | Unfocused TUS |
| `ConcaveCircularTransducer` | Spherical bowl | HIFU, focused TUS |
| `ConvexCircularTransducer` | Spherical dome | Diverging field |
| `FocusedCircularTransducer` | Cylindrical focus (one axis) | Line-focused TUS |

![FlatCircularTransducer](../examples/assets/gallery_flat_circular.png)
![ConcaveCircularTransducer](../examples/assets/gallery_concave.png)
![FocusedCircularTransducer](../examples/assets/gallery_focused_circular.png)

### Composite arrays

`CustomTransducer` lets you place multiple mono-element transducers at
arbitrary positions and orientations -- useful for TUS helmets, ring arrays,
or any non-standard layout.

![CustomTransducer — TUS helmet](../examples/assets/gallery_custom_helmet.png)

## Delays and apodization

**Delays** control *when* each element fires. By introducing time offsets, the
wavefronts from different elements can be made to converge at a focal point.

**Apodization** controls *how much* each element contributes. A rectangular
window activates a sub-aperture; tapered windows (Hanning, Hamming) reduce
sidelobes at the cost of a wider main lobe.

Both can be recomputed for different focal points without recreating the
transducer object.

## Subdivision density

The `no_sub_x` and `no_sub_y` parameters control how many patches each element
is divided into. More subdivisions:

- Improve accuracy of the SIR computation
- Increase memory usage and computation time
- Are especially important for curved surfaces

A good starting point is `no_sub_x=2, no_sub_y=4` for linear arrays and
`no_sub=20-30` for circular transducers.
