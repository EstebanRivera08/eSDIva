---
icon: lucide/ruler
---

# Geometry Functions

Every transducer in eSDIva is a mosaic of small flat **rectangular patches**,
because the spatial impulse response has a closed-form (trapezoidal) solution only
for a rectangle. Curved and arbitrary surfaces are therefore *tessellated* into
rectangles before any field is computed. Two tilers in
`esdiva.utilities.surface_subdivision` do this; the transducer classes call them for
you, but you can call them directly to build a custom aperture.

A patch is only valid in the far field of its own size — `w ≪ √(4·l·c/f)`, with `l`
the patch-to-field distance. That is why finer subdivision (`no_sub_*`,
`no_sub_diameter`, `n_rings`) means more patches and a more accurate field: each
patch stays small enough for the trapezoidal SIR to hold.

## Parametrization: how a surface is described

A surface is a map from a 2-D parameter domain `(u, v)` to 3-D space,
`r(u, v) → (x, y, z)` in metres. The two tangent vectors `∂r/∂u`, `∂r/∂v` span the
local tangent plane; their cross product is the patch normal (the direction the
patch radiates along). Each patch is the flat rectangle in that tangent plane, with
physical width `wu` and height `wv` — the quantities the SIR kernels consume.

Two natural parametrizations, one per tiler:

| Parametrization | Domain | Best for |
|-----------------|--------|----------|
| **Cartesian `(u, v)`** | a rectangle in the parameter plane | flat, mildly curved, or single-axis surfaces |
| **Radial `(r, φ)`** | rings × azimuth | any axisymmetric cap `z(r)` — bowl, dome, aspheric |

The choice matters because the grid must not **shear**: if the two tangents stop
being orthogonal, the "rectangles" become parallelograms and the patch area the SIR
assumes is wrong. A cartesian grid shears on a doubly-curved cap; a radial grid
never does on a surface of revolution.

## Tessellation: the two tilers

### `subdivide_radial` — axisymmetric caps

```python
from esdiva.utilities.surface_subdivision import subdivide_radial
import numpy as np

mosaic = subdivide_radial(
    sag=lambda r: -np.sqrt(0.05**2 - r**2) + 0.05,   # z(r) in metres
    radius=0.02,       # aperture radius (m)
    n_rings=20,        # coarse rings, centre → rim
)
```

Concentric rings, each split into near-square azimuthal patches. For a surface of
revolution the meridian tangent `(cos, sin, z′)` and the azimuth tangent
`(−sin, cos, 0)` are exactly orthogonal, so **every patch is a true rectangle**, the
outer ring is the exact circular rim (no staircase), and `Σ wv = 2πr_c` per ring
gives ~100 % coverage. Refinement here thins the *innermost* rings (where azimuthal
patches would otherwise overlap at the hub) — the opposite of the cartesian tiler.
It warns when the meridian normal turns more than `max_ring_turn_deg` (default 15°)
between rings, meaning the flat patches chord too coarsely across the curvature:
raise `n_rings`. This is the engine behind every circular transducer and
`RadialProfileTransducer`.

### `subdivide_cartesian` — flat / mildly curved / single-axis

```python
from esdiva.utilities.surface_subdivision import subdivide_cartesian
import numpy as np

mosaic = subdivide_cartesian(
    surface_fn=lambda u, v: np.array([u, v, 0.0]),   # r(u, v) in metres
    u_range=(-0.01, 0.01), v_range=(-0.006, 0.006),
    n_u=40, n_v=24,
)
```

A grid in `(u, v)`; each cell becomes a flat rectangle in the tangent plane at its
centre. An optional `inside_fn` masks the aperture shape — cells fully outside are
dropped, boundary cells are subdivided `border_refine²` ways so a curved rim is
resolved cleanly. `normalize_patch_size=True` sets each patch to the parameter-space
step (right when `u`, `v` are already arc length, so patches come out uniform).
It warns when the tangents depart from orthogonal by more than `max_shear_deg`
(default 5°): on a doubly-curved cap the grid shears and the round rim staircases —
switch to `subdivide_radial`.

### What both return

A dict describing the mosaic, consumed directly by the SIR kernels and the
plotting routines:

| Key | Meaning |
|-----|---------|
| `corners` | the four 3-D vertices of each patch |
| `centers` | patch centre positions (m) |
| `normals` | outward unit normal per patch |
| `tangents_u`, `tangents_v` | the two in-plane unit tangents |
| `wu`, `wv` | physical patch width and height (m) |
| `el_idx` | which element each patch belongs to |
| `coverage` | tiled area ÷ true surface area (≈ 1.0 when well resolved) |

`coverage` is the quick sanity check: far from 1.0 means the tessellation is
under- or over-covering the surface and the field amplitude will be off.

## Choosing a tiler

- Axisymmetric cap (bowl, dome, lens, any `z(r)`) → **`subdivide_radial`**.
- Flat piston, linear-array element, single-axis (cylindrical) curve → **`subdivide_cartesian`**.
- Doubly-curved but not a surface of revolution → cartesian with a fine grid, and
  heed the shear warning; there is no shear-free tiling for a general doubly-curved
  patch.

See [Mono-element Transducers](mono-elements.md) for `RadialProfileTransducer`,
which wraps `subdivide_radial` around an arbitrary `z(r)`, and the
[Transducers API](../api/transducers.md) for the subdivision parameters exposed on
each class.
