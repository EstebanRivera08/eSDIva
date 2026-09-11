---
icon: lucide/layout-grid
---

# Multi-element Transducers

Arrays of independently driven elements — electronic steering and focusing via
per-element delays and apodization. See the
[Transducers API](../api/transducers.md) and
[Example 3 — Multi-element 3-D](../examples/example03_multielements_monochromatic_CW.md).

| | |
|---|---|
| ![Linear array](../examples/assets/ex01_gallery_linear.png) | ![Convex array](../examples/assets/ex01_gallery_convex.png) |
| ![Matrix array](../examples/assets/ex01_gallery_matrix.png) | ![Custom array](../examples/assets/ex01_gallery_custom_helmet.png) |

Radiated pressure fields for these arrays live in the [Emission guide](emission.md).

## At a glance

| Type | Layout | Steering |
|------|--------|---------|
| `LinearArrayTransducer` | 1-D row along X | Lateral + depth (XZ) |
| `ConvexArrayTransducer` | Elements on convex arc | Wide-angle sector |
| `MatrixArrayTransducer` | 2-D grid of elements | Full 3-D volumetric |
| `CustomTransducer` | Arbitrary positions | Per-element definition |

## Basic usage

```python
from esdiva.transducers import LinearArrayTransducer

tx = LinearArrayTransducer(
    n_elements=64,
    element_width_mm=0.25,
    element_height_mm=12.0,
    kerf_mm=0.05,
    no_sub_x=2,
    no_sub_y=4,
    frequency_Hz=5e6,
)

# Focus at (x=0, y=0, z=30 mm)
tx.compute_delays(focus_mm=[0, 0, 30])
tx.compute_apodization(focus_mm=[0, 0, 30], FoverD=2.0)
```

Delays and apodization can be recomputed for any focal point without recreating the transducer object.

## Elevation lens

A linear array focuses electronically in the imaging plane, but **not** in
elevation — the out-of-plane direction the image cannot show. Physical probes
solve that with an acoustic lens, and eSDIva models it as real geometry: pass
`elevation_focus_mm` and each element's patches are placed on a cylindrical arc.

```python
tx = LinearArrayTransducer(
    n_elements=64, element_width_mm=0.108, element_height_mm=4.0,
    kerf_mm=0.002, no_sub_x=1, no_sub_y=6,
    elevation_focus_mm=12.0,          # RADIUS of curvature, in mm
    frequency_Hz=12.5e6,
)
tx.elevation_focus_depth_mm     # 11.832 — where it actually focuses
tx.elevation_lens_sag           # +1.678e-4 m, signed
```

`no_sub_y` must be at least 2, or there are no patches to put on the curve.

!!! warning "`elevation_focus_mm` is a radius, not a focal depth"
    The lens is referenced with its **rim at `z = 0`** (the Field II
    `xdc_focused_array` datum), so the arc's centre of curvature — the point every
    patch is equidistant from, which is the definition of the focus — sits one
    sagitta *shallower* than the radius. The gap is `h²/(8R)`: 0.17 mm for a 4 mm
    aperture at `R = 12 mm`. Read the real value from
    `elevation_focus_depth_mm` rather than assuming it equals what you passed in.

**Sign: concave or convex.** Positive curves the surface away from the medium
(concave, converging — an ordinary elevation lens). Negative makes it bulge into
the medium (convex, diverging), spreading the elevation beam instead; its focus is
virtual, and `elevation_focus_depth_mm` returns a negative depth. A flat aperture
returns `inf`.

**Why it matters for measurements.** The elevation beam at the focus is about
`λ·z/H` wide — 0.36 mm for a 4 mm aperture at 12 mm, against 1.7 mm for the same
aperture unfocused at 22 mm. Anything quantitative that averages over the
resolution cell inherits that width: a Doppler velocity is the power-weighted mean
over the cell, so an unfocused elevation aperture reads a vessel comparable to the
beam as slower than it is. See
[Flow Simulation](flow-simulation.md).

!!! note "The time origin already accounts for it"
    A curved surface sits off the `z = 0` plane a beamformer measures depth from,
    so reception subtracts the signed sag (once per aperture) from `coords["t0"]`.
    Nothing to do at the call site: a point target lands at its true depth for
    flat, concave and convex apertures alike. Emission needs no such term — its
    `t0` is a physical origin, not a beamforming reference.
