---
icon: lucide/circle
---

# Mono-element Transducers

!!! warning "Coming soon"
    This section is under active development. Content will be added in a future release.

    For now, see the [Transducers API reference](../api/transducers.md) for parameter details and the [Transducer Gallery](../examples/example1_transducer_gallery.md) example for usage patterns.

## At a glance

| Type | Shape | Focusing |
|------|-------|---------|
| `FlatCircularTransducer` | Flat disk | None — diverging field |
| `ConcaveCircularTransducer` | Spherical bowl | Geometric (focus at z > 0) |
| `ConvexCircularTransducer` | Spherical dome | Geometric (diverging) |
| `FocusedCircularTransducer` | Cylindrical arc | Line-focus along one axis |

All mono-element transducers have `n_elements = 1`. Electronic delays have no effect — focusing is achieved through the physical curvature of the surface.

### Z-convention

For all curved circular transducers the **rim is at z = 0**. The `focus_mm` parameter sets the axial depth from the rim to the geometric focus:

```python
tx = ConcaveCircularTransducer(
    diameter_mm=64.0,
    focus_mm=63.0,   # depth from rim to focus
    frequency_Hz=0.5e6,
    no_sub_diameter=30,
)
```
