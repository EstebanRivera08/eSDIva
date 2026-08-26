---
icon: lucide/radio-tower
---

# Transducers

All transducers share the `TransducerBase` interface: patch geometry, per-element
`compute_delays` / `compute_apodization`, rigid `transform`, and 3-D `show`. Units
are mm at the API surface. See the [Transducers user guide](../user-guide/transducers.md).

## Shared interface — TransducerBase

::: esdiva.transducers.TransducerBase
    options:
      members:
        - compute_delays
        - compute_apodization
        - set_apodization
        - set_impulse_response
        - set_excitation
        - transform
        - clean
        - get_mesh
        - show

## Array transducers

::: esdiva.transducers.LinearArrayTransducer
    options:
      members: false

::: esdiva.transducers.ConvexArrayTransducer
    options:
      members: false

::: esdiva.transducers.MatrixArrayTransducer
    options:
      members: false

## Mono-element transducers

::: esdiva.transducers.FlatCircularTransducer
    options:
      members: false

::: esdiva.transducers.ConcaveCircularTransducer
    options:
      members: false

::: esdiva.transducers.ConvexCircularTransducer
    options:
      members: false

::: esdiva.transducers.FocusedCircularTransducer
    options:
      members: false

## Custom & imported

::: esdiva.transducers.CustomTransducer
    options:
      members: false

::: esdiva.transducers.from_fieldii_rect_data

::: esdiva.transducers.from_fieldii_xdc_data
