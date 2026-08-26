---
icon: lucide/radio-tower
---

# Transducers

All transducers share the `TransducerBase` interface: patch geometry, per-element
`compute_delays` / `compute_apodization`, rigid `transform`, and 3-D `show`. Units
are mm at the API surface. See the [Transducers user guide](../user-guide/transducers.md).

## Shared interface — TransducerBase

::: sondi.transducers.TransducerBase
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

::: sondi.transducers.LinearArrayTransducer
    options:
      members: false

::: sondi.transducers.ConvexArrayTransducer
    options:
      members: false

::: sondi.transducers.MatrixArrayTransducer
    options:
      members: false

## Mono-element transducers

::: sondi.transducers.FlatCircularTransducer
    options:
      members: false

::: sondi.transducers.ConcaveCircularTransducer
    options:
      members: false

::: sondi.transducers.ConvexCircularTransducer
    options:
      members: false

::: sondi.transducers.FocusedCircularTransducer
    options:
      members: false

## Custom & imported

::: sondi.transducers.CustomTransducer
    options:
      members: false

::: sondi.transducers.from_fieldii_rect_data

::: sondi.transducers.from_fieldii_xdc_data
