---
icon: lucide/radio-tower
---

# Transducers

All transducers share the `TransducerBase` interface: patch geometry, per-element
`compute_delays` / `compute_apodization`, rigid `transform`, and 3-D `show`. Units
are mm at the API surface. See the [Transducers user guide](../user-guide/transducers.md).

## Shared interface — TransducerBase

::: pyfield.transducers.TransducerBase
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

::: pyfield.transducers.LinearArrayTransducer
    options:
      members: false

::: pyfield.transducers.ConvexArrayTransducer
    options:
      members: false

::: pyfield.transducers.MatrixArrayTransducer
    options:
      members: false

## Mono-element transducers

::: pyfield.transducers.FlatCircularTransducer
    options:
      members: false

::: pyfield.transducers.ConcaveCircularTransducer
    options:
      members: false

::: pyfield.transducers.ConvexCircularTransducer
    options:
      members: false

::: pyfield.transducers.FocusedCircularTransducer
    options:
      members: false

## Custom & imported

::: pyfield.transducers.CustomTransducer
    options:
      members: false

::: pyfield.transducers.from_fieldii_rect_data

::: pyfield.transducers.from_fieldii_xdc_data
