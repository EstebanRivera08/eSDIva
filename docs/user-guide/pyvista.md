---
icon: lucide/box
---

# PyVista Integration

!!! warning "Coming soon"
    This section is under active development. Content will be added in a future release.

    See the [STL Meshes](../examples/example8_importstl_petridish.md) and [STL + Simulation](../examples/example9_monoelement_petridish.md) examples for composing 3-D scenes.

## Overview

PyField's PyVista helpers allow composing rich 3-D scenes combining:

- **Pressure volumes** — isosurfaces or volumetric rendering of `p(x, y, z)`
- **Transducer meshes** — patch geometry coloured by apodization or delays
- **STL meshes** — arbitrary experimental geometry (e.g., petri dishes, phantoms)
- **Brain anatomy** — atlas-registered anatomical structures

All helpers operate on a shared `pv.Plotter` instance, so scenes are built incrementally and rendered in a single interactive window.
