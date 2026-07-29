---
icon: lucide/brain
---

# Brain Atlas Integration

!!! warning "Coming soon"
    This section is under active development. Content will be added in a future release.

    See the [Mouse Brain Atlas](../examples/example12_txconcave_mousebrain.md) and [Rat Brain Targeting](../examples/example13_txlinear_ratbrainzones.md) examples for working atlas integration code.

## Overview

PyField integrates with the [BrainGlobe Atlas API](https://brainglobe.info/documentation/brainglobe-atlasapi/index.html) to map acoustic pressure fields onto anatomical brain structures.

The `BG_Atlas` class in `pyfield.utilities` wraps the BrainGlobe API and provides:

- Loading and querying rat and mouse brain atlases
- Coordinate registration between the transducer frame and atlas space
- Overlaying pressure volumes onto brain anatomy in PyVista scenes
- Targeting specific anatomical regions by name

## Supported atlases

| Atlas | Species | Resolution |
|-------|---------|-----------|
| `allen_mouse_25um` | Mouse | 25 µm |
| `whs_sd_rat_39um` | Rat (SD) | 39 µm |

Atlas data is downloaded automatically on first use via BrainGlobe.
