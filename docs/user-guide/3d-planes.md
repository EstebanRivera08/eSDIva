---
icon: lucide/layers
---

# 3D Planes

!!! warning "Coming soon"
    This section is under active development. Content will be added in a future release.

    See the [STL + Simulation example](../examples/example15_monoelement_petridish.md) for an example of slice planes in a 3-D scene.

## Overview

3-D plane rendering places flat cross-section slices inside a PyVista 3-D scene. This is useful for showing the focal spot in the context of surrounding anatomy or experimental geometry.

Unlike [3D Volume](3d-volume.md) isosurfaces, plane views preserve the exact field values at each slice position without interpolation artefacts from isosurface meshing.
