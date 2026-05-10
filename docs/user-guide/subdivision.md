---
icon: lucide/grid
---

# Patch Subdivision

!!! warning "Coming soon"
    This section is under active development. Content will be added in a future release.

    See [Example 10](../examples/example10_subdivide_parametric_surface.md) for a visual demonstration of `subdivide_parametric_surface` on an ellipsoidal cap.

## Overview

Every transducer surface is decomposed into small flat rectangular patches. Two parameterisations are available for curved surfaces:

### Spherical-cap method

Ring-based tiling in spherical coordinates. Patches are distributed along iso-latitude rings, with refined smaller patches near the pole. Works correctly at any curvature, including hemispheres.

**Default for** `ConcaveCircularTransducer` and `ConvexCircularTransducer`.

### Cartesian method

Arc-length reparameterised Cartesian grid: `x = R·sin(sₓ/R)`, `y = R·sin(sᵧ/R)`. The arc-length parameter step `ds` maps uniformly across the aperture, preventing Jacobian blow-up near the rim of high-curvature surfaces.

**Optional** via `method="cartesian"`. Supports `normalize_patch_size=True` to force patch widths equal to the arc-length step, eliminating Jacobian-driven size variation.

### Key parameters

| Parameter | Effect |
|-----------|--------|
| `no_sub_diameter` | Total number of patches across the diameter |
| `ratio_big_patches` | Fraction of aperture covered by the main grid (rest is border refinement) |
| `refine_factor` | Subdivision factor for border patches |
| `normalize_patch_size` | Force uniform patch sizes (cartesian only) |
