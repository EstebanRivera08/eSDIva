---
icon: lucide/rectangle-horizontal
---

# Rectangular Aperture SIR

!!! warning "Coming soon"
    This section is under active development. Content will be added in a future release.

## Overview

The analytical SIR of a flat rectangular piston is the building block of PyField's patch-based method. Every transducer surface — regardless of curvature — is approximated as a mosaic of small flat rectangles, each contributing one rectangular-aperture SIR.

The far-field SIR for a rectangular patch of width $2a$ and height $2b$ centred at the origin has a closed-form expression involving edge-diffraction terms. PyField evaluates this expression efficiently using Numba JIT compilation, parallelised over field points.

See `src/pyfield/h_sir/farfield_rect_patch.py` for the implementation.
