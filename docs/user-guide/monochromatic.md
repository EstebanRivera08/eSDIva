---
icon: lucide/activity
---

# Monochromatic Simulation

!!! warning "Coming soon"
    This section is under active development. Content will be added in a future release.

    For a working example, see [Linear Array (CW)](../examples/example3_lineartx_monochromatic.md) or the [Quickstart](../index.md).

## Overview

Monochromatic simulation computes the steady-state pressure field for a continuous wave at the transducer's centre frequency. It returns a 3-D spatial pressure field `p(x, y, z)`.

```python
from pyfield.psimulation import PyField

sim = PyField(tx)
x, y, z, p = sim(field_points, method="auto")
# p.shape == (Nx, Ny, Nz)
```

Internally, the SIR is evaluated at each field point and the monochromatic response is computed as the Fourier transform of the SIR at the centre frequency.

## When to use

- Beam pattern analysis and -6 dB focal spot characterisation
- Comparing transducer geometries or subdivision settings
- Quick field preview before a full transient run
