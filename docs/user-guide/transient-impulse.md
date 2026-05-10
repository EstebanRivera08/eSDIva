---
icon: lucide/waves
---

# Transient Impulse Simulation

!!! warning "Coming soon"
    This section is under active development. Content will be added in a future release.

    See the [Transient Simulation example](../examples/example5_lineartx_transient.md) for a working pulsed simulation.

## Overview

Transient impulse simulation returns the spatial impulse response `h(t, x, y, z)` without convolving an excitation pulse. This is the raw SIR output for each field point.

```python
sim = PyField(tx, fs=200e6)
x, y, z, p = sim(field_points, method="auto", monochromatic=False)
# p.shape == (Nt, Nx, Ny, Nz)
```

## When to use

- Inspecting the raw SIR at specific field points
- Debugging transducer geometry by examining arrival times
- Applying a custom convolution outside PyField
