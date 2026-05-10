---
icon: lucide/radio
---

# Transient + Excitation

!!! warning "Coming soon"
    This section is under active development. Content will be added in a future release.

    See the [Transient Simulation example](../examples/example5_lineartx_transient.md) for a working end-to-end example with excitation.

## Overview

Full time-domain simulation: the spatial impulse response is convolved with a user-defined excitation pulse. Returns `p(t, x, y, z)`.

```python
import numpy as np
from pyfield.psimulation import PyField

fs = 200e6
fc = tx.fc
n_cycles = 2
t_pulse = np.arange(0, n_cycles / fc, 1 / fs)
excitation = np.hanning(len(t_pulse)) * np.sin(2 * np.pi * fc * t_pulse)

sim = PyField(tx, fs=fs)
x, y, z, t, p = sim(field_points, method="auto", excitation=excitation)
# p.shape == (Nt, Nx, Ny, Nz)
```

## When to use

- Realistic pulsed propagation waveforms
- Tissue heating estimates and acoustic intensity maps
- Comparing pulse shapes and their effect on the focal pattern
