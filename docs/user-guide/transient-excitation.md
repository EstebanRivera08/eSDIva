---
icon: lucide/radio
---

# Transient + Excitation

The SIR is convolved (in the frequency domain) with a user excitation pulse,
giving the full time-domain pressure wavefront `p(t, x, y, z)`. Pass one global
pulse `(L,)` or a per-element pulse `(L, E)`.

```python
import numpy as np
from esdiva.emission import Emission

fs, fc, n_cycles = 200e6, tx.fc, 2
t = np.arange(0, n_cycles / fc, 1 / fs)
excitation = np.hanning(len(t)) * np.sin(2 * np.pi * fc * t)

sim = Emission(tx, fs=fs, excitation=excitation)
p, coords = sim(field_points, method="auto")
# dict input : p.shape == (Nt, Nx, Ny, Nz), coords has "x","y","z","t0","dt"
```

![Steered plane-wave transient — matrix array](../examples/assets/ex04_dw_transient.gif)

## When to use

- Realistic pulsed propagation waveforms (DW, steered PW, focused).
- Acoustic intensity / heating maps.
- Comparing pulse shapes and their effect on the focal pattern.

See [Example 4 — Diverging Wave](../examples/example04_lineararray_excitation_DW.md)
and [Example 5 — Steered Plane Wave](../examples/example05_matrixarray_pulsed_steeredPW.md).
