---
icon: lucide/audio-lines
---

# Simulation Modes

PyField supports two simulation modes: **monochromatic** (continuous wave) and
**transient** (pulsed). This guide explains when to use each and what to expect.

## Monochromatic simulation

Computes the steady-state pressure field for a continuous wave at the
transducer's centre frequency. The output is a 3D spatial pressure field
`p(x, y, z)`.

```python
from pyfield.psimulation import PyField

sim = PyField(tx)
x, y, z, p = sim(field_points, method="auto")
# p.shape == (Nx, Ny, Nz)
```

**When to use**: beam pattern analysis, focal spot characterization, comparing
transducer designs.

![Monochromatic CW pressure field — linear array](../examples/assets/lineartx_monochromatic.png)

## Transient simulation

Computes the time-domain pressure field by convolving the SIR with an
excitation pulse. The output is a 4D spatio-temporal pressure field
`p(t, x, y, z)`.

```python
import numpy as np

# Define excitation pulse
fs = 200e6
fc = tx.fc
n_cycles = 2
t_pulse = np.arange(0, n_cycles / fc, 1 / fs)
excitation = np.sin(2 * np.pi * fc * t_pulse)

sim = PyField(tx, fs=fs)
x, y, z, p = sim(
    field_points,
    method="auto",
    excitation=excitation,
)
# p.shape == (Nt, Nx, Ny, Nz)
```

**When to use**: realistic pulse propagation, tissue heating estimates,
waveform analysis.

![Transient wavefront animation — pulsed linear array](../examples/assets/pressure_field_video.gif)

## Choosing a computation method

| Method | Description | Best for |
|--------|-------------|----------|
| `"naive"` | Direct sample-by-sample evaluation | Small problems, reference |
| `"sdi"` | Sparse Delta Integration | Large/dense field grids |
| `"auto"` | Automatic selection | General use (recommended) |

The `"auto"` method examines the problem size and selects whichever approach
will be faster. In practice, always use `"auto"` unless you have a specific
reason to force one method.

## Field grid tips

- Use a **dict** for regular grids (most common):
  ```python
  {"x_extent": [-5, 5], "y_extent": [0, 0], "z_extent": [5, 50],
   "dx": 0.1, "dy": 0, "dz": 0.2}
  ```
- Set one extent to `[v, v]` with `d=0` to compute a 2D plane
- Finer grids give smoother plots but use more memory and time
- PyField warns if the estimated memory exceeds safe limits
