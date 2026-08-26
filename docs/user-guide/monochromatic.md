---
icon: lucide/activity
---

# Monochromatic Simulation

Monochromatic mode computes the steady-state pressure amplitude for a continuous
wave at the transducer centre frequency `fc`. The SIR is evaluated per field point
and its Fourier component at `fc` is taken — returning a 3-D amplitude field
`p(x, y, z)`.

```python
from esdiva.emission import Emission

sim = Emission(tx, monochromatic=True)
p, coords = sim(field_points, method="auto")
# dict input : p.shape == (Nx, Ny, Nz), coords has "x", "y", "z"
# (N,3) input: p.shape == (N_points,), you reshape
```

![Monochromatic pressure — matrix array field](../examples/assets/ex03_matrix_array_field.png)

## When to use

- Beam-pattern analysis and −6 dB focal-spot characterisation.
- Comparing transducer geometries or subdivision settings.
- Fast field preview before a full transient run.

See [Example 3 — Multi-element 3-D](../examples/example03_multielements_monochromatic_CW.md).
