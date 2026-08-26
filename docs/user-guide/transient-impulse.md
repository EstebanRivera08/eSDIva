---
icon: lucide/waves
---

# Transient Impulse Simulation

Transient impulse mode returns the raw spatial impulse response `h(t, x, y, z)` —
no excitation convolution. This is the pure geometric SIR sampled at `fs`.

```python
sim = Emission(tx, fs=200e6)            # excitation=None, monochromatic=False
p, coords = sim(field_points, method="auto")
# dict input : p.shape == (Nt, Nx, Ny, Nz), coords has "x","y","z","t0","dt"
# (N,3) input: p.shape == (Nt, N_points)
```

## When to use

- Inspecting the raw SIR / arrival times at chosen field points.
- Debugging transducer geometry.
- Applying a custom convolution outside SonDI.

For a full pulsed wavefront, add an excitation → [Transient + Excitation](transient-excitation.md).
