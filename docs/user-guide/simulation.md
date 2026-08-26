---
icon: lucide/audio-lines
---

# Simulation

eSDIva splits simulation into two families, both driven by the **SDI** spatial
impulse response engine:

<div class="grid cards" markdown>

-   :lucide-radio-tower: **[Emission](emission.md)**

    ---

    Forward pressure field radiated by the transducer — **monochromatic** CW,
    **transient** pulsed, or with power-law **attenuation**. Returns `p(x, y, z)`
    (CW) or `p(t, x, y, z)` (transient).

-   :lucide-activity: **[Reception (RF)](reception.md)**

    ---

    Pulse-echo **RF** from point scatterers: PSF, phantoms, FMC, and PW/DW
    sequences. Returns per-element channel data `(Erx, Nt)`. Field II-accurate,
    >20× faster on large apertures.

</div>

---

## Field input format

Emission field points and reception scatterer lattices share the same grid dict:

```python
field_points = {
    "x_extent": [-5, 5],    # mm — lateral
    "y_extent": [-0.5, 0.5],  # mm — elevation
    "z_extent": [5, 55],    # mm — axial (depth)
    "dx": 0.1,
    "dy": 1.0,
    "dz": 0.2,
}
```

Set one axis to a single value with its `d=0` to compute a 2-D plane. Emission
also accepts a raw `(N, 3)` mm array; you then reshape the flat output yourself.

## Method selection (SDI)

| Method | Meaning | When |
|--------|---------|------|
| `"auto"` | Picks FST or SDI from grid size | Always recommended |
| `"FST"` | Fully-Sampled Trapezoid — direct evaluation | Small grids, reference |
| `"sdi"` | Sparse Delta Integration — sparse deltas + integration | Large / dense grids |

Both give the same result; SDI is the fast path for large apertures (**>100×** on
emission, **>20×** on reception) with negligible difference from Field II.

## Medium properties

Override defaults in the constructor (shared by `Emission` and `Reception`):

```python
sim = Emission(tx, c=1540, rho=1.0, fs=200e6, alpha0=None)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `c` | 1540 m/s | Speed of sound |
| `rho` | 1.0 kg/m³ | Medium density |
| `fs` | 100 MHz | Sampling frequency |
| `alpha0` | `None` | Power-law attenuation dB/(MHz·cm); `None` disables it |
