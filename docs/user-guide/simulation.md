---
icon: lucide/audio-lines
---

# Simulation

PyField supports two simulation modes — monochromatic and transient — both driven by the same `PyField` simulator class.

<div class="grid cards" markdown>

-   :lucide-activity: **[Monochromatic](monochromatic.md)**

    ---

    Steady-state continuous-wave pressure field `p(x, y, z)`. Fast and ideal for beam pattern analysis and transducer design comparison.

-   :lucide-waves: **[Transient Impulse](transient-impulse.md)**

    ---

    Pulsed simulation using only the spatial impulse response (no excitation convolution). Returns `h(t, x, y, z)`.

-   :lucide-radio: **[Transient + Excitation](transient-excitation.md)**

    ---

    Full time-domain simulation: SIR convolved with a user-defined excitation pulse. Returns `p(t, x, y, z)`.

</div>

---

## Common interface

All modes use the same entry point:

```python
from pyfield.psimulation import PyField

sim = PyField(tx)
p, coords = sim(field_points, method="auto")                # monochromatic
p, coords = sim(field_points, method="auto", monochromatic=False)  # transient impulse
p, coords = sim(field_points, method="auto", excitation=pulse)  # transient + excitation
```

## Method selection

| Method | Description | When to use |
|--------|-------------|-------------|
| `"auto"` | Automatic selection | Always recommended |
| `"FST"` | Sample-by-sample evaluation | Small grids, reference results |
| `"sdi"` | Sparse Delta Integration | Large or dense field grids |

## Field input format

### Structured grid (dict)

```python
field_points = {
    "x_extent": [-5, 5],    # mm
    "y_extent": [-0.5, 0.5],
    "z_extent": [5, 55],
    "dx": 0.1,
    "dy": 1.0,
    "dz": 0.2,
}
```

Set one dimension to a single value with `d=0` to compute a 2-D plane.

Returns `p.shape = (Nx, Ny, Nz)` (monochromatic) or `(Nt, Nx, Ny, Nz)` (transient), with `coords` containing `"x"`, `"y"`, `"z"` arrays.

### Raw point array

```python
pts = np.array([[x1, y1, z1], [x2, y2, z2], ...])  # (N, 3) in mm
p, coords = sim(pts, method="auto")
# p.shape == (N_points,)  — monochromatic
# p.shape == (Nt, N_points)  — transient
```

Raw input returns a flat pressure array. User handles reshaping. `coords` omits `"x"`, `"y"`, `"z"` (only `"t0"`, `"dt"` for transient). Use dict input or `compute_mesh=True` for automatic grid construction.

## Medium properties

Override defaults in the `PyField` constructor:

```python
sim = PyField(tx, c=1540, rho=1.0, fs=200e6, alpha0=0)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `c` | 1540 m/s | Speed of sound |
| `rho` | 1.0 kg/m³ | Medium density |
| `fs` | 200 MHz | Sampling frequency |
| `alpha0` | 0 dB/(MHz·cm) | Attenuation coefficient |
