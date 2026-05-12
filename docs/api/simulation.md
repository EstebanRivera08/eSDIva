---
icon: lucide/audio-lines
---

# Simulation

## PyField

`PyField` wraps the SIR kernel and provides a callable interface.

```python
from pyfield.psimulation import PyField

sim = PyField(transducer)
pressure, coords = sim(field_points_mm, method="auto")
```

### Arguments

| Parameter | Type | Description |
|-----------|------|-------------|
| `transducer` | `TransducerBase` | Any configured transducer |

### `__call__(field_points_mm, *, method, excitation)`

| Parameter | Type | Description |
|-----------|------|-------------|
| `field_points_mm` | `dict` or `(N,3) ndarray` | Structured grid dict or raw point coordinates in mm |
| `method` | `str` | `"naive"`, `"SDI"`, or `"auto"` |
| `excitation` | `ndarray or None` | Excitation pulse for transient simulation |

**Return values**

| Symbol | Shape | Description |
|--------|-------|-------------|
| `pressure` | see below | Pressure field |
| `coords` | `dict` | Coordinate metadata (see below) |

**Structured grid** (dict input):

- Monochromatic: `p.shape = (Nx, Ny, Nz)`, `coords = {"x", "y", "z"}`
- Transient: `p.shape = (Nt, Nx, Ny, Nz)`, `coords = {"x", "y", "z", "t0", "dt"}`

**Raw point array** (`(N,3)` ndarray input):

- Monochromatic: `p.shape = (N_points,)`, `coords = {}`
- Transient: `p.shape = (Nt, N_points)`, `coords = {"t0", "dt"}`

User handles reshaping for raw input. Use dict input or `compute_mesh=True` for automatic grid construction.

### Methods

| Name | Description |
|------|-------------|
| `"naive"` | Direct evaluation — O(M·P) |
| `"sdi"` | Sparse Delta Integration — same result, faster for dense grids |
| `"auto"` | Picks the faster method based on `M` and `P` |

---

## Simulation methods

Both methods compute the same transducer SIR by summing the contribution of
each patch `m` over all `M` patches:

```
h_tx(r_p, t) = Σ_m  a_m · h_m(r_p, t − τ_m)
```

where `a_m` and `τ_m` are the apodization weight and delay of patch `m`, and
`h_m` is its spatial impulse response — a **trapezoid** in time whose four
corners are set by the patch geometry and the field-point direction cosines:

```
t₁ = l/c − (Δt₁ + Δt₂)/2
t₂ = t₁ + Δt₁          Δt₁ = |xᵤ| · wx / c
t₃ = t₁ + Δt₂          Δt₂ = |yᵤ| · wy / c
t₄ = t₁ + Δt₁ + Δt₂
```

`l` is the patch-to-field-point distance, `(xᵤ, yᵤ)` are direction cosines,
and `wx`, `wy` are the physical patch half-widths.

### Naive method

Fills the trapezoid sample-by-sample for every `(patch, field-point)` pair.
Straightforward but scales as O(M · P · T_trap) where `T_trap` is the number
of samples within `[t₁, t₄]`.

### Sparse Delta Integration (SDI)

The trapezoid is the double time-integral of four weighted Dirac deltas.
Taking two derivatives of `h_m`:

```
∂h_m/∂t   = s · [u(t−t₁) − u(t−t₂) − u(t−t₃) + u(t−t₄)]

∂²h_m/∂t² = s · [δ(t−t₁) − δ(t−t₂) − δ(t−t₃) + δ(t−t₄)]
```

where `s = h_max / (t₂ − t₁)` is the trapezoid slope and `u`, `δ` are the
Heaviside step and Dirac delta.

Instead of filling the trapezoid, SDI places **4 weighted delta samples** per
patch into a sparse accumulator and then integrates twice.  The inner loop
touches only 4 time samples per `(patch, field-point)` pair regardless of
`T_trap`, which gives a large speedup when the field grid is dense.

```
for each (patch m, field point p):
    place ±s at t₁, t₂, t₃, t₄  →  d²h accumulator
cumsum twice  →  h_tx
```

The `"auto"` method selects naive or SDI based on the ratio of M·P to the
expected trapezoid width, choosing whichever is faster for the given problem.

---

## Monochromatic simulation

Set up the transducer and call without an excitation signal.  The output is a
3-D complex pressure field (envelope).

```python
sim = PyField(tx)
p, coords = sim(pts_mm, method="auto")
# Structured (dict): p.shape == (Nx, Ny, Nz)
# Raw array:         p.shape == (N_points,)
```

**Linear array — XZ plane (Domino probe)**

![Monochromatic pressure — linear array 2D](../examples/assets/lineartx_monochromatic.png)

**Linear array — 3-D isosurface view (pressure)**

![Monochromatic pressure — linear array 3D](../examples/assets/linear_array_pressure_field.png)

**Matrix array — 3D isosurface view (pressure + TX)**

![Monochromatic pressure — matrix array 2D](../examples/assets/matrix_array_field.png)

---

## Transient simulation

Pass a time-domain excitation pulse.  The output is a 4-D real pressure field
with time along axis 0.

```python
import numpy as np

fs = 100e6           # sampling frequency
fc = tx.fc           # centre frequency
n_cycles = 3
t_pulse = np.arange(0, n_cycles / fc, 1 / fs)
excitation = np.sin(2 * np.pi * fc * t_pulse) * np.hanning(len(t_pulse))

p, coords = sim(pts_mm, method="auto", excitation=excitation)
# Structured (dict): p.shape == (Nt, Nx, Ny, Nz)
# Raw array:         p.shape == (Nt, N_points)
```

---

## Field grid helper

Build a dense 3-D grid of points:

```python
import numpy as np

x = np.linspace(-5, 5, 40)
y = np.array([0.0])          # single plane
z = np.linspace(5, 60, 120)
pts = np.array(np.meshgrid(x, y, z)).T.reshape(-1, 3)
```

---

## Delta-k diagnostic

After a simulation, inspect `sim.sub_elem_delta_k` to verify the SIR
accuracy condition for every patch and field point.  Use
`plot_deltak_distribution(sim)` for a visual summary.

```python
from pyfield.utilities import plot_deltak_distribution

fig = plot_deltak_distribution(sim, per_element=True)
```
