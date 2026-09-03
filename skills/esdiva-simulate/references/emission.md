# Emission — transmitted pressure fields

`Emission` turns the aperture's spatial impulse response `h(r, t)` into pressure at
every field point. What it returns depends only on how you construct it.

```python
from esdiva.emission import Emission
sim = Emission(tx, c=1540.0, rho=1.0, fs=100e6, alpha0=None, verbose=True)
p, coords = sim(field_points_mm, method="auto")
```

## The four modes

| Constructor | Physics computed | Shape | Field II equivalent |
|---|---|---|---|
| `Emission(tx, monochromatic=True)` | `\|H(r, ω_c)\|` — CW amplitude at `fc` | `(Nx, Ny, Nz)` | `calc_h` → FFT → `fc` bin |
| `Emission(tx)` | `ρ₀·h(r, t)` — the raw SIR | `(Nt, Nx, Ny, Nz)` | `calc_h` |
| `Emission(tx, fs=..., excitation=e)` | `ρ₀ · d(e ⊛ ir_tx)/dt ⊛ h(r, t)` — pulsed pressure | `(Nt, Nx, Ny, Nz)` | `calc_hp` |
| `Emission(tx, fs=..., excitation=e_LE)` | same, one excitation per element, `e_LE` is `(L, E)` | `(Nt, Nx, Ny, Nz)` | `calc_hp` per element |

The time derivative is the physics: pressure follows the *velocity* derivative, so a
transient field is the differentiated drive convolved with the SIR. If
`tx.impulse_response` is set it enters this chain; otherwise the bare excitation is
used as the normal velocity.

Monochromatic mode is a spatial map, not a snapshot in time — it answers "how strong
is the beam here", not "where is the wavefront now". Use it for beam profiles,
−6 dB widths, depth of field. Use transient mode for wavefronts, pulse shape,
time-of-flight, and anything feeding reception.

## The field grid

A dict in millimetres. Extents are inclusive endpoints; a zero-width axis with a
zero step collapses that dimension to a single plane.

```python
field_points = {"x_extent": [-10, 10], "y_extent": [0, 0], "z_extent": [0.5, 50],
                "dx": 0.1, "dy": 0, "dz": 0.2}
```

Cost is linear in the number of points × the number of patches. Prototype on the
XZ plane (`y_extent: [0, 0]`) before asking for a volume. Keep `z_extent` away from
0: the SIR is singular on the aperture face, and the near field within a fraction of
an element width is not physically meaningful anyway.

## Return contract

- `coords["x"]`, `coords["y"]`, `coords["z"]` — axis vectors in mm.
- Transient only: `coords["t0"]` (start of the time axis, s) and `coords["dt"]`
  (= `1/fs`). Rebuild the axis with
  `t = coords["t0"] + np.arange(p.shape[0]) * coords["dt"]`.
- `sim.time_log` — wall time split into `time_grid_s` / `hsir_s` / `fft_s`, i.e.
  geometry vs. signal processing. Read it before optimising anything.

## Method

`method="auto"` (default) picks between `"FST"` — the fully sampled trapezoid, the
classic Field II-style evaluation — and `"sdi"`, the sparse delta integration that
places only the SIR's breakpoints and integrates twice. Both compute the *same* SIR
under the same assumptions; SDI is much cheaper on large grids. Pin the method only
to benchmark or to reproduce a reference.

## Attenuation

`alpha0` is the power-law coefficient in dB/(cm·MHz^y) and `freq_power` is `y`;
`None` disables attenuation (the default). The transfer function is causal
(Kramers–Kronig consistent), so it disperses the pulse as well as damping it — a
pulse under attenuation arrives slightly reshaped, not merely smaller.

`fast_attenuation=True` applies one shared transfer function; setting it `False`
forces the per-element loop, which is slower but correct when elements see
different path lengths. Attenuation is applied through the excitation convolution,
so in mode 2 (`excitation=None`) it is ignored — pass an excitation if attenuation
must apply.

## Derived quantities

Peak pressure and intensity come straight from the transient field:

```python
peak  = np.max(np.abs(p), axis=0)                       # spatial peak pressure (Pa)
ispta = np.sum(p ** 2, axis=0) * coords["dt"] / (rho * c) / PRP   # I_SPTA
```

with `PRP` the pulse repetition period. There is no hidden normalisation: pressure
is in pascals for `rho` in kg/m³ and an excitation in m/s.

## Plotting

```python
from esdiva.plotting import plot2D_pressure_slices, plot2D_transient_slices
plot2D_pressure_slices(p, coords=coords, db_scale=True)   # mono 3-D or transient 4-D
plot2D_transient_slices(p, coords=coords)                 # transient planes
```

3-D: `plot3D_pressure_vol`, `plot3D_pressure_slices`, `plot3D_transient_slices`
(PyVista). Movies: `save_pyvista_movie`, `save_matplotlib_animation` (needs the
`video` extra). In a notebook pass `notebook=True` / install the `jupyter` extra.
