# Emission — transmitted pressure fields

`Emission` turns the aperture's spatial impulse response `h(r, t)` into pressure at
every field point. What it returns depends only on how you construct it.

```python
from esdiva.emission import Emission
sim = Emission(tx, c=1540.0, rho=1.0, fs=100e6, alpha0=None, verbose=True)
p, coords = sim(field_points_mm, method="auto")
```

## What an emission field is — and is not

Every `Emission` result is a **linear** wave in a **homogeneous, lossless-or-power-law
fluid**: one sound speed `c`, one density `ρ₀`, no medium map, no boundaries except the
rigid baffle in the aperture plane. Straight rays at that single `c` set every delay. So
the following are outside the method, not missing features:

- **Skull, bone, layered tissue, fat/muscle interfaces, any `c(r)` or `ρ(r)` map** —
  hence no refraction, no reflection, no transmission loss, no aberration or
  skull-induced defocusing. A "transcranial" beam cannot be simulated here; the beam
  eSDIva gives you is the free-field one that would exist without the skull.
- **Nonlinear propagation** — no harmonics, no shock, no saturation, at any drive
  amplitude. The chain is one convolution.
- **Cavitation, heating, radiation force, streaming** — not wave-field quantities.
  `I_SPTA` and a peak-negative pressure *can* be formed from the linear field, but they
  are free-field, non-derated, non-saturating numbers: they overestimate what a real
  path delivers. Do not present them as a safety index (MI/TI) for a real exposure.
- **Standing waves, reverberation, a reflecting wall or interface** — the field
  radiates outward into an unbounded medium and never comes back.

Modelling caveats that shape accuracy rather than forbid a study:

- **Rigid (hard) baffle** *(not yet configurable)*. The aperture plane is rigid — the `1/2πR` in the SIR is
  exactly that assumption. Real probes sit in a soft/inactive housing, so eSDIva is
  optimistic at large angles off the normal, and radiation behind the array is absent.
  A soft-baffle / obliquity option (Field II's `xdc_baffle`) is a planned addition, not a
  limit of the method.
- **Far-field trapezoid per patch.** A patch's SIR is the trapezoid seen from far
  *relative to that patch* (direction cosines and one centre distance). The remedy is
  subdivision: shrink patches until the field stops moving. Accuracy is therefore worst
  right on the aperture face — another reason `z_extent` should start away from 0.
- **Sub-sample patches are widened to one sample** *(not yet exact)*. A patch whose SIR is narrower than
  `1/fs` is stretched to one bin (area conserved). Under-sampling shows up as amplitude
  error, not as an obvious artefact — this is why `fs` must be 20–50× `fc`.
- **An elevation lens is pure geometry** *(a lens material layer is not yet modelled)*. A "lens" is a curved aperture surface, not a
  refracting material layer: no lens sound speed, no lens attenuation, no reverberation
  inside it.
- **The transducer is an ideal velocity source.** No electrical impedance, no element
  crosstalk, no mechanical resonance beyond whatever you supply as
  `tx.impulse_response` — and that response is one array for the whole aperture
  (per-element impulse responses are *not yet* supported; per-element *excitation*
  already is, via an `(L, E)` array).
- **Attenuation is one global power law** (`alpha0`, `freq_power`) applied over the whole
  path — a per-region attenuation map is *not yet* available — and it is ignored when
  `excitation=None`.
- **Pressures are uncalibrated unless you calibrate them.** With the default `rho=1.0`
  (Field II convention) and an arbitrary excitation, the field is linear-scale and
  correct in *relative* terms; real pascals need a real `ρ₀` and a drive in m/s.

If a request needs a heterogeneous medium or a nonlinear term, say so before writing
code, then offer the free-field question eSDIva does answer (beam width, focal gain,
depth of field, aperture design) and point to a full-wave solver (k-Wave, Stride, an
FDTD/pseudospectral code) for the rest. Full table with the reason for each exclusion:
`references/physics.md` § "What eSDIva cannot compute".

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
