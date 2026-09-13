# Emission — transmitted pressure fields

`Emission` turns the aperture's spatial impulse response `h(r, t)` into pressure at
every field point. What it returns depends only on how you construct it.

```python
from esdiva.emission import Emission
sim = Emission(tx, c=1540.0, rho=1.0, fs=100e6, alpha0=None, verbose=True)
p, coords = sim(field_points_mm)   # method: None = measured-fastest (see "Method")
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

- **Baffle: rigid by default, soft on request.** `tx.baffle = "rigid"` (default, the
  `1/2πR` Rayleigh integral with no obliquity) or `"soft"` (pressure-release housing,
  Field II's `xdc_baffle(Th, 1)`): each **patch** is weighted by `cosθ = max(0, n·u)`
  between its own normal and the direction to the point — per patch, not per element
  (per-element weighting is wrong in the near field and on curved apertures, up to −62 %
  / −42 % measured), zero behind the patch. Applies on transmit and, by reciprocity, on
  receive; both SIR methods support it (not `ReceptionPaired`).
- **Far-field trapezoid per patch.** A patch's SIR is the trapezoid seen from far
  *relative to that patch* (direction cosines and one centre distance). The remedy is
  subdivision: shrink patches until the field stops moving. Accuracy is therefore worst
  right on the aperture face — another reason `z_extent` should start away from 0.
- **Sub-sample patches (temporal SIR only).** The sampled SIR widens a patch whose SIR is
  narrower than `1/fs` to one bin (area conserved), which multiplies its spectrum by
  exactly `sinc(πf/fs)` per clamped axis (0.4 % at 5 MHz / 100 MHz on axis, 2.5 % at
  40 MHz). The spectral SIR is exact (no clamp). Keep `fs` ≥ ~18× the highest in-band
  frequency for < 1 % on the temporal path.
- **An elevation lens is pure geometry** *(a lens material layer is not yet modelled)*. A "lens" is a curved aperture surface, not a
  refracting material layer: no lens sound speed, no lens attenuation, no reverberation
  inside it.
- **The transducer is an ideal velocity source.** No electrical impedance, no element
  crosstalk, no mechanical resonance beyond whatever you supply as
  `tx.impulse_response` — and that response is one array for the whole aperture
  (per-element impulse responses are *not yet* supported; per-element *excitation*
  already is, via an `(L, E)` array).
- **Attenuation is one global power law** (`alpha0`, `freq_power`) applied over the whole
  path — a per-region attenuation map is *not yet* available.
- **Pressure is in pascals** for `rho` in kg/m³ and a velocity pulse in m/s, independent
  of `fs` (pinned by a Rayleigh test: on-axis far field of a small piston equals
  `ρ·A/(2πz)·v'(t − z/c)` within 1 %). With the default `rho=1.0` (Field II convention)
  the field is Pa per (kg/m³); set a real `ρ₀` and a real drive for absolute numbers.

If a request needs a heterogeneous medium or a nonlinear term, say so before writing
code, then offer the free-field question eSDIva does answer (beam width, focal gain,
depth of field, aperture design) and point to a full-wave solver (k-Wave, Stride, an
FDTD/pseudospectral code) for the rest. Full table with the reason for each exclusion:
`references/physics.md` § "What eSDIva cannot compute".

## The four modes

| Constructor | Physics computed | Shape | Field II equivalent |
|---|---|---|---|
| `Emission(tx, monochromatic=True)` | `ρ₀·ω_c·\|H(r, ω_c)\|` — pressure amplitude at exactly `fc` per 1 m/s | `(Nx, Ny, Nz)` | `calc_h` → FFT at `fc` |
| `Emission(tx)` | `ρ₀·h(r, t)` — the raw SIR | `(Nt, Nx, Ny, Nz)` | `calc_h` |
| `Emission(tx, fs=..., excitation=e)` | `ρ₀ · d(e ⊛ ir_tx)/dt ⊛ h(r, t)` — pulsed pressure | `(Nt, Nx, Ny, Nz)` | `calc_hp` |
| `Emission(tx, fs=..., excitation=e_LE)` | same, one excitation per element, `e_LE` is `(L, E)` | `(Nt, Nx, Ny, Nz)` | `calc_hp` per element |

The time derivative is the physics: pressure follows the *velocity* derivative, so a
transient field is the differentiated drive convolved with the SIR. If
`tx.impulse_response` is set it enters this chain at full length (`L + L_ir − 1`);
otherwise the bare excitation is used as the normal velocity.

**Transient pressure is SIGNED** (compression > 0, rarefaction < 0), like pyMUST's
`mkmovie`/`simus` (`irfft`, never its magnitude). Peak negative pressure is `-p.min(0)`;
use `abs`/an envelope for field maps. Monochromatic output is an amplitude (≥ 0).

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

Only the SIR source changes; everything after it (drives, attenuation, baffle, transfer
function) is the same frequency-domain chain:

- `"spectral"` — the closed-form SIR spectrum `H(r, ω)` (per patch: area × two sincs ×
  a delay phasor), exact, evaluated only on the pulse's band (one bin monochromatic).
- `"temporal"` (= `"sdi"`), `"fst"`, `"auto"` — sample `h(r, t)` (FST fills the trapezoid,
  SDI places its corners and integrates twice), then FFT.

`method=None` (default) picks the faster, measured on a 64-element array: **spectral**
for monochromatic fields (2×; 23× with per-element attenuation) and for per-element
drives or attenuation (2.3× — one FFT for all elements instead of one per element);
**temporal** for every other transient (1.3–2.5×, also with attenuation, a transfer
function or a soft baffle — each is one multiply per bin either way). Both give the same
field (≤ ~1 %); pin a method to benchmark or to reproduce a reference.

## Attenuation

`alpha0` is the power-law coefficient in dB/(cm·MHz^y) and `freq_power` is `y`;
`None` disables attenuation (the default). The transfer function is causal
(Kramers–Kronig consistent), so it disperses the pulse as well as damping it — a
pulse under attenuation arrives slightly reshaped, not merely smaller. Dispersion is
referenced at `fc`: the phase speed at `fc` is exactly your `c`, and higher frequencies
travel slightly faster (tissue-like positive dispersion).

Path origin: `fast_attenuation=True` (default) measures every path from the transducer
centre; `False` uses each element's centre (per-element paths, better near field).
Attenuation applies in every mode, including the raw SIR (`excitation=None`).

## Derived quantities

Peak pressure and intensity come straight from the transient field:

```python
peak  = np.max(np.abs(p), axis=0)                       # spatial peak pressure (Pa)
pnp   = -np.min(p, axis=0)                              # peak negative pressure (Pa)
ispta = np.sum(p ** 2, axis=0) * coords["dt"] / (rho * c) / PRP   # I_SPTA
```

with `PRP` the pulse repetition period. There is no hidden normalisation: pressure
is in pascals for `rho` in kg/m³ and an excitation in m/s, whatever `fs`.

## Plotting

```python
from esdiva.plotting import plot2D_pressure_slices, plot2D_transient_slices
plot2D_pressure_slices(p, coords=coords, db_scale=True)   # mono 3-D or transient 4-D
plot2D_transient_slices(p, coords=coords)                 # transient planes
```

3-D: `plot3D_pressure_vol`, `plot3D_pressure_slices`, `plot3D_transient_slices`
(PyVista). Movies: `save_pyvista_movie`, `save_matplotlib_animation` (needs the
`video` extra). In a notebook pass `notebook=True` / install the `jupyter` extra.
