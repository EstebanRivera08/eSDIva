# The physics eSDIva computes

Written for the ultrasound researcher who wants to know what the numbers mean and
where they can go wrong. Nothing here is eSDIva-specific dogma — it is the
Tupholme–Stepanishen framework the package implements.

## The spatial impulse response

For a baffled aperture `S` radiating into a homogeneous lossless fluid, the velocity
potential at a field point `r` from a normal velocity `v(t)` uniform over the
aperture is a convolution,

    φ(r, t) = v(t) ⊛ h(r, t),    h(r, t) = ∫_S δ(t − |r − r_s|/c) / (2π|r − r_s|) dS

and the pressure is `p = ρ₀ ∂φ/∂t`. The **spatial impulse response** `h(r, t)` is
pure geometry: it is the area of the aperture lying on the sphere of radius `ct`
centred on `r`, divided by that radius. All of the diffraction physics — near-field
structure, edge waves, the transition to the far field — is in `h`; the transducer's
electro-acoustics is entirely in `v`.

Two consequences you feel immediately:

- `h` is a sequence of **sharp edges**, not a smooth band-limited signal. Its
  discontinuities occur when the expanding sphere first touches, crosses, or leaves
  an aperture edge. This is why `fs` must be far above the pulse bandwidth
  (100–200 MHz for a few-MHz probe) even though the *pressure* is band-limited.
- On the aperture face `h` is singular, and it changes rapidly within a fraction of
  an element width. Field points there are numerically hostile and physically
  uninteresting; start the grid a little away from `z = 0`.

## Patches: exact pieces, approximate assembly

eSDIva evaluates `h` exactly for a flat **rectangular patch**, then sums over
patches. The exactness is per patch; the approximation is the assumption that the
aperture is well represented by that tiling and that the drive is uniform across a
patch. So:

- More subdivision → smaller patches → the SIR converges. The convergence check
  (double `no_sub_x`/`no_sub_y`, confirm the field barely moves) is the only honest
  way to justify a subdivision setting.
- A curved aperture (bowl, convex array) tiled with flat rectangles has a *stair-cased*
  rim. That is why the circular classes expose `ratio_big_patches` and
  `refine_factor`: refine where the boundary is misrepresented, not everywhere.
- Cost is `O(field points × patches)`. Doubling subdivision in both directions
  quadruples the runtime.

## FST vs SDI — same integral, different bookkeeping

- **FST** (Fully Sampled Trapezoid) evaluates the patch SIR on every output sample.
  It is the classic Field II-style approach and the reference implementation.
- **SDI** (Sparse Delta Integration) exploits the fact that the trapezoidal patch
  SIR is piecewise linear: its *second* derivative is a handful of delta functions
  at the geometric breakpoints. SDI places only those deltas and integrates twice
  (a double cumulative sum) to recover the SIR.

Under identical assumptions the two produce the same SIR. SDI's cost is set by the
number of breakpoints, not by the number of samples, so it wins as grids and
sampling rates grow. `method="auto"` chooses per problem; pin a method only to
benchmark or to reproduce a published reference.

One numerical caveat worth knowing when comparing runs: the double cumulative sum
accumulates in float64 but stores float32, so SDI and FST agree to ~0.004 % of peak
rather than to the last bit. Compare fields with a relative tolerance, never with
`atol=0`.

## Emission: from SIR to pressure

`p(r, t) = ρ₀ · ∂/∂t [ e(t) ⊛ ir_tx(t) ] ⊛ h(r, t)`. The derivative is not a
convention — pressure responds to the *acceleration* of the surface, so a symmetric
drive gives an antisymmetric pressure pulse. With `excitation=None` the simulator
returns `ρ₀·h` itself, which is the right object to compare against Field II
`calc_h` but is **not** a pressure waveform.

Monochromatic mode is `|H(r, ω_c)|`, the magnitude of the SIR's Fourier transform at
the centre frequency: the steady-state CW amplitude map. It contains no time axis
and no pulse shape, so it answers beam-width and depth-of-field questions and cannot
answer time-of-flight ones.

## Pulse-echo: where the third derivative went

The pulse-echo signal from a point scatterer is

    v_pe = ρ₀/2c₀² · E_m ⊛ ∂³v/∂t³ ,    h_two-way = h_tx ⊛ h_rx

Three time derivatives appear: one from emission, one from the scattering, one from
reception. In practice you never apply them explicitly, because a physical
excitation reaches the medium through the transmit and receive impulse responses,
and `E_m ⊛ ∂³v/∂t³ ∝ e ⊛ h_e ⊛ h_r` — the band-limited pulse model already carries
them. eSDIva and Field II share this convention, so `pulse_echo_rf` equals Field II
`calc_scat` (`≡ calc_hhp` for a unit point) without any correction factor.

The practical corollary is the imaging checklist's first rule: **without impulse
responses the pulse model is wrong**, not merely unrealistic. Driving with a bare
excitation and no `impulse_response` leaves the aperture's diffraction tails
dominating the spectrum, which widens the PSF by roughly 60 % and lifts the
sidelobes. If a simulated PSF looks too wide, check this before blaming geometry.

## Time origin and the beamforming reference

`coords["t0"]` is **not** the instant of the first sample in a raw sense — it is
chosen so that an echo peaks at its geometric round-trip time. Two things have
already been removed: the transmit bulk delay (`delays.max()`, so the axis is
beam-axis referenced) and the two-way pulse lag. A beamformer therefore samples at
`(t_tx + t_rx − t0)·fs` with no lag correction. Adding one back is the single most
common way to get an image that is sharp but axially displaced by half a pulse
length. USTB's `initial_time` and MUST's `dasmtx` use the same convention; raw
Field II `calc_scat` does not, which is what `t_offset_s` exists for.

## Attenuation

Real tissue attenuates as a power law, `α(f) = α₀ f^y` with `y ≈ 1–1.5` in soft
tissue. A pure amplitude law is non-causal, so eSDIva uses the Kramers–Kronig
consistent form: the same `α₀`, `y` also fix the frequency-dependent phase velocity.
The consequence is physical — an attenuated pulse is *reshaped and delayed*, not
merely scaled — and it matters when you compare arrival times between attenuating
and non-attenuating runs.

`y = 1` is a removable singularity in the dispersion relation (`tan(yπ/2)` diverges),
so it is handled by its own branch; if you are testing attenuation, test `y = 1`
separately from `y ≠ 1`.

Attenuation enters through the excitation convolution, so a run with
`excitation=None` ignores it.

## Coordinate frame

X lateral (across the elements), Y elevation (out of the imaging plane), Z axial
(depth, the direction the aperture radiates). The array sits at `z = 0` looking
toward `+z`. A focal point with `z > 0` converges; `z < 0` is a virtual source
behind the aperture, i.e. a diverging wave. Public API is millimetres, internals are
SI, and `element_centers` is one of the few internals users touch directly — it is
in metres.

## Sampling and grid choices, in one place

| Choice | Rule of thumb | Symptom if wrong |
|---|---|---|
| `fs` | 20–50× `fc` (100–200 MHz typical) | quantised SIR edges, jagged pulse, wrong peak amplitude |
| `no_sub_x`/`no_sub_y` | patch ≲ λ/2; verify by doubling | field changes when refined; ringing near the aperture |
| Field-grid `dx`, `dz` | ≲ λ/4 to resolve a beam profile | aliased lobe structure, missed nulls |
| `z_extent` start | keep off the aperture face | huge values, near-field noise |
| Scatterer density | 5–10 per resolution cell | grainy non-Rayleigh "speckle" |
| Decimation | `downsampling=` after simulation | (lowering `fs` instead corrupts the SIR) |

## Field II correspondence

| eSDIva | Field II |
|---|---|
| `Emission(tx)` (no excitation) | `calc_h` |
| `Emission(tx, monochromatic=True)` | `calc_h` → FFT → `fc` bin |
| `Emission(tx, excitation=e)` | `calc_hp` |
| `Reception.pulse_echo_rf` | `calc_scat` (≡ `calc_hhp` for a unit point) |
| `Reception.synthetic_aperture_rf` | `calc_scat_all` |
| `Reception.scan_focusline` | `calc_scat` with receive focusing |

With `rho=1.0` (the default) the non-pressure quantities match numerically to
floating-point precision.

## When a result surprises you

Do not name a cause you have not tested. "Those sidelobes are the element pitch" or
"that artefact is the missing impulse response" are hypotheses until a control run
excludes the alternatives — change one thing (pitch, subdivision, pulse model,
frequency) and see whether the feature moves as that cause predicts. A physics claim
recorded without its discriminating test is an opinion that will outlive the session
and mislead the next reader.
