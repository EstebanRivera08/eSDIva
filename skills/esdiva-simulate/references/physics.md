# The physics eSDIva computes

Written for the ultrasound researcher who wants to know what the numbers mean and
where they can go wrong. Nothing here is eSDIva-specific dogma — it is the
Tupholme–Stepanishen framework the package implements.

## What eSDIva cannot compute — read this before designing a study

The whole framework rests on one integral: a **linear** wave propagating in a
**homogeneous, non-scattering fluid** with a single sound speed `c` and a single
density `ρ₀`. Every geometric quantity — the delay to a field point, the SIR
breakpoints, the round-trip time of an echo — is computed as a straight ray at that
one speed. There is no medium map anywhere in the package, so a request that needs
the wave to *change* as it travels cannot be answered, however the script is
written. Refusals below are structural, not missing features — and they are the **same limits
Field II has**, because both implement the same Jensen/Tupholme–Stepanishen model. That
is usually the clearest thing to tell a user: if Field II cannot do it, neither can
eSDIva, for the same reason.

| Not possible | Why the method forbids it |
|---|---|
| Transcranial / through-bone propagation, aberration correction, skull-induced defocusing | Requires a heterogeneous `c(r)`, `ρ(r)` and shear-wave conversion in bone. eSDIva has one scalar `c`, one `ρ₀`, and a fluid-only formulation — no elastic solid, no mode conversion. |
| Refraction, reflection or transmission at a tissue interface (fat/muscle, tissue/bone, lens layers) | An interface is an impedance discontinuity. The SIR integral assumes free-field propagation from the aperture with no boundaries other than the rigid baffle in the aperture plane. |
| Layered or spatially varying media, sound-speed maps, temperature-dependent `c` | `c` is a single constructor scalar used for every delay. |
| Multiple scattering, reverberation, shadowing behind a strong target, clutter from the body wall | Scatterers are independent Born (weak, single-scattering) point targets. Each contributes `amplitude × h_tx ⊛ h_rx` and never sees another scatterer's field. |
| Nonlinear propagation, harmonic imaging, shock formation, mechanical/thermal index for HIFU dosimetry | The whole chain is a linear convolution. There is no second-order term, so no harmonic is generated at any drive level. |
| Absolute pressures in Pa without calibration | The default `rho=1.0` (Field II convention) and an arbitrary excitation amplitude make the output linear-scale but not calibrated. Amplitudes are meaningful in *relative* terms unless the user supplies real `ρ₀` and a calibrated drive. |
| Elastography, shear-wave propagation | The medium is an inviscid fluid — no shear branch, no viscoelasticity. (Flow and Doppler are *not* in this row: scatterers are static within one call, but advancing them between calls, as Field II users do, gives motion and decorrelation.) |
| Streaming, cavitation, radiation force, heating | Not wave-field quantities — they need a nonlinear or thermal solver. |
| Region-dependent attenuation, an attenuation map | `alpha0`/`freq_power` are global scalars applied along the whole propagation distance. |

### "Never" versus "not yet"

Everything in the table above is a property of the *model*, so it will not arrive in a
future release — closing any of those rows means a different solver (k-Wave, Stride, an
FDTD/pseudospectral code), not a new eSDIva version. Field II sits in exactly the same
place.

A few things a user may ask for are instead **current-version gaps**. Say "not yet",
not "impossible", and give the workaround where one exists:

| Not yet | Where it stands today |
|---|---|
| Soft-baffle / obliquity weighting (Field II's `xdc_baffle`) | The rigid baffle is assumed and unavoidable. It flatters response at large angles off the normal; nothing else changes. |
| Per-element impulse responses, a separate receive-electronics transfer function | One `impulse_response` per transducer. Per-element *excitation* already exists on emission (`(L, E)`). |
| Frequency-dependent scatterer amplitude (Rayleigh `f⁴`, scatterer size) | Amplitudes are frequency-flat scalars. Approximate by simulating scatterer classes separately and filtering each result. |
| Moving scatterers inside one `sequence_rf` call | Advance the positions yourself between `pulse_echo_rf` calls (`v/PRF` per emission) and stack — this is how flow is done in Field II. Only the checkpointing convenience is missing. |
| Attenuation that varies by region | `alpha0` is one global power law. |
| A lens as a material layer (lens sound speed, lens loss) | A lens is a curved aperture surface: the focusing geometry is right, the layer physics is absent. |
| Exact sub-sample patch response | A patch whose SIR is narrower than `1/fs` is widened to one sample bin (area conserved). Raise `fs` rather than working around it. |

Noise, TGC, ADC quantisation and element crosstalk are **deliberately** absent, not
pending: the RF is a clean, unamplified signal so the user controls the SNR. Add noise
yourself before quoting any CNR or contrast number.

**What the brain-atlas feature actually does.** eSDIva can register a computed field
onto an anatomical atlas and report, per structure, what fraction of the beam lands
there. The field is still computed in **homogeneous water/tissue**: the skull is not
in the acoustic model, the beam is not defocused or attenuated by bone, and the
overlay is a *targeting* and *coverage* aid, not a transcranial simulation. Saying
otherwise misleads someone planning a real neuromodulation experiment.

**What to say instead.** Name the physical reason (one sentence: "that needs a
heterogeneous medium; eSDIva propagates through a single sound speed"), then offer
the nearest question eSDIva *can* answer — free-field beam shape, focal geometry,
aperture design, PSF, the imaging sequence itself — and point the user at a
full-wave solver (k-Wave, Stride, Kranion, an FDTD/pseudospectral code) for the part
that needs a medium map. Never patch around it by faking a skull as a set of
scatterers or by lowering `c`: single-scattering point targets in a homogeneous
medium do not reproduce refraction, and neither does a global speed change.

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

eSDIva evaluates `h` in closed form for a flat **rectangular patch** — a trapezoid in
time, valid in that patch's own far field (the patch is seen through one centre
distance and two direction cosines) — then sums over patches. Two approximations
therefore sit under every field: the aperture is represented by that tiling with a
uniform drive per patch, and each patch is far enough from the field point for its
trapezoid to hold. Both are controlled by the same knob — subdivision. So:

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
