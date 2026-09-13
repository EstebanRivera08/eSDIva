# Reception — pulse-echo RF, and how to beamform it

`Reception` simulates the signal a receive aperture records when a transmit aperture
insonifies point scatterers. The pulse-echo signal is the two-way SIR convolved with
the drive,

    v_pe = ρ₀/2c₀² · E_m ⊛ ∂³v/∂t³ ,   with   h_two-way = h_tx ⊛ h_rx

and that third derivative is **already baked into** the band-limited excitation and
the TX/RX impulse responses (`E_m ⊛ ∂³v/∂t³ ∝ e ⊛ h_e ⊛ h_r`). Neither eSDIva nor
Field II applies an explicit ∂³ — which is why the output matches Field II
`calc_scat` directly.

```python
from esdiva.reception import Reception
sim = Reception(tx, rx, c=1540.0, rho=1.0, fs=100e6,
                excitation=e, method="spectral", alpha0=None, verbose=True)
```

`tx` and `rx` are separate transducers. If they are physically the same probe pass
`rx = tx.copy()` — sharing one object lets the transmit delays/apodization corrupt
the receive channels, and `sequence_rf` refuses it.

## What this RF is a model of — and what it cannot contain

The scattering model is Jensen's (JASA 89(1), 1991), the one Field II implements:
**weak (first-order Born) scattering from independent point targets in a homogeneous
medium**. Each scatterer contributes `amplitude × (h_tx ⊛ h_rx)` and nothing else.
Everything below is therefore absent from the RF by construction — it will never
appear, no matter how the phantom is built:

- **Multiple scattering, reverberation, body-wall clutter, comet tails** — a scatterer
  never sees another scatterer's field.
- **Shadowing and enhancement.** A dense or strongly reflecting region does not weaken
  the echoes behind it; the acoustic shadow under a stone or a rib is not there.
- **Specular reflectors.** Amplitudes are frequency-independent, angle-independent
  scalars, so there is no impedance-controlled reflection coefficient and no
  mirror-like surface. A bone surface, a needle, a vessel wall or a catheter modelled
  as a sheet of points gives a bright layer, not real specular behaviour (no
  angle-dependent dropout, no reverberation between two interfaces).
- **Frequency-dependent scattering** *(not yet)*. No Rayleigh `f⁴` law, no scatterer-size
  effects — approximate it by simulating scatterer classes separately and filtering each;
  the spectrum of an echo is the pulse spectrum shaped only by diffraction and
  attenuation.
- **Refraction and true 3-D aberration.** One global `c` sets both the simulation and
  the geometry, so there is no skull, no fat layer, no sound-speed map. Two related
  studies *are* supported and worth offering instead: a global speed **mismatch**
  (beamform with `c' ≠ c_sim` — a uniform error, not aberration), and a **near-field
  phase screen** (per-element delay/amplitude errors written into an event's `delays`
  and `apodization`, and channel shifts applied to the returned RX data). A phase
  screen at the aperture is the standard first-order aberration model; what is missing
  is refraction distributed along the propagation path.
- **Motion is resolved between emissions, not within one.** `sequence_rf` accepts a
  moving cloud — `(N_events, N_scat, 3)`, one cloud per emission — so flow, Doppler
  and speckle decorrelation are first-class (see "Moving scatterers" below). What it
  cannot show is an intra-pulse Doppler frequency shift: the scatterers are frozen
  while each echo is formed, and the RF carries only the phase shift *between*
  emissions. That is what every Doppler estimator reads, and it is the same modelling
  boundary Field II works within. The medium is still linear and non-viscous, so this
  buys Doppler and decorrelation, not elastography or shear waves.
- **Nonlinearity.** No harmonic imaging, no pulse-inversion contrast, no microbubbles.
- **Electronics — absent by design, not pending.** No thermal or electronic noise, no
  TGC, no ADC quantisation, no element crosstalk, and one impulse response per
  transducer (per-element responses and a separate receive-electronics transfer
  function are *not yet* supported). The
  RF is a noiseless, unamplified, infinite-dynamic-range signal — add noise yourself if
  the study needs a realistic SNR or a CNR that means something.
- **Absolute echo calibration.** Amplitudes are relative; there is no backscatter
  coefficient in physical units.

Two consequences worth stating to a user out loud, because they change how a result
should be read: a simulated B-mode has **no clutter floor** other than the beam's own
sidelobes and the speckle you created, so contrast numbers are optimistic against a
real scanner; and an anechoic lesion is *perfectly* anechoic, so its measured CNR is
bounded by your scatterer statistics, not by physics.

Attenuation, when enabled, is one global power law over the whole path — a per-region
map is *not yet* available. Full table of exclusions with the reason for each:
`references/physics.md` § "What eSDIva cannot compute".

## The calls

| Call | Returns | Use for |
|---|---|---|
| `pulse_echo_rf(pos_mm, amp)` (= `__call__`) | `(Erx, Nt)` | one transmit, per-channel RF |
| `pulse_echo_rf(pos_mm, per_scatterer=True)` | `(N_scat, Erx, Nt)` | the PSF of each point |
| `sequence_rf(pos_mm, amp, tx_events)` | `(N_events, Erx, Nt)` | PW / DW / multi-event acquisitions |
| `synthetic_aperture_rf(...)` | per-element DW basis | FMC, Field II `calc_scat_all` |
| `scan_focusline(focus_mm, pos_mm, amp, FoverD=...)` | `(Nt,)` | one focused B-mode line |

`scan_focusline` recomputes TX **and** RX focus from `focus_mm` and beamforms on
receive inside the SIR kernel — that is Field II `calc_scat`'s own receive
beamforming: about `E_rx`× cheaper than per-channel RF plus an external sum, and
with no sample-interpolation loss. Loop it over lines to build a conventional B-mode.

`downsampling=N` decimates the time axis with anti-aliasing after simulation. Always
prefer it to lowering `fs`: the SIR needs the fine grid, the output does not.

## `method=` — how the two-way convolution is evaluated

All backends compute the same physics; they differ in cost.

- `"spectral"` (default) — closed-form one-way spectra, `Σ_TX·Σ_RX = F{Δδ_pe}`, no
  forward FFT, cost ∝ M (patches), exact, band-limited bins only. Supports per-patch
  one-way attenuation. Use this.
- `"fst"` / `"sdi"` / `"auto"` — sample both SIRs and FFT-convolve
  (`ReceptionConventional`); the string names its SIR-sampling kernel. Use to
  cross-check `spectral`.
- `ReceptionPaired(tx, rx, ...)` (separate class; `method="paired"` raises) — the pedagogic two-way delta train. Exact, no FFT, but cost ∝ M², so
  far slower; it warns on construction. For teaching or auditing the kernel only.

## Scatterers

Three ways to supply targets:

1. **Explicit points** — `(N, 3)` mm positions plus `(N,)` amplitudes. Wires,
   isolated targets, resolution studies.
2. **A grid dict** — the same `x_extent`/`dx`… form as emission, giving a regular
   lattice of unit points. That is a **PSF map, not a phantom**: a periodic lattice
   returns coherent lattice echoes, not speckle.
3. **A phantom** — for tissue:

```python
from esdiva.utilities import make_phantom
pos_mm, amp = make_phantom(extents_mm, n_scatterers=200_000,
                           echogenicity_map=g, seed=0)
```

Positions are uniform random in the box and amplitudes are `N(0,1)·map(r)`, so an
anechoic region (`map = 0`) is silent and a region of gain `g` has echo energy ∝ `g²`.
Fully developed speckle needs **5–10 scatterers per resolution cell** (cell ≈ λ·F#
laterally × half a pulse length axially); fewer gives a grainy, non-Rayleigh texture
that is not tissue.

`sim.show(pos_mm, amp)` previews the setup in 3-D (both meshes, scatterers faded by
amplitude). Run it before every long acquisition.

## Sequences and checkpointing

An event is a dict carrying `"delays"` and `"apodization"` `(E,)` arrays — the
transmit focal law. Build them from the wavefront you want:

```python
centers = probe.element_centers                  # (E, 3) in METRES
d = np.linalg.norm(centers - vs_mm * 1e-3, axis=1) / c
event = {"delays": (d - d.min()).astype(np.float32),
         "apodization": np.ones(len(centers), np.float32),
         "virtual_source_mm": vs_mm}             # extra key, consumed by das_volume
rf, coords = sim.sequence_rf(pos_mm, amp, events, out_path="rf_store")
```

With `out_path` each event is written the moment it finishes, so a crash costs at
most the event in flight and re-running resumes. The store fingerprints probe,
medium, excitation, scatterers and events — a changed configuration raises with a
diff instead of silently mixing incompatible data. `checkpoint_chunks=N` splits one
event into N scatterer chunks (RF is linear in the scatterers); zero-amplitude
sentinel points pin one common time grid so chunks sum sample-exactly. Size chunks
at roughly 10–15 min each.

`pulse_echo_rf` and `synthetic_aperture_rf` accept the same two arguments and route
through `sequence_rf`.

**Recommend `sequence_rf` for any phantom study, static or moving.** The same result
is reachable by hand — loop `pulse_echo_rf` per transmit (and per cloud, if the
medium moves) and stack — and the physics is identical, so offer that when a user
wants to see the steps. But steer real runs to `sequence_rf`: it is the only path
with on-disk checkpointing (a killed 6-hour job resumes instead of returning
nothing), the config fingerprint that refuses to mix incompatible data,
`checkpoint_chunks=` for bounding loss inside one event, and `t0_per_event`
collected for the beamformer. Reserve `pulse_echo_rf` for a single acquisition or a
PSF map.

```python
from esdiva.io import RFDataset
ds = RFDataset("rf_store"); ds.summary()
rf, coords = ds.load_all()
ds.to_hdf5("channels.h5")      # UFF-compatible fields for MATLAB / USTB
```

`coords["dt"]` is shared; `coords["t0_per_event"]` gives each event its own
beam-axis time origin (the grid depends on that event's delays, and on where its
scatterers are). Beamform each event with **its own** `t0`.

## Moving scatterers (flow, Doppler, decorrelation)

Give `sequence_rf` a `(N_events, N_scat, 3)` stack instead of `(N_scat, 3)` and every
emission sees its own cloud. eSDIva has no slow-time clock, so the trajectory is the
user's to write — for constant flow that is one line:

```python
t = np.arange(n_events) / prf                        # slow-time, s
pos = pos0[None] + v_mm_s[None] * t[:, None, None]   # (N_events, N_scat, 3)
rf, coords = sim.sequence_rf(pos, amp, events, out_path="rf_store")
```

Any pulsatile, cardiac or decorrelating trajectory is the same call. `amplitudes` may
also be per event, `(N_events, N_scat)`: the scatterer count is fixed for the
sequence, so a zero amplitude is how a scatterer that has left the region of interest
— or a destroyed contrast bubble — is retired. Checkpointing works unchanged, and the
fingerprint covers the positions by value, so two different flows cannot be mixed in
one store. `synthetic_aperture_rf` takes the same stack (one cloud per transmit
group) to model motion during a slow FMC acquisition.

Two things to get right, and to tell the user about:

- **Aliasing.** The unambiguous axial velocity is `v_nyquist = c·PRF/(4·fc)`. Beyond
  it the inter-emission phase wraps and the estimate folds.
- **Beam-to-flow angle.** Only the axial component is visible: flow perpendicular to
  the beam produces no Doppler shift at all. A vessel must be tilted.

The scatterers are frozen during each emission, so this models the phase shift
*between* emissions — what every Doppler estimator reads — not an intra-pulse
frequency shift. eSDIva ships no Doppler estimator or clutter filter; `example22`
shows both written in a few lines of numpy.

### Doppler: what was measured, and what was ruled out

These come from `example22`, each with the control that established it. Quote
them as measurements, not as intuition — and do not re-derive the refuted ones.

**Established.**

- **Measure the centre frequency on the BEAMFORMED signal, not the channel RF.**
  Delay-and-sum low-passes the data (interpolation between RF samples plus
  coherent summation across the aperture), so the beamformed echo centres
  *below* the channel echo: 4.27 against 4.46 MHz on a 5 MHz probe. Since
  `v ∝ 1/f`, feeding the channel figure to the estimator biases every velocity
  low by 4.5 %, uniformly, at every radius and every speed. *Test:* a plug-flow
  control went 0.955 → 0.998 of its known velocity, and the beamformed spectrum
  (4.268 MHz) independently matched the frequency the velocity error demanded
  (4.261 MHz) to 0.1 %. This is why Loupas's estimator takes its frequency from
  the same IQ it processes.
- **Never use the probe's nominal frequency.** The worked example measures
  2.45 MHz on a 3 MHz probe — a −18 % velocity bias before anything else goes
  wrong. The ratio held at 0.85, 0.84 and 0.82 of nominal across 5, 12.5 and
  3 MHz rebuilds of it, and was still measured every time rather than assumed.
- **Scale the centroid's integration BAND to the probe, never absolute numbers.**
  A band left over from another probe truncates the echo spectrum and rescales
  every velocity. Measured: `(2, 10) MHz` carried over from a 5 MHz probe cut the
  top off a 12.5 MHz echo and reported 8.2 MHz instead of 10.5 — a 28 % velocity
  error, on top of a pipeline that was otherwise correct.
- **A vessel-core ratio above 1.0 is physically impossible — use it as a free
  check.** A resolution-cell average of a profile that is *peaked* at the core
  cannot exceed the truth there; averaging a maximum with its neighbours can only
  pull it down. A core reading of 1.19× is what exposed the band bug above. Any
  velocity calibration can be sanity-checked this way without a control run.
- **A plug-flow control separates a scale error from a gradient error.** Same
  phantom, geometry, sequence and processing, one velocity for all blood. A flat
  ratio across radius = uniform scale error; the Poiseuille case's 0.80 → 1.47
  pattern is the gradient sitting on top of it. This is the single most useful
  control for any velocity-estimation bug.
- **Elevation is the axis that decides whether a velocity is trustworthy, and it
  is a DESIGN parameter, not just a modelling one.** An unfocused aperture
  averages in out-of-plane blood, which is slower. Measured across a redesign:
  a flat 4 mm aperture at 22 mm (elevation slice ~1.7 mm, 0.56 of the lumen) read
  0.80× at the vessel core and 1.47× at the wall; an elevation LENS focused on the
  vessel, slice 0.36 mm against a 2.0 mm lumen (0.18), read 0.93× / 1.27×. The
  redesign moved frequency, depth and lens together, so it demonstrates rather
  than isolates — but the design lesson stands: size the elevation slice against
  the target before blaming the estimator.
- **`λ·z/H` is NOT established as the elevation width's depth law** — it was
  claimed here and then refuted. It matches numerically at one depth, which is
  the trap. *Test:* a vessel tilted 60° spans 18–26 mm within a single
  acquisition, so only depth varies; the measured core ratio over 19–25 mm is
  flat (+0.016, band scatter ±0.017) where `λ·z/H` predicts +0.048 to +0.096.
  Report a fitted elevation width as measured for that geometry. (Untested
  hypothesis for the discrepancy: an unfocused aperture's far field starts near
  `H²/λ` — 52 mm for 4 mm at 5 MHz — so the cone does not apply at these depths.)
- **Model the elevation average over the BLOOD CHORD only** (`|y| < √(R²−r²)`):
  outside the lumen there is no blood and so no flow signal to average in.
  Ignoring that inflates the residual 1.6× (0.76 → 1.19 cm/s) and biases the
  fitted width low (0.69 → 0.50 mm).
- **Noise must be added before any sensitivity claim.** Noiseless RF gives a
  longer ensemble nothing to average down: velocity scatter fell only
  2.58 → 2.50 cm/s and every sequence detected 100 % of the lumen. Add it with
  `esdiva.utilities.add_noise`, one `reference` shared by every sequence being
  compared.
- **For an image plane cutting a vessel along a diameter, the true mean velocity
  is `(2/3)·v_peak·cosθ`**, not the `(1/2)·v_peak` of a circular cross-section
  average — the voxels sample the radius uniformly. Getting this wrong reports a
  bias that is not there.

**Refuted — do not re-open without new evidence.** Each was tried as the cause of
a uniform 4.5 % deficit and failed: wall-filter order (0 → 2 moves the mean only
6.44 → 6.73 cm/s), a depth-restricted centre frequency (moves the *wrong* way),
receive-aperture angle alone (17° → 7° recovers 2.4 %), and transit-time
decorrelation (flat across a 4× change in displacement per lag; worth +0.3 %).

**A trap in the decimation test:** decimating slow time to probe decorrelation
also halves the Nyquist velocity. Run it on a control slow enough to stay
unambiguous, or the estimate folds and the result is meaningless.

## What the RF output actually is (writing your own beamformer)

The built-in beamformers are a convenience, not the interface. The simulator's job
ends at a fully specified RF array, and every quantity you need to reconstruct it
yourself is in the return values plus the transducer objects. Most users eventually
write their own beamformer — Fourier-domain migration, Stolt/f-k, model-based or
adjoint reconstruction, REFoCUS, sparse recovery, a learned network. Feed them
directly.

**The array.** `rf[..., e, n]` is the voltage on receive element `e` at sample `n`,
in the simulator's pressure units (linear, unnormalised, no TGC, no filtering, no
noise). Axis order is always `[event, receive element, time]`; `pulse_echo_rf` drops
the event axis, `per_scatterer=True` prepends a scatterer axis instead.

**The time axis.** One sample period `dt = coords["dt"] = 1/fs` for everything; the
origin is per event.

```python
t = coords["t0"] + np.arange(rf.shape[-1]) * coords["dt"]     # seconds
t = coords["t0_per_event"][e] + np.arange(rf.shape[-1]) * coords["dt"]
```

Traces are zero-padded at the **end** to a common length, so only the origin differs
between events. `t0` is the **beamforming reference**: it is set so that an echo
peaks at its geometric round-trip time, with the two-way pulse lag and the transmit
bulk delay already removed. A custom beamformer therefore samples at

```python
n = (t_tx + t_rx - t0) * fs        # no pulse-lag term, ever
```

where `t_tx` is the transmit travel time from the wavefront's time origin to the
voxel and `t_rx = |r_voxel − r_e| / c`. This is what USTB calls `initial_time` and
what MUST's `dasmtx` assumes, so eSDIva RF drops into either without a correction.
Add a term only for a physical system delay you are deliberately modelling.

**The geometry.** `rx.element_centers` is `(E, 3)` in **metres** — the receive
positions, in the same frame as the field, after any `transform()`. Convert to mm
only at the display boundary. `rx.n_elements`, `rx.fc`, `rx.delays`,
`rx.apodization` complete the aperture description.

**The transmit wavefront.** It is defined entirely by the event's `delays` array,
not by a convention. Element `e` fires at `τ_e = delays[e] − max(delays)` (the bulk
delay is what `t0` already removed), so any transmit model can be fitted from it:

```python
tau = event["delays"] - event["delays"].max()          # firing instants, s
# spherical source behind the array (diverging wave):
t_ref = np.mean(tau - np.linalg.norm(centers - r_vs, axis=1) / c)
t_tx  = t_ref + np.linalg.norm(r_voxel - r_vs) / c
# plane wave with unit direction n:
t_ref = np.mean(tau - centers @ n / c)
t_tx  = t_ref + r_voxel @ n / c
```

For a focused transmit (`z_vs > 0`) the signs flip on the way in: `t_ref =
mean(τ_e + |r_e − r_vs|/c)` and `t_tx = t_ref − |r − r_vs|/c` above the focus. The
means are exact when the delays were built from that source and a least-squares fit
otherwise — which is precisely how `das_volume` stays convention-agnostic.

**A complete minimal DAS**, as a template for anything more ambitious:

```python
t0, dt = coords["t0"], coords["dt"]
centers = rx.element_centers                        # (E, 3) m
img = np.zeros(len(voxels_m))
for e, ce in enumerate(centers):
    t_rx = np.linalg.norm(voxels_m - ce, axis=1) / c
    idx = (t_tx + t_rx - t0) / dt                   # fractional sample index
    img += np.interp(idx, np.arange(rf.shape[-1]), rf[e])   # linear interpolation
```

Two things to get right in any custom reconstruction: interpolate (nearest-sample
indexing costs resolution at these frequencies — linear is the floor, cubic or a
Hilbert/IQ interpolation is better), and apply the receive aperture *growth* (F-number
mask) yourself if you want depth-independent resolution, since the raw RF contains
every channel at every depth.

**Checks that catch a wrong custom beamformer.** Simulate one point scatterer at a
known position with `per_scatterer=True`, and verify (a) the peak of channel `e`
lands within a sample of `(t_tx + t_rx − t0)·fs`, and (b) the reconstructed point
appears at the position you put it, not offset by half a pulse length — a constant
axial offset of `≈ pulse_length/2` means a lag term was re-applied that `t0` had
already removed.

**Exporting to another tool.** `RFDataset.to_hdf5(path)` (or
`save_rf_hdf5(path, rf, coords, ...)`) writes one self-describing HDF5 file with the
channel data and timing under UFF-compatible names (`sampling_frequency`,
`initial_time`, `sound_speed`) for MATLAB, USTB, or your own pipeline.

## Built-in beamformers

```python
from esdiva.beamforming import das_volume, das_rca_volume, DAS_focused_scanline, envelope_db
vol, axes = das_volume(rf[e:e+1],
                       {"dt": coords["dt"], "t0_per_event": t0[e:e+1]},
                       [event], probe, grid_mm,
                       c=1540.0, fnum=0.5, rx_apodization="rect")
```

- `das_volume` — general 3-D DAS for TX aperture = RX aperture. Each event carries
  **one** geometric key: `virtual_source_mm` (`z < 0` diverging, `z > 0` focused,
  `z ≈ 0` synthetic aperture) or `angles_deg` (plane wave, `α` or `(θx, θy)`). The
  transmit time origin is recovered from the event's own delays, so no min- or
  max-referenced delay convention is assumed.
- `das_rca_volume` — row–column probes, whose bar elements need their own geometry.
- `DAS_focused_scanline` — one line from per-channel RF.
- `envelope_db` / `esdiva.utilities.to_dB` — Hilbert envelope and log compression.

`t_offset_s` defaults to `0.0` and **stays there** for eSDIva RF: `coords["t0"]` is
already the beamforming reference. Use it only for foreign data (raw Field II
`calc_scat`, which still carries the two-way pulse lag) or a real system delay.

Compound coherently: Hilbert-transform each event's volume along the axial axis and
sum the complex IQ, then take the magnitude. Summing envelopes discards the phase
and blurs the result.

Receive apodization on a simulated array should usually be `"rect"` — element
directivity already tapers the aperture and a second window over-tapers it.
`coherence_weight=True` multiplies each voxel by its aperture coherence factor,
suppressing incoherent clutter; it is a display enhancement, so report plain DAS
numbers and treat CF as a ceiling.

## Imaging-study checklist

1. Set **both** `tx.impulse_response` and `rx.impulse_response` (2-cycle burst at
   `fc`) and drive with the bare excitation. Skipping them widens the PSF ~60 % and
   raises sidelobe clutter. The RF checkpoint fingerprint does **not** cover impulse
   responses — delete the store after changing the pulse model.
2. Derive PW/DW virtual sources per probe from the coverage rule: every corner of
   the reconstruction volume inside every event's cone. Never copy a virtual-source
   layout between probes.
3. Phantom: 5–10 scatterers per resolution cell, anechoic targets ≥ 3 PSF radii
   across, wires dim (≈ +10 dB) and far from contrast targets.
4. Preview with `sim.show(...)`, then run **one** event and inspect the speckle
   before launching the full sequence.
5. **Estimate the run time on the machine that will run it**, before committing:

   ```python
   from esdiva.utilities import estimate_sequence_runtime
   est = estimate_sequence_runtime(sim, positions_mm, n_emissions=n_angles * n_frames)
   ```

   It times short probes on subsets of the real phantom with the real probe and
   excitation, then projects. Cost is dominated by **scatterers × emissions**
   (patch count matters far less), and the constant is a property of the machine
   — cores, memory bandwidth, BLAS threading — so it varies by more than an order
   of magnitude between a laptop and a compute node. Never quote a timing
   measured elsewhere as the expected cost of someone's run; say which machine it
   came from. Probe with the largest fraction you can afford: cost per scatterer
   grows with cloud size before levelling off, so small subsets under-predict.
   Anything projected beyond a few hours wants `out_path=`.
6. Beamform with `t_offset_s=0.0` and each event's own `t0`.
7. Metrics: TGC estimated from speckle only, ROIs and margins scaled in PSF units
   (λz/D, not mm), plain DAS numbers, ~30 dB display window.
