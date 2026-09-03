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
- **Motion is not automatic** *(first-class support not yet)*. One `sequence_rf` call freezes the scatterers: every
  event sees the identical medium, so a sequence gives no Doppler or decorrelation by
  itself. Flow *is* simulatable the way it is in Field II — loop the emissions
  yourself, advancing the scatterer positions by `v·(1/PRF)` between calls to
  `pulse_echo_rf`, and stack the results. You lose `sequence_rf`'s checkpointing, and
  the medium is still linear and non-viscous, so this buys Doppler and speckle
  decorrelation, not elastography or shear waves.
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
- `"paired"` — the pedagogic two-way delta train. Exact, no FFT, but cost ∝ M², so
  far slower; it warns when selected. For teaching or auditing the kernel only.

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

```python
from esdiva.io import RFDataset
ds = RFDataset("rf_store"); ds.summary()
rf, coords = ds.load_all()
ds.to_hdf5("channels.h5")      # UFF-compatible fields for MATLAB / USTB
```

`coords["dt"]` is shared; `coords["t0_per_event"]` gives each event its own
beam-axis time origin (the grid depends on that event's delays). Beamform each event
with **its own** `t0`.

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
5. Beamform with `t_offset_s=0.0` and each event's own `t0`.
6. Metrics: TGC estimated from speckle only, ROIs and margins scaled in PSF units
   (λz/D, not mm), plain DAS numbers, ~30 dB display window.
