# Example 21 — Volumetric ultrafast imaging with matrix arrays

A full pulse-echo **volume imaging case study**: a matrix probe transmits a
few tens of diverging waves, every element receives, and a 3-D delay-and-sum
with coherent compounding turns the stored RF into a volumetric B-mode of a
tissue-mimicking phantom. It runs at channel counts (up to 3025) where
Field II needs ~20× longer per event (~5 days for the flagship acquisition
vs an afternoon here), and every stage of the pipeline was verified against
independent references along the way. The lessons that verification bought —
physics traps, design rules, and a symptom→cause table — live in
[`TROUBLESHOOTING.md`](TROUBLESHOOTING.md): read it before designing your
own simulation.

## The three steps

| Script | What it does |
|---|---|
| `step1_define_phantom_TX_RX.py` | **All definitions, one place**: pick the scenario (probe + drive frequency), the phantom, the diverging-wave sequence, the drive burst + probe impulse response, the beamforming grid. The other scripts import from here. |
| `step2_acquire_RF.py` | Run the acquisition with `ReceptionSDI(method="spectral")`, one TX event at a time, checkpointed to `out/<scenario>/RF/` (`RFDataset`: crash-safe, resumable, refuses a silently-changed config). |
| `step3_beamforming.py` | Load the RF, beamform **each event** with the general `das_volume` beamformer, form IQ (Hilbert along z), **compound the per-event IQ coherently**, save the IQ volume to `out/<scenario>/IQ/`, apply depth-only TGC, measure contrast/SNR/PSF, save B-mode figures + `out/<scenario>/metrics.json`. |

Everything is grouped per scenario, and the beamformed volume is a stored
product — the visualization scripts read it, they never re-beamform:

```
out/<scenario>/RF/                # checkpointed acquisition (RFDataset)
out/<scenario>/IQ/iq_volume.npz   # compounded complex IQ + voxel axes (step 3)
out/<scenario>/metrics.json       # timings + contrast/SNR/PSF metrics
figures/<scenario>/               # every plot of that scenario
```

Pick the scenario by editing `SCENARIO` in step 1 or via the environment:

```bash
SCENARIO=vermon uv run examples/example21_rca_volume/step2_acquire_RF.py
SCENARIO=vermon uv run examples/example21_rca_volume/step3_beamforming.py
```

Auxiliary scripts (all follow the same scenario switch):

- `preview_phantom.py` — truth slices + 3-D scene of the phantom **before**
  spending simulation hours.
- `visualize_beamformed_volume.py` — 3-D renders of the beamformed volume
  (setup scene, sigmoid volume render, MPR cut-planes, truth-vs-image slices).
- `psf_grid.py` — 12 isolated points through the full sequence, beamformed
  per ring subset: the pure PSF vs position and vs compounding count.

Companion notes: [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md) — the design
rules and failure modes behind every choice in these scripts.

## The scenarios

| | `zeus5` (flagship) | `zeus10` (cautionary) | `vermon` (real probe) |
|---|---|---|---|
| Probe | ZeUS 55×55, 3025 ch | same probe | Vermon-type 32×32, 1024 ch |
| Pitch / aperture | 0.30 mm / 16.5 mm | 0.30 mm / 16.5 mm | 0.30 mm / 9.6 mm |
| Drive | 5 MHz → pitch **0.97λ** | 10 MHz → pitch **2λ** | 3 MHz → pitch **0.58λ** |
| Sequence | 21 DW, z=−20 mm, tilt ≤16.7° | 9 DW, z=−40 mm | 25 DW, z=−10 mm |
| Volume | 11×7×10 mm, z 10–20 | **same** | **same** |
| Scatterers | ~319k (10/cell) | ~1.3M (5/cell) | ~23k (10/cell) |
| Cost (spectral kernel) | ~4–6 h | ~8 h | **minutes** |
| Expected image | diffraction-limited (0.3–0.4 mm PSF), cyst ≈ −18 dB | **aliased**: grating clutter fills the cyst | textbook at its ~0.8 mm PSF |

The `vermon` scenario is also the **end-to-end pipeline test**: same phantom,
25 events in a few minutes — run steps 2–3 on it before committing hours to
a ZeUS acquisition. Because the acquisition is checkpointed per event, an
interrupted run (or one killed mid-way) simply resumes where it stopped.

Why the drive frequency decides everything (the 10 MHz → 5 MHz story): the
elements are fixed at 0.30 mm pitch. At 10 MHz (λ=0.154 mm) that is **2λ** —
the echo field is spatially undersampled (sampling a wave needs λ/2), so the
beamformer cannot tell the focused direction from aliased ones and every
voxel collects faint coherent copies of speckle from elsewhere. Bright
lesions survive; an anechoic cyst — an *absence* of signal — gets painted
over by clutter that no software (coherence weighting included: the aliased
copies are coherent) can reject. At 5 MHz the same pitch is 0.97λ: grating
lobes leave the field, the ~1λ element face hears wide-angle echoes, and one
single unfocused shot already out-images the entire 10 MHz compound. Real
probes have the bandwidth to be driven at half their centre frequency.

All three scenarios image the **same phantom** (identical volume, targets
and seed — the "contrast ladder"): an anechoic cyst tube (r=2 mm) at the
volume mid-depth (where the compound transmit focus peaks), a ×4 hyperechoic
tube below it on the same vertical and a second ×4 tube column beside it, a
×4 sphere above the cyst **offset in elevation** (absent at y=0, present in
the +y slice — the proof the volume is genuinely 3-D), and dim PSF wires
(amplitude 4 ≈ +10 dB over speckle) on the clear lane between the columns:
three lateral (along y) at fixed depths plus one axial (along z at y=0)
crossing them, so the lateral PSF is read continuously with depth. Only what
the physics dictates changes per scenario: the **scatterer count** — speckle
needs ≥ ~5–10 random scatterers per resolution cell `(λz/D)² · pulse/2` to
be fully developed (Rayleigh envelope, SNR = mean/std = 1.91), and that cell
shrinks with frequency and aperture, which is why the 10 MHz run needs 1.3 M
scatterers and the Vermon one 23k for the same statistics — and the
**virtual sources**, re-derived per probe from the coverage and
steep-wavefront rules (`TROUBLESHOOTING.md` §2): two probes are comparable
when each runs at its own coverage limit, not when they share coordinates.

One thing is **not** optional: every probe builder sets
`probe.impulse_response` (a 2-cycle burst at `fc`). A physical probe
band-passes the signal twice through its piezo impulse responses — the
chain is `drive ⊛ h_tx ⊛ h_rx`, which the Reception class convolves in the
frequency domain. Simulating without them models ideal broadband elements,
and the aperture impulse-response tails then dominate the received spectrum:
the PSF widens ~60 % and the sidelobe skirt rises ~12 dB. Full story and the
checkpoint caveat: `TROUBLESHOOTING.md` §1.

## The beamformer: one DAS for every transmission basis

`pyfield.beamforming.das_volume` beamforms **any** transmit scheme with one
call, assuming TX aperture = RX aperture. Each event dict carries the same
`delays`/`apodization` given to `sequence_rf` plus one geometric key:

- `virtual_source_mm=[x, y, z]` — spherical wavefront: `z<0` diverging wave,
  `z>0` focused transmit (converges to the focus, then diverges — the
  virtual-source model), `z≈0` single-element synthetic aperture;
- `angles_deg=(θx, θy)` — steered plane wave.

The transmit time origin is recovered **from the event's own delays**
(element `e` fires at `delays_e − max(delays)` in the data's time frame,
because the simulator's `t0` is beam-axis referenced), so no min/max delay
reference convention has to be assumed — the convention trap that once
misplaced points by `(d_max−d_min)/2c ≈ 1 mm` cannot occur. Receive is a
dynamic radial aperture `|r_xy − r_e,xy| ≤ z/(2·F#)`, rect or Hann, with
optional coherence-factor weighting. Verified in
`tests/unit/test_das_volume.py`: a synthetic point echo reconstructs at its
exact position for all four bases.

**Which probes fit `das_volume`?** Any aperture whose elements act as
point-like receivers: it only uses `element_centers`, so the layout may be
flat, curved, sparse, or a ring — the direct-path `t_rx = |r − r_e|/c` and
the delay-recovered `t_tx` stay exact (only the `fnum` gate, defined per
depth `z`, becomes approximate for strongly curved apertures). **Not RCA**:
its long row/column elements receive at the *nearest point of the bar*
(a stationary-phase arrival), not at the element centre — use
`das_rca_volume`, which models exactly that segment geometry.

Step 3 applies it per event, converts each RF volume to IQ (analytic signal
along z) and sums the IQ complex — coherent compounding — then applies one
depth-only TGC gain curve. The default settings that matter (full-aperture
rect receive, the pulse-centre `t_offset_s`, the wire-free TGC median) each
guard against a measured failure mode: `TROUBLESHOOTING.md` §4–5.

## Reference results (plain DAS + TGC, 30 dB display)

ZeUS 5 MHz, 21 DW (3.75 h acquisition, seconds to beamform):

| Metric | value | truth |
|---|---|---|
| Anechoic cyst r=2 mm | **−17.5 dB** | −∞ |
| Lesion ×4 r=1 (side / under the cyst) | **+12.0 / +12.0 dB** | +12 |
| Sphere ×4 r=0.8 (elevation) | **+12.2 dB** | +12 |
| Speckle SNR | 1.86 | 1.91 (Rayleigh) |
| Wire lateral FWHM z=11/15/19 | 0.30 / 0.40 / 0.40 mm | diffraction ≈ 0.3–0.4 |
| Wire axial FWHM | 0.15–0.25 mm | pulse-limited |

Every number sits at its physical ground truth — the acquisition, the
kernel, the beamformer and the phantom statistics all close simultaneously.
The Vermon scenario images the identical phantom at its own ~0.8 mm PSF in
minutes (an earlier ×2-scaled campaign of the same design read cyst
−24.5 dB, lesion +10.6, sphere +11.3, Rayleigh speckle). The zeus10 scenario
is the negative control: run it to *see* spatial aliasing eat an anechoic
target.

Every design rule behind these numbers, the failure modes that were hit
getting there, and a symptom→cause table live in
[`TROUBLESHOOTING.md`](TROUBLESHOOTING.md).

## Artifacts

- `out/<scenario>/RF/` — the checkpointed acquisition (compressed chunk
  files + fingerprinted manifest; resumable).
- `out/<scenario>/IQ/iq_volume.npz` — the compounded complex IQ volume +
  voxel axes (what the visualization scripts read).
- `out/<scenario>/metrics.json` — timings + all metrics.
- `figures/<scenario>/` — B-modes, truth-vs-image panels, 3-D renders,
  PSF ladders.
