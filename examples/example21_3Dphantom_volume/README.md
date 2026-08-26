# Example 21 — 3-D volume imaging with a matrix array

This folder is a complete **pulse-echo volume imaging experiment**, from a
tissue-mimicking phantom to a beamformed 3-D B-mode. A matrix probe transmits a
few tens of **diverging waves**, every element records the echoes, and a 3-D
delay-and-sum with **coherent compounding** reconstructs the volume. It runs at
channel counts (up to 3025) where a classical spatial-impulse-response simulator
would need days per event; here a flagship acquisition takes an afternoon.

The point of the example is not one pretty image — it is to show the *whole
chain* (define → acquire → beamform → measure) and the handful of physics
choices that decide whether the final volume is trustworthy or garbage.

## The pipeline

Three scripts, run in order. Each imports its definitions from step 1, so there
is a single source of truth for the geometry, phantom, and sequence.

| Script | Stage | What it produces |
|---|---|---|
| `step1_define_phantom_TX_RX.py` | **Define** | The scenario (probe + drive frequency), the scatterer phantom, the diverging-wave sequence, the drive pulse + probe impulse response, and the beamforming grid. Nothing is simulated here — it is the experiment's parameter file. |
| `step2_acquire_RF.py` | **Acquire** | Runs `Reception(method="spectral")` one transmit event at a time and writes the raw per-channel RF to `out/<scenario>/RF/`. The store is checkpointed: a killed run resumes where it stopped, and it refuses to continue if you silently change the config. |
| `step3_beamforming.py` | **Beamform + measure** | Loads the RF, beamforms **each event** with `das_volume`, forms IQ (analytic signal along depth), **sums the per-event IQ coherently**, applies a depth-only time-gain curve, then measures contrast / speckle SNR / PSF width and saves the B-mode figures and `metrics.json`. |

Helper scripts (same scenario switch):

- `preview_phantom.py` — render the phantom truth and the acquisition geometry
  **before** spending simulation hours.
- `visualize_beamformed_volume.py` — 3-D renders of the reconstructed volume
  (volume render, cut-planes, truth-vs-image slices).
- `psf_grid.py` — 12 isolated point targets through the full sequence,
  beamformed per sub-aperture: the pure PSF vs position and vs compounding.

The beamformed volume is a **stored product** — visualization scripts read it,
they never re-beamform:

```
out/<scenario>/RF/                # checkpointed raw channel data
out/<scenario>/IQ/iq_volume.npz   # compounded complex IQ + voxel axes
out/<scenario>/metrics.json       # timings + contrast/SNR/PSF numbers
```

## Running it

```bash
uv run examples/example21_3Dphantom_volume/step2_acquire_RF.py
uv run examples/example21_3Dphantom_volume/step3_beamforming.py
```

Pick the probe with the `SCENARIO` environment variable (or edit it in step 1):

```bash
SCENARIO=vermon uv run examples/example21_3Dphantom_volume/step2_acquire_RF.py
```

Start with `vermon`: it images the same phantom in a few minutes and is the
end-to-end pipeline test. Only commit hours to a ZeUS run once that passes.

## The phantom (one shared truth)

All scenarios image the **same** scatterer cloud (same volume, targets, and
random seed) so probes are compared on identical ground truth — a "contrast
ladder":

- an **anechoic cyst** tube (r = 2 mm) at mid-depth, where the compound
  transmit focus is strongest;
- two **hyperechoic** (×4) tube columns, one under the cyst and one beside it;
- a ×4 **sphere** offset in elevation — absent in the central slice, present in
  the +y slice, proving the reconstruction is genuinely 3-D;
- dim **PSF wires** (~+10 dB over speckle) on a clear lane: three lateral at
  fixed depths plus one axial, so the point-spread width is read continuously
  with depth.

Build it with `sondi.utilities.make_phantom`, which places random scatterers
and draws their amplitudes from `N(0,1)` times an echogenicity map. Two things
change per scenario, both dictated by physics, not taste:

- **Scatterer count** — fully developed speckle needs ≥ 5–10 random scatterers
  per resolution cell, and the cell shrinks with frequency (~λ³), so the 10 MHz
  run needs ~1.3 M scatterers where the 3 MHz one needs ~23 k.
- **Virtual-source layout** — re-derived per probe from the coverage rule
  below, never copied between probes.

## The scenarios

| | `zeus5` (flagship) | `zeus10` (high-freq) | `vermon` (real probe) |
|---|---|---|---|
| Probe | ZeUS 55×55, 3025 ch | same probe | Vermon-type 32×32, 1024 ch |
| Pitch / aperture | 0.30 mm / 16.5 mm | 0.30 mm / 16.5 mm | 0.30 mm / 9.6 mm |
| Drive | 5 MHz (pitch 0.97λ) | 10 MHz (pitch 2λ) | 3 MHz (pitch 0.58λ) |
| Sequence | 21 DW, z = −20 mm | 9 DW, z = −40 mm | 25 DW, z = −10 mm |
| Volume | 11×7×10 mm, z 10–20 | same | same |
| Scatterers | ~319 k | ~1.3 M | ~23 k |
| Cost | ~4–6 h | ~8 h | minutes |

The drive frequency sets pitch/λ (the pitch is fixed at 0.30 mm). Classical
sampling theory predicts grating-lobe clutter above 1λ, and we first blamed the
10 MHz images on it — wrongly: a rerun produced clean images at 2λ pitch. Treat
pitch/λ as a number to check per system, not a verdict. What frequency really
decides here is *cost*, through the scatterer count above.

## The beamformer

`sondi.beamforming.das_volume` beamforms **any** transmit scheme (plane wave,
diverging wave, focused, synthetic aperture) in one call, assuming the transmit
and receive apertures coincide. Each event carries its `delays`/`apodization`
plus one geometric key:

- `virtual_source_mm=[x, y, z]` — a spherical wavefront: `z < 0` diverging,
  `z > 0` focused, `z ≈ 0` single-element synthetic aperture;
- `angles_deg=(θx, θy)` — a steered plane wave.

The transmit time origin is recovered from **each event's own delay vector**, so
you do not have to tell the beamformer which element the delays were referenced
to — a common source of ~1 mm axial misplacement (see troubleshooting). Receive
uses a depth-dependent radial aperture set by the f-number.

`das_volume` works for any aperture of point-like elements (flat, curved,
sparse, ring). It does **not** fit row-column (RCA) probes, whose long bars
receive at the nearest point of the bar rather than the element centre — those
use `das_rca_volume`.

## Reference results

`zeus5`, 21 diverging waves, plain DAS + depth TGC, 30 dB display:

| Metric | measured | ground truth |
|---|---|---|
| Anechoic cyst (r = 2 mm) | −17.5 dB | −∞ |
| Lesion ×4 (side / under cyst) | +12.0 / +12.0 dB | +12 |
| Sphere ×4 (elevation) | +12.2 dB | +12 |
| Speckle SNR | 1.86 | 1.91 (Rayleigh) |
| Wire lateral FWHM (z = 11/15/19) | 0.30 / 0.40 / 0.40 mm | 0.3–0.4 (diffraction) |
| Wire axial FWHM | 0.15–0.25 mm | pulse-limited |

Every number lands at its physical limit at once — acquisition, beamformer, and
phantom statistics all close together, which is the real validation. `vermon`
reproduces the same targets at its coarser ~0.8 mm PSF in minutes.

## Troubleshooting — read before running your own

Each item below cost us at least one wasted acquisition. General lesson first:
**do not name a cause you have not tested.** We attributed early 10 MHz problems
to grating lobes and impulse-response handling; a controlled rerun falsified
both. A wrong-but-confident physics explanation is worse than none.

**Always set the impulse responses.** A real probe band-passes the echo twice,
`drive ⊛ h_tx ⊛ h_rx`. If you leave `probe.impulse_response` unset you are
modelling ideally broadband elements, and the low-frequency tails of the
*aperture* diffraction response take over the spectrum: the received centroid
drops (we measured 3.0 → 1.86 MHz), the lateral PSF widens ~60 %, and the
sidelobe skirt rises from about −22 to −10 dB and becomes the dominant clutter —
enough to fill an anechoic cyst. The tell-tale: point targets far wider than the
diffraction limit *while every arrival time is exact*. When timing is right but
the PSF is fat, look at the spectrum, not the beamformer. (The RF checkpoint
fingerprint does **not** cover the impulse responses — after changing the pulse,
delete `out/<scenario>/RF/` by hand.)

**Cover the whole volume with every transmit.** A diverging wave from a virtual
source only insonifies the cone through the aperture edges. With half-aperture
`a`, the wave from `(r, −z_s)` reaches laterally out to `a + (a−r)·z/z_s` at
depth `z`. If any volume corner falls outside any event's cone the failure is
silent — the image just bands in brightness with depth and the contrast numbers
stop making sense. Note too that compounding sharpens the synthesized focus only
until the tilt span matches the aperture's own half-angle; beyond that, extra
transmits only lower the sidelobe pedestal (~1/N). Derive the sources for each
probe and volume; comparable probes each run at their own coverage limit.

**Design the phantom for speckle.** Below ~5–10 scatterers per resolution cell
the texture — and every contrast number read from it — is an artefact of the
particular random draw. Make anechoic targets at least ~3 PSF radii across or
they fill in from their own blurred edges. Keep wires dim and few: a dense
scatterer line integrates coherently and can paint sidelobe arcs across the
whole frame. Preview the phantom and run a single event to check speckle
statistics before launching the full acquisition.

**Beamform honestly.**

- *Double apodization.* Elements about one wavelength wide already taper the
  aperture through their own directivity. Adding a Hann receive window on top
  halves the effective aperture (we measured 0.98 vs 0.52 mm FWHM on the same
  RF). Prefer a rectangular receive window and a low f-number.
- *Pulse-centre lag.* A band-limited pulse peaks about half its length after the
  geometric arrival, so a naive delay-and-sum places the whole image too deep.
  `das_volume` reads this lag from the RF metadata automatically; a custom
  beamformer must add it.
- *Delay reference.* Referencing transmit delays to the earliest vs the latest
  element shifts the time origin by `(d_max − d_min)/c` — about a millimetre.
  `das_volume` avoids this by reading each event's own delays; a single 0° plane
  wave is the test that exposes the mistake.

**Measure honestly.** Take the depth-gain curve from speckle-only regions, never
across an anechoic target. Scale every region of interest with the PSF (in units
of `λz/D`), not in fixed millimetres, or the same margins that are clean on a
sharp probe sit inside a wire's mainlobe on a blurrier one. Coherence-factor
weighting flatters the contrast number but destroys speckle texture and lets
coherent clutter through — report plain DAS and quote CF only as a ceiling.

### Symptom → likely cause

| Symptom | Likely cause | Check |
|---|---|---|
| PSF wide, timing exact | impulse responses not set | received spectrum centroid |
| Sidelobe arcs across the frame | missing impulse responses, or a too-bright wire | spectrum; wire amplitude |
| Brightness banding with depth | virtual-source coverage violated | per-event cone vs volume corners |
| Anechoic target filled (≫ PSF) | clutter floor | received spectrum |
| Anechoic target filled (~PSF) | resolution, not clutter | enlarge target or aperture |
| Point misplaced ~1 mm | delay-reference convention | one 0° plane wave |
| Whole image shifted deep | pulse-centre lag not applied | beamformer time offset |
| Metrics collapse on a new probe | fixed-mm ROIs on a different PSF | rescale ROIs by PSF |
| Kernel edits have no effect | stale numba cache | delete `__pycache__/*.nb*` |
