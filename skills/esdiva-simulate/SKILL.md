---
name: esdiva-simulate
description: Help someone use the eSDIva ultrasound field simulator — build a transducer (linear/convex/matrix array, circular piston or bowl, custom or Field II import), run an emission pressure-field simulation (CW or transient), run a pulse-echo RF simulation (PSF, phantom, plane-/diverging-wave sequence), beamform it or feed the RF to their own beamformer, choose a working PyVista/Matplotlib backend, and explain the SIR/SDI physics behind a result. Use whenever the user mentions eSDIva, a spatial impulse response, an acoustic pressure field, a transducer aperture, a point spread function, RF channel data, delay-and-sum, or an ultrasound imaging simulation.
---

# Simulating with eSDIva

eSDIva computes acoustic fields with the Tupholme–Stepanishen **spatial impulse
response** (SIR) method: an aperture is discretised into rectangular patches, each
patch's SIR is evaluated exactly, and pressure follows by convolution with the
excitation. Its users are ultrasound researchers and students, not programmers:
explain the acoustics, keep the Python out of the way, get a first result on screen
quickly.

## Helping someone who is new to the package

Turn the physical question into the smallest simulation that answers it:

1. **Ask what they want to see**, not what they want to call. Beam shape, wavefront,
   PSF, B-mode image, channel data for their own reconstruction — each maps to one
   template.
2. **Start from a ready-made probe** (`transducers.Domino()`, 128-element linear;
   `transducers.Zeus_Matrix()`, 55×55 matrix) unless they have specific element
   dimensions. Fewer numbers to get wrong on day one.
3. **Copy a template, then edit it** — `templates/` holds four runnable starting
   points. Do not compose a script from scratch when one of them is 80 % of the way.
4. **Run something small first.** A coarse 2-D plane returns in seconds and catches
   unit errors, wrong focus signs and empty grids before a long run.
5. **Explain the output in physical terms.** "`p` is `(Nt, Nx, Ny, Nz)`: the pressure
   in pascals at each point for each time sample" — then show the plot. Say what a
   feature *is* (edge wave, near-field structure, sidelobe), not just that it exists.
6. **Say what a number costs.** If a request implies hours of compute, say so up
   front and offer the cheaper version first.

Ask where the result should go — a `.py` script or a Jupyter notebook — because it
changes the plotting arguments (see `references/visualization.md`). Templates are
`# %%`-celled `.py` files: they run as scripts *and* map one-to-one onto notebook
cells, so emit whichever the user wants from the same content. Never keep two
divergent copies of one template.

## Route the request

Open the file named below — paths are relative to this skill's own folder, wherever
it is installed. Read one reference, not all of them.

| User wants | Read |
|---|---|
| A probe / aperture / geometry, delays, apodization, moving a probe in space | `references/transducers.md` |
| A pressure field: beam plot, CW amplitude, propagating wavefront, intensity, attenuation | `references/emission.md` |
| RF channel data, PSF, phantom image, PW/DW sequence, FMC, DAS, B-mode | `references/reception.md` |
| What the RF output means, feeding it to a *custom* or third-party beamformer, exporting to USTB/MATLAB | `references/reception.md` § "What the RF output actually is" |
| A figure that actually appears — notebook vs desktop vs headless, PyVista backends, saving movies | `references/visualization.md` |
| "Why does the field look like this", method choice, sampling, `t0`, Field II equivalence | `references/physics.md` |

Templates: `emission_cw.py` (beam profile), `emission_transient.py` (wavefront),
`reception_psf.py` (point spread function + timing check),
`reception_sequence_das.py` (DW sequence → RF → image, with a hand-written
beamformer beside the built-in one).

## Rules that decide whether the result is right

1. **Units.** Everything public is millimetres and carries an `_mm` suffix; every
   internal is SI (m, s). `element_centers` is metres — it is the one internal users
   touch directly.
2. **`no_sub_x` / `no_sub_y` are required and keyword-only** on array probes. They
   set patch size, which sets accuracy: a patch must be small compared with the
   wavelength or the SIR is quantised. Start at 2–4 per element side and check the
   result stops changing when you double it.
3. **Return shapes.** `Emission(tx, monochromatic=True)` → `(Nx, Ny, Nz)` CW
   amplitude at `fc`. Any transient emission → `(Nt, Nx, Ny, Nz)` with
   `coords["t0"]` and `coords["dt"]`. `Reception.pulse_echo_rf` → `(Erx, Nt)`.
4. **`coords["t0"]` is the beamforming reference, not the first sample's instant.**
   The two-way pulse lag and the transmit bulk delay are already subtracted, so an
   echo peaks at its *geometric* round-trip time. Any beamformer, built-in or
   home-made, reads the sample at `(t_tx + t_rx − t0)·fs` with **no** lag term.
   Re-applying it displaces the image axially by half a pulse length. (Same
   convention as USTB `initial_time` and MUST `dasmtx`; raw Field II `calc_scat`
   output does still carry the lag — that is what `t_offset_s` is for.)
5. **Imaging needs impulse responses.** For any PSF / phantom / B-mode study set
   both `tx.impulse_response` and `rx.impulse_response` (typically a 2-cycle burst
   at `fc`) and drive with a bare excitation. Skipping them widens the PSF by
   roughly 60 % and raises sidelobe clutter — aperture diffraction tails then
   dominate the spectrum.
6. **Receive stays unfocused.** Reception returns per-element RF without summing, so
   receive delays or apodization are baked into every channel. Copy the probe
   (`rx = tx.copy()`) *before* applying the transmit focal law; focusing belongs in
   the beamformer.
7. **Sampling.** `fs` defaults to 100 MHz and must oversample the pulse heavily — the
   SIR is a train of sharp edges, not a band-limited signal. 100–200 MHz for a
   few-MHz probe. Decimate afterwards with `downsampling=`, never by lowering `fs`.
8. **Method selection.** Leave `method="auto"` for emission (`"FST"` = classic fully
   sampled trapezoid, `"sdi"` = sparse delta integration; same answer, faster on
   large grids) and `method="spectral"` for `Reception`, unless benchmarking.
9. **Cost is the grid.** Runtime scales with field points × patches. Prototype on a
   coarse 2-D plane (`y_extent: [0, 0]`, `dy: 0`), then refine. Warn before
   launching anything that will take hours, and use `out_path=` checkpointing for it.

## Working style

- Comment the **physics**, not the array mechanics. Say why a `dt` factor or a delay
  sign is there; do not restate what a line literally does.
- Do not assert a physical cause you have not tested. "The sidelobes come from the
  pitch" is a hypothesis until a control run excludes the alternatives — label it as
  one, or run the discriminating simulation (change that one parameter and check the
  feature moves as predicted).
- Sanity-check before the long run: `sim.show(...)` previews a pulse-echo setup in
  3-D, and one event of a sequence tells you whether the speckle is plausible.
- If something looks wrong, check the cheap causes first — units (mm vs m), the sign
  of the focus `z`, a grid that starts on the aperture face, missing impulse
  responses — before suspecting the physics.
