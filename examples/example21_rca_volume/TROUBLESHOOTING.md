# Troubleshooting & design notes

Practical lessons from building this case study. Every item below cost at
least one wasted acquisition or a day of head-scratching. Check them before
designing a new simulation; come back when an image looks wrong.

## 1. Set the transducer impulse response — always

A physical probe band-passes the signal **twice** through its piezo impulse
response: the pulse-echo chain is `e ⊛ h_e ⊛ h_r` (drive ⊛ TX impulse
response ⊛ RX impulse response). If you skip `probe.impulse_response`, you
are simulating ideally broadband elements — and the low-frequency tails of
the *aperture* (diffraction) impulse responses take over the received
spectrum. It does not look like a subtle error; it looks like a bad probe:

- received spectral centroid drops (measured: 3.0 → 1.86 MHz for a 3 MHz
  drive), so the effective wavelength grows;
- the lateral PSF widens ~60 % beyond the diffraction limit;
- the near-in sidelobe skirt rises from −22 to −10 dB and paints iso-range
  arcs; over a whole speckle volume this skirt was our dominant clutter —
  adding the impulse responses alone took an anechoic cyst from −2 dB
  (barely visible) to −24 dB.

The symptom to watch for: point targets 50–100 % wider than λz/D **while
every arrival time is geometrically exact**. If timing is right and the PSF
is still wide, look at the received spectrum, not at the beamformer.

One warning: the RF checkpoint fingerprint covers the excitation and the
probe geometry but *not* the impulse responses. If you change the pulse
model, delete `out/<scenario>/RF` yourself — resume cannot detect it.

## 2. Placing virtual sources

A diverging wave from a virtual source at `(x_vs, z_vs)` (behind the array,
`z_vs < 0`) only insonifies the **cone from the source through the aperture
edges**. Three rules follow:

1. **Coverage.** Check that every corner of the imaging volume lies inside
   every event's cone: with half-aperture `a`, the wave from `(r, −z_s)`
   reaches, at depth `z`, laterally out to `a + (a − r)·z/z_s` on the far
   side. Violating this doesn't fail loudly — the deep/lateral parts of the
   volume are simply missing from some events, and the image shows depth-
   dependent brightness banding and contrast that refuses to make sense.
2. **Steep wavefronts feed grating lobes.** The closer the source, the more
   obliquely its wavefront crosses the outer elements. Harmless at fine
   pitch (≤ 1λ); at coarse pitch (the same probe driven at a higher
   frequency) every extra degree of obliquity pushes more spatially aliased
   energy into the image. Coarse pitch ⇒ keep the sources far and the tilts
   small.
3. **Tilt saturates.** Compounding narrows the synthesized transmit focus
   only until the transmit tilt span matches the aperture's own half-angle;
   beyond that, extra sources only grind down the sidelobe pedestal (~1/N).
   A small aperture over a wide, deep volume therefore cannot buy much
   transmit focusing — rule 1 caps the tilt long before rule 3 does.

Consequence: **derive the sources from the rules for each probe and volume;
never copy a layout between probes.** Two probes are comparable when both
run at their own coverage limit, not when they share coordinates.

## 3. Designing the phantom

- **Scatterer density.** Fully developed speckle (Rayleigh envelope,
  mean/std = 1.91) needs ≥ ~5–10 random scatterers per resolution cell
  `(λz/D)² · (pulse length)/2`. Fewer, and the texture — and every contrast
  number — is an artefact of the particular random draw. The cell scales
  ~λ³, which is why a 10 MHz volume needs 50× more scatterers than the same
  volume at 3 MHz.
- **Anechoic targets must out-size the PSF** (radius ≥ ~3 PSF), or they fill
  in from their own blurred edges no matter how clean the system is.
- **Wires are treacherous.** A dense scatterer line integrates everything
  inside a resolution cell coherently, so its brightness scales with the
  PSF volume — a wire that is fine at one frequency paints full-frame
  sidelobe arcs at another. Keep wires dim (~+10 dB over speckle, amplitude
  4 here), few, and well clear of contrast targets; brighter wires wash
  entire depth planes.
- **Preview before you burn hours.** `preview_phantom.py` renders the truth
  and the acquisition geometry in seconds; a single-event acquisition
  validates speckle statistics in minutes. Both have caught phantom-design
  mistakes that would have wasted multi-hour runs.

## 4. Measuring image quality honestly

- **TGC on tissue, not on the cyst.** The depth-gain curve must come from
  speckle-only regions. A median over a plane containing a large anechoic
  target dips inside it and injects a spurious gain bump at that depth —
  the digital version of setting the TGC sliders on the lesion.
- **Scale every ROI and margin with the PSF.** Exclusion margins tuned on a
  0.3 mm-PSF probe sit *inside* a wire's mainlobe on a 0.8 mm-PSF probe;
  the background then contains target energy and the metrics silently
  degrade (a +12 dB lesion once read +0.4 dB from this alone). Use λz/D at
  the target depth as the unit, not millimetres.
- **Background depth matters — until TGC.** The compound transmit focus is
  a depth-dependent brightness hump, so *before* TGC, contrast must be
  measured against same-depth speckle (a whole-volume background once made
  a filled cyst read +6 dB "brighter" than its surroundings). *After* a
  depth-only TGC the mean is flat in depth by construction, and any clean
  speckle is a fair reference.
- **Coherence-factor weighting is a contrast knob, not the truth.** It
  recovers the contrast *number* (a +12 dB lesion reading +4 dB under plain
  DAS comes back in full) but destroys the speckle texture (SNR falls far
  below Rayleigh) — and it cannot reject *coherent* clutter such as grating
  lobes. Report plain DAS; quote CF as a ceiling.
- **Display range:** after TGC, speckle fills ~30 dB of greyscale; a 40+ dB
  window makes normal sidelobe structure look like artefacts.

## 5. Beamforming pitfalls

- **Double apodization.** Elements about one wavelength wide already taper
  the aperture through their own directivity (an edge element viewing a
  mid-depth voxel at ~30° weighs ×0.6). Adding a Hann receive taper on top
  halves the effective aperture: measured 0.98 mm point FWHM (f#1 Hann) vs
  0.52 mm (full-aperture rect) on the same RF. Default here: `fnum=0.5`,
  `rx_apodization="rect"`.
- **Pulse-centre lag.** A band-limited pulse peaks ~half its length after
  the geometric arrival; pass that lag as `t_offset_s` or the whole image
  is biased deep (here: `3(L−1)/2/fs` for the drive ⊛ TX-IR ⊛ RX-IR chain).
- **Delay-reference conventions bite.** Whether delays are referenced to
  the earliest or latest element changes the transmit time origin by
  `(d_max − d_min)/c` — enough to misplace a point by a millimetre.
  `das_volume` sidesteps this by recovering the time origin from each
  event's own delay vector, so any convention beamforms correctly.
- **Which probes fit `das_volume`?** Any aperture of point-like elements —
  flat, curved, ring, sparse (it only uses `element_centers`). Not RCA: its
  long bars receive at the nearest point of the bar, not at the element
  centre — use `das_rca_volume` for that geometry.

## 6. Symptom → cause

| Symptom | Likely cause | Check / fix |
|---|---|---|
| Point wider than λz/D, timing exact | missing impulse responses (§1) | received spectrum centroid |
| Sidelobe arcs across the frame | impulse responses (§1) or a too-bright wire (§3) | spectrum; wire amplitude |
| Brightness bands vs depth, contrast nonsensical | coverage rule violated (§2) | per-event cone vs volume corners |
| Anechoic target filled, lesions fine | clutter floor: grating lobes (coarse pitch) or sidelobe skirt | pitch/λ; CF as diagnostic (rejects incoherent skirt, passes grating) |
| Anechoic target filled, ~PSF-sized | resolution, not clutter (§3) | enlarge target or aperture |
| Compound ≈ single event | usually masked by speckle, not broken | isolated-point run; per-event phase at a wire voxel |
| Point misplaced ~1 mm | delay-reference convention (§5) | one 0° plane wave — no convention can hide |
| Metrics degrade on a new probe, image looks fine | fixed-mm ROIs on a different PSF (§4) | PSF-scaled margins |
| Whole image shifted in depth | missing `t_offset_s` (§5) | pulse-centre lag |
| Kernel edits "have no effect" | stale numba cache | delete `__pycache__/*.nb*` |
