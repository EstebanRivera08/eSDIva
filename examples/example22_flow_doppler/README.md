# Example 22 — Ultrafast compound Doppler

A controlled replication of **ultrafast compound Doppler imaging**, run on a
phantom whose true velocity field we set ourselves — which the original
experiment, on real blood, could not have.

> **Bercoff, J., Montaldo, G., Loupas, T., Savery, D., Mézière, F., Fink, M. &
> Tanter, M. (2011).** *Ultrafast compound Doppler imaging: providing full blood
> flow characterization.* **IEEE Transactions on Ultrasonics, Ferroelectrics and
> Frequency Control 58**(1), 134–147.
> [doi:10.1109/TUFFC.2011.1780](https://doi.org/10.1109/TUFFC.2011.1780) ·
> [PubMed 21244981](https://pubmed.ncbi.nlm.nih.gov/21244981/)

Nothing here is invented. The method, the design rule and the claims are the
paper's; this folder re-runs them against a known ground truth.

## The claim being tested

Conventional colour Doppler scans the box one focused line at a time, and each
line needs its own ensemble ("packet") of emissions to measure a velocity, so it
costs `N_lines × E`. Ultrafast compounding insonifies the **whole box** with
every emission, so one ensemble of compound frames images everything at once,
costing `N_angles × E`. The saving is `N_lines / N_angles` — reported as up to
16×, and 10–15× in practice for a colour box of ~100 lines imaged with ~9
angles. The paper's second option is to keep the acquisition time and spend it
on a much longer ensemble instead, buying **sensitivity** rather than speed.

## The three sequences

One phantom, one probe, one pulse, one Doppler PRF, one estimator, one noise
level. They differ **only** in how the transmit spends its emissions.

| Sequence | Transmit | Emissions | Slow-time samples |
|---|---|---|---|
| **Conventional focused** | 72 focused lines, one packet each | 576 | 8 |
| **Compounded ultrafast** | 9 tilted plane waves per frame | 72 | 8 |
| **Compounded ultrafast, long ensemble** | 9 tilted plane waves per frame | 288 | 32 |

### "Ensemble" means slow-time samples, not frames of equal cost

The ensemble is the number of **slow-time samples the estimator sees**, and that
is the quantity that must match for the comparison to be fair. What one sample
*costs* is what differs:

- a conventional sample is **one focused emission** on that line, so 8 samples
  on each of 72 lines is 576 emissions;
- a compounded sample is **one compound frame of 9 emissions**, so 8 samples is
  9 × 8 = 72 emissions — and the whole box is imaged at once.

Both estimators work from 8 numbers per voxel. Only the bill differs, by 8.0×.

## The design rule, and how it constrains this sequence

```
N_angles = PRF_max / PRF_doppler        PRF_max = c / (2·z_max)
```

`PRF_max` is the depth limit — you cannot fire again until the last echo is
back. Here `z_max = 16.5 mm` gives `PRF_max = 46.7 kHz`, so nine angles cap the
Doppler PRF at **5 kHz**, and with it the unambiguous velocity
`v_nyquist = c·PRF_doppler/(4·f)` = 18.4 cm/s. Compounding is never free: it is
always paid for out of the velocity range. That is why the vessel peaks at
20 cm/s rather than something faster — the sequence was designed against the
limit, not the other way round.

## The probe

A 3 MHz linear array, with every dimension tied to the wavelength
(λ = 0.513 mm):

| | | why |
|---|---|---|
| 64 elements, 0.24 mm pitch | 15.4 mm aperture | half-wavelength pitch, so the grating-lobe condition never binds and the angles can be steered as wide as the aperture allows |
| 14 mm elevation aperture | — | elevation is the axis the image cannot show |
| elevation lens, radius 17.46 mm | **line focus at 16.00 mm** | putting the focus on the vessel is what keeps the slice thin |

A voxel collects blood from above and below the imaging plane too, where the
vessel is narrower and the flow slower, so an unfocused elevation aperture
flattens the measured profile. The lens puts that slice at **0.61 mm against a
4 mm lumen**.

**`elevation_focus_mm` is the lens RADIUS, not the focal depth.** The rim is the
`z = 0` datum, so the line focus — the arc's centre of curvature — sits one
sagitta shallower. Here the sagitta is **1.46 mm, nearly three wavelengths**, so
the difference is not a rounding detail: `step1` asks the probe
(`elevation_focus_depth_mm`) rather than assuming.

## Results

Vessel peak 45 cm/s at 60° → true axial peak **22.5 cm/s**. The image plane cuts
the vessel along a *diameter*, so its voxels sample the radius uniformly and the
true mean over them is `(2/3)·v_peak·cosθ` = **15.44 cm/s** — not the
`(1/2)·v_peak` that averaging over the circular cross-section would give. Step 3
evaluates it voxel by voxel from the profile that built the phantom, so that
distinction cannot be got wrong twice.

| Sequence | Emissions | Vessel mean | Scatter | Detected | Echo centre |
|---|---|---|---|---|---|
| Conventional focused | 576 | 15.11 cm/s | 8.10 cm/s | 70 % | 2.58 MHz |
| Compounded ultrafast | 72 | 15.58 cm/s | 11.10 cm/s | 64 % | 2.45 MHz |
| Compounded ultrafast, long ensemble | 288 | 14.32 cm/s | 6.50 cm/s | 75 % | 2.45 MHz |

**Claim 1 holds.** Compounded ultrafast matches conventional to **0.47 cm/s —
3.1 % of the true vessel mean — using 8.0× fewer emissions**, and both land within
2 % of the 15.44 cm/s truth (0.98× and 1.01×). The gap is comfortably inside the
run-to-run scatter of an 8-sample ensemble, which is 8–11 cm/s here.

**Claim 2 holds.** Spending the conventional budget on a 4× longer ensemble
instead of on scanning lines cuts the velocity scatter **11.10 → 6.50 cm/s** and
raises the fraction of the lumen carrying a usable flow signal from **64 % to
75 %** — the difference between the speckled colour map of the short-ensemble
rows and the filled one at the bottom.

Step 3 asserts both claims, so the example fails loudly if the simulator, the
beamformer or the estimator regresses.

## What the colour map gets right, and what it cannot

The long-ensemble map reads **0.91×** the true velocity at the vessel core and
**1.36×** at the wall. The profile comes back slightly flattened, and that is not
a calibration error — it is worth understanding because it applies to every
Doppler measurement ever made.

**A Doppler estimate is never the velocity at a point.** The lag-one
autocorrelation is computed from the echo of a whole **resolution cell**, and
every scatterer inside it contributes a phasor turning at its own rate. The phase
of their sum is approximately the **power-weighted mean velocity in the cell**.
Each voxel reports a local average of the velocity field, exactly as a B-mode
voxel reports a local average of echogenicity: Doppler inherits the imaging
system's resolution, applied to velocity.

So the flattening is set by the ratio of the cell to the vessel, and this example
is built to make that ratio small:

All three are **measured on a point target**, not computed from `λ·F#` — that
formula under-predicted the lateral width by 2.3× here, and a resolution cell
quoted from a formula is how a phantom ends up under-sampled.

| | measured | fraction of the 4.0 mm lumen |
|---|---|---|
| lateral, 9-angle compound | 1.25 mm | 0.31 |
| elevation slice at the lens focus | 0.61 mm | 0.15 |
| axial | 0.32 mm | 0.08 |

The lateral axis is the weak one, and it is inherent to the transmit: a single
**focused** transmit resolves 0.70 mm where 9-angle compounding gives 1.25 mm.
Compounding sharpens only until the tilt span matches the aperture's own
half-angle — measured, ±5° gives 1.90 mm, ±20° gives 1.25 mm, and ±26° gives
nothing more. That asymmetry is worth knowing when reading the two colour maps
against each other.

Near the **wall** the reading still runs high, and that is geometry rather than
blur alone: the cell reaches inward to faster flow while its outward side holds
no blood and contributes no flow power, so the average is one-sided.

### Elevation is the axis that bites, and designing it away is the evidence

An earlier version of this example used a **flat** elevation aperture, where the
slice was 0.56 of the lumen. It read **0.80×** at the core and **1.47×** at the
wall. Focusing the elevation aperture on the vessel, so the slice is 0.15 of the
lumen, moved that to **0.91× / 1.36×** — and the fraction of the lumen carrying a
usable flow signal roughly doubled.

**Read that as a demonstration, not a controlled experiment.** The redesign
changed the frequency, the depth and the lens together, so it does not isolate
the elevation term by itself. What it does show is that a geometry designed for a
thin elevation slice reports the profile far more faithfully — and that the axis
the image cannot show is the one that decides whether a velocity number is
trustworthy.

!!! warning "`λ·z/H` is not a safe formula for the blur width"
    On the flat-aperture configuration, fitting an elevation average to the
    measured profile gave σ ≈ 0.7 mm, which is suggestively close to `λ·z/H`. It
    was claimed here as a parameter-free prediction, and **a depth test refuted
    it.** Because the vessel is tilted it spans a range of depths inside a single
    acquisition — same probe, pulse, ensemble and estimator, only the depth
    changing — and a `λ·z/H` blur would have to grow 25 % across that span:

    | | change in core ratio over the span |
    |---|---|
    | measured | **+0.016** (band-to-band scatter ±0.017 — flat) |
    | `λ·z/H`, averaged over the blood chord | +0.048 |
    | `λ·z/H`, averaged without that restriction | +0.096 |

    Both depth-scaled models over-predict a trend the data does not show. One
    untested explanation is that an unfocused aperture's far field only begins
    near `H²/λ` (52 mm for a 4 mm aperture at 5 MHz), so the `λ·z/H` cone does not
    apply at 18–26 mm at all. Treat a fitted elevation width as measured for its
    geometry, not as a formula to carry elsewhere.

    A related modelling point that *is* settled: average over the **blood chord**
    only (`|y| < √(R² − r²)`). Outside the lumen there is no blood, so there is no
    flow signal to average in; ignoring that inflates the residual by 1.6×
    (0.76 → 1.19 cm/s) and biases the fitted width low.

## Three ways to get the centre frequency wrong

Velocity scales as `1/f`, so every error in `f` is an error of the same size in
every voxel. All three of these were made in this example and caught.

**1. Using the probe's nominal rating.** The two-way chain band-passes the drive
twice and the aperture adds its own response, so the echo here centres at
**2.45 MHz on a 3 MHz probe**. Assuming 3 MHz reads every velocity **18 % low**.
The ratio is remarkably stable across redesigns — 0.85, 0.84 and 0.82 of nominal
at 5, 12.5 and 3 MHz — but it is still measured every run, never assumed.

**2. Measuring it on the wrong signal.** Delay-and-sum low-passes the data — it
interpolates between RF samples and sums coherently across an aperture — so the
beamformed signal centres *below* the raw channel RF. Measure it on the signal the
estimator actually processes. On the earlier 5 MHz configuration this was worth
4.5 %: the channel centroid read 4.46 MHz against 4.27 MHz beamformed, and two
independent measurements pinned the beamformed value — the spectrum gave
4.268 MHz, the velocity error against known truth demanded 4.261 MHz, agreeing to
0.1 %. Fixing it took a plug-flow control from 0.955 to 0.998 of its known
velocity. It is why Loupas's estimator takes its frequency from the very IQ it
processes.

**3. Integrating over a band that no longer fits the probe.** The centroid is
computed over a frequency band, and this example's band was left at
`(2, 10) MHz` — correct for the old 5 MHz probe, and quietly fatal for a 12.5 MHz
one, because it discarded everything above 10 MHz. It reported **8.2 MHz instead
of 10.5**, scaling every velocity up by 28 %.

*How it was caught, and the general lesson:* the vessel core read **1.19×** the
true velocity. A resolution-cell average of a profile that is *peaked* at the
core cannot exceed the truth there — averaging a maximum with its neighbours can
only pull it down. A core ratio above 1.0 is therefore physically impossible, and
it is a free consistency check on any velocity calibration. `echo_center_frequency`
now defaults to the full band up to Nyquist, and both call sites pass a band
scaled to `FC` rather than absolute numbers.

Estimating the local RF frequency rather than assuming it is exactly the
correction **Loupas et al. (1995)** introduced over the classic **Kasai et al.
(1985)** estimator — *IEEE UFFC* **42**(4), 672–688. `echo_center_frequency` here
is a global, whole-frame simplification of that idea: weaker than Loupas's local
two-dimensional autocorrelation, not better.

### Previously refuted, on the 5 MHz configuration

Each of these was tested as the cause of a uniform velocity deficit and **failed**;
recorded so they are not re-tried. Wall-filter order (0 → 2 moved the mean only
6.44 → 6.73 cm/s); a depth-restricted centre frequency (moved the *wrong* way);
receive-aperture angle alone (17° → 7° recovered 2.4 %); transit-time
decorrelation (flat across a 4× change in displacement per lag, worth +0.3 %).

*A trap in that last one:* decimating slow time to probe decorrelation also halves
the Nyquist velocity. An earlier version of the test decimated the fast plug, put
12 cm/s above the 10.6 cm/s limit, and folded the estimate to a negative velocity.
The control must be slow enough that decimation stays unambiguous.

## The pipeline

| Script | Stage |
|---|---|
| `step1_define_flow_phantom.py` | Probe, pulse, vessel, all three sequences, the design-rule check. Run it directly for a 3-D preview. |
| `step2_acquire_RF.py` | Three checkpointed `sequence_rf` runs → `out/RF_{A,B,C}/`. Estimates its own runtime first. |
| `step3_doppler_processing.py` | Beamforms each sequence, runs the **identical** estimator on all three, asserts the paper's claims. |
| `step4_visualize.py` | The comparison figures. Reads stored products; never re-beamforms. |
| `bias_budget.py` | The plug-flow control and the measured error budget above. |

`doppler_tools.py` holds the processing chain — `rf2iq`, `wfilt`, `iq2doppler`,
`doppler_spectrum` — named for the conventional chain so the steps map onto what
any ultrasound processing toolbox calls them. eSDIva ships no Doppler estimator
or clutter filter; these are a few lines of numpy each, kept in the open so every
assumption behind a velocity number is visible.

```bash
uv run examples/example22_flow_doppler/step1_define_flow_phantom.py   # preview
uv run examples/example22_flow_doppler/step2_acquire_RF.py            # the long one
uv run examples/example22_flow_doppler/step3_doppler_processing.py
uv run examples/example22_flow_doppler/step4_visualize.py
uv run examples/example22_flow_doppler/bias_budget.py                 # optional
```

**Estimate the runtime on your own machine before starting.** Step 2 calls
`esdiva.utilities.estimate_sequence_runtime`, which times short probes on the
real phantom and projects. Cost per emission is a property of the hardware —
cores, memory bandwidth, threading — and varies by more than an order of
magnitude between a laptop and a compute node, so any timing quoted here is not
a prediction for your machine. It is not even stable on one machine: an identical
computation measured 31 s and 42 s per emission minutes apart, purely from
background load. For reference only: `estimate_sequence_runtime` projected
**42 min** for the 936 emissions here on a 20-core laptop.

## What Doppler actually measures

Not a frequency shift inside one echo — the **phase an echo gains between
emissions**, because the blood moved `v/PRF` in between. A scatterer receding by
`Δz` lengthens the round trip by `2Δz`, turning the phase by `4π·f·Δz/c`.

`sequence_rf` reproduces exactly that: one scatterer cloud **per emission**.

```python
positions = np.stack([np.vstack([tissue, blood_at(m)])
                      for m in range(n_emissions)])   # (N_emissions, N_scat, 3)
rf, coords = sim.sequence_rf(positions, amplitudes, events, out_path="out/RF_B")
```

eSDIva has no slow-time clock, so the trajectory is yours — which is why a
pulsatile or recirculating flow is no harder than a constant one. Scatterers are
frozen *within* each emission, so the RF carries the inter-emission phase a
Doppler estimator reads, not an intra-pulse frequency shift. Same modelling
boundary Field II works within.

**One subtlety that would silently corrupt the comparison:** each sequence
advances the blood at *its own* emission rate. The compounded ones fire all 9
angles of a frame back to back, so consecutive emissions are `1/(9·PRF_doppler)`
apart; the conventional one sends a single beam per slow-time sample,
`1/PRF_doppler` apart, with events ordered **line-major** (a whole packet on one
line, then the next). Get this wrong and one sequence's velocities are scaled by
9 — which would read as a physics result rather than a bookkeeping error.

## How each sequence is beamformed

**Conventional.** A focused transmit illuminates *one line*, so its echoes are
beamformed onto that line's voxel column only — reconstructing a whole image from
one focused transmit would invent data the transmit never insonified. The 8
emissions of a packet become that column's slow time.

**Ultrafast.** Every emission insonifies the whole box, so the 9 angles of one
frame are summed **coherently** (that is what synthesises a transmit focus
everywhere), and each compound frame is one slow-time sample. Compounding never
crosses frames — that would average away the phase carrying the flow.

## Receiver noise is added on purpose

eSDIva's RF is **noiseless by design**, and without noise the sensitivity claim
cannot be tested at all: a longer ensemble has nothing to average down. The first
run showed exactly that — the velocity scatter fell only 2.58 → 2.50 cm/s and
every sequence detected 100 % of the lumen. Step 3 therefore adds white Gaussian
receiver noise with `esdiva.utilities.add_noise`, derived once and applied
identically to all three, because noise belongs to the amplifier and not to the
sequence.

## Two statistics that do not work here, and why

- **A single-voxel gate velocity.** With an 8-sample ensemble this is dominated
  by estimator variance; comparing one voxel between sequences measures noise,
  not the method. They are compared by their *mean over the whole lumen*.
- **A maximum-velocity envelope from 8 slow-time samples.** A spectral edge
  cannot be resolved against noise from 8 samples — a real pulsed-wave system
  builds its spectrogram from hundreds. It is reported for the two short
  sequences but only *asserted* for the long-ensemble one.

## Sized to run on a laptop — and what to turn up

This study runs end to end on an ordinary laptop, and that is deliberate. None of
the choices below is a limitation of eSDIva; each keeps the acquisition short
while leaving the physics being demonstrated intact. They are knobs, and turning
one up costs runtime and nothing else — a workstation run starts from here and
simply spends more.

| Choice here | Why it is small | Turn it up when you want |
|---|---|---|
| Blood −14 dB under tissue | Real blood–tissue contrast is 40–60 dB; Field II vessel studies use 60 dB. At 60 dB the flow needs a far longer ensemble to clear the clutter. | a realistic clutter-rejection study — raise contrast **and** ensemble together |
| 8-sample ensembles | The claim is about *cost per slow-time sample*, which 8 shows as well as 64. | spectral Doppler, or any variance-limited measurement |
| 72 lines × 9 angles, 18 × 12 mm box | The single most expensive choice: a wider box needs more lines AND a longer vessel to cross it. The reported speed-up is `N_lines/N_angles` by construction. | a clinical ~100-line box, which reads 10–15× |
| tissue at 1.2 per resolution cell | Blood is what the Doppler claims rest on, so blood gets the textbook 5 per cell and tissue takes the compromise. | quantitative B-mode speckle statistics |
| `no_sub_y = 6` across the lens | Converged: 6 matches 10 patches to 0.04 mm in target depth and to the same PSF, for ~18 % less compute. | nothing — this one is free |
| Wall filter = order-0 polynomial | The phantom's tissue is perfectly still, so subtracting the slow-time mean is exactly the right filter for it. | tissue actually moves — raise the order, or `method="eig"`, the SVD family **Demené et al. (2015)** showed dominates for ultrafast data (*IEEE Trans. Med. Imaging* **34**(11), 2271–2285) |

All of it lives as named constants in `step1_define_flow_phantom.py`, so scaling
the study up is editing one file and re-running.

**One real gotcha, not a sizing choice:** the RF checkpoint fingerprint covers
the excitation and the geometry but **not** the impulse responses. Change the
pulse model and delete `out/` yourself — resume cannot detect it.

## References

1. **Bercoff, J. et al.** (2011). Ultrafast compound Doppler imaging: providing
   full blood flow characterization. *IEEE Trans. Ultrason. Ferroelectr. Freq.
   Control* **58**(1), 134–147. doi:10.1109/TUFFC.2011.1780 — **the paper this
   example replicates.**
2. **Kasai, C., Namekawa, K., Koyano, A. & Omoto, R.** (1985). Real-time
   two-dimensional blood flow imaging using an autocorrelation technique.
   *IEEE Trans. Sonics Ultrason.* **32**(3), 458–464 — the lag-one estimator in
   `iq2doppler`.
3. **Loupas, T., Powers, J. T. & Gill, R. W.** (1995). An axial velocity
   estimator for ultrasound blood flow imaging, based on a full evaluation of
   the Doppler equation by means of a two-dimensional autocorrelation approach.
   *IEEE Trans. Ultrason. Ferroelectr. Freq. Control* **42**(4), 672–688 — why
   the centre frequency is measured rather than assumed.
4. **Demené, C. et al.** (2015). Spatiotemporal clutter filtering of ultrafast
   ultrasound data highly increases Doppler and fUltrasound sensitivity.
   *IEEE Trans. Med. Imaging* **34**(11), 2271–2285 — the SVD wall filter
   available as `wfilt(method="eig")`.
5. **Jensen, J. A.** (1996). Field: A program for simulating ultrasound systems.
   *Med. Biol. Eng. Comput.* **34**, 351–353 — the moving-scatterer flow model
   this simulation shares.
