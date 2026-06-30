# `_research/` — PyField vs Field II benchmarking (concave PSF)

Working notes + scripts from the session that validated PyField's pulse-echo PSF
against Field II for the concave Ø16 mm, focus 80 mm transducer (3 MHz, fs 100 MHz,
scatterer at z = 30 mm). Reference data: `../rf_concave_psf.mat` (Field II env, dB,
decimated ÷5 → 50 ns grid) and `../fieldiiexamples/concave_data.mat` (Field II
`xdc_get` element geometry, 208 flat rects).

---

## Problems encountered & solutions

### 1. ReceptionSDI leakage on finer subdivision — FIXED
**Symptom:** changing `no_sub_diameter` + `refine_factor=2` produced sustained
vertical streaks in time (DC/ramp tails) in `ReceptionSDI` only. Conventional
`Reception` was unaffected.

**Cause:** `_compute_rectangle_SIR_params` clamps `Dt1,Dt2 → dt` for sub-sample
patches, so `slope = h_max/Dt1 ~ area/dt²` explodes. The PE-SDI weight is
`slope_e · slope_r` (**squared**, ~1e20). The delta buffer was **float32** → ULP
≈ 1e13 at that scale → catastrophic cancellation across patch pairs → DC residual
→ ramp after the single cumsum.

**Fix:** float64 accumulation buffer in the PE-SDI kernel
(`src/pyfield/hsir/transducer_sir_pe.py`: `out` + cumsum store → float64, cast to
float32 at the `compute_pe_sdi` boundary). Leak gone (tail/peak ~1e-3, stable
208→1680 patches); 49 reception/SDI tests pass.

### 2. Reception(FST) vs ReceptionSDI amplitude mismatch — DIAGNOSED (`16_FST_vs_sdi_amp.py`)
**Symptom:** conventional `Reception` peak amplitude differs from `ReceptionSDI`.

**Cause:** exactly a factor of `dt = 1/fs`. Ratio FST/SDI = `fs` (50/100/200 MHz →
50e6/100e6/200e6, <1% error), independent of `fc`. `FST·dt == SDI` (amplitude
ratio 1.005, waveform corr 0.945 = the usual FST-vs-SDI numerical diff, not scale).

The factor lives in the two-way SIR convolution `h_tx ⊛ h_rx`:
  * FST does it as an FFT product `irfft(H_tx·H_rx)` = DISCRETE conv = (1/dt)·continuous;
  * SDI does it by delta placement (δ⊛δ, weights multiply → continuous) + cumsum.
All exc/IR convolutions are FFT products in BOTH engines, so they cancel — FST
just has ONE extra FFT-product conv (the SIR-SIR one) that omits the continuous-
convolution `dt`. Since `(h_tx⊛h_rx)(t)=∫h_tx h_rx dτ ≈ dt·Σ`, **conventional
`Reception` is too big by `fs`; SDI is correct for that step.**

**Fix (applied):** folded one `dt` into `scale` in `Reception._compute_rf_inner`
(`scale = rho/(2c²)·dt`), with a comment explaining the continuous-vs-discrete
convolution origin. Post-fix `peak(FST)/peak(SDI) = 1.005`, corr 0.945 — agree to
~0.5% (not bit-identical: sampled-trapezoid+`(jω)ⁿ` vs quantised-delta+cumsum are
different operators, so a few-% residual is expected). 49 reception/SDI tests pass.
Normalised PSFs unaffected.

### 3. "PyField looks temporally smoother than Field II" — EXPLAINED (PyField is correct)
The visible difference is **one on-axis interference null**: Field II dips deeper
(~7.8 dB) than PyField (~5.6 dB). Everything else (lobe positions, FWHM, lateral
profile) matches; PyField's *lateral* nulls are if anything sharper.

Systematically ruled out (all with numbers):

| Hypothesis | Test | Verdict |
|---|---|---|
| Wrong convolution chain | FWHM 990 vs 1000 ns; drop exc → 680 ns | chain correct |
| Time-sampling / `Dt→dt` clamp | fs 100/200/400 MHz | flat 5.6 dB → not it |
| Far-field trapezoid / patch count | no_sub 16→160 (1→0.1 mm) | converges 5.6→6.1 → not it |
| Bowl tiling geometry | import Field II's exact rects | not it |
| Bowl faceting (flat rects) | Rayleigh on Field II's 208 rects | 5.7 dB → not it |
| Signal-chain variant | sweep `(jω)^{0,1,2}·exc·ir^{1,2}` on gold SIR | none hit 7.8 @ 1000 ns |
| Envelope processing | decimate ÷5 like Field II | 6.0 vs 6.1 → not it |

**Gold-standard verdict (Rayleigh-Sommerfeld integral):** the exact continuous
physics gives a **6.1 dB** on-axis null — matching PyField (5.6–6.1), **not** Field
II (7.8). The user's convergence argument is right: PyField's far-field trapezoid
sum converges to the exact Rayleigh integral (one-way SIR at `no_sub=64` overlays
the gold top-hat). **PyField is faithful to exact physics; Field II's extra ~1.7 dB
null depth is internal to its `calc_hhp` (time-bin SIR integration / derivative
model) and is not ground truth.**

---

## The Rayleigh gold standard

`rayleigh_gold_standard.py` — the reusable benchmark.

Direct numerical Rayleigh-Sommerfeld SIR (baffled aperture):

    h(r_p, t) = ∫_S  δ(t − |r_p − r_s|/c) / (2π|r_p − r_s|)  dS

Discretisation: tile the aperture into K tiny surface elements (area `dA_k`,
distance `R_k`); each drops weight `dA_k/(2π R_k)` at delay `R_k/c`, linearly split
across 2 time bins, scaled by `fs` (unit-area delta → sampled density). As K → ∞
this is the **exact** continuous SIR — the limit both PyField and Field II converge
to. Validated: 26k / 288k / 1.1M elements all give the same null depth.

### What it outputs
- **Left panel:** one-way SIR on-axis — Rayleigh gold vs PyField `no_sub=16` / `64`
  (convergence check; `no_sub=64` overlays the gold top-hat).
- **Right panel:** on-axis pulse-echo envelope, **peak-aligned** — Rayleigh gold,
  PyField FST, PyField SDI, Field II. (Peak-aligned because absolute `t0`
  conventions differ across engines + the manual gold chain: exc/IR group delay,
  SIR-only `t0`. Align by peak to compare shape + null depth fairly.)
- **Console:** the central-null depth (dB) for all four.

### Run
```bash
cd examples
uv run python _research/rayleigh_gold_standard.py
```

---

## How to extend the benchmark

**Add a field point / sweep depth:** change `ZS` (scatterer depth, mm) and `CI`
(on-axis lateral index). For a full lateral PSF, loop `X` and call `rayleigh_sir`
per point (the chain helper `on_axis_psf` shows the per-point recipe).

**Compare a different quantity:** the gold pulse-echo chain is
`(jω)¹ · exc · ir² · (h_gold ⊛ h_gold)` — `calc_hhp`. For scattered RF
(`calc_scat`) use `(jω)³`. Keep the SAME chain on the gold and on PyField so the
comparison isolates the SIR engine, not the chain.

**Add another transducer:** replace `sample_bowl` with a dense surface sampler for
the new geometry (return `(K,3)` points in metres + `(K,)` areas). Everything else
is geometry-agnostic. For flat/faceted apertures, bilinearly sample each rectangle
(see the deleted faceted test in git history, or sample the patch corners).

**Tighten convergence:** raise `n_theta, n_phi` in `sample_bowl` until the null
depth stops moving (already converged at 400×720 = 288k here).

### IR-count check (is Field II's deeper null from using 1 IR not 2?)
No. Field II `calc_hhp(Th, Th)` convolves BOTH the emit and receive aperture
impulse responses → `ir²`, same as PyField. On the exact gold SIR:

| chain | dip | FWHM |
|---|---|---|
| `(jω)¹·exc·ir²` (= calc_hhp) | 6.1 dB | 920 ns |
| `(jω)¹·exc·ir¹` (one IR) | 10.1 dB | **700 ns** |

One IR makes the PSF far too narrow (700 vs Field II's 1000 ns) — wrong. `ir²` is
correct and gives the right FWHM. The IR count is NOT the source of Field II's
deeper null. (Reproduced in the script's console "IR-count check".)

### Gotchas
- **`t0` from the apex, not the rim:** anchor the Rayleigh time grid on the TRUE
  minimum surface distance (`min |surf − field_pt|`), not `sqrt(R_AP²+z²)` (that is
  the rim distance and is LATER than the on-axis apex arrival → the early SIR falls
  in negative bins and the gold curve gets truncated on the left).
- **Time axis:** the gold one-way SIR is referenced to its `t0`; the pulse-echo
  autoconvolution `h⊛h` is therefore referenced to `2·t0` (not `t0`). The exc/IR
  group delay is NOT added to `t0` (matches PyField's `coords["t0"]` convention).
  One-way panel overlays directly on PyField's axis; for the pulse-echo cross-engine
  overlay use peak-alignment (engine `t0` conventions differ).
- **`concave_data.mat` layout** (Field II `xdc_get`, shape `(26, 208)`, columns =
  patches): rows 7-9 = patch centre, rows 10-21 = 4 corners (x,y,z, winding order),
  row 4 = apodization, rows 2-3 = width/height (1 mm).
- **PyField one-way SIR:** `Emission(tx)(pts)` with no excitation returns the raw
  SIR `h` (Mode 2 / pulsed-pure), not `dh/dt`.
- **Scale convention:** PyField uses `ρ/(2c²)`, Field II `ρ/2` — raw amplitudes
  differ by `c²`. Normalised/envelope comparisons unaffected.

---

## Other scripts here
- `05_remaining_diff.py` — earlier on-axis FWHM / lateral-width diff probe.
- `01`–`04`, `overlay_kfix.png`, `remaining_diff.png` — earlier SIR-shape / k-offset
  / SDI double-integration probes (pre-gold-standard).
