# Investigation Prompt — PyField vs Field II Reception "Smoothing" Discrepancy

## Goal

Find and fix the root cause of the residual **lateral/temporal "smoothing"** (loss of
edge-wave crispness) seen when comparing PyField's pulse-echo reception against Field II,
which caps cross-correlation at **~0.93** even after all geometry issues are resolved.

**Key observation to pursue:** In **Emission** (`calc_hp`-equivalent forward pressure) the
agreement with Field II is high — the discrepancy is small. The smoothing is **specific to
Reception** (`Reception` and `ReceptionSDI`, i.e. the pulse-echo path). So the cause is most
likely in something that lives **only on the pulse-echo side**: the transfer-function / IR
handling, the `(jω)³` derivative model, the `h_pe = h_tx ⊛ h_rx` combination, the PE-SDI
delta construction, or the scale/convolution chain — **not** the shared SIR geometry.

## What was already established this session (do NOT re-investigate — treated as solved)

### 1. Time/axial alignment — FIXED
- Concave PE peak was ~0.66 µs too deep. Decomposed:
  - **0.52 µs** = geometry convention mismatch. Field II `xdc_concave(R, Rfocus, ele)` places
    the bowl **apex at z=0**, `Rfocus` = radius of curvature. PyField
    `ConcaveCircularTransducer` places the **rim at z=0**, apex recessed to `z = -sag`, and
    `focus_mm` = axial focal depth (`R = √(focus_mm² + r_ap²)`). Same scatterer coord `z` is
    ~`sag` closer to the PyField surface → `2·sag/c` late.
  - **0.15 µs** = pulse/derivative model (residual, see below).
- Fix applied in `compare_psf_fieldii.py`: shift field points by the PyField apex-z so both
  measure depth from the apex. Peak Δ dropped 0.65 → 0.13 µs; on-axis dB error 48 → ~5 dB.
- `pe_t0 = t0_tx + t0_rx` is self-consistent (kernel writes deltas at absolute times via
  `floor((t_corner − t0)·fs)`); the `0.5·size_patch/c` term in `compute_time_grid` does NOT
  shift the peak. Timing is not the smoothing cause.

### 2. Spatial "completely different" — was a PLOTTING ARTIFACT
- `compare_psf_fieldii.py` gave each panel its own time extent; PyField's longer SIR tail
  stretched its panels. On a **common time axis** the three are 98% similar (2D corr
  PyField↔Field II = 0.982 full, 0.93 above −30 dB; naive↔SDI = 0.998).
- Fix applied: all 2D panels share Field II's time window (`ylim`); time-axis line plots
  share `xlim`.

### 3. The "smoothing"/blur — NOT geometry, NOT tiling, NOT patch count, NOT fs (ALL RULED OUT)
Decisive experiments:
| Test | Result |
|------|--------|
| `no_sub_diameter` 16 → 48 (208 → 6636 patches) | corr 0.93 → 0.94, no sharpening |
| `fs` 100 → 800 MHz | corr 0.933 → 0.930, FLAT, images identical (rules out `dt`-clamp at `farfield_rect_patch.py:47-51`) |
| `refine_factor` 1 → 3, `method` cartesian → spherical | corr 0.933 → 0.942, marginal |
| **Import Field II's EXACT 208 patches** via `from_fieldii_xdc_data` (`concave_data.mat`) | corr **0.92/0.93** — identical to PyField's own tiling |

**Conclusion:** With Field II's *exact* rectangles (same centers, same 1 mm elements, apex
z=0, RX channels summed), PyField reproduces the SAME blur and SAME ~0.93 ceiling. Therefore
the smoothing is **localized to the per-element SIR kernel and/or the pulse-echo signal
chain — the parts common to every transducer path.** Geometry/tiling/rim/patch-count/fs are
eliminated.

### 4. naive vs SDI ~0.02 µs (1–2 sample) phase shift — INHERENT, not a bug
- `Reception` (naive): `h_tx ⊛ h_rx` on SIR grid, then `(jω)³` applied analytically in freq
  domain. `ReceptionSDI`: 3 derivatives baked into `Dh_pe` as 16 quantized Dirac deltas/pair
  + `+2.0` bin-offset convention + 1 cumsum.
- Difference = delta-bin quantization vs analytic differentiation. Shrinks in µs as `fs→∞`,
  never bit-identical. Expected to agree only to ~SDI tolerance (corr >0.95; here 0.998).

## Leading hypotheses for the remaining ~0.93 ceiling (where to focus next)

Ranked by suspicion given "Emission OK, Reception smoothed":

1. **Per-element SIR shape (far-field trapezoid vs Field II's per-element response).**
   PyField uses ONE far-field trapezoid per patch. At z=30 mm a 1 mm element is ~4× past the
   far-field validity limit `w << √(4·l·c/f) ≈ 0.25 mm`. Field II's per-element response is
   sharper (internal subdivision and/or exact-rectangle Lockwood–Willette formula).
   - **BUT** subdividing PyField patches to 6636 did NOT sharpen → either the trapezoid does
     not converge as expected, OR a second ceiling (below) dominates. **Resolve this tension
     first.** Suggested unit test: compare a SINGLE flat rectangle's PyField trapezoid SIR
     against the closed-form exact rectangle SIR at a near-normal field point — does PyField
     converge to it as the patch is subdivided?

2. **Pulse-echo derivative / TF model — REASON EMISSION ≠ RECEPTION.**
   - Emission applies fewer derivatives; Reception applies `(jω)³` (CLAUDE.md §5/§6:
     `v_pe = (ρ₀/2c₀²)·E_m ⊛ ∂³v/∂t³`). If the derivative COUNT or the way `h_pe = h_tx ⊛ h_rx`
     is combined differs from Field II's `calc_hhp` model, the pulse-echo spectrum is shaped
     differently → broader/smoother envelope **only in reception**. This matches the symptom.
   - Check: how many time derivatives does Field II `calc_hhp`/`calc_scat` apply for
     pulse-echo (2 vs PyField's 3)? An extra derivative or a different `jω` power changes the
     effective bandwidth → smoothing. **High-priority suspect.**
   - Check the transfer-function / IR multiply chain in `reception.py` and `reception_sdi.py`
     (`fft_v_pe`, `fft_ir_tx`, `fft_ir_rx`): is anything applied twice, with wrong sign, or
     band-limiting the result more than Field II does?

3. **`h_tx ⊛ h_rx` double-SIR broadening.** Convolving two approximate (trapezoid-broadened)
   SIRs squares the broadening — could explain why Reception smooths more than Emission (which
   uses h_tx only). Quantify: compare a single-element `h_tx` vs Field II `calc_h`, then
   `h_pe` vs Field II `calc_hhp`, to see if broadening enters at the SIR stage or the
   convolution stage.

## Concrete next steps

1. **Isolate Emission vs Reception quantitatively.** Run forward pressure (`Emission`) for the
   concave at the same points and correlate vs Field II `calc_hp` — confirm Emission corr is
   high (e.g. >0.98). This pins the defect to the pulse-echo-only operations.
2. **Single-rectangle SIR unit test** (hypothesis 1): PyField trapezoid vs exact analytic
   rectangle SIR; test convergence under subdivision.
3. **Derivative-count audit** (hypothesis 2): verify Field II `calc_hhp` derivative order;
   compare against PyField `(jω)³`. Try `(jω)²` and compare envelope sharpness/corr.
4. **Stage-by-stage broadening** (hypothesis 3): compare `h`, then `h_pe`, then post-pulse RF
   against Field II equivalents to find which stage injects the smoothing.
5. If hypothesis 1 confirmed: scope **auto-subdivision of each patch to the far-field limit**
   (`w < √(4·l·c/f)`) before the trapezoid — lower-risk than implementing exact-rectangle SIR.

## Reference geometry / files

- Field II ref RF: `examples/rf_concave_psf.mat` (concave Ø16 mm, Rfocus=80 mm, 3 MHz,
  scatterers z=30 mm, 101 lateral −10..10 mm, fs=100 MHz, 2-cycle Hanning IR, plain-sine exc).
- Field II exact geometry: `examples/fieldiiexamples/concave_data.mat` = raw `xdc_get(Th,'rect')`
  matrix, shape **(26, 208)**. Row map (0-indexed): row0 phys#, row1 math#, row2/3 width/height
  (1 mm), row4 apod, rows7-9 center (x,y,z), **rows10-21 = the 4 corners** (×3 coords), row22
  delay. Build with `FieldIITransducer(quads, apod, delay, frequency_hz=3e6)`. **It is
  per-patch elements (n_elements=208) → monostatic RX must SUM channels** (`rf.sum(axis=2)`).
- Comparison script: `examples/compare_psf_fieldii.py` (already has apex-shift + common-axis
  fixes). `examples/example06_concave_PSF.py` is the naive-vs-SDI-only demo (no Field II).
- Kernel: `src/pyfield/hsir/farfield_rect_patch.py` (trapezoid; `dt`-clamp at lines 47-51).
- Reception: `src/pyfield/reception/reception.py` (`Reception`, `(jω)³`),
  `src/pyfield/reception/reception_sdi.py` (`ReceptionSDI`, PE-SDI),
  `src/pyfield/hsir/transducer_sir_pe.py` (`compute_pe_sdi`, `+2.0` offset).
- Compat import: `src/pyfield/transducers/fieldii_compat.py`.

## Constraints / gotchas

- Windows: run with `PYTHONUTF8=1 PYTHONIOENCODING=utf-8` (cp1252 chokes on µ/→ in verbose).
- `pytest filterwarnings = error` — handle UserWarnings (e.g. patch coverage >105%).
- After editing Numba kernels, clear `.nbi/.nbc` cache (CLAUDE.md Risky §6).
- Scale convention: PyField `rho/(2c²)`, Field II `rho/2`; raw amplitude differs by ~c² but
  normalized PSF is unaffected — compare envelopes/normalized, never raw amplitude.
- SIR comparisons need tolerance (float32 cumsum tail ~0.004%): `rtol≈0.005`, never `atol=0`.
