# PE SDI Validation — Troubleshoot Log

Investigation of the pulse-echo SDI (PE SDI) implementation vs Field II reference.
Date: 2026-05-29.  Diagnostic scripts: `diag_pe_sdi_timing.py`, `diag_fieldii_matfile.py`.

---

## Issue 1 — PE SDI delta offset: +1 vs +2

### Symptom
`compare_psf_fieldii.py` showed PyField RF peak systematically later than expected.
Internal test (CASE 2, flat 4-patch array) showed PE SDI peak 1 sample early vs
`dh_tx * d2h_rx` FFT-conv reference.

### Root cause
`_place_pe_sdi_2d` used `kf = (t_event - t0) * fs + 1.0` (same as single SDI).
But single SDI places d2h events so that after 1 cumsum the first step is at bin
`Ne = floor((t1 - t0)*fs) + 1`.
For PE SDI, the combined event of TX corner i and RX corner j must land at
`Ne_i + Nr_j` so both cumsums align.  With offset +1, the PE event lands at
`floor((t1e + t1r - pe_t0)*fs) + 1`, but the reference sum is
`Ne_i + Nr_j = floor(...) + 2`.  One-sample shift (10 ns at 100 MHz).

### Fix
Changed offset from `+1.0` to `+2.0` in all 4 TX-corner placements in
`_place_pe_sdi_2d` (`sir_derivatives.py`).

### Verification
- CASE 2 (flat 4-patch): timing diff = 0.00 ns, amplitude ratio = 1.009.
- CASE 4 (concave 208-patch): timing diff = 0.00 ns, amplitude ratio = 0.9989.
- All existing unit tests pass.

---

## Issue 2 — Apparent 59% pre-onset leakage in concave Dh_pe

### Symptom
`diag_pe_sdi_timing.py` CASE 3 reported:

```
Leakage ratio (before/peak): 5.91e-01
Dh_pe first nonzero at idx 129, expected onset at idx 131
```

59% of peak amplitude before the "expected onset". Appeared to confirm user's
concern about SDI leakage.

### Root cause (diagnostic, not code bug)
`exp_first_idx` was computed as `floor((min_rt - pe_t0) * fs + 1.5)` where
`min_rt = 2 * min_dist / c` (direct round-trip to closest patch).

The actual first SDI event is at `t1e + t1r - pe_t0` where
`t1 = dist/c - 0.5*(Dt1 + Dt2)`.  This is legitimately earlier than `min_rt`
by `0.5*(Dt1+Dt2)` per side.  The "pre-onset" window (indices 0–130) contains
the very start of the real signal (indices 129–130), not spurious energy.

Indices 0–128 are exactly zero (no leakage whatsoever).  The 59% ratio is because
the expected_onset estimate is 2 samples too late.

### Real leakage measurement
RF signal pre-onset amplitude / peak: **7.86e-7** — negligible float noise.

### No code fix needed.

---

## Issue 3 — DC tail in Dh_pe (float32 cancellation)

### Symptom
`diag_pe_sdi_timing.py` CASE 3:

```
Dh_pe tail value at idx exp_last + 20: -7.397e+19
Peak:                                   1.391e+23
Ratio:                                  5.3e-4  (0.053%)
```

Dh_pe does not return to zero after the last SDI event.

### Root cause
`_place_pe_sdi_2d` accumulates delta weights into a `float32` array.
With 208^2 × 16 ≈ 700k events of magnitude ~1e14 each, float32 relative
precision ~1e-7 gives cancellation residual ~7e12 per event pair.
Over all pairs, the uncancelled sum can reach ~1e19.

The cumsum step already uses a float64 accumulator (CLAUDE.md risky impl #1),
so the error does not compound further.  But the delta-placement residual
propagates as a constant DC offset in Dh_pe.

### Impact
The DC offset in Dh_pe is **multiplied by FFT(excitation)[0]** during the
reception convolution.  The excitation is a Hanning-windowed sine: zero mean.
`FFT(ir)[0] = sum(ir) ≈ 0`.  The DC component is suppressed by ~6 orders of
magnitude.  Final RF pre-onset ratio: 7.86e-7 → no visible artifact.

### Mitigation
For very large arrays (M >> 200) or strong scatterers, a float64 delta buffer
would reduce the tail further.  Not implemented — the 0.053% tail is physically
negligible for all tested cases.

---

## Issue 4 — Field II mat file: 5x DT error in Python loading

### Symptom
`compare_psf_fieldii.py` reported:

```
PyField peak:  40.845 us
Field II peak: 38.780 us
dt_peak = 2065 ns
```

2 µs difference appeared to indicate a major PE SDI timing error.

### Root cause
The Matlab script `example_concave_psf.m` downsamples the RF envelope:

```matlab
env = abs(hilbert(RF_data(1:5:600,:)));   % every 5th row → 50 ns/sample
[N,M] = size(env);                         % N = 120
mesh(xpoints, ((0:N-1)/fs + start_time)*1e6, env)  % time axis uses 1/fs !
```

`RF_data(1:5:600,:)` selects every 5th row (step size 5 in Matlab `start:step:stop`
syntax), so `env` has samples spaced `5/fs = 50 ns` apart.
But the time axis in `mesh` uses `(0:N-1)/fs` (10 ns steps), compressing the
display by 5×.  The mat file stores this downsampled envelope plus the
`start_time` (= time of sample 1 of the FULL RF_data, i.e., 10 ns spacing).

Python loaded with `DT = 1/fs = 10 ns` instead of `DT = 5/fs = 50 ns`.

### Corrected peak
Field II corrected peak = `t0 + 36 * 50 ns = 38.420 + 1.800 = 40.220 µs`.

| Metric | Before fix | After fix |
|--------|-----------|-----------|
| Field II peak | 38.780 µs | 40.220 µs |
| PyField peak  | 40.845 µs | 40.845 µs |
| Difference    | 2065 ns   | **625 ns** |

### Fix
Changed `fii_dt = 1.0 / FS` → `fii_dt = 5.0 / FS` in:
- `diag_fieldii_matfile.py`
- `compare_psf_fieldii.py`

---

## Residual 625 ns difference (not a bug)

After fixing the DT error, 625 ns remains between PyField (40.845 µs) and
Field II (40.220 µs).  Two independent contributors:

### 4a — Different signal chain (excitation convolution count)

Field II `calc_hhp` with `xdc_impulse(Th, Hanning_ir)` and `xdc_excitation(Th, plain_exc)`:

```
RF_fii  ∝  d(Hanning_ir * plain_exc)/dt  *  d²h_pe/dt²
        =  d³(Hanning_ir * plain_exc * h_pe)/dt³
```

Both TX and RX in Field II use `Th`, so both get Hanning_ir applied.
Signal chain: `plain_exc ⊛ Hanning_ir_tx ⊛ Hanning_ir_rx`.

**Original bug in `compare_psf_fieldii.py`** (before 2026-06-01 fix):
```python
ir = sin(2πf₀t)                # plain sine — WRONG
tx.impulse_response = ir
tx.excitation = ir * hanning    # Hanning on excitation — WRONG
```
PyField chain: `(Hanning_sin) ⊛ plain_sin ⊛ plain_sin` = `Hanning × sin³`.
Field II chain: `plain_sin ⊛ Hanning_sin ⊛ Hanning_sin` = `Hanning² × sin³`.
One fewer Hanning window → broader PSF, 330 ns group delay difference.

**Fixed (2026-06-01)**:
```python
ir = sin(2πf₀t) * hanning(...)  # Hanning on IR — CORRECT
tx.impulse_response = ir
tx.excitation = sin(2πf₀t)      # plain sine — CORRECT
```
Now PyField chain = Field II chain exactly.

### 4b — Geometry approximation (ALREADY FIXED)

The TROUBLESHOOT doc (2026-05-29) documented that `_compute_pe_sdi_ppar`
used global direction cosines. This has since been fixed: both
`farfield_rect_patch.py` and `transducer_sir_pe.py` now project the
displacement vector onto local patch frame tangents (`eu`, `ev`) before
computing `xp`, `yp`. The ~295 ns contribution from this approximation
no longer applies.

### Net (after both fixes)
Signal chain fix removes 330 ns. Geometry approximation already fixed.
Residual timing difference should be ≤ 50 ns (numerical/discretisation).

---

## PSF spatial agreement

### Measurement method matters

Field II stores the downsampled envelope.  Lateral FWHM measured as a time-slice
at the global peak row underestimates the true width because off-axis scatterers
peak at different times.

For PyField: each lateral scatterer is an independent simulation.  The on-axis
peak is at 40.856 µs; x = ±10 mm peaks at 39.886 µs (0.97 µs spread).
A time-slice at the on-axis peak gives 0.80 mm FWHM (misleadingly narrow).
Max-over-time gives the correct 7.20 mm FWHM.

### Comparison (all at z = 30 mm, focus = 80 mm, diameter = 16 mm)

Measurements below are from the **old** (wrong-signal-chain) run.  After the
2026-06-01 fix (`compare_psf_fieldii.py`) the PSF should be re-measured and
this table updated.

| Metric | Field II | PyField (old) | PyField (fixed, TBD) |
|--------|---------|---------------|----------------------|
| Peak time (on-axis) | 40.220 µs | 40.856 µs | ~40.3 µs (expected) |
| Lateral FWHM (-6 dB, max-t) | 5.20 mm\* | 7.20 mm | closer to 5.2 mm |
| Axial FWHM (-6 dB) | 0.750 µs | 0.940 µs | closer to 0.75 µs |

\* Field II measured as time-slice (no per-x max), likely slightly underestimates.

PSF spatial structure (main lobe shape, absence of spurious side-lobes)
qualitatively agrees between PyField and Field II.

---

## Summary

| Problem | Real? | Fixed? | Impact |
|---------|-------|--------|--------|
| PE SDI offset +1 vs +2 | YES | YES | 10 ns timing shift, corrected |
| 59% leakage in Dh_pe | NO (diagnostic artefact) | N/A | None |
| DC tail 0.053% | YES (float32) | Accepted | Killed by zero-mean excitation |
| Mat file DT 10ns vs 50ns | YES | YES | 2065 ns apparent error → 625 ns real |
| Signal chain: wrong IR/exc assignment | YES | YES (2026-06-01) | Extra Hanning window → PSF broadening + 330 ns |
| Geometry: global vs local cosines | YES | YES (pre-existing fix) | Both kernels already use local patch frames |
| Scale convention: rho/2c² vs rho/2 | YES (convention) | Documented | Raw amplitude differs by c²; normalised PSF unaffected |
| Lateral FWHM 0.8mm (wrong) | NO (time-slice artefact) | Documented | Max-over-time gives correct 7.2mm |
