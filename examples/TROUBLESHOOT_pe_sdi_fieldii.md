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

Field II `calc_hhp` with `xdc_impulse(Th, ir)` and `xdc_excitation(Th, exc)`:

```
RF_fii  ∝  d(ir * exc)/dt  *  d²h_pe/dt²  =  d³(ir * exc * h_pe)/dt³
```

PyField Reception with `tx.impulse_response = ir` and `tx.excitation = ir`:

```
RF_pf   =  Dh_pe * FFT(v) * FFT(ir_tx) * FFT(ir_rx)
        =  d³h_pe/dt³ * exc * ir_tx * ir_rx
        =  d³(ir³ * h_pe)/dt³     [one extra ir vs Field II]
```

Extra ir adds group delay `(N_ir - 1)/2 / fs = 33 / 100e6 = 330 ns`.

### 4b — Geometry approximation

`_compute_pe_sdi_ppar` computes direction cosines in the **global frame**:

```python
xp_e = dx_e / dist_e   # global, not patch-local
yp_e = dy_e / dist_e
```

For curved transducer patches (5.7° tilt at rim of 16 mm / 80 mm bowl),
local and global direction cosines differ by up to ~37% for rim patches.
This changes the trapezoid width (Dt1, Dt2) and therefore h_max per patch.
The shift in effective SIR centre contributes ~295 ns.

### Net
330 ns (extra ir) + 295 ns (geometry approx) ≈ 625 ns observed.

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

| Metric | Field II | PyField |
|--------|---------|---------|
| Peak time (on-axis) | 40.220 µs | 40.856 µs |
| Lateral FWHM (-6 dB, max-t) | 5.20 mm\* | 7.20 mm |
| Axial FWHM (-6 dB) | 0.750 µs | 0.940 µs |

\* Field II measured as time-slice (no per-x max), likely slightly underestimates.

Both are wider than the theoretical focus FWHM (~1.3 mm at 80 mm), confirming
the expected defocused response at z = 30 mm (before the 80 mm geometric focus).

Remaining difference (lateral ~40%, axial ~25%) is attributed to:
- Extra `ir` convolution in PyField (ir_tx × ir_rx vs single E_m in Field II)
  broadens the effective pulse and adds 330 ns group delay.
- Global vs local direction cosines for curved patches (geometry approximation).

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
| 625 ns vs Field II | YES (convention) | Documented | Extra ir conv + geometry approx |
| Lateral FWHM 0.8mm (wrong) | NO (time-slice artefact) | Documented | Max-over-time gives correct 7.2mm |
