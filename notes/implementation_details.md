# PyField SDI — Implementation Reference

Non-obvious details from all refactoring batches.
Ordered by importance. Focus on continuous→discrete and PE SDI history.

---

## SECTION 1 — Trapezoidal SIR: Continuous Physics

For rectangular patch `m`, half-widths `wx`, `wy`, local tangent axes `e_u`, `e_v`:

```
dist = |r_p - r_m|
u    = (r_p - r_m) / dist        # unit direction vector

xp = (d · e_u) / dist  = (dx*eu[0]+dy*eu[1]+dz*eu[2]) / dist   # LOCAL frame
yp = (d · e_v) / dist  = (dx*ev[0]+dy*ev[1]+dz*ev[2]) / dist   # LOCAL frame

Dt1  = min(wx*|xp|, wy*|yp|) / c   # shorter side crossing time
Dt2  = max(wx*|xp|, wy*|yp|) / c   # longer  side crossing time
```

If `Dt1 < dt`: clamp to `dt` (prevents division-by-zero in `h_max = area/Dt2`).
If `Dt2 < dt`: clamp to `dt`.

```
t1   = dist/c - (Dt1+Dt2)/2 + delay     # first corner TOF
t2   = t1 + Dt1
t3   = t1 + Dt2
t4   = t1 + Dt1 + Dt2
h_max = wx*wy / (2π * Dt2 * dist)       # plateau amplitude
slope = h_max / Dt1                      # rising/falling slope
```

**Area invariant**: `∫h dt = wx*wy / (2π*dist)`.  
**Far-field validity**: `w ≪ sqrt(4*dist*c/f)` — controls subdivision density.

---

## SECTION 2 — SDI: Continuous → Discrete

### Continuous: 2nd derivative = 4 weighted Dirac deltas

```
d²h/dt² = slope * [+δ(t-t1) - δ(t-t2) - δ(t-t3) + δ(t-t4)]
```

Signs: `(+, -, -, +)`. Recover h via double integration:
```
d²h/dt²  →[cumsum, NO dt]→  dh/dt  →[cumsum, ×dt]→  h
```

### Discrete delta placement: offset `+1`

```python
kf       = (t_corner - t0) * fs + 1.0    # ← +1 is critical
kf_floor = int(floor(kf))
w_ceil   = kf - kf_floor                 # fractional upper weight
w_floor  = 1.0 - w_ceil
out[p, kf_floor]   += slope * w_floor    # 2 writes per corner = 8 total
out[p, kf_floor+1] += slope * w_ceil
```

**Why +1**: after cumsum, a delta at bin `N` produces a step at bin `N`.
The discrete step must land at `N = floor((t_corner - t0)*fs) + 1`
— the first bin AFTER `t_corner`. Using `+0` would step one sample early.

### First cumsum (d²h → dh): NO `dt` multiplication

**Why no dt**: SDI deltas encode amplitude as `slope` (already dimensionally
correct). The cumsum of a delta distribution gives a step directly —
no Euler-step `dt` factor needed. `d2h` values are NOT continuous samples,
they are discrete Dirac delta weights.

```python
acc = np.float64(0.0)       # FLOAT64 MANDATORY
acc += np.float64(arr[k])
out[k] = np.float32(acc)    # write back float32 — NO * dt
```

### Second cumsum (dh → h): WITH `dt` multiplication

`dh` is a piecewise-linear (continuous) function sampled on a grid.
Numerical integration requires the `dt` time-step factor.

```python
out[k] = np.float32(acc) * dt32    # ×dt here
```

### Float64 accumulator: MANDATORY for both cumsums

At SDI event magnitudes (~4×10¹⁰), float32 ULP ≈ 4096.
Large cancelling events leave a DC residual → linear ramp after double cumsum.
Float64 accumulator + float32 write-back reduces tail to ~0.004% of peak.

**h_new ≠ h_ref in tail by design**: the reference kernel (`farfield_rect_patch.py`)
writes first cumsum in-place to the same float32 array, then re-reads.
Modular design (separate cumsum functions) uses a new array → different
intermediate rounding → different tail sign. Both are correct to within
float32 precision. Use `rtol=0.005, atol=0.005×peak` for SIR comparisons.

### Standard SIR test tolerance

```python
peak = float(np.abs(expected).max())
np.testing.assert_allclose(actual, expected, rtol=0.005, atol=0.005 * peak)
```

Real physics bugs (wrong patch, geometry error) → errors > 1% of peak.
Float32 arithmetic artifacts → < 0.005% of peak.

### d2h_all ≠ d2h_per_element.sum() (float32 non-associativity)

`compute_d2h` and `compute_d2h_per_element` accumulate in different order.
Float32 addition non-associative → difference up to 1 ULP (4096) at event scale.
After cumsum → constant offset of ~4096–6144 in `dh` plateau region.
**Never compare them with atol=0.** Same tolerance as above.

---

## SECTION 3 — Time Grid

```python
t0 = (min_dist - 0.5*(wx + wy)) / c
```

Subtracts `0.5*(wx+wy)/c` because earliest possible SIR corner is
`t1 = dist/c - (Dt1+Dt2)/2` where `(Dt1+Dt2)/2 ≤ (wx+wy)/(2c)`.
Without this, early events (especially for on-axis points near large patches)
fall before index 0 and are silently lost.

**PE SDI time grid:**
```python
pe_t0 = tx_t0 + rx_t0          # sum of single-path starts
pe_T  = tx_T + rx_T - 1        # linear convolution output length
```

**Field II offset**: Field II uses `~2*min_dist/c` (no patch-width correction).
PyField `pe_t0` is `(wx+wy)/c ≈ 694 ns` earlier for a 1×1 mm patch at 100 MHz.
Correct behavior — PyField grid starts at true first SIR event.

---

## SECTION 4 — PE SDI: Pulse-Echo Combined (CRITICAL — history of issues)

### Theory

```
zeta_pe = d²h_tx/dt² ⊛ d²h_rx/dt²   →  16 Dirac deltas per (m_e, m_r) pair
Dh_pe   = ∫zeta_pe dt  =  dh_tx/dt ⊛ d²h_rx/dt²   (only 1 integration needed)
```

Signal chain (no derivative on excitation — already absorbed into Dh_pe):
```
rf = (E_tx × E_rx × v) ⊛ Dh_pe × (rho / 2c²)
```

### Discrete placement: offset `+2` (not +1)

```python
kf = (t_corner_e + t_corner_r - pe_t0) * fs + 2.0    # +2 for PE SDI
```

**Why +2**: FFT convolution of `dh_tx` (step at `Ne = floor(t_e)*fs + 1`) and
`d²h_rx` (delta at `Nr = floor(t_r)*fs + 1`) places first nonzero at bin
`Ne + Nr = floor(...) + 2`. PE SDI must match exactly.
Using `+1` → 1-sample early = 10 ns at 100 MHz axial shift.

**Confirmed**: `diag_pe_sdi_timing.py` CASE 2 — timing diff vs FFT-conv = 0.00 ns.

### Only 1 cumsum in PE SDI kernel

`zeta_pe` is a delta distribution (order 2). One integration → `Dh_pe`.
Adding `* dt` to this cumsum is WRONG — reduces output by factor ~fs.

**WRONG** (produced 8-orders-of-magnitude too small output, Batch 4 Issue 1):
```python
out[p, k] = np.float32(acc * dt)   # NEVER DO THIS
```

**CORRECT**:
```python
out[p, k] = np.float32(acc)        # bare cumsum, no dt
```

The cumsum runs INSIDE the `prange(P)` loop (per-point), after all patches:
```python
for p in prange(P):
    for m_r ...
        for m_e ...:
            _place_pe_sdi_2d(out, ...)   # scatter 16 deltas
    # cumsum (per-point, no race condition):
    acc = float64(0.0)
    for k in range(T):
        acc += float64(out[p, k])
        out[p, k] = float32(acc)
```

### Weight per (m_e, m_r) pair

```
weight = slope_e * slope_r
```

16 sign combinations: `sign_e[i] * sign_r[j]` where each is `(+1,-1,-1,+1)`.

### ppar vs mpar cumsum consistency

Batch 4 Issue 2: after reverting `* dt` with `replace_all=True`, `_compute_pe_sdi_mpar`
still had `* dt` because Unicode "× dt" in a comment caused the pattern to not match.

**Lesson**: always verify ALL instances matched after `replace_all`. Check both kernels
individually if sign/content differs in surrounding context.

### Testing PE SDI

**Cannot compare raw Dh_pe vs Dh_ref at delta level** (Batch 4 Issue 3):
linear interpolation and float32 quantization produce different error patterns.

**Correct**: compare AFTER excitation convolution (actual use case).
After FFT convolution with a narrowband pulse, interpolation artifacts wash out:
```python
rf_pe  = irfft(rfft(Dh_pe)  * rfft(exc, n=nfft))[:pe_T]
rf_ref = irfft(rfft(Dh_ref) * rfft(exc, n=nfft))[:pe_T]
peak_ratio = abs(peak_pe / peak_ref - 1) < 0.05    # ~0.003
correlation > 0.95                                  # ~0.984
```

**ppar vs mpar tolerance**: thread-local reduction in mpar changes accumulation
order → up to 1 ULP difference per sample. Use `rtol=0.01, atol=0.005×peak`.

### DC tail in Dh_pe (float32 delta buffer)

With M²×16 delta events at ~10¹⁴ magnitude, float32 residual accumulates to ~10¹⁹.
Mitigations:
1. Float64 accumulator in cumsum prevents amplification
2. Zero-mean excitation: `FFT(v)[0] ≈ 0` → DC suppressed ~6 orders of magnitude
3. Measured final RF pre-onset: 7.86×10⁻⁷ relative — negligible

### PE SDI current status (as of 2026-05-29)

| Issue | Found | Fixed |
|-------|-------|-------|
| Delta offset +1 → must be +2 | Batch 4 | Yes |
| Cumsum `* dt` → must be bare | Batch 4 | Yes |
| mpar cumsum `* dt` not reverted (replace_all miss) | Batch 4 | Yes |
| Numba parallel hang (6×1D array params) | Session 2 | Yes → packed (M,6) |
| Direction cosines: global frame for curved transducers | Session 2 | Yes |
| DC float32 tail ~0.05% | Session 2 | Accepted |
| End-to-end Reception vs Field II RF | Session 2 | Pending validation |

---

## SECTION 5 — Direction Cosines: Local Frame (Curved Transducers)

**WRONG (before fix, only correct for flat transducers):**
```python
xp = dx / dist    # = u_x  (global x — wrong if e_u ≠ [1,0,0])
yp = dy / dist    # = u_y  (global y — wrong if e_v ≠ [0,1,0])
```

**CORRECT (general):**
```python
xp = (dx*frames[m,0] + dy*frames[m,1] + dz*frames[m,2]) * inv_dist  # u · e_u
yp = (dx*frames[m,3] + dy*frames[m,4] + dz*frames[m,5]) * inv_dist  # u · e_v
```

**Error quantified** (16mm/80mm concave bowl, rim patch 5.7° tilt):
- Global `|u_x| = 0.258`, local `|u·e_u| = 0.160` → 39% error in Dt2, h_max
- Larger bowls or short-focus transducers: error becomes severe

For flat transducers: `e_u = [1,0,0]`, `e_v = [0,1,0]` → no-op.

### Why (M,6) packing (not 6 separate arrays)

6 separate 1D arrays `(eu0, eu1, eu2, ev0, ev1, ev2)` indexed by same `m`
in `prange` kernel → Numba parfors alias analysis → LLVM compilation hang
**>40 minutes**. One `(M,6)` array compiles in **~2s**.

```python
frames = np.ascontiguousarray(np.concatenate([eu, ev], axis=1))  # (M, 6)
```

---

## SECTION 6 — Emission: Global vs Per-Element Excitation Consistency

**Critical bug (Batch 3)**: global excitation path (`exc.ndim == 1`) computed
one global SDI dh → convolved. Per-element path computed E separate dh_e → convolved.
These produced different intermediate float32 cumsums → constant offset of 6144
in `dh` plateau → 150× relative differences near zero crossings after FFT conv.

**Fix**: both paths must use IDENTICAL per-element dh computation:

```python
# Global path now tiles exc and calls element loop:
exc_tiled = np.tile(exc[:, np.newaxis], (1, n_elements))   # (L, E)
pressure = _element_loop_to_pressure(points_m, exc_tiled, ...)
```

**Why**: `compute_d2h` (all patches) ≠ `sum_e compute_d2h_per_element(e)` due
to float32 non-associativity (see Section 2). Element loop uses same kernel with
same accumulation order → bit-identical with per-element path.

### Memory architecture: element loop

Peak memory O(batch_P × nfft), E-independent:
```
for e in range(E):
    dh_e = compute_dh_per_element(pts_batch, patches_of_e)  # (batch_P, 1, T)
    acc += irfft(rfft(dh_e) * fft_exc[e])                   # accumulate in freq domain
```

**NOT**: pre-allocate `(P, E, T)` dh → 142 GB for E=128, P=60501.

---

## SECTION 7 — Signal Chain Comparison: PyField vs Field II

**Field II full Born formula (Angelsen 1980, Jensen 1992):**
```
v_pe  = (ρ₀/2c₀²) × E_m ⊛ ∂³v/∂t³     ← 3 derivatives on excitation
h_pe  = h_tx(r₁→r₅) ⊛ h_rx(r₅→r₁)     ← no derivatives on SIR
```

**PyField (redistribute all 3 derivatives onto SIR side):**
```
v_pe' = (ρ₀/2c₀²) × E_m × v            ← no derivatives
Dh_pe = dh_tx/dt ⊛ d²h_rx/dt²          ← 3 derivatives on SIR
```

Frequency domain: **both** give `RF = (ρ₀/2c₀²) × (−jω³) × V × E_m × H_tx × H_rx`.
Exactly equivalent — redistribution via commutativity of convolution/differentiation.

**Where the 3 derivatives come from:**
- 1st `d/dt`: SIR radiation equation `p ∝ ∂v_n/∂t ⊛ h` (Stepanishen 1971)
- 2nd+3rd `d²/dt²`: Born scattering source terms from Δρ and Δc in wave equation

**f_m scattering function (Born approximation):**
```
f_m(r) = Δρ(r)/ρ₀ − 2Δc(r)/c₀
```

Setting `tx.excitation = ir` AND `tx.impulse_response = ir` is physically valid.
Both Field II and PyField apply the same total signal content.

---

## SECTION 8 — Pre-Existing Bugs (Do Not Fix, Do Not Call)

### `h_sir.__call__` is broken

```python
self.x, self.y, self.z, points = check_field_points(field_points_mm)  # returns 1 value
```

Unpacking 4 values from 1-value return → `ValueError` always.
**Do not call `h_sir.__call__`**. Use `compute_derivative` or `Emission`/`Reception`.

### PyField raw-array path passes mm as metres

Non-dict input to `PyField.__call__` skips the `× 1e-3` conversion →
distance calculations 1000× too large → garbage SIR.
**Always use dict input for PyField.** For Emission/Reception: explicit `× 1e-3`.

### `from_sir_to_pressure`: attenuation skipped without excitation

When `excitation=None`, function returns `h_sir` directly — no IRFFT step.
`alpha0` parameter silently ignored. Callers requiring attenuation must provide excitation.

---

## SECTION 9 — Attenuation Wiring

```python
H_att = causal_attenuation_tf(freqs_hz, distances_m, alpha0, freq_power, f0_hz)
# H_att shape: (..., N_freq)  — leading dims = distances shape
# FFT batch shape: (N_freq, batch_P)
fft_pressure *= H_att.T    # transpose before multiply: (N_freq, batch_P)
```

`H_att` returns complex128. Multiplying float64 FFT batch by complex128 promotes
to complex128 → `irfft` returns real output correctly.

Causal power-law model (y ≠ 1):
```
H_att = exp(-alpha0*|ω|^y*d) × exp(-j*alpha0*|ω|^y*tan(y*π/2)*d)
```
`tan(y*π/2)` diverges as y→1. Never test y=1 continuity by approaching from
y=1.001 — pole on the general-formula side. Test `y=1` branch independently.

---

## SECTION 10 — Numba Cache and Compilation

**Clear cache after any kernel change:**
```powershell
Get-ChildItem -Path "src\pyfield\h_sir\__pycache__" -Filter "*.nb?" | Remove-Item -Force
```

Symptom of stale cache: fix "has no effect". First run after clear: slow (compile).
Subsequent runs: instant.

All parallel kernels: `@njit(parallel=True, fastmath=True, cache=True)`.

---

## SECTION 11 — PSF Measurement

**Wrong**: time-slice at on-axis peak row → underestimates lateral FWHM.
Off-axis scatterers peak at different times; a single time-slice misses them.

**Correct**: `max(abs(rf), axis=time)` per lateral position → true envelope peak
per lateral point → measure FWHM from that.

Example: 30 mm depth, 80 mm focus, 16 mm bowl —
time-slice FWHM: ~0.8–5.2 mm, max-over-time FWHM: 7.2 mm (correct).

---

## SECTION 12 — Miscellaneous Pitfalls

### `n_elements` must come from `delays.shape[0]`

`tx.n_elements` not universal across all `TransducerBase` subclasses.
`delays.shape[0]` always correct regardless of transducer type.

### Tail zeroing for pulsed mode

After computing h_sir or dh with SDI, the float32 DC tail persists in samples
after the last SIR event:
```python
idx_end = _compute_active_window(points_m, t0, T, dt)
h[idx_end:, :] = 0.0   # zero tail before FFT or return
```
Without this, "static cone" artifact appears in pulsed-mode output.

### H_att broadcasting convention

`causal_attenuation_tf` returns `(batch, N_freq)`. FFT arrays are `(N_freq, batch)`.
Always `.T` before multiply:
```python
fft_out = fft_h * fft_exc * H_att.T
```

### Mat file DT loading (Field II comparison scripts)

Field II Matlab script may downsample RF envelope (e.g., `RF(1:5:end)` = every
5th row). Python loading with `DT = 1/fs` (10 ns) instead of `DT = 5/fs` (50 ns)
produces 5× apparent timing error. Check step size when loading `.mat` files.

---

## Quick Reference: Integration Rules

| Operation | Offset | dt factor | Notes |
|-----------|--------|-----------|-------|
| Single SDI: d²h placement | `+1.0` | — | step lands at first bin after t_corner |
| PE SDI: Dh_pe placement | `+2.0` | — | matches Ne+Nr of FFT conv reference |
| d²h → dh (1st cumsum) | — | **NO** | delta distribution: bare cumsum |
| dh → h (2nd cumsum) | — | **YES** | continuous function: Euler step |
| PE SDI: zeta → Dh_pe cumsum | — | **NO** | delta distribution: bare cumsum |
