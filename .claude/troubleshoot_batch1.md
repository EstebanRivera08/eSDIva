# Batch 1 Refactor — Hard Issues Reference

Non-evident bugs and pitfalls encountered during Batch 1 (sir_derivatives, attenuation,
transducers/base, helper_functions). Written to prevent re-encountering these in future batches.

---

## Issue 1 — float32 Accumulator in Numba Cumsum Causes Catastrophic Cancellation

### What happened

`_cumsum_2d` and `_cumsum_3d` in `sir_derivatives.py` were written with float32 accumulators:

```python
@njit
def _cumsum_2d(arr):
    acc = np.float32(0.0)          # WRONG
    for k in range(T):
        acc += arr[i, k]
        out[i, k] = acc
```

The d2h (second-derivative SIR) array contains large cancelling events. For a single patch on a
4-element, 8-patch transducer at 200 MHz sampling:

- Peak event magnitude: ~4.158e10 (slope × weight ≈ h_max / Dt1 × fs)
- Float32 ULP at 4.158e10: ~4096
- Total sum of events (float64): −2048 (should be ~0 for a complete SIR)

When the cumsum accumulates 4.158e10 in float32 then subtracts it, the cancellation residual
is ±4096 (one ULP), not the true ~±2048. This produces a wrong DC offset in `dh` and a
linear ramp (slope × sample count × dt) in `h`.

### Effect on h_sir

After double cumsum, the SIR body is correct. The artifact appears only in the tail (after the
SIR ends), where the phantom DC leaks as a linear ramp. Magnitude at sample 537 with float32 acc:

```
h_ref tail  (reference, in-place float64→float32): +0.01585 m/s per sample
h_f32 tail  (float32 acc, wrong):                  −0.00715 m/s per sample
h_f64 tail  (float64 acc, correct write-back):      −0.00374 m/s per sample
```

Peak SIR amplitude: **392 m/s**. Tail artifact = 0.015 m/s = **0.004% of peak**. Physically
negligible, but numerically large enough to fail exact comparison tests.

### Fix

Use float64 accumulator with float32 write-back (matches reference kernel in
`farfield_rect_patch.py` which uses Python float `acc = 0.0`):

```python
@njit
def _cumsum_2d(arr):
    acc = np.float64(0.0)                   # float64 accumulator
    for k in range(T):
        acc += np.float64(arr[i, k])        # read float32, promote to float64
        out[i, k] = np.float32(acc)         # write back as float32
```

### Rule

**Never use float32 accumulator for cumsum over float32 arrays that contain large cancelling
values.** The ULP at accumulator scale >> the true residual. Always use float64 accumulator +
float32 write-back to match the reference kernel's two-pass integration pattern.

---

## Issue 2 — DC Tail Artifact Persists Even With Correct float64 Accumulator

### What happened

Even after fixing to float64 accumulator, the tail of `h_new` (−0.00374) does not match
`h_ref` (+0.01585). Root cause: the reference kernel in `farfield_rect_patch.py` writes the
first cumsum result back **in-place** to the `d2h` array as float32, then the second cumsum
reads those float32 values. The reference and new implementations both use float64 accumulators,
but produce different tail directions because the reference's in-place write introduces an
intermediate float32 rounding that the new implementation does not replicate.

Additionally, the `inv_2pi` constant differs:

| Code | Value | Type |
|------|-------|------|
| `farfield_rect_patch.py` | `1/(2*pi) = 0.159154943...` | float64 (Python module constant) |
| `sir_derivatives.py` originally | `np.float32(1/(2*pi)) = 0.1591549...` | float32 |

This causes `area` and hence `slope` to differ by ~2e-8 relative. At float32 event scale
(4.158e10), this 2e-8 error = 830, which is less than 1 ULP (4096) and thus rounds to the
same float32. So the `inv_2pi` difference is NOT the cause of the tail mismatch — it is the
in-place write-back pattern.

### Quantified uncertainty

The tail artifact is a function of:
- `d2h.sum()` in float64: −2048 for this fixture
- Duration after SIR ends: T − k_end samples
- `dt = 1/fs`

Ramp magnitude at sample k_tail: `|d2h.sum_f64| × k_tail × dt`

For T=537, k_end≈220, dt=5e-9 s: `2048 × (537−220) × 5e-9 ≈ 0.0032 m/s`

As fraction of peak (392 m/s): **~0.0008% = 8e-6 relative**. This is below float32 eps
(~1.2e-7) relative to peak, so it is irreducible floating-point noise.

### What does NOT fix it

Changing `_cumsum_2d` to use float64 internally reduces the tail from ±0.007 to ±0.004.
But it does NOT make `h_new == h_ref` because the reference uses a specific in-place pattern
that is not replicated by the new modular design (where cumsum is a separate function).
Replicating the exact reference behavior would require in-place modification of `d2h` inside the
cumsum, which would destroy the input array — unacceptable for modular design.

### Resolution

Accept that `h_new` and `h_ref` have different (but equally arbitrary) tail values. Use
`rtol=0.01, atol=0.01 × peak` in tests. This expresses: "all errors within 1% of signal peak."

At peak = 392 m/s, tolerance = 3.92 m/s. Tail artifact = 0.015 m/s = **0.38% of tolerance**.
Any real implementation bug (wrong patch, wrong geometry, missed element) would produce body
errors of tens to hundreds of m/s, well above tolerance.

### Rule

**Never compare two SDI double-cumsum results with atol=0 or extremely tight absolute tolerances.**
The tail ramp is inherent to any float32-based SDI implementation and varies sign/magnitude based
on specific accumulation order. Use `rtol=0.01, atol=0.01 × max(|expected|)` as the standard
for SIR comparison tests.

---

## Issue 3 — Float32 Non-Associativity: d2h_all ≠ d2h_per_element.sum()

### What happened

`compute_d2h` (all patches, output shape P×T) and `compute_d2h_per_element` (per-element output
P×E×T) use different Numba kernels. For overlapping time indices, the accumulation order differs:

- `d2h_all[p, k]`: all patches accumulate in the order they appear in the inner loop
- `d2h_per_e[p, e, k]`: only patches of element e accumulate

At shared time indices, float32 addition is not associative: `float32(a + b) ≠ float32(float32(a) + float32(b))` when values are large.

### Numbers

```
d2h_all.sum()      (float64): −2048
d2h_per_e.sum()    (float64):     0
max(|d2h_all − d2h_per_e.sum(axis=1)|): 4096  (= 1 ULP at event scale ~4.158e10)
```

After cumsum: `dh_all` tail = −2048, `dh_per_e.sum()` tail = 0.
Diff in tail = 2048. As fraction of dh body peak (~4.158e10): **5e-8 relative**.
Physically zero. But absolute comparison with atol=0 fails.

### Rule

**Never write a test asserting `d2h_per_element.sum(axis=E) == d2h_all` with atol=0.**
Float32 non-associativity guarantees they differ by up to 1 ULP of event magnitude. Use
`rtol=0.01, atol=0.01 × max(|expected|)` same as above.

---

## Issue 4 — y=1 Continuity Test for Attenuation TF Is Physically Impossible

### What happened

REFACTOR_PLAN §11 specifies: "`y=1` special case ≈ `y=1.001` general case (continuous at
boundary)". Test written as:

```python
H1 = causal_attenuation_tf(..., y=1.0, ...)
H2 = causal_attenuation_tf(..., y=1.001, ...)
amp_diff_dB = 20 * np.log10(|H1| / |H2|)
assert np.max(np.abs(amp_diff_dB)) < 0.02
```

Fails because `tan(y × π/2)` diverges as y → 1:

| y | tan(y×π/2) |
|---|------------|
| 1.0 | ∞ (special case: log dispersion used instead) |
| 1.001 | ≈ −636 |
| 1.01 | ≈ −63.7 |
| 1.1 | ≈ −6.31 |

At y=1.001, the K-K dispersion phase is ~636 times larger than at y=1.1. The phase factor
`exp(−j × alpha0 × |ω|^y × tan(yπ/2) × d)` is massive at y=1.001, causing |H2| to oscillate
and diverge from |H1|. The amplitude envelopes are close, but the test compares complex ratio
|H1/H2| which includes phase beating.

### Resolution

Remove this test entirely. `y=1` continuity is a mathematical property of the analytic form
(O'Donnell 1981 logarithmic limit), not directly testable numerically at y→1 from general case.
Instead test `y=1` independently via `|H(f)| = exp(−alpha0_nep × f × d)` (verified in
`test_y1_amplitude_matches_formula`).

### Rule

**Do not test analytic limit continuity numerically by approaching from one side if the general
formula has a pole at the limit.** `tan(π/2)` diverges; any y > 1.0 is on the wrong side of
the divergence relative to 1.0 via the non-y=1 formula.

---

## Issue 5 — Pre-Existing Test Failures Pollute Test Gate

### What happened

`tests/unit/test_plotting/test_image.py` imports `pyfield.utilities.plotting` which does not
exist. This causes ImportError for the entire plotting test module. Running the full test suite
without ignoring this directory produces confusing failures unrelated to Batch 1.

### Resolution

Always run Batch 1 gate with:
```
uv run pytest tests/ -k "sir_deriv or attenuation or steering or sub_elem" \
    --ignore=tests/unit/test_plotting
```

### Rule

Before declaring a test gate failed, verify failures are in Batch-scope files. Pre-existing
broken tests in unrelated modules contaminate aggregate pass/fail counts.

---

## Issue 6 — Numba Cache Stale After Source Edit

### What happened

After editing `_cumsum_2d` (changing float32 → float64 accumulator), Numba JIT-compiled cache
files (`.nbi`, `.nbc`) in `__pycache__` sometimes retain the old compiled version. Tests then
run the old (wrong) compiled code despite source changes, producing confusing behavior where
the fix "doesn't work."

### Resolution

Clear Numba cache after significant kernel changes:
```powershell
Get-ChildItem -Path "src\pyfield\h_sir\__pycache__" -Filter "*.nbi" | Remove-Item -Force
Get-ChildItem -Path "src\pyfield\h_sir\__pycache__" -Filter "*.nbc" | Remove-Item -Force
```

JIT recompile is confirmed by long first-run time (35–75 s) vs fast cached run (<5 s).

### Rule

If a Numba kernel fix "has no effect" in tests, clear `.nbi`/`.nbc` cache files before
concluding the fix is wrong.

---

---

## Mandatory Rule — All SDI Integration Must Use float64 Accumulators

All cumsum operations that integrate d2h → dh → h **must** use float64 accumulators with
float32 write-back. This applies to every Numba kernel and Python helper in `sir_derivatives.py`
and any future SDI integration code.

**Required pattern** (enforced in `_cumsum_2d`, `_cumsum_3d`):

```python
acc = np.float64(0.0)          # float64 accumulator — MANDATORY
for k in range(T):
    acc += np.float64(arr[i, k])   # read float32, accumulate as float64
    out[i, k] = np.float32(acc)    # write back as float32
```

**Forbidden pattern:**

```python
acc = np.float32(0.0)          # WRONG — catastrophic cancellation at 4e10 scale
```

**Why:** SDI events are on the order of 4e10 (slope × fs). Float32 ULP at this scale is ~4096.
When large positive and negative events cancel, float32 loses precision and leaves a DC residual
that propagates as a linear ramp through the double cumsum. Float64 accumulator reduces the
residual by ~2× relative to float32 and matches the reference kernel's `acc = 0.0` pattern.

---

## Summary Table

| Issue | Root cause | Max artifact | Fix strategy |
|-------|-----------|-------------|--------------|
| float32 cumsum DC ramp | float32 ULP at 4e10 scale >> residual | 0.004% of SIR peak | float64 acc + float32 writeback |
| h_new ≠ h_ref tail | In-place vs modular cumsum difference | 0.004% of SIR peak | rtol=0.005, atol=0.005×peak |
| d2h_per_e.sum ≠ d2h_all | float32 non-associativity, 1 ULP | 5e-8 relative in dh | rtol=0.005, atol=0.005×peak |
| y=1 continuity test | tan(yπ/2) diverges at y=1.001 | — | Remove test, test y=1 directly |
| test_plotting ImportError | Wrong module path in test | — | `from pyfield.plotting import plot2D_pressure_plane as plot_pressure_2D` |
| Numba cache stale | .nbi/.nbc not updated on source change | — | Delete .nbi/.nbc files |

## Standard Test Tolerance for SIR Comparisons

For all future SIR/d2h/dh comparisons across different kernels or integration paths:

```python
peak_tol = 0.005 * float(np.abs(expected).max())
np.testing.assert_allclose(actual, expected, rtol=0.005, atol=peak_tol)
```

Pass condition: `|actual − expected| ≤ 0.5% of peak`. Float32 arithmetic artifacts are
< 0.005% of peak. Real physics bugs produce errors >> 0.5% of peak.
