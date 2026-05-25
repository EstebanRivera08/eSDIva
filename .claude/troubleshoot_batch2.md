# Batch 2 Refactor — Hard Issues Reference

Non-evident bugs and pitfalls encountered during Batch 2 (h_sir.compute_derivative,
h_sir/__init__ exports, sir_to_pressure attenuation wiring). Written to prevent
re-encountering these in future batches.

---

## Issue 1 — `h_sir.__call__` Has Pre-Existing Bug (4-Value Unpack from 1-Value Return)

### What happened

`h_sir.__call__` imports `check_valid_field_points as check_field_points` then does:

```python
self.x, self.y, self.z, points = check_field_points(field_points_mm)
```

But `check_valid_field_points` returns **one value** (the validated dict or array).
Unpacking 4 values from it raises `ValueError: not enough values to unpack`.

This means `h_sir.__call__` is currently broken for all inputs. It was NOT fixed as
part of Batch 2 (out of scope — `PyField` is the primary API).

### Resolution for `compute_derivative`

Used `create_3D_spatial_grid_from_points` for dict inputs (returns metres), and direct
`× 1e-3` conversion for array inputs:

```python
if isinstance(field_points_mm, dict):
    self.x, self.y, self.z, points = create_3D_spatial_grid_from_points(field_points_mm)
else:
    self.x = self.y = self.z = None
    pts = np.asarray(field_points_mm, dtype=np.float32)
    points = pts * np.float32(1e-3)  # mm → m
```

### Rule

**Do NOT call `h_sir.__call__` anywhere in tests or in Batch 3/4 code.** Use `compute_derivative`
or `PyField` directly. If fixing `__call__` becomes necessary, replace
`check_valid_field_points as check_field_points` with `create_3D_spatial_grid_from_points`.

---

## Issue 2 — PyField Raw-Array Path Passes mm to `compute_time_grid` (Latent Bug)

### What happened

In `PyField.__call__`, when `field_points_mm` is a raw ndarray (not dict):

```python
is_structured = isinstance(field_points_mm, dict)
if is_structured or create_meshgrid:
    x, y, z, field_points_mm = create_3D_spatial_grid_from_points(field_points_mm)
else:
    x, y, z = None, None, None
    # field_points_mm stays in mm!
```

Then `compute_sir(field_points_mm, ...)` is called with mm values.
`compute_time_grid` and `compute_h_sir` expect metres.

Result: distance calculations 1000× too large, SIR is garbage.

### Status

Pre-existing bug. Not fixed in Batch 2 (only dict inputs work correctly in practice).

### Rule

**For Batch 3/4 Emission/Reception classes: always go through `create_3D_spatial_grid_from_points`
for dict inputs and explicit `× 1e-3` conversion for array inputs.** Do not copy PyField's
raw-array path.

---

## Issue 3 — Attenuation Only Wired in Excitation Path of `from_sir_to_pressure`

### What happened

`from_sir_to_pressure` has two execution paths:
1. `excitation is None` → `Pressure_flat = h_sir` (direct return)
2. `excitation is not None` → FFT convolution path

Attenuation (H_att multiply before irfft) only makes sense in path 2 because path 1 has no
IRFFT. If `alpha0` is provided with `excitation=None`, attenuation is silently ignored.

### Implication

Tests for "alpha0=None bit-identical" work fine. Tests for "attenuation decreases amplitude"
must use excitation. If future code relies on attenuation without excitation (e.g., raw SIR
post-processing), a separate FFT-multiply-IFFT step must be added.

### Rule

**`alpha0` parameter in `from_sir_to_pressure` is a no-op when `excitation=None`.**
Document this at any call site that might pass `alpha0` without excitation.

---

## Issue 4 — H_att.T Required for Broadcasting Against H_batch

### What happened

`causal_attenuation_tf(freqs_hz, distances_m[start:end], ...)` returns shape
`(cols, N_freq)` (leading dims = distances shape, trailing dim = freq).

`H_batch = rfft(h_pad)` has shape `(N_freq, cols)`.

Direct multiply `H_batch * H_att` fails (shape mismatch).
Correct form: `H_batch * fft_dExcitation * H_att.T` where `H_att.T` has shape `(N_freq, cols)`.

```python
H_att = causal_attenuation_tf(freqs_att, distances_arr[start:end], alpha0, freq_power, f0_hz)
# H_att: (cols, N_freq) → complex128
fft_Pressure = fft_Pressure * H_att.T   # (N_freq, cols) * (N_freq, cols)
```

Note: `H_att` is `complex128`. Multiplying `float64 H_batch` by `complex128 H_att.T` promotes
`fft_Pressure` to `complex128`. `irfft` of complex input returns real — correct behavior.

### Rule

**Always transpose H_att before multiplying into the FFT batch.** Convention:
`causal_attenuation_tf` returns `(..., N_freq)`, but FFT batch arrays are `(N_freq, batch)`.

---

## Issue 5 — Local Variable `Pressure_flat` Shadows Outer Variable in Closure

### What happened

Original `_process_batch` closure had:

```python
def _process_batch(start):
    ...
    Pressure_flat = np.abs(outputfft[:T, :])   # local, shadows outer
    return start, end, Pressure_flat
```

The outer function also has `Pressure_flat = np.zeros(...)`. Python assignment inside
a function creates a local variable, not an assignment to the closure-captured name.
The code worked (returned the local), but was confusing and a future mutation bug risk.

### Resolution

Renamed local to `Pressure_batch`:

```python
def _process_batch(start):
    ...
    Pressure_batch = np.abs(outputfft[:T, :])
    return start, end, Pressure_batch
```

### Rule

**In batch-processing closures: never name local results the same as outer accumulator arrays.**
The outer array is written via the loop (`Pressure_flat[:, start:end] = out`); the local return
value should have a distinct name.

---

## Issue 6 — `n_elements` Must Come from `delays.shape[0]`, Not `tx.n_elements`

### What happened

`tx.n_elements` attribute exists on `LinearArrayTransducer` but may not exist on all
`TransducerBase` subclasses (e.g., single-element, circular). Using it directly would
break on transducer types that don't define it.

`self.delays` is always `transducer.delays`, an ndarray of shape `(n_elements,)`.
`int(self.delays.shape[0])` is always correct regardless of transducer type.

### Rule

**Always derive `n_elements` from `delays.shape[0]` (or equivalently `apodization.shape[0]`).**
Do not use `tx.n_elements` directly in Emission/Reception unless `TransducerBase` guarantees it.

---

## Issue 7 — `do_attenuation` Closure Capture in `ThreadPoolExecutor`

### What happened

`_process_batch` is a closure that captures `do_attenuation`, `distances_arr`, `freqs_att`,
`alpha0`, `freq_power`, `f0_hz` from the enclosing scope. These are set before the closure is
defined, so they are available at call time.

`ThreadPoolExecutor` calls `_process_batch` from worker threads. All captured variables are
read-only (no mutation). NumPy slicing `distances_arr[start:end]` creates a new array per call.
`causal_attenuation_tf` creates new arrays internally. No race conditions.

### Potential issue for Batch 3/4

If `Emission.__call__` uses a similar pattern with mutable state (e.g., `self.alpha0` could
change between batch calls if someone modifies it externally), closures over `self` attributes
risk capturing stale values mid-computation. Safe for now since batches run synchronously.

### Rule

**Closures over `self` attributes inside `ThreadPoolExecutor` are safe only if the instance
is not mutated during the parallel execution.** For Emission/Reception, ensure `alpha0`, `distances_m`
etc. are captured into local variables before entering the executor.

---

## Issue 8 — `sir_derivatives.py` Needed `ruff format` After Batch 1

### What happened

`sir_derivatives.py` was created in Batch 1 but not passed through `ruff format`. Batch 2's
`just pre-commit` run triggered ruff format and reformatted it. Pre-existing failures in
`utilities/__init__.py` (F401) and `TorchField*` + `plotting3D.py` (ty errors) are unrelated
to Batch 2.

### Rule

**Run `uv run ruff format src/` and `uv run ruff check --fix src/` on any new file before
committing.** The pre-existing failures in `utilities/__init__.py` and `cache/` must NOT be
fixed as part of any Batch commit (different scope).

---

## Summary Table

| Issue | Scope | Root cause | Fix strategy |
|-------|-------|-----------|--------------|
| `h_sir.__call__` broken | Pre-existing | Wrong import alias → 4-value unpack | Use `compute_derivative` instead |
| PyField raw-array → mm passed as m | Pre-existing | Missing `× 1e-3` in non-dict path | Always use `create_3D_spatial_grid_from_points` |
| Attenuation skipped with no excitation | Design | No IRFFT in excitation=None path | Document; callers must provide excitation |
| H_att shape mismatch | Implementation | `(cols, N_freq)` vs `(N_freq, cols)` | `H_att.T` before multiply |
| Local var shadows outer accumulator | Style/risk | Same name in closure and outer | Rename local to `Pressure_batch` |
| `tx.n_elements` not universal | Robustness | Attribute not in TransducerBase | Use `delays.shape[0]` |
| Closure capture thread safety | Design | Shared read in ThreadPoolExecutor | OK for read-only; localise for mutable state |
| sir_derivatives.py unformatted | Process | ruff format not run post-Batch 1 | Format new files before commit |

---

## Attenuation Wiring Checklist for Batch 3 (Emission)

When wiring attenuation into `Emission.__call__`:

1. Compute `distances_m` via `compute_attenuation_distances` before FFT loop.
2. Pass `alpha0, freq_power, f0_hz, distances_m` to `from_sir_to_pressure` (now supports them).
3. `distances_m` shape must be `(P,)` for `from_sir_to_pressure`. If per-element attenuation
   needed, call `causal_attenuation_tf` directly with `(P, E)` distances.
4. `alpha0=None` path in both functions is bit-identical → no performance cost for un-attenuated runs.
5. For CW (monochromatic) path: use `from_sir_to_monochromatic_pressure` which also accepts `alpha0`.
6. Guard: `do_attenuation = alpha0 is not None and distances_m is not None` — if user provides
   `alpha0` but not `distances_m`, attenuation is silently skipped. Consider warning.
