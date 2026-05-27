# Refactor Plan: Emission / Reception / Attenuation

Full spec lives in `PROMPT2.md`. Read the referenced sections before implementing each batch.
Physics background: `.claude/rules/physics-context.md`, `.claude/rules/attenuation.md`.

Execute in order — each batch depends on the previous being tested and passing.

---

## Dependency graph

```
Batch 1 (foundation — no cross-deps, files independent)
  └─▶ Batch 2 (wiring — needs sir_derivatives + attenuation from B1)
        └─▶ Batch 3 (Emission — needs all of B1+B2)
              └─▶ Batch 4 (Reception — needs all previous)
```

---

## Batch 1 — Foundation ✓ COMPLETE

**All files independent. Can be implemented in any order or in parallel.**

### Files to create / modify

| Action | File | Spec section |
|--------|------|-------------|
| MODIFY | `src/pyfield/utilities/helper_functions.py` | §1.5 |
| MODIFY | `src/pyfield/transducers/base.py` | §4.2, §7, §7.2 |
| CREATE | `src/pyfield/h_sir/sir_derivatives.py` | §1.2, §1.3, §1.6 |
| CREATE | `src/pyfield/psimulation/attenuation.py` | §2 |

### What to do per file

**`helper_functions.py`** (§1.5):
- `compute_sub_elem_attributes`: add `sub_el_idx_arr: int32[M]` to return tuple (7th element becomes 8th, add new last).
- Already iterates `sub_el_idx` — collect into array, return it.
- Update all call sites in same file and `h_sir/h_sir.py`, `psimulation/PyField.py`.

**`transducers/base.py`** (§4.2, §7, §7.2):
- Add `_impulse_response` and `_excitation` attributes to `TransducerBase.__init__`.
- Add `impulse_response` and `excitation` properties with getter/setter (spec in §4.2).
- Extend `compute_delays` signature: add `angle_steering_deg=None` kwarg (spec + formula in §7.2).
- Mutual exclusivity: `focus_mm` and `angle_steering_deg` both non-None → `ValueError`.
- Mono-element: still warns and returns zeros regardless of mode.

**`sir_derivatives.py`** (§1.2, §1.3, §1.6) — new file:
- Implement all 4 Numba kernels: `compute_d2h`, `compute_dh`, `compute_d2h_per_element`, `compute_dh_per_element`.
- All `@njit(parallel=True, fastmath=True)`, `prange` over P by default.
- Per-element variants: extra args `sub_el_idx: int32[M]`, `n_elements: int`. Accumulate into `out[p, sub_el_idx[m], k]`.
- Integration helpers: `integrate_d2h_to_dh`, `integrate_dh_to_h` — work on any `(..., T)` shape.
- Reference kernel: `compute_h_sir_patch_parallel` — `prange` over M.
- Patches-parallel variants (`_mpar` suffix) using thread-local reduction pattern (§1.6).
- Python wrappers accept `parallel_axis="points"|"patches"`, dispatch to correct Numba kernel.
- Memory: Python wrappers accept `batch_size_points` to chunk P at Python level.
- Reuse `compute_rectangle_SIR_params` logic from `farfield_rect_patch.py` — do not import from it, duplicate the helper or factor into shared internal module.

**`attenuation.py`** (§2) — new file:
- `causal_attenuation_tf(freqs_hz, distances_m, alpha0_dB, y, f0_hz)` — accepts any leading shape on `distances_m`, returns `(..., N_freq)`. Both absorption and K-K dispersion terms. Special branch for `y == 1`.
- `convert_alpha0_to_nepers(alpha0_dB, y)`.
- `compute_attenuation_distances(field_points_m, transducer_center_m, patch_centers_m, mode)` — returns `(P,)` or `(P, M)`.
- `reduce_patch_distances_to_element(distances_pm, sub_el_idx, n_elements, reduce="mean")` — supports `"mean"`, `"min"`, `"max"`. Returns `(P, E)`.
- `compute_reception_distances(scatterer_positions_m, tx_center_m, rx_element_centers_m)` — two-path model, returns `(P, E_rx)`.

### Test gate (must pass before Batch 2)

```
uv run pytest tests/ -k "sir_deriv or attenuation or steering or sub_elem"
```

Key assertions (see §11 for full list):
- `compute_d2h(pts, ...) + 2 cumsums ≈ compute_h_sir(pts, ...)` — `rtol=1e-4`, float32
- `sum over E of compute_d2h_per_element == compute_d2h` — grouping correctness
- `angle_steering_deg=0` flat array → `delays == 0`
- `angle_steering_deg=30` flat linear → delays monotone proportional to x-position
- `angle_steering_deg=0` depth-staggered custom array → delays proportional to z
- `alpha0=None` path in `causal_attenuation_tf` → identity (H=1)
- `y=1` special case output ≈ `y=1.001` general case (continuous at boundary)

---

## Batch 2 — Wiring ✓ COMPLETE

**Needs Batch 1 complete and tested.**

### Files to modify

| Action | File | Spec section |
|--------|------|-------------|
| MODIFY | `src/pyfield/h_sir/h_sir.py` | §1.4 |
| MODIFY | `src/pyfield/h_sir/__init__.py` | §5 (h_sir exports) |
| MODIFY | `src/pyfield/psimulation/sir_to_pressure.py` | §3 (signal chain), §2.2 (rules) |

### What to do per file

**`h_sir/h_sir.py`** (§1.4):
- Add `compute_derivative(self, field_points_mm, *, derivative="h", per_element=False, parallel_axis="points")` method.
- Dispatches to `compute_h_sir` (existing) for `derivative="h"`, else to new kernels from `sir_derivatives.py`.
- Returns `(t0, result)` where result shape is `(T, P)` or `(T, P, E)`.
- Update call site for `compute_sub_elem_attributes` to unpack new `sub_el_idx_arr`.

**`h_sir/__init__.py`**:
- Export new kernels: `compute_d2h`, `compute_dh`, `compute_d2h_per_element`, `compute_dh_per_element`, `integrate_d2h_to_dh`, `integrate_dh_to_h`.

**`sir_to_pressure.py`** (§3.3 signal chain):
- Add `alpha0`, `freq_power`, `f0_hz`, `distances_m` params to existing functions — all default `None` = no attenuation.
- When `alpha0` is not None: call `causal_attenuation_tf`, multiply in frequency domain before `irfft`.
- `alpha0=None` must produce bit-identical output to current (no extra ops).

### Test gate (must pass before Batch 3)

```
uv run pytest tests/ -k "h_sir or sir_to_pressure"
```

Key assertions:
- `compute_derivative(derivative="h")` output matches existing `compute_h_sir` output — same transducer, same points
- `sir_to_pressure` with `alpha0=None` → identical output to unmodified function
- `sir_to_pressure` with `alpha0=0.5` → amplitude decreases monotonically with distance

---

## Batch 3 — Emission ✓ COMPLETE

**Needs Batch 1 + Batch 2 complete and tested.**

### Files to create / modify

| Action | File | Spec section |
|--------|------|-------------|
| CREATE | `src/pyfield/psimulation/emission.py` | §3 |
| REWRITE | `src/pyfield/psimulation/PyField.py` | §3.5 |
| MODIFY | `src/pyfield/psimulation/__init__.py` | §5 (exports) |
| MODIFY | `src/pyfield/__init__.py` | §5 (top-level exports) |

### What to do per file

**`emission.py`** (§3) — new file:
- `Emission` class: constructor params `(transducer, *, c, rho, fs, alpha0, freq_power, excitation, monochromatic, verbose)`.
- `.set(name, value)` method with `_SETTABLE` dict validation (§6).
- `__call__(field_points_mm, *, method="auto")` — full excitation dispatch logic (§3.3):
  - `monochromatic=True` → CW path
  - `excitation=None` → raw SIR (pulsed default)
  - `excitation.ndim==1` → global excitation convolution
  - `excitation.ndim==2` → per-element path using `compute_dh_per_element`
- Private `_compute_sir` and `_compute_sir_derivative` methods (§3.4).
- Attenuation: if `alpha0` not None, apply via `causal_attenuation_tf` in freq domain.
- TX `impulse_response` if not None: `V_eff(f) = FFT(excitation) * FFT(ir_tx)`.

**`PyField.py`** (§3.5) — full rewrite to thin wrapper:
- Subclass of `Emission` with `monochromatic=True` default.
- `DeprecationWarning` in `__init__`.
- No other logic.

**`psimulation/__init__.py`** (§5):
- Export `Emission`, `Reception` (placeholder until B4), `PyField`, `causal_attenuation_tf`, `compute_reception_distances`, `reduce_patch_distances_to_element`.

### Test gate (must pass before Batch 4)

```
uv run pytest tests/ -k "emission or pyfield"
```

Key assertions (§11):
- Old `PyField(tx)(field_points)` → same pressure + `DeprecationWarning`
- `Emission(tx, monochromatic=True)` == old `PyField(tx)` output
- `Emission(tx, excitation=pulse)` == old `PyField(tx, excitation=pulse)` output
- Uniform per-element excitation (same pulse repeated E times) == global excitation — same output
- `alpha0=None` path identical to no-attenuation baseline
- `ir_tx=None` == `ir_tx=delta` — identity IR behavior

### Actual Implementation Notes (divergences from spec)

**Additional constructor parameters not in original spec:**
- `transfer_function=None` — callable `TF(freq) -> array` applied multiplicatively
  in freq domain alongside excitation (modes 3 and 4).
- `fast_attenuation=False` — controls per-element loop trigger for attenuation.
  `False` (default): per-element loop, element-center distances (accurate).
  `True`: TX-center distance, no E-loop (fast approximation).

**Per-element loop trigger** (replaces spec's `excitation.ndim==2` only):
```python
use_per_element = (self.alpha0 is not None and not self.fast_attenuation) or per_elem_exc
```
This means modes 2 and 3 also enter per-element loop when `alpha0` is set
and `fast_attenuation=False`.

**SIR kernel used**: `compute_h_sir` from `farfield_rect_patch.py` for all paths —
**not** `compute_dh_per_element` from `sir_derivatives.py`.  Per-element loop
computes `h_sir` separately per element by passing element-filtered patch arrays
directly to `compute_h_sir`.

**FFT backend**: `scipy.fft.rfft/irfft` with `workers=-1` (multithreaded),
**not** `numpy.fft`. All FFT done on float32 → complex64 throughout
(half memory vs float64; no upcast/downcast pattern).

**Memory strategy (key bug fix)**: Per-element loop pre-allocates a single
`h_pad_buf = np.zeros((batch_P, nfft), float32)` outside all loops.
Each element writes its h_sir into `h_pad[:, :T]`; tail `[:, T:]` stays zero.
`scipy.fft.rfft` on an already-nfft-length input creates no internal
zero-padding buffer — eliminates `E × n_batches × ~140 MB` allocations
that caused OS swap with explicit per-call zero-padding.

**Loop structure**: P-outer, E-inner. `acc_H (cols, N_freq)` accumulates all E
elements in freq domain per P-batch. One `irfft + abs` per P-batch (not per element).
Total irfft calls = n_batches (not E×n_batches), preserving inter-element interference.

**ETA print**: After first P-batch completes, prints estimated total time:
`First batch: Xs → estimated total: ~Y min (FFT-bound: E×n_batches batches)`.

**Monochromatic reshape fix**: `pressure_flat.reshape(Nx, Ny, Nz)` was wrong
(assumes x-outer ordering). Fixed to use `reshape_to_mapped_points(x, y, z, flat)[0]`
which correctly reverses the z-outer meshgrid flattening.

**Benchmark results** (LinearArrayTransducer E=128, P=60501, dx=dz=0.1mm, 12.5 MHz):

| Mode | Time |
|------|------|
| 1. Monochromatic global | ~29 s |
| 2. Pulsed global | ~11 s |
| 3. Global excitation | ~24 s |
| 4. Per-element (E=128) | ~17 min |

Per-element is FFT-bound: `E × n_batches × t_rfft = 128 × 15 × 0.53 s ≈ 1020 s`.
Irreducible on CPU for this grid size; reducible by coarsening grid or using GPU.

### Post-Batch 3 Performance Fix

**Problem**: Pulsed transient (mode 2) regressed from ~8 s to >100 s on
`bigelementstx/visualize_transient.py` (CustomTransducer, 128 elements,
P≈14K, M≈192K patches).

**Root causes**:
1. `_compute_active_window` — Python for-loop over all P points, each doing
   NumPy ops on M patches. O(P) Python iterations × O(M) per iteration ≈ 50–100 s.
2. `compute_time_grid` called twice — once in `__call__`, again inside `_compute_sir`.
   Each call runs O(P×M) Numba distance scan.

**Fix**:
1. Removed `_compute_active_window` entirely. SDI tail index now uses
   `info["max_time"]` returned by `compute_h_sir` (already computed inside
   the Numba kernel at zero extra cost).
2. `_compute_sir` accepts `time_grid_params` kwarg to reuse pre-computed
   `(time_grid, t0, dt, T)` from `__call__`. No redundant O(P×M) scan.

**Other cleanup applied**:
- `Pressure_flat` → `pressure_flat` (consistent snake_case).
- `rho` scaling unified: all modes (monochromatic + transient) now multiply
  by `self.rho` in a single common exit path.
- Duplicate reshape/coords logic merged into one block after mode dispatch.
- Dispatch flags documented with inline comment.

---

## Architecture Decision: Element-Loop Memory Strategy

**Decided during Batch 3 implementation (post-compaction). Applies to Batch 4.**

### Problem

Pre-computing `dh (T, P, E)` monolithically for large arrays causes OOM:
- Domino E=128, P=60501 → 142 GB
- Zeus_Matrix E=3025, P=~3000 → 397 MB per FFT batch in `_from_dh_per_element_to_pressure`

E-chunking within `_from_dh_per_element_to_pressure` mitigated but didn't eliminate the problem: each `(nfft, batch_P, E_chunk) float64` allocation still large for matrix arrays.

### Solution: Loop over elements, not chunks of elements

Instead of:
```
dh (T, P, E) → P-batch loop → E-chunk loop → FFT → accumulate
```

Do:
```
for e in range(n_elements):
    dh_e (T, P) → FFT → multiply by exc_e → accumulate into pressure
```

For each element e:
1. Filter patches: `mask = sub_el_idx_arr == e`; `patches_e = patches[mask]`
2. Call `compute_dh_per_element(points_batch, patches_e, n_elements=1)` → `(cols, 1, T)` → squeeze to `(cols, T)`
3. Zero tail: `dh_e[idx_e:, :] = 0.0` (transposed)
4. FFT + multiply: `H = rfft(dh_e.T, n=nfft)` → multiply by `fft_exc[:, e:e+1]`
5. Accumulate: `acc += irfft(H * fft_exc_e, n=nfft, axis=0)`

Peak memory per P-batch: `3 × (nfft × batch_P) × 8 bytes float64` ≈ 3 × 256 MB = 768 MB (independent of E).

### Benefits

| Property | E-chunk | Element-loop |
|----------|---------|-------------|
| Memory scaling | O(P × E_chunk × nfft) | O(P × nfft) — E-independent |
| Per-element excitation | Needs (L, E) pre-tiled | Natural: pick exc[:, e] |
| Per-element attenuation | Needs H_att (P, E, N_freq) pre-computed | Natural: compute H_att_e (P, N_freq) per e |
| TX/RX unified | Only TX | Same pattern for RX d2h_rx_e |

### Impact on Batch 4 (Reception)

Reception uses `compute_pe_sdi` (combined TX+RX delta placement) per RX element. Same element-loop applies:

```
for e_rx in range(n_rx_elements):
    rx_patches_e = patches[sub_el_idx == e_rx]
    compute_pe_sdi(all_tx_patches, rx_patches_e) → Dh_pe (P, T)
    FFT(Dh_pe) → multiply by FFT(v'_pe) → multiply by H_att → IFFT
    weight by scattering amplitudes → sum → rf[:, e_rx]
```

Each iteration: `O(P × nfft)` memory. Total sequential, no concurrent allocations.

---

## Batch 4 — Reception

**Needs all previous batches complete and tested.**

### Files to create / modify

| Action | File | Spec section |
|--------|------|-------------|
| CREATE | `src/pyfield/psimulation/reception.py` | §4 |
| MODIFY | `src/pyfield/psimulation/__init__.py` | §5 (add Reception export) |

### What to do per file

**`reception.py`** (§4) — new file:
- `Reception` class: constructor params `(tx, rx, *, c, rho, fs, alpha0, freq_power, excitation, downsampling, verbose)`.
- `.set(name, value)` — same `_SETTABLE` pattern as `Emission` (§6), without `monochromatic`.
- `__call__(scatterer_positions_mm, scattering_amplitudes, *, method="sdi")`:
  - Per RX element loop: call `compute_pe_sdi` with all TX patches + RX element patches (combined delta placement) → `Dh_pe (P, T)`.
  - FFT `Dh_pe`. Frequency-domain product chain:
    `RF_e(f) = Dh_pe × V'_pe × H_att`
    where `V'_pe = FFT(excitation) * FFT(ir_tx) * FFT(ir_rx)`.
  - H_att shape `(P, N_freq)` via `compute_reception_distances` → `causal_attenuation_tf`. Skip if `alpha0=None`.
  - IFFT, weight by `f_m(s)`, sum over scatterers → `rf[:, e_rx]`.
  - Scale by `rho / (2 * c^2)`.
  - Apply `downsampling` if not None.
  - Returns `(Nt, E_rx)`, `{"t0": float, "dt": float}`.
- Note: existing `compute_d2h` / `compute_dh` kernels in `sir_derivatives.py` are preserved for `Emission` use; Reception uses `compute_pe_sdi` exclusively.
- Memory strategy (§4.5 — element-loop): loop over RX elements, `compute_pe_sdi` per iteration produces `Dh_pe (P, T)`; O(P × nfft) peak independent of E_rx. Batch P if needed.
- `compute_sequence(scatterer_positions_mm, scattering_amplitudes, tx_events, *, time_btw_tx=None, method="sdi")`:
  - `tx_events`: list of dicts `{"delays": ndarray, "apodization": ndarray}`.
  - Loop: set TX delays/apod, call `__call__`, restore TX state after all events.
  - Returns `(N_events, Nt, E_rx)`, coords with `"time_btw_tx"` stored if provided.
- `compute_all(scatterer_positions_mm, scattering_amplitudes, *, method="sdi")`:
  - Each TX element transmits (delta or `self.excitation`), all RX receive.
  - Returns `(E_tx, Nt, E_rx)`, coords.

### Test gate

```
uv run pytest tests/ -k "reception"
```

Key assertions (§11):
- Single on-axis scatterer → symmetric RF across elements
- TX == RX (same object) → valid pulse-echo result
- `compute_sequence` with 1 event == `__call__` with same delays/apod
- `alpha0=None` → no attenuation applied (H_att term absent from chain)
- `downsampling=10` → output `Nt` is `floor(Nt_full / 10)`

---

## Notes

- Do not modify `farfield_rect_patch.py` or `hsir_SDI.py` at any point.
- Do not modify `plotting/`, `cache/`, `scans/`.
- Run `just pre-commit` after each batch before marking complete.
- Float32 kernels, float64 for FFT ops (upcast/downcast pattern from existing `sir_to_pressure.py`).
- All user-facing distances in mm, internal in metres.
