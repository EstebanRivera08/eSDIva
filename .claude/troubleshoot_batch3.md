# Troubleshoot — Batch 3: Emission / Per-Element Excitation

## Failing Test
`TestEmissionPerElementExcitation::test_uniform_per_element_matches_global`

Assertion: uniform per-element excitation (same pulse × E elements) must equal
global excitation output within rtol=1e-3.

---

## Diagnosis Path

### Attempt 1 — Boundary truncation hypothesis (wrong root cause)

**Hypothesis**: global path uses `h_sir *_t bwd_diff_trunc(exc)*fs`. The
`np.diff(exc, prepend=0)` has length L=80 and misses the boundary term
`bwd_diff(exc)[L] = -exc[L-1]`. This creates a constant ~1.227e10 offset
starting at sample n = (idx_last_h_nonzero + L).

**Fix attempted**: changed `sir_to_pressure.py` to differentiate h_sir instead
of exc: `bwd_diff(h_sir)*fs *_t exc`.

**Result**: test still failed with 79% element mismatch and max relative
difference ~150×. Waveforms were completely different, not just offset by
a constant. **Reverted.**

### Attempt 2 — SDI shift hypothesis (correct direction, wrong granularity)

**Hypothesis**: global path uses naive h_sir then numeric diff; per-element
path uses SDI dh. The SDI kf formula `kf = (t-t0)*fs + 1.0` shifts events by
+1 sample. After convolution with exc and abs(), near zero crossings this 1-sample
shift causes relative differences up to 150×.

**Fix attempted**: changed global path in `emission.py` to use
`_compute_sir_derivative(per_element=False)` (global SDI dh) and call
`_from_dh_per_element_to_pressure` with E=1.

**Result**: still failed (79% mismatch, ~97× max relative diff). Waveform
now changed (SDI used), but still doesn't match per-element.

### Root Cause Found — float32 fastmath ULP in intermediate cumsum tails

**Diagnosis**: compared `dh_global` vs `sum_e dh_pe_e` directly:

```
d2h_g [170] = d2h_pe_sum[170] + 4096   # 1 ULP at ~3.5e10 scale
d2h_g [189] = d2h_pe_sum[189] - 4096
d2h_g [190] = d2h_pe_sum[190] + 256
```

The 4096 difference is exactly 1 ULP of float32 at the ~3.5e10 scale,
caused by `fastmath=True` in the two different Numba kernels
(`_compute_d2h_ppar` vs `_compute_d2h_per_element_ppar`) reordering
floating-point ops differently.

After `integrate_d2h_to_dh` (cumsum with float64 accumulator, float32
write-back), the 1-ULP difference at sample 170 propagates as a **constant
offset** of 6144 in dh across the intermediate plateau (samples 171-188):

```
dh_g     [171:189] = +2048  (plateau residual)
dh_pe_sum[171:189] = -4096  (sum of per-element residuals: 0 + (-2048) + (-2048) + 0)
diff                = +6144  (constant)
```

This intermediate tail is NOT zeroed by `_compute_active_window` (which only
zeros the final tail after the last event). The 6144 constant difference in
the active window propagates through FFT convolution, creating ~1e-5 absolute
errors that show up as 150× relative differences near zero crossings.

**Why is the final test sensitive to this?** Both signals look similar in
magnitude but are oscillating. Near zero crossings, p_global[n] ≈ small,
p_pe[n] ≈ small+6144_contribution, so relative diff is huge.

---

## Fix Applied

**`emission.py` — global excitation path** (`exc.ndim == 1`):

Changed from: compute global SDI dh (1 cumsum) → convolve with exc

Changed to: **compute per-element dh (E separate cumsums) → tile exc to (L, E)
→ call `_from_dh_per_element_to_pressure`**.

```python
n_elements = int(self.delays.shape[0])
dh, t0 = self._compute_sir_derivative(points_m, derivative="dh", per_element=True)
idx_e = self._compute_active_window(points_m, t0, dh.shape[0], 1.0 / self.fs)
dh[idx_e:, :, :] = 0.0
effective_exc = self._apply_ir_to_excitation(exc, self.tx.impulse_response)
exc_tiled = np.tile(effective_exc[:, np.newaxis].astype(np.float32), (1, n_elements))
pressure = self._from_dh_per_element_to_pressure(dh, x, y, z, exc_tiled, distances_m)
```

**Why this works**: for uniform excitation, the per-element path computes
`|sum_e irfft(rfft(dh_pe_e) * rfft(exc))[:T]|`. The global path (with tiled
exc) now computes the IDENTICAL expression using the SAME dh_pe arrays.
Result is bit-for-bit identical.

**Trade-off**: global path now does E separate cumsums instead of 1. For
E=64 elements this is ~64× more SDI accumulation work. Acceptable since
the alternative (summing cumsums) is fundamentally inconsistent.

---

## Other Fix: PyField.normalize backward compatibility

**Problem**: integration test `test_normalize_option` called
`sim(grid, normalize=True)`. New PyField wrapper had `**kwargs` silently
absorbing `normalize`, so normalization was never applied.

**Fix**: added explicit `normalize=False` parameter to `PyField.__call__`,
with post-processing: `pressure = pressure / max(abs(pressure))` when True.

---

## Files Modified (Batch 3)
- `src/pyfield/psimulation/emission.py` — Emission class, global excitation path fix
- `src/pyfield/psimulation/PyField.py` — normalize parameter restored
- `src/pyfield/psimulation/sir_to_pressure.py` — minor (linter touched it during session; attenuation import removed by linter since Emission no longer calls it with excitation)
- `src/pyfield/h_sir/sir_derivatives.py` — new file (SDI derivative kernels)
- `src/pyfield/psimulation/attenuation.py` — new file (attenuation transfer function)
- `tests/unit/test_psimulation/test_emission.py` — new Batch 3 test file

## Gate Result
`uv run pytest tests/ -k "emission or pyfield"` → **29 passed**
Full suite (excl. visual): **135 passed**

---

## Post-Batch 3 OOM Fixes (same session, continued after context compaction)

### Problem 1: Global excitation path OOM for large P or large E

Original approach (`_compute_sir_derivative(per_element=True)`) allocated the full
`dh (T, P, E)` monolithically — 142 GB for Domino (E=128, P=60501).

First fix (batch P in global path): worked for small E but `_from_dh_per_element_to_pressure`
still allocated `(nfft, batch_size, E)` float64 ≈ 397 MB per batch for Zeus_Matrix (E=3025).

**Final fix**:
1. `_from_dh_per_element_to_pressure` refactored to loop over E-chunks:
   - `pe_cap = 128MB // (nfft * 8)` → `E_chunk = min(E, pe_cap)`, `batch_P = pe_cap // E_chunk`
   - Sequential nested P×E loops; each iteration allocates `(nfft, batch_P, E_chunk)` ≤ ~512 MB peak
   - Removed ThreadPoolExecutor (numpy FFT already multi-threaded internally)
2. Global excitation path (`exc.ndim==1`) batches P before calling `_from_dh_per_element_to_pressure`:
   - `batch_P = 512MB // (n_elements * T * 4)`

### Problem 2: Monochromatic CW OOM after h_sir allocated

`h_sir` (1 GB) already in memory; `rfft(h_sir[:, batch])` → complex128 (16 B/elem, not 8).
ThreadPoolExecutor ran multiple batches concurrently → 72 MB × N_workers more allocations.

**Fix** in `from_sir_to_monochromatic_pressure` (`sir_to_pressure.py`):
- Switched to `rfft` (halves vs full fft)
- Cap formula corrected: `8MB // (T_rfft * 16)` (complex128 bytes)
- Replaced ThreadPoolExecutor with sequential `for` loop → only 1 batch in flight at a time

### Problem 3: SDI cumsum tail visible in pulsed mode ("static cone")

**Fix** in `emission.py` pulsed path (`exc is None`):
```python
idx_e_h = self._compute_active_window(points_m, t0, T_h, dt_h)
h[idx_e_h:, :] = 0.0
```

### Problem 4: Transient plot vmin/vmax broken (colorbar starting at wrong value)

Root cause: `to_dB(display_frames)` without `vmax=p_max`; no explicit `vmax=0` to imshow.
First frame (before wavefront arrives) has very small dB max → imshow auto-scales vmax to
that negative value (e.g., -50 dB), matplotlib swaps with user's vmin (-40 dB) → colorbar
shows [-50, -40] instead of [vmin, 0].

**Fix** in `plot2D_pressure_slices` (`plotting2D.py`):
1. Compute `p_max` from **full** `pressure_field` before decimation
2. Call `to_dB(display_frames, vmax=p_max)` for consistent normalisation
3. Pop `vmin`/`vmax` from kwargs; set `vmax_plot=0` for dB
4. Pass explicit `vmin=vmin, vmax=vmax_plot` to all `imshow` calls

### Problem 5: method="naive" not supported for excitation modes

**Fix** in `emission.py` global excitation path (`exc.ndim==1`):
- If `method=="naive"`: fall back to `_compute_sir(naive)` + `from_sir_to_pressure(excitation=...)`
- If `method` is "auto"/"sdi": use per-element dh path (unchanged)

### Files Modified (post-compaction)
- `src/pyfield/psimulation/emission.py` — OOM batch+E-chunk fix, naive routing, pulsed tail zero
- `src/pyfield/psimulation/sir_to_pressure.py` — rfft + sequential + 8MB cap, noisy print removed
- `src/pyfield/plotting/plotting2D.py` — vmin/vmax transient fix, p_max from full field

### Final Gate
`uv run pytest tests/ -k "emission or pyfield"` → **29 passed**

---

## Post-Batch 3 Refactor: Element-Loop Memory Strategy (next session)

### Problem: E-chunking still OOM for large matrix arrays

After OOM fixes above, `_from_dh_per_element_to_pressure` with E-chunking was still
allocating `(nfft, batch_P, E_chunk) float64` ≈ 397 MB for Zeus_Matrix (E=3025).
The per-element excitation path pre-computed `dh (T, P, E)` monolithically — 142 GB
for Domino (E=128, P=60501).

### Solution: Element-loop — O(P×nfft) peak, E-independent

**Architecture decision**: loop over E elements instead of chunks.

For each element `e`:
1. Filter patches: `mask_e = sub_el_idx_arr == e`; `n_patches_e = mask_e.sum()`
2. Create `sub_el_idx_e = zeros(n_patches_e, int32)` (all-zero = single-element call)
3. Call `compute_dh_per_element(pts_batch, filtered_patches, ..., sub_el_idx_e, n_elements=1)` → `(cols, 1, T)`
4. Zero tail: `dh_e[:, 0, idx_e:] = 0.0`
5. FFT + multiply: `H = rfft(dh_e[:, 0, :].T)` → `fft_P_e = H * fft_exc[:, e:e+1]`
6. Accumulate: `acc += irfft(fft_P_e, nfft, axis=0)`

Peak memory per P-batch: `3 × (nfft × batch_P) × 8 bytes float64` ≈ 768 MB (E-independent).

**Why same result as monolithic**: `compute_dh_per_element` accumulates `out[p, sub_el_idx[m], k]`.
Filtering to `mask_e = idx == e` and remapping to `sub_el_idx_e = 0` produces exactly
`out_monolithic[:, e, :]` — bit-identical.

**Uniform excitation test still passes** (rtol=1e-3): global path tiles `(L,) → (L, E)`,
per-element path gets `(L, E)` directly; both call `_element_loop_to_pressure` with
identical inputs → same dh_e per element → bit-identical accumulation.

### Files Modified (element-loop refactor)
- `src/pyfield/psimulation/emission.py`:
  - Removed `ThreadPoolExecutor` import (no longer needed)
  - Removed `_from_dh_per_element_to_pressure` (E-chunked)
  - Added `_element_loop_to_pressure(points_m, excitations_LE, distances_m, t0, T, dt, idx_e)`
  - Refactored `exc.ndim==1` SDI path: removed `compute_dh_per_element` P-batch + `_from_dh_pe_to_pressure` calls; now `compute_time_grid` → tile exc → `_element_loop_to_pressure`
  - Refactored `exc.ndim==2` path: removed `_compute_sir_derivative(per_element=True)` → `_from_dh_pe_to_pressure`; now `compute_time_grid` → `_element_loop_to_pressure`
- `REFACTOR_PLAN.md` — added "Architecture Decision: Element-Loop Memory Strategy" section
- `PROMPT2.md` — updated §3.3 excitation dispatch, §4.5 reception memory strategy, §9 performance constraints

### Gate
`uv run pytest tests/ -k "emission or pyfield"` → **29 passed**
`uv run pytest tests/ --ignore=tests/unit/test_plotting` → **135 passed**
