# Emission Class — Function Call Workflows

Reference for understanding how `Emission.__call__` routes through internal functions
for each simulation mode, and where time is spent.

Reception class workflow will be added here once implemented.

---

## Mode Decision Tree

```
Emission.__call__(field_points_mm)
           │
           ├─ monochromatic=True ──────────────────────────────────────────┐
           │                                                                │
           │   use_per_element? ──── False ──► [A] Mono Global             │
           │                   └─── True  ──► [B] Mono Per-Element         │
           │                                                                │
           └─ monochromatic=False                                           │
                      │                                                     │
                      │   use_per_element? ──── True  ──► [E] Per-Element  │
                      │                                                     │
                      ├─ exc=None, alpha0=None ─────────► [C] Pulsed Pure  │
                      │                                                     │
                      └─ exc=(L,) or fast_attenuation ──► [D] Global FFT  ─┘


  use_per_element = (alpha0 is not None and not fast_attenuation) OR exc.ndim == 2
```

---

## Shared Preamble (every call)

Runs before any mode dispatch. Cost is negligible (<< 1 s).

```
__call__(field_points_mm)
  │
  ├─ [dict input] create_3D_spatial_grid_from_points(field_points_mm)
  │    → x (Nx,), y (Ny,), z (Nz,), points_m (P, 3) in metres
  │
  ├─ compute per_elem_exc, use_per_element flags
  │
  ├─ _announce_mode(exc, use_per_element)    [print mode header]
  │
  ├─ [per_elem_exc] validate exc.shape[1] == n_elements
  │
  ├─ compute_time_grid(P, M, points_m, centers, wx, wy, c, fs, delays)
  │    → time_grid (T,), t0 (float), dt (float), T (int)
  │    cost: O(P × M) distance scan, ~0.05–0.5 s for large grids
  │
  ├─ [alpha0 and global att path]
  │    compute_attenuation_distances(points_m, tx_center) → distances_m (P,)
  │
  └─ [exc not None and tx.impulse_response not None]
       _apply_ir_to_excitation(exc, ir)    [np.convolve, fast]
```

---

## [A] Monochromatic Global

**Trigger**: `monochromatic=True`, `alpha0=None` or `fast_attenuation=True`

```
_mono_global(points_m, distances_m, method)
  │
  ├─ _compute_sir(points_m, method)
  │    ├─ compute_time_grid(...)              [redundant: also called in preamble]
  │    ├─ compute_h_sir(P, M, T, dt,          ◄── BOTTLENECK: Numba, O(P × M × avg_T)
  │    │    time_grid, points_m, centers,           ~25–35 s for P=60k, M=1280
  │    │    wx_arr, wy_arr, inv_c, fs,
  │    │    apod, delays, method_flag)
  │    │    → h (P, T) float32
  │    └─ return h.T, t0   → (T, P)
  │
  ├─ _compute_active_window(points_m, t0, T, dt)
  │    → idx_e    [SDI tail guard; Python loop over patches, fast]
  │
  ├─ h[idx_e:, :] = 0.0
  │
  └─ from_sir_to_monochromatic_pressure(h, None, None, None, fc, fs,
          alpha0=alpha0, distances_m=distances_m)
       ├─ np.fft.rfft(h[:, batch], axis=0)   [float64 upcast, sequential batches]
       │    → H (T//2+1, batch) complex128
       ├─ |H[fc_idx, :]|                      [extract single frequency bin]
       └─ × |H_att| if alpha0                 [scalar multiply per point]
       → pressure_flat (P,)

└─ reshape_to_mapped_points(x, y, z, pressure_flat)[0]  → (Nx, Ny, Nz)
```

| Step | Cost | Notes |
|------|------|-------|
| `compute_h_sir` | ~25–35 s | Numba JIT, parallelised over P |
| `from_sir_to_monochromatic_pressure` | ~2–5 s | Sequential FFT batches, float64 |
| `compute_time_grid` | ~0.05 s | Called twice (minor redundancy) |

**Dominant cost**: Numba SIR kernel.

---

## [B] Monochromatic Per-Element

**Trigger**: `monochromatic=True`, `alpha0 is not None` and `fast_attenuation=False`

```
_mono_per_element(points_m, T, dt, time_grid, method_flag)
  │
  ├─ _extract_patch_slices()    → list of E tuples (centers_e, wx_e, wy_e, apod_e, delays_e)
  │
  ├─ exp_vec = exp(-j2πfc × time_grid)   (T,) complex64    [one-time, fast]
  │
  ├─ for e in range(n_elements):   ◄── E iterations (tqdm progress bar)
  │    │
  │    ├─ _compute_h_sir_batch(points_m, T, dt, time_grid, method_flag, patch_slices[e])
  │    │    → compute_h_sir(P, M_e, T, ...)   ◄── BOTTLENECK: Numba, M_e = M/E patches
  │    │    → h_e (P, T) float32              total: E × Numba(M/E) = same work as global
  │    │
  │    ├─ for p_batch:
  │    │    ├─ h_e[batch].astype(complex64) @ exp_vec   → H_e_fc (cols,)  [dot product]
  │    │    └─ [alpha0] _causal_tf_at_fc(dist_e_b)      → H_att_e_b (cols,) complex64
  │    │         causal_attenuation_tf at single freq fc only — cheap
  │    │
  │    ├─ acc_flat[batch] += H_e_fc × H_att_e_b
  │    └─ del h_e    [free immediately]
  │
  └─ abs(acc_flat)  → (P,) float32

└─ reshape_to_mapped_points(x, y, z, pressure_flat)[0]  → (Nx, Ny, Nz)
```

| Step | Cost | Notes |
|------|------|-------|
| E × `compute_h_sir(M/E patches)` | same total FLOPs as [A] | Parallelised over P each time |
| dot product `h_e @ exp_vec` | negligible | (cols, T) × (T,) per batch |
| `causal_tf_at_fc` | negligible | single frequency, no FFT |

**Dominant cost**: same Numba kernel as [A], just called E times with M/E patches each.
Total arithmetic equivalent — overhead from E Python loop calls is small.

---

## [C] Pulsed Pure

**Trigger**: `excitation=None`, `alpha0=None`

Legacy path through `from_sir_to_pressure`.

```
_compute_sir(points_m, method)      ◄── BOTTLENECK: Numba
  → h (T, P), t0

_compute_active_window(...)  → idx_e_h
h[idx_e_h:, :] = 0.0

from_sir_to_pressure(h, None, None, None, fs, rho=rho, excitation=None)
  [excitation=None branch]: returns h unchanged  →  Pressure_flat = h (T, P)

reshape_to_mapped_points(x, y, z, Pressure_flat) × rho  → (Nt, Nx, Ny, Nz)
```

| Step | Cost | Notes |
|------|------|-------|
| `compute_h_sir` | ~8–12 s | Same kernel as [A], but less post-processing |
| `from_sir_to_pressure` | ~0 s | h returned directly when `excitation=None` |
| reshape + rho | < 1 s | |

**Dominant cost**: Numba SIR kernel.
**Fastest transient mode** — no FFT post-processing at all.

---

## [D] Global FFT

**Trigger**: `excitation=(L,)` or (`alpha0 is not None` and `fast_attenuation=True`);
`use_per_element=False`

```
_transient_global(points_m, t0, T, dt, time_grid, distances_m, method, exc_1d)
  │
  ├─ rfftfreq(nfft, 1/fs)  → freqs (N_freq,) float32
  │
  ├─ [exc_1d not None]
  │    rfft(exc_1d, n=nfft, workers=-1)      [scipy, fast, single call]
  │    j2pif × fft_exc → fft_exc (N_freq,) complex64   [derivative in freq domain]
  │
  ├─ [transfer_function] TF = transfer_function(freqs)  (N_freq,) complex64
  │
  ├─ [alpha0] causal_attenuation_tf(freqs, distances_m, ...)
  │    → H_att (P, N_freq) complex64    [pre-computed for all P at once]
  │
  └─ for p_batch in range(n_batches):   [n_batches = ceil(P / batch_P)]
       │
       ├─ _compute_h_sir_batch(pts_batch, T, dt, time_grid, method_flag)
       │    → compute_h_sir(cols, M, T, ...)  ◄── BOTTLENECK part 1: Numba
       │    → h_b (cols, T) float32
       │
       ├─ rfft(h_b, n=nfft, axis=1, workers=-1)    ◄── BOTTLENECK part 2: scipy FFT
       │    [zero-pads internally from T→nfft; n > T → scipy internal buffer allocated]
       │    → H (cols, N_freq) complex64
       │
       ├─ H *= fft_exc         if exc
       ├─ H *= TF              if TF
       ├─ H *= H_att[batch]    if alpha0
       │
       └─ abs(irfft(H, n=nfft, axis=1, workers=-1)[:, :T]).T
            → Pressure_flat[:, batch]   float32

└─ reshape_to_mapped_points(x, y, z, Pressure_flat) × rho  → (Nt, Nx, Ny, Nz)
```

| Step | Cost | Notes |
|------|------|-------|
| `compute_h_sir` per batch | ~1.5–2 s/batch | Numba, cols × M points |
| `rfft(h_b, n=nfft)` | ~0.3–0.5 s/batch | scipy multi-threaded; n>T → internal zero-pad buffer per call |
| `irfft` | ~0.3 s/batch | One per batch |
| `causal_attenuation_tf` | ~0.5 s one-time | Full (P, N_freq) pre-computed |
| Total | ~24 s | P=60k, 15 batches |

**Dominant cost**: Numba SIR + scipy FFT interleaved per batch.
**Note**: `rfft(h_b, n=nfft)` creates an internal zero-padding buffer of
`(cols, nfft) float32` per call because `n > len(h_b[0])`.
For large grids (n_batches × 280 MB per call) this can stress memory.

---

## [E] Per-Element Transient

**Trigger**: `use_per_element=True`, `monochromatic=False`.
Handles pulsed (`exc=None`), global (`exc=(L,)`), and per-element (`exc=(L,E)`) excitation
plus any attenuation — all share one code path.

```
_transient_per_element(points_m, t0, T, dt, time_grid, method_flag, exc)
  │
  ├─ _extract_patch_slices()   → E tuples   [fast, one-time]
  │
  ├─ rfftfreq(nfft, 1/fs), j2pif   (N_freq,) complex64
  │
  ├─ [exc not None] pre-compute fft_exc_list (one-time, outside all loops):
  │    exc=(L,)   → fft_e = j2pif × rfft(exc, n=nfft)    shared ref × E
  │    exc=(L,E)  → [j2pif × rfft(exc[:, e], n=nfft) for e in E]
  │
  ├─ [transfer_function] TF = transfer_function(freqs)   [one-time]
  │
  ├─ h_pad_buf = zeros((batch_P, nfft), float32)   ◄── PRE-ALLOCATED ONCE
  │    tail [:, T:] stays zero for all iterations
  │
  └─ for p_batch in range(n_batches):   ◄── outer loop: n_batches iterations
       │
       ├─ acc_H = zeros((cols, N_freq), complex64)
       │
       └─ for e in range(n_elements):   ◄── inner loop: E iterations
            │
            ├─ _compute_h_sir_batch(pts_batch, T, dt, time_grid, method_flag, patch_slices[e])
            │    → compute_h_sir(cols, M_e, T, ...)   M_e = M/E patches
            │    → h_e_b (cols, T) float32     ◄── BOTTLENECK part 1: Numba
            │
            ├─ h_pad[:cols, :T] = h_e_b        [write into pre-alloc buffer; tail stays 0]
            │
            ├─ rfft(h_pad[:cols], axis=1, workers=-1)   ◄── BOTTLENECK part 2: scipy FFT
            │    [h_pad already length nfft → no internal scipy zero-pad buffer]
            │    → H_e (cols, N_freq) complex64         ~0.5 s per call
            │
            ├─ H_e *= fft_exc_list[e]   if exc
            ├─ H_e *= TF                if TF
            ├─ [alpha0]
            │    dist_e_b = norm(pts_batch - elem_centers[e], axis=1)   (cols,)
            │    causal_attenuation_tf(freqs, dist_e_b, ...)   → H_att_e_b (cols, N_freq)
            │    H_e *= H_att_e_b
            │
            └─ acc_H += H_e;  del H_e
            │
       ├─ irfft(acc_H, n=nfft, axis=1, workers=-1)[:, :T].T   ◄── ONE per batch
       │    → Pressure_flat[:, batch]
       │
       └─ [ib == 0] print ETA: t_first × n_batches → estimated total

└─ reshape_to_mapped_points(x, y, z, Pressure_flat) × rho  → (Nt, Nx, Ny, Nz)
```

| Step | Cost | Notes |
|------|------|-------|
| `compute_h_sir` per (batch, element) | ~0.35 s | Numba, M/E patches, cols points |
| `rfft(h_pad)` per (batch, element) | ~0.53 s | scipy multi-thread; **h_pad already nfft → no buffer** |
| `irfft(acc_H)` | ~0.3 s/batch | ONE call per batch, not per element |
| `causal_attenuation_tf` | ~0.02 s/(batch×elem) | (cols, N_freq) per element |
| **Total** per batch | ~68 s | (Numba + rfft) × E per batch |
| **Total** | ~17 min | 15 batches × 68 s |

**Total rfft call count**: `E × n_batches = 128 × 15 = 1920`
**Total irfft call count**: `n_batches = 15`
**Memory peak per batch**: `O(batch_P × nfft)` — E-independent.

---

## Bottleneck Summary

| Mode | Primary bottleneck | Secondary | Typical time |
|------|-------------------|-----------|-------------|
| [A] Mono Global | `compute_h_sir` (Numba, all M patches) | `rfft` for mono extraction | ~29 s |
| [B] Mono Per-Element | `compute_h_sir` × E (M/E patches each) | dot product h·exp (negligible) | ~29 s |
| [C] Pulsed Pure | `compute_h_sir` (Numba) | none — h returned directly | ~11 s |
| [D] Global FFT | `compute_h_sir` per batch + `rfft` per batch | `causal_attenuation_tf` (one-time) | ~24 s |
| [E] Per-Element | `rfft` × E × n_batches (FFT-bound) | `compute_h_sir` × E × n_batches | ~17 min |

Benchmark conditions: `LinearArrayTransducer` E=128, M=1280, P=60501 (201×1×301),
T=4608, nfft=8192, fs=200 MHz, fc=12.5 MHz.

---

## Memory Architecture (Per-Element Mode)

The memory design in `_transient_per_element` eliminates the main cause of
OS-swap-induced slowdown seen in earlier versions:

```
WRONG (earlier): zeros((cols, nfft), float32) inside E-loop
  → E × n_batches allocations = 128 × 15 × 140 MB = 268 GB traffic

WRONG (middle): rfft(h_b, n=nfft) with n > len(h_b)
  → scipy creates internal zero-pad buffer (cols, nfft) per call anyway
  → same 268 GB traffic, just hidden inside scipy

CORRECT (current): h_pad_buf = zeros((batch_P, nfft), float32) ONCE
  → h_pad[:, :T] = h_e_b each iteration, tail stays zero
  → rfft(h_pad) receives already-nfft input → no scipy buffer
  → ONE allocation, reused 1920 times
```

Freq-domain accumulation (not time-domain) keeps irfft count at n_batches:
```
irfft(H_0 + H_1 + ... + H_{E-1}) = irfft(H_0) + irfft(H_1) + ...   [linear]
→ accumulate in freq domain → one irfft per batch → interference preserved
```

---

## Key Internal Functions

| Function | Module | What it does |
|----------|--------|-------------|
| `compute_h_sir` | `h_sir/farfield_rect_patch.py` | Numba JIT SIR, `(P, T) float32` |
| `compute_time_grid` | `utilities/helper_functions.py` | Min/max TOF scan → `T, t0, dt, time_grid` |
| `_extract_patch_slices` | `emission.py` | Pre-split patches per element, cache outside E-loop |
| `_compute_h_sir_batch` | `emission.py` | Wrapper: passes element patch arrays to `compute_h_sir` |
| `_batch_P` | `emission.py` | Batch size from 400 MB budget |
| `causal_attenuation_tf` | `psimulation/attenuation.py` | Causal power-law H_att, K-K dispersion |
| `from_sir_to_monochromatic_pressure` | `psimulation/sir_to_pressure.py` | FFT→single bin→reshape |
| `from_sir_to_pressure` | `psimulation/sir_to_pressure.py` | Full FFT convolution with excitation |
| `reshape_to_mapped_points` | `utilities/helper_functions.py` | `(T, P) → (T, Nz, Nx, Ny) → transpose (T, Nx, Ny, Nz)` |

---

## Potential Optimisations (not yet implemented)

| Optimisation | Target mode | Expected gain |
|-------------|-------------|---------------|
| GPU FFT (cupy/torch) | [E] per-element | E × n_batches rfft → ~10–50× |
| Coarser grid | all | P ↓ → fewer batches |
| Pre-compute all-element h_sir then split | [E] | Avoids E-loop overhead if memory permits |
| Async Numba + FFT pipeline | [D][E] | Overlap SIR compute and FFT |
| Remove double `compute_time_grid` in [A] | [A] | Minor: ~0.1 s |
