# Emission — Computation Flowchart

## Dispatch overview

```mermaid
flowchart TD
    A["Emission.__call__(field_points_mm, method)"]

    A --> PARSE

    subgraph PARSE["Input & Attenuation Setup"]
        P1{"dict input?"}
        P1 -->|yes| P2["parse grid dict<br/>→ x, y, z, points_m  shape P×3"]
        P1 -->|no|  P3["asarray / reshape<br/>→ points_m  shape P×3"]
        P2 & P3 --> P4{"alpha0 ≠ None?"}
        P4 -->|yes| P5["compute_attenuation_distances<br/>→ distances_m  shape P"]
        P4 -->|no|  P6["distances_m = None"]
    end

    PARSE --> DISP{"Mode dispatch<br/>(monochromatic, exc.ndim)"}

    DISP -->|"monochromatic=True"| M1
    DISP -->|"exc=None"| M2
    DISP -->|"exc.ndim=1, naive"| M3A
    DISP -->|"exc.ndim=1, sdi/auto"| M3B
    DISP -->|"exc.ndim=2"| M4
```

---

## Mode 1 — Monochromatic CW

```mermaid
flowchart TD
    M1A["compute_h_sir  →  h  T×P  float32<br/>farfield_rect_patch — prange over P<br/>cost: O(M · P · T̄)"]
    M1B["rfft(h)  →  H  N_freq×P  complex128"]
    M1C["abs(H[fc_idx, :])  →  amplitude  P<br/>reshape  →  Nx×Ny×Nz"]
    M1A --> M1B --> M1C
```

**Key**: single SIR kernel + one FFT per column, then index at `fc`. No time-domain output.

---

## Mode 2 — Pulsed (raw SIR)

```mermaid
flowchart TD
    M2A["compute_h_sir  →  h  T×P  float32<br/>SAME kernel as Mode 1<br/>cost: O(M · P · T̄)"]
    M2B["zero SDI tail:  h[idx_e:, :] = 0"]
    M2C["rho · fs · bwd_diff(h, axis=0)  →  dp  T×P<br/>pure time-domain — NO FFT"]
    M2D["abs(dp)  →  reshape  →  Nt×Nx×Ny×Nz"]
    M2A --> M2B --> M2C --> M2D
```

**Key**: identical SIR kernel as Mode 1. Differentiation is a finite-difference — trivial cost.

---

## Mode 3a — Global excitation, naive

```mermaid
flowchart TD
    M3A1["compute_h_sir naive  →  h  T×P<br/>cost: O(M · P · T)  — sample-by-sample loop"]
    M3A2["zero SDI tail"]
    M3A3["rfft(h)  ×  rfft(exc)  →  irfft  →  abs(p)<br/>cost: O(P · T · log T)"]
    M3A4["reshape  →  Nt×Nx×Ny×Nz"]
    M3A1 --> M3A2 --> M3A3 --> M3A4
```

---

## Mode 3b — Global excitation, SDI/auto  ⚠️ CURRENTLY SLOW

```mermaid
flowchart TD
    T1["tile exc  L → L×E<br/>E identical columns (same pulse for every element)"]
    T2["rfft(exc_tiled)  →  fft_exc  N_freq×E"]
    T3["for p_start in range(0, P, batch_P):"]

    subgraph BATCH["P-batch  (batch_P = min(512 MB ÷ E÷T÷4,  256 MB ÷ nfft÷8))"]
        B1["compute_dh_per_element(batch_P pts, all M patches, n_elements=E)<br/>→  dh_b  batch_P × E × T  float32<br/>⚠️  _d2h_per_element_ppar: O(M · batch_P · T)  — prange/P OK<br/>⚠️  _cumsum_3d:          O(E · batch_P · T) — now parallel via prange(P×E)"]
        B2["zero tail: dh_b[:, :, idx_e:] = 0"]
        B3["acc = zeros(nfft, batch_P)  float64"]
        ELOOP["for e in range(E):  — E = 128 iterations for Domino"]
        B4["h_pad = zeros(nfft, batch_P)<br/>h_pad[:T,:] = dh_b[:,e,:].T  (upcast float32 → float64)<br/>H = rfft(h_pad, axis=0)   →  N_freq × batch_P<br/>fft_P_e = H × fft_exc[:,e]<br/>acc += irfft(fft_P_e, nfft)"]
        B5["abs(acc[:T,:])  →  Pressure_flat[:, p_start:p_end]"]
        B1 --> B2 --> B3 --> ELOOP --> B4 --> ELOOP
        B4 -->|e=E-1| B5
    end

    T1 --> T2 --> T3 --> BATCH
    T3 -->|"next batch"| T3
```

**Why so slow**: for global excitation the SDI kernel still computes **E separate dh arrays** (one per element, summing M/E patches each). The work is proportional to **E × P × T** — the same as Mode 4. Tiling to `(L, E)` doesn't reduce the SDI cost at all.

---

## Mode 4 — Per-element excitation  (L×E)

```mermaid
flowchart TD
    E1["exc already L×E — each column is one element's pulse"]
    E2["rfft(exc)  →  fft_exc  N_freq×E"]
    E3["for p_start in range(0, P, batch_P):"]

    subgraph BATCH4["P-batch  (same memory caps as Mode 3b)"]
        F1["compute_dh_per_element(batch_P pts, all M patches, n_elements=E)<br/>→  dh_b  batch_P × E × T  float32<br/>cost: identical to Mode 3b"]
        F2["zero tail"]
        F3["acc = zeros(nfft, batch_P)"]
        ELOOP4["for e in range(E):"]
        F4["H = rfft(dh_b[:,e,:].T)<br/>fft_P_e = H × fft_exc[:,e]   ← each column is different here<br/>acc += irfft(fft_P_e)"]
        F5["abs(acc[:T,:])  →  Pressure_flat[:, p_start:p_end]"]
        F1 --> F2 --> F3 --> ELOOP4 --> F4 --> ELOOP4
        F4 -->|e=E-1| F5
    end

    E1 --> E2 --> E3 --> BATCH4
    E3 -->|"next batch"| E3
```

**Key**: per-element exc is the **natural and correct use case** for this code path. Every column of `fft_exc` is different, so E separate FFT multiplications are unavoidable.

---

## Summary: computational cost per mode

| Mode | SDI kernel | SDI output size | FFTs | Expected time |
|------|-----------|-----------------|------|---------------|
| 1 — CW | global `compute_h_sir` | T×P | 1 per P col | fast |
| 2 — Pulsed | global `compute_h_sir` | T×P | none (bwd_diff only) | **baseline** |
| 3a — Global, naive | global `compute_h_sir` naive | T×P | 1 per P col | slow (naive SIR) |
| **3b — Global, SDI** | **per-element `compute_dh_per_element`** | **T×P×E ← !** | **E per batch** | **~E× too slow** |
| 4 — Per-element | per-element `compute_dh_per_element` | T×P×E | E per batch | expected (unavoidable) |

---

## What Mode 3b SHOULD do

```mermaid
flowchart TD
    F1["compute_dh  →  dh  T×P  float32<br/>GLOBAL: single SDI kernel, same cost as Mode 2<br/>cost: O(M · P · T̄)"]
    F2["zero tail: dh[idx_e:, :] = 0"]
    F3["rfft(dh.T)  →  H  N_freq×P  ← ONE rfft call"]
    F4["H ×= rfft(exc)[:, np.newaxis]  ← ONE multiply, not E multiplies"]
    F5["irfft(H)  →  p  T×P<br/>abs(p)  →  reshape  →  Nt×Nx×Ny×Nz"]
    F1 --> F2 --> F3 --> F4 --> F5
```

Expected timing: **Mode 2 time + FFT overhead** (a few seconds, not ~200 s).

**Why it isn't done this way**: the test `test_uniform_per_element_matches_global` checks
that the global path and per-element path agree to `rtol=1e-3`. Due to `fastmath=True`
floating-point reordering, `_compute_d2h_ppar` (global) and the sum of
`_compute_d2h_per_element_ppar` differ by 1 ULP (~4096 at float32 scale of ~3.5×10¹⁰).
After cumsum this propagates as a constant offset (~6144) in an intermediate plateau of dh.
After convolution + abs, this tiny absolute difference causes large *relative* differences
near zero crossings → `rtol=1e-3` fails even though the signals are physically equivalent.

The fix is to restore the fast global dh path and relax the test metric (use `atol` on the
absolute pressure error, or compare peak-normalised energy rather than pointwise `rtol`).
