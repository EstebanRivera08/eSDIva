# Pulse-Echo Reception — three formulations, complexity, and optimization plan

Working document for choosing and optimizing the pulse-echo RF kernels. It states
the three formulations that compute the same physical signal, derives the
computational complexity of each (including SIR build, cumsum, and FFT/iFFT costs),
tabulates their bottlenecks and advantages, anchors the orders against measured
Domino timings, and lists the concrete optimizations worth applying. The final
section is written as a prompt/context block to drive an implementation plan.

Companion to `ARCHITECTURE.md` → "Pulse-Echo Post-Processing & Depth Binning".

---

## 0. Notation

| Symbol | Meaning | Domino value |
|---|---|---|
| `P` | number of point scatterers | 10–10⁴ |
| `E` | RX output channels (= RX elements) | 128 |
| `M_tx` | total TX patches (whole active aperture) | 1280 |
| `M_Erx` | patches per **single** RX element | 10 |
| `T` | RF / SIR time-series length (samples) | ~10³ |
| `L` | excitation-chain length `exc ⊛ ir_tx ⊛ ir_rx` (samples) | ~46 |
| `L_w` | length of integrated kernel `w = I⁴ v_pe` | ~`L` |
| `N_d` | attenuation depth bins (atten variant only) | 20–50 |
| `nfft` | FFT length, `next_pow2(T + L − 1)` | ~2·T |
| `C_fft` | one transform cost `≈ nfft·log₂ nfft` | — |
| `I` | time-integration operator; `I⁴` = ÷(jω)⁴ in frequency | — |
| `⊛` | temporal convolution | — |

One **linear convolution via FFT** = 1 rfft + spectral multiply + 1 irfft
≈ `2·C_fft + nfft`. Below, "1 FFT-conv" means this whole cost.

Geometry facts used throughout:
- Every scatterer sees the **whole** TX aperture (`M_tx` patches) and **one** RX
  element at a time (`M_Erx` patches). So `M_tx ≫ M_Erx` — the exploitable asymmetry.
- TX is **shared across RX elements** within one transmit event (hoistable).
- RF is **linear in scatterers** (superposition → amortizable over `P`).

---

## 1. The unifying identity (why there are only three forms)

All three methods compute the **same** `p_pe`. Each one-way SIR is the double integral
of its piecewise-constant 2nd derivative (a set of corner deltas):
`h_tx = I² D²h_tx`, `h_rx = I² D²h_rx`. Substituting and moving integrals through the
convolutions gives one identity chain — each equality *is* one of the three methods:

    p_pe = v_pe ⊛ (h_tx ⊛ h_rx)                                    ← (1) Conventional
         = v_pe ⊛ I⁴(D²h_tx ⊛ D²h_rx) = v_pe ⊛ (I⁴ Δδ_pe)          ← (2) Truncated SDI PE
         = (I⁴ v_pe) ⊛ Δδ_pe = w ⊛ Δδ_pe
         = Σ_i Σ_j a_i a_j · w(t − τ_i − τ_j)                       ← (3) Complete SDI PE

    Δδ_pe ≡ D²h_tx ⊛ D²h_rx   (analytic; deltas ⊛ deltas = deltas, 16·M_tx·M_Erx of them)
    w     ≡ I⁴ v_pe           (analytic; precomputed once)

They differ only in **how `p_pe` is evaluated** — at which stage the four integrals are
applied and whether the two-way SIR is kept factored or expanded:

1. **Conventional** — `p_pe = v_pe ⊛ (h_tx ⊛ h_rx)`. Build each one-way SIR by
   placing its 2nd-derivative deltas (`8·M` sample-writes) and **double-cumsuming**
   (`2T`) to recover `h_tx`, `h_rx`; then **two FFT-convs**. The convolution is
   M-independent, but **the SIR build is linear in M** (`8M` writes + `2T` per SIR).
2. **Truncated SDI PE** — `p_pe = v_pe ⊛ (I⁴ Δδ_pe)`, where
   `Δδ_pe ≡ D²h_tx ⊛ D²h_rx` is the **analytic** convolution of the two
   second-derivative delta trains. Convolution of deltas is again deltas → `Δδ_pe`
   is `16·M_tx·M_Erx` deltas (4 TX-corner × 4 RX-corner per patch pair). Place them,
   then realize `I⁴` **entirely in Fourier** as `÷(jω)⁴`, folded into the single
   spectral multiply with `v_pe` — **no time-domain cumsum** (this also removes the
   float32 cumsum-cancellation hazard, gotcha #1). **1 FFT-conv** total.
3. **Complete SDI PE** — push all four integrals onto the velocity,
   `w ≡ I⁴ v_pe` (analytic, precomputed once), so the convolution collapses to a
   closed sum of shifted kernels: `p_pe = Σ_i Σ_j a_i a_j · w(t − τ_i − τ_j)`.
   **No FFT, no cumsum** — just `16·M_tx·M_Erx` scaled, shifted copies of `w`.

**Separability is the whole story for the convolution stage.** The weights
`a_i a_j = a_tx ⊗ a_rx` are **rank-1** and the shifts `τ_i + τ_j = τ_tx ⊕ τ_rx` are an
**outer sum** → the inner double sum is a separable bilinear form. FFT-convolution
*exploits* that separability (cost `T log T`, **independent of patch count**); pair
enumeration (methods 2, 3) *discards* it and pays `M²` to rediscover it. Convolution
of two sparse delta trains is fundamentally `min(M², T log T)` — no rewrite beats that.
Enumeration only wins when `M` is tiny, or when the physics **breaks separability**
(per-path attenuation, §6) and forces enumeration anyway. (Note: even the conventional
path is not globally M-free — its *SIR build* is linear in M; only its *convolution* is.)

---

## 2. Method 1 — Conventional `Reception` : `v_pe ⊛ (h_tx ⊛ h_rx)`

Builds the two one-way SIRs separately and convolves. SIR build via `naive`
(per-sample trapezoid loop) or `sdi` (delta train + cumsum).

### Algorithmic flow
```
precompute  fft_v = rfft(v_pe)                                   # once
for p in scatterers:
    build h_tx(p)            # M_tx patches → SIR(T)   [naive: M_tx·T ; sdi: M_tx + cumsum T]
    g(p) = v_pe ⊛ h_tx(p)    # hoisted TX side: 1 FFT-conv          (shared over all E)
    for e in RX elements:
        build h_rx(p,e)      # M_Erx patches → SIR(T)
        p_pe(p,e) = g(p) ⊛ h_rx(p,e)    # 1 FFT-conv
        RF[e] += a_p · p_pe(p,e)
```

### Complexity
| Phase | Cost |
|---|---|
| TX SIR build | `P·(M_tx + T)`  (sdi) or `P·M_tx·T` (naive) |
| TX hoisted conv | `P · C_fft` |
| RX SIR build | `P·E·(M_Erx + T)` |
| RX two-way conv | **`P·E · C_fft`**  ← dominant |
| **Total** | **`O(P·E·T·log T)`** |

The `naive`/`sdi` choice only changes the sub-dominant SIR build, so total time is
nearly identical between them (confirmed empirically: `naive ≈ sdi`). The two-way FFT
conv `P·E·C_fft` is the floor, and it is the **same FFT Field II runs** — so the
speedup over Field II comes only from the faster SIR build and from PyField's batched
FFT, landing at ~2× (Amdahl: SIR is a small fraction of the budget).

---

## 3. Method 2 — Truncated SDI PE : `v_pe ⊛ (I⁴ Δδ_pe)`  (current `ReceptionSDI`)

Forms `Δδ_pe = D²h_tx ⊛ D²h_rx` **explicitly** (the kernel places 16 deltas per patch
pair), then realizes the full `I⁴` in the frequency domain as `÷(jω)⁴`
(`n_integrations=4`) — **no time-domain cumsum** (removed: it added ½-sample group
delay and risked float32 cancellation, gotcha #1). The kernel's three SIR derivatives
are relocated onto the exc/IR chain by the same spectral multiply.

### Algorithmic flow
```
precompute  fft_v, fft_ir_tx, fft_ir_rx, (jω)^-4                 # once
for e in RX elements:
    Δδ_pe = compute_pe_sdi(...)        # (P, T): place 16·M_tx·M_Erx deltas → prange over P
    if summed & no atten:
        Δ_sum = a · Δδ_pe              # collapse scatterers (BLAS matvec)
        RF[e] = irfft( rfft(Δ_sum) · (jω)^-4 · fft_v · fft_ir_tx · fft_ir_rx )   # 1 FFT-conv
    else:                              # per_scatterer or attenuation
        H = rfft(Δδ_pe, axis=1)        # P FFTs (no amortization)
        ...· (jω)^-4 · filters / per-scatterer H_att...
        RF[e] = Σ_p a_p · irfft(H)
```

### Complexity (summed, no attenuation)
| Phase | Cost |
|---|---|
| Pair-product placement | **`P·E · 16·M_tx·M_Erx`**  ← dominant for arrays |
| FFT (amortized over P) | `E · C_fft` |
| **Total** | **`O(P·E · M_tx·M_Erx)`** |

No cumsum term — `I⁴` is now entirely in the FFT multiply (`÷(jω)⁴`).

### Complexity (per_scatterer **or** attenuation)
FFT amortization is lost — `P` forward FFTs per element:
`O(P·E·M_tx·M_Erx + P·E·C_fft)` — **both** terms worse than Method 1 → never use for arrays.

### Why it loses on arrays
Method 2 replaces Method 1's `P·E·C_fft` with `P·E·16·M_tx·M_Erx`. The crossover is

    16·M_tx·M_Erx   vs   C_fft ≈ T·log₂ T

Domino: `16·1280·10 = 204 800` vs `T·log₂T ≈ 10³·10 ≈ 2×10⁴` → ~**10× worse**, matching
the measured `pe_sdi 0.2×` vs `conventional 2.2×` (≈11× gap). The `M_tx·M_Erx` pair
count is irreducible by construction — it is the M²-in-patches wall.

---

## 4. Method 3 — Complete SDI PE : `Σ_i Σ_j a_i a_j · w(t − τ_i − τ_j)`

Precompute the single integrated waveform `w = I⁴ v_pe` once, then accumulate shifted,
scaled copies — **no FFT, no cumsum**.

### Algorithmic flow
```
w = I⁴ v_pe            # once: 4 integrations of the exc/IR chain (length L_w)
for e in RX elements:
    for p in scatterers:
        for i in TX patches (M_tx):
            for j in RX patches (M_Erx):
                RF[e, k:k+L_w] += a_p · a_i · a_j · w   at offset τ_i + τ_j
```

### Complexity
| Phase | Cost |
|---|---|
| Precompute `w` | `O(L_w)` (once) |
| Pair accumulation | **`P·E · 16·M_tx·M_Erx · L_w`**  ← dominant |
| **Total** | **`O(P·E · M_tx·M_Erx · L_w)`** |

Strictly Method 2 × `L_w` (it splats a length-`L_w` copy per pair instead of one delta
+ a single shared FFT). So for arrays it is the slowest. Its value is elsewhere:

- **FFT-free** — no `nfft` padding, no spectral setup.
- **Exact** — `w` evaluated at the continuous shifted time `τ_i + τ_j`: no SDI
  sample-binning interpolation error and no float32 cumsum cancellation
  (gotcha #1). Ideal as a **golden reference** for a single scatterer / monoelement.
- **Per-path attenuation rides free** (§6) — the unique regime where enumeration is
  not a fallback but the *correct* method.

---

## 5. Master comparison

### Complexity (dominant term)
| | Method 1 Conventional | Method 2 Truncated SDI | Method 3 Complete SDI |
|---|---|---|---|
| Form | `v_pe ⊛ h_tx ⊛ h_rx` | `v_pe ⊛ I⁴Δδ_pe` | `Σ a_i a_j w(t−τ_i−τ_j)` |
| SIR build | `P(M_tx+T) + PE(M_Erx+T)` | (folded into pair product) | (folded into pair product) |
| Pair work | — | `PE·16·M_tx·M_Erx` | `PE·16·M_tx·M_Erx·L_w` |
| Cumsum | `PE·T` (SIR build) | none (`÷(jω)⁴` in FFT) | none |
| FFT/iFFT | `PE·C_fft` | `E·C_fft` (summed) / `PE·C_fft` (atten) | none |
| **Total** | **`O(PE·T·logT)`** | **`O(PE·M_tx·M_Erx)`** | **`O(PE·M_tx·M_Erx·L_w)`** |
| Crossover vs M1 | — | wins iff `16 M_tx M_Erx < T·logT` | wins iff `16 M_tx M_Erx L_w < T·logT` |

### Bottlenecks & advantages
| Method | Bottleneck | Advantage | Best regime |
|---|---|---|---|
| **1 Conventional** | `PE` FFT-convs (same FFT as Field II) | M-independent; rank-1 separability exploited; depth-binning shrinks `T` | **Arrays, many scatterers, weak/no atten** |
| **2 Truncated SDI** | `M_tx·M_Erx` pair product per (scat,elem) | FFT amortized over `P` → `E` FFTs; FFT-light for small `M` | **Small `M` (PSF, monoelement, few patches)** |
| **3 Complete SDI** | `M_tx·M_Erx·L_w` (slowest at scale) | FFT-free; **exact** (no interp/cumsum error); **per-path attenuation free** | **Reference / single scatterer / per-path-atten near-field** |

### Empirical anchor — Domino linear (128 elem, M_tx=1280, M_Erx=10, fs=100MHz)
| `P` | Field II | Conventional (M1) | ReceptionSDI (M2) | M2/M1 |
|---|---|---|---|---|
| 10 | 0.012 s | 0.031 s (0.4×) | 0.078 s (0.2×) | 2.5× |
| 100 | 0.088 s | 0.084 s (1.0×) | 0.559 s (0.2×) | 6.6× |
| 1 000 | 0.842 s | 0.428 s (2.0×) | 4.76 s (0.2×) | 11× |
| 10 000 | 8.526 s | 3.95 s (2.2×) | 43.8 s (0.2×) | 11× |

All methods linear in `P` (separable superposition). M2's ~11× constant penalty at
scale = the predicted `16·M_tx·M_Erx / T·logT`. **For arrays, Method 1 is correct and
Method 2 is the worst option** — exactly what the benchmark shows.

---

## 6. Attenuation — the regime that flips the verdict

Power-law attenuation `H_att(ω, d)` depends on propagation distance `d`. Current
`ReceptionSDI` applies it **per scatterer-center** (`reception_sdi.py:411`, one
distance `d_pe` per (scatterer, element)) — an approximation, and it costs M2 its FFT
amortization (`P` FFTs/element). True attenuation is **per path**: pair `(i,j)`
travels `d_ij = c·(TOF_tx,i + TOF_rx,j)` (geometric TOF, **not** electronic-delayed
`τ`). The exact weight `a_i a_j · H_att(ω, d_ij)` depends on **both** `i` and `j` →
**no longer rank-1 separable** → **FFT (Method 1) cannot factor it**. Enumeration is
forced — and Method 3 carries it for free:

    precompute  w_d = I⁴ v_pe ⊛ h_att(d)   for N_d depth bins        # N_d FFTs, GLOBAL, shared
    RF[e] = Σ_p a_p Σ_{i,j} a_i a_j · w_{bin(d_ij)}(t − τ_i − τ_j)    # per-pair kernel lookup

Same `O(PE·M_tx·M_Erx·L_w)` as plain Method 3 — attenuation becomes a table lookup.
This is **more accurate** than the per-scatterer approximation wherever the aperture
path-spread is large (near field, strong focusing, low F-number) and is a capability
the separable FFT path structurally cannot provide. Novel vs both PyField-now and
Field II (which also approximates per-scatterer). Niche: **per-path attenuated PSF /
monoelement / near-field focused**.

---

## 7. Optimization opportunities in the current implementation

Ordered by leverage. Each is independent unless noted.

1. **Regime router (highest leverage).** Dispatch on the cheap inequality
   `16·M_tx·M_Erx  ⋛  T·log₂T` (and `attenuation` flag, `per_scatterer` flag):
   arrays/weak-atten → Method 1; small `M` → Method 2; reference/per-path-atten →
   Method 3. The benchmark proves a single class cannot win all regimes. Add
   `_regime_select(M_tx, M_Erx, T, attenuation, per_scatterer)` and route in
   `ReceptionBase.pulse_echo_rf`. Calibrate the constant once against measured timings.

2. **Accumulate-in-kernel before the FFT (Method 2, summed/no-atten).** The cumsum is
   now gone, but `compute_pe_sdi` still returns a full `(P, T)` delta buffer that
   `reception_sdi.py:383` collapses with `a @ Δδ_pe`. Amplitude-accumulate **all**
   scatterers' deltas into one `(T,)` buffer **inside** the kernel instead: memory
   `(P,T)·8B → (T,)·8B` (Domino P=10⁴: 320 MB → 32 KB, cache-resident) and kills the
   matvec. Caveat: `prange` over `P` then races the shared buffer → thread-local
   buffers + reduction, or re-tile the parallel axis (ties into the patch-vs-scatterer
   parallelization question in §8). Summed-no-atten path only.

3. **prange granularity for PSF.** `compute_pe_sdi` parallelizes over `P`
   (`transducer_sir_pe.py:175`); useless when `P=1` (single-point PSF) — the
   `M_tx` loop runs serial. Add a `per_scatterer`/`P==1` path that pranges over
   `m_e` (TX patches) instead.

4. **Per-path attenuated Method 3 (new capability).** Implement the depth-binned
   kernel family `w_d` (§6) as the attenuation backend for small-`M`/reference runs.
   Gated behind the router (only when enumeration is already chosen). Validate the
   near-field accuracy gain vs the per-scatterer approximation before committing.

5. **Route attenuated arrays to Method 1.** Current M2 attenuation path is worst-case
   (`P` FFTs/elem **and** the pair product). The router (1) already covers this:
   `attenuation && large M → Method 1 + per-scatterer H_att`.

6. **Confirm TX-hoist in Method 1.** Verify `g(p) = v_pe ⊛ h_tx(p)` is computed once
   per scatterer and reused across `E` (not recomputed per element). If not, it is a
   free `E×` cut on the TX-conv term.

7. **Depth-binning / decimation on Method 1** (the only `T·logT`-shrinking lever for
   arrays — already exposed as `downsampling=`; see `ARCHITECTURE.md`). This is what
   pushes Method 1 past Field II's 2× toward larger margins at high `P`.

---

## 8. Planning prompt / context

> **Goal:** restructure PyField `ReceptionSDI` to expose the three
> mathematically-equivalent pulse-echo formulations behind a `method` flag (mirroring
> `Reception`): `conventional` (Method 1), `truncated` (Method 2), `complete`
> (Method 3) — so the three formalisms can be run and compared on identical inputs.
> Each has a different complexity regime and set of advantages (§5). I still think
> `truncated`/`complete` may have a faster *implementation* than `conventional` even
> though the algorithm differs — e.g. parallelizing over patches instead of
> scatterers, batching over scatterers (rarely >500k), or vectorizing the pair sum.
> Treat the M² wall (§1) as the thing to push against in wall-clock for realistic `P`,
> not assume insurmountable.
>
> **Invariants:** all three forms must remain bit-comparable to Field II `calc_scat`
> (corr ~1.0) on the no-attenuation path; `coords["t0"]` beam-axis referencing
> unchanged; public `pulse_echo_rf` / `sequence_rf` / `synthetic_aperture_rf` /
> `scan_focusline` signatures unchanged (`method` is an internal dispatch flag).
>
> **Decision gates:** profile `compute_pe_sdi` (placement vs FFT) on Domino
> `P∈{10²,10³,10⁴}` before committing a parallelization rewrite — confirms whether the
> patch-pair placement or the FFT dominates. Before the per-path attenuation kernel
> (§6), run the accuracy harness quantifying per-path vs per-scatterer error vs
> depth/F-number — confirms it is worth the `L_w` cost.

