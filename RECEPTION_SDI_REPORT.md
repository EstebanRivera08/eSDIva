# Report — SDI in Pulse-Echo Reception

Findings from implementing and benchmarking the three pulse-echo formulations in
`ReceptionSDI` (`conventional` / `paired` / `factored`). Companion to `development.md`
(theory) and `examples/_research/reception_method_benchmark.py` (measurements).

## 1. Implementation details

Three evaluations of the same RF equation `p_pe = v_pe ⊛ h_tx ⊛ h_rx`. Each one-way SIR is
a trapezoid; its 2nd derivative `D²h` is 4 signed corner deltas (+,−,−,+ × slope).

| method | mechanism | kernel | placement cost | forward FFT |
|---|---|---|---|---|
| `conventional` | sample both SIRs (cumsum), FFT-convolve | `compute_h_sir` | `M` build | 2 fwd + 1 inv / scatterer (depth-binned) |
| `paired` `i4=fft` | place `Δδ_pe = D²h_tx ⊛ D²h_rx` (16/pair), `÷(jω)⁴` | `compute_pe_sdi` | `16·M_tx·M_Erx` | 1 fwd + 1 inv |
| `paired` `i4=splat` | splat `w=I⁴v_pe` per pair | `compute_pe_complete` | `16·M_tx·M_Erx · nfft` | 0 |
| `factored` | analytic `Σ_TX·Σ_RX`, `÷(jω)⁴` | `compute_oneway_spectrum_band` | `(M_tx+M_Erx)·N_band` | **0 fwd** + 1 inv |

Key mechanics:

- **Convolution theorem unifies the SDI forms**: `F{Δδ_pe} = F{Δδ^e ⊛ Δδ^r} = Σ_TX·Σ_RX`
  exactly. So the factored spectrum equals the paired spectrum at the FFT bins — same
  `inv_jw_pow` / `scale`, no separate scaling constant.
- **Analytic spectrum = sum of corner phasors**
  `Σ(ω) = Σ_m slope_m Σ_i σ_i e^{−jω(t_i−t0)}`, built by a **geometric phasor recurrence**
  (uniform frequency grid → constant `e^{−jΔω·Δt}` step), so no per-bin sin/cos.
- **Band-limiting**: `RF(ω) = G(ω)·S(ω)`, `G = V·IR·÷(jω)⁴` is band-limited by the pulse, so
  `Σ` is evaluated only on the `N_band ≪ N_freq` in-band bins (≈ 80/2048 here, ≈25×). Exact —
  out-of-band is killed by `G`.
- **TX-share**: `Σ_TX` is built once and reused for every RX element.
- **Per-patch one-way attenuation** folded into each phasor (patch-to-point distance is
  already computed in `_patch_corner_times`); the TX×RX product carries the true round trip.
  Free in `factored`, infeasible cheaply in `conventional`.
- **float64 is mandatory** for summed/integrated delta trains: sub-sample patches give slope
  ~1e10, areas (and the PE product) reach ~1e20, which must cancel almost completely to leave
  the tiny in-band signal. (Found + fixed a latent bug: `compute_pe_sdi_summed` returned
  float32 → `paired` correlation collapsed to ~0.4 on depth-spread arrays.)

## 2. Performance (measured, 8-core CPU, fs = 100 MHz)

| regime | conventional | factored | paired(fft) | winner |
|---|---|---|---|---|
| PSF, 64-el, per_scatterer | 0.069 | 0.071 | 0.211 | conv ≈ factored |
| compact 16-el, 100 scat | 0.300 | 0.224 | **0.031** | paired |
| large 128-el (2304 patch), 100 scat | **2.47** | 2.87 | 9.08 | conventional |

All methods agree to correlation ≥ 0.997.

**Why `factored` did NOT deliver the theorized ~25×:** the bound assumed `conventional` pays
`P·E_rx` full forward FFTs. In practice `conventional` already **depth-bins** (short per-bin
FFTs) and collapses scatterers in frequency before a single inverse FFT, so its effective cost
is near-linear, not `P·E_rx·T·logT`. `factored` removes the forward FFT but pays
`O(P·M·N_band)` — a **non-uniform DFT** (off-grid corner times), exactly the `O(M·N)` cost the
FFT was built to avoid. The two converge.

**CPU verdict:** all three land within ~1.2× on their good regimes. SDI rearrangement does
**not** beat an already-optimized FFT pipeline on CPU. The real wins are elsewhere (§5).

## 3. What could be done better

1. **Batch RX elements into one kernel call.** `factored` makes `E_rx` separate Numba calls
   (dispatch overhead). One `(E_rx, P, N_band)` kernel + one batched inverse FFT removes it —
   likely the biggest cheap CPU win.
2. **Tune the router.** `_PE_FFT_CONST = 4` is approximate and mis-routes the depth-spread
   summed case (picks `paired` where `conventional` wins). Add scatterer-count + depth-spread
   terms; re-fit per CPU.
3. **Shrink `N_band` further.** Tighter band threshold, or evaluate `Σ` on a decimated
   frequency grid + spectral interpolation. Scales `factored` directly.
4. **Mixed-precision phasor recurrence** (complex64 with periodic re-anchoring) to bound drift
   — halves inner-loop bandwidth.
5. **Pruned inverse FFT** for `per_scatterer` (only the band is nonzero).

## 4. New ideas — SDI as a reception speed tool

The CPU rearrangement doesn't win, but SDI's structure opens paths a sampled pipeline cannot:

1. **GPU is where `factored` dominates.** The analytic phasor sum is dense FLOPs with **no
   irregular scatter** (unlike delta placement) and **no FFT** — a perfect map-reduce.
   `conventional`'s per-scatterer FFTs batch poorly on GPU. The asymptotic story likely flips.
   *Strongest follow-up.*
2. **NUFFT — the genuine "best of both worlds."** A type-1 NUFFT computes `Σ_m e^{−jωt_m}`
   for all bins in `O(M + N logN)` — linear patch + FFT speed — at controllable (not exact)
   accuracy, re-introducing gridding. The principled middle between paired-placement and
   analytic-exact.
3. **Low-rank over the scatterer grid.** Neighbouring scatterers share smooth corner-delay
   fields → SVD/Chebyshev the delay+amplitude field, evaluate `r ≪ P` basis points, interpolate.
   Turns `P` spectra into `r`. Big for dense PSF/imaging grids.
4. **Element shift-in-Fourier.** Linear array: element `e` geometry = element 0 translated →
   corner delays differ by a smooth phase ramp. Build a reference element once, phase-shift in
   Fourier for the rest. `E` spectrum builds → `1 + E` cheap multiplies. (TX-share is the
   trivial case of this.)
5. **Reciprocity fold** (monostatic `TX = RX`): `Σ_TX = Σ_RX` → square it, halve the builds.
6. **Exactness → coarser `fs`.** No quantization/aliasing means a lower `fs` suffices for the
   same accuracy → fewer bins everywhere. Indirect but real.

## 5. SDI strengths in reception

- **Sparse SIR representation** — a handful of corner deltas, not `T` samples. Cheap when truly
  sparse (few patches, monoelement, PSF).
- **Closed-form, exact frequency domain** — no time sampling, no interpolation, no `nfft`
  padding of the SIR. More accurate than both `conventional` and paired-placement.
- **Separability** — the two-way delay is an outer sum → `factored` is linear in patch count
  (the M² wall is broken algebraically).
- **Composability** — attenuation, K-K dispersion, impulse responses and `I⁴` all multiply in
  one spectral step; per-patch attenuation is essentially free.
- **Derivative bookkeeping for free** — `D²` *is* the delta train; `I⁴` in Fourier carries zero
  group delay (sample-aligned, no cumsum drift).
- **No irregular memory in the analytic form** — GPU/SIMD friendly.

## 6. SDI limitations in reception

- **Off-grid → non-uniform DFT.** The analytic spectrum is `O(M·N)`; it loses to FFT's
  `O(N logN)` for large `M·N` unless band-limited. Exactness ⊕ FFT-speed — pick one (only NUFFT
  bridges, approximately).
- **Float32 cancellation hazard.** Huge delta areas (~1e20) cancel → float64 mandatory for any
  summed/integrated train; easy to reintroduce bugs.
- **Far-field trapezoid assumption.** SDI is valid only where the rectangular-patch far-field
  SIR holds — accuracy is subdivision-dependent.
- **M² wall for exact pair enumeration** (`paired`) unless the path is separable.
- **Band-limiting required** for `factored` to be competitive — it fails for a near-delta /
  wideband excitation (band = whole spectrum), where the router falls back to `conventional`.
- **Does not beat `conventional`'s depth-binning on CPU** for large depth-spread summed fields.
- **`per_scatterer` cannot fold the scatterer sum** → `P` inverse FFTs (fine only when `P` is
  small, i.e. a PSF).

---

**Bottom line:** on CPU, SDI's algebraic rearrangement buys *exactness, per-path attenuation,
and graceful M-scaling*, not raw speed — the FFT pipeline is already near-optimal. The real
speed upside is **GPU** (factored's dense, scatter-free shape) and **redundancy exploitation**
(low-rank scatterer grids, element phase-ramps), neither yet built. Those are the highest-value
next steps.
