# Pulse-Echo Reception — three formulations, complexity, and the spectral fast path

Working document for the pulse-echo RF kernels in `ReceptionSDI` / `Reception`. It
states the three formulations that compute the same physical signal, derives each one's
computational complexity (SIR build, cumsum, FFT/iFFT, closed-form spectrum), tabulates
their bottlenecks, and — the focus of the current revision — explains **why the spectral
formulation is the fast path at high scatterer counts**, including the three changes that
made it so: the cancellation-free factored one-way spectrum, depth-binning of the spectral
window, and the fused multi-element two-way kernel.

Companion to `ARCHITECTURE.md` → "Pulse-Echo Post-Processing & Depth Binning".

---

## 0. Notation

| Symbol | Meaning | Reference value |
|---|---|---|
| `P` | number of point scatterers | 10 – 10⁵ |
| `E` | RX output channels (= RX elements) | 64 – 128 |
| `M_tx` | total TX patches (whole active aperture) | ~1500 |
| `M_Erx` | patches per **single** RX element | `no_sub_x·no_sub_y` ~ 24 |
| `M_rx` | total RX patches = `E·M_Erx` | ~1500 |
| `T` | RF / SIR time-series length (samples) | ~10³–10⁴ |
| `L` | excitation-chain length `exc ⊛ ir_tx ⊛ ir_rx` (samples) | ~50–100 |
| `fs`, `BW` | sample rate / signal bandwidth | 100 MHz / ~5 MHz |
| `nfft` | FFT length, `next_pow2(T + L − 1)` | ~2·T |
| `N_band` | in-band frequency bins, `≈ (BW/fs)·nfft` | see §4 |
| `C_fft` | one transform cost `≈ nfft·log₂ nfft` | — |
| `I` | time-integration operator; `I⁴` = ÷(jω)⁴ in frequency | — |
| `⊛` | temporal convolution | — |

Geometry facts used throughout:
- Every scatterer sees the **whole** TX aperture (`M_tx` patches) and (for per-channel RF)
  **one** RX element at a time (`M_Erx` patches). TX is **shared across RX elements** within
  one transmit event (hoistable / reusable).
- RF is **linear in scatterers** (superposition → amortizable over `P`).
- The received signal is **band-limited** by `exc ⊛ ir_tx ⊛ ir_rx`: only `N_band` of the
  `nfft/2` frequency bins carry signal.

---

## 1. The unifying identity (why there are only three forms)

All three methods compute the **same** pulse-echo response `p_pe = v_pe ⊛ h_tx ⊛ h_rx`.
Each one-way SIR is the double integral of its piecewise-constant 2nd derivative — a
sparse train of trapezoid-corner deltas: `h_tx = I² D²h_tx`, `h_rx = I² D²h_rx`. `D²h` of
one rectangular patch is **four signed Diracs** at its corner times. Substituting and moving
integrals through the convolutions gives one identity chain; each equality *is* one method:

    p_pe = v_pe ⊛ (h_tx ⊛ h_rx)                                  ← (1) Conventional
         = (I⁴ v_pe) ⊛ (D²h_tx ⊛ D²h_rx) = w ⊛ Δδ_pe            ← (2) Paired
         = F⁻¹{ V_pe·(jω)⁻⁴ · Σ_TX(ω) · Σ_RX(ω) }                ← (3) Spectral

    Δδ_pe ≡ D²h_tx ⊛ D²h_rx   (deltas ⊛ deltas = deltas: 16·M_tx·M_Erx of them)
    w     ≡ I⁴ v_pe           (the integrated drive, precomputed once)
    Σ_TX(ω) ≡ F{D²h_tx}       (closed-form sum of 4 corner phasors per patch)

They differ only in **how** the same product is evaluated:

1. **Conventional** — build each one-way SIR by sampling (place its corner deltas,
   double-cumsum to recover `h_tx`, `h_rx`), then convolve by FFT. The convolution is
   patch-count independent, but the SIR build is linear in M. (Delegated to `Reception`.)
2. **Paired** — convolve the two corner-delta trains analytically into the 16-delta two-way
   train `Δδ_pe`, enumerating all `M_tx·M_Erx` patch pairs; push the four integrations onto
   the drive once (`w = I⁴ v_pe`) and splat a shifted copy of `w` per corner event. No FFT,
   no cumsum — exact, but `O(M²)`. (`compute_pe_complete`.)
3. **Spectral** — never form the pairs and never sample the SIR. Each one-way SIR spectrum
   `Σ_TX(ω)`, `Σ_RX(ω)` is written in **closed form** (a sum of corner phasors per patch),
   evaluated only on the in-band frequencies, and the two are multiplied (convolution ⇒
   product). The four integrations `I⁴` and the exc/IR chain are one downstream spectral
   multiply, and a single inverse FFT per element returns the RF.
   (`compute_twoway_spectrum_summed` + `compute_oneway_spectrum_band`.)

**Separability is the structure being exploited or paid for.** The two-way weights
`a_i a_j` are rank-1 and the shifts `τ_i + τ_j` are an outer sum, so the two-way SIR is a
separable bilinear form. Conventional FFT-conv and the spectral product both *exploit* this
(cost independent of the pair count); paired enumeration *discards* it and pays `M²` to
rediscover it. Paired only wins when `M` is tiny or when the physics **breaks separability**
(per-path attenuation, §7) and forces enumeration anyway.

---

## 2. Method 1 — Conventional `Reception` : `v_pe ⊛ (h_tx ⊛ h_rx)`

Builds the two one-way SIRs (sampled) and convolves by FFT. Includes a depth-binned fast
path: scatterers are grouped by depth so each bin uses a short time window → small `nfft`.

### Complexity (summed, depth-binned)
| Phase | Cost |
|---|---|
| TX SIR build | `P·(M_tx + T_bin)` |
| RX SIR build | `P·E·(M_Erx + T_bin)` |
| Forward FFTs (per scatterer, per element) | **`P·E · C_fft(nfft_bin)`**  ← dominant |
| Scatterer sum + 1 inverse FFT / element | `E · C_fft(nfft_bin)` |
| **Total** | **`O(P·E · nfft_bin·log nfft_bin)`** |

The forward-FFT term is the floor — it is the **same FFT Field II runs**. Depth-binning
shrinks `nfft_bin` (hence the floor) but cannot remove the `P·E` forward transforms.

---

## 3. Method 2 — Paired SDI PE : `Σ_i Σ_j a_i a_j · w(t − τ_i − τ_j)`

Precompute `w = I⁴ v_pe` once; for each of the 16 corner events of every TX–RX patch pair
add a shifted, scaled copy of `w`. No FFT, no cumsum.

### Complexity
| Phase | Cost |
|---|---|
| Precompute `w` | `O(nfft)` (once) |
| Pair accumulation | **`P·E · 16·M_tx·M_Erx · len(w)`**  ← dominant |
| **Total** | **`O(P·E · M_tx·M_Erx · len(w))`** |

Slowest at array scale (the `M²·len(w)` wall). Its value: FFT-free, **exact** (continuous
shift `τ_i+τ_j`, no sample-binning or cumsum-cancellation error), and **per-path attenuation
rides free** (§7). Used for tiny apertures (PSF, monoelement) and as the golden reference.

---

## 4. Method 3 — Spectral SDI PE : `F⁻¹{ V_pe·(jω)⁻⁴ · Σ_TX · Σ_RX }`  (the fast path)

The spectral form is the default for band-limited drives and the focus of this revision.
It builds each one-way SIR spectrum analytically and multiplies — **no forward FFT at all**.

### 4.1 The closed-form one-way spectrum, in cancellation-free factored form

`D²h` of one patch is four corner deltas at `t1,t2,t3,t4`, so its Fourier transform is a
four-phasor sum. Using the corner-time structure `t2=t1+Δt1, t3=t1+Δt2, t4=t1+Δt1+Δt2`,
that sum **factors**:

    e^{-jωt1} − e^{-jωt2} − e^{-jωt3} + e^{-jωt4}
        = e^{-jω t1}·(1 − e^{-jωΔt1})·(1 − e^{-jωΔt2}) ,

and with `1 − e^{-jx} = 2j·sin(x/2)·e^{-jx/2}` it collapses to a **real envelope times one
phasor** at the patch-centre arrival `t_c = (t1+t4)/2 = l/c`:

    S_patch(ω) = -4·slope·sin(ωΔt1/2)·sin(ωΔt2/2)·e^{-jω(t_c − t0)} .

The aperture spectrum is `Σ_TX(ω) = Σ_patches S_patch`, swept over the uniform `omega` grid
by complex-multiply recurrence (the half-angle phasors advance by a constant factor; no
`sin`/`cos` per bin). Per-patch causal attenuation `exp(-α|f|^y d)·(dispersion)` is folded
into each patch term using the patch-to-point distance, so the TX×RX product carries the
true round-trip loss.

**Why the factored form matters numerically.** The naive four-phasor sum adds `±slope`
terms — and `slope = h_max/Δt1` is *large* for thin patches — relying on cancellation to
land the small physical value. That cancellation is what forced complex128 accumulation. The
factored form computes the small value **directly** via `sin`: for small `Δt1`,
`4·slope·sin(ωΔt1/2) → 2·h_max·ω`, bounded by the physical plateau and never inflated. The
scatterer sum is then well conditioned, so complex64/float32 works (validated:
complex64 vs complex128 relerr ≈ 2·10⁻⁷). It also uses **3 swept phasors instead of 4**.

### 4.2 Cost, and the depth-span trap that depth-binning removes

For one window (no binning), the summed spectral RF costs, per RX element:

    build Σ_RX,e : P·M_Erx·N_band     product+sum : P·N_band     1 inverse FFT : C_fft

plus the shared TX build `P·M_tx·N_band` (once). Total over `E` elements:

    O( P·N_band·(M_tx + M_rx)  +  P·E·N_band  +  E·C_fft ) ,   M_rx = E·M_Erx.

The dominant factor is `N_band`. Crucially **`N_band = (BW/fs)·nfft` and `nfft ∝ T` spans
the arrival window of *all* scatterers**, so a deep or wide field inflates `N_band` linearly
with depth span. (Earlier this was wrongly assumed bandwidth-only; it is not — it scales
with the time record.) A single-window spectral run on a deep field therefore carries a
large `N_band` and loses.

**Depth-binning fixes this.** Group scatterers by depth into bins; each bin spans a tight
arrival window → small `nfft_bin` → small `N_band_bin = (BW/fs)·nfft_bin`. All bins share one
global sample lattice, so each bin's RF adds back at an integer sample offset (no
resampling). The bin count is **floor-aware** (`_auto_depth_bins`, shared with conventional):
shrink windows only until `nfft_bin` hits its `next_pow2(L)` floor — past that, extra bins do
not shrink `nfft` and only add overhead. Per-bin cost has the small `N_band_bin`; summed over
bins the total is `O(P·N_band_floor·(M_tx + M_rx))` — **independent of the field's depth
span**. (Conventional already binned; this session brought the same trick to spectral.)

### 4.3 The fused multi-element two-way kernel

The summed two-way spectrum per element, `S_e(ω) = Σ_p a_p·Σ_TX(ω;r_p)·Σ_RX,e(ω;r_p)`, is
evaluated by one fused kernel (`_twoway_summed_points`) rather than a Python loop over
elements. Per scatterer the TX spectrum `Σ_TX` is built **once** and reused while sweeping
all RX elements; the result accumulates straight into the per-element output. This removes,
relative to the earlier per-element loop:

- the **per-element kernel relaunch** (one launch per bin instead of `E` per bin),
- the **re-streaming of `Σ_TX`** from memory `E` times (it stays in cache, reused),
- every **`(P, N_band)` intermediate** (nothing of that size is materialized).

It parallelizes over scatterer chunks (race-free per-chunk buffers) and accumulates in
complex128 internally — cheap because `N_band` is small after binning, and well conditioned
thanks to the factored form (§4.1). RX patches are laid out element-by-element (CSR offsets
`rx_ptr`), so `focused_sum` (the beamformed scan line) is just the single-group case.

---

## 5. Why the spectral path is faster than conventional / Field II

Compare the two dominant terms directly (depth-binned, summed, no attenuation):

    Conventional :  P·E · nfft_bin·log₂(nfft_bin)        (forward FFTs — the Field II floor)
    Spectral     :  P·N_band_bin·(M_tx + M_rx) + P·E·N_band_bin

With a full aperture both sides (`M_tx = M_rx = E·M_Erx`) the spectral build dominates and
`≈ 2·P·E·M_Erx·N_band_bin`. Substituting `N_band_bin = (BW/fs)·nfft_bin`, the ratio is

    spectral / conventional  ≈  2·M_Erx·N_band_bin / ( nfft_bin·log₂ nfft_bin )
                             =  2·M_Erx·(BW/fs) / log₂(nfft_bin) .

The `nfft_bin` **cancels** — the ratio is set by three things, none of them the record
length:

1. **No forward FFT (the big one).** Conventional pays `P·E` forward transforms (one per
   scatterer per element); that is the same FFT Field II runs and its irreducible floor.
   Spectral has **zero** forward FFTs — the spectrum is closed form — and only `E` inverse
   FFTs per bin (after the scatterer sum), which is negligible. The whole `nfft·log` floor is
   gone, replaced by the analytic build.
2. **Band-limiting.** Spectral evaluates only the `N_band` in-band bins; the FFT processes the
   full `nfft` length regardless. The `(BW/fs)` factor is this saving.
3. **Few patches per element.** The remaining cost scales with `M_Erx` (patches per element),
   typically small (`no_sub_x·no_sub_y`).

So spectral wins whenever `2·M_Erx·(BW/fs) < log₂(nfft_bin)` — i.e. band-limited drives
(small `BW/fs`), modest per-element subdivision, and deep windows (large `log`). For the
reference probe (`M_Erx=24`, `BW/fs≈0.05`, `log₂(256)≈8`) the ratio is `≈0.3`, i.e. ~3× — the
measured ~0.7–0.8× also pays the shared TX build (`P·M_tx·N_band`), the inverse FFTs, and
Python/setup overhead, which the asymptotic ratio omits.

The three implementation changes map onto this directly: **factored form** makes the analytic
build cheap and float32-safe (enabling the no-FFT path to stay accurate); **depth-binning**
keeps `N_band_bin` at its floor regardless of field extent; **the fused kernel** removes the
per-element launch/streaming overhead that otherwise dominated at the small per-bin sizes.

---

## 6. Master comparison

### Complexity (dominant term, summed, no attenuation)
| | M1 Conventional | M2 Paired | M3 Spectral |
|---|---|---|---|
| Form | `v_pe ⊛ h_tx ⊛ h_rx` | `Σ a_i a_j w(t−τ_i−τ_j)` | `F⁻¹{V_pe(jω)⁻⁴ Σ_TX Σ_RX}` |
| Forward FFT | `P·E·C_fft` (floor) | none | **none** |
| Inverse FFT | `E·C_fft` | none | `E·C_fft` (per bin) |
| Patch work | SIR build `P(M_tx+M_rx)` | `P·E·16·M_tx·M_Erx·len(w)` | spectrum build `P·N_band·(M_tx+M_rx)` |
| Cumsum | in SIR build | none | none |
| **Total** | **`O(PE·nfft_bin·log)`** | **`O(PE·M_tx·M_Erx·len(w))`** | **`O(P·N_band_bin·(M_tx+M_rx))`** |

### Bottlenecks & best regime
| Method | Bottleneck | Best regime |
|---|---|---|
| **1 Conventional** | `P·E` forward FFTs (Field II floor) | wideband / near-delta drive; fallback |
| **2 Paired** | `M_tx·M_Erx·len(w)` | tiny aperture (PSF, monoelement); per-path attenuation; reference |
| **3 Spectral** | analytic build `P·N_band·M` (no forward FFT) | **band-limited drive — arrays + high P (default)** |

`method="auto"`: spectral when a band-limited excitation/IR is present (the usual case),
conventional for a near-delta/wideband drive (no band-limiting benefit), paired only for a
handful of patches.

### Empirical anchor — 64-element linear, `M_tx = M_rx = 1536`, `M_Erx = 24`, `fs = 100 MHz`, `fc = 5 MHz`
Depth-binned spectral (current, fused) vs conventional (depth-binned, Field II-style):

| `P` | depth (mm) | bins | conventional | spectral | spectral/conv |
|---|---|---|---|---|---|
| 6 000 | 20–90 | 46 | 2.12 s | 1.47 s | **0.69×** |
| 20 000 | 20–130 | 112 | 5.04 s | 4.13 s | **0.82×** |

Spectral is faster than conventional at high `P`, and the margin grows with the scatterer
count (more bins × elements = more per-element overhead removed by the fused kernel). Binned
spectral reproduces the single-window spectral RF exactly (correlation 1.0); complex64 vs
complex128 accumulation agrees to ~2·10⁻⁷ (the factored form). Before this session's three
changes the same spectral path was ~1.3× *slower* than conventional at `P=20 000`.

---

## 7. Attenuation — the regime that flips the verdict toward paired

Power-law attenuation `H_att(ω, d)` depends on propagation distance. The **spectral** path
folds it **per patch** (each one-way patch term carries its own patch-to-point distance), so
the TX×RX product gives a true per-path round trip at no extra asymptotic cost — the fast
path supports attenuation directly. The **conventional** path applies it per scatterer-center
(one distance per scatterer/element), an approximation. **Paired** carries the exact per-path
form for free via a depth-binned kernel family `w_d = I⁴ v_pe ⊛ h_att(d)`: the pair weight
becomes a table lookup `w_{bin(d_ij)}`, where `d_ij` is the true geometric two-way TOF of the
pair — `O(PE·M_tx·M_Erx·len(w))` as plain paired. This is the unique regime where enumeration
is the *correct* method, not a fallback: the per-path weight depends on both `i` and `j`, so
it is **not rank-1 separable** and the FFT/spectral product structurally cannot factor it.
Niche: per-path attenuated PSF / monoelement / near-field focused.

---

## 8. Implementation status & remaining optimizations

**Implemented this session:**
- **Spectral formulation** (`compute_oneway_spectrum_band`, `compute_twoway_spectrum_summed`)
  in cancellation-free **factored-sin** form — closed-form one-way spectra, no forward FFT,
  float32/complex64-safe.
- **Depth-binning of the spectral path** (`_rf_spectral_binned`) with a **floor-aware,
  shared** `_auto_depth_bins` (bins until `nfft_bin` hits the `next_pow2(L)` floor; same rule
  for conventional and spectral).
- **Fused multi-element two-way kernel** (`_twoway_summed_points`): `Σ_TX` built once per
  scatterer and reused across all RX elements; no `(P, N_band)` intermediates; one launch per
  bin. Makes binned spectral beat conventional at high `P` (§6).
- `method="auto"` router: spectral (band-limited) / conventional (wideband) / paired (few
  patches).

**Remaining opportunities (not yet done), by leverage:**
1. **Far-field element collapse (Dirichlet).** In the factored form each patch is
   `envelope(ω)·e^{-jω t_c}`; for subpatches on a regular grid the phasor sum is a geometric
   series → product of two Dirichlet kernels, collapsing the per-element patch count. Valid
   per-axis where the *group* extent satisfies the Fraunhofer bound `L ≫ D²/λ` (holds laterally
   at imaging depths; fails in elevation for tall elements until deep field). Adaptive,
   per-axis; removes the `M` factor from the build where it applies.
2. **GPU port of the spectral kernel.** The `P×M×N_band` factored-sin loop is memory-light,
   phasor-recurrence, complex64-friendly — ideal for CUDA/cupy; 10–100× ceiling at production
   channel/frame counts.
3. **Working-rate decimation.** Run the internal grid at `fs' ≈ 4·BW` instead of decimating
   after; shrinks `nfft`, `N_band`, `len(w)` for all paths (spectral is analytic in ω, so
   nearly free).
4. **Minimal analytic ω grid.** Spectral can use any ω grid; a sub-band grid + chirp-Z back to
   the sample lattice would cut `N_band` further (needs the CZT resample step).
5. **Per-path attenuated paired kernel** (§7) — the depth-binned `w_d` family, gated behind the
   router for small-`M`/reference runs.

**Invariants** (must stay true): all formulations bit-comparable to Field II `calc_scat`
(corr ~1.0) on the no-attenuation path; `coords["t0"]` beam-axis referencing unchanged;
public `pulse_echo_rf` / `sequence_rf` / `synthetic_aperture_rf` / `scan_focusline`
signatures unchanged (`method` is an internal dispatch flag).
