# Pulse-Echo SDI: analytical-kernel reformulation — math & performance

Analysis of moving the SIR integrals onto the excitation in the pulse-echo SDI
formulation (`p_pe = (I⁴ v_pe) ⊛ Δδ`), and why it does **not** beat the conventional
FFT path for arrays. Companion to `ARCHITECTURE.md` →
"Pulse-Echo Post-Processing & Depth Binning".

## Notation

- `I` = time-integration operator; `I²`, `I⁴` = repeated integration.
- `δ` = Dirac delta; `⊛` = temporal convolution.
- `M_tx`, `M_rx` = number of TX / RX patches (for one RX element).
- `v_pe` = excitation/IR chain; `T` = SIR sample length; `L_w` = length of `I⁴ v_pe`.

## 1. Derivation (this is correct)

Single rectangular patch: the trapezoidal SIR has a piecewise-constant 2nd
derivative = 4 weighted Diracs at the corner times `τ_k`:

    h_m'' = Σ_{k=1..4} s_k δ(t − τ_k)        ⇒   h_m = I²[ Σ_k s_k δ(t − τ_k) ]

Summing over patches gives delta trains for each aperture:

    h_tx = I² D_tx ,   D_tx = Σ_{m,k} a_m s_{m,k} δ(t − τ_{m,k})   (4·M_tx deltas)
    h_rx = I² D_rx                                                 (4·M_rx deltas)

Two-way SIR (one TX aperture, one RX element):

    h_pe = h_tx ⊛ h_rx = (I² D_tx) ⊛ (I² D_rx) = I⁴ (D_tx ⊛ D_rx) = I⁴ Z

    Z = D_tx ⊛ D_rx = Σ_{i,j} g_i g_j δ( t − (τ_i + τ_j) )
    N_pair = (4·M_tx)(4·M_rx) = 16 · M_tx · M_rx        ← the delta product "Δδ"

Recorded RF:

    p_pe = v_pe ⊛ h_pe = v_pe ⊛ (I⁴ Z)

Integration commutes with convolution, `I(f ⊛ g) = (If) ⊛ g`, so push the four
integrals onto the excitation and define `w ≡ I⁴ v_pe` (precomputed once):

    p_pe = (I⁴ v_pe) ⊛ Z = Σ_{i,j} g_i g_j · w( t − τ_i − τ_j )      ✅ valid

This is the proposed analytical kernel: a sum of shifted, scaled copies of a single
fixed waveform `w`. The mathematics is exact.

## 2. Performance verdict — it is slower for arrays

Evaluating the sum = accumulate `N_pair` shifted copies of `w` (length `L_w`):

    cost ≈ O( N_pair · L_w ) = O( 16 · M_tx · M_rx · L_w )   per (scatterer, element)

Per (scatterer, element), compared with the existing kernels:

| Method | How `h_pe` is formed | Cost |
|---|---|---|
| **Analytical kernel** (this note) | expand `Z`, convolve each pair with `w` | `O(M_tx·M_rx · L_w)` |
| **`ReceptionSDI`** | place `Z` deltas (1 write each), **1 cumsum**, 1 FFT⊛`v_pe` | `O(M_tx·M_rx + T + nfft·log nfft)` |
| **`Reception`** (conventional) | build `D_tx`, `D_rx` separately, FFT-convolve | `O(M_tx + M_rx + nfft·log nfft)` |

Two independent reasons it loses:

1. **The `M_tx·M_rx` pair count is irreducible.** Forming the two-way SIR as a
   *product* of two delta trains has `16·M_tx·M_rx` pairs by construction. Reordering
   the integrals changes nothing here. This `O(M²)`-in-patches term is exactly why
   `ReceptionSDI` is slow for real arrays (Domino `M=1280` → ~1.6M pairs *per
   scatterer per element*).

2. **It de-optimises the integration.** `ReceptionSDI` applies all four integrals
   with **one cumsum** (`O(T)`, once). The analytical kernel instead spreads them
   onto `v_pe` and pays an `O(L_w)` convolution **per pair**, multiplying the already
   quadratic term by `L_w` (tens–hundreds). So it is strictly worse than
   `ReceptionSDI`.

**Key point:** the integrals are *not* the bottleneck (cumsum is `O(T)`). Forming the
delta **product** at all is. The user's observation that the delta formulation didn't
speed things up is explained by (1), not by the integrals.

## 3. What beats it (and Field II)

Do **not** multiply the delta trains. Build `h_tx` and `h_rx` **separately**
(`O(M)` each) and convolve them with **one FFT** (`O(T·log T)`) — conventional
`Reception`. For `M ≫ √(T·log T)` this wins by orders of magnitude; depth-binning
shrinks `T·log T` further (see `ARCHITECTURE.md`). This is why `Reception` +
depth-binning beats Field II `calc_scat_multi` (≈2× at `N_scat = 10⁴`) while
`ReceptionSDI` does not. No integral-reordering beats convolution-via-FFT for large
`M`, because the `M²` pair count of the product form is irreducible.

## 4. Where the analytical kernel *is* the right tool — small `M`

For **monoelement / few-patch** transducers, `N_pair = 16·M_tx·M_rx` is tiny, so
`O(M²·L_w)` is cheap and the form has two genuine advantages over the cumsum kernels:

- **FFT-free** — just shifted adds of `w`.
- **Exact** — no SDI sample-interpolation error and no float32 cumsum cancellation
  (the float64-accumulator hazard, gotcha #1 in `ARCHITECTURE.md`), because `w` is
  evaluated analytically at the exact shifted times.

`ReceptionSDI` already targets the small-`M` niche; the `w = I⁴ v_pe` accumulation
could be a cleaner, exact, cumsum-free variant there. For arrays (`M ~ 10³`) it is the
wrong regime — the `M²` wall dominates.

## Summary

| | Analytical kernel `(I⁴v_pe)⊛Z` | `ReceptionSDI` | `Reception` (FFT, binned) |
|---|---|---|---|
| Patch scaling | `O(M²·L_w)` | `O(M²)` | `O(M + T log T)` |
| FFT needed | no | yes (1) | yes |
| Exact (no cumsum/interp error) | **yes** | no | no |
| Best regime | tiny `M` (monoelement) | small `M` | **arrays (`M ≫ 1`)** |
