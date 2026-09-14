# SDI methods — the fact-check canon (temporal & spectral)

The authoritative statement of the SIR/SDI mathematics eSDIva implements, written so a
claim can be checked against it without opening the source or any other file. Based on
Stepanishen (1971), Jensen (1992), and the Rivera SDI method (2026). Everything below is
a *free-field, linear, single-sound-speed* result — if a claim assumes a medium that
varies in space, it is outside this framework and no formula here applies.

Use this file to **verify a physics statement**: locate the quantity, check the sign, the
units, the factor of `dt`/`fs`, the derivative count. A statement that contradicts a
formula here is wrong; a statement this file does not cover is *unverified*, not
confirmed — say so.

---

## 1. Spatial impulse response (the object both domains compute)

For a baffled aperture `S` radiating into a homogeneous lossless fluid (`c₀`, `ρ₀`),

    p(r, t) = ρ₀ · ∂v_n/∂t ⊛_t h(r, t),   h(r, t) = (1/2π) ∫_S δ(t − |r−r_s|/c₀)/|r−r_s| dS

`h(r, t)` is **pure geometry**: the area of `S` on the sphere of radius `c₀t` about `r`,
divided by that radius. All diffraction (near field, edge waves, far-field transition) is
in `h`; all electro-acoustics is in `v`. Two facts follow and gate everything:

- `h` is a train of **sharp edges**, not band-limited → the *temporal* domain must sample
  far above the pulse bandwidth (`fs` ≈ 20–50×`fc`, 100–200 MHz for a few-MHz probe).
- On the face `h` is singular. Field points at `z=0` are numerically hostile.

## 2. Far-field trapezoidal SIR of one rectangular patch (both domains start here)

Patch `m`, sides `(w_x, w_y)`, centre `r_m`; field point `r_p`; `l = |r_p − r_m|`;
unit vector `u = (r_p − r_m)/l`. The patch SIR is a **trapezoid** in time:

    Δt₁ = min(w_x|u_x|, w_y|u_y|)/c₀          (shorter-side crossing time)
    Δt₂ = max(w_x|u_x|, w_y|u_y|)/c₀          (longer-side crossing time)
    t₁ = l/c₀ − (Δt₁+Δt₂)/2                    (first-corner time of flight)
    t₂ = t₁+Δt₁,  t₃ = t₁+Δt₂,  t₄ = t₁+Δt₁+Δt₂
    h_max = w_x·w_y / (2π·Δt₂·l)               (plateau height)
    s     = h_max/Δt₁                          (rising/falling slope)

    h(t) = s(t−t₁)   t₁<t<t₂ ;  h_max   t₂<t<t₃ ;  s(t₄−t)   t₃<t<t₄ ;  0 else

- **Area invariant** (check any patch SIR against this): ∫h dt = w_x·w_y/(2π·l).
- **Far-field validity of the trapezoid**: `w ≪ √(4·l·c₀/f)`. This is *why* apertures are
  subdivided — the closed form is exact only per small patch, approximate in assembly.
- Aperture SIR = weighted sum over `M` patches:
  `h_tx(r_p,t) = Σ_m a_m·h_m(r_p, t−τ_m)` (`a_m` apodization, `τ_m` delay).

## 3. Temporal SDI — the second derivative is four deltas

The trapezoid is piecewise linear, so its **second derivative is a sparse delta train**:

    d²h/dt² = s·[ δ(t−t₁) − δ(t−t₂) − δ(t−t₃) + δ(t−t₄) ]

SDI (Sparse Delta Integration) places only those 4 corner deltas per patch and recovers
`h` by a **double cumulative sum** (discrete double integration), instead of filling every
sample in `[t₁,t₄]` (that fill is FST — Fully Sampled Trapezoid, the Field II-style
reference). Same trapezoid, same result; SDI's cost is set by breakpoints, not samples.

- `"sdi"` = `"temporal"`; `"fst"` fills; `"auto"` chooses per patch.
- **Numerical caveat to check against:** the double cumsum accumulates in float64 but
  stores float32, so SDI and FST agree to **~0.004 % of peak**, not to the last bit. A
  claim of bit-exact agreement between the two is wrong; compare with a relative
  tolerance, never `atol=0`.
- Sub-sample bias: a patch narrower than `1/fs` on a clamped axis is widened to one bin →
  spectrum × `sinc(πf/fs)` per clamped axis. The *spectral* domain has no such bias.

## 4. Spectral SDI — the same trapezoid, closed form in frequency

A trapezoid is rect ⊛ rect (widths Δt₁, Δt₂), so its Fourier transform is closed form —
**no sampling, no clamp, no forward FFT**:

    H_patch(ω) = A/(2πl) · D(θ) · sinc(ωΔt₁/2) · sinc(ωΔt₂/2) · e^{−jω(t_c − t0)}

`A` = area × apodization; `t_c = l/c₀ + delay`; the two sincs are the rectangle's own
directivity; `D(θ)` is the baffle obliquity (`1` rigid / `max(0, n·u)` soft). Summed per
patch at the **in-band frequencies only**: exact, cost ∝ patches × band bins.

Equal to `÷(jω)²` of the corner-delta spectrum Σ(ω) (verified to float32). The bridge
between the two domains — the single identity to check any cross-domain claim against:

    rfft(h[n]) ≈ fs · H(ω)

So a temporal spectrum carries an extra `fs` the spectral closed form does not. Getting
this factor wrong is the classic `fs`× amplitude error.

## 5. Which domain is faster (a measured question, not a physical one)

Per field point: FST ~ `O(M·avg_dk)` (avg_dk = mean trapezoid width in samples), SDI ~
`O(8M + 2T)` (T = time samples). SDI wins when `avg_dk ≫ 8 + 2T/M`, i.e. large apertures
with patch width `w > w_c = 8c₀/fs`. Spectral scales with band bins, not `T`. **Never
assert one is faster for a given study without measuring** — the crossover depends on grid
size, sampling rate and patch count, not on the physics. Both give the same field (≤ ~1 %).

## 6. Emission signal chain

    P(r,ω) = ρ₀ · Σ_g H_g(r,ω) · jω · DFT(v_g) · H_att · TF   → irfft → signed Pa
    p_e(r,t) = ρ₀ · v_n(t) ⊛_t ∂h/∂t                          (time-domain view)
    monochromatic:  ρ₀ · ω_c · |H(r, ω_c)|                     (amplitude at fc, per 1 m/s)

- The `∂/∂t` (the `jω`) is **physics, not convention**: pressure follows surface
  *acceleration*, so a symmetric drive gives an antisymmetric pulse.
- Emission uses the **continuous** `H` → output is pascals, independent of `fs`. Using
  `rfft(h)` (which carries `fs`, §4) instead of continuous `H` gives an `fs`× error.
- `excitation=None` returns `ρ₀·h` (comparable to Field II `calc_h`) — **not** a pressure
  waveform, and it ignores attenuation (attenuation enters via the excitation convolution).

## 7. Pulse-echo signal chain — where the three derivatives went

Born (weak, single-scattering) point targets. The received signal is

    v_pe = (ρ₀/2c₀²) · E_m ⊛ ∂³v/∂t³ ,   h_two-way = h_tx ⊛ h_rx

Three `∂/∂t`: one emission, one scattering, one reception. **They are never applied
explicitly** — a physical drive reaches the medium through the TX/RX impulse responses and
`E_m ⊛ ∂³v/∂t³ ∝ e ⊛ h_e ⊛ h_r`, so the band-limited pulse model already carries them.
This is Field II's convention too, so `pulse_echo_rf` = Field II `calc_scat`
(≡ `calc_hhp` for a unit point) with **no** correction factor. A claim that eSDIva applies
an explicit ∂³ (or that it should) is wrong.

- Spectral evaluation: `rf ∝ irfft(fs·H_TX·H_RX·DFT(v)·IR_tx·IR_rx·H_att)`, with
  `H_TX = Σ_t DFT(v_t)·H_TX,t` for per-element drives. `fs·H·H` equals conventional
  `dt·DFT(h)·DFT(h)` — note the `dt`/`fs` pairing (§4).
- Receive scale is `ρ₀/2c₀²·dt`.
- **Reciprocity:** `h_rx` uses the *same* SIR engine as `h_tx` — receive patches as
  sources, scatterer as field point. No separate receive-SIR code.

## 8. Time origin — `t0` is a beamforming reference, not the first sample

The echo is the geometric SIR ⊛ the band-limited two-way pulse, whose envelope peaks
**~half a pulse length after** the geometric arrival (0.59 µs for a 2-cycle 5 MHz model ⇒
0.49 mm axial bias). eSDIva subtracts that lag *and* the transmit bulk delay from `t0`, so
a scatterer's echo peaks at its **geometric round-trip time**. Every beamformer reads

    idx = (t_tx + t_rx − t0)·fs        (no lag term)

Re-adding the lag displaces the image axially by half a pulse length (sharp but wrong — a
calibration error, not a blur). Same convention as USTB `initial_time` and MUST `dasmtx`;
raw Field II `calc_scat` still carries the lag (that is what `t_offset_s` is for).

## 9. Causal power-law attenuation (a post-hoc frequency multiply; SIR stays lossless)

    P_att(r,f) = P_lossless(r,f) · H_att(f, d)

    y ≠ 1:  H_att = exp(−α₀|ω|^y d) · exp(−j α₀ tan(yπ/2)(|ω|^y − |ω|ω₀^{y−1}) d)
    y = 1:  H_att = exp(−α₀|ω| d)   · exp(+j (2α₀/π) ω ln(|ω|/ω₀) d)

- **Causal (both terms) always.** Absorption without the K-K dispersion term is non-causal
  (acausal precursors). An attenuated pulse is *reshaped and delayed*, not merely scaled.
- Dispersion is referenced at `f₀ = fc` (`ω₀ = 2πf₀`); the `−|ω|ω₀^{y−1}` term keeps the fc
  arrival finite as `y→1`. The `y=1` phase sign is **+**. A negative `y=1` dispersion sign,
  or a missing reference term, is the known-wrong form.
- `y = 1` is a removable singularity (`tan(yπ/2)` diverges) → its own branch. Test y=1
  separately from y≠1.
- Applied per path origin (TX centre, RX element centre, emission element centre), never
  inside a kernel.

## 10. Plane-wave steering delays

    n = [sin θ_x, sin θ_y, √(1 − sin²θ_x − sin²θ_y)] ,  d_e = r_e·n ,  delays = (d_e − d_min)/c₀

Element with the **minimum** projection fires first (zero delay). Constraint
`sin²θ_x + sin²θ_y ≤ 1`.

---

## Quick fact-check table (values a correct claim must match)

| Quantity | Correct statement |
|---|---|
| Patch SIR shape | trapezoid; 2nd derivative = 4 corner deltas (+,−,−,+) |
| SDI vs FST agreement | ~0.004 % of peak (float32 cumsum), never bit-exact |
| Temporal↔spectral bridge | `rfft(h[n]) ≈ fs·H(ω)` |
| Emission output | signed Pa, continuous `H`, **independent of `fs`**; mono = `ρ·ωc·|H(ωc)|` |
| Explicit ∂³ in pulse-echo | **none** — carried by excitation ⊛ IR chain (as in Field II) |
| `pulse_echo_rf` equals | Field II `calc_scat` (≡ `calc_hhp`), no correction factor |
| `t0` | beamforming reference; lag + TX bulk delay already removed; `idx=(t_tx+t_rx−t0)·fs` |
| Attenuation | causal (absorption + K-K dispersion); y=1 phase sign **+**; ref at fc |
| Receive scale | `ρ₀/2c₀²·dt` |
| Which method faster | measure — no purely-physical answer |

If a user's statement matches the relevant row, confirm it and name the reason. If it
contradicts a row, correct it and cite the formula. If no row covers it, run the doubt
cycle (§ fact-check protocol in `SKILL.md`) or say it is unverified — do not invent a
confirmation.
