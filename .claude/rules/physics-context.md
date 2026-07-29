---
paths:
  - "src/pyfield/hsir/**"
  - "src/pyfield/emission/**"
  - "src/pyfield/reception/**"
  - "src/pyfield/transducers/**"
---

# Physics Context — SIR/SDI Theory

Reference for understanding physical meaning of code changes.
Based on Stepanishen (1971), Jensen (1992), and Rivera SDI method (2026).

## 1. Spatial Impulse Response (SIR)

Pressure from vibrating aperture S at field point r_p:

    p_m(r_p, t) = rho_0 * dv_n/dt * h_m(r_p, t)     [convolution in time]

Where h_m is the SIR from patch m to point p:

    h_m(r_p, t) = (1/2pi) * integral_S [ delta(t - |r_m - r_p|/c_0) / |r_m - r_p| ] dS

Physical meaning: sum of spherical wavelets from each aperture element at retarded time.
Equivalent view (reciprocity): backward-projected sphere from field point intersecting aperture.

## 2. Far-Field Trapezoidal SIR (Rectangular Patch)

For patch m with dimensions (w_mx, w_my), center r_m, field point r_p:
- Distance: l = |r_p - r_m|
- Unit vector: u = (u_x, u_y, u_z) = (r_p - r_m) / l

Trapezoid parameters:
- dt1 = min(w_mx*|u_x|, w_my*|u_y|) / c_0     (shorter side crossing time)
- dt2 = max(w_mx*|u_x|, w_my*|u_y|) / c_0     (longer side crossing time)
- t1 = l/c_0 - (dt1 + dt2)/2                    (first corner TOF)
- t2 = t1 + dt1,  t3 = t1 + dt2,  t4 = t1 + dt1 + dt2
- h_max = w_mx * w_my / (2*pi * dt2 * l)        (plateau amplitude)
- slope = h_max / dt1

Piecewise SIR:
- Rising:  s*(t - t1)    for t1 < t < t2
- Plateau: h_max         for t2 < t < t3
- Falling: s*(t4 - t)    for t3 < t < t4
- Zero elsewhere

**Area invariant**: integral(h_m) = w_mx * w_my / (2*pi*l)

**Far-field validity**: w << sqrt(4*l*c_0/f)  — this is why transducers must subdivide
into small rectangular patches.

## 3. SDI Method (Sparse Delta Integration)

Key insight: 2nd derivative of trapezoid = 4 weighted Dirac deltas.

    d2h/dt2 = s * [delta(t-t1) - delta(t-t2) - delta(t-t3) + delta(t-t4)]

Recover trapezoid by double integration:

    h_m(r_p, t) = integral integral [ s * delta_comb(t'') ] dt'' dt'

Discrete: 8 sample points per trapezoid (2 per corner time) instead of filling
all samples in [t1, t4].

## 4. Transducer = Sum of M Patches

    h_tx(r_p, t) = sum_{m=1}^{M} a_m * h_m(r_p, t - tau_m)

Where a_m = apodization, tau_m = delay for patch m.

SDI form (linearity of integration):

    h_tx(r_p, t) = integral integral [ sum_m a_m * s_m * delta_comb(t'' - tau_m) ] dt'' dt'

**Why rectangular patches**: SIR has closed-form trapezoidal solution only for
rectangles. All transducer geometries (circular, curved, arbitrary) must be decomposed
into rectangular patches for this method to work.

## 5. Performance: SDI vs FST

Per field point:
- FST: O(M * avg_dk)  where avg_dk = average trapezoid width in samples
- SDI:   O(8M + 2T)     where T = total time samples

SDI wins when: avg_dk >> 8 + 2T/M

Critical patch size: w_c = 8*c_0/f_s
SDI only faster when w > w_c.

Average trapezoid width (hemispherical distribution): avg_dk ~ f_s * w / c_0

Heuristic: SDI better for large apertures (A >> 2*w*l_max) with patches above w_c.

## 6. Receive SIR and Acoustic Reciprocity

**Key principle**: By Green's function reciprocity, h(r_m → r_p, t) = h(r_p → r_m, t).

Transmit SIR: patch m is source, field point r_p is receiver.
Receive SIR: scatterer at r_p is source, patch m on receive transducer is receiver.

By reciprocity, the receive SIR is computed identically to transmit: treat the receive
transducer patches as sources and the scatterer position as the field point. **Same
h_sir function, same code path.** No separate receive SIR implementation needed.

    h_rx(r_scatterer, t) = h_sir(receive_patches → r_scatterer)
                         = same computation as h_sir(transmit_patches → r_field_point)

This means: for pulse-echo, compute h_tx with TX geometry and h_rx with RX geometry,
both using the same SIR engine, both treating scatterer positions as field points.

## 7. Pulse-Echo and Scattering Model

Medium perturbations (Born approximation):
- rho(r) = rho_0 + d_rho(r),  c(r) = c_0 + d_c(r)
- Scattering function: f_m(r) = d_rho/rho_0 - 2*d_c/c_0

Pulse-echo SIR:

    h_pe(r, t) = h_tx(r, t) *_t h_rx(r, t)

Where h_tx and h_rx are computed via same SIR engine (section 6 reciprocity).

Received signal:

    p_r(t) = v_pe(t) * f_m(r) *_r h_pe(r, t)

Where:
- v_pe = (rho_0 / 2*c_0^2) * E_m * d3v/dt3      (pulse-echo waveform)
- E_m(t) = receive electro-mechanical transfer function

Alternative form (derivatives on SIRs, used in SDI):

    p_r(t) = (rho_0/2c_0^2) * f_m(r) *_r [ (E_m*v) *_t (dh_tx/dt *_t d2h_rx/dt2) ]

## Symbol → Code Mapping

| Symbol | Code variable | Meaning |
|--------|--------------|---------|
| c_0 | `c` | Speed of sound (m/s) |
| rho_0 | `rho` | Medium density (kg/m^3) |
| f_s | `fs` | Sampling frequency (Hz) |
| w_mx, w_my | `wu`, `wv` | Patch dimensions (m) |
| l | `dist` / `l_mp` | Patch-to-point distance (m) |
| u_x, u_y, u_z | `ux`, `uy`, `uz` | Unit direction components |
| t1..t4 | `t1`..`t4` | Corner times-of-flight (s) |
| h_max | `h_max` | Trapezoid plateau height |
| s | `slope` | Trapezoid slope = h_max/dt1 |
| a_m | apodization array | Per-patch weight |
| tau_m | delay array | Per-patch delay (s) |
| M | `n_patches` | Total number of patches |

## 8. Emitted Pressure Field

For pulsed excitation v_n(t) = delta(t), emitted pressure ∝ SIR:

    p_e,delta(r, t) ∝ h(r, t)

Monochromatic pressure amplitude at center frequency omega_c:

    p_e,wc(r) = |P_e(r, omega_c)| ∝ |H(r, omega_c)|

Spatial distribution determined by magnitude of SIR Fourier transform at omega_c.

For arbitrary excitation (same or per-element):

    p_e(r, t) = rho_0 * v_n(t) *_t dh(r, t)/dt

**SDI benefit**: truncate after first integration step (skip one cumsum).
Convolution done in frequency domain. Additional transfer functions
(attenuation, bandwidth, dispersion) multiply in same domain. Inverse FFT
recovers time-domain pressure.

## 9. Receive Signals (Born Approximation)

Received pressure at element position r_m:

    p_r(r_m, t) = E_e(t) *_t integral_S p_s(r_p, t) dS

Compact form (Angelsen 1980, Jensen 1991):

    p_r(r_m, t) = v_pe(t) *_t f_m(r_p) *_r h_pe(r_m,p, t)

Where:
- f_m(r) = d_rho/rho_0 - 2*d_c/c_0           (scattering function)
- h_pe(r, t) = d2/dt2 [h_tx(r, t) *_t h_rx(r, t)]  (pulse-echo SIR)
- v_pe(t) = (rho_0 / 2c_0^2) * E_e(t) *_t dv/dt     (pulse-echo waveform)

Redistributing derivatives onto SIRs (associativity of convolution):

    p_r(r_m, t) = (rho_0/2c_0^2) * f_m(r_p) *_r
                  [(E_m * v) *_t (dh_tx/dt *_t d2h_rx/dt2)]

**SDI benefit for receive**: first- and second-order temporal derivatives of SIR
mean SDI can be truncated before integration stage. Saves O(T) operations per
truncation, where T = temporal sampling length.

## 9.1 PE SDI: three ways to evaluate the same pulse-echo RF

The pulse-echo RF of one scatterer is p_pe = v_pe *_t h_tx *_t h_rx. Each one-way SIR is
a trapezoid whose 2nd derivative is four corner deltas (d2h = sum of 4 signed Diracs,
signs +,-,-,+, each scaled by the rising slope). PyField evaluates p_pe three ways; all
agree to corr ~1.0 with each other and Field II. Reception's `method=` picks one:

**fst / sdi / auto (conventional)** — sample both one-way SIRs (place corner deltas,
double-cumsum to a trapezoid) and FFT-convolve. SIR build is linear in patch count M; the
convolution is M-independent. (`ReceptionConventional`, with a depth-bin fast path; the
string names its SIR-sampling kernel.)

**paired** — convolve the two corner-delta trains *analytically* (deltas *_t deltas =
deltas), giving the two-way train

    Δδ_pe = d2h_tx *_t d2h_rx = 16 Dirac deltas per (m_e, m_r) patch pair

at t_event = t_e_corner + t_r_corner (4 TX corners × 4 RX corners). Push the four
integrations onto the excitation once, w = I4 v_pe, then for each of a pair's 16 corner
events splat a shifted, scaled copy of w: p_pe = Σ_ij a_i a_j w(t − τ_i − τ_j). No FFT,
no cumsum — the output is the RF directly. Cost ∝ M_tx·M_rx·len(w), so it is the exact
reference path for compact apertures (a PSF, a monoelement). (`compute_pe_complete`.)

**spectral** — never form the pairs. The Fourier transform of one aperture's corner
train is closed form (a sum of four phasors per patch),

    Σ(ω) = Σ_m slope_m [ e^{-jω t1} − e^{-jω t2} − e^{-jω t3} + e^{-jω t4} ]

so the two-way SIR spectrum is the PRODUCT of the one-way spectra (convolution ⇒
multiply), Σ_TX·Σ_RX = F{Δδ_pe}, and h_tx *_t h_rx = ÷(jω)^4 · Σ_TX·Σ_RX. This builds no
time-domain SIR and does NO forward FFT — cost is linear in patch count (M_tx + M_rx),
and exact (no time sampling, no interpolation). Because the received signal is
band-limited by the excitation/IR, Σ is evaluated only on the in-band bins
(N_band ≪ N_freq). For the summed RF, `compute_twoway_spectrum_summed` builds the TX
spectrum once per scatterer and reuses it across every RX element, summing Σ_TX·Σ_RX over
scatterers in one fused pass; for the per-scatterer PSF, `compute_oneway_spectrum_band`
builds one element's Σ at a time. Per-patch one-way attenuation (§10) is multiplied into
each patch phasor for free, using the patch-to-point distance — the TX×RX product then
carries the true round-trip loss, which conventional cannot do cheaply.

**I4 scaling.** Δδ_pe / Σ hold delta *areas* (no width). ÷(jω) is a continuous
integrator weighting each sample by dt, under-counting by fs=1/dt, so `inv_jw_pow`
carries one ×fs. Doing all four integrations in Fourier (vs a time-domain cumsum) carries
zero group delay → sample-aligned with conventional, and avoids float32 cumsum
cancellation.

**Excitation.** v_pe = (rho_0 / 2c_0^2) · (E_m * v). No explicit derivative on v — the
physical d3v/dt3 is carried by the band-limited excitation/IR chain (same as Field II).
This differs from Emission, where the chain has an explicit dv/dt.

Code: `compute_pe_complete` (paired) / `compute_oneway_spectrum_band` +
`compute_twoway_spectrum_summed` (spectral) in `transducer_sir_pe_sdi.py`. The
conventional path delegates to `ReceptionConventional` (`farfield_rect_patch.compute_h_sir`).

## 9.2 Pulse-centre lag — a beamforming correction, NOT part of the RF

The reception RF is the raw echo referenced to the **geometric** round-trip time
`t0` (nearest-patch arrival). But the recorded echo is the geometric SIR convolved
with the band-limited two-way pulse `exc ⊛ ir_tx ⊛ ir_rx`, whose envelope peaks
about **half a pulse length after** the geometric arrival. So a delay-and-sum that
reads the geometric time `t_tx + t_rx − t0` lands the point-spread function ~0.5–1
mm too deep (≈1 µs for a 2-cycle 5 MHz pulse). Every beamformer must instead read
the sample at `t_geom + coords["pulse_center_lag_s"]`.

This lag is **stored in `coords["pulse_center_lag_s"]`** by the reception
simulators (`ReceptionBase._pulse_center_lag_s`, from the drive + element impulse
responses), NOT applied to the RF samples. Rationale: the RF is the physical echo;
baking the lag into it would (a) misrepresent the raw signal and (b) double-count
in the built-in beamformers, which already add it. The DAS beamformers
(`das_volume`, `das_rca_volume`) default `t_offset_s=None` →
auto-read the lag from `coords`; pass a float to override, `0.0` to disable.

**When writing a custom beamformer** (e.g. a torch/differentiable one), you MUST
add this term yourself: `idx = (t_tx + t_rx − t0 + coords["pulse_center_lag_s"])·fs`.
Forgetting it biases every PSF axially by a constant ~half-pulse depth. (Distinct
from the per-event TX time reference, which must be recovered from the event's own
delays — `t_ref = mean_e(τ_e ∓ |r_e − r_vs|/c)` — or the events desynchronise and
the compounded PSF splits into one ray per transmit.)

## 10. Attenuation in SIR Simulations

**Core approach**: Post-hoc frequency-domain transfer function. SIR stays lossless.

    P_att(r, f) = P_lossless(r, f) * H_att(f, d)

One complex multiply per field point per frequency bin. Fits into existing
freq-domain convolution step.

### Causal Power-Law Model (recommended)

General case (y != 1), from Szabo (1994), Holm (2019):

    H_att(omega, d) = exp(-alpha0 * |omega|^y * d)                    [absorption]
                    * exp(-j * alpha0 * |omega|^y * tan(y*pi/2) * d)  [K-K dispersion]

Special case (y = 1), O'Donnell (1981):

    H_att(omega, d) = exp(-alpha0 * |omega| * d)
                    * exp(-j * (2*alpha0/pi) * omega * ln(|omega|/omega0) * d)

Parameters: alpha0 [Np/m/Hz^y], y (tissue: 1.0–1.3), d [m], omega0 = 2*pi*f0.

Always use causal (both terms). Non-causal (amplitude-only, Jensen 1993) produces
acausal precursors. Causal correction cost = zero.

### Distance for H_att

Two options:
1. Per field point: d = |r| from transducer center (fast, approximate)
2. Per subaperture: d_i = |r - r_i| per patch (accurate near-field)

### Historical context

**Full model** (correct physics, not tractable): attenuation inside SIR integral,
each ray different kernel. Breaks closed-form solution.

**Far-field approximation** (Jensen): apply attenuation to 1-D pulse only.
For typical clinical parameters (0.5 dB/MHz/cm, 60–100 mm depth), reproduces
PSF almost as well as full model.

**When SIR insufficient** (need full-wave solvers):
- Nonlinear propagation
- Spatially heterogeneous attenuation
- Strongly scattering media (Born breaks down)

Key refs: Holm 2019 (theory), Kelly & McGough 2013/2022 (SIR-specific).
See attenuation.md for implementation rules.

## 11. Plane-Wave Steering Delays

    n = [sin(theta_x), sin(theta_y), sqrt(1 - sin^2(theta_x) - sin^2(theta_y))]
    d_e = element_centers @ n
    delays = (d_e - d_min) / c

Physical: the emitted plane wave travels along the unit normal `n`, so an element's
emit time is proportional to its projection `d_e = r_e·n` onto that direction. The
element with the **minimum** projection fires first (zero delay); larger projections
fire progressively later. Constraint: `sin^2(theta_x) + sin^2(theta_y) <= 1`.
