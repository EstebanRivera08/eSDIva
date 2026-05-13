---
paths:
  - "src/pyfield/h_sir/**"
  - "src/pyfield/psimulation/**"
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

## 5. Performance: SDI vs Naive

Per field point:
- Naive: O(M * avg_dk)  where avg_dk = average trapezoid width in samples
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

## 8. Attenuation in SIR Simulations

**Critical constraint**: Full dispersive attenuation inside SIR integral is
theoretically correct but computationally intractable. Each aperture point has
different propagation distance → each ray needs different attenuation kernel →
breaks SIR closed-form solution.

**Full model** (correct physics, not tractable):

    h_att(t, r) = integral_S a(t - tau, |r + r1|) * delta(tau - |r + r1|/c) / |r + r1| dS dtau

Each ray from aperture surface S has its own attenuation kernel a(t, |r|).

**Attenuation transfer function** (homogeneous medium):

    |A(f, |r|)| = exp(-alpha * |r|) * exp(-beta * (f - f0) * |r|)

- alpha: frequency-independent amplitude decay
- beta: frequency-dependent decay slope (dB/MHz/cm → Np/m/Hz conversion needed)
- f0: center frequency

Split into frequency-independent term exp(-alpha * |r|) and frequency-dependent
term exp(-beta * (f - f0) * |r|).

**Phase and causality**: Linear phase (Kak & Dines) → non-causal.
Minimum-phase model (Gurumurthy & Arthur):

    A(f, |r|) = exp(-alpha |r|) exp(-beta(f-f0)|r|)
                * exp(-j2pi f (tau_b + tau_m beta/pi^2)|r|)
                * exp(j (2f/pi) beta |r| ln(2pi f))

Gives causal dispersive attenuation impulse response.

**Far-field approximation**: When field point far from aperture, all rays arrive
at nearly same time with similar propagation distance:

    h_att(t, r) ≈ h(t, r) *_t a(t, |r|)

Simple temporal convolution of SIR with single attenuation kernel.

**Mean-distance approximation**: Compute mean propagation distance |r_mid| over
aperture, use single attenuation kernel a_mid(t) for all rays. Non-stationary
convolution approximated as stationary. Accuracy depends on SIR duration,
aperture curvature, attenuation strength.

**Practical rule** (empirically validated, Jensen): For typical clinical parameters
(0.5 dB/MHz/cm, concave transducer, focal depths 60–100 mm), applying attenuation
to 1-D pulse only and keeping SIR unchanged reproduces PSF almost as well as
full model.

**When approximation breaks down**:
- Strong attenuation + large aperture (rays differ significantly in path length)
- Near-field of large curved transducers
- Heterogeneous media (attenuation varies spatially)
- Nonlinear propagation → requires k-Wave, FDTD, or pseudo-spectral solvers
