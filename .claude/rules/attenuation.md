---
paths:
  - "src/pyfield/attenuation/**"
  - "src/pyfield/emission/**"
  - "src/pyfield/reception/**"
  - "src/pyfield/hsir/**"
---

# Attenuation Implementation Rules

Guidelines for implementing attenuation in PyField SIR-based simulations.
See physics-context.md §8–10 for underlying theory.

## Core Principle

**Never modify the SIR computation to include attenuation.**
The SIR (h_sir) stays purely geometric. Attenuation applied as frequency-domain
transfer function multiplied onto the SIR spectrum:

    P_att(r, f) = P_lossless(r, f) * H_att(f, d)

One complex multiply per field point per frequency bin. Zero extra compute.

## Recommended Model: Causal Power-Law Transfer Function

**Best candidate for PyField.** Supported by Szabo (1994), Holm (2019),
Kelly & McGough (2013, 2021, 2022). Used in FOCUS, DiffUS.

Always use causal model (adds Kramers-Kronig dispersion phase). Cost = zero.
Accuracy = better than non-causal (Field II, SIMUS).

### General case (y != 1):

    H_att(omega, d) = exp(-alpha0 * |omega|^y * d)
                    * exp(-j * alpha0 * |omega|^y * tan(y*pi/2) * d)

First term = absorption (amplitude decay). Second term = K-K dispersion (phase).

### Special case (y = 1):

tan(pi/2) diverges. Use logarithmic dispersion (O'Donnell 1981):

    H_att(omega, d) = exp(-alpha0 * |omega| * d)
                    * exp(-j * (2*alpha0/pi) * omega * ln(|omega|/omega0) * d)

### Parameters:
- `alpha0`: attenuation coefficient [Np/m/Hz^y]
- `y`: power-law exponent (tissue: 1.0–1.3)
- `d`: propagation distance [m]
- `omega0`: reference angular frequency = 2*pi*f0 [rad/s]
- `f0`: reference frequency (typically transducer center frequency) [Hz]

### Phase velocity (dispersive):

General (y != 1):

    1/c_p(omega) = 1/c_0 + alpha0 * tan(y*pi/2) * (|omega|^(y-1) - omega0^(y-1))

y = 1:

    1/c_p(omega) = 1/c_0 - (2*alpha0/pi) * ln(|omega|/omega0)

## Distance Options

Two approaches for propagation distance `d`:

1. **Per field point**: `d = |r|` from transducer center. Fast, approximate.
2. **Per subaperture**: `d_i = |r - r_i|` from each patch centroid. Apply H_att
   before summation over patches. More accurate near-field.

Option 1 sufficient for most cases. Option 2 needed for large aperture + strong
attenuation + near-field.

## Integration in SDI Pipeline

After computing H(r, omega) via SDI:

    H_sir = compute_sir_frequency_domain(...)     # (N_points, N_freq)
    H_att = causal_attenuation_tf(freq, dist, alpha0, y, f0)  # (N_points, N_freq)
    P = rho0 * dV * H_sir * H_att                 # one extra multiply
    p_t = irfft(P)                                 # back to time domain

Fits naturally into existing freq-domain convolution step where excitation
is already multiplied.

## Attenuation Parameter Convention

- User-facing: `alpha0` in dB/(MHz^y·cm) — matches clinical/literature convention.
- Internal conversion to Np/m/Hz^y for computation:
  `alpha0_neper = alpha0_dB * 100 / (20 * log10(e) * 1e6^y)`
- Store `alpha0` (dB units) on PyField/medium; convert at computation time.

## Frequency-Independent Shortcut

For quick amplitude-only decay (no spectral distortion):

    p_att(r) = p(r) * exp(-alpha_neper * |r|)

where `alpha_neper = alpha0 * f0 * 100 / (20 * log10(e))` at center frequency f0.

Use only when spectral content unimportant (e.g., monochromatic CW fields).

## What NOT To Do

1. Do not add attenuation terms inside `farfield_rect_patch.py` or SIR kernels.
2. Do not assume single attenuation value for all field points — distance-dependent.
3. Do not use non-causal (amplitude-only) model. Add K-K dispersion phase.
   Non-causal produces acausal precursors (Kelly & McGough 2022).
4. Do not apply attenuation to SIR directly unless implementing far-field
   approximation explicitly and documenting validity assumptions.

## When SIR + Attenuation TF Is Insufficient

If simulation requires any of these, SIR method itself is insufficient:
- Nonlinear propagation
- Spatially heterogeneous attenuation (varying alpha0 along path)
- Strongly scattering media (Born approximation breaks down)

Recommend k-Wave, FDTD, or pseudo-spectral solvers instead.

**Note**: Frequency-power-law attenuation (f^y where y = 1.0–1.3) IS supported
by the causal TF approach above. This does NOT require full-wave solvers.

## Key References

| Ref | What | Year |
|-----|------|------|
| Jensen 1993 | Non-causal amplitude-only TF (Field II) | 1993 |
| Szabo 1994 | Time-domain wave eq for power-law media | 1994 |
| O'Donnell 1981 | K-K relationship, y=1 logarithmic dispersion | 1981 |
| Holm 2019 | *Waves with Power-Law Attenuation* (book, best single theory ref) | 2019 |
| Kelly & McGough 2013 | Causal impulse response for circular sources | 2013 |
| Kelly & McGough 2022 | Causal vs noncausal Green's fn comparison | 2022 |
| Garcia 2022 | SIMUS non-causal implementation | 2022 |
