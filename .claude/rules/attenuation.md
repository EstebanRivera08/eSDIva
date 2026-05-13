---
paths:
  - "src/pyfield/psimulation/**"
  - "src/pyfield/h_sir/**"
---

# Attenuation Implementation Rules

Guidelines for implementing attenuation in PyField SIR-based simulations.
See physics-context.md §8 for underlying theory.

## Core Principle

**Never modify the SIR computation to include attenuation.**
The SIR (h_sir) stays purely geometric. Attenuation is applied externally,
to the excitation pulse or to the resulting RF signal.

## Standard Approach: Attenuate the Excitation

    v_att(t, |r|) = v(t) *_t a(t, |r|)

- Compute attenuation impulse response a(t, |r|) from transfer function A(f, |r|).
- Convolve with excitation pulse before or after SIR convolution.
- Distance |r| = propagation distance from transducer to field point.

This is distance-dependent: each field point gets a differently attenuated pulse.

## Attenuation Parameter Convention

- User-facing: `alpha0` in dB/(MHz·cm) — matches clinical/literature convention.
- Internal conversion to Np/(Hz·m) for computation:
  `beta_neper = alpha0 * 100 / (20 * log10(e) * 1e6)`
- Store `alpha0` on PyField/medium; convert at computation time.

## Frequency-Independent Shortcut

For quick amplitude-only decay (no spectral distortion):

    p_att(r) = p(r) * exp(-alpha_neper * |r|)

where `alpha_neper = alpha0 * f0 * 100 / (20 * log10(e))` at center frequency f0.

Use only when spectral content unimportant (e.g., monochromatic CW fields).

## What NOT To Do

1. Do not add attenuation terms inside `farfield_rect_patch.py` or SIR kernels.
2. Do not assume single attenuation value for all field points — it is depth-dependent.
3. Do not use linear-phase attenuation models — they are non-causal.
   Use minimum-phase (Gurumurthy & Arthur) if phase matters.
4. Do not apply attenuation to SIR directly unless implementing far-field
   approximation explicitly and documenting validity assumptions.

## When SIR + Pulse Attenuation Is Insufficient

If simulation requires any of these, SIR method itself is insufficient:
- Nonlinear propagation
- Frequency-power-law attenuation (e.g., tissue-realistic f^1.1 law)
- Spatially heterogeneous attenuation
- Full dispersive modeling

Recommend k-Wave, FDTD, or pseudo-spectral solvers instead. Do not attempt
to hack these into the SIR framework.
