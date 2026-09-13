---
icon: lucide/waves
---

# SIR kernels

The two building blocks every simulator uses: the spatial impulse response of a set of
rectangular patches, either **sampled in time** or **in closed form in frequency**. They
describe the same far-field trapezoid and are linked by `rfft(h[n]) ≈ fs·H(ω)`, so
`Emission` and `Reception` can use either (`method=`) and add their transfer functions
(excitation, attenuation, obliquity) in one frequency domain.

- `compute_h_sir` — `h(r, t)` on a time grid (FST or SDI kernel).
- `compute_h_sir_spectrum` — `H(r, ω)` at any frequencies: per patch
  `A/(2πl)·D(θ)·sinc(ωΔt1/2)·sinc(ωΔt2/2)·e^{-jω(t_c−t0)}`, exact, no sampling.

::: esdiva.hsir.compute_h_sir_spectrum

::: esdiva.hsir.compute_h_sir
