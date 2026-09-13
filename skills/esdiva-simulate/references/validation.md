# Validation — what has been verified, how, and what not to reintroduce

Every statement below was checked by a discriminating experiment (numbers are from those
runs, 2026-09). Quote them with the test that backs them; do not upgrade a hypothesis
into a finding without one.

## Guarantees pinned by the test suite

| Guarantee | Evidence / test |
|---|---|
| Closed-form `H(ω)` = continuous FT of the sampled trapezoid SIR | ≤ 1.1e-4 (off-axis), 8.9e-4 (on-axis) vs `dt·rfft(h)` at 1 GHz — `tests/unit/test_h_sir/test_sir_spectral.py` |
| Spectral ≡ temporal, every feature combination | Emission: monochromatic/transient × global/per-element × attenuation × rigid/soft; Reception: drive × attenuation × baffle (8 cases) — ≤ 2 % of peak, typically 0.3–1.4 % |
| Emission is signed and in pascals | On-axis far field of a small piston = `ρ·A/(2πz)·v'(t − z/c)` within 1–3 %, positive correlation > 0.99, 1/z spreading — `test_far_field_is_signed_rayleigh` |
| Soft baffle = `cosθ` per patch, zero behind | Ratio soft/rigid = 0.8000 for `cosθ = 0.8` — `test_soft_baffle_is_cos_theta_and_zero_behind` |
| Attenuation dispersion is physical | High frequencies arrive first (y = 0.5, 1, 1.1, 1.5), continuous across y = 1, zero dispersion phase at `f0 = fc` — `test_attenuation.py` |
| No silent numerical drift | 8 canonical scenarios (one per example family, both methods) pinned to `tests/regression/golden.npz`; regenerate with `just regen-golden` only after an intentional change, stated in the commit |
| Examples still run | `just test-examples` runs every numbered example headless (plotters off-screen, closed after one render) |

## Which SIR method is faster (measured, 64-element linear array)

Both give the same physics; speed depends on the problem.

| Emission (12.5k points) | spectral | temporal | default (`method=None`) |
|---|---|---|---|
| monochromatic | 0.30 s | 0.60 s | spectral |
| monochromatic + per-element attenuation | 1.2 s | 29 s | spectral |
| transient, global drive (± attenuation, TF, soft baffle) | 4.1–6.4 s | 2.9–5.0 s | temporal |
| transient, per-element drive or attenuation | 34–98 s | 80–217 s | spectral |

| Reception (20k scatterers) | spectral | temporal |
|---|---|---|
| global drive | 5.9 s | 35.5 s |
| + attenuation | 8.7 s | 914 s |
| per-element drive | 6.8 s | 353 s |
| PSF, 50 points | 0.70 s | 1.06 s |

Refuted hypothesis: "attenuation / obliquity / a transfer function make spectral faster".
They are one multiply per frequency bin on either path; what flips the choice is the
number of radiating groups (temporal pays one sampled SIR + one FFT per element).

## Physics decisions and the evidence behind them

- **Dispersion is referenced at `fc`.** The y ≠ 1 phase carries the `−|f|·f0^(y−1)` term,
  otherwise `c` is the phase speed at f → 0 and the arrival at `fc` diverges as y → 1
  (−374 ns vs −34 ns over 5 cm at y = 1.1; 1695 m/s instead of 1540 at y = 1.01). The
  y = 1 phase is `+j(2α/π)f·ln(f/f0)d`, the y → 1 limit — it was sign-flipped (negative
  dispersion, echo 40 ns late). Found by a fresh-context adversarial review.
- **Obliquity is per patch, not per element.** Per-element `cosθ` errs −62 % at
  (0, 2.5, 1) mm next to a 0.3×5 mm element and −22 % far off a concave bowl; per patch
  converges to Field II's time-resolved `cosφ = z/(ct)`. `cosθ` is Field II parity; the
  exact Dirichlet kernel's `1/(jkR)` term is dropped (≤ 0.06 % at 5 MHz, R ≥ 2 mm; up to
  26° phase at 0.5–1 MHz within a few mm of the face).
- **Sub-sample clamp.** The temporal SIR widens a patch narrower than `1/fs` to one bin:
  its spectrum is multiplied by exactly `sinc(πf/fs)` per clamped axis (0.9745 / 0.9959 /
  0.9990 at 40 / 100 / 200 MHz for 5 MHz). Resizing patches cannot avoid it — a point in
  front of a patch sees Δt → 0 whatever its width. The spectral SIR has no clamp.
- **Attenuation path.** TX centre (+ RX element centre in reception), or per element in
  emission. Versus exact per-patch paths: ≈ 3 % RF difference in the near field (3–6 mm,
  19 mm aperture, 0.5 dB/MHz/cm), ≈ 0.3 % deep.
- **Signed transient pressure.** Same convention as pyMUST (`mkmovie` = `irfft`, `simus` =
  `irfft(conj(·))`); only its RMS map `RP` is non-negative. `abs` would break
  superposition and hide the peak negative pressure (MI).
- **No time-window margin for dispersion.** After the dispersion fix a lone echo vs one
  with the window opened early loses ≤ 1.2e-3 of its energy (up to 1 dB/MHz/cm, 150 mm).

## Bugs fixed — do not reintroduce

| Bug | Symptom | Guard |
|---|---|---|
| Attenuation y ≠ 1 without `f0` reference; y = 1 sign flip | Wrong arrival times/dispersion, default `freq_power=1` affected | dispersion tests |
| Emission truncated `exc ⊛ ir` to `len(exc)` | 56 % of pulse energy lost (2-cycle drive + 2-cycle IR) | `test_ir_equals_preconvolved_pulse` |
| Emission transient returned `abs(irfft)` | Rectified pressure, no rarefaction | Rayleigh test (correlation sign) |
| Emission missing the convolution `dt` | Output = `fs` × pascals (1e8× at 100 MHz) | Rayleigh test (amplitude) |
| Monochromatic read the nearest FFT bin | 1.2 % error vs exact `fc` | golden values |
| Pulse-centre lag used `exc.size` | `t0` off for per-element `(L, E)` drives (L·E samples) | reception parity |
| Elevation-lens sag added instead of subtracted | Image shifted 2·sag in depth, sharp PSF | `test_lens_time_origin.py` |

## Making a new claim

Label an untested cause as a hypothesis. To promote it: isolate it in a control run or a
parameter sweep, have a fresh reviewer try to refute it, and record the discriminating
test next to the conclusion (here, in a docstring, or in a test).
