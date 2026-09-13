---
icon: lucide/trending-down
---

# Attenuation

Power-law tissue attenuation is a **causal frequency-domain transfer function**
multiplied onto the (lossless) SIR spectrum — absorption plus the Kramers-Kronig
dispersion phase, so waveforms stay causal. The SIR kernels themselves are never
modified, so attenuation works identically with the spectral and temporal kernels, in
**emission and in reception**.

    H_att(f, d) = exp(−α₀|f|^y d) · exp(−j·[dispersion phase, referenced at fc]·d)

| Parameter | Meaning |
|-----------|---------|
| `alpha0` | Attenuation coefficient in dB/(MHz`^y`·cm); `None` disables it |
| `freq_power` | Power-law exponent `y` (1.0 = linear-with-frequency; tissue 1.0–1.3) |
| `fast_attenuation` | Emission only: path from the transducer centre (`True`) or from each element (`False`, better near field) |

The dispersion is **referenced at the centre frequency**: the phase speed at `fc` equals
your `c`, and higher frequencies travel slightly faster (tissue-like positive
dispersion), continuously across `y = 1`.

## Emission

The path `d` runs from the transducer centre to each field point (or from each element
with `fast_attenuation=False`). Works in monochromatic mode (amplitude decay at `fc`) and
in transient mode (full spectral distortion, including the raw SIR).

```python
from esdiva.emission import Emission

sim = Emission(tx, fs=200e6, excitation=pulse,
               alpha0=0.5,      # dB/(MHz^y · cm)
               freq_power=1.1)  # power-law exponent y
p, coords = sim(field_points)

sim_near = Emission(tx, fs=200e6, excitation=pulse, alpha0=0.5,
                    fast_attenuation=False)   # one path per element
```

![Monochromatic field with brain attenuation](../examples/assets/ex11_attenuation_brain.png)

## Reception

The echo travels out and back, so `d` is the round trip transducer centre → scatterer →
each receive element's centre, with every reception method.

```python
from esdiva.reception import Reception

sim = Reception(tx, rx, fs=200e6, excitation=pulse, alpha0=0.5, freq_power=1.1)
rf, coords = sim.pulse_echo_rf(scatterer_pos_mm, scatterer_amp)   # deeper echoes weaker
```

Deeper echoes lose amplitude and high frequencies faster than shallow ones, so the echo
spectrum shifts down with depth — the reason real scanners apply time-gain compensation.

See [Example 11 — Attenuation](../examples/example11_lineararray_attenuations.md).
