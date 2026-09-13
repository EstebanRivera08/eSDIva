---
icon: lucide/trending-down
---

# Attenuation

Power-law tissue attenuation is applied as a **causal frequency-domain transfer
function** multiplied onto the (lossless) SIR spectrum — absorption plus the
Kramers-Kronig dispersion phase, so waveforms stay causal. The SIR kernel itself
is never modified.

```python
sim = Emission(tx, fs=200e6, excitation=exc,
               alpha0=0.5,      # dB/(MHz^y · cm) — clinical convention
               freq_power=1.1)  # power-law exponent y (tissue 1.0–1.3)
p, coords = sim(field_points)
```

| Parameter | Meaning |
|-----------|---------|
| `alpha0` | Attenuation coefficient in dB/(MHz`^y`·cm); `None` disables it |
| `freq_power` | Power-law exponent `y` (1.0 = linear-with-frequency) |
| `fast_attenuation` | Emission path origin: `True` the transducer centre; `False` each element's centre (better near field) |

The dispersion is **referenced at the centre frequency**: the phase speed at `fc` equals
your `c`, and higher frequencies travel slightly faster (tissue-like positive
dispersion), continuously across `y = 1`.

![Monochromatic field with brain attenuation](../examples/assets/ex11_attenuation_brain.png)

Works in monochromatic (amplitude decay at `fc`) and transient (full spectral
distortion, including the raw SIR) modes, and in reception along the round trip
transducer centre → scatterer → receive-element centre, with every SIR method. See
[Example 11 — Attenuation](../examples/example11_lineararray_attenuations.md).
