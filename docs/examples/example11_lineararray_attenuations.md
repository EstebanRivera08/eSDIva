# Example 11: CW Pressure Field with Tissue Attenuation

Side-by-side CW fields of a 10 MHz linear array in water (lossless) and in
brain tissue (causal power-law attenuation with Kramers–Kronig dispersion,
α₀ = 0.59 dB/(cm·MHz^y), y = 1.3 — ITIS database values).

## What you will learn

- Enabling causal attenuation: `Emission(alpha0=..., freq_power=...)`
- How attenuation reshapes a high-frequency focused beam
- Building a dB attenuation map: the map uses `fast_attenuation=True`
  (TX-centre distance), whose brain/water ratio is exactly the smooth
  distance-dependent loss `exp(−α·f^y·d)`. The per-element model cannot be
  used for this ratio: its per-element Kramers–Kronig dispersion shifts the
  interference fringes, so the pointwise ratio oscillates along the fringes.

## Output

![Water — no attenuation](assets/ex11_attenuation_water.png)
![Brain tissue — causal K-K attenuation](assets/ex11_attenuation_brain.png)
![Attenuation map (dB)](assets/ex11_attenuation_map.png)

## Run it

```bash
uv run examples/example11_lineararray_attenuations_monochromatic_CW.py
```

## Key code

```python
from sondi.emission import Emission

sim_water = Emission(tx, monochromatic=True, fs=100e6)
p_water, coords = sim_water(plane)

sim_brain = Emission(tx, monochromatic=True, fs=100e6,
                     alpha0=0.5912, freq_power=1.3)
p_brain, coords = sim_brain(plane)

att_map_db = 20 * np.log10(p_brain / p_water)
```

[View full script on GitHub](https://github.com/EstebanRivera08/SonDI/blob/main/examples/example11_lineararray_attenuations_monochromatic_CW.py)
