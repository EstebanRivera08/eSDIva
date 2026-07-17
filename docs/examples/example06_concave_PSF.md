# Example 6: Concave Transducer — Pulse-Echo PSF

Computes the pulse-echo point spread function (PSF) of a spherically focused
single-element transducer, comparing the two reception backends: conventional
(`Reception`, Field II-style SIR convolution) and PE-SDI (`ReceptionSDI`).
Both agree to a fraction of a percent — the comparison doubles as a
self-validation of the reception engine.

## What you will learn

- Sweeping a point scatterer laterally to build a PSF image
- `pulse_echo_rf(..., per_scatterer=True)` — one RF trace per scatterer
- Setting the transducer `impulse_response` / `excitation` chain
- Raw-RF and log-envelope comparison between backends

## Output

![Concave PSF comparison (FST vs SDI)](assets/ex06_concave_psf_comparison.png)

## Run it

```bash
uv run examples/example06_concave_PSF.py
```

## Key code

```python
from pyfield.reception import Reception, ReceptionSDI
from pyfield.transducers import ConcaveCircularTransducer

tx = ConcaveCircularTransducer(diameter_mm=16, focus_mm=80, frequency_Hz=3e6,
                               no_sub_diameter=16)
tx.impulse_response = pulse
tx.excitation = pulse
rx = tx.copy()

sim = ReceptionSDI(tx, rx, fs=100e6, c=1540)
rf, coords = sim.pulse_echo_rf(scatterer_positions_mm, per_scatterer=True)
# rf.shape = (N_scat, E_rx, Nt) — the PSF, one trace per lateral position
```

[View full script on GitHub](https://github.com/EstebanRivera08/PyField/blob/main/examples/example06_concave_PSF.py)
