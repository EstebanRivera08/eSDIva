---
icon: lucide/wrench
---

# Utilities

Helper modules for geometry construction, brain atlas integration, and 3-D visualization.

<div class="grid cards" markdown>

-   :lucide-brain: **[Brain Atlas](brain-atlas.md)**

    ---

    BrainGlobe-based integration for mapping acoustic fields onto anatomical structures. Supports rat and mouse brain atlases.

-   :lucide-box: **[PyVista Integration](pyvista.md)**

    ---

    Helpers for composing PyVista 3-D scenes: pressure volumes, transducer meshes, STL imports, and custom lighting.

-   :lucide-ruler: **[Geometry Functions](geometry.md)**

    ---

    Functions for computing transducer geometry, coordinate transforms, and spatial utilities used across the library.

</div>

## Planning a long acquisition

```python
from esdiva.utilities import estimate_sequence_runtime

est = estimate_sequence_runtime(sim, positions_mm, n_emissions=n_angles * n_frames)
```

Times a few short pulse-echo probes on subsets of your real phantom, with your
real probe and excitation, and projects the wall time of the full sequence. Cost
is dominated by **scatterers × emissions**, but the constant is a property of the
machine — cores, memory bandwidth, threading — and varies by more than an order
of magnitude between a laptop and a compute node.

!!! warning "A timing from another machine is not a prediction for yours"
    It is not even stable on one machine: an identical computation has measured
    31 s and 42 s per emission minutes apart under different background load.
    Measure on the hardware that will run the job, and probe with the largest
    fraction you can afford — cost per scatterer grows with cloud size before
    levelling off, so small subsets under-predict.

## Receiver noise

eSDIva's RF is **noiseless by design**: the SIR model computes what an ideal
receiver would record. Any study whose conclusion depends on detectability —
Doppler sensitivity, contrast at depth, how many emissions must be averaged —
needs noise added explicitly, so that it is a stated assumption of the experiment
rather than a hidden property of the simulator.

```python
from esdiva.utilities import add_noise, rf_rms

noisy = add_noise(rf, snr_db=20, rng=0)          # 20 dB channel SNR

reference = rf_rms(rf_a)                          # comparing two sequences:
a = add_noise(rf_a, 6, reference=reference, rng=1)
b = add_noise(rf_b, 6, reference=reference, rng=2)
```

!!! note "Use one `reference` when comparing sequences"
    Noise belongs to the receiver, not to the transmit scheme. Letting each
    acquisition derive its own level from its own RMS quietly hands the weaker
    sequence a quieter amplifier and destroys the comparison — pass a single
    `reference` computed once.

TGC, ADC quantisation and element crosstalk are deliberately absent and are left
to the user.
