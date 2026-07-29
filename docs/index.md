---
icon: lucide/waves
hide:
  - toc
---

<div class="hero-banner" markdown>

# PyField

Acoustic field simulator for ultrasound transducers, based on the Tupholme–Stepanishen Spatial Impulse Response formulation and accelerated by the Sparse Delta Integration (SDI) method.

[![PyPI](https://img.shields.io/pypi/v/pyfield?color=9575cd&logo=pypi&logoColor=white&label=PyPI)](https://pypi.org/project/pyfield/)
[![Python](https://img.shields.io/pypi/pyversions/pyfield?color=00897b&logo=python&logoColor=white)](https://pypi.org/project/pyfield/)
[![License](https://img.shields.io/github/license/EstebanRivera08/PyField?color=9575cd)](https://github.com/EstebanRivera08/PyField/blob/main/LICENSE)
[![Stars](https://img.shields.io/github/stars/EstebanRivera08/PyField?color=9575cd&logo=github)](https://github.com/EstebanRivera08/PyField/stargazers)

[Get Started :lucide-arrow-right:](user-guide/getting-started.md){ .md-button .md-button--primary }
[GitHub :lucide-github:](https://github.com/EstebanRivera08/PyField){ .md-button }

</div>

---

## Install

=== "From GitHub"

    ```bash
    uv add git+https://github.com/EstebanRivera08/PyField.git
    ```

=== "Development"

    ```bash
    git clone https://github.com/EstebanRivera08/PyField.git
    cd PyField && uv sync
    ```

---

## Quickstart

```python
from pyfield.transducers import LinearArrayTransducer
from pyfield.emission import Emission
from pyfield.plotting import plot2D_pressure_slices

# 64-element linear array focused at 30 mm depth
tx = LinearArrayTransducer(
    n_elements=64,
    element_width_mm=0.25,
    element_height_mm=12.0,
    kerf_mm=0.05,
    no_sub_x=2,
    no_sub_y=4,
    frequency_Hz=5e6,
)
tx.compute_delays(focus_mm=[0, 0, 30])
tx.compute_apodization(focus_mm=[0, 0, 30], FoverD=2.0)

# Field grid (all distances in mm)
field_points = {
    "x_extent": [-5, 5],
    "y_extent": [-0.5, 0.5],
    "z_extent": [5, 55],
    "dx": 0.1,
    "dy": 1.0,
    "dz": 0.2,
}

# Run monochromatic simulation and visualize
sim = Emission(tx, monochromatic=True)
p, coords = sim(field_points, method="auto")
plot2D_pressure_slices(p, coords=coords, db_scale=True, vmin=-40)
```

---

## Why PyField

!!! tip "Sparse Delta Integration — the core of PyField"
    The **SDI** method reformulates the spatial impulse response as a sparse train
    of Dirac deltas integrated in time or frequency, evaluated with **Numba-parallel
    CPU kernels** (no GPU required). For large apertures this delivers
    **>100× faster emission** and **>20× faster reception (RF)** than sample-by-sample
    evaluation — with **negligible difference from Field II** — making full RF and
    phantom simulations practical on a laptop.

<div class="grid cards" markdown>

-   :lucide-zap: **SDI — very fast SIR**

    ---

    Sparse Delta Integration in time **and** frequency domain. >100× emission and
    >20× reception speedup on large apertures, Field II-accurate.

-   :lucide-cpu: **Numba-parallel, CPU-only**

    ---

    Parallel JIT kernels run on any multi-core CPU. No CUDA, no GPU dependency —
    fast RF simulation anywhere.

-   :lucide-activity: **Emission & Reception**

    ---

    CW / transient / attenuated emission fields, and full pulse-echo **RF** for
    PSF, phantom, FMC, and sequence (PW/DW) studies.

-   :lucide-radio-tower: **Rich transducer library**

    ---

    Linear, convex, matrix arrays; flat, concave, convex, focused circular
    mono-elements; fully custom geometries. Import Field II probes.

-   :lucide-box: **3-D visualization**

    ---

    Interactive PyVista scenes: transducer geometry, pressure volumes, STL meshes,
    and brain anatomy composed in one renderer.

-   :lucide-brain: **Brain atlas integration**

    ---

    Map acoustic fields onto anatomy via BrainGlobe. Rat and mouse atlases out of
    the box.

</div>

!!! note "Background theory"
    The SIR/SDI derivations are covered in the accompanying PyField paper — see
    [Citing PyField](citing.md).
