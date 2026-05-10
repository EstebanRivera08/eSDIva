---
icon: lucide/waves
hide:
  - toc
---

<div class="hero-banner" markdown>

# PyField

Acoustic field simulator for ultrasound transducers.  
Built on the Tupholme–Stepanishen Spatial Impulse Response method.

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
from pyfield.psimulation import PyField
from pyfield.utilities import plot_pressure_planes

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
sim = PyField(tx)
x, y, z, p = sim(field_points, method="auto")
plot_pressure_planes(x, y, z, p, db_scale=True, vmin=-40)
```

---

## Features

<div class="grid cards" markdown>

-   :lucide-cpu: **SIR-based computation**

    ---

    Patch-based Spatial Impulse Response engine derived from the Tupholme–Stepanishen method. Accurate for arbitrary transducer geometries.

-   :lucide-radio-tower: **Rich transducer library**

    ---

    Linear, convex, and matrix arrays; flat, concave, convex, and focused circular mono-elements; fully custom multi-element configurations.

-   :lucide-zap: **Monochromatic & transient**

    ---

    Compute steady-state continuous-wave fields or full time-domain pulsed simulations with user-defined excitation pulses.

-   :lucide-brain: **Brain atlas integration**

    ---

    Register acoustic fields onto anatomical structures via the BrainGlobe API. Includes rat and mouse atlases out of the box.

-   :lucide-box: **3-D visualization**

    ---

    Interactive 3-D scenes with PyVista: compose transducer geometry, pressure volumes, STL meshes, and brain anatomy in one renderer.

-   :lucide-flask-conical: **Research-grade accuracy**

    ---

    Validated against SIR benchmarks. Naive and SDI methods with automatic selection for optimal speed and numerical accuracy.

</div>
