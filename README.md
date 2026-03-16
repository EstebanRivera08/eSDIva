

> [!WARNING]  
> PyField is currently under development. The API is subject to change, and some features may be incomplete or unstable.

PyField is an open‑source Spatial Impulse Response (SIR) and pressure‑field simulation library that supports arbitrary transducer geometries composed of small rectangular patches with apodization and delays.
PyField implements both the naïve and Sparse Delta Integration (SDI) methods for computing SIRs following the Tupholme–Stepanishen formulation.

> [!NOTE]  
> PyField is designed as complementary material to the work presented in [reference]. Its goal is to provide fundamental building blocks that researchers can inspect, reuse, contribute to, or adapt. It also leaves room for community‑driven extensions that integrate naturally with the broader scientific Python ecosystem.  
> Utilities such as the integration with the GlobeBrain atlas may still evolve to improve robustness.

### Main Features

- **Transducer objects** — Tools to create and assemble common linear and matrix array probe types. These utilities compute geometric focal laws, generate apodization windows for specified F/D ratios, and more.

- **SIR simulation** — The `H_sir` module computes discrete spatial impulse responses \( h(r, t) \) produced by apertures discretized into rectangular patches. It includes naïve, SDI, and automatic methods implemented with Numba‑accelerated kernels for field‑point‑parallel execution.

- **Pressure simulation** — Converts time‑domain SIRs into acoustic pressure fields. Supports monochromatic fields (spatial‑only) and broadband transient simulations with defined excitation pulses (producing spatio‑temporal pressure matrices).

- **Brain Atlas Integration** — Maps pressure simulations onto standard brain atlases for neuro‑ultrasound research.

- **Visualization** — Rich plotting utilities using PyVista for visualizing transducers, pressure fields, and brain atlases.

---

## Installation

### 1. Set up a virtual environment

We recommend installing PyField in a virtual environment to avoid dependency conflicts with other Python packages. Using [uv](https://docs.astral.sh/uv/guides/install-python/), you can create a new project folder with a virtual environment as follows:

```bash
uv init new_project
```

If you already have a project folder, create a virtual environment with:

```bash
uv venv
```

### 2. Install PyField

To install the latest development version from GitHub:

```bash
uv add git+https://github.com/EstebanRivera08/PyField.git
```

PyField will soon be available on PyPI.

---

If you want, I can also help you:

- refine the tone (more formal, more friendly, more concise)  
- reorganize the README for clarity  
- add badges, examples, or a quickstart section  
- write API documentation or docstrings  

Just tell me the style you’re aiming for.


### 3. Check installation

Check that PyField is correctly installed by opening a Python interpreter and
importing the package:

```python
import pyfield
```

If no error is raised, you have installed pyfield correctly.

## Quick Start

From `pyfield` folder, you can run the examples with a single command:

```bash
uv run example2_ratbrainzones.py
```

You can also use this script as a reference for building your own ultrasound simulations.

```python
import numpy as np
import pyvista as pv

import pyfield
from pyfield.psimulation import PyField
from pyfield.transducers import MatrixArrayTransducer
from pyfield.utilities import add_pressure_vol, add_transducer_mesh, create_vol_mesh

## -------Define the transducer focus and F/D and field points grid--------
focus_mm = np.array([0, 0, 3])
FoverD = 1

## Define the field points grid

field_point_mm = {
    "x_extent_mm": [-0.5, 0.5],
    "y_extent_mm": [-0.5, 0.5],
    "z_extent_mm": [1.5, 4.5],
    "dx_mm": 0.015,
    "dy_mm": 0.015,
    "dz_mm": 0.025,
}

# Define transducer

matrix_array_probe = MatrixArrayTransducer(
    N_elem_x=17,
    N_elem_y=17,
    elem_width_mm=0.2,
    elem_height_mm=0.2,
    kerf_x_mm=0.05,
    kerf_y_mm=0.05,
    no_sub_x=2,
    no_sub_y=2,
    frequency_Hz=10e6,
)

# Prepare transducer for simulation

delays = matrix_array_probe.compute_delays(focus_mm=focus_mm)
apodization = matrix_array_probe.compute_apodization(focus_mm=focus_mm, FoverD=FoverD)
matrix_array_probe.plot_delays_apodization()
matrix_array_probe.show(notebook=True, jupyter_backend="static", scalars="Delays")

# Perform simulation

matrix_field = PyField(matrix_array_probe)
x, y, z, p_matrixfield = matrix_field(field_point_mm, method="auto")

# Visualize the results

# TX + Pressure
transducer_mesh = matrix_array_probe.get_mesh()
pressure_mesh = create_vol_mesh(
    x, y, z, p_matrixfield / p_matrixfield.max(), scalars="Pressure"
)
plotter_matrix = pv.Plotter(window_size=(600, 600), notebook=False)
plotter_matrix = add_pressure_vol(pressure_mesh, plotter=plotter_matrix)
plotter_matrix = add_transducer_mesh(transducer_mesh, plotter=plotter_matrix)

plotter_matrix.show()
```

## Citing PyField

If you use PyField in your research, please cite it using the following reference:

Or in BibTeX format:

