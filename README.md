

>[!WARNING]
> PyField is currently under active development. The API is subject
> to change, and features may be incomplete or unstable.


PyField is an open-source Spatial Impulse Response (SIR) and Pressure Field simulation
program that supports arbitrary transducer geometries made up from small rectangular
partches with apodization and delays.

PyField supports the Naive and Sparse Delta Integration (SDI) method for calculation of
SIR following the Tuphome-Stepanishen develpment.

>[!NOTE]
> Because PyField is designed as complementary material of what is presented in [],
> PyField aims to provide the fundamental building blocks for researchers to inspect,
> reuse, contribute or adapt the code and leaves room for community-driven extension
> that may integrate naturally with the broader scientific python ecosystem.
> Utilities such us the integration with the GlobeBrain-atlas might experience
> modifications to make it more robust. 

- **Transducer objects** : Routines to create and construct common linear and matrix
  arrays probe types. These routines provide utilities to compute geometric focal laws,
  to generate apodization windows for specified F/D ratios and to accept.
- **SIR simulation** : The H_sir module computes discrete spatial impuse responses
  $h(r,t)$ produced by an aperture discretized into rectangular patches. Within the
  methods to compute the response, this module counts with the naive, SDI and auto
  methods supported in numba-accelerated kernels for field-point-parallel execution.
- **Pressure simulation** : This module converts time-domain SIRs into acoustic pressure
  fields. The pressure field can be monochromatic (solely depending in space) or
  broadband transient simulations with defined excitation pulses (giving spatio-temporal
  pressure matrices as result).
- **Brain Atlas Integraion** : Map pressure simulations to standard brain atlases for
  neuro-ultrasound technologies. 
- **Visualization** : Rich plotting utilities using PyVista for visualization of
  transducers, pressure fields and brain atlases.

## Installation

### 1. Setup a virtual environment


We recommend to install pyfield in a virtual environment to avoid dependecy conflicts
wiht other Python packages. Using
[uv](https://docs.astral.sh/uv/guides/install-python/),
you may create a new project folder with a virtual environmen as follows:

```bash
uv init new_project
```

If you already have a project folder, you may create a virtual environment as follows:

```bash
uv venv
```

### 2. Install PyField

To install the lates devepment version from GitHub :

```bash
uv add git+https://github.com/EstebanRivera08/PyField.git
```
Soon it will be present in PyPi...

### 3. Check installation

Check that PyField is correctly installed by opening a Python interpreter and
importing the package:

```python
import pyfield
```

If no error is raised, you have installed pyfield correctly.

## Quick Start

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

