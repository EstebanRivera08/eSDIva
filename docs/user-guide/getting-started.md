# Getting Started

This guide walks you through installing PyField and running your first simulation.

## Installation

### Prerequisites

- Python >= 3.11
- [uv](https://docs.astral.sh/uv/) (recommended package manager)

### Install from GitHub

```bash
uv add git+https://github.com/EstebanRivera08/PyField.git
```

### Verify installation

```python
import pyfield
print(pyfield.__version__)
```

## Your first simulation

### 1. Create a transducer

PyField models transducers as collections of rectangular patches. Start with a
simple linear array:

```python
from pyfield.transducers import LinearArrayTransducer

tx = LinearArrayTransducer(
    n_elements=64,
    element_width_mm=0.25,
    element_height_mm=12.0,
    kerf_mm=0.05,
    no_sub_x=2,
    no_sub_y=4,
    frequency_Hz=5e6,
)
```

### 2. Configure focusing

Set electronic delays and apodization to focus the beam:

```python
tx.compute_delays(focus_mm=[0, 0, 30])
tx.compute_apodization(focus_mm=[0, 0, 30], FoverD=2.0)
```

### 3. Define the field grid

Specify the spatial region where pressure will be computed (all distances in mm):

```python
field_points = {
    "x_extent": [-5, 5],
    "y_extent": [-0.5, 0.5],
    "z_extent": [5, 55],
    "dx": 0.1,
    "dy": 1.0,
    "dz": 0.2,
}
```

### 4. Run the simulation

```python
from pyfield.psimulation import PyField

sim = PyField(tx)
x, y, z, p = sim(field_points, method="auto")
```

### 5. Visualize results

```python
from pyfield.utilities import plot_pressure_planes

plot_pressure_planes(x, y, z, p, db_scale=True, vmin=-40)
```

## Key concepts

- **Patch-based discretization**: every transducer surface is approximated by
  small flat rectangular patches. The `no_sub_x` and `no_sub_y` parameters
  control how many patches per element -- more patches means higher accuracy
  but slower computation.

- **Unit convention**: user-facing APIs use millimeters (`_mm` suffix);
  internal computations use SI units (metres, seconds).

- **Coordinate system**: X = lateral, Y = elevation, Z = axial (depth).

## Next steps

- [Transducer types](transducers.md) -- learn about all available geometries
- [Simulation modes](simulation.md) -- monochromatic vs transient
- [Visualization](visualization.md) -- 2D and 3D plotting options
