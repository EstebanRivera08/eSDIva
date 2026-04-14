# Example 2: Mono-element Pressure Fields

Computes and plots the monochromatic (CW) pressure field for all four circular
transducer types: flat piston, concave bowl, cylindrical focus, and convex dome.

## What you will learn

- Creating mono-element (single-element) circular transducers
- Running a CW simulation with `PyField`
- Visualising XZ pressure planes with `plot_pressure_planes`

## Transducers covered

| Type | Geometry |
|------|----------|
| `FlatCircularTransducer` | Flat piston disc (D = 25 mm) |
| `ConcaveCircularTransducer` | Spherical bowl (D = 40 mm, R = 60 mm) |
| `FocusedCircularTransducer` | Cylindrical line-focus (D = 20 mm, R = 40 mm) |
| `ConvexCircularTransducer` | Convex dome (D = 20 mm, R = 10 mm) |

## Output

![FlatCircularTransducer pressure field](assets/mono_flat.png)
![ConcaveCircularTransducer pressure field](assets/mono_concave.png)
![FocusedCircularTransducer pressure field](assets/mono_focused.png)
![ConvexCircularTransducer pressure field](assets/mono_convex.png)

## Run it

```bash
uv run examples/example2_monoelement_transducers.py
```

## Key code

```python
from pyfield.psimulation import PyField
from pyfield.transducers import ConcaveCircularTransducer
from pyfield.utilities import plot_pressure_planes

bowl = ConcaveCircularTransducer(
    diameter_mm=40.0,
    radius_of_curvature_mm=60.0,
    no_sub=30,
    frequency_Hz=1e6,
)

sim = PyField(bowl, c=1540.0, fs=100e6)
x, y, z, p = sim(xz_grid, method="auto")
plot_pressure_planes(x, y, z, p / p.max())
```

[View full script on GitHub](https://github.com/EstebanRivera08/PyField/blob/main/examples/example2_monoelement_transducers.py)
