# Example 3: Multi-element Transducers — 3-D Visualisation

Computes a monochromatic focused pressure field for both a linear array
(Domino) and a matrix array (Zeus_Matrix), then renders the transducer
geometry and pressure volume together in 3-D using PyVista.

## What you will learn

- Simulating focused fields for linear and matrix arrays side by side
- Building volumetric pressure meshes with `create_3Dvol_mesh`
- Composing 3-D scenes with `add_transducer_mesh` and `add_pressure_vol`
- Controlling PyVista camera positions and grid annotations

## Output

![Linear array 3-D scene](assets/ex03_linear_array_field.png)
![Linear array pressure volume](assets/ex03_linear_array_pressure_field.png)
![Matrix array 3-D scene](assets/ex03_matrix_array_field.png)
![Matrix array pressure volume](assets/ex03_matrix_array_pressure_field.png)

## Run it

```bash
uv run examples/example03_multielements_monochromatic_CW.py
```

## Key code

```python
from sondi.transducers import Domino
from sondi.emission import Emission
from sondi.plotting import add_pressure_vol, add_transducer_mesh, create_3Dvol_mesh

probe = Domino()
probe.compute_delays(focus_mm=[-2, 0, 8])
probe.compute_apodization(focus_mm=[-2, 0, 8], FoverD=1)

sim = Emission(probe, monochromatic=True)
p, coords = sim(field_point_mm, method="auto")

# Build PyVista meshes
tx_mesh = probe.get_mesh()
pr_mesh = create_3Dvol_mesh(p / p.max(), coords["x"], coords["y"], coords["z"], scalars="Pressure")

# Render
plotter = pv.Plotter()
plotter = add_pressure_vol(pr_mesh, plotter=plotter)
plotter = add_transducer_mesh(tx_mesh, plotter=plotter)
plotter.show()
```

[View full script on GitHub](https://github.com/EstebanRivera08/SonDI/blob/main/examples/example03_multielements_monochromatic_CW.py)
