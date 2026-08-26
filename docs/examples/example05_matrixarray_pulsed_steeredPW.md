# Example 5: Matrix Array — Steered Plane Wave (3-D Transient)

A 32 × 32 matrix array (3 MHz, sub-λ pitch) transmits a plane wave steered
10° off-axis. The transient field is computed on two orthogonal planes and
animated in 3-D together with the transducer mesh.

## What you will learn

- Plane-wave steering delays via `compute_delays(angle_steering_deg=(θx, θy))`
- Computing planes instead of full volumes (same physics, far cheaper)
- Synchronising plane time axes with `align_to_common_time`
- 3-D animation with `plot3D_transient_slices` (GIF export)

## Output

![Matrix-array steered plane-wave transient (3-D)](../assets/ex05_matrix_pw_3d.gif)

## Run it

```bash
uv run examples/example05_matrixarray_pulsed_steeredPW.py
```

## Key code

```python
from esdiva.emission import Emission
from esdiva.plotting import add_transducer_mesh, plot3D_transient_slices
from esdiva.transducers import MatrixArrayTransducer
from esdiva.utilities import align_to_common_time

tx = MatrixArrayTransducer(
    n_elements_x=32, n_elements_y=32,
    element_width_mm=0.275, element_height_mm=0.275,
    kerf_x_mm=0.025, kerf_y_mm=0.025,
    no_sub_x=1, no_sub_y=1, frequency_Hz=3e6,
)
tx.compute_delays(angle_steering_deg=(10, 0))

sim = Emission(tx, fs=50e6, excitation=excitation)
p_xz, c_xz = sim(PLANE_XZ)
p_yz, c_yz = sim(PLANE_YZ)

t, [p_xza, p_yza] = align_to_common_time([(p_xz, c_xz), (p_yz, c_yz)])

planes = [
    {"plane": "xz", "data": p_xza.squeeze(), "translation": (0, 0, 0)},
    {"plane": "yz", "data": p_yza.squeeze(), "translation": (0, 0, 0)},
]
plotter = add_transducer_mesh(tx.get_mesh(), scalars="Delays")
plot3D_transient_slices(planes, coords=coords, plotter=plotter, time_array=t, db_scale=True)
```

[View full script on GitHub](https://github.com/EstebanRivera08/eSDIva/blob/main/examples/example05_matrixarray_pulsed_steeredPW.py)
