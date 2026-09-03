# Building an eSDIva transducer

A transducer is a set of flat rectangular **patches**. Every patch radiates as a
small piston; its exact SIR is known analytically, and the aperture response is the
sum over patches. So the geometry object is not decoration — patch size and
placement *are* the numerical accuracy of the simulation.

All constructors are **keyword-only** and take millimetres. `frequency_Hz` sets `fc`
(used by the monochromatic mode, apodization windows and the default pulse).

## Choosing a class

```python
import esdiva.transducers as transducers
```

| Class | Geometry | Required kwargs |
|---|---|---|
| `LinearArrayTransducer` | 1-D row of rectangles, optional elevation lens | `n_elements`, `element_width_mm`, `element_height_mm`, `kerf_mm`, `no_sub_x`, `no_sub_y` |
| `ConvexArrayTransducer` | curved 1-D row (abdominal probe) | as above **+** `radius_of_curvature_mm` |
| `MatrixArrayTransducer` | 2-D grid of rectangles | `n_elements_x`, `n_elements_y`, `element_width_mm`, `element_height_mm`, `kerf_x_mm`, `kerf_y_mm`, `no_sub_x`, `no_sub_y` |
| `FlatCircularTransducer` | flat piston disc | `diameter_mm` |
| `ConcaveCircularTransducer` | spherical bowl, geometric focus in front (HIFU/TUS) | `diameter_mm`, `focus_mm` |
| `ConvexCircularTransducer` | spherical dome, virtual focus behind (lens) | `diameter_mm`, `focus_mm` |
| `FocusedCircularTransducer` | disc with single-axis curvature (line focus) | `diameter_mm`, `focus_mm`, `focus_axis` |
| `CustomTransducer` | any list of the above placed at arbitrary positions/normals (helmet, dual probe) | `elements`, `positions_mm`, `normals` |
| `FieldIITransducer` | built from `xdc_get(Th, 'all')` patch data | via `from_fieldii_xdc_data` / `from_fieldii_rect_data` / `from_fieldii_patch_arrays` |

Ready-made probes: `transducers.Domino()` (128-element linear) and
`transducers.Zeus_Matrix()` (55×55 matrix). Use them for demos and quick checks
instead of retyping element dimensions.

```python
tx = transducers.LinearArrayTransducer(
    n_elements=64, element_width_mm=0.25, element_height_mm=12.0,
    kerf_mm=0.05, no_sub_x=2, no_sub_y=4, frequency_Hz=5e6,
)
```

## Subdivision — the one number that decides accuracy

`no_sub_x` / `no_sub_y` split each element into `no_sub_x × no_sub_y` patches. The
SIR of a patch is exact, but the *aperture* SIR is only as good as the assumption
that a patch is small enough that its response does not vary across it. Practical
rule: patches of order λ/2 or smaller, i.e. more subdivision for tall elevation
dimensions (`element_height_mm` 12 mm at λ ≈ 0.3 mm is why `no_sub_y > no_sub_x`).

Circular apertures use `no_sub_diameter` instead, with `ratio_big_patches` and
`refine_factor` controlling how the rim is refined (the rim is where a coarse
rectangular tiling misrepresents the curved boundary).

Always run the convergence check once: double the subdivision, confirm the field
changes by less than your tolerance, keep the cheaper setting.

## Beamforming state (multi-element only)

Delays are seconds, apodization is dimensionless; both live on the transducer and
are recomputed in place for a new focus — you never rebuild the probe.

```python
tx.compute_delays(focus_mm=[0, 0, 30])                    # geometric focal law
tx.compute_delays(angle_steering_deg=15)                  # steered plane wave
tx.compute_delays(focus_mm=[0, 0, -10])                   # z < 0 → diverging wave
tx.compute_apodization(focus_mm=[0, 0, 30], FoverD=2.0,
                       apodization_type="hanning")        # F-number-limited aperture
tx.plot_delays_apodization()                              # inspect both
```

**z-convention:** the array sits at z = 0 and radiates toward +z. A focus at
`z > 0` converges; `z < 0` is a *virtual* source behind the aperture and produces a
diverging wave — that is how DW transmits are built. `FoverD` is the F-number: it
sets how much of the aperture is open, so a small `FoverD` means a wide aperture,
a tighter focus, and stronger sidelobes.

For a mono-element bowl the focus is *geometric* (`focus_mm` in the constructor);
there are no electronic delays to compute.

## Moving a probe in space

`transform(T_matrix)` applies a 4×4 homogeneous transform (translation in mm) to
**all** computed geometry — patch quads, patch frames, element centres — so the
simulation and the 3-D view stay consistent.

```python
rx = tx.copy()
rx.transform(T)          # e.g. a second probe facing the first
sim.set("rx", rx)        # simulators snapshot geometry at construction — re-set it
tx.clean()               # revert to the canonical pose
```

The `sim.set(...)` line is the usual bug: transforming a probe after building the
simulator changes nothing until the simulator is told.

## Inspecting

- `tx.show()` — 3-D PyVista view of the patches, coloured by apodization.
- `tx.plot_delays()` / `tx.plot_apodization()` / `tx.plot_delays_apodization()`.
- `tx.element_centers` — `(E, 3)` in **metres** (internal convention), handy for
  writing your own focal law.

## Impulse response

`tx.impulse_response = ir` (or `tx.set_impulse_response(ir)`) attaches the
electro-acoustic response of the element. It is mandatory for imaging work
(see the imaging checklist in `reception.md`) and ignored by pure-SIR emission runs.
