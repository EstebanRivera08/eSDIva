---
icon: lucide/layers
---

# Moving & Combining Volumes

eSDIva propagates in a **homogeneous** medium, so a volume computed once moves rigidly
with the probe that produced it: tilt or shift the probe, and the field (or PSF) tilts or
shifts with it. You never need to re-simulate a moved probe — you move the result.

[`Field`][esdiva.fields.Field] wraps any scalar volume on a grid and lets you **move**
it, bring several onto **one common grid**, and **combine** them point by point. It is
not tied to pressure (or to Field II): an `Emission` pressure field, a beamformed PSF
or envelope from `das_volume`, an intensity map — all are a `Field`.

## Recommended steps

Every use follows the same five steps:

```python
from esdiva import Field

# 1. One common grid — the same dict Emission takes.
canvas = Field.canvas({"x_extent": [-10, 10], "y_extent": [-5, 5],
                       "z_extent": [20, 60], "dx": 0.1, "dy": 0.1, "dz": 0.1})

# 2. Wrap each result, 3. pose it with the same 4x4 you passed tx.transform(T).
views = [Field(vol, coords).transform(T) for (vol, coords), T in results]

# 4. Bring every view onto the canvas, 5. combine.
total = sum(v.resample(canvas) for v in views)
total.plot()        # 3-D render — or use total.data / total.coords as arrays
```

`coords` is what the simulator or beamformer returned: `Emission` gives `"x"`/`"y"`/`"z"`,
`das_volume` gives `"x_mm"`/`"y_mm"`/`"z_mm"` — both are accepted. Every operation
returns a new `Field`; the original is never modified.

## Which combination is physically right

| You want | Write | Why |
|---|---|---|
| Compound views **fired at different times** (tilted probes, orthogonal probes, power Doppler) | `sum(v.resample(canvas) ** 2 for v in views)` | Echoes from separate firings do not interfere: **power** adds. |
| Views that are **one coherent wavefront** (fired together) | `sum(v.resample(canvas) for v in views)` on **signed** transient frames | Pressures superpose. A monochromatic `\|P\|` has lost its phase — never sum it as if coherent. |
| The **two-way** (pulse-echo) CW beam | `tx_field * rx_field` | Transmit and receive act in sequence, so their amplitude responses multiply. |
| A **difference map** (two methods, two poses, moved vs re-simulated) | `a - b` | Point-by-point comparison. |
| Rescale | `2 * a`, `a / a.data.max()` | Scalars act on every point. |

## Two things that decide whether the result is right

**Simulate each source wider than the canvas.** A tilted view must still cover every
canvas point; where it doesn't reach, `resample` returns **0** — which looks like a real
dropout. To see a pose's coverage, resample a field of ones:

```python
covered = Field(np.ones_like(vol), coords).transform(T).resample(canvas).data > 0.999
```

**Sample at λ/3 or finer.** `resample` interpolates trilinearly. On a λ/3 grid a moved
field matches the re-simulated moved probe to ~2 % of peak beyond the near field
(32-element 5 MHz array tilted 15°); refine the grid to shrink it. A pure
translation by a whole number of grid steps is exact (xAM sound-sheet study: 2.5e-6 of peak).

## A PSF on a common canvas

The same steps hold for beamformed PSFs, e.g. two orthogonal probes imaging one target:

```python
from scipy.signal import hilbert

vol, axes = das_volume(rf, coords, events, probe, grid_mm)   # in the probe's own frame
psf = Field(np.abs(hilbert(vol, axis=2)), axes, scalars="PSF")

canvas = Field.canvas(roi_grid, scalars="PSF")
power = sum(psf.transform(T).resample(canvas) ** 2 for T in (T_probe_a, T_probe_b))
db = 10 * np.log10(power.data / power.data.max())
```

## NumPy on a field

A `Field` behaves like its sample array wherever the axis order is kept, and the
result stays a `Field` on the same grid and pose:

- element-wise NumPy functions — `np.abs(f)`, `np.sqrt(f)`, `np.log10(f)`,
  `np.maximum(a, b)`;
- `f.shape`, `f.ndim`, `f.dtype`, and `np.asarray(f)` for the raw samples;
- reductions over **time** of a transient: `f.mean(axis=0)`, `f.max(axis=0)` (peak
  pressure), `f.min(axis=0)` (peak rarefaction), `f.sum(axis=0)`. Without `axis`
  they return one number, e.g. `f / f.max()` normalises.

```python
rms = np.sqrt((field**2).mean(axis=0))   # RMS pressure map, still a Field
```

There is no `transpose`/`reshape`: the axis order `(t, x, y, z)` is what ties each
sample to its coordinate. For a spatial reduction take `np.asarray(f)` first.

## Drawing

- `field.plot()` — one-shot 3-D render on a new plotter.
- `field.add_to(plotter)` — add to a scene you are composing; call `plotter.show()` last.
- `field.to_mesh()` — the posed PyVista mesh. Its points carry the pose, so a rotated
  field **tilts** (a plain `create_3Dvol_mesh` `ImageData` would only shift).
- Transient `(Nt, …)` data: pick a frame with `time_index=` before posing or combining.

[Example 25](../examples/advanced/rca_xam_sound_sheet.md) uses `Field` arithmetic on transient fields to build a cross-amplitude-modulation sound sheet, and moves the sheet instead of re-simulating it.
