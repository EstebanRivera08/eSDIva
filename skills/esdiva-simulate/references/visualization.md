# Showing the result — and picking the right backend

Two rendering stacks, two sets of rules. **Matplotlib** draws the 2-D field maps and
waveforms; **PyVista** draws everything 3-D (transducer meshes, pressure volumes,
pulse-echo setups, STL geometry, brain atlases). PyVista is the one that needs a
deliberate backend choice — the same call that opens an interactive window on a
desktop silently renders nothing (or hangs) in a notebook, on a headless server, or
in CI.

**Ask where the figure is going before writing the call.** Desktop script, Jupyter
notebook, or a saved file for a paper — the answer changes the arguments, not the
code around them.

## Matplotlib (2-D)

```python
from esdiva.plotting import plot2D_pressure_slices, plot2D_transient_slices
plot2D_pressure_slices(p, coords=coords, db_scale=True)   # mono 3-D or transient 4-D
plot2D_transient_slices(p, coords=coords)                 # transient planes
plt.show()
```

Script or notebook, this just works. Headless (CI, a server, a background run): set
`MPLBACKEND=Agg` in the environment and save with `fig.savefig(...)` instead of
`plt.show()` — otherwise the figure is built and discarded, with a
"FigureCanvasAgg is non-interactive" warning.

Transient input animates; `plot2D_pressure_slices` on a 4-D array plays the
wavefront rather than drawing one frame.

## PyVista (3-D) — the backend decision

Every 3-D entry point takes the same two arguments: `notebook` and
`jupyter_backend`. They appear on `transducer.show()`, `Reception.show()`,
`plot3D_pressure_vol`, `plot3D_pressure_slices`, `plot3D_transient_slices`.

| Where you are | Call it like this | Needs |
|---|---|---|
| Desktop script / IDE | `tx.show()` — the defaults (`notebook=False`) open a native interactive window | a display |
| Jupyter / JupyterLab, want to rotate the scene | `tx.show(notebook=True, jupyter_backend="trame")` | `pip install "esdiva[jupyter]"` (trame + ipywidgets) |
| Jupyter, just want a picture in the output cell | `tx.show(notebook=True, jupyter_backend="static")` | nothing extra; safest in a notebook shared as HTML |
| Headless server, CI, batch figure generation | `plot3D_pressure_vol(..., off_screen=True, save_path="fig.png")` | an offscreen GL context (`pyvista.start_xvfb()` on bare Linux) |

Rules of thumb:

- `notebook=True` without `jupyter_backend` leaves the choice to PyVista's global
  setting, which is why the same notebook behaves differently on two machines. Be
  explicit.
- `"trame"` is interactive but heavier and needs a live kernel — a notebook exported
  to HTML shows an empty cell. `"static"` renders a PNG into the cell and survives
  export. Prefer `"static"` for anything you will share, `"trame"` while exploring.
- Never pass `notebook=True` from a plain script: the figure goes to a widget that
  has nowhere to display.
- `Reception.show(...)` additionally takes `save_path`, `off_screen` and
  `return_plotter` — `return_plotter=True` hands back the `pyvista.Plotter` so you
  can add your own meshes (an STL vessel, a target marker) before showing it.

## Saving

```python
from esdiva.plotting import save_pyvista_screenshot, save_pyvista_movie, save_matplotlib_animation
```

`save_pyvista_screenshot` for a still, `save_pyvista_movie` for an orbit or a
propagating wavefront (needs `pip install "esdiva[video]"` — imageio + ffmpeg), and
`save_matplotlib_animation` for 2-D transient sequences. The `plot3D_*` helpers also
write a still directly when given `save_path`, switching themselves to off-screen
rendering when they do.

## Composing your own 3-D scene

The `add_*` helpers all take a shared `pyvista.Plotter` (as the keyword `plotter=`)
and never call `show()`, so a scene is built incrementally and shown once, last. The
pressure helper takes a *mesh*, not `(p, coords)` — build it with `create_3Dvol_mesh`:

```python
import pyvista as pv
from esdiva.plotting import (
    add_transducer_mesh, add_pressure_vol, add_stl_mesh, add_markers, create_3Dvol_mesh,
)

pl = pv.Plotter(notebook=True)          # backend decision happens HERE
mesh = create_3Dvol_mesh(p, coords["x"], coords["y"], coords["z"], scalars="Pressure")
add_pressure_vol(mesh, plotter=pl)
add_transducer_mesh(tx.get_mesh(), plotter=pl)
add_stl_mesh("petri_dish.stl", plotter=pl)
add_markers(targets_mm, plotter=pl)
pl.show(jupyter_backend="static")
```

The `notebook`/`jupyter_backend` decision belongs to the `Plotter` and its `show()`,
exactly as in the table above.

**The one-shot `plot3D_*` trap.** `plot3D_pressure_vol(..., show_fig=True)` calls
`show()` internally, which on a desktop backend *closes* the render window — the plotter
it returns is then spent and anything you add to it afterwards never appears. To overlay
(a second field, the transducer, a marker), leave `show_fig=False` (the default), add
with the `add_*` helpers above, and call `show()` yourself at the end.

## Moving and combining volumes — `Field`

The medium is homogeneous, so a computed volume rotates and translates *with* the
aperture that produced it. You do not re-simulate a steered or displaced probe — you
move the result. `Field` (`from esdiva import Field`) wraps any `(data, coords)` scalar
volume: an `Emission` pressure field, a `das_volume` PSF/envelope (its `x_mm`/`y_mm`/
`z_mm` axes are accepted as-is), an intensity map. Not Field II, and not pressure-only.

**Recommended steps** — the same for every case:

```python
from esdiva import Field

canvas = Field.canvas(grid)                       # 1. one common grid (Emission's grid dict)
views  = [Field(vol, coords).transform(T)         # 2. wrap each result, 3. pose it (tx's 4x4)
          for (vol, coords), T in results]
total  = sum(v.resample(canvas) for v in views)   # 4. bring every view onto the canvas
total.plot()                                      # 5. draw — or use total.data / total.coords
```

| Case | Combine on the canvas | Why |
|---|---|---|
| Compound views fired at different times (tilts, orthogonal probes, power Doppler) | `sum(v.resample(canvas) ** 2 for v in views)` | incoherent → **power** adds |
| Views fired together (one coherent wavefront) | `sum(v.resample(canvas) for v in views)` on SIGNED transient frames | coherent → pressure adds; a monochromatic `\|P\|` has lost its phase, so never sum it coherently |
| Two-way CW beam (TX then RX) | `tx_field * rx_field` | transmit and receive act in sequence → product |
| Compare two results (method, pose, re-simulation) | `a - b` | point-by-point difference map |

Rules that decide whether the result is right:
- **Simulate each source volume wider than the canvas.** A tilted view must still cover
  every canvas point; `resample` returns **0** where it doesn't reach, which reads as a
  real dropout. Coverage mask of a pose: `Field(np.ones_like(vol), coords).transform(T).resample(canvas).data > 0.999`.
- Grid step ≤ λ/3: `resample` is trilinear; on a λ/3 grid moved-vs-re-simulated agrees to
  ~2 % of peak beyond the near field (example 25).
- Runnable demo of all four cases: `examples/example25_field_canvas.py`.

Details:
- **`transform(T)`** is deferred and chainable, and returns a *copy* (the field is
  immutable), so one simulation can be placed at many poses without recomputing.
- **`to_mesh()`** returns a `StructuredGrid` with the pose baked into explicit point
  coordinates, so a steered field **tilts** correctly. (Raw `create_3Dvol_mesh` returns
  `ImageData`, whose lazy direction matrix the contouring ignores — a rotated `ImageData`
  only *shifts*. Pass `structured=True`, or just use `Field`, when the volume
  will be reoriented.)
- **`add_to(plotter)`** draws into a shared plotter (the overlay path); **`plot()`** is
  the one-shot (new plotter, shows by default).
- **`resample(onto)`** puts a posed field on another's grid; `+ - * / **` then act point
  by point (scalars too). `a + b` on differently-posed fields resamples `b` onto `a`'s
  grid for you, but an explicit canvas is clearer and keeps every view on equal terms.
- A transient `(Nt, …)` is a time series, not one thing in space: pick a frame with
  `to_mesh(time_index=…)` before posing, resampling or summing.

## When nothing appears

- Empty notebook cell → `notebook=True` was missing, or `"trame"` was used in an
  exported notebook. Try `jupyter_backend="static"`.
- Script hangs or crashes on a server → no display; use `off_screen=True` with
  `save_path`, and `pyvista.start_xvfb()` on a bare Linux box.
- `ModuleNotFoundError: trame` → install the extra: `pip install "esdiva[jupyter]"`.
- Movie export fails → `pip install "esdiva[video]"`.
- Matplotlib figure never shows and warns about `FigureCanvasAgg` → headless
  backend; save the figure instead of showing it.
