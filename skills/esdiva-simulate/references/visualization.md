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

The `add_*` helpers all take a shared `pyvista.Plotter`, so a scene is built
incrementally and shown once:

```python
import pyvista as pv
from esdiva.plotting import add_transducer_mesh, add_pressure_vol, add_stl_mesh, add_markers

pl = pv.Plotter(notebook=True)          # backend decision happens HERE
add_transducer_mesh(pl, tx)
add_pressure_vol(pl, p, coords)
add_stl_mesh(pl, "petri_dish.stl")
add_markers(pl, targets_mm)
pl.show(jupyter_backend="static")
```

The `notebook`/`jupyter_backend` decision belongs to the `Plotter` and its `show()`,
exactly as in the table above.

## When nothing appears

- Empty notebook cell → `notebook=True` was missing, or `"trame"` was used in an
  exported notebook. Try `jupyter_backend="static"`.
- Script hangs or crashes on a server → no display; use `off_screen=True` with
  `save_path`, and `pyvista.start_xvfb()` on a bare Linux box.
- `ModuleNotFoundError: trame` → install the extra: `pip install "esdiva[jupyter]"`.
- Movie export fails → `pip install "esdiva[video]"`.
- Matplotlib figure never shows and warns about `FigureCanvasAgg` → headless
  backend; save the figure instead of showing it.
