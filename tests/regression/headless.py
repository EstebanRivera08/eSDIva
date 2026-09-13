"""Run one example script with no windows: ``python headless.py path/to/example.py``.

Every PyVista plotter is forced off-screen and ``show()`` renders once then closes it;
Matplotlib uses Agg and ``plt.show()`` closes the figures. Scenes are still built and
rendered (so rendering errors surface), but nothing opens on screen and no plotter is
left alive. Used by `test_examples.py` and ``just figures`` (with ``ESDIVA_SAVE_FIG=1``).
"""

import runpy
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pyvista as pv  # noqa: E402

pv.OFF_SCREEN = True
_init = pv.Plotter.__init__


def _init_off_screen(self, *args, **kwargs):
    kwargs["off_screen"] = True
    _init(self, *args, **kwargs)


def _render_and_close(self, *args, **kwargs):
    self.render()
    self.close()


pv.Plotter.__init__ = _init_off_screen
pv.Plotter.show = _render_and_close
plt.show = lambda *args, **kwargs: plt.close("all")

script = sys.argv[1]
sys.argv = [script]
sys.path.insert(
    0, str(__import__("pathlib").Path(script).parent)
)  # `from config import`
runpy.run_path(script, run_name="__main__")
