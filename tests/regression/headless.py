"""Run one example script with no windows: ``python headless.py path/to/example.py``.

Every PyVista plotter is forced off-screen and ``show()`` renders once then closes it;
Matplotlib uses Agg and ``plt.show()`` closes the figures. Scenes are still built and
rendered (so rendering errors surface), but nothing opens on screen and no plotter is
left alive. Used by `test_examples.py` and ``just figures``.

With ``ESDIVA_SAVE_FIG=1`` every figure, animation and 3-D scene the script shows is
also saved to ``docs/examples/assets/<script>_<n>.png|gif``, so the examples
themselves carry no figure-saving code.
"""

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.animation as animation  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pyvista as pv  # noqa: E402

script = Path(sys.argv[1]).resolve()
SAVE = os.environ.get("ESDIVA_SAVE_FIG", "0") == "1"
FIG_FOLDER = Path(__file__).parents[2] / "docs" / "examples" / "assets"
_count = 0
_animations = []


def _next_path(suffix):
    global _count
    _count += 1
    FIG_FOLDER.mkdir(parents=True, exist_ok=True)
    return FIG_FOLDER / f"{script.stem}_{_count}{suffix}"


pv.OFF_SCREEN = True
_init = pv.Plotter.__init__


def _init_off_screen(self, *args, **kwargs):
    kwargs["off_screen"] = True
    _init(self, *args, **kwargs)


def _render_and_close(self, *args, **kwargs):
    self.render()
    if SAVE:
        self.screenshot(str(_next_path(".png")))
    self.close()


_anim_init = animation.Animation.__init__


def _record_animation(self, *args, **kwargs):
    _anim_init(self, *args, **kwargs)
    _animations.append(self)


def _save_and_close(*args, **kwargs):
    if SAVE:
        animated = {a._fig for a in _animations}
        for a in _animations:
            a.save(_next_path(".gif"), writer=animation.PillowWriter(fps=10))
        for num in plt.get_fignums():
            fig = plt.figure(num)
            if fig not in animated:
                fig.savefig(_next_path(".png"), dpi=150, bbox_inches="tight")
    _animations.clear()
    plt.close("all")


import esdiva.plotting  # noqa: E402

_plot3d_transient = esdiva.plotting.plot3D_transient_slices


def _plot3d_transient_saved(*args, **kwargs):
    # PyVista animations are written by the plotting call itself, not by show().
    if SAVE and kwargs.get("save_path") is None:
        movie = _next_path(".gif")
        kwargs.update(save_path=str(movie.parent), file_name=movie.name)
    return _plot3d_transient(*args, **kwargs)


esdiva.plotting.plot3D_transient_slices = _plot3d_transient_saved
pv.Plotter.__init__ = _init_off_screen
pv.Plotter.show = _render_and_close
animation.Animation.__init__ = _record_animation
plt.show = _save_and_close

sys.argv = [str(script)]
sys.path.insert(0, str(script.parent))  # Advanced studies import sibling modules.
sys.path.insert(0, str(Path(__file__).parents[2] / "examples"))  # `from config import`.
scope = {"__name__": "__main__", "__file__": str(script)}
exec(compile(script.read_text(encoding="utf-8"), str(script), "exec"), scope)
_save_and_close()  # Figures the script built but never showed.
# Free the script's VTK objects while the interpreter is alive, so their __del__
# does not run during shutdown and print "Exception ignored ..." noise.
scope.clear()
pv.close_all()
__import__("gc").collect()
