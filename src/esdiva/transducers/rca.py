"""Row-column addressed (RCA) array: two crossed linear arrays on one face."""

from typing import Any, Callable, Optional, Tuple

import numpy as np
import pyvista as pv

from .base import _finish_plotter
from .linear import LinearArrayTransducer

ROT_Z_90 = np.array(
    [[0.0, -1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0, 0, 0, 1.0]]
)
"""4x4 rotation of +90° about z: turns the columns (bars along y) into the rows."""


class RowColumnAddressedTransducer:
    """Row-column addressed (RCA) probe: N columns crossed with N rows on one face.

    A fully populated N×N matrix needs N² channels. An RCA keeps 2N: the top
    electrodes are N long strips ("columns", bars along y, steering in x) and the
    bottom electrodes N strips at 90° ("rows", bars along x, steering in y). One
    set fires, typically the orthogonal set listens, so each array focuses in one
    direction only and the bar length gives no elevation focus unless a lens is
    added. That is enough for 3-D imaging: plane waves or sound sheets swept along
    one array, received and beamformed on the other.

    Each set is an ordinary `LinearArrayTransducer` (rectangular bar elements,
    the same SIR patches, the same delay/apodization API), so both are exposed as
    attributes and plug straight into the simulators::

        rca = RowColumnAddressedTransducer(n_elements=64, element_width_mm=0.09,
                                           kerf_mm=0.01, no_sub_x=1, no_sub_y=32,
                                           frequency_Hz=15e6)
        p, coords = Emission(rca.columns)(grid)             # transmit on columns
        rf, coords = Reception(rca.columns, rca.rows)(...)  # receive on rows
        das_rca_volume(rf, coords, ..., rx_length_mm=rca.bar_length_mm)

    Rows and columns occupy the same physical face (both at ``z = 0``), so their
    patches overlap in space; that is harmless because only one set radiates or
    receives in any one simulation.

    Parameters
    ----------
    n_elements : int
        Bars per array (N columns and N rows).
    element_width_mm : float
        Bar width in mm (pitch = width + kerf).
    kerf_mm : float
        Gap between bars in mm.
    no_sub_x : int
        Patches across a bar's width.
    no_sub_y : int
        Patches along a bar's length. A bar is as long as the whole aperture, so
        it needs many patches to stay in each patch's far field.
    frequency_Hz : float, optional
        Centre frequency in Hz.
    elevation_focus_mm : float, optional
        Signed radius of a cylindrical lens along each bar's length (as in
        `LinearArrayTransducer`); ``None`` is the flat, lens-free RCA.
    elevation_profile_mm : callable, optional
        Free lens shape ``y_mm -> z_mm`` along each bar, exclusive with
        `elevation_focus_mm`.
    """

    def __init__(
        self,
        *,
        n_elements: int,
        element_width_mm: float,
        kerf_mm: float,
        no_sub_x: int,
        no_sub_y: int,
        frequency_Hz: Optional[float] = None,
        elevation_focus_mm: Optional[float] = None,
        elevation_profile_mm: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> None:
        self.n_elements = n_elements
        self.pitch_mm = element_width_mm + kerf_mm
        # Square RCA: each bar spans the full aperture of the crossing array.
        self.bar_length_mm = n_elements * self.pitch_mm - kerf_mm
        self.columns = LinearArrayTransducer(
            n_elements=n_elements,
            element_width_mm=element_width_mm,
            element_height_mm=self.bar_length_mm,
            kerf_mm=kerf_mm,
            no_sub_x=no_sub_x,
            no_sub_y=no_sub_y,
            elevation_focus_mm=elevation_focus_mm,
            elevation_profile_mm=elevation_profile_mm,
            frequency_Hz=frequency_Hz,
        )
        self.rows = self.columns.copy()
        self.rows.transform(ROT_Z_90)
        self.columns.name, self.rows.name = "RCA columns", "RCA rows"
        self.fc = self.columns.fc

    def transform(self, T_matrix) -> None:
        """Rigidly move both arrays together (4x4, translation in mm).

        Parameters
        ----------
        T_matrix : (4, 4) array-like
            Homogeneous rigid-body transform, as for any transducer.
        """
        self.columns.transform(T_matrix)
        self.rows.transform(T_matrix)

    def show(
        self,
        *,
        offset_mm: float = 0.1,
        window_size: Tuple[int, int] = (800, 600),
        scalars: str = "Apodization",
        notebook: bool = False,
        jupyter_backend: Optional[str] = None,
        kwargs_columns: Optional[dict] = None,
        kwargs_rows: Optional[dict] = None,
    ) -> None:
        """Draw both arrays, each coloured by its own apodization or delays.

        Columns and rows share one physical face, so drawn in place they overlap
        and flicker. The rows are therefore drawn `offset_mm` behind the columns,
        along the face normal. **The offset is for display only**: the transducer
        geometry and every simulation keep both arrays on the same face.

        Parameters
        ----------
        offset_mm : float, default: 0.1
            Display-only gap (mm) between columns (front) and rows (behind).
            ``0`` draws them in their true, coincident position.
        window_size : (int, int)
            Pixel dimensions of the render window.
        scalars : {'Apodization', 'Delays'}
            Which cell array to colour both arrays by.
        notebook : bool
            Enable Jupyter notebook rendering.
        jupyter_backend : str, optional
            Backend string passed to PyVista (``'static'``, ``'trame'`` …).
        kwargs_columns, kwargs_rows : dict, optional
            Forwarded to ``plotter.add_mesh()`` for each array (e.g. ``cmap``,
            ``opacity``, ``color``); they override the defaults (columns ``cool`` /
            rows ``winter`` for apodization, ``gnuplot`` / ``rainbow`` for delays).
            ``color`` paints the array in one solid colour; its scalar bar is
            replaced by a legend entry (``label``, default: the array name).
        """
        plotter = pv.Plotter(window_size=window_size, notebook=notebook)
        title = "Apodization" if scalars == "Apodization" else "Delays (s)"
        # Mean patch normal = radiating direction of the face (follows transform/aim).
        normal = self.rows.sub_patch_frames["normals"].mean(axis=0)
        shift_mm = -offset_mm * normal / np.linalg.norm(normal)

        # Distinct colormaps tell the two bar sets apart at a glance.
        cmaps = {"Apodization": ("cool", "winter"), "Delays": ("gnuplot", "rainbow")}
        cmap_cols, cmap_rows = cmaps[scalars]
        for array, shift, user_kwargs, x_bar, cmap in (
            (self.columns, None, kwargs_columns, 0.8, cmap_cols),
            (self.rows, shift_mm, kwargs_rows, 0.1, cmap_rows),
        ):
            # Distinct titles keep two scalar bars (PyVista merges same-titled bars).
            kwargs: dict[str, Any] = {
                "cmap": cmap,
                "scalar_bar_args": {
                    "title": f"{array.name} \n {title}",
                    "vertical": True,
                    "position_x": x_bar,
                    "position_y": 0.1,
                },
                **(user_kwargs or {}),
            }
            # A solid colour has no scalar bar, so name it in a legend instead.
            if "color" in kwargs:
                kwargs.setdefault("label", array.name)
            array.add_to_plotter(plotter, scalars=scalars, shift_mm=shift, **kwargs)

        if any("color" in (kw or {}) for kw in (kwargs_columns, kwargs_rows)):
            plotter.add_legend(
                bcolor="white", face="rectangle", size=(0.25, 0.1), loc="upper center"
            )

        if offset_mm:
            plotter.add_text(
                f"Rows drawn {offset_mm:g} mm behind columns (display only)",
                position="lower_edge",
                font_size=8,
            )
        _finish_plotter(plotter, jupyter_backend)

    def __repr__(self) -> str:
        return (
            f"RowColumnAddressedTransducer({self.n_elements}+{self.n_elements} bars, "
            f"pitch={self.pitch_mm:.3f} mm, bar_length={self.bar_length_mm:.2f} mm, "
            f"fc={self.fc / 1e6:.2f} MHz)"
        )
