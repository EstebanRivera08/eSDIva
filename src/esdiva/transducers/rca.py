"""Row-column addressed (RCA) array: two crossed linear arrays on one face."""

from typing import Callable, Optional

import numpy as np
import pyvista as pv

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

    def show(self, **kwargs) -> None:
        """Draw both arrays: columns coloured by apodization, rows as wireframe.

        Parameters
        ----------
        **kwargs
            Forwarded to ``pyvista.Plotter.add_mesh`` for the columns.
        """
        plotter = pv.Plotter()
        kwargs.setdefault("scalars", "Apodization")
        kwargs.setdefault("cmap", "cool")
        kwargs.setdefault("show_edges", True)
        plotter.add_mesh(self.columns.get_mesh(), **kwargs)
        plotter.add_mesh(self.rows.get_mesh(), style="wireframe", color="k")
        plotter.add_axes()
        plotter.show_grid(xtitle="X (mm)", ytitle="Y (mm)", ztitle="Z (mm)")
        plotter.show()

    def __repr__(self) -> str:
        return (
            f"RowColumnAddressedTransducer({self.n_elements}+{self.n_elements} bars, "
            f"pitch={self.pitch_mm:.3f} mm, bar_length={self.bar_length_mm:.2f} mm, "
            f"fc={self.fc / 1e6:.2f} MHz)"
        )
