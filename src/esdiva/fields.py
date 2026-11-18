"""Pressure fields you can move in space and compound.

A simulation returns a pressure volume sampled on an axis-aligned grid together
with its coordinates — but the medium is homogeneous, so the *same* field just
rotates and translates with the aperture that produced it. You therefore do not
re-simulate to look at a steered or displaced probe: you move the result.

`PressureField` wraps the ``(data, coords)`` pair an `Emission` (or a beamformed
`Reception`) returns and gives it three physical operations:

- **move** it rigidly in space (`transform`), the same 4x4 transform you apply to
  the transducer, so field and aperture stay registered;
- **render** it (`to_mesh` / `add_to` / `plot`), with the pose baked into
  explicit point coordinates so it tilts rather than shifts;
- **compound** two or more fields (`p1 + p2` / `compound`), summing them only
  where their volumes overlap — the coherent sum of several steered or displaced
  emissions on a shared grid.

Pressure is in pascals (or an amplitude in the units the simulation used);
coordinates are in millimetres, matching the public API everywhere else.
"""

from __future__ import annotations

import numpy as np

from .plotting import add_pressure_vol, create_3Dvol_mesh

__all__ = ["PressureField"]


class PressureField:
    """A simulated pressure volume that can be moved, drawn and compounded.

    Parameters
    ----------
    data : (Nx, Ny, Nz) or (Nt, Nx, Ny, Nz) numpy.ndarray
        Pressure samples on the grid — monochromatic amplitude ``(Nx, Ny, Nz)``
        or a transient ``(Nt, Nx, Ny, Nz)``. For spatial operations (moving,
        rendering, compounding) a transient must hold a single time frame
        (``Nt == 1``) or one is chosen with ``time_index``; a whole time series
        is not a single thing in space.
    coords : dict
        Grid coordinates as returned by the simulator: 1-D arrays ``"x"``,
        ``"y"``, ``"z"`` in millimetres (a transient also carries ``"t0"`` /
        ``"dt"``, left untouched here).
    scalars : str, optional
        Name of the scalar array on the rendered mesh. Default ``"Pressure"``.

    Notes
    -----
    The field is immutable: `transform` returns a new `PressureField` with the
    transform composed, leaving the original untouched, so a single simulation
    can be placed at several poses without recomputing it.
    """

    def __init__(self, data, coords, *, scalars="Pressure"):
        data = np.asarray(data)
        if data.ndim not in (3, 4):
            raise ValueError(
                f"data must be (Nx, Ny, Nz) or (Nt, Nx, Ny, Nz), got {data.shape}."
            )
        for axis in ("x", "y", "z"):
            if axis not in coords:
                raise ValueError(
                    f"coords is missing '{axis}' — a PressureField needs the grid "
                    "axes the simulator returns (run a grid simulation, not a point "
                    "cloud)."
                )
        self.data = data
        self.coords = coords
        self.scalars = scalars
        # A field is backed by either arrays+coords (the usual case) or a
        # prebuilt mesh (a compounded result lives on a resampled grid with no
        # axis-aligned coords). The pose is accumulated and applied once, at
        # mesh build, so chaining moves costs nothing until the field is drawn.
        self._mesh = None
        self._transform = np.eye(4)

    # -- moving in space ---------------------------------------------------

    def transform(self, matrix) -> PressureField:
        """Rigidly move the field (rotation then translation), returning a copy.

        Use the *same* 4x4 homogeneous transform you pass to
        `transducer.transform` (translation column in millimetres): the field
        and the aperture that made it then move together and stay registered.
        The move is deferred — baked into explicit point coordinates only when
        the field is rendered — so composing several moves is free.

        Parameters
        ----------
        matrix : (4, 4) array-like
            Homogeneous rigid-body transform; translation in millimetres.

        Returns
        -------
        PressureField
            A new field with ``matrix`` composed onto the current pose.
        """
        matrix = np.asarray(matrix, dtype=float)
        if matrix.shape != (4, 4):
            raise ValueError(f"transform matrix must be (4, 4), got {matrix.shape}.")
        moved = PressureField.__new__(PressureField)
        moved.data = self.data
        moved.coords = self.coords
        moved.scalars = self.scalars
        moved._mesh = self._mesh
        moved._transform = matrix @ self._transform
        return moved

    # -- rendering ---------------------------------------------------------

    def to_mesh(self, *, time_index=None):
        """Build the PyVista mesh, with the accumulated pose baked in.

        Returns a `StructuredGrid` (not `ImageData`): the pose lives in explicit
        point coordinates, so a steered field tilts correctly instead of merely
        shifting. Feed the result to any PyVista call, or to `add_pressure_vol`.

        Parameters
        ----------
        time_index : int, optional
            Which time frame to render for a transient field. Required when the
            field has more than one frame; ignored otherwise.

        Returns
        -------
        pyvista.StructuredGrid
            The field as explicit points, posed and ready to draw.
        """
        if self._mesh is not None:
            mesh = self._mesh
        else:
            mesh = create_3Dvol_mesh(
                self._frame(time_index),
                self.coords["x"],
                self.coords["y"],
                self.coords["z"],
                scalars=self.scalars,
                structured=True,
            )
        if not np.allclose(self._transform, np.eye(4)):
            mesh = mesh.transform(self._transform, inplace=False)
        return mesh

    def add_to(self, plotter, *, time_index=None, **kwargs):
        """Draw this field into an existing plotter (the overlay path).

        `add_pressure_vol` never calls ``show()``, so repeated `add_to` calls
        compose several fields (and the transducer) into one scene — the caller
        calls ``plotter.show()`` last. This sidesteps the trap where a one-shot
        plot shows (and so closes) the window before you can overlay anything.

        Parameters
        ----------
        plotter : pyvista.Plotter
            Plotter to draw into.
        time_index : int, optional
            Time frame for a transient field (see `to_mesh`).
        **kwargs
            Forwarded to `esdiva.plotting.add_pressure_vol`.

        Returns
        -------
        pyvista.Plotter
            The same plotter, with this field added.
        """
        return add_pressure_vol(
            self.to_mesh(time_index=time_index), plotter=plotter, **kwargs
        )

    def plot(self, *, time_index=None, show=True, **kwargs):
        """Render this field on its own plotter (the one-shot path).

        Parameters
        ----------
        time_index : int, optional
            Time frame for a transient field (see `to_mesh`).
        show : bool, optional
            Call ``plotter.show()`` before returning. Default True. Pass False
            to keep composing with `add_to` afterwards.
        **kwargs
            Forwarded to `esdiva.plotting.add_pressure_vol`.

        Returns
        -------
        pyvista.Plotter
            The plotter (shown when ``show`` is True).
        """
        plotter = add_pressure_vol(self.to_mesh(time_index=time_index), **kwargs)
        if show:
            plotter.show()
        return plotter

    # -- compounding -------------------------------------------------------

    def __add__(self, other) -> PressureField:
        """Compound two fields: sum them where their volumes overlap.

        Returns a field on *this* field's grid: the other field is resampled
        onto these points and added where it reaches, left as-is where it does
        not. The coherent sum of two steered or displaced emissions.
        """
        if not isinstance(other, PressureField):
            return NotImplemented
        return PressureField.compound([self, other], scalars=self.scalars)

    @classmethod
    def compound(cls, fields, *, onto=None, scalars="Pressure") -> PressureField:
        """Sum several fields on a shared grid, over their overlap only.

        Each field is resampled onto the canvas grid and added only where it
        actually covers it, so non-overlapping regions keep whichever field is
        present there. The on-grid coherent sum used to compound several steered
        or displaced emissions.

        Parameters
        ----------
        fields : sequence of PressureField
            Fields to compound; their current poses are honoured.
        onto : PressureField, optional
            Field whose grid is the canvas. Defaults to the first field. Points
            of any field falling outside this canvas are dropped, so pass the
            widest field (or a dedicated enclosing grid) as the canvas.
        scalars : str, optional
            Scalar name on the result mesh. Default ``"Pressure"``.

        Returns
        -------
        PressureField
            A field backed by the summed canvas mesh.
        """
        fields = list(fields)
        if not fields:
            raise ValueError("compound needs at least one field.")
        canvas = (onto or fields[0]).to_mesh()
        name = canvas.point_data.keys()[0]
        total = np.zeros(canvas.n_points, dtype=float)
        for field in fields:
            # Resample onto the canvas; vtkValidPointMask marks where the field
            # actually covers it, so only genuine overlap is summed.
            sampled = canvas.sample(field.to_mesh())
            mask = np.asarray(sampled["vtkValidPointMask"], dtype=bool)
            total[mask] += np.asarray(sampled[name])[mask]
        result = canvas.copy()
        result.clear_data()
        result[scalars] = total
        return cls._from_mesh(result, scalars=scalars)

    # -- internals ---------------------------------------------------------

    def _frame(self, time_index):
        """Return the single 3-D volume to render / move / compound."""
        if self.data.ndim == 3:
            return self.data
        nt = self.data.shape[0]
        if time_index is not None:
            return self.data[time_index]
        if nt == 1:
            return self.data[0]
        raise ValueError(
            f"This transient field has {nt} time frames; a frame is not a single "
            "thing in space. Pass time_index= to pick one."
        )
        # ponytail: single-frame only. Posing/compounding every frame of a
        # transient would mean one mesh per sample — add that if a moving
        # transient movie is ever needed.

    @classmethod
    def _from_mesh(cls, mesh, *, scalars="Pressure"):
        """Wrap an already-built (e.g. compounded) mesh, with no axis grid."""
        obj = cls.__new__(cls)
        obj.data = None
        obj.coords = None
        obj.scalars = scalars
        obj._mesh = mesh
        obj._transform = np.eye(4)
        return obj


def _self_check():
    """Round-trip a pose and compound two offset boxes (runnable sanity check)."""
    x = np.linspace(-2, 2, 9)
    coords = {"x": x, "y": x.copy(), "z": x.copy()}
    data = np.ones((9, 9, 9))
    field = PressureField(data, coords)

    # A rotation then its inverse must bring the mesh points back.
    theta = np.deg2rad(37.0)
    c, s = np.cos(theta), np.sin(theta)
    rot = np.eye(4)
    rot[[0, 0, 2, 2], [0, 2, 0, 2]] = [c, s, -s, c]  # rotate about y
    there = field.transform(rot).to_mesh().points
    back = field.transform(np.linalg.inv(rot) @ rot).to_mesh().points
    assert np.allclose(back, field.to_mesh().points), "pose round-trip failed"
    assert not np.allclose(there, field.to_mesh().points), "rotation was a no-op"

    # Two identical boxes, one shifted by half a box: the overlap sums to 2, the
    # non-overlap stays 1 (each field present alone there).
    shift = np.eye(4)
    shift[0, 3] = 2.0
    total = (field + field.transform(shift)).to_mesh()["Pressure"]
    assert np.isclose(total.max(), 2.0), f"overlap should sum to 2, got {total.max()}"
    assert np.isclose(total.min(), 1.0), f"non-overlap should stay 1, got {total.min()}"
    print("PressureField self-check passed.")


if __name__ == "__main__":
    _self_check()
