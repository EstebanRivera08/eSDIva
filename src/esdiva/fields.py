"""Pressure fields you can move in space and combine.

A simulation returns a pressure volume on an axis grid. The medium is
homogeneous, so the *same* field rotates and translates with its aperture — you
move the result instead of re-simulating a steered or displaced probe.

`PressureField` wraps the ``(data, coords)`` an `Emission` (or beamformed
`Reception`) returns and gives it three operations:

- **move** it (`transform`) with the 4x4 you pass the transducer, so field and
  aperture stay registered;
- **render** it (`to_mesh` / `add_to` / `plot`) with the pose baked into points,
  so it tilts rather than shifts;
- **combine** fields as linear algebra: `resample` each onto one grid, then
  ``a + b``, ``a - b``, ``2 * a`` — the coherent sum (or difference) of views.

Pressure is in pascals (or the simulation's amplitude units); coordinates in
millimetres.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates

from .plotting import add_pressure_vol, create_3Dvol_mesh

__all__ = ["PressureField"]


class PressureField:
    """A pressure volume that can be moved, drawn and compounded.

    Parameters
    ----------
    data : (Nx, Ny, Nz) or (Nt, Nx, Ny, Nz) numpy.ndarray
        Pressure samples: monochromatic amplitude, or a transient (spatial ops
        need a single frame — ``Nt == 1`` or one picked with ``time_index``).
    coords : dict
        Grid axes from the simulator: 1-D ``"x"``, ``"y"``, ``"z"`` in mm.
    scalars : str, optional
        Scalar name on the rendered mesh. Default ``"Pressure"``.

    Notes
    -----
    Immutable: `transform` returns a new field, so one simulation can be placed
    at many poses without recomputing.
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
        # Backed by arrays+coords, or by a prebuilt mesh (a compounded result on a
        # resampled grid). The pose is accumulated and applied once at mesh build,
        # so chaining moves is free until the field is drawn.
        self._mesh = None
        self._transform = np.eye(4)

    # -- moving in space ---------------------------------------------------

    def transform(self, matrix) -> PressureField:
        """Rigidly move the field (the same 4x4 you pass the transducer).

        Deferred and chainable; translation column in millimetres.

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
        moved = self._copy()
        moved._transform = matrix @ self._transform
        return moved

    # -- rendering ---------------------------------------------------------

    def to_mesh(self, *, time_index=None):
        """Build the posed PyVista mesh.

        Returns a `StructuredGrid` with the pose in explicit points, so a steered
        field tilts (not merely shifts).

        Parameters
        ----------
        time_index : int, optional
            Time frame for a transient field; required when ``Nt > 1``.

        Returns
        -------
        pyvista.StructuredGrid
            The field as posed explicit points.
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
        """Draw into an existing plotter (the overlay path; caller shows last).

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
        """Render on a fresh plotter (the one-shot path).

        Parameters
        ----------
        time_index : int, optional
            Time frame for a transient field (see `to_mesh`).
        show : bool, optional
            Call ``plotter.show()`` before returning. Default True.
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

    # -- linear algebra on a shared grid -----------------------------------

    def resample(self, onto) -> PressureField:
        """Sample this field onto ``onto``'s grid (zero where it does not reach).

        The one step that brings differently-posed fields onto a common grid; `+`,
        `-`, scalar `*` then act point-by-point, so the coherent sum of views is
        ``sum(f.resample(canvas) for f in fields)``. On the common case (regular
        source grid, axis-aligned canvas) it maps the canvas points back through
        this field's inverse pose and interpolates the base grid directly — the
        field is never rotated or cell-located, an order of magnitude faster.

        Parameters
        ----------
        onto : PressureField
            Field whose sample points (with its pose) are the target.

        Returns
        -------
        PressureField
            This field's pressure on ``onto``'s grid.
        """
        # Fast path: uniform source grid, unposed axis canvas — no mesh, no locator.
        if (
            self.data is not None
            and onto.coords is not None
            and np.allclose(onto._transform, np.eye(4))
            and self._uniform_spacing() is not None
        ):
            return self._resample_trilinear(onto)
        # Fallback: mesh-backed source or posed/non-grid target has no base grid to
        # index — probe with VTK; vtkValidPointMask zeros uncovered points.
        target = onto.to_mesh().copy()
        target.clear_data()
        probed = target.sample(self.to_mesh())
        mask = np.asarray(probed["vtkValidPointMask"], dtype=bool)
        target[self.scalars] = np.where(mask, np.asarray(probed[self.scalars]), 0.0)
        return PressureField._from_mesh(target, scalars=self.scalars)

    def __mul__(self, scalar) -> PressureField:
        """Scale the pressure (``2 * field``); ``-1`` negates, giving ``a - b``."""
        if not np.isscalar(scalar):
            return NotImplemented
        out = self._copy()
        if out._mesh is not None:
            mesh = out._mesh.copy()
            mesh[self.scalars] = np.asarray(mesh[self.scalars]) * scalar
            out._mesh = mesh
        else:
            out.data = self.data * scalar
        return out

    __rmul__ = __mul__

    def __truediv__(self, scalar) -> PressureField:
        return self * (1.0 / scalar)

    def __neg__(self) -> PressureField:
        return self * -1.0

    def __sub__(self, other) -> PressureField:
        return self + (-other)

    def __radd__(self, other) -> PressureField:
        # Let sum([...]) work: its 0 seed is the additive identity.
        if other == 0 or other is None:
            return self
        return self.__add__(other)

    def __add__(self, other) -> PressureField:
        """Coherent sum (``a + b``), resampling ``other`` onto this grid if needed."""
        if not isinstance(other, PressureField):
            return NotImplemented
        # Both already on one resampled grid: add the point values directly.
        if (
            self._mesh is not None
            and other._mesh is not None
            and self._mesh.n_points == other._mesh.n_points
        ):
            # ponytail: equal point count stands in for "same grid" — true once both
            # were resampled onto one canvas, the documented way to add.
            total = np.asarray(self._mesh[self.scalars]) + np.asarray(
                other._mesh[other.scalars]
            )
            result = self._mesh.copy()
            result.clear_data()
            result[self.scalars] = total
            return PressureField._from_mesh(result, scalars=self.scalars)
        # Same axis grid and pose, still arrays: add them, stay lightweight.
        if (
            self._mesh is None
            and other._mesh is None
            and self._on_same_axis_grid(other)
        ):
            out = self._copy()
            out.data = self.data + other.data
            return out
        # Different grids: put both on this field's grid, then add their meshes.
        base = PressureField._from_mesh(self.to_mesh(), scalars=self.scalars)
        return base + other.resample(self)

    # -- internals ---------------------------------------------------------

    def _copy(self) -> PressureField:
        """Structural copy sharing data/mesh/pose; the caller mutates one field."""
        out = PressureField.__new__(PressureField)
        out.data = self.data
        out.coords = self.coords
        out.scalars = self.scalars
        out._mesh = self._mesh
        out._transform = self._transform
        return out

    def _on_same_axis_grid(self, other) -> bool:
        """True when two array fields share axes and pose (addable as arrays)."""
        if self.coords is None or other.coords is None:
            return False
        same_axes = all(
            np.array_equal(self.coords[a], other.coords[a]) for a in ("x", "y", "z")
        )
        return same_axes and np.allclose(self._transform, other._transform)

    def _uniform_spacing(self):
        """``(dx, dy, dz)`` if every axis is evenly spaced, else ``None``.

        Trilinear sampling needs a regular grid so a millimetre is a constant
        index step; a singleton axis (a 2-D plane) reports 1.0.
        """
        spacing = []
        for axis in ("x", "y", "z"):
            c = np.asarray(self.coords[axis], dtype=float)
            if c.size < 2:
                spacing.append(1.0)
                continue
            step = np.diff(c)
            if not np.allclose(step, step[0]):
                return None
            spacing.append(float(step[0]))
        return tuple(spacing)

    def _resample_trilinear(self, onto) -> PressureField:
        """Interpolate onto an axis canvas in this field's own frame (see `resample`)."""
        vol = self._frame(None)
        ox, oy, oz = (float(self.coords[a][0]) for a in ("x", "y", "z"))
        dx, dy, dz = self._uniform_spacing()
        cx, cy, cz = (np.asarray(onto.coords[a], dtype=float) for a in ("x", "y", "z"))
        gx, gy, gz = np.meshgrid(cx, cy, cz, indexing="ij")
        world = np.stack([gx.ravel(), gy.ravel(), gz.ravel(), np.ones(gx.size)])
        # Canvas points (mm) carried into this field's local frame, then to indices.
        local = np.linalg.inv(self._transform) @ world
        index = [(local[0] - ox) / dx, (local[1] - oy) / dy, (local[2] - oz) / dz]
        vals = map_coordinates(vol, index, order=1, mode="constant", cval=0.0)
        return PressureField(vals.reshape(gx.shape), onto.coords, scalars=self.scalars)

    def _frame(self, time_index):
        """The single 3-D volume to render / move / compound."""
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
        # ponytail: single-frame only. Add per-frame posing if a moving transient
        # movie is ever needed.

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

    # Two identical boxes, one shifted by half a box: overlap sums to 2, the rest 1.
    shift = np.eye(4)
    shift[0, 3] = 2.0
    total = (field + field.transform(shift)).to_mesh()["Pressure"]
    assert np.isclose(total.max(), 2.0), f"overlap should sum to 2, got {total.max()}"
    assert np.isclose(total.min(), 1.0), f"non-overlap should stay 1, got {total.min()}"

    # Linearity: 2*f doubles, f + (-f) cancels.
    assert np.isclose((2.0 * field).to_mesh()["Pressure"].max(), 2.0), "2*f failed"
    assert np.allclose((field - field).to_mesh()["Pressure"], 0.0), (
        "f - f should vanish"
    )

    # resample a grid-aligned shift onto a wider canvas: it lands on canvas nodes,
    # so the box must cross unchanged (all 0/1, every one accounted for).
    cx = np.arange(-3, 5.0 + 1e-9, 0.5)
    cyz = np.linspace(-2, 2, 9)
    canvas = PressureField(
        np.zeros((cx.size, 9, 9)), {"x": cx, "y": cyz, "z": cyz.copy()}
    )
    moved = field.transform(shift).resample(canvas).data
    assert np.all(np.isclose(moved, 0.0) | np.isclose(moved, 1.0)), "box not preserved"
    assert np.isclose(moved.sum(), data.size), (
        f"want {data.size} ones, got {moved.sum()}"
    )

    # A rotated view must stay bounded and finite (trilinear interpolates 0..1).
    rotated = field.transform(rot).resample(canvas).data
    assert np.isfinite(rotated).all() and rotated.max() <= 1.0 + 1e-9, (
        "rotate unbounded"
    )
    print("PressureField self-check passed.")


if __name__ == "__main__":
    _self_check()
