"""Volumes you can move in space and combine: pressure fields, PSFs, envelopes.

A simulation returns a scalar volume on an axis grid — a pressure field from
`Emission`, a PSF or envelope from `Reception` + beamforming, an intensity map.
The medium is homogeneous, so the *same* volume rotates and translates with its
aperture: move the result instead of re-simulating a steered or displaced probe.

`Field` wraps ``(data, coords)`` and gives it three operations:

- **move** it (`transform`) with the 4x4 you pass the transducer, so volume and
  aperture stay registered;
- **combine** volumes as linear algebra on one grid: build a `canvas`, `resample`
  each posed volume onto it, then ``a + b``, ``a - b``, ``a * b``, ``2 * a`` act
  point by point (coherent sum of views, difference maps, TX x RX products);
- **render** it (`to_mesh` / `add_to` / `plot`) with the pose baked into points,
  so it tilts rather than shifts.

Recommended steps::

    canvas = Field.canvas({"x_extent": [-10, 10], "y_extent": [-5, 5],
                           "z_extent": [20, 60], "dx": 0.1, "dy": 0.1, "dz": 0.1})
    views = [Field(p, coords).transform(T) for (p, coords), T in results]
    total = sum(v.resample(canvas) for v in views)   # every view on one grid
    total.plot()                                     # or total.data / total.coords

Values keep the simulation's units (Pa for pressure); coordinates are in mm.
"""

from __future__ import annotations

import operator

import numpy as np
from scipy.ndimage import map_coordinates

from .plotting import add_pressure_vol, create_3Dvol_mesh
from .utilities.helper_functions import create_spatial_grid_from_dict

__all__ = ["Field"]


class Field:
    """A scalar volume (pressure, PSF, envelope, ...) that can be moved, drawn and combined.

    Parameters
    ----------
    data : (Nx, Ny, Nz) or (Nt, Nx, Ny, Nz) numpy.ndarray
        Samples on the grid: a monochromatic amplitude, a PSF/envelope volume, or a
        transient (spatial ops need a single frame — ``Nt == 1`` or one picked
        with ``time_index``).
    coords : dict
        Grid axes in mm, as returned: 1-D ``"x"``, ``"y"``, ``"z"`` (simulators) or
        ``"x_mm"``, ``"y_mm"``, ``"z_mm"`` (`das_volume`).
    scalars : str, default: "Pressure"
        Name of the quantity on the rendered mesh and its colour bar (e.g. ``"PSF"``).

    Notes
    -----
    Immutable: every operation returns a new field, so one simulation can be
    placed at many poses without recomputing.
    """

    def __init__(self, data, coords, *, scalars="Pressure"):
        data = np.asarray(data)
        if data.ndim not in (3, 4):
            raise ValueError(
                f"data must be (Nx, Ny, Nz) or (Nt, Nx, Ny, Nz), got {data.shape}."
            )
        # Simulators name the axes "x"/"y"/"z"; beamformers "x_mm"/"y_mm"/"z_mm".
        coords = {**coords}
        for axis in ("x", "y", "z"):
            if axis not in coords and f"{axis}_mm" in coords:
                coords[axis] = coords[f"{axis}_mm"]
            if axis not in coords:
                raise ValueError(
                    f"coords is missing '{axis}' — a Field needs the grid axes the "
                    "simulator or beamformer returns (a grid, not a point cloud)."
                )
        self.data = data
        self.coords = coords
        self.scalars = scalars
        # Backed by arrays+coords, or by a prebuilt mesh (a result on a posed or
        # irregular grid). The pose is accumulated and applied once at mesh build,
        # so chaining moves is free until the field is drawn or resampled.
        self._mesh = None
        self._transform = np.eye(4)

    @classmethod
    def canvas(cls, grid, *, scalars="Pressure") -> Field:
        """An all-zero field on a regular grid: the common target for `resample`.

        Use it as the one grid every posed volume is brought onto before they are
        added, subtracted or multiplied. Make it wide enough to hold every view —
        points of a view that fall outside the canvas are dropped.

        Parameters
        ----------
        grid : dict
            The same grid dict `Emission` takes: ``x_extent``, ``y_extent``,
            ``z_extent`` (``[min, max]`` in mm) and ``dx``, ``dy``, ``dz`` (mm).
        scalars : str, default: "Pressure"
            Name of the quantity the canvas will carry.

        Returns
        -------
        Field
            Zeros on that grid.
        """
        x, y, z, _ = create_spatial_grid_from_dict(grid)
        return cls(
            np.zeros((x.size, y.size, z.size)),
            {"x": x, "y": y, "z": z},
            scalars=scalars,
        )

    # -- moving in space ---------------------------------------------------

    def transform(self, matrix) -> Field:
        """Rigidly move the field (the same 4x4 you pass the transducer).

        Deferred and chainable; translation column in millimetres.

        Parameters
        ----------
        matrix : (4, 4) array-like
            Homogeneous rigid-body transform; translation in millimetres.

        Returns
        -------
        Field
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
        show : bool, default: True
            Call ``plotter.show()`` before returning.
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

    def resample(self, onto) -> Field:
        """Sample this field onto ``onto``'s grid (zero where it does not reach).

        The one step that brings differently-posed volumes onto a common grid; the
        arithmetic operators then act point by point, so the coherent sum of views
        is ``sum(f.resample(canvas) for f in fields)``. When ``onto`` is an unposed
        axis grid (a `canvas`) the result keeps plain ``data``/``coords`` arrays:
        canvas points are mapped back through this field's inverse pose and
        trilinearly interpolated — no mesh, an order of magnitude faster.

        Parameters
        ----------
        onto : Field
            Field whose sample points (with its pose) are the target — usually a
            `canvas`.

        Returns
        -------
        Field
            This field's values on ``onto``'s grid.
        """
        if (
            self.data is not None
            and onto.coords is not None
            and np.allclose(onto._transform, np.eye(4))
            and self._uniform_spacing() is not None
        ):
            return self._resample_trilinear(onto)
        # Mesh-backed source or posed target has no base grid to index — probe with
        # VTK; vtkValidPointMask zeros the uncovered points.
        target = onto.to_mesh().copy()
        target.clear_data()
        probed = target.sample(self.to_mesh())
        mask = np.asarray(probed["vtkValidPointMask"], dtype=bool)
        target[self.scalars] = np.where(mask, np.asarray(probed[self.scalars]), 0.0)
        return Field._from_mesh(target, scalars=self.scalars)

    def _combine(self, other, op) -> Field:
        """Apply ``op`` point by point, resampling ``other`` onto this grid if needed."""
        if np.isscalar(other):
            out = self._copy()
            if out._mesh is not None:
                mesh = out._mesh.copy()
                mesh[self.scalars] = op(np.asarray(mesh[self.scalars]), other)
                out._mesh = mesh
            else:
                out.data = op(self.data, other)
            return out
        if not isinstance(other, Field):
            return NotImplemented
        # Same axis grid and pose, still arrays: operate on them directly.
        if (
            self._mesh is None
            and other._mesh is None
            and self._on_same_axis_grid(other)
        ):
            out = self._copy()
            out.data = op(self.data, other.data)
            return out
        # Both already on one resampled grid.
        if (
            self._mesh is not None
            and other._mesh is not None
            and self._mesh.n_points == other._mesh.n_points
        ):
            # ponytail: equal point count stands in for "same grid" — true once both
            # were resampled onto one canvas, the documented way to combine.
            result = self._mesh.copy()
            result.clear_data()
            result[self.scalars] = op(
                np.asarray(self._mesh[self.scalars]),
                np.asarray(other._mesh[other.scalars]),
            )
            return Field._from_mesh(result, scalars=self.scalars)
        # Different grids: put other on this field's grid first.
        if self._mesh is None and np.allclose(self._transform, np.eye(4)):
            return self._combine(other.resample(self), op)
        base = Field._from_mesh(self.to_mesh(), scalars=self.scalars)
        return base._combine(other.resample(self), op)

    def __add__(self, other) -> Field:
        """Coherent sum ``a + b`` (``other`` resampled onto this grid if needed)."""
        return self._combine(other, operator.add)

    def __radd__(self, other) -> Field:
        # Let sum([...]) work: its 0 seed is the additive identity.
        if other == 0 or other is None:
            return self
        return self._combine(other, operator.add)

    def __sub__(self, other) -> Field:
        """Difference ``a - b``, e.g. two methods or two poses compared point by point."""
        return self._combine(other, operator.sub)

    def __mul__(self, other) -> Field:
        """``2 * a`` scales; ``a * b`` multiplies point by point (TX x RX, masks)."""
        return self._combine(other, operator.mul)

    __rmul__ = __mul__

    def __truediv__(self, other) -> Field:
        return self._combine(other, operator.truediv)

    def __pow__(self, exponent) -> Field:
        """``a ** 2``: envelope to power, for incoherent (power) compounding of views."""
        return self._combine(exponent, operator.pow)

    def __neg__(self) -> Field:
        return self * -1.0

    # -- internals ---------------------------------------------------------

    def _copy(self) -> Field:
        """Structural copy sharing data/mesh/pose; the caller mutates one field."""
        out = Field.__new__(Field)
        out.data = self.data
        out.coords = self.coords
        out.scalars = self.scalars
        out._mesh = self._mesh
        out._transform = self._transform
        return out

    def _on_same_axis_grid(self, other) -> bool:
        """True when two array fields share axes and pose (combinable as arrays)."""
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

    def _resample_trilinear(self, onto) -> Field:
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
        return Field(vals.reshape(gx.shape), onto.coords, scalars=self.scalars)

    def _frame(self, time_index):
        """The single 3-D volume to render / move / combine."""
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
    """Round-trip a pose, compound offset boxes, multiply fields (runnable check)."""
    x = np.linspace(-2, 2, 9)
    coords = {"x": x, "y": x.copy(), "z": x.copy()}
    data = np.ones((9, 9, 9))
    field = Field(data, coords)

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
    total = (field + field.transform(shift)).data
    assert np.isclose(total.max(), 2.0), f"overlap should sum to 2, got {total.max()}"
    assert np.isclose(total.min(), 1.0), f"non-overlap should stay 1, got {total.min()}"

    # Linearity: 2*f doubles, f - f cancels, f * f squares.
    assert np.isclose((2.0 * field).data.max(), 2.0), "2*f failed"
    assert np.allclose((field - field).data, 0.0), "f - f should vanish"
    assert np.allclose((3 * field * field).data, 3.0), "f * f failed"

    # Recommended path: canvas from a grid dict, resample, sum.
    canvas = Field.canvas(
        {
            "x_extent": [-3, 5],
            "y_extent": [-2, 2],
            "z_extent": [-2, 2],
            "dx": 0.5,
            "dy": 0.5,
            "dz": 0.5,
        }
    )
    moved = field.transform(shift).resample(canvas).data
    assert np.all(np.isclose(moved, 0.0) | np.isclose(moved, 1.0)), "box not preserved"
    assert np.isclose(moved.sum(), data.size), (
        f"want {data.size} ones, got {moved.sum()}"
    )
    both = sum(f.resample(canvas) for f in (field, field.transform(shift)))
    assert np.isclose(both.data.max(), 2.0), "canvas sum lost the overlap"

    # A rotated view must stay bounded and finite (trilinear interpolates 0..1).
    rotated = field.transform(rot).resample(canvas).data
    assert np.isfinite(rotated).all() and rotated.max() <= 1.0 + 1e-9, (
        "rotate unbounded"
    )
    print("Field self-check passed.")


if __name__ == "__main__":
    _self_check()
