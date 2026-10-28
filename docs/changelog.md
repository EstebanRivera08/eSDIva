---
icon: lucide/history
---

# Changelog

All notable changes to eSDIva are documented here.

---

## [Unreleased]

### Fixed

- **Z-convention for curved transducers** — rim is now always at `z = 0`
  across all tiling methods and transducer types.
  - `ConcaveCircularTransducer` (spherical & cartesian): pole moved from
    `z = 0` to `z = -sag`; rim now at `z = 0`.
  - `FocusedCircularTransducer`: curved-axis edges at `z = 0`, centre at
    `z = -sag` (was centre at `z = 0`, edges lifted toward `z = +sag`).
  - `ConvexCircularTransducer` was already correct (apex at `z = +sag`, rim
    at `z = 0`).
  - `FlatCircularTransducer` unaffected (always at `z = 0`).

- **ConvexCircularTransducer cartesian parameterisation** — replaced direct
  `(x, y)` Cartesian grid with arc-length reparameterisation
  `x = R·sin(sx/R)`, matching `ConcaveCircularTransducer`. The old grid
  caused Jacobian → ∞ near the rim of high-curvature surfaces (e.g.
  hemispheres), producing patch sizes up to ~35 mm instead of the expected
  ~5 mm.

### Added

- **`subdivide_radial`** — polar concentric-ring tiler for any axisymmetric
  surface `z(r)`; exact rectangular patches, exact circular rim, no shear.
- **`RadialProfileTransducer`** — single element with an arbitrary radial
  profile `z(r)` (dome, bowl, aspheric lens); `max z(r)` pinned to `z = 0`.
- **`plotting.visualize_element`** — four-panel element diagnostic (surface
  profile `z(r)`, directivity, mesh, xz plane) from one monochromatic emission;
  `reference`/`reference_profile` overlay a target (dotted) against the found (red).
- **Example 23** — model a diverging multilens element as a rigid cap, comparing a
  refractive-doublet, a spherical `r²`, and a free aspheric fit with `scipy.optimize`;
  includes an animated GIF of the aspheric fit converging toward the target directivity.
- **`normalize_patch_size`** parameter on `ConcaveCircularTransducer` and
  `ConvexCircularTransducer` (cartesian method only, default `False`).
  When `True`, patch `wu`/`wv` are set to the arc-length parameter step
  (`ddu`, `ddv`) ignoring local Jacobian stretch. Produces uniform patch
  sizes across the aperture; critical for hemispheres.

- **`normalize_patch_size`** parameter on `subdivide_cartesian`
  (public API, same semantics as above).

### Changed

- **`subdivide_parametric_surface` renamed to `subdivide_cartesian`** (and now
  warns when the surface tangents shear from orthogonal). `subdivide_spherical_cap`
  removed — the curved circular transducers tile via `subdivide_radial`.
- **Circular transducer `method="spherical"` renamed to `method="radial"`**
  (`ConcaveCircularTransducer`, `ConvexCircularTransducer`).

---

## Earlier versions

No changelog maintained before this point.
