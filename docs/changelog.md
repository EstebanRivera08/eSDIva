---
icon: lucide/history
---

# Changelog

All notable changes to SonDI are documented here.

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

- **`normalize_patch_size`** parameter on `ConcaveCircularTransducer` and
  `ConvexCircularTransducer` (cartesian method only, default `False`).
  When `True`, patch `wu`/`wv` are set to the arc-length parameter step
  (`ddu`, `ddv`) ignoring local Jacobian stretch. Produces uniform patch
  sizes across the aperture; critical for hemispheres.

- **`normalize_patch_size`** parameter on `subdivide_parametric_surface`
  (public API, same semantics as above).

---

## Earlier versions

No changelog maintained before this point.
