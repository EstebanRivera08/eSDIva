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
- **`RadialProfileTransducer.virtual_source_m`** — the cap's acoustic centre (centre
  of curvature) from a least-squares circle fit of its surface; the point a beamformer
  must reference a diverging/converging lens element to instead of the `z = 0` plane.
- **`plotting.visualize_element`** — 2×2-panel element diagnostic (top row surface
  profile `z(r)` and directivity, bottom row mesh and xz plane) from one monochromatic
  emission; `reference`/`reference_profile` overlay a target (dotted) against the found (red).
- **Example 23** — model a diverging multilens element as a rigid cap, comparing a
  refractive-doublet, a strictly-convex, and a free-aspheric fit with `scipy.optimize`;
  includes an animated GIF of the aspheric fit converging toward the target directivity.
- **Example 24** — assemble the paper's 252-element (18×14) diverging-multilens matrix
  from the example-23 cap with `CustomTransducer`; reproduce its depth-resolved Fig. 2
  point PSF (three targets, full-aperture receive) with the central-4×4 16-source
  diverging sequence (`sequence_rf` + `das_volume`), referencing the beamformer to each
  lens's `virtual_source_m`, and a grayscale, linear-opacity dB volume render.
- **`normalize_patch_size`** parameter on `ConcaveCircularTransducer` and
  `ConvexCircularTransducer` (cartesian method only, default `False`).
  When `True`, patch `wu`/`wv` are set to the arc-length parameter step
  (`ddu`, `ddv`) ignoring local Jacobian stretch. Produces uniform patch
  sizes across the aperture; critical for hemispheres.

- **`normalize_patch_size`** parameter on `subdivide_cartesian`
  (public API, same semantics as above).

### Changed

- **`RadialProfileTransducer` surface callable is now millimetres** — the constructor
  argument `surface_sag_m` (metres in/out) became **`surface_sag_mm`** (`r_mm -> z_mm`),
  matching the mm convention of `diameter_mm` on the same constructor. The callable is
  converted to SI internally. **Breaking:** pass a mm profile and drop the `1e-3` factors.
- **`subdivide_parametric_surface` renamed to `subdivide_cartesian`** (and now
  warns when the surface tangents shear from orthogonal). `subdivide_spherical_cap`
  removed — the curved circular transducers tile via `subdivide_radial`.
- **Circular transducer `method="spherical"` renamed to `method="radial"`**
  (`ConcaveCircularTransducer`, `ConvexCircularTransducer`).

---

## Earlier versions

No changelog maintained before this point.
