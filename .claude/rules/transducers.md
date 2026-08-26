---
paths:
  - "src/sondi/transducers/**"
---

# Transducer Design Rules

## Why Rectangular Patches

All transducer geometries decompose into small rectangular patches because the SIR
has a closed-form trapezoidal solution only for rectangles (see physics-context rule).
Patch dimensions must satisfy far-field condition: w << sqrt(4*l*c_0/f).

`no_sub_x` and `no_sub_y` control subdivision density and simulation accuracy.

## Coordinate System

- X: Lateral (across array elements)
- Y: Elevation (perpendicular to imaging plane)
- Z: Axial (beam propagation, depth)

## Z-Convention for Curved Transducers

Apex (surface centre) at z = 0, matching Field II `xdc_concave`/`xdc_convex`
and transducer datasheets. Focus/medium in +z direction.
- **ConcaveCircular**: apex at z = 0, rim at z = +sag, focus at z = +focus_mm (= R)
- **ConvexCircular**: apex at z = 0, rim at z = -sag, virtual focus at z = -R
- **FocusedCircular**: rim-referenced (2026-07-02, Field II lens convention) — face
  (curved-axis rim) at z = 0, centre line dished back to z = -sag; exposes
  `elevation_lens_sag = sag`; line focus at z = +R
- **FlatCircular**: entire face at z = 0

`sag = R - sqrt(R^2 - (D/2)^2)`

### Elevation lens datum (LinearArray / ConvexArray)

The cylindrical elevation lens (`elevation_focus_mm`) is **rim-referenced**: the element
face (rim, at `y = ±height/2`) sits at `z = 0` and the surface dishes **back** toward the
backing, so the centre (`y = 0`) is the deepest point at `z = -sag_edge`, where
`sag_edge = R - sqrt(R^2 - (height/2)^2)` and `R = elevation_focus`. This matches Field II
`xdc_focused_array` / `xdc_convex_focused_array`, which keep the flat element face at z = 0.
(`FocusedCircular` uses the same rim-referenced datum since 2026-07-02.)

`elevation_lens_sag` is **settable**: subclasses supply the geometric default via
`_default_elevation_lens_sag()`; assigning a value in metres overrides it (needed for
`FieldIITransducer` imports — or pass `elevation_focus_mm=` to the import, which
computes the sag from the imported patches' elevation extent). Assign `None` to
restore the default.

Reception adds the lens **group delay** `elevation_lens_sag / c` **per aperture** (TX once,
RX once) to `coords["t0"]`: the pulse-echo origin is referenced to the first-arriving rim,
but the focused elevation echo peaks one lens transit later. Exposed as the transducer
property `elevation_lens_sag` (metres; 0 for flat apertures). With this, a native
elevation-focused linear matches Field II pulse-echo RF at lag 0 (corr ~0.97 single
scatterer; residual is curved-patch far-field discretization, improves with `no_sub_y`).

## focus_mm Definition (Concave / Convex / Focused)

`focus_mm` = **focal length = radius of curvature** (the datasheet value, =
Field II `xdc_concave` `Rfocus`). `R = focus_mm` directly.
- `focus_mm = D/2` → hemisphere (`R = D/2`)
- Must satisfy `focus_mm >= D/2` (aperture radius); else ValueError
- Stored internally as `self.radius_of_curvature` in metres
- Spherical-cap subdivision still builds with rim at z=0 then shifts z by ±sag
  (`_shift_frames_z`) so the apex lands at z=0

## Mono-Element vs Multi-Element

`compute_delays()` and `compute_apodization()` = element-level beamforming
(linear, convex, matrix arrays).

For mono-element (all circular types):
- `compute_delays()` → ignored with warning (entire surface acts simultaneously)
- `compute_apodization()` → uniform weights with warning
- **Patch-wise apodization** still meaningful (edge apodization) via `set_apodization()`
- Geometric focusing = physical curvature, not electronic delays

## Surface Subdivision Methods

**ConcaveCircular / ConvexCircular** — `method` parameter:
- `"cartesian"` (default): arc-length reparameterised grid (`subdivide_parametric_surface`)
- `"spherical"`: ring-based tiling (`subdivide_spherical_cap`), preferred at high curvature or hemispheres

**Shared parameters** (all 4 circular types):
- `ratio_big_patches` (0-1, default 0.85): inner ring refinement (spherical) or border
  refinement zone (cartesian/flat)
- `refine_factor` (int, default 3): sub-rings (spherical) or subdivision factor (cartesian)

**Overlap warnings**: spherical warns if patch/R > 0.3; cartesian warns if coverage > 102%.

**`normalize_patch_size`** (bool, default False): sets wu/wv to arc-length step ignoring
Jacobian stretch. Produces uniform patches. Critical for hemispheres (Jacobian diverges
at rim). Available on ConcaveCircular and ConvexCircular (cartesian only).

**ConvexCircular cartesian**: uses arc-length reparameterization (`x = R*sin(sx/R)`)
matching ConcaveCircular. Previous direct `(x,y)` caused Jacobian → infinity near rim.

Removed params: `center_refine`, `border_refine`, `filled_radius_with_big_patches`,
`subdivision_method`.

## Adding New Transducer Type

1. Inherit from `TransducerBase`
2. Implement `_compute_element_centers()` — element positions
3. Implement `_build_subdivisions()` — rectangular patches
4. Export in `src/sondi/transducers/__init__.py`

Design for generalization — backward compatibility matters since new transducers
will be added over time.

## Lazy Geometry

`TransducerBase` uses lazy-loaded properties for element centers, patch vertices.
Computation deferred until first access.

## Unit Convention

User-facing APIs: millimeters (`_mm` suffix).
Internal computation: SI units (meters, seconds).
