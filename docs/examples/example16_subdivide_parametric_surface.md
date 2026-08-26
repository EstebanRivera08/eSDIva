---
icon: lucide/grid-3x3
---

# Example 16: Parametric Surface Subdivision in rectangular patches (for transducers)

Demonstrates `subdivide_parametric_surface`, the public utility that tiles any
C1 parametric surface with flat tangent-plane rectangles — the patch
representation the SIR kernel requires, and the machinery behind the curved
circular transducers.

The example uses an **ellipsoidal cap** — a surface with strong curvature
variation from centre to rim — to show how patches follow the local tangent
plane and how `border_refine` controls how closely the mosaic follows the
aperture boundary.

[Source on GitHub](https://github.com/EstebanRivera08/SonDI/blob/main/examples/example16_subdivide_parametric_surface.py)

---

## Step 1 — Define the surface

```python
import numpy as np
from sondi.utilities.surface_subdivision import subdivide_parametric_surface

# Ellipsoidal cap: z(x, y) = c * sqrt(1 - x²/a² - y²/b²)
a, b, c = 30e-3, 20e-3, 15e-3   # semi-axes in metres
R_ap = 15e-3                      # aperture radius (circular mask)

def ellipsoid_cap(x, y):
    arg = max(1.0 - (x / a) ** 2 - (y / b) ** 2, 0.0)
    return np.array([x, y, c * np.sqrt(arg)])
```

Any `(u, v) → ndarray(3,)` function works here — no analytic derivatives
required.

---

## Step 2 — Subdivide

```python
frames = subdivide_parametric_surface(
    ellipsoid_cap,
    u_range=(-R_ap, R_ap),
    v_range=(-R_ap, R_ap),
    n_u=10, n_v=10,
    inside_fn=lambda x, y: x ** 2 / a ** 2 + y ** 2 / b ** 2 <= 1.0,
    normal_sign=1.0,
    border_refine=3,
)
```

Cells cut by the aperture boundary (`inside_fn` mixed across the cell
corners) are subdivided into `border_refine² = 9` sub-patches, so the mosaic
follows the elliptical rim closely without wasting patches in the interior.

---

## Step 3 — Inspect the output

```python
# frames is a dict with keys:
#   'corners'    — list of (4,3) arrays, one per patch (corner vertices in metres)
#   'centers'    — (M,3) patch centres on the surface
#   'normals'    — (M,3) unit outward normals
#   'tangents_u' — (M,3) first tangent axis
#   'tangents_v' — (M,3) second tangent axis (orthogonal to normal and tu)
#   'wu', 'wv'   — (M,) half-widths of each patch (metres)
#   'el_idx'     — (M,) element index per patch (all 0 for single-element)
#   'coverage'   — fraction of theoretical surface area covered by patches

print(f"Patches accepted: {frames['centers'].shape[0]}")
print(f"Coverage        : {frames['coverage']:.1%}")
```

---

## Step 4 — Matplotlib: 3-D mosaic + top-down area map

The left panel shows each flat rectangular patch in its local tangent plane,
coloured by area.  Outward normals are drawn as red arrows.  The right panel
is a top-down scatter plot showing how patch centres distribute across the
circular aperture.

![Ellipsoidal cap — 3-D mosaic and top-down area map](assets/ex16_subdivision_ellipsoid_cap.png)

---

## Step 5 — PyVista: theoretical surface vs flat patch mosaic

The flat patch mosaic (coloured by area) with the theoretical
surface as a semi-transparent cyan overlay.  The slight mismatch between
patches and surface near the rim is why the coverage metric can exceed 100 %:
flat tangent-plane patches lift slightly above the curved surface and their
areas sum to more than the actual curved area.

![Theoretical vs approximated surface](assets/ex16_subdivision_ellipsoid_cap_pyvista.png)

---

## Step 6 — Effect of subdivision density

Running the same subdivision at `n_u = n_v` = 6, 12, and 24 shows the accuracy
knob of the SIR method: each flat patch must stay inside the far-field limit
`w << sqrt(4·l·c/f)`, so a denser grid follows the curved surface more
faithfully at the cost of more patches (and SIR evaluations):

![subdivision density comparison](assets/ex16_subdivision_density_comparison.png)
