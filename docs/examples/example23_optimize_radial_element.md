---
icon: lucide/target
---

# Example 23: A Diverging Multilens Element as a Rigid Cap

The multi-lens array of Nardone et al. (*Nat. Commun.* 2025, s41467-025-64911-z)
widens each element's beam with a diverging epoxy/silicone doublet lens. eSDIva is
linear and homogeneous — **no refraction** — so the lens can only be modelled as a
curved **rigid cap** whose surface sag reproduces the same aperture phase. This example
builds three candidate caps and compares each against the paper's measured directivity
(a Bessel bell crossing **−6 dB at 33.17°**), using
[`visualize_element`][esdiva.plotting.visualize_element]:

1. **refractive doublet** — the paper's lens, collapsed through a refractive-index
   (Snell) model to one effective cap `z(r) = −(n₂−n₁)·(R−√(R²−r²))` (no optimization).
2. **strictly convex cap** — four even-power terms `z(r) = −λ·Σ cₖ(r/R)^{2(k+1)}` with
   **non-negative** coefficients, so the sag is monotone (apex at the centre, no dip) —
   the best purely convex rigid cap.
3. **free aspheric** — the same four terms but coefficients free to go negative, so a
   central dimple may form if it sharpens the agreement with the bell.

[Source on GitHub](https://github.com/EstebanRivera08/eSDIva/blob/main/examples/example23_optimize_radial_element.py)

---

## The parametric surface

An axisymmetric cap is **even in `r`** (symmetric about the axis, apex pinned to
`z = 0`), so only even powers appear — a Taylor expansion of `z(r²)`:

```python
import numpy as np

C0, FC = 1540.0, 1e6
LAMBDA_MM = C0 / FC * 1e3

def sag_of(coeffs, diameter_mm):
    r_ap, c = diameter_mm / 2 * 1e-3, np.asarray(coeffs, float)
    return lambda r: -LAMBDA_MM * 1e-3 * sum(
        c[k] * np.clip(np.asarray(r, float) / r_ap, 0, 1) ** (2 * (k + 1))
        for k in range(len(c))
    )
```

With **non-negative** coefficients every term increases outward, so `z(r)` decreases
monotonically — a convex *diverging* cap (apex at the centre). Allowing negative
coefficients lets a central dimple form (the free aspheric).

---

## Fitting to the directivity

```python
from scipy.optimize import minimize
from scipy.special import j1
from esdiva import Emission

# loss = mean-squared dB mismatch against the jinc bell inside the main lobe (±40°),
# with BOUNDED Nelder–Mead — the bounds keep D and the coeffs in range directly;
# clipping them inside the loss would flatten the objective past the edge and the
# simplex would drift onto that plateau. With a FREE diameter the loss vs D is
# multimodal, so the free-aspheric fit restarts from several diameters (multistart)
# and keeps the lowest-MSE result — a single start is not reproducible.
res = minimize(loss, x0, method="Nelder-Mead",
               bounds=[(3.0, 5.5)] + [coeff_bounds] * n_terms)
```

!!! note "Well-posedness"
    Matching the main lobe alone is **under-determined** — a smooth convex cap and a
    central-dimple aspheric fit nearly the same bell, and with a free diameter Nelder–Mead
    can even walk to the diameter bound. Two guards keep the example well-posed: the
    coefficient sign (non-negative → convex; free → the dimple can appear) separates the
    two shapes, and **multistart** over the diameter (restart from several diameters, keep
    the lowest-MSE basin) makes each fit reproducible on an interior diameter. The two
    solutions are near-degenerate at ±40°, which is exactly why constraining the sign is
    what cleanly distinguishes "best convex cap" from "best free cap".

---

## The three caps

Each is shown with the same four panels: **surface profile** `z(r)` and **directivity**
(found solid-red vs reference dotted-black), the patch **mesh**, and the **xz** pressure
plane. The refractive doublet reaches ~30°; both fitted caps reach ~34° but by different
shapes — the **strictly convex** cap (D≈4.5 mm) stays monotone, while the **free aspheric**
(D≈3.8 mm) develops a shallow central dimple that pulls the shoulder in a little tighter.
Neither fully removes the ±45° shoulder (spherical aberration) that the paper's real
multilens suppresses — a single rigid cap trades aperture phase for beam width, it cannot
reshape the lens's two-material refraction.

![Refractive doublet](assets/ex23_refraction.png)
![Strictly convex cap](assets/ex23_convex.png)
![Free aspheric](assets/ex23_aspheric.png)

---

## Watching the fit converge

Each Nelder–Mead iterate of the free-aspheric fit is rendered with the same four panels,
so the surface visibly deforms as the directivity closes on the bell — the central dimple
forming and the shoulder pulling in over the run.

![Aspheric fit evolution](assets/ex23_optimization.gif)
