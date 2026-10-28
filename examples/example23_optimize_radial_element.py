"""
Example 23: Three ways to mimic a diverging multilens element with a rigid cap.

The multi-lens array of Nardone et al. (Nat. Commun. 2025, s41467-025-64911-z)
widens each element's beam with a diverging epoxy/silicone doublet lens. eSDIva is
linear and homogeneous — no refraction — so the lens can only be modelled as a curved
*rigid* cap whose surface sag reproduces the same aperture phase. This example builds
and compares three candidate caps against the paper's measured directivity (a Bessel
bell crossing −6 dB at 33.17°):

1. **refractive doublet** — the paper's lens itself, collapsed through a refractive-index
   (Snell) model to one effective cap ``z(r) = −(n₂−n₁)·(R−√(R²−r²))`` (no optimization).
2. **spherical cap (r²)** — a single-term convex cap ``z(r) = −λ·c₀·(r/R)²``, with ``c₀``
   fitted on the paper's real 4.5 mm aperture (diameter fixed).
3. **free aspheric** — four even-power terms ``z(r) = −λ·Σ c_k·(r/R)^{2(k+1)}``, fitted
   freely including the diameter (coefficients may go negative → a non-convex aspheric).

The surface is even in ``r`` (axisymmetric, apex pinned to z=0), so only even powers
appear. Each cap is shown with `plotting.visualize_element`: surface profile and
directivity (found solid-red vs reference dotted-black), mesh, and xz field.

Run with:
    uv run python examples/example23_optimize_radial_element.py
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
from config import FIG_FOLDER, SAVE_FIG
from scipy.optimize import minimize
from scipy.special import j1

from esdiva import Emission
from esdiva.plotting import visualize_element
from esdiva.transducers import RadialProfileTransducer

C0, FC = 1540.0, 1e6
LAMBDA_MM = C0 / FC * 1e3
TARGET_HALF_DEG = 33.17  # paper's measured −6 dB half-angle with the lens
ANGLE = np.linspace(-60, 60, 241)
FIT = np.abs(ANGLE) <= 40.0  # fit the main lobe (±40°), where the bell shape matters
D_BOUNDS = (2.5, 5.5)

# Paper's doublet collapsed to one effective diverging cap (both faces R = 3.2 mm,
# epoxy c=2570 then silicone c=1015 → index contrast n₂−n₁ = c0/c₂ − c0/c₁).
R_LENS_M = 3.2e-3
N_CONTRAST = C0 / 1015.0 - C0 / 2570.0
D_PHYS_MM = 4.5  # the paper's piezo element size


def physical_sag(r):
    """Effective rigid-cap sag ``z(r) = −(n₂−n₁)·(R−√(R²−r²))`` (metres) of the lens."""
    rr = np.asarray(r, float)
    return -N_CONTRAST * (R_LENS_M - np.sqrt(np.clip(R_LENS_M**2 - rr**2, 0.0, None)))


def sag_of(coeffs, diameter_mm):
    """Radial sag ``z(r) = −λ·Σ c_k (r/R)^{2(k+1)}`` (metres) for a given diameter."""
    r_ap, c = diameter_mm / 2 * 1e-3, np.asarray(coeffs, float)
    return (
        lambda r: -LAMBDA_MM
        * 1e-3
        * sum(
            c[k] * np.clip(np.asarray(r, float) / r_ap, 0, 1) ** (2 * (k + 1))
            for k in range(len(c))
        )
    )


def make_element(diameter_mm, coeffs, no_sub_diameter=40):
    return RadialProfileTransducer(
        diameter_mm=diameter_mm,
        surface_sag_m=sag_of(coeffs, diameter_mm),
        frequency_Hz=FC,
        no_sub_diameter=no_sub_diameter,
        ratio_big_patches=0.7,
        refine_factor=2,
    )


def directivity_db(element):
    """Emitted ``|P|`` (dB, peak-normalised) on a 35 mm arc over ``ANGLE``."""
    th = np.deg2rad(ANGLE)
    pts = np.stack([35.0 * np.sin(th), np.zeros_like(th), 35.0 * np.cos(th)], 1)
    p = np.abs(
        Emission(element, fs=100e6, c=C0, monochromatic=True, verbose=False)(pts)[0]
    ).squeeze()
    return 20 * np.log10(p / p.max())


def half_angle_deg(dir_db):
    pos = ANGLE >= 0
    below = np.where(dir_db[pos] <= -6.0)[0]
    return float(ANGLE[pos][below[0]]) if below.size else float("nan")


# Target: unapodised-disc directivity ``2·J1(x)/x`` scaled to the paper half-angle.
beta = 2.21536 / np.sin(np.deg2rad(TARGET_HALF_DEG))
x = np.where(np.abs(ANGLE) < 1e-6, 1e-6, beta * np.sin(np.deg2rad(ANGLE)))
reference_db = np.clip(
    20 * np.log10(np.clip(np.abs(2 * j1(x) / x), 1e-6, None)), -20, None
)


def optimize_cap(n_terms, coeff_bounds, diameter_mm=None):
    """Fit ``n_terms`` even-power coefficients to the bell, seeded from the physical cap.

    ``coeff_bounds`` of ``(0, +)`` forces a convex diverging cap; allowing negatives
    lets the surface become a non-convex aspheric. If ``diameter_mm`` is given the
    diameter is fixed (only the shape is fitted — e.g. the sphere on the paper's real
    4.5 mm aperture); otherwise it is a free variable found by multistart."""
    r = np.linspace(0.0, D_PHYS_MM / 2 * 1e-3, 400)
    u = r / (D_PHYS_MM / 2 * 1e-3)
    basis = np.stack([u ** (2 * (k + 1)) for k in range(n_terms)], 1)
    seed = np.clip(
        np.linalg.lstsq(basis, -physical_sag(r) / (LAMBDA_MM * 1e-3), rcond=None)[0],
        coeff_bounds[0],
        coeff_bounds[1],
    )

    def loss(p):
        with warnings.catch_warnings():  # coarse trial profiles under-resolve by design
            warnings.simplefilter("ignore")
            dir_db = directivity_db(make_element(p[0], p[1:], no_sub_diameter=26))
        return float(
            np.mean((np.clip(dir_db, -20, None)[FIT] - reference_db[FIT]) ** 2)
        )

    # Multistart over the diameter: with a free diameter the loss vs D is multimodal
    # (a narrow best basin plus a flat shelf at the large-D bound), so a single
    # Nelder-Mead from one start can walk to the bound and is not reproducible. Restart
    # from a few diameters and keep the lowest-MSE result. A fixed diameter needs one.
    if diameter_mm is None:
        starts, d_bound = [np.r_[d0, seed] for d0 in (3.0, 3.7, 4.5, 5.2)], D_BOUNDS
    else:
        starts, d_bound = [np.r_[diameter_mm, seed]], (diameter_mm, diameter_mm)
    best = None
    for x0 in starts:
        res = minimize(
            loss,
            x0,
            method="Nelder-Mead",
            bounds=[d_bound] + [coeff_bounds] * n_terms,
            options={"maxiter": 400, "xatol": 1e-3, "fatol": 1e-6},
        )
        if best is None or res.fun < best.fun:
            best = res
    return float(best.x[0]), best.x[1:]


# Build the three candidate caps. The refractive doublet is the exact Snell sag (no fit);
# the sphere fits one convex term at the paper aperture; the aspheric fits four terms + D.
el_refr = RadialProfileTransducer(
    diameter_mm=D_PHYS_MM,
    surface_sag_m=physical_sag,
    frequency_Hz=FC,
    no_sub_diameter=40,
    ratio_big_patches=0.7,
    refine_factor=2,
)
D_sph, c_sph = optimize_cap(
    1, (0.0, 3.0), diameter_mm=D_PHYS_MM
)  # sphere on the paper aperture
D_asp, c_asp = optimize_cap(
    4, (-2.0, 2.0)
)  # free even polynomial (size floats), may be non-convex
el_sph = make_element(D_sph, c_sph)
el_asp = make_element(D_asp, c_asp)

models = [
    (el_refr, "refractive doublet (D=4.50 mm)", "refraction"),
    (el_sph, f"spherical cap r², D fixed to {D_sph:.2f} mm", "spherical"),
    (el_asp, f"free aspheric, 4 terms (D={D_asp:.2f} mm)", "aspheric"),
]
for el, name, _tag in models:
    print(f"  {name:34s} -6 dB = {half_angle_deg(directivity_db(el)):.1f} deg")

# Reference surface for the profile panels = the physical refraction cap.
r_ref_mm = np.linspace(0.0, D_PHYS_MM / 2, 200)
z_ref_um = physical_sag(r_ref_mm * 1e-3) * 1e6
z_ref_um -= z_ref_um.max()

if SAVE_FIG:
    FIG_FOLDER.mkdir(exist_ok=True)
for el, title, tag in models:
    save_path = str(FIG_FOLDER / f"ex23_{tag}.png") if SAVE_FIG else None
    visualize_element(
        el,
        c_m_s=C0,
        reference=(ANGLE, reference_db),
        reference_profile=(r_ref_mm, z_ref_um),
        title=title,
        save_path=save_path,
    )
    if SAVE_FIG:
        print(f"  Saved: {save_path}")
if not SAVE_FIG:
    plt.show()
