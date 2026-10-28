"""
Example 23: Can one rigid curved cap mimic a diverging multilens element?

The multi-lens array of Nardone et al. (Nat. Commun. 2025, s41467-025-64911-z)
widens each element's beam with a diverging epoxy/silicone doublet lens. eSDIva is
linear and homogeneous — no refraction — so the lens can only be modelled as a
curved *rigid* cap whose surface sag reproduces the same aperture phase. Collapsing
the paper's doublet through a refractive-index (Snell) model gives one effective
diverging cap ``z(r) = −(n₂−n₁)·(R−√(R²−r²))`` (R = 3.2 mm, n₂−n₁ ≈ 0.92).

This script seeds `scipy.optimize` from that physical cap and tunes a polynomial sag
``z(r) = −λ·Σ c_k (r/R)^{2(k+1)}`` to the paper's measured directivity (−6 dB at
33.17°), then shows with `plotting.visualize_element` how closely the cap matches.
Seeding from the physical cap matters: a random start matches the far-field bell but
lands on a non-physical (non-diverging) surface.

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
N_COEFFS = 10
ANGLE = np.linspace(-70, 70, 281)
FIT = np.abs(ANGLE) <= 50.0  # fit the whole bell over ±50°, not just the half-angle
Z_AXIS = np.linspace(0.5, 60, 300)  # on-axis sampling for the diverging penalty

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


def make_element(diameter_mm, coeffs, no_sub_diameter):
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


def axial_norm(element):
    """On-axis ``|P|`` vs ``Z_AXIS``, peak-normalised (flat/rising = diverging)."""
    pts = np.stack([np.zeros_like(Z_AXIS), np.zeros_like(Z_AXIS), Z_AXIS], 1)
    p = np.abs(
        Emission(element, fs=100e6, c=C0, monochromatic=True, verbose=False)(pts)[0]
    ).squeeze()
    return p / p.max()


def half_angle_deg(dir_db):
    pos = ANGLE >= 0
    below = np.where(dir_db[pos] <= -6.0)[0]
    return float(ANGLE[pos][below[0]]) if below.size else float("nan")


def profile_mm(coeffs, diameter_mm, n=200):
    """Surface profile ``(r, z)`` in mm, apex pinned to z=0 as the simulator tiles it."""
    r_mm = np.linspace(0.0, diameter_mm / 2, n)
    z_mm = sag_of(coeffs, diameter_mm)(r_mm * 1e-3) * 1e3
    return r_mm, z_mm - z_mm.max()


# Target: unapodised-disc directivity ``2·J1(x)/x`` scaled to the paper half-angle.
beta = 2.21536 / np.sin(np.deg2rad(TARGET_HALF_DEG))
x = np.where(np.abs(ANGLE) < 1e-6, 1e-6, beta * np.sin(np.deg2rad(ANGLE)))
reference_db = np.clip(
    20 * np.log10(np.clip(np.abs(2 * j1(x) / x), 1e-6, None)), -20, None
)

# Physical seed: fit the paper's effective cap to the polynomial basis.
r_seed = np.linspace(0.0, D_PHYS_MM / 2 * 1e-3, 400)
u_seed = r_seed / (D_PHYS_MM / 2 * 1e-3)
basis = np.stack([u_seed ** (2 * (k + 1)) for k in range(N_COEFFS)], 1)
c_seed = np.linalg.lstsq(basis, -physical_sag(r_seed) / (LAMBDA_MM * 1e-3), rcond=None)[
    0
]


def loss(p):
    diameter_mm = float(np.clip(p[0], 2.5, 6.0))
    with (
        warnings.catch_warnings()
    ):  # coarse trial profiles are expected to under-resolve
        warnings.simplefilter("ignore")
        el = make_element(diameter_mm, np.clip(p[1:], -2, 2), no_sub_diameter=26)
        dir_db, ax = directivity_db(el), axial_norm(el)
    shape = float(np.mean((np.clip(dir_db, -20, None)[FIT] - reference_db[FIT]) ** 2))
    # Keep the cap diverging: penalise any on-axis climb past the near field.
    far = Z_AXIS[:-1] > diameter_mm**2 / (4 * LAMBDA_MM)
    focus = float(np.clip(np.diff(ax)[far], 0, None).sum())
    return shape + 2.0 * focus


res = minimize(
    loss,
    np.r_[D_PHYS_MM, c_seed],
    method="Nelder-Mead",
    options={"maxiter": 400, "xatol": 1e-3, "fatol": 1e-5},
)
D = float(np.clip(res.x[0], 2.5, 6.0))
coeffs = np.clip(res.x[1:], -2, 2)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    seed_half = half_angle_deg(directivity_db(make_element(D_PHYS_MM, c_seed, 40)))
el = make_element(D, coeffs, no_sub_diameter=40)
opt_half = half_angle_deg(directivity_db(el))
print(f"physical seed: D = {D_PHYS_MM:.2f} mm, -6 dB = {seed_half:.1f} deg")
print(
    f"optimized:     D = {D:.3f} mm, -6 dB = {opt_half:.1f} deg  (target {TARGET_HALF_DEG} deg)"
)
print(f"coeffs = {np.array2string(coeffs, precision=3)}")

if SAVE_FIG:
    FIG_FOLDER.mkdir(exist_ok=True)

# Profile check: how the surface changed from the physical seed to the optimum.
rs_mm, zs_mm = profile_mm(c_seed, D_PHYS_MM)
ro_mm, zo_mm = profile_mm(coeffs, D)
fig_p, axp = plt.subplots(figsize=(6, 4))
axp.plot(rs_mm, zs_mm * 1e3, "k--", lw=2, label=f"physical seed (D={D_PHYS_MM:.1f} mm)")
axp.plot(ro_mm, zo_mm * 1e3, "C0", lw=2, label=f"optimized (D={D:.2f} mm)")
axp.set(xlabel="r (mm)", ylabel="z (µm)", title="radial surface profile z(r)")
axp.legend()
axp.grid(alpha=0.3)
fig_p.tight_layout()
if SAVE_FIG:
    fig_p.savefig(FIG_FOLDER / "ex23_profiles.png", dpi=130)
    print(f"  Saved: {FIG_FOLDER / 'ex23_profiles.png'}")

save_path = str(FIG_FOLDER / "ex23_optimized_element.png") if SAVE_FIG else None
fig = visualize_element(
    el,
    c_m_s=C0,
    reference=(ANGLE, reference_db),
    title=f"multilens cap vs paper directivity (D = {D:.2f} mm)",
    save_path=save_path,
)
if SAVE_FIG:
    print(f"  Saved: {save_path}")
else:
    plt.show()
