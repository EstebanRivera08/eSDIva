"""
Example 23: Three ways to mimic a diverging multilens element with a rigid cap.

The multilens array of Haidour et al. (Nat. Commun. 2025, s41467-025-64911-z) widens each
element's beam with a diverging epoxy/silicone doublet lens. eSDIva is linear and
homogeneous — no refraction — so the lens is modelled as a rigid curved cap whose surface
sag reproduces the same aperture phase. Three candidate caps are fitted to the paper's
measured directivity (a Bessel bell crossing −6 dB at 33.17°):

1. **refractive doublet** — the lens's own Snell sag ``z(r) = −(n₂−n₁)·(R−√(R²−r²))``, no fit.
2. **strictly convex cap** — four even-power terms, coefficients ≥ 0 (monotone, no dip).
3. **free aspheric** — same terms, coefficients free (a central dimple may form if it fits).

Each cap is shown with `plotting.visualize_element` (profile, directivity, mesh, xz field).

Run with:
    uv run python examples/example23_optimize_radial_element.py
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
from config import FIG_FOLDER, SAVE_FIG
from esdiva import Emission
from esdiva.plotting import visualize_element
from esdiva.transducers import RadialProfileTransducer
from scipy.optimize import minimize
from scipy.special import j1

C0, FC = 1540.0, 1e6
LAMBDA_MM = C0 / FC * 1e3
TARGET_HALF_DEG = 33.17  # paper's measured −6 dB half-angle with the lens
ANGLE = np.linspace(-60, 60, 241)
FIT = np.abs(ANGLE) <= 40.0  # fit the main lobe (±40°), where the bell shape matters
D_BOUNDS = (3.0, 5.5)

# Paper's doublet collapsed to one effective diverging cap (both faces R = 3.2 mm,
# epoxy c=2570 then silicone c=1015 → index contrast n₂−n₁ = c0/c₂ − c0/c₁).
R_LENS_MM = 3.2  # doublet radius of curvature (mm)
N_CONTRAST = C0 / 1015.0 - C0 / 2570.0
D_PHYS_MM = 4.5  # the paper's piezo element size


def physical_sag(r_mm):
    """Lens Snell sag ``z(r) = −(n₂−n₁)·(R−√(R²−r²))`` (mm)."""
    rr = np.asarray(r_mm, float)
    return -N_CONTRAST * (R_LENS_MM - np.sqrt(np.clip(R_LENS_MM**2 - rr**2, 0.0, None)))


def sag_of(coeffs, diameter_mm):
    """Even-power sag ``z(r) = −λ·Σ c_k (r/R)^{2(k+1)}`` (mm)."""
    r_ap, c = diameter_mm / 2, np.asarray(coeffs, float)
    return lambda r: (
        -LAMBDA_MM
        * sum(
            c[k] * np.clip(np.asarray(r, float) / r_ap, 0, 1) ** (2 * (k + 1))
            for k in range(len(c))
        )
    )


def make_element(diameter_mm, coeffs, no_sub_diameter=40):
    return RadialProfileTransducer(
        diameter_mm=diameter_mm,
        surface_sag_mm=sag_of(coeffs, diameter_mm),
        frequency_Hz=FC,
        no_sub_diameter=no_sub_diameter,
        ratio_big_patches=0.7,
        refine_factor=2,
    )


def directivity_db(element):
    """Emitted ``|P|`` (dB, peak-normalised) on a 35 mm arc vs ``ANGLE``."""
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


def optimize_cap(n_terms, coeff_bounds, diameter_mm=None, trace=None):
    """Fit ``n_terms`` even-power coefficients to the bell, seeded from the physical cap.

    ``coeff_bounds`` ``(0, +)`` forces a convex cap; allowing negatives gives a non-convex
    aspheric. ``diameter_mm`` given → diameter fixed (shape only); else it is a free
    variable found by multistart. Pass an empty list as ``trace`` to receive the winning
    start's ``[D, *coeffs]`` iterate path (for the GIF)."""
    r_mm = np.linspace(0.0, D_PHYS_MM / 2, 400)
    u = r_mm / (D_PHYS_MM / 2)
    basis = np.stack([u ** (2 * (k + 1)) for k in range(n_terms)], 1)
    seed = np.clip(
        np.linalg.lstsq(basis, -physical_sag(r_mm) / LAMBDA_MM, rcond=None)[0],
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

    # Free diameter: loss vs D is multimodal (one best basin + a flat shelf at the large-D
    # bound), so restart from a few diameters and keep the lowest MSE. Fixed D needs one.
    if diameter_mm is None:
        starts, d_bound = [np.r_[d0, seed] for d0 in (3.0, 3.7, 4.5, 5.2)], D_BOUNDS
    else:
        starts, d_bound = [np.r_[diameter_mm, seed]], (diameter_mm, diameter_mm)
    best, best_path = None, None
    for x0 in starts:
        path = [np.array(x0, float)]  # record each accepted simplex iterate
        res = minimize(
            loss,
            x0,
            method="Nelder-Mead",
            bounds=[d_bound] + [coeff_bounds] * n_terms,
            options={"maxiter": 400, "xatol": 1e-3, "fatol": 1e-6},
            callback=lambda xk: path.append(np.array(xk, float)),
        )
        if best is None or res.fun < best.fun:
            best, best_path = res, path
    if trace is not None:
        trace[:] = best_path
    return float(best.x[0]), best.x[1:]


# Reference surface for the profile panels = the physical refraction cap.
r_ref_mm = np.linspace(0.0, D_PHYS_MM / 2, 200)
z_ref_um = physical_sag(r_ref_mm) * 1e3  # mm -> µm
z_ref_um -= z_ref_um.max()


def optimized_aspheric(no_sub_diameter=40):
    """Fit and return the free-aspheric diverging cap (model 3; reused by example 24)."""
    D, c = optimize_cap(4, (-2.0, 2.0))
    return make_element(D, c, no_sub_diameter=no_sub_diameter)


def animate_fit(trace, gif_path, n_frames=60):
    """Stitch the aspheric fit's iterates into a GIF, one `visualize_element` per frame.

    The viewer watches the surface deform toward the target bell. The profile y-axis is
    pinned (0 to −1000 µm) so only the curve moves, not the axis."""
    from PIL import Image  # ships with matplotlib — no extra dependency

    steps = np.unique(np.linspace(0, len(trace) - 1, n_frames).astype(int))
    frames, tmp = [], FIG_FOLDER / "_frame.png"
    for i in steps:
        el = make_element(trace[i][0], trace[i][1:])
        ha = half_angle_deg(directivity_db(el))
        fig = visualize_element(
            el,
            c_m_s=C0,
            reference=(ANGLE, reference_db),
            reference_profile=(r_ref_mm, z_ref_um),
            profile_ylim=(-1000.0, 50.0),
            title=f"aspheric fit — step {i} (D={trace[i][0]:.2f} mm, −6 dB {ha:.0f}°)",
            save_path=str(tmp),
        )
        plt.close(fig)
        frames.append(Image.open(tmp).convert("RGB").resize((900, 760)))
    tmp.unlink()
    frames[0].save(
        gif_path, save_all=True, append_images=frames[1:], duration=150, loop=0
    )


if __name__ == "__main__":
    # Three candidate caps: Snell sag (no fit), convex (terms ≥ 0), aspheric (terms free).
    el_refr = RadialProfileTransducer(
        diameter_mm=D_PHYS_MM,
        surface_sag_mm=physical_sag,
        frequency_Hz=FC,
        no_sub_diameter=40,
        ratio_big_patches=0.7,
        refine_factor=2,
    )
    D_cvx, c_cvx = optimize_cap(4, (0.0, 2.0))
    asp_trace = []  # [D, *coeffs] at each Nelder-Mead iterate — animated below
    D_asp, c_asp = optimize_cap(4, (-2.0, 2.0), trace=asp_trace)
    el_cvx = make_element(D_cvx, c_cvx)
    el_asp = make_element(D_asp, c_asp)

    models = [
        (el_refr, "refractive doublet (D=4.50 mm)", "refraction"),
        (el_cvx, f"strictly convex cap, 4 terms (D={D_cvx:.2f} mm)", "convex"),
        (el_asp, f"free aspheric, 4 terms (D={D_asp:.2f} mm)", "aspheric"),
    ]
    for el, name, _tag in models:
        print(f"  {name:34s} -6 dB = {half_angle_deg(directivity_db(el)):.1f} deg")

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
    if SAVE_FIG:
        gif_path = str(FIG_FOLDER / "ex23_optimization.gif")
        animate_fit(asp_trace, gif_path)
        print(f"  Saved: {gif_path}  ({len(asp_trace)} iterates)")
    else:
        plt.show()
