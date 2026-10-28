"""
Example 23: Optimize a radial-profile element for a target directivity.

Shapes the surface ``z(r)`` of a single `RadialProfileTransducer` so its emitted
beam matches a target −6 dB half-angle. The profile is a polynomial sag
``z(r) = −λ · Σ c_k (r/R)^{2(k+1)}``; `scipy.optimize` tunes the diameter and the
coefficients to fit an ideal Bessel directivity, then `plotting.visualize_element`
shows the mesh, directivity, on-axis field and xz plane of the result.

Run with:
    uv run examples/example23_optimize_radial_element.py
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
TARGET_HALF_DEG = 33.0
N_COEFFS = 10
ANGLE = np.linspace(-70, 70, 281)
FIT = np.abs(ANGLE) <= 45.0


def sag_of(coeffs, diameter_mm):
    """Radial sag ``z(r) = −λ·Σ c_k (r/R)^{2(k+1)}`` (metres) for a given diameter."""
    r_ap, c = diameter_mm / 2 * 1e-3, np.asarray(coeffs, float)
    return lambda r: -LAMBDA_MM * 1e-3 * sum(
        c[k] * np.clip(np.asarray(r, float) / r_ap, 0, 1) ** (2 * (k + 1)) for k in range(len(c))
    )


def directivity_db(element):
    """Emitted ``|P|`` (dB, peak-normalised) on a 35 mm arc over ``ANGLE``."""
    th = np.deg2rad(ANGLE)
    pts = np.stack([35.0 * np.sin(th), np.zeros_like(th), 35.0 * np.cos(th)], 1)
    p = np.abs(Emission(element, fs=100e6, c=C0, monochromatic=True, verbose=False)(pts)[0]).squeeze()
    return 20 * np.log10(p / p.max())


# Ideal bell: an unapodised disc directivity ``2·J1(x)/x`` scaled to the target.
beta = 2.21536 / np.sin(np.deg2rad(TARGET_HALF_DEG))
x = np.where(np.abs(ANGLE) < 1e-6, 1e-6, beta * np.sin(np.deg2rad(ANGLE)))
reference_db = np.clip(20 * np.log10(np.clip(np.abs(2 * j1(x) / x), 1e-6, None)), -20, None)


def loss(p):
    diameter_mm = float(np.clip(p[0], 2.5, 6.0))
    with warnings.catch_warnings():  # coarse trial profiles are expected to under-resolve
        warnings.simplefilter("ignore")
        el = RadialProfileTransducer(
            diameter_mm=diameter_mm, surface_sag_m=sag_of(np.clip(p[1:], -2, 2), diameter_mm),
            frequency_Hz=FC, no_sub_diameter=16,
        )
        dir_db = directivity_db(el)
    return float(np.mean((np.clip(dir_db, -20, None)[FIT] - reference_db[FIT]) ** 2))


p0 = np.r_[4.0, np.full(N_COEFFS, 0.1)]
res = minimize(loss, p0, method="Nelder-Mead",
               options={"maxiter": 200, "xatol": 1e-3, "fatol": 1e-4})
D = float(np.clip(res.x[0], 2.5, 6.0))
coeffs = np.clip(res.x[1:], -2, 2)
print(f"D = {D:.3f} mm, coeffs = {np.array2string(coeffs, precision=3)}")

el = RadialProfileTransducer(
    diameter_mm=D, surface_sag_m=sag_of(coeffs, D), frequency_Hz=FC, no_sub_diameter=30,
)
save_path = str(FIG_FOLDER / "ex23_optimized_element.png") if SAVE_FIG else None
if SAVE_FIG:
    FIG_FOLDER.mkdir(exist_ok=True)
fig = visualize_element(el, c_m_s=C0, reference=(ANGLE, reference_db),
                        title=f"optimized radial element (D = {D:.2f} mm)", save_path=save_path)
if SAVE_FIG:
    print(f"  Saved: {save_path}")
else:
    plt.show()
