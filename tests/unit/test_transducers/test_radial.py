"""Radial subdivision and RadialProfileTransducer: geometry + far-field checks."""

import numpy as np
import pytest

from esdiva import Emission
from esdiva.transducers import (
    ConvexCircularTransducer,
    RadialProfileTransducer,
)
from esdiva.utilities import subdivide_radial

R = 3e-3  # radius of curvature for the spherical test caps
CONVEX = lambda r: np.sqrt(np.clip(R**2 - np.asarray(r, float) ** 2, 0.0, None)) - R
CONCAVE = lambda r: R - np.sqrt(np.clip(R**2 - np.asarray(r, float) ** 2, 0.0, None))
ASPHERIC = lambda r: -8e-4 * (np.asarray(r, float) / 2e-3) ** 2 - 3e-4 * (np.asarray(r, float) / 2e-3) ** 4


def _max_corner_angle_dev(element):
    """Largest deviation (deg) of any patch corner angle from 90°."""
    q = np.array(element.sub_quad_verts)  # (M, 4, 3)
    c0, c1, c2, c3 = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    def angle(a, b, c):
        u, v = a - b, c - b
        cos = (u * v).sum(-1) / (np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1))
        return np.degrees(np.arccos(np.clip(cos, -1, 1)))

    corners = np.stack(
        [angle(c3, c0, c1), angle(c0, c1, c2), angle(c1, c2, c3), angle(c2, c3, c0)], 1
    )
    return float(np.max(np.abs(corners - 90.0)))


@pytest.mark.parametrize("sag", [CONVEX, CONCAVE, ASPHERIC])
def test_patches_are_exact_rectangles(sag):
    el = RadialProfileTransducer(diameter_mm=4.0, surface_sag_m=sag, no_sub_diameter=20)
    assert _max_corner_angle_dev(el) < 1e-6


@pytest.mark.parametrize("sag", [CONVEX, CONCAVE])
def test_coverage_near_unity(sag):
    frames = subdivide_radial(sag, 2e-3, 12)
    assert 0.98 <= frames["coverage"] <= 1.02


def test_flat_piston_matches_analytic_jinc():
    # A flat disc radiates the jinc pattern D(θ) = 2 J1(ka sinθ)/(ka sinθ);
    # the radial tiling must reproduce its −6 dB (half-pressure) half-angle.
    from scipy.special import j1

    diameter_mm, fc, c = 10.0, 1e6, 1540.0
    el = RadialProfileTransducer(
        diameter_mm=diameter_mm, surface_sag_m=lambda r: np.zeros_like(np.asarray(r, float)),
        frequency_Hz=fc, no_sub_diameter=24,
    )
    ang = np.linspace(0, 30, 601)
    th = np.deg2rad(ang)
    pts = np.stack([150.0 * np.sin(th), np.zeros_like(th), 150.0 * np.cos(th)], 1)
    p = np.abs(Emission(el, fs=100e6, c=c, monochromatic=True, verbose=False)(pts)[0]).squeeze()
    measured = float(ang[np.argmax((20 * np.log10(p / p.max())) <= -6.0)])

    ka = 2 * np.pi / (c / fc) * (diameter_mm / 2 * 1e-3)
    x = np.linspace(1e-6, ka, 2000)
    analytic = float(np.degrees(np.arcsin(x[np.argmax(np.abs(2 * j1(x) / x) <= 0.5)] / ka)))
    assert abs(measured - analytic) < 1.5


def test_method_radial_rename():
    # "radial" is the new ring method; "spherical" was removed (no deprecation shim).
    ConvexCircularTransducer(diameter_mm=10.0, focus_mm=6.0, no_sub_diameter=16, method="radial")
    with pytest.raises(ValueError):
        ConvexCircularTransducer(diameter_mm=10.0, focus_mm=6.0, method="spherical")
    with pytest.raises(ImportError):
        from esdiva.utilities import subdivide_spherical_cap  # noqa: F401
