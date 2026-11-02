"""Radial subdivision and RadialProfileTransducer: geometry + far-field checks."""

import numpy as np
import pytest

from esdiva import Emission
from esdiva.transducers import (
    ConvexCircularTransducer,
    RadialProfileTransducer,
)
from esdiva.utilities import subdivide_radial

R_MM = (
    3.0  # radius of curvature for the spherical test caps (mm, transducer convention)
)
R = R_MM * 1e-3  # same, in metres (virtual_source_m and subdivide_radial are SI)


# Sag callables are mm->mm (the RadialProfileTransducer user convention).
def CONVEX(r_mm):
    return np.sqrt(np.clip(R_MM**2 - np.asarray(r_mm, float) ** 2, 0.0, None)) - R_MM


def CONCAVE(r_mm):
    return R_MM - np.sqrt(np.clip(R_MM**2 - np.asarray(r_mm, float) ** 2, 0.0, None))


def ASPHERIC(r_mm):
    u = np.asarray(r_mm, float) / 2.0  # aperture radius 2 mm
    return -0.8 * u**2 - 0.3 * u**4


def _max_corner_angle_dev(element):
    """Largest deviation (deg) of any patch corner angle from 90°."""
    q = np.array(element.sub_quad_verts)  # (M, 4, 3)
    c0, c1, c2, c3 = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    def angle(a, b, c):
        u, v = a - b, c - b
        cos = (u * v).sum(-1) / (
            np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1)
        )
        return np.degrees(np.arccos(np.clip(cos, -1, 1)))

    corners = np.stack(
        [angle(c3, c0, c1), angle(c0, c1, c2), angle(c1, c2, c3), angle(c2, c3, c0)], 1
    )
    return float(np.max(np.abs(corners - 90.0)))


@pytest.mark.parametrize("sag", [CONVEX, CONCAVE, ASPHERIC])
def test_patches_are_exact_rectangles(sag):
    el = RadialProfileTransducer(
        diameter_mm=4.0, surface_sag_mm=sag, no_sub_diameter=20
    )
    assert _max_corner_angle_dev(el) < 1e-6


@pytest.mark.parametrize("sag", [CONVEX, CONCAVE])
def test_coverage_near_unity(sag):
    # subdivide_radial is the SI tiler; wrap the mm sag callable to m->m.
    frames = subdivide_radial(lambda r_m: sag(r_m * 1e3) * 1e-3, 2e-3, 12)
    assert 0.98 <= frames["coverage"] <= 1.02


def test_flat_piston_matches_analytic_jinc():
    # A flat disc radiates the jinc pattern D(θ) = 2 J1(ka sinθ)/(ka sinθ);
    # the radial tiling must reproduce its −6 dB (half-pressure) half-angle.
    from scipy.special import j1

    diameter_mm, fc, c = 10.0, 1e6, 1540.0
    el = RadialProfileTransducer(
        diameter_mm=diameter_mm,
        surface_sag_mm=lambda r_mm: np.zeros_like(np.asarray(r_mm, float)),
        frequency_Hz=fc,
        no_sub_diameter=24,
    )
    ang = np.linspace(0, 30, 601)
    th = np.deg2rad(ang)
    pts = np.stack([150.0 * np.sin(th), np.zeros_like(th), 150.0 * np.cos(th)], 1)
    p = np.abs(
        Emission(el, fs=100e6, c=c, monochromatic=True, verbose=False)(pts)[0]
    ).squeeze()
    measured = float(ang[np.argmax((20 * np.log10(p / p.max())) <= -6.0)])

    ka = 2 * np.pi / (c / fc) * (diameter_mm / 2 * 1e-3)
    x = np.linspace(1e-6, ka, 2000)
    analytic = float(
        np.degrees(np.arcsin(x[np.argmax(np.abs(2 * j1(x) / x) <= 0.5)] / ka))
    )
    assert abs(measured - analytic) < 1.5


def test_virtual_source_recovers_sphere_centre():
    # A spherical cap radiates from its centre of curvature. CONVEX is a diverging
    # cap (apex at z=0, centre BEHIND at z=−R); CONCAVE is a bowl pinned at its rim
    # (apex the rim, centre IN FRONT at z=+R−sag). The circle fit must recover both.
    sag = R - np.sqrt(R**2 - (2e-3) ** 2)  # rim sagitta of the 4 mm concave cap
    cases = {
        CONVEX: -R,  # diverging: virtual source behind the aperture
        CONCAVE: R - sag,  # converging: focus in front, shifted by the rim pin
    }
    for profile, expect_z in cases.items():
        el = RadialProfileTransducer(
            diameter_mm=4.0, surface_sag_mm=profile, no_sub_diameter=20
        )
        vs = el.virtual_source_m
        np.testing.assert_allclose(vs[:2], 0.0, atol=1e-12)  # on-axis
        np.testing.assert_allclose(vs[2], expect_z, rtol=1e-3)


def test_virtual_source_flat_is_aperture_plane():
    # A flat piston has no curvature → the virtual source collapses to z=0.
    el = RadialProfileTransducer(
        diameter_mm=4.0,
        surface_sag_mm=lambda r_mm: np.zeros_like(np.asarray(r_mm, float)),
        no_sub_diameter=20,
    )
    np.testing.assert_array_equal(el.virtual_source_m, [0.0, 0.0, 0.0])


def test_method_radial_rename():
    # "radial" is the new ring method; "spherical" was removed (no deprecation shim).
    ConvexCircularTransducer(
        diameter_mm=10.0, focus_mm=6.0, no_sub_diameter=16, method="radial"
    )
    with pytest.raises(ValueError):
        ConvexCircularTransducer(diameter_mm=10.0, focus_mm=6.0, method="spherical")
    with pytest.raises(ImportError):
        from esdiva.utilities import subdivide_spherical_cap  # noqa: F401
