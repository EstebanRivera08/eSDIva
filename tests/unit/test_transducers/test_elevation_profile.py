"""Elevation profiles: the free-shape lens must reproduce the circular one exactly,
and the RCA's two arrays must be the same bars crossed at 90 degrees."""

import numpy as np
import pytest

from esdiva.transducers import (
    ConvexArrayTransducer,
    ElevationProfileTransducer,
    LinearArrayTransducer,
    RowColumnAddressedTransducer,
)

R_MM, H_MM = 8.0, 5.0


def _circle(radius_mm):
    """Signed circular lens written by hand as a free profile (mm -> mm)."""
    r, s = abs(radius_mm), np.sign(radius_mm)
    return lambda y: s * (r - np.sqrt(r**2 - y**2))


def _linear(**lens):
    return LinearArrayTransducer(
        n_elements=3,
        element_width_mm=0.3,
        element_height_mm=H_MM,
        kerf_mm=0.05,
        no_sub_x=2,
        no_sub_y=6,
        frequency_Hz=5e6,
        **lens,
    )


@pytest.mark.parametrize("radius_mm", [R_MM, -R_MM])
def test_profile_reproduces_circular_lens(radius_mm):
    arc = _linear(elevation_focus_mm=radius_mm)
    free = _linear(elevation_profile_mm=_circle(radius_mm))
    np.testing.assert_allclose(free.sub_quad_verts, arc.sub_quad_verts, atol=1e-10)
    assert free.elevation_lens_sag == pytest.approx(arc.elevation_lens_sag, rel=1e-9)
    # Sign: concave recessed (+), convex protruding (-).
    assert np.sign(free.elevation_lens_sag) == np.sign(radius_mm)


def test_convex_array_takes_profile_and_signed_lens():
    kw = dict(
        n_elements=4,
        element_width_mm=0.3,
        element_height_mm=H_MM,
        kerf_mm=0.05,
        radius_of_curvature_mm=40,
        no_sub_x=1,
        no_sub_y=6,
        frequency_Hz=3e6,
    )
    arc = ConvexArrayTransducer(elevation_focus_mm=-R_MM, **kw)
    free = ConvexArrayTransducer(elevation_profile_mm=_circle(-R_MM), **kw)
    np.testing.assert_allclose(free.sub_quad_verts, arc.sub_quad_verts, atol=1e-10)
    assert arc.elevation_lens_sag < 0


def test_element_is_one_array_element():
    """A single profile element is exactly element 0 of the array, re-centred."""
    elem = ElevationProfileTransducer(
        element_width_mm=0.3,
        element_height_mm=H_MM,
        elevation_profile_mm=lambda y: y**2 / 20,
        no_sub_x=2,
        no_sub_y=6,
    )
    arr = _linear(elevation_profile_mm=lambda y: y**2 / 20)
    first = np.array(arr.sub_quad_verts[: elem.n_sub_patches]) - arr.element_centers[0]
    np.testing.assert_allclose(elem.sub_quad_verts, first, atol=1e-15)
    # Rims at z = 0, centre recessed by the parabola's rim height.
    assert elem.elevation_lens_sag == pytest.approx((H_MM / 2) ** 2 / 20 * 1e-3)


def test_lens_arguments_are_exclusive_and_need_patches():
    with pytest.raises(ValueError, match="not both"):
        _linear(elevation_focus_mm=R_MM, elevation_profile_mm=_circle(R_MM))
    with pytest.raises(ValueError, match="no_sub_y"):
        LinearArrayTransducer(
            n_elements=2,
            element_width_mm=0.3,
            element_height_mm=H_MM,
            kerf_mm=0.05,
            no_sub_x=1,
            no_sub_y=1,
            elevation_profile_mm=_circle(R_MM),
        )


def test_rca_rows_are_columns_rotated():
    rca = RowColumnAddressedTransducer(
        n_elements=8,
        element_width_mm=0.09,
        kerf_mm=0.01,
        no_sub_x=1,
        no_sub_y=8,
        frequency_Hz=15e6,
    )
    assert rca.bar_length_mm == pytest.approx(8 * 0.1 - 0.01)
    cols, rows = rca.columns.element_centers, rca.rows.element_centers
    np.testing.assert_allclose(rows[:, 1], cols[:, 0], atol=1e-15)  # rows step in y
    np.testing.assert_allclose(rows[:, [0, 2]], 0.0, atol=1e-15)
    # Each row bar is long along x: its patches span the full aperture in x.
    xs = np.array(rca.rows.sub_quad_verts)[..., 0]
    assert np.ptp(xs) == pytest.approx(rca.bar_length_mm * 1e-3)
