"""Tests for pyfield.transducers.linear.LinearArrayTransducer."""

import numpy as np
import pytest

from pyfield.transducers import LinearArrayTransducer


class TestLinearCreation:
    def test_creation_defaults(self, linear_4elem):
        assert linear_4elem.n_elements == 4
        assert linear_4elem.type == "linear"
        assert linear_4elem.fc == 5e6

    def test_stored_dimensions_si(self, linear_4elem):
        """Dimensions are stored in SI units (metres)."""
        assert linear_4elem.elem_width == pytest.approx(0.25e-3)
        assert linear_4elem.elem_height == pytest.approx(5.0e-3)
        assert linear_4elem.kerf == pytest.approx(0.05e-3)

    def test_pitch(self, linear_4elem):
        expected_pitch = (0.25 + 0.05) * 1e-3
        assert linear_4elem.pitch == pytest.approx(expected_pitch)

    def test_default_frequency_warning(self, capsys):
        """When no frequency_Hz is given, a warning is printed."""
        LinearArrayTransducer(
            n_elements=2,
            element_width_mm=0.25,
            element_height_mm=5.0,
            kerf_mm=0.05,
            no_sub_x=1,
            no_sub_y=1,
        )
        captured = capsys.readouterr()
        assert "1 MHz" in captured.out


class TestLinearElementCenters:
    def test_count(self, linear_4elem):
        centers = linear_4elem.element_centers
        assert centers.shape == (4, 3)

    def test_symmetric(self, linear_8elem):
        """Element centers should be symmetric around x=0."""
        centers = linear_8elem.element_centers
        x_coords = centers[:, 0]
        assert abs(x_coords.sum()) < 1e-12

    def test_z_at_zero(self, linear_4elem):
        """All element centers sit on the z=0 plane."""
        centers = linear_4elem.element_centers
        np.testing.assert_allclose(centers[:, 2], 0.0)

    def test_y_at_zero(self, linear_4elem):
        """All element centers sit on the y=0 plane."""
        centers = linear_4elem.element_centers
        np.testing.assert_allclose(centers[:, 1], 0.0)


class TestLinearSubdivisions:
    def test_count(self, linear_4elem):
        """Total patches = n_elements * no_sub_x * no_sub_y."""
        expected = 4 * 1 * 2  # 8 patches
        assert len(linear_4elem.sub_quad_verts) == expected
        assert len(linear_4elem.sub_el_idx) == expected

    def test_quad_shape(self, linear_4elem):
        """Each subdivision quad has 4 vertices of 3 coordinates."""
        for quad in linear_4elem.sub_quad_verts:
            assert quad.shape == (4, 3)

    def test_n_sub_patches_property(self, linear_4elem):
        assert linear_4elem.n_sub_patches == 4 * 1 * 2


class TestLinearElevationLens:
    """Elevation-lens datum and sag (the Field II `xdc_focused_array` convention).

    The cylindrical lens references the element face (rim, y = ±height/2) at z = 0 and
    dishes the surface back, so the centre (y = 0) is the deepest point at −sag, where
    ``sag = R − √(R² − (height/2)²)``. This matches Field II, which keeps the flat
    element face at z = 0; reception adds ``sag/c`` per aperture as the lens group delay.
    """

    @staticmethod
    def _lensed(elev_mm=8.0, h_mm=1.5, R_check=None):
        return LinearArrayTransducer(
            n_elements=4,
            element_width_mm=0.25,
            element_height_mm=h_mm,
            kerf_mm=0.05,
            no_sub_x=1,
            no_sub_y=10,
            elevation_focus_mm=elev_mm,
            frequency_Hz=5e6,
        )

    def test_sag_zero_when_flat(self, linear_4elem):
        """A flat aperture (no lens) has zero elevation sag."""
        assert linear_4elem.elevation_lens_sag == 0.0

    def test_sag_formula(self):
        """Sag equals R − √(R² − (height/2)²) in metres."""
        tx = self._lensed(elev_mm=8.0, h_mm=1.5)
        R, h = 8e-3, 1.5e-3
        assert tx.elevation_lens_sag == pytest.approx(R - np.sqrt(R**2 - (h / 2) ** 2))

    def test_rim_at_zero_centre_recessed(self):
        """Rim patches sit at z≈0; the centre is recessed to ≈ −sag (rim-referenced)."""
        tx = self._lensed(elev_mm=8.0, h_mm=1.5)
        z = np.stack(tx.sub_quad_verts)[:, :, 2]
        assert z.max() == pytest.approx(0.0, abs=1e-9)
        assert z.min() == pytest.approx(-tx.elevation_lens_sag, abs=1e-9)


class TestLinearDelays:
    def test_shape(self, linear_4elem):
        delays = linear_4elem.compute_delays(focus_mm=[0, 0, 20])
        assert delays.shape == (4,)

    def test_minimum_is_zero(self, linear_4elem):
        delays = linear_4elem.compute_delays(focus_mm=[0, 0, 20])
        assert delays.min() == pytest.approx(0.0)

    def test_non_negative(self, linear_4elem):
        delays = linear_4elem.compute_delays(focus_mm=[0, 0, 20])
        assert np.all(delays >= 0)

    def test_symmetric_focus_symmetric_delays(self, linear_8elem):
        """On-axis focus should yield symmetric delays."""
        delays = linear_8elem.compute_delays(focus_mm=[0, 0, 30])
        np.testing.assert_allclose(delays, delays[::-1], atol=1e-12)


class TestLinearApodization:
    def test_shape(self, linear_4elem):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            apod = linear_4elem.compute_apodization(focus_mm=[0, 0, 20], FoverD=1.0)
        assert apod.shape == (4,)

    def test_non_negative(self, linear_4elem):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            apod = linear_4elem.compute_apodization(focus_mm=[0, 0, 20], FoverD=1.0)
        assert np.all(apod >= 0)

    def test_none_type_defaults_rect(self, linear_4elem):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            linear_4elem.compute_apodization(
                focus_mm=[0, 0, 20], FoverD=1.0, apodization_type=None
            )
        assert linear_4elem.apodization_type == "rect"
