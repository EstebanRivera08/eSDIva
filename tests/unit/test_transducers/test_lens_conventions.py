"""Lens datum and lens-sag conventions (Field II correspondence)."""

import numpy as np
import pytest

from esdiva.transducers import FocusedCircularTransducer
from esdiva.transducers.fieldii_compat import FieldIITransducer


class TestSettableLensSag:
    def test_override_and_restore(self, linear_4elem):
        assert linear_4elem.elevation_lens_sag == 0.0
        linear_4elem.elevation_lens_sag = 1.2e-4
        assert linear_4elem.elevation_lens_sag == 1.2e-4
        linear_4elem.elevation_lens_sag = None  # back to geometric default
        assert linear_4elem.elevation_lens_sag == 0.0

    def test_negative_sag_is_a_convex_lens(self, linear_4elem):
        # Negative is meaningful, not an error: it is a surface that bulges INTO
        # the medium, shortening every path instead of lengthening it.
        linear_4elem.elevation_lens_sag = -1e-4
        assert linear_4elem.elevation_lens_sag == -1e-4

    def test_rejects_non_finite(self, linear_4elem):
        with pytest.raises(ValueError):
            linear_4elem.elevation_lens_sag = float("nan")


class TestFieldIILensSag:
    def _quads(self, half_h_mm):
        """One flat 1 mm × 2·half_h patch centred at the origin."""
        hw, hh = 0.5e-3, half_h_mm * 1e-3
        return [
            np.array(
                [[-hw, -hh, 0], [hw, -hh, 0], [hw, hh, 0], [-hw, hh, 0]],
                dtype=np.float64,
            )
        ]

    def test_sag_from_elevation_focus(self):
        tx = FieldIITransducer(
            self._quads(half_h_mm=2.0),
            [1.0],
            [0.0],
            frequency_hz=5e6,
            elevation_focus_mm=8.0,
        )
        R, half_h = 8.0e-3, 2.0e-3
        expected = R - np.sqrt(R**2 - half_h**2)
        assert tx.elevation_lens_sag == pytest.approx(expected)

    def test_default_is_zero(self):
        tx = FieldIITransducer(self._quads(half_h_mm=2.0), [1.0], [0.0])
        assert tx.elevation_lens_sag == 0.0

    def test_focus_smaller_than_aperture_rejected(self):
        with pytest.raises(ValueError):
            FieldIITransducer(
                self._quads(half_h_mm=2.0), [1.0], [0.0], elevation_focus_mm=1.0
            )


class TestFocusedCircularDatum:
    def test_rim_at_z0_centre_dished_back(self):
        """Field II lens datum: face (curved-axis rim) at z = 0, centre at -sag."""
        tx = FocusedCircularTransducer(
            diameter_mm=10.0, focus_mm=15.0, no_sub_diameter=8, frequency_Hz=1e6
        )
        R, R_ap = 15.0e-3, 5.0e-3
        sag = R - np.sqrt(R**2 - R_ap**2)
        # Patch centres lie ON the surface (corners sit in tangent planes and
        # may dip ~10 µm past it): deepest centre = centre line at -sag, and
        # the curved-axis rim reaches (close to) the z = 0 face plane.
        cz = tx.sub_patch_frames["centers"][:, 2]
        assert cz.min() == pytest.approx(-sag, abs=5e-6)
        z = np.concatenate([q[:, 2] for q in tx.sub_quad_verts])
        assert z.max() == pytest.approx(0.0, abs=1e-5)
        assert tx.elevation_lens_sag == pytest.approx(sag)


class TestCylindricalLensGeometry:
    """The lens is built as geometry, so its geometry is what gets checked."""

    @staticmethod
    def _probe(radius_mm, height_mm=4.0, no_sub_y=24):
        from esdiva.transducers import LinearArrayTransducer

        return LinearArrayTransducer(
            n_elements=1,
            element_width_mm=0.1,
            element_height_mm=height_mm,
            kerf_mm=0.02,
            no_sub_x=1,
            no_sub_y=no_sub_y,
            elevation_focus_mm=radius_mm,
            frequency_Hz=5e6,
        )

    def test_sag_is_signed_by_curvature(self):
        """Concave recedes (+), convex protrudes (-), magnitudes identical."""
        concave, convex = self._probe(12.0), self._probe(-12.0)
        assert concave.elevation_lens_sag > 0
        assert convex.elevation_lens_sag < 0
        assert concave.elevation_lens_sag == pytest.approx(-convex.elevation_lens_sag)

    @pytest.mark.parametrize("radius_mm", [12.0, -12.0])
    def test_patches_are_equidistant_from_the_stated_focus(self, radius_mm):
        """The defining property of a lens: one point sees every patch alike.

        That point is the arc's centre of curvature, which sits one sagitta
        shallower than the radius because the rim is the z = 0 datum. For a
        convex lens it is virtual, behind the array.
        """
        probe = self._probe(radius_mm)
        centres = np.asarray(probe.sub_quad_verts).mean(axis=1)
        focus_m = probe.elevation_focus_depth_mm * 1e-3
        d = np.linalg.norm(centres - np.array([0.0, 0.0, focus_m]), axis=1)
        assert np.ptp(d) < 1e-9  # metres
        # Patch centres are chord midpoints, so they sit a fraction of a micron
        # inside the arc - a discretisation offset, not a geometry error.
        assert d.mean() == pytest.approx(abs(radius_mm) * 1e-3, rel=1e-4)

    def test_flat_aperture_has_no_elevation_focus(self):
        probe = self._probe(None, no_sub_y=4)
        assert probe.elevation_lens_sag == 0.0
        assert probe.elevation_focus_depth_mm == float("inf")
        z = np.asarray(probe.sub_quad_verts)[..., 2]
        assert np.allclose(z, 0.0)

    def test_radius_smaller_than_half_height_rejected(self):
        with pytest.raises(ValueError, match="cannot span"):
            self._probe(1.0, height_mm=4.0)
