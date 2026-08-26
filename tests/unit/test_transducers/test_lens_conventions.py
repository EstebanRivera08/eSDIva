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

    def test_rejects_negative(self, linear_4elem):
        with pytest.raises(ValueError):
            linear_4elem.elevation_lens_sag = -1e-4


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
