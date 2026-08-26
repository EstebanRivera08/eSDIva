"""Frozen matrix-array apodization profiles.

The MatrixArrayTransducer aperture convention (directivity-cone footprint,
element-size centring) was validated against an external simulator; these
snapshots lock the exact weights so any change to the convention fails loudly.
"""

import numpy as np
from numpy.testing import assert_allclose

from esdiva.transducers import MatrixArrayTransducer


def _matrix_6x5():
    return MatrixArrayTransducer(
        n_elements_x=6,
        n_elements_y=5,
        element_width_mm=0.3,
        element_height_mm=0.25,
        kerf_x_mm=0.02,
        kerf_y_mm=0.03,
        no_sub_x=1,
        no_sub_y=1,
        frequency_Hz=5e6,
    )


# fmt: off
_HANNING_SNAPSHOT = np.array([
    0.08637288, 0.3125,     0.5920085,  0.81813562, 0.9045085,  0.81813562,
    0.0954915,  0.3454915,  0.6545085,  0.9045085,  1.0,        0.9045085,
    0.08637288, 0.3125,     0.5920085,  0.81813562, 0.9045085,  0.81813562,
    0.0625,     0.22612712, 0.42838137, 0.5920085,  0.6545085,  0.5920085,
    0.0329915,  0.11936438, 0.22612712, 0.3125,     0.3454915,  0.3125,
])
_CIRCULAR_SNAPSHOT = np.array([
    0., 1., 1., 1., 1., 1.,
    0., 1., 1., 1., 1., 1.,
    0., 1., 1., 1., 1., 1.,
    0., 0., 1., 1., 1., 1.,
    0., 0., 0., 0., 0., 0.,
])
# fmt: on


class TestMatrixApodizationFrozen:
    def test_hanning_profile_unchanged(self):
        tx = _matrix_6x5()
        apod = tx.compute_apodization(
            focus_mm=[0.3, -0.2, 4.0], FoverD=1.5, apodization_type="hanning"
        )
        assert_allclose(apod, _HANNING_SNAPSHOT, rtol=0, atol=1e-8)

    def test_circular_profile_unchanged(self):
        tx = _matrix_6x5()
        apod = tx.compute_apodization(
            focus_mm=[0.0, 0.0, 3.0], FoverD=2.0, apodization_type="circular"
        )
        assert_allclose(apod, _CIRCULAR_SNAPSHOT, rtol=0, atol=0)
