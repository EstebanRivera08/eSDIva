import numpy as np
from numpy.testing import assert_allclose

from esdiva.utilities import fwhm_mm, tone_burst


def test_tone_burst_matches_windowed_sine():
    fc, fs = 5e6, 100e6
    t = np.arange(0, 2 / fc, 1 / fs)
    expected = np.sin(2 * np.pi * fc * t) * np.hanning(t.size)
    burst = tone_burst(fc, 2, fs)
    assert burst.dtype == np.float32
    assert_allclose(burst, expected, atol=1e-7)


def test_fwhm_of_gaussian():
    x = np.linspace(-5, 5, 2001)
    sigma = 0.7
    p = np.exp(-(x**2) / (2 * sigma**2))
    assert_allclose(fwhm_mm(p, x), 2 * np.sqrt(2 * np.log(2)) * sigma, rtol=1e-4)


def test_fwhm_lobe_touching_edge_is_clipped():
    x = np.linspace(0, 1, 11)
    assert_allclose(fwhm_mm(np.ones(11), x), 1.0)
