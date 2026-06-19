"""Tests for the surviving pulse-echo SDI kernels.

Two analytic forms of the pulse-echo response are implemented in
``transducer_sir_pe_sdi.py``: the paired form (`compute_pe_complete`, which splats the
integrated drive per patch pair) and the spectral form (`compute_oneway_spectrum_band`,
the closed-form one-way SIR-delta spectrum). This test checks the spectrum kernel produces
a sane, position-dependent spectrum — the end-to-end equivalence of the three ReceptionSDI
methods is covered in test_reception.py.
"""

import warnings

import numpy as np

from pyfield.hsir.transducer_sir_pe_sdi import compute_oneway_spectrum_band
from pyfield.transducers import LinearArrayTransducer
from pyfield.utilities.helper_functions import compute_sub_elem_attributes, compute_time_grid

import pytest


@pytest.fixture
def simple_rx():
    """4-element linear array, 2 patches per element (uniform, no focusing delays)."""
    return LinearArrayTransducer(
        n_elements=4,
        element_width_mm=0.25,
        element_height_mm=5.0,
        kerf_mm=0.05,
        no_sub_x=1,
        no_sub_y=2,
        frequency_Hz=5e6,
    )


def _rx_window(rx, points, c=1540.0, fs=200e6):
    """One-way receive window origin t0 (seconds) for the analytic spectrum reference."""
    rx_c, _, _, rx_M, _, rx_wx, rx_wy, _ = compute_sub_elem_attributes(rx)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        _, rx_t0, _, _ = compute_time_grid(
            points.shape[0],
            rx_M,
            points,
            rx_c,
            float(rx_wx.max()),
            float(rx_wy.max()),
            c,
            fs,
            rx.delays,
            verbose=False,
        )
    return rx_t0


def _band_omega(fs=200e6, nfft=4096):
    """Uniform in-band angular-frequency grid over the 2-8 MHz transducer pass-band."""
    freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
    band = (freqs > 2e6) & (freqs < 8e6)
    return (2.0 * np.pi * freqs[band]).astype(np.float64)


class TestOnewaySpectrumBasic:
    """The closed-form one-way SIR spectrum must be non-trivial and position-dependent."""

    def test_spectrum_nonzero_and_position_dependent(self, simple_rx):
        c, fs = 1540.0, 200e6
        points = np.array([[0.0, 0.0, 20e-3], [3e-3, 0.0, 28e-3]], dtype=np.float32)
        rx_c, rx_a, rx_d, _, _, rx_wx, rx_wy, _ = compute_sub_elem_attributes(simple_rx)
        omega = _band_omega(fs)
        spec = compute_oneway_spectrum_band(
            points, rx_c, rx_wx, rx_wy, rx_a, rx_d, 1.0 / c, _rx_window(simple_rx, points), omega, 1.0 / fs
        )
        assert spec.shape == (2, omega.shape[0])
        assert spec.dtype == np.complex64
        assert np.any(np.abs(spec) > 0), "spectrum should be non-zero in-band."
        # Two scatterers at different depths cannot share the same spectrum.
        assert not np.allclose(spec[0], spec[1])
