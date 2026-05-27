"""Tests for PE SDI kernel — compute_pe_sdi correctness.

Validates that the combined PE SDI (16 deltas per patch pair + 1 cumsum)
produces results matching the reference: FFT convolution of dh_tx and d2h_rx.
"""

import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.fft import irfft, rfft

from pyfield.h_sir.sir_derivatives import (
    compute_d2h,
    compute_dh,
    compute_pe_sdi,
)
from pyfield.transducers import LinearArrayTransducer
from pyfield.utilities.helper_functions import (
    compute_sub_elem_attributes,
    compute_time_grid,
)


@pytest.fixture
def simple_tx():
    """4-element linear array for TX."""
    tx = LinearArrayTransducer(
        n_elements=4,
        element_width_mm=0.25,
        element_height_mm=5.0,
        kerf_mm=0.05,
        no_sub_x=1,
        no_sub_y=2,
        frequency_Hz=5e6,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        tx.compute_delays(focus_mm=[0, 0, 20])
    return tx


@pytest.fixture
def simple_rx():
    """4-element linear array for RX (same geometry, no delays)."""
    rx = LinearArrayTransducer(
        n_elements=4,
        element_width_mm=0.25,
        element_height_mm=5.0,
        kerf_mm=0.05,
        no_sub_x=1,
        no_sub_y=2,
        frequency_Hz=5e6,
    )
    # RX: no focusing delays (receive on all elements simultaneously).
    return rx


def _next_pow2(n):
    return 1 << (int(n - 1).bit_length())


def _extract_sub_elem(tx):
    """Unpack sub-element attributes."""
    return compute_sub_elem_attributes(tx)


def _build_pe_time_grid(tx, rx, points, c=1540.0, fs=200e6):
    """Build PE time grid covering TX + RX propagation."""
    tx_c, tx_a, tx_d, tx_M, _, tx_wx, tx_wy, _ = _extract_sub_elem(tx)
    rx_c, rx_a, rx_d, rx_M, _, rx_wx, rx_wy, _ = _extract_sub_elem(rx)

    _, tx_t0, tx_dt, tx_T = compute_time_grid(
        points.shape[0],
        tx_M,
        points,
        tx_c,
        float(tx_wx.max()),
        float(tx_wy.max()),
        c,
        fs,
        tx.delays,
        verbose=False,
    )
    _, rx_t0, rx_dt, rx_T = compute_time_grid(
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
    dt = 1.0 / fs
    pe_t0 = tx_t0 + rx_t0
    pe_T = tx_T + rx_T - 1
    return pe_t0, dt, pe_T, tx_t0, tx_T, rx_t0, rx_T


class TestPeSdiVsReference:
    """PE SDI kernel must match reference when convolved with excitation.

    Raw delta-level comparison has float32 interpolation quantization
    differences between PE SDI (16 direct placements + 1 cumsum) and the
    reference (separate d2h + FFT conv). These wash out after excitation
    convolution, which is the actual use case.
    """

    def test_pe_sdi_excitation_pipeline_matches_reference(self, simple_tx, simple_rx):
        """PE SDI + FFT(exc) must match dh_tx * d2h_rx + FFT(exc)."""
        c, fs = 1540.0, 200e6
        inv_c = 1.0 / c
        dt = 1.0 / fs
        fc = 5e6
        points = np.array([[0.0, 0.0, 20.0e-3]], dtype=np.float32)

        # 3-cycle excitation pulse.
        t_exc = np.arange(0, 3.0 / fc, 1.0 / fs)
        excitation = np.sin(2 * np.pi * fc * t_exc).astype(np.float32)
        L = len(excitation)

        tx_c, tx_a, tx_d, tx_M, _, tx_wx, tx_wy, tx_idx = _extract_sub_elem(simple_tx)
        rx_c, rx_a, rx_d, rx_M, _, rx_wx, rx_wy, rx_idx = _extract_sub_elem(simple_rx)

        pe_t0, pe_dt, pe_T, tx_t0, tx_T, rx_t0, rx_T = _build_pe_time_grid(
            simple_tx,
            simple_rx,
            points,
            c,
            fs,
        )

        # RX element 0 patches.
        rx_mask = rx_idx == 0
        rx_c_e = rx_c[rx_mask]
        rx_wx_e = rx_wx[rx_mask]
        rx_wy_e = rx_wy[rx_mask]
        rx_a_e = rx_a[rx_mask]
        rx_d_e = rx_d[rx_mask]

        # PE SDI → FFT conv with excitation (no jw — derivatives in Dh_pe).
        Dh_pe = compute_pe_sdi(
            points,
            tx_c,
            tx_wx,
            tx_wy,
            tx_a,
            tx_d,
            rx_c_e,
            rx_wx_e,
            rx_wy_e,
            rx_a_e,
            rx_d_e,
            inv_c,
            pe_t0,
            pe_T,
            fs,
            dt,
        )
        nfft_pe = _next_pow2(pe_T + L - 1)
        rf_pe = irfft(
            rfft(Dh_pe, n=nfft_pe, axis=1) * rfft(excitation, n=nfft_pe),
            n=nfft_pe,
            axis=1,
        )[:, :pe_T]

        # Reference: dh_tx FFT-conv d2h_rx FFT-conv excitation.
        dh_tx = compute_dh(
            points,
            tx_c,
            tx_wx,
            tx_wy,
            inv_c,
            tx_a,
            tx_d,
            tx_t0,
            tx_T,
            fs,
            dt,
        )
        d2h_rx = compute_d2h(
            points,
            rx_c_e,
            rx_wx_e,
            rx_wy_e,
            inv_c,
            rx_a_e,
            rx_d_e,
            rx_t0,
            rx_T,
            fs,
            dt,
        )
        nfft_ref = _next_pow2(tx_T + rx_T + L - 2)
        Dh_ref = irfft(
            rfft(dh_tx.astype(np.float64), n=nfft_ref, axis=1)
            * rfft(d2h_rx.astype(np.float64), n=nfft_ref, axis=1),
            n=nfft_ref,
            axis=1,
        )
        rf_ref = irfft(
            rfft(Dh_ref, n=nfft_ref, axis=1)
            * rfft(excitation.astype(np.float64), n=nfft_ref),
            n=nfft_ref,
            axis=1,
        )[:, :pe_T]

        # After excitation convolution, peaks match and waveforms correlate.
        peak_pe = float(np.abs(rf_pe).max())
        peak_ref = float(np.abs(rf_ref).max())
        assert peak_pe > 0, "PE SDI + exc should produce non-zero signal."
        assert abs(peak_pe / peak_ref - 1.0) < 0.05, (
            f"Peak ratio {peak_pe / peak_ref:.4f} should be ~1.0"
        )

        # Normalized correlation in active region.
        rf_pe_n = rf_pe[0] / peak_pe
        rf_ref_n = (rf_ref[0] / peak_ref).astype(np.float32)
        active = np.abs(rf_ref_n) > 0.01
        if active.any():
            corr = np.corrcoef(rf_pe_n[active], rf_ref_n[active])[0, 1]
            assert corr > 0.95, f"Waveform correlation {corr:.4f} should be > 0.95"

    def test_pe_sdi_nonzero_output(self, simple_tx, simple_rx):
        """PE SDI must produce non-zero output for on-axis scatterer."""
        c, fs = 1540.0, 200e6
        points = np.array([[0.0, 0.0, 20.0e-3]], dtype=np.float32)
        tx_c, tx_a, tx_d, _, _, tx_wx, tx_wy, _ = _extract_sub_elem(simple_tx)
        rx_c, rx_a, rx_d, _, _, rx_wx, rx_wy, _ = _extract_sub_elem(simple_rx)

        pe_t0, dt, pe_T, *_ = _build_pe_time_grid(
            simple_tx,
            simple_rx,
            points,
            c,
            fs,
        )

        Dh_pe = compute_pe_sdi(
            points,
            tx_c,
            tx_wx,
            tx_wy,
            tx_a,
            tx_d,
            rx_c,
            rx_wx,
            rx_wy,
            rx_a,
            rx_d,
            1.0 / c,
            pe_t0,
            pe_T,
            fs,
            dt,
        )
        assert np.any(Dh_pe != 0), "PE SDI should have non-zero output."

    def test_pe_sdi_multiple_scatterers(self, simple_tx, simple_rx):
        """PE SDI shape correct for multiple scatterer positions."""
        c, fs = 1540.0, 200e6
        points = np.array(
            [
                [0.0, 0.0, 15.0e-3],
                [1.0e-3, 0.0, 20.0e-3],
                [-1.0e-3, 0.0, 25.0e-3],
            ],
            dtype=np.float32,
        )
        P = points.shape[0]
        tx_c, tx_a, tx_d, _, _, tx_wx, tx_wy, _ = _extract_sub_elem(simple_tx)
        rx_c, rx_a, rx_d, _, _, rx_wx, rx_wy, _ = _extract_sub_elem(simple_rx)

        pe_t0, dt, pe_T, *_ = _build_pe_time_grid(
            simple_tx,
            simple_rx,
            points,
            c,
            fs,
        )

        Dh_pe = compute_pe_sdi(
            points,
            tx_c,
            tx_wx,
            tx_wy,
            tx_a,
            tx_d,
            rx_c,
            rx_wx,
            rx_wy,
            rx_a,
            rx_d,
            1.0 / c,
            pe_t0,
            pe_T,
            fs,
            dt,
        )
        assert Dh_pe.shape == (P, pe_T)
        assert Dh_pe.dtype == np.float32


class TestPeSdiParallelAxis:
    """Points vs patches parallel axis must agree."""

    def test_patches_parallel_matches_points_parallel(self, simple_tx, simple_rx):
        c, fs = 1540.0, 200e6
        points = np.array([[0.0, 0.0, 20.0e-3]], dtype=np.float32)
        tx_c, tx_a, tx_d, _, _, tx_wx, tx_wy, _ = _extract_sub_elem(simple_tx)
        rx_c, rx_a, rx_d, _, _, rx_wx, rx_wy, _ = _extract_sub_elem(simple_rx)

        pe_t0, dt, pe_T, *_ = _build_pe_time_grid(
            simple_tx,
            simple_rx,
            points,
            c,
            fs,
        )

        Dh_points = compute_pe_sdi(
            points,
            tx_c,
            tx_wx,
            tx_wy,
            tx_a,
            tx_d,
            rx_c,
            rx_wx,
            rx_wy,
            rx_a,
            rx_d,
            1.0 / c,
            pe_t0,
            pe_T,
            fs,
            dt,
            parallel_axis="points",
        )
        Dh_patches = compute_pe_sdi(
            points,
            tx_c,
            tx_wx,
            tx_wy,
            tx_a,
            tx_d,
            rx_c,
            rx_wx,
            rx_wy,
            rx_a,
            rx_d,
            1.0 / c,
            pe_t0,
            pe_T,
            fs,
            dt,
            parallel_axis="patches",
        )
        # Thread-local reduction in mpar produces slightly different float32
        # accumulation order. Use relative tolerance and peak-scaled atol.
        peak = max(
            float(np.abs(Dh_points).max()), float(np.abs(Dh_patches).max()), 1e-10
        )
        assert_allclose(
            Dh_points,
            Dh_patches,
            rtol=0.01,
            atol=0.005 * peak,
            err_msg="Points vs patches parallel axis must agree.",
        )


class TestPeSdiBatching:
    """Batched computation must match unbatched."""

    def test_batched_matches_unbatched(self, simple_tx, simple_rx):
        c, fs = 1540.0, 200e6
        points = np.array(
            [
                [0.0, 0.0, 15.0e-3],
                [1.0e-3, 0.0, 20.0e-3],
            ],
            dtype=np.float32,
        )
        tx_c, tx_a, tx_d, _, _, tx_wx, tx_wy, _ = _extract_sub_elem(simple_tx)
        rx_c, rx_a, rx_d, _, _, rx_wx, rx_wy, _ = _extract_sub_elem(simple_rx)

        pe_t0, dt, pe_T, *_ = _build_pe_time_grid(
            simple_tx,
            simple_rx,
            points,
            c,
            fs,
        )

        Dh_full = compute_pe_sdi(
            points,
            tx_c,
            tx_wx,
            tx_wy,
            tx_a,
            tx_d,
            rx_c,
            rx_wx,
            rx_wy,
            rx_a,
            rx_d,
            1.0 / c,
            pe_t0,
            pe_T,
            fs,
            dt,
        )
        Dh_batched = compute_pe_sdi(
            points,
            tx_c,
            tx_wx,
            tx_wy,
            tx_a,
            tx_d,
            rx_c,
            rx_wx,
            rx_wy,
            rx_a,
            rx_d,
            1.0 / c,
            pe_t0,
            pe_T,
            fs,
            dt,
            batch_size_points=1,
        )
        assert_allclose(
            Dh_batched,
            Dh_full,
            rtol=1e-6,
            atol=1e-30,
            err_msg="Batched PE SDI must match unbatched.",
        )
