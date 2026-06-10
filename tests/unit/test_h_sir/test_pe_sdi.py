"""Tests for PE SDI kernel — compute_pe_sdi correctness.

Validates that the combined PE SDI (16 deltas per patch pair + 1 cumsum)
produces results matching the reference: FFT convolution of dh_tx and d2h_rx.
"""

import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.fft import irfft, rfft

from pyfield.hsir.farfield_rect_patch import compute_h_sir as _compute_h_sir_ref
from pyfield.hsir.transducer_sir_pe import compute_pe_sdi, compute_pe_sdi_summed
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
    """PE SDI pipeline must match the two-way SIR reference.

    compute_pe_sdi returns the raw Δδ_pe delta train (D²h_tx ⊛ D²h_rx). Applying
    I⁴ as ÷(jω)⁴ cancels those four derivatives, so the pipeline reconstructs the
    two-way SIR h_tx ⊛ h_rx; convolving with the excitation gives the pulse-echo RF.
    The reference builds h_tx ⊛ h_rx ⊛ exc independently from the one-way SIRs.
    Float32 placement quantization washes out after the excitation convolution.
    """

    def test_pe_sdi_excitation_pipeline_matches_reference(self, simple_tx, simple_rx):
        """PE SDI + I⁴ + FFT(exc) must match (h_tx ⊛ h_rx) ⊛ exc."""
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

        from scipy.fft import rfftfreq as _rfftfreq

        # Pipeline: raw Δδ_pe → I⁴ via ÷(jω)⁴ → conv excitation.
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
        freqs_pe = _rfftfreq(nfft_pe, d=1.0 / fs)
        jw_pe = 1j * 2.0 * np.pi * freqs_pe
        inv4 = np.zeros_like(jw_pe)
        nz = freqs_pe > 0
        inv4[nz] = (1.0 / jw_pe[nz]) ** 4
        inv4 *= fs  # delta areas: ÷(jω) weights samples by dt, restore with fs.
        rf_pe = irfft(
            rfft(Dh_pe, n=nfft_pe, axis=1) * inv4 * rfft(excitation, n=nfft_pe),
            n=nfft_pe,
            axis=1,
        )[:, :pe_T]

        # Reference: two-way SIR h_tx ⊛ h_rx (one-way SIRs, no derivatives) ⊛ exc.
        from pyfield.utilities.helper_functions import compute_time_grid as _ctg

        time_grid_tx, _, _, _ = _ctg(
            1,
            tx_M,
            points,
            tx_c,
            float(tx_wx.max()),
            float(tx_wy.max()),
            c,
            fs,
            simple_tx.delays,
            verbose=False,
        )
        h_tx, _ = _compute_h_sir_ref(
            1,
            tx_M,
            tx_T,
            dt,
            time_grid_tx,
            points,
            tx_c,
            tx_wx,
            tx_wy,
            inv_c,
            fs,
            tx_a,
            tx_d,
            0,
            None,
            None,
        )  # method_flag=0 (naive) avoids float32 cumsum drift artifact  # (1, tx_T)

        M_rx_e = int(rx_mask.sum())
        time_grid_rx_e, _, _, _ = _ctg(
            1,
            M_rx_e,
            points,
            rx_c_e,
            float(rx_wx_e.max()),
            float(rx_wy_e.max()),
            c,
            fs,
            simple_rx.delays,
            verbose=False,
        )
        h_rx_e, _ = _compute_h_sir_ref(
            1,
            M_rx_e,
            rx_T,
            dt,
            time_grid_rx_e,
            points,
            rx_c_e,
            rx_wx_e,
            rx_wy_e,
            inv_c,
            fs,
            rx_a_e,
            rx_d_e,
            0,
            None,
            None,
        )  # (1, rx_T)

        nfft_ref = _next_pow2(tx_T + rx_T + L)
        H_tx = rfft(h_tx.astype(np.float64), n=nfft_ref, axis=1)
        H_rx_e = rfft(h_rx_e.astype(np.float64), n=nfft_ref, axis=1)
        # /fs: discrete h_tx ⊛ h_rx FFT-conv = fs × the continuous two-way SIR.
        h_pe_ref = H_tx * H_rx_e / fs
        rf_ref = irfft(
            h_pe_ref * rfft(excitation.astype(np.float64), n=nfft_ref),
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
            assert corr > 0.90, f"Waveform correlation {corr:.4f} should be > 0.90"

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


class TestPeSdiKernelVariants:
    """Optimised kernels (§7.2 summed, §7.3 patch-parallel) match the base kernel."""

    def _setup(self, simple_tx, simple_rx, points):
        c, fs = 1540.0, 200e6
        tx_c, tx_a, tx_d, _, _, tx_wx, tx_wy, _ = _extract_sub_elem(simple_tx)
        rx_c, rx_a, rx_d, _, _, rx_wx, rx_wy, _ = _extract_sub_elem(simple_rx)
        pe_t0, dt, pe_T, *_ = _build_pe_time_grid(simple_tx, simple_rx, points, c, fs)
        args = (tx_c, tx_wx, tx_wy, tx_a, tx_d, rx_c, rx_wx, rx_wy, rx_a, rx_d)
        return args, (1.0 / c, pe_t0, pe_T, fs, dt)

    def test_summed_matches_amps_matvec(self, simple_tx, simple_rx):
        """compute_pe_sdi_summed ≡ amps @ compute_pe_sdi (§7.2)."""
        points = np.array(
            [[0.0, 0.0, 15e-3], [1e-3, 0.0, 20e-3], [-1e-3, 0.0, 25e-3]],
            dtype=np.float32,
        )
        amps = np.array([1.0, 0.7, -1.3], dtype=np.float32)
        args, tail = self._setup(simple_tx, simple_rx, points)
        full = compute_pe_sdi(points, *args, *tail)
        summed = compute_pe_sdi_summed(points, *args[:10], amps, *tail)
        ref = amps @ full
        assert summed.shape == (full.shape[1],)
        # rtol loose: the raw delta areas reach ~1e20, where compute_pe_sdi's float32
        # output has ULP ~1e13 while the summed kernel accumulates in float64 — they
        # differ by float32 quantization, not logic (gotcha #2 non-associativity).
        assert_allclose(summed, ref, rtol=1e-3, atol=1e-4 * np.abs(ref).max())

    def test_patch_parallel_matches_point_parallel(self, simple_tx, simple_rx):
        """P==1 (patch-parallel kernel) ≡ the point-parallel kernel for the same point."""
        p0 = np.array([[0.5e-3, 0.0, 22e-3]], dtype=np.float32)
        args, tail = self._setup(simple_tx, simple_rx, p0)
        single = compute_pe_sdi(p0, *args, *tail)  # P==1 → patch-parallel
        # P==2 (duplicate point) uses the point-parallel kernel; row 0 is the same point.
        dup = np.repeat(p0, 2, axis=0)
        both = compute_pe_sdi(dup, *args, *tail)
        assert single.shape[0] == 1
        assert_allclose(single[0], both[0], rtol=1e-5, atol=1e-30)


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
