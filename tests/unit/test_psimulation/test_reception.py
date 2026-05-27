"""Tests for Reception class — pulse-echo RF simulation."""

import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pyfield.psimulation import Reception
from pyfield.transducers import LinearArrayTransducer


@pytest.fixture
def simple_tx():
    """4-element linear array for TX with focused delays."""
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
    """4-element linear array for RX (no focusing delays)."""
    rx = LinearArrayTransducer(
        n_elements=4,
        element_width_mm=0.25,
        element_height_mm=5.0,
        kerf_mm=0.05,
        no_sub_x=1,
        no_sub_y=2,
        frequency_Hz=5e6,
    )
    return rx


@pytest.fixture
def on_axis_scatterer():
    """Single on-axis scatterer at 20mm depth."""
    pos = np.array([[0, 0, 20]], dtype=np.float32)  # mm
    amp = np.array([1.0], dtype=np.float32)
    return pos, amp


class TestReceptionBasic:
    """Basic Reception functionality."""

    def test_rf_output_shape(self, simple_tx, simple_rx, on_axis_scatterer):
        """RF output shape must be (Nt, E_rx)."""
        pos, amp = on_axis_scatterer
        sim = Reception(simple_tx, simple_rx, verbose=False)
        rf, coords = sim(pos, amp)

        assert rf.ndim == 2
        assert rf.shape[1] == simple_rx.n_elements
        assert rf.dtype == np.float32
        assert "t0" in coords
        assert "dt" in coords

    def test_rf_nonzero(self, simple_tx, simple_rx, on_axis_scatterer):
        """RF must be non-zero for on-axis scatterer."""
        pos, amp = on_axis_scatterer
        sim = Reception(simple_tx, simple_rx, verbose=False)
        rf, _ = sim(pos, amp)
        assert np.any(rf != 0), "RF should be non-zero for on-axis scatterer."

    def test_self_echo_valid(self, simple_tx, on_axis_scatterer):
        """TX == RX (same transducer) must produce valid result."""
        pos, amp = on_axis_scatterer
        sim = Reception(simple_tx, simple_tx, verbose=False)
        rf, coords = sim(pos, amp)

        assert rf.ndim == 2
        assert rf.shape[1] == simple_tx.n_elements
        assert np.any(rf != 0)

    def test_excitation_none_pure_pe_sir(self, simple_tx, simple_rx, on_axis_scatterer):
        """excitation=None → pure PE SIR derivative (no excitation conv)."""
        pos, amp = on_axis_scatterer
        sim = Reception(simple_tx, simple_rx, excitation=None, verbose=False)
        rf, _ = sim(pos, amp)
        assert rf.ndim == 2
        assert np.any(rf != 0)


class TestReceptionAttenuation:
    """Attenuation handling."""

    def test_no_attenuation_default(self, simple_tx, simple_rx, on_axis_scatterer):
        """alpha0=None → no attenuation applied."""
        pos, amp = on_axis_scatterer
        sim = Reception(simple_tx, simple_rx, alpha0=None, verbose=False)
        rf_noatt, _ = sim(pos, amp)
        assert np.any(rf_noatt != 0)

    def test_attenuation_reduces_amplitude(self, simple_tx, simple_rx):
        """alpha0 > 0 must reduce RF amplitude compared to no attenuation."""
        pos = np.array([[0, 0, 30]], dtype=np.float32)  # mm, deeper point
        amp = np.array([1.0], dtype=np.float32)

        sim_noatt = Reception(simple_tx, simple_rx, alpha0=None, verbose=False)
        rf_noatt, _ = sim_noatt(pos, amp)

        sim_att = Reception(
            simple_tx, simple_rx, alpha0=0.5, freq_power=1.0, verbose=False
        )
        rf_att, _ = sim_att(pos, amp)

        # Attenuation should reduce peak amplitude.
        peak_noatt = np.abs(rf_noatt).max()
        peak_att = np.abs(rf_att).max()
        if peak_noatt > 1e-10:
            assert peak_att < peak_noatt, (
                f"Attenuation must reduce amplitude: {peak_att} should be < {peak_noatt}"
            )


class TestReceptionDownsampling:
    """Downsampling output."""

    def test_downsampling_reduces_nt(self, simple_tx, simple_rx, on_axis_scatterer):
        """downsampling=10 → output Nt = ceil(Nt_full / 10)."""
        pos, amp = on_axis_scatterer
        sim = Reception(simple_tx, simple_rx, verbose=False)

        rf_full, _ = sim(pos, amp)
        rf_ds, coords_ds = sim(pos, amp, downsampling=10)

        Nt_full = rf_full.shape[0]
        Nt_ds = rf_ds.shape[0]
        expected_Nt = len(range(0, Nt_full, 10))
        assert Nt_ds == expected_Nt, f"Expected {expected_Nt} samples, got {Nt_ds}."
        assert coords_ds["dt"] == 10 * (1.0 / sim.fs)


class TestReceptionSet:
    """Runtime parameter update via .set()."""

    def test_set_alpha0(self, simple_tx, simple_rx):
        sim = Reception(simple_tx, simple_rx, verbose=False)
        sim.set("alpha0", 0.5)
        assert sim.alpha0 == 0.5

    def test_set_excitation(self, simple_tx, simple_rx):
        exc = np.sin(2 * np.pi * 5e6 * np.arange(0, 1e-6, 5e-9)).astype(np.float32)
        sim = Reception(simple_tx, simple_rx, verbose=False)
        sim.set("excitation", exc)
        assert sim.excitation is not None

    def test_set_unknown_raises(self, simple_tx, simple_rx):
        sim = Reception(simple_tx, simple_rx, verbose=False)
        with pytest.raises(ValueError, match="Unknown parameter"):
            sim.set("nonexistent", 42)

    def test_set_wrong_type_raises(self, simple_tx, simple_rx):
        sim = Reception(simple_tx, simple_rx, verbose=False)
        with pytest.raises(TypeError):
            sim.set("c", "fast")

    def test_set_tx_refreshes(self, simple_tx, simple_rx):
        """Setting 'tx' must refresh sub-element attributes."""
        sim = Reception(simple_tx, simple_rx, verbose=False)
        old_centers = sim._tx_centers.copy()

        new_tx = LinearArrayTransducer(
            n_elements=2,
            element_width_mm=0.5,
            element_height_mm=5.0,
            kerf_mm=0.1,
            no_sub_x=1,
            no_sub_y=1,
            frequency_Hz=5e6,
        )
        sim.set("tx", new_tx)
        assert sim._tx_centers.shape != old_centers.shape


class TestReceptionSequence:
    """compute_sequence with multiple TX events."""

    def test_single_event_matches_call(self, simple_tx, simple_rx, on_axis_scatterer):
        """compute_sequence with 1 event == __call__ with same delays/apod."""
        pos, amp = on_axis_scatterer
        sim = Reception(simple_tx, simple_rx, verbose=False)

        rf_single, coords_single = sim(pos, amp)
        rf_seq, coords_seq = sim.compute_sequence(
            pos,
            amp,
            [{"delays": simple_tx.delays, "apodization": simple_tx.apodization}],
        )

        assert rf_seq.shape[0] == 1  # 1 event
        assert rf_seq.shape[2] == simple_rx.n_elements
        assert_allclose(
            rf_seq[0, : rf_single.shape[0], :],
            rf_single,
            rtol=1e-5,
            atol=1e-30,
            err_msg="Single-event compute_sequence must match __call__.",
        )

    def test_sequence_restores_tx_state(self, simple_tx, simple_rx, on_axis_scatterer):
        """TX delays/apod must be restored after compute_sequence."""
        pos, amp = on_axis_scatterer
        sim = Reception(simple_tx, simple_rx, verbose=False)

        orig_delays = simple_tx.delays.copy()
        orig_apod = simple_tx.apodization.copy()

        events = [
            {
                "delays": np.zeros_like(orig_delays),
                "apodization": np.ones_like(orig_apod),
            },
        ]
        sim.compute_sequence(pos, amp, events)

        np.testing.assert_array_equal(simple_tx.delays, orig_delays)
        np.testing.assert_array_equal(simple_tx.apodization, orig_apod)


class TestReceptionRepr:
    """String representation."""

    def test_repr(self, simple_tx, simple_rx):
        sim = Reception(simple_tx, simple_rx, verbose=False)
        r = repr(sim)
        assert "Reception" in r
        assert "1540" in r
