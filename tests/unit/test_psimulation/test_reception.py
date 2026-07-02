"""Tests for Reception class — pulse-echo RF simulation."""

import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pyfield.reception import ReceptionSDI
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
        """RF output shape must be (E_rx, Nt)."""
        pos, amp = on_axis_scatterer
        sim = ReceptionSDI(simple_tx, simple_rx, verbose=False)
        rf, coords = sim(pos, amp)

        assert rf.ndim == 2
        assert rf.shape[0] == simple_rx.n_elements
        assert rf.dtype == np.float32
        assert "t0" in coords
        assert "dt" in coords

    def test_rf_nonzero(self, simple_tx, simple_rx, on_axis_scatterer):
        """RF must be non-zero for on-axis scatterer."""
        pos, amp = on_axis_scatterer
        sim = ReceptionSDI(simple_tx, simple_rx, verbose=False)
        rf, _ = sim(pos, amp)
        assert np.any(rf != 0), "RF should be non-zero for on-axis scatterer."

    def test_self_echo_valid(self, simple_tx, on_axis_scatterer):
        """TX == RX (same transducer) must produce valid result.

        ``simple_tx`` is focused, so reusing it as RX legitimately triggers the
        non-default RX delays/apodization warning (they are applied per element).
        """
        pos, amp = on_axis_scatterer
        with pytest.warns(UserWarning, match="per receive element"):
            sim = ReceptionSDI(simple_tx, simple_tx, verbose=False)
        rf, coords = sim(pos, amp)

        assert rf.ndim == 2
        assert rf.shape[0] == simple_tx.n_elements
        assert np.any(rf != 0)

    def test_excitation_none_pure_pe_sir(self, simple_tx, simple_rx, on_axis_scatterer):
        """excitation=None → pure PE SIR derivative (no excitation conv)."""
        pos, amp = on_axis_scatterer
        sim = ReceptionSDI(simple_tx, simple_rx, excitation=None, verbose=False)
        rf, _ = sim(pos, amp)
        assert rf.ndim == 2
        assert np.any(rf != 0)


class TestReceptionAttenuation:
    """Attenuation handling."""

    def test_no_attenuation_default(self, simple_tx, simple_rx, on_axis_scatterer):
        """alpha0=None → no attenuation applied."""
        pos, amp = on_axis_scatterer
        sim = ReceptionSDI(simple_tx, simple_rx, alpha0=None, verbose=False)
        rf_noatt, _ = sim(pos, amp)
        assert np.any(rf_noatt != 0)

    def test_attenuation_reduces_amplitude(self, simple_tx, simple_rx):
        """alpha0 > 0 must reduce RF amplitude compared to no attenuation."""
        pos = np.array([[0, 0, 30]], dtype=np.float32)  # mm, deeper point
        amp = np.array([1.0], dtype=np.float32)

        sim_noatt = ReceptionSDI(simple_tx, simple_rx, alpha0=None, verbose=False)
        rf_noatt, _ = sim_noatt(pos, amp)

        sim_att = ReceptionSDI(
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
        sim = ReceptionSDI(simple_tx, simple_rx, verbose=False)

        rf_full, _ = sim(pos, amp)
        rf_ds, coords_ds = sim(pos, amp, downsampling=10)

        Nt_full = rf_full.shape[1]  # (E_rx, Nt) → time is last axis
        Nt_ds = rf_ds.shape[1]
        expected_Nt = len(range(0, Nt_full, 10))
        assert Nt_ds == expected_Nt, f"Expected {expected_Nt} samples, got {Nt_ds}."
        assert coords_ds["dt"] == 10 * (1.0 / sim.fs)


class TestReceptionSet:
    """Runtime parameter update via .set()."""

    def test_set_alpha0(self, simple_tx, simple_rx):
        sim = ReceptionSDI(simple_tx, simple_rx, verbose=False)
        sim.set("alpha0", 0.5)
        assert sim.alpha0 == 0.5

    def test_set_excitation(self, simple_tx, simple_rx):
        exc = np.sin(2 * np.pi * 5e6 * np.arange(0, 1e-6, 5e-9)).astype(np.float32)
        sim = ReceptionSDI(simple_tx, simple_rx, verbose=False)
        sim.set("excitation", exc)
        assert sim.excitation is not None

    def test_set_unknown_raises(self, simple_tx, simple_rx):
        sim = ReceptionSDI(simple_tx, simple_rx, verbose=False)
        with pytest.raises(ValueError, match="Unknown parameter"):
            sim.set("nonexistent", 42)

    def test_set_wrong_type_raises(self, simple_tx, simple_rx):
        sim = ReceptionSDI(simple_tx, simple_rx, verbose=False)
        with pytest.raises(TypeError):
            sim.set("c", "fast")

    def test_set_tx_refreshes(self, simple_tx, simple_rx):
        """Setting 'tx' must refresh sub-element attributes."""
        sim = ReceptionSDI(simple_tx, simple_rx, verbose=False)
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
    """sequence_rf with multiple TX events."""

    def test_single_event_matches_call(self, simple_tx, simple_rx, on_axis_scatterer):
        """sequence_rf with 1 event == __call__ with same delays/apod."""
        pos, amp = on_axis_scatterer
        sim = ReceptionSDI(simple_tx, simple_rx, verbose=False)

        rf_single, coords_single = sim(pos, amp)
        rf_seq, coords_seq = sim.sequence_rf(
            pos,
            amp,
            [{"delays": simple_tx.delays, "apodization": simple_tx.apodization}],
        )

        assert rf_seq.shape[0] == 1  # 1 event
        assert rf_seq.shape[1] == simple_rx.n_elements  # (Nev, E_rx, Nt)
        assert_allclose(
            rf_seq[0, :, : rf_single.shape[1]],
            rf_single,
            rtol=1e-5,
            atol=1e-30,
            err_msg="Single-event sequence_rf must match __call__.",
        )

    def test_sequence_restores_tx_state(self, simple_tx, simple_rx, on_axis_scatterer):
        """TX delays/apod must be restored after sequence_rf."""
        pos, amp = on_axis_scatterer
        sim = ReceptionSDI(simple_tx, simple_rx, verbose=False)

        orig_delays = simple_tx.delays.copy()
        orig_apod = simple_tx.apodization.copy()

        events = [
            {
                "delays": np.zeros_like(orig_delays),
                "apodization": np.ones_like(orig_apod),
            },
        ]
        sim.sequence_rf(pos, amp, events)

        np.testing.assert_array_equal(simple_tx.delays, orig_delays)
        np.testing.assert_array_equal(simple_tx.apodization, orig_apod)

    def test_t0_per_event(self, simple_tx, simple_rx, on_axis_scatterer):
        """Each event reports its own beam-axis t0; first equals coords['t0']."""
        pos, amp = on_axis_scatterer
        sim = ReceptionSDI(simple_tx, simple_rx, verbose=False)
        n_el = simple_tx.n_elements
        events = [
            {"delays": np.zeros(n_el, dtype=np.float32)},
            {"delays": np.full(n_el, 1e-7, dtype=np.float32)},
        ]
        _, coords = sim.sequence_rf(pos, amp, events)
        t0s = coords["t0_per_event"]
        assert t0s.shape == (2,)
        assert t0s[0] == coords["t0"]
        # Zero delays vs a uniform 100 ns bulk: t0 is beam-axis referenced
        # (delays.max() subtracted), so event 2's origin sits 100 ns earlier —
        # exactly what cancels the bulk-shifted echo inside the trace.
        assert_allclose(t0s[1], t0s[0] - 1e-7, atol=1e-12)


class TestReceptionFormulations:
    """method selector: auto router + conventional/spectral/paired equivalence."""

    @staticmethod
    def _exc(fs=100e6, fc=5e6):
        t = np.arange(0, 2.0 / fc, 1.0 / fs)
        return (np.sin(2 * np.pi * fc * t) * np.hanning(len(t))).astype(np.float32)

    @staticmethod
    def _big_tx(n=32):
        # Many patches per element (3×6) so the paired M² placement clearly exceeds the
        # patch-independent transform cost → the router leaves the paired regime.
        return LinearArrayTransducer(
            n_elements=n,
            element_width_mm=0.25,
            element_height_mm=10.0,
            kerf_mm=0.05,
            no_sub_x=3,
            no_sub_y=6,
            frequency_Hz=5e6,
        )

    def test_invalid_method_raises(self, simple_tx, simple_rx):
        with pytest.raises(ValueError, match="Unknown method"):
            ReceptionSDI(simple_tx, simple_rx, method="bogus", verbose=False)
        sim = ReceptionSDI(simple_tx, simple_rx, verbose=False)
        with pytest.raises(ValueError, match="Unknown method"):
            sim.set("method", "bogus")

    def test_all_methods_agree(self, simple_tx, simple_rx):
        """conventional ≈ spectral ≈ paired produce the same physical RF."""
        exc = self._exc()
        simple_tx.impulse_response = exc  # band-limited chain → splat is exact
        simple_rx.impulse_response = exc
        pos = np.array([[0, 0, 18], [1.0, 0, 22], [-1.5, 0, 26]], dtype=np.float32)
        amp = np.array([1.0, 0.8, 1.2], dtype=np.float32)
        cases = {
            "conventional": {"method": "conventional"},
            "spectral": {"method": "spectral"},
            "paired": {"method": "paired"},
        }
        out = {}
        for name, kw in cases.items():
            sim = ReceptionSDI(
                simple_tx,
                simple_rx,
                fs=100e6,
                c=1540,
                excitation=exc,
                verbose=False,
                **kw,
            )
            out[name], _ = sim.pulse_echo_rf(pos, amp)
        n = min(v.shape[-1] for v in out.values())

        def corr(a, b):
            return np.corrcoef(a[..., :n].ravel(), b[..., :n].ravel())[0, 1]

        ref = out["conventional"]
        pk_ref = float(np.abs(ref).max())
        for name, v in out.items():
            assert corr(v, ref) > 0.99, name
            assert abs(float(np.abs(v).max()) / pk_ref - 1) < 0.05, name

    def test_router_paired_for_monoelement(self):
        """auto on a near-monoelement aperture → paired (its splat cost is tiny there).

        `paired` splats the full integrated drive per patch pair, so it only undercuts the
        transform when there are barely any patches — a single 1×1 element each way.
        """
        exc = self._exc()
        mono = LinearArrayTransducer(
            n_elements=1,
            element_width_mm=0.25,
            element_height_mm=5.0,
            kerf_mm=0.05,
            no_sub_x=1,
            no_sub_y=1,
            frequency_Hz=5e6,
        )
        sim = ReceptionSDI(mono, mono, excitation=exc, verbose=False)
        sim.pulse_echo_rf(np.array([[0, 0, 20]], dtype=np.float32), per_scatterer=True)
        assert sim._last_method == "paired"

    def test_router_spectral_for_large_aperture(self):
        """auto on a many-patch aperture with band-limited drive → spectral."""
        exc = self._exc()
        sim = ReceptionSDI(
            self._big_tx(), self._big_tx(), excitation=exc, verbose=False
        )
        sim.pulse_echo_rf(np.array([[0, 0, 30]], dtype=np.float32))
        assert sim._last_method == "spectral"

    def test_router_conventional_without_band_limit(self):
        """auto with no excitation/impulse-response on a large aperture → conventional."""
        sim = ReceptionSDI(self._big_tx(), self._big_tx(), verbose=False)  # wideband
        sim.pulse_echo_rf(
            np.array([[0, 0, 30]], dtype=np.float32),
            amplitudes=np.array([1.0], dtype=np.float32),
        )
        assert sim._last_method == "conventional"

    def test_spectral_binning_matches_single_window(self):
        """Depth-binned spectral RF == single-window spectral RF (binning is exact).

        Many scatterers spread over depth force depth binning (>1 bin); each bin's window
        is snapped to one global sample lattice and added back at an integer offset, so the
        binned RF must reproduce the unbinned single-window result. This also exercises the
        fused two-way kernel that builds the summed spectrum.
        """
        exc = self._exc()
        tx = self._big_tx()
        rng = np.random.default_rng(0)
        P = 1500
        pos = np.column_stack(
            [rng.uniform(-5, 5, P), np.zeros(P), rng.uniform(20, 60, P)]
        ).astype(np.float32)
        amp = rng.standard_normal(P).astype(np.float32)

        binned = ReceptionSDI(
            tx, tx, fs=100e6, excitation=exc, method="spectral", verbose=False
        )
        n_bins = binned._auto_depth_bins(pos * 1e-3, max(int(tx.delays.shape[0]), 2))
        assert n_bins > 1, "test needs a depth spread that triggers binning"
        rf_b, _ = binned.pulse_echo_rf(pos, amp)
        assert binned._last_method == "spectral"

        single = ReceptionSDI(
            tx,
            tx,
            fs=100e6,
            excitation=exc,
            method="spectral",
            n_depth_bins=1,
            verbose=False,
        )
        rf_s, _ = single.pulse_echo_rf(pos, amp)

        n = min(rf_b.shape[-1], rf_s.shape[-1])
        corr = np.corrcoef(rf_b[..., :n].ravel(), rf_s[..., :n].ravel())[0, 1]
        assert corr > 0.999

    def test_explicit_method_overrides_router(self, simple_tx, simple_rx):
        """A non-auto method is used verbatim (no regime select)."""
        pos = np.array([[0, 0, 20]], dtype=np.float32)
        amp = np.array([1.0], dtype=np.float32)
        sim = ReceptionSDI(simple_tx, simple_rx, method="conventional", verbose=False)
        sim.pulse_echo_rf(pos, amp)
        assert sim._last_method == "conventional"

    def test_paired_attenuation_not_supported(self, simple_tx, simple_rx):
        exc = self._exc()
        sim = ReceptionSDI(
            simple_tx,
            simple_rx,
            fs=100e6,
            excitation=exc,
            alpha0=0.5,
            method="paired",
            verbose=False,
        )
        with pytest.raises(NotImplementedError, match="attenuation"):
            sim.pulse_echo_rf(np.array([[0, 0, 20]], dtype=np.float32))

    def test_spectral_attenuation_decays_with_depth(self, simple_tx, simple_rx):
        """spectral folds per-patch attenuation in: deeper scatterers echo weaker."""
        exc = self._exc()
        pos = np.array([[0, 0, 20], [0, 0, 40]], dtype=np.float32)
        amp = np.ones(2, dtype=np.float32)
        sim_att = ReceptionSDI(
            simple_tx,
            simple_rx,
            fs=100e6,
            excitation=exc,
            alpha0=0.8,
            method="spectral",
            verbose=False,
        )
        sim_no = ReceptionSDI(
            simple_tx,
            simple_rx,
            fs=100e6,
            excitation=exc,
            method="spectral",
            verbose=False,
        )
        psf_att, _ = sim_att.pulse_echo_rf(pos, amp, per_scatterer=True)
        psf_no, _ = sim_no.pulse_echo_rf(pos, amp, per_scatterer=True)
        r20 = np.abs(psf_att[0]).max() / np.abs(psf_no[0]).max()
        r40 = np.abs(psf_att[1]).max() / np.abs(psf_no[1]).max()
        assert r20 < 1.0  # attenuation reduces amplitude
        assert r40 < r20  # deeper scatterer attenuates more


class TestReceptionRepr:
    """String representation."""

    def test_repr(self, simple_tx, simple_rx):
        sim = ReceptionSDI(simple_tx, simple_rx, verbose=False)
        r = repr(sim)
        assert "Reception" in r
        assert "1540" in r
