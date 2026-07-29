"""Tests for Reception class — pulse-echo RF simulation."""

import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pyfield.reception import Reception
from pyfield.transducers import LinearArrayTransducer
from pyfield.utilities.helper_functions import create_3D_spatial_grid_from_points


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
        sim = Reception(simple_tx, simple_rx, verbose=False)
        rf, coords = sim(pos, amp)

        assert rf.ndim == 2
        assert rf.shape[0] == simple_rx.n_elements
        assert rf.dtype == np.float32
        assert "t0" in coords
        assert "dt" in coords

    def test_rf_nonzero(self, simple_tx, simple_rx, on_axis_scatterer):
        """RF must be non-zero for on-axis scatterer."""
        pos, amp = on_axis_scatterer
        sim = Reception(simple_tx, simple_rx, verbose=False)
        rf, _ = sim(pos, amp)
        assert np.any(rf != 0), "RF should be non-zero for on-axis scatterer."

    def test_grid_dict_input(self, simple_tx, simple_rx):
        """A grid dict must equal the same lattice passed as explicit points."""
        grid = {
            "x_extent": [-1.0, 1.0],
            "y_extent": [0.0, 0.0],
            "z_extent": [18.0, 22.0],
            "dx": 1.0,
            "dy": 1.0,
            "dz": 2.0,
        }
        sim = Reception(simple_tx, simple_rx, verbose=False)
        rf_dict, _ = sim(grid)
        *_, pts_m = create_3D_spatial_grid_from_points(grid)
        rf_pts, _ = sim(pts_m * 1e3)
        # The two paths quantise positions differently (mm-float32 vs m-float64
        # before the final float32 cast), so compare against the RF peak.
        peak = np.abs(rf_pts).max()
        assert peak > 0
        assert_allclose(rf_dict, rf_pts, atol=1e-3 * peak)

    def test_self_echo_valid(self, simple_tx, on_axis_scatterer):
        """TX == RX (same transducer) must produce valid result.

        ``simple_tx`` is focused, so reusing it as RX legitimately triggers the
        non-default RX delays/apodization warning (they are applied per element).
        """
        pos, amp = on_axis_scatterer
        with pytest.warns(UserWarning, match="per receive element"):
            sim = Reception(simple_tx, simple_tx, verbose=False)
        rf, coords = sim(pos, amp)

        assert rf.ndim == 2
        assert rf.shape[0] == simple_tx.n_elements
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

        Nt_full = rf_full.shape[1]  # (E_rx, Nt) → time is last axis
        Nt_ds = rf_ds.shape[1]
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
    """sequence_rf with multiple TX events."""

    def test_single_event_matches_call(self, simple_tx, simple_rx, on_axis_scatterer):
        """sequence_rf with 1 event == __call__ with same delays/apod."""
        pos, amp = on_axis_scatterer
        sim = Reception(simple_tx, simple_rx, verbose=False)

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
        sim = Reception(simple_tx, simple_rx, verbose=False)

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
        sim = Reception(simple_tx, simple_rx, verbose=False)
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
            Reception(simple_tx, simple_rx, method="bogus", verbose=False)
        sim = Reception(simple_tx, simple_rx, verbose=False)
        with pytest.raises(ValueError, match="Unknown method"):
            sim.set("method", "bogus")

    def test_all_methods_agree(self, simple_tx, simple_rx):
        """fst ≈ spectral ≈ paired produce the same physical RF."""
        exc = self._exc()
        simple_tx.impulse_response = exc  # band-limited chain → splat is exact
        simple_rx.impulse_response = exc
        pos = np.array([[0, 0, 18], [1.0, 0, 22], [-1.5, 0, 26]], dtype=np.float32)
        amp = np.array([1.0, 0.8, 1.2], dtype=np.float32)
        cases = {
            "fst": {"method": "fst"},
            "spectral": {"method": "spectral"},
            "paired": {"method": "paired"},
        }
        out = {}
        for name, kw in cases.items():
            # paired warns (pedagogic reference); silence it here.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                sim = Reception(
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

        ref = out["fst"]
        pk_ref = float(np.abs(ref).max())
        for name, v in out.items():
            assert corr(v, ref) > 0.99, name
            assert abs(float(np.abs(v).max()) / pk_ref - 1) < 0.05, name

    def test_default_method_is_spectral(self):
        """The default (unspecified) method is the fast spectral core."""
        exc = self._exc()
        sim = Reception(self._big_tx(), self._big_tx(), excitation=exc, verbose=False)
        sim.pulse_echo_rf(np.array([[0, 0, 30]], dtype=np.float32))
        assert sim._last_method == "spectral"

    def test_auto_delegates_to_conventional(self):
        """method='auto' routes to the conventional backend (its SIR-kernel auto-picker)."""
        sim = Reception(self._big_tx(), self._big_tx(), method="auto", verbose=False)
        sim.pulse_echo_rf(
            np.array([[0, 0, 30]], dtype=np.float32),
            amplitudes=np.array([1.0], dtype=np.float32),
        )
        assert sim._last_method == "auto"

    def test_paired_warns_pedagogic(self, simple_tx, simple_rx):
        """Selecting the pedagogic 'paired' method warns that it is slow."""
        with pytest.warns(UserWarning, match="pedagogic"):
            Reception(simple_tx, simple_rx, method="paired", verbose=False)

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

        binned = Reception(
            tx, tx, fs=100e6, excitation=exc, method="spectral", verbose=False
        )
        n_bins = binned._auto_depth_bins(pos * 1e-3, max(int(tx.delays.shape[0]), 2))
        assert n_bins > 1, "test needs a depth spread that triggers binning"
        rf_b, _ = binned.pulse_echo_rf(pos, amp)
        assert binned._last_method == "spectral"

        single = Reception(
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

    def test_explicit_method_used_verbatim(self, simple_tx, simple_rx):
        """A conventional-family method delegates and is reported verbatim."""
        pos = np.array([[0, 0, 20]], dtype=np.float32)
        amp = np.array([1.0], dtype=np.float32)
        sim = Reception(simple_tx, simple_rx, method="fst", verbose=False)
        sim.pulse_echo_rf(pos, amp)
        assert sim._last_method == "fst"

    def test_paired_attenuation_not_supported(self, simple_tx, simple_rx):
        exc = self._exc()
        with pytest.warns(UserWarning, match="pedagogic"):
            sim = Reception(
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
        sim_att = Reception(
            simple_tx,
            simple_rx,
            fs=100e6,
            excitation=exc,
            alpha0=0.8,
            method="spectral",
            verbose=False,
        )
        sim_no = Reception(
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


class TestSequenceCheckpoint:
    """sequence_rf(out_path=...) — checkpointed, resumable acquisition."""

    @staticmethod
    def _events(tx):
        return [
            {"delays": np.zeros(4, np.float32), "apodization": np.ones(4, np.float32)},
            {
                "delays": np.full(4, 1e-7, np.float32),
                "apodization": np.ones(4, np.float32),
            },
        ]

    def test_checkpoint_matches_in_ram(
        self, simple_tx, simple_rx, on_axis_scatterer, tmp_path
    ):
        """Disk round-trip must return exactly the in-RAM sequence result."""
        pos, amp = on_axis_scatterer
        sim = Reception(simple_tx, simple_rx, verbose=False)
        events = self._events(simple_tx)

        rf_ram, coords_ram = sim.sequence_rf(pos, amp, events)
        rf_ck, coords_ck = sim.sequence_rf(pos, amp, events, out_path=tmp_path / "ds")

        np.testing.assert_array_equal(rf_ck, rf_ram)
        np.testing.assert_array_equal(
            coords_ck["t0_per_event"], coords_ram["t0_per_event"]
        )
        assert coords_ck["dt"] == coords_ram["dt"]

    def test_resume_skips_completed_events(
        self, simple_tx, simple_rx, on_axis_scatterer, tmp_path
    ):
        """Re-running recomputes ONLY missing events, and the result is intact."""
        pos, amp = on_axis_scatterer
        sim = Reception(simple_tx, simple_rx, verbose=False)
        events = self._events(simple_tx)
        out = tmp_path / "ds"

        rf_full, _ = sim.sequence_rf(pos, amp, events, out_path=out)

        calls = []
        orig = sim.pulse_echo_rf
        sim.pulse_echo_rf = lambda *a, **k: (calls.append(1), orig(*a, **k))[1]

        # Everything on disk: nothing recomputed.
        rf_again, _ = sim.sequence_rf(pos, amp, events, out_path=out)
        assert len(calls) == 0
        np.testing.assert_array_equal(rf_again, rf_full)

        # Simulated crash: one event file lost — only that one recomputed.
        (out / "rf_event_0001.npz").unlink()
        rf_resumed, _ = sim.sequence_rf(pos, amp, events, out_path=out)
        assert len(calls) == 1
        np.testing.assert_array_equal(rf_resumed, rf_full)

    def test_changed_setup_refused(
        self, simple_tx, simple_rx, on_axis_scatterer, tmp_path
    ):
        """Same folder + different scatterers must raise, never mix data."""
        pos, amp = on_axis_scatterer
        sim = Reception(simple_tx, simple_rx, verbose=False)
        events = self._events(simple_tx)
        out = tmp_path / "ds"
        sim.sequence_rf(pos, amp, events, out_path=out)

        with pytest.raises(ValueError, match="DIFFERENT"):
            sim.sequence_rf(pos, amp * 2.0, events, out_path=out)

    @staticmethod
    def _cloud(n=30):
        rng = np.random.default_rng(7)
        pos = np.column_stack(
            [
                rng.uniform(-2, 2, n),
                rng.uniform(-1, 1, n),
                rng.uniform(15, 25, n),
            ]
        ).astype(np.float32)
        return pos, rng.standard_normal(n).astype(np.float32)

    def test_chunked_sum_is_chunk_count_invariant(self, simple_tx, simple_rx, tmp_path):
        """Splitting the cloud into 2 or 3 chunks must give the same RF.

        The grid sentinels pin one time grid per event regardless of the chunk
        count, so the only difference is float32 summation order — any real
        misalignment or lost/duplicated scatterer breaks this hard.
        """
        pos, amp = self._cloud()
        sim = Reception(simple_tx, simple_rx, verbose=False)
        events = self._events(simple_tx)

        rf2, c2 = sim.sequence_rf(
            pos, amp, events, out_path=tmp_path / "k2", checkpoint_chunks=2
        )
        rf3, c3 = sim.sequence_rf(
            pos, amp, events, out_path=tmp_path / "k3", checkpoint_chunks=3
        )
        np.testing.assert_array_equal(c2["t0_per_event"], c3["t0_per_event"])
        peak = np.abs(rf2).max()
        np.testing.assert_allclose(rf2, rf3, rtol=1e-4, atol=1e-5 * peak)

    def test_chunked_physics_matches_unchunked(self, simple_tx, simple_rx, tmp_path):
        """Chunked RF must be the same waveform as the unchunked one.

        The sentinels widen the time window slightly (earlier ``t0``), so the
        two grids sample the identical band-limited signal at offset times —
        compare after interpolating onto the unchunked absolute time axis.
        """
        pos, amp = self._cloud()
        sim = Reception(simple_tx, simple_rx, verbose=False)
        events = self._events(simple_tx)[:1]

        rf_ref, c_ref = sim.sequence_rf(pos, amp, events, out_path=tmp_path / "u")
        rf_ck, c_ck = sim.sequence_rf(
            pos, amp, events, out_path=tmp_path / "c", checkpoint_chunks=3
        )
        assert c_ck["t0"] <= c_ref["t0"]  # sentinel margin only widens.

        dt = c_ref["dt"]
        t_ref = c_ref["t0"] + dt * np.arange(rf_ref.shape[2])
        t_ck = c_ck["t0"] + dt * np.arange(rf_ck.shape[2])
        peak = np.abs(rf_ref).max()
        for e in range(rf_ref.shape[1]):
            aligned = np.interp(t_ref, t_ck, rf_ck[0, e].astype(np.float64))
            np.testing.assert_allclose(aligned, rf_ref[0, e], atol=0.02 * peak)

    def test_chunked_resume_recomputes_only_missing_chunk(
        self, simple_tx, simple_rx, tmp_path
    ):
        """Losing one chunk file must cost exactly one chunk, not an event."""
        pos, amp = self._cloud()
        sim = Reception(simple_tx, simple_rx, verbose=False)
        events = self._events(simple_tx)
        out = tmp_path / "ds"
        rf_full, _ = sim.sequence_rf(
            pos, amp, events, out_path=out, checkpoint_chunks=3
        )

        calls = []
        orig = sim.pulse_echo_rf
        sim.pulse_echo_rf = lambda *a, **k: (calls.append(1), orig(*a, **k))[1]
        (out / "rf_event_0004.npz").unlink()  # event 1, chunk 1.
        rf_resumed, _ = sim.sequence_rf(
            pos, amp, events, out_path=out, checkpoint_chunks=3
        )
        assert len(calls) == 1
        np.testing.assert_array_equal(rf_resumed, rf_full)

    def test_chunks_require_out_path(self, simple_tx, simple_rx):
        pos, amp = self._cloud()
        sim = Reception(simple_tx, simple_rx, verbose=False)
        with pytest.raises(ValueError, match="out_path"):
            sim.sequence_rf(pos, amp, self._events(simple_tx), checkpoint_chunks=4)

    def test_shared_tx_rx_with_events_refused(self, simple_tx):
        """Same object as TX and RX + per-event delays must raise: the event's
        TX delays would also time-shift every receive channel (RX weights are
        applied per element), silently corrupting the RF."""
        pos, amp = self._cloud()
        with pytest.warns(UserWarning, match="per receive element"):
            sim = Reception(simple_tx, simple_tx, verbose=False)
        with pytest.raises(ValueError, match="same transducer"):
            sim.sequence_rf(pos, amp, self._events(simple_tx))

    def test_pulse_echo_checkpointed_matches_direct(
        self, simple_tx, simple_rx, tmp_path
    ):
        """pulse_echo_rf(out_path=...) must equal the direct call exactly.

        The checkpointed path wraps the CURRENT TX focus into a one-event
        sequence; without chunking there are no sentinels, so the RF and time
        grid are bit-identical — and the focus state is fingerprinted, so a
        refocused re-run on the same folder refuses.
        """
        pos, amp = self._cloud()
        sim = Reception(simple_tx, simple_rx, verbose=False)

        rf_direct, c_direct = sim.pulse_echo_rf(pos, amp)
        rf_ck, c_ck = sim.pulse_echo_rf(pos, amp, out_path=tmp_path / "pe")
        np.testing.assert_array_equal(rf_ck, rf_direct)
        # t0 may differ by one float64 ulp (different summation order).
        np.testing.assert_allclose(c_ck["t0"], c_direct["t0"], rtol=0, atol=1e-12)
        assert c_ck["dt"] == c_direct["dt"]

        simple_tx.compute_delays(focus_mm=[0, 0, 40])  # different focus state.
        with pytest.raises(ValueError, match="DIFFERENT"):
            sim.pulse_echo_rf(pos, amp, out_path=tmp_path / "pe")

        with pytest.raises(ValueError, match="per_scatterer"):
            sim.pulse_echo_rf(pos, amp, per_scatterer=True, out_path=tmp_path / "psf")

    def test_synthetic_aperture_checkpointed_matches_in_ram(
        self, simple_tx, simple_rx, tmp_path
    ):
        """FMC via checkpoints must equal the in-RAM run (one file per group)."""
        pos, amp = self._cloud()
        sim = Reception(simple_tx, simple_rx, verbose=False)

        rf_ram, c_ram = sim.synthetic_aperture_rf(
            pos, amp, decimation=2, countdown=False
        )
        rf_ck, c_ck = sim.synthetic_aperture_rf(
            pos, amp, decimation=2, out_path=tmp_path / "fmc"
        )
        assert rf_ram.shape == (
            simple_tx.n_elements,
            simple_rx.n_elements,
            rf_ram.shape[2],
        )
        np.testing.assert_array_equal(rf_ck, rf_ram)
        assert c_ck["t0"] == c_ram["t0"] and c_ck["dt"] == c_ram["dt"]
