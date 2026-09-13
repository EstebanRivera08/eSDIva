"""Tests for Emission class — Batch 3 test gate."""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_excitation(fs=200e6, fc=5e6, n_cycles=2):
    t = np.arange(0, n_cycles / fc, 1.0 / fs)
    return np.sin(2.0 * np.pi * fc * t).astype(np.float32)


def _make_emission(tx, **kwargs):
    from esdiva.emission import Emission

    return Emission(tx, **kwargs)


# ---------------------------------------------------------------------------
# Emission instantiation and .set()
# ---------------------------------------------------------------------------


class TestEmissionInit:
    def test_set_valid(self, small_linear_transducer):
        sim = _make_emission(small_linear_transducer)
        sim.set("alpha0", 0.5)
        assert sim.alpha0 == 0.5
        sim.set("alpha0", None)
        assert sim.alpha0 is None

    def test_set_monochromatic(self, small_linear_transducer):
        sim = _make_emission(small_linear_transducer)
        sim.set("monochromatic", True)
        assert sim.monochromatic is True

    def test_set_unknown_raises(self, small_linear_transducer):
        sim = _make_emission(small_linear_transducer)
        with pytest.raises(ValueError, match="Unknown parameter"):
            sim.set("nonexistent", 42)

    def test_set_wrong_type_raises(self, small_linear_transducer):
        sim = _make_emission(small_linear_transducer)
        with pytest.raises(TypeError):
            sim.set("c", "not_a_float")


# ---------------------------------------------------------------------------
# Emission.__call__ — monochromatic (CW) path
# ---------------------------------------------------------------------------


class TestEmissionMonochromatic:
    def test_structured_grid_output_shape(
        self, small_linear_transducer, small_field_grid
    ):
        sim = _make_emission(small_linear_transducer, monochromatic=True)
        p, coords = sim(small_field_grid)
        assert p.ndim == 3
        assert "x" in coords and "y" in coords and "z" in coords

    def test_raw_points_output_shape(self, small_linear_transducer):
        sim = _make_emission(small_linear_transducer, monochromatic=True)
        pts = np.array([[0.0, 0.0, 20.0], [0.0, 0.0, 25.0]], dtype=np.float32)
        p, coords = sim(pts)
        assert p.shape == (2,)


# ---------------------------------------------------------------------------
# Emission.__call__ — pulsed path (exc=None)
# ---------------------------------------------------------------------------


class TestEmissionPulsed:
    def test_output_shape_structured(self, small_linear_transducer, small_field_grid):
        sim = _make_emission(small_linear_transducer)
        p, coords = sim(small_field_grid)
        assert p.ndim == 4
        assert "t0" in coords and "dt" in coords
        assert "x" in coords

    def test_output_shape_raw_points(self, small_linear_transducer):
        sim = _make_emission(small_linear_transducer)
        pts = np.array([[0.0, 0.0, 20.0]], dtype=np.float32)
        p, coords = sim(pts)
        assert p.ndim == 2
        assert p.shape[1] == 1
        assert "t0" in coords


# ---------------------------------------------------------------------------
# Emission.__call__ — global excitation path
# ---------------------------------------------------------------------------


class TestEmissionGlobalExcitation:
    def test_output_has_time_coords(self, small_linear_transducer, small_field_grid):
        exc = _make_excitation()
        sim = _make_emission(small_linear_transducer, excitation=exc)
        p, coords = sim(small_field_grid)
        assert p.ndim == 4
        assert "t0" in coords
        assert "dt" in coords

    def test_alpha0_none_same_as_no_attenuation(self, small_linear_transducer):
        exc = _make_excitation()
        pts = np.array([[0.0, 0.0, 20.0]], dtype=np.float32)

        sim_base = _make_emission(small_linear_transducer, excitation=exc)
        sim_att = _make_emission(small_linear_transducer, excitation=exc, alpha0=None)

        p_base, _ = sim_base(pts)
        p_att, _ = sim_att(pts)

        np.testing.assert_array_equal(p_base, p_att)


# ---------------------------------------------------------------------------
# Emission.__call__ — per-element excitation path
# ---------------------------------------------------------------------------


class TestEmissionPerElementExcitation:
    def test_uniform_per_element_matches_global(self, small_linear_transducer):
        """Uniform per-element excitation (same pulse × E) must equal global excitation."""
        n_el = small_linear_transducer.n_elements
        pulse = _make_excitation()
        exc_global = pulse
        exc_per_elem = np.tile(pulse[:, np.newaxis], (1, n_el))  # (L, E)

        pts = np.array([[0.0, 0.0, 20.0]], dtype=np.float32)

        # Same SIR source on both sides: this checks the per-element sum, not the method.
        sim_global = _make_emission(
            small_linear_transducer, excitation=exc_global, method="temporal"
        )
        sim_pe = _make_emission(
            small_linear_transducer, excitation=exc_per_elem, method="temporal"
        )

        p_global, _ = sim_global(pts)
        p_pe, _ = sim_pe(pts)

        # Both paths compute the same physics but via different float32 accumulation
        # orders, so tiny dh differences cause large *relative* errors near zero
        # crossings of the abs-valued signal.  Compare peak amplitude and total
        # energy instead, which are robust to zero-crossing phase shifts.
        peak = float(max(p_global.max(), p_pe.max()))
        np.testing.assert_allclose(
            p_pe.max(),
            p_global.max(),
            rtol=1e-3,
            err_msg="Peak amplitude must match between per-element and global paths.",
        )
        np.testing.assert_allclose(
            np.sum(p_pe**2),
            np.sum(p_global**2),
            rtol=1e-3,
            err_msg="Total signal energy must match between per-element and global paths.",
        )
        # Tight absolute tolerance: sample-wise difference bounded to 0.1 % of peak.
        np.testing.assert_allclose(
            p_pe,
            p_global,
            atol=peak * 1e-3,
            err_msg="Uniform per-element excitation must match global excitation output.",
        )

    def test_wrong_n_elements_raises(self, small_linear_transducer):
        n_el = small_linear_transducer.n_elements
        pulse = _make_excitation()
        bad_exc = np.tile(pulse[:, np.newaxis], (1, n_el + 1))  # wrong E
        pts = np.array([[0.0, 0.0, 20.0]], dtype=np.float32)
        sim = _make_emission(small_linear_transducer, excitation=bad_exc)
        with pytest.raises(ValueError, match="Per-element excitation"):
            sim(pts)

    def test_output_has_time_coords(self, small_linear_transducer, small_field_grid):
        n_el = small_linear_transducer.n_elements
        pulse = _make_excitation()
        exc_pe = np.tile(pulse[:, np.newaxis], (1, n_el))
        sim = _make_emission(small_linear_transducer, excitation=exc_pe)
        p, coords = sim(small_field_grid)
        assert "t0" in coords and "dt" in coords


# ---------------------------------------------------------------------------
# Transfer function
# ---------------------------------------------------------------------------


class TestTransferFunction:
    def test_identity_tf_matches_no_tf_global(self, small_linear_transducer):
        """TF(f)=1 must give same result as no transfer function (global exc)."""
        pulse = _make_excitation()
        pts = np.array([[0.0, 0.0, 20.0]], dtype=np.float32)

        sim_base = _make_emission(small_linear_transducer, excitation=pulse)
        sim_tf = _make_emission(
            small_linear_transducer,
            excitation=pulse,
            transfer_function=lambda f: np.ones_like(f),
        )

        p_base, _ = sim_base(pts)
        p_tf, _ = sim_tf(pts)

        peak = float(max(p_base.max(), p_tf.max()))
        np.testing.assert_allclose(p_tf, p_base, atol=peak * 1e-5)

    def test_identity_tf_matches_no_tf_per_element(self, small_linear_transducer):
        """TF(f)=1 must give same result as no TF (per-element exc)."""
        n_el = small_linear_transducer.n_elements
        pulse = _make_excitation()
        exc_pe = np.tile(pulse[:, np.newaxis], (1, n_el))
        pts = np.array([[0.0, 0.0, 20.0]], dtype=np.float32)

        sim_base = _make_emission(small_linear_transducer, excitation=exc_pe)
        sim_tf = _make_emission(
            small_linear_transducer,
            excitation=exc_pe,
            transfer_function=lambda f: np.ones_like(f),
        )

        p_base, _ = sim_base(pts)
        p_tf, _ = sim_tf(pts)

        peak = float(max(p_base.max(), p_tf.max()))
        np.testing.assert_allclose(p_tf, p_base, atol=peak * 1e-5)

    def test_zero_tf_gives_zero_pressure(self, small_linear_transducer):
        """TF(f)=0 must suppress all output."""
        pulse = _make_excitation()
        pts = np.array([[0.0, 0.0, 20.0]], dtype=np.float32)

        sim = _make_emission(
            small_linear_transducer,
            excitation=pulse,
            transfer_function=lambda f: np.zeros_like(f),
        )
        p, _ = sim(pts)
        assert p.max() == 0.0

    def test_set_transfer_function(self, small_linear_transducer):
        """set('transfer_function', ...) updates TF and rejects non-callables."""
        sim = _make_emission(small_linear_transducer)
        sim.set("transfer_function", lambda f: np.ones_like(f))
        assert sim.transfer_function is not None
        sim.set("transfer_function", None)
        assert sim.transfer_function is None
        with pytest.raises(TypeError):
            sim.set("transfer_function", 42)


# ---------------------------------------------------------------------------
# spectral (closed-form H) ≡ temporal (sampled h → FFT), every mode
# ---------------------------------------------------------------------------


class TestSpectralTemporalParity:
    _PTS = np.array([[0, 0, 8], [2, 0, 15], [-3, 1, 22], [6, 0, 12]], np.float32)

    @pytest.mark.parametrize(
        "kw",
        [
            dict(monochromatic=True),
            dict(monochromatic=True, alpha0=0.5, fast_attenuation=False),
            dict(excitation="global", alpha0=0.7, freq_power=1.2),
            dict(excitation="per_element", alpha0=0.5, fast_attenuation=False),
        ],
        ids=["mono", "mono-att-el", "global-att", "per-element-att"],
    )
    @pytest.mark.parametrize("baffle", ["rigid", "soft"])
    def test_same_field(self, small_linear_transducer, kw, baffle):
        tx = small_linear_transducer
        tx.baffle = baffle
        kw = dict(kw)
        pulse = _make_excitation(fs=100e6)
        if kw.get("excitation") == "global":
            kw["excitation"] = pulse
        elif kw.get("excitation") == "per_element":
            kw["excitation"] = pulse[:, None] * np.arange(1, tx.n_elements + 1)
        p_s, _ = _make_emission(tx, method="spectral", fs=100e6, **kw)(self._PTS)
        p_t, _ = _make_emission(tx, method="fst", fs=100e6, **kw)(self._PTS)
        tx.baffle = "rigid"
        n = min(len(p_s), len(p_t))
        np.testing.assert_allclose(p_s[:n], p_t[:n], atol=2e-2 * np.abs(p_t).max())

    def test_soft_baffle_lowers_off_axis_field(self, small_linear_transducer):
        tx = small_linear_transducer
        pt = np.array([[12.0, 0, 8.0]], np.float32)  # cosθ ≈ 0.55
        rigid, _ = _make_emission(tx, monochromatic=True)(pt)
        tx.baffle = "soft"
        soft, _ = _make_emission(tx, monochromatic=True)(pt)
        tx.baffle = "rigid"
        assert 0.45 < soft[0] / rigid[0] < 0.65

    def test_transfer_function_applies_in_monochromatic(self, small_linear_transducer):
        pts = self._PTS
        p1, _ = _make_emission(small_linear_transducer, monochromatic=True)(pts)
        p2, _ = _make_emission(
            small_linear_transducer,
            monochromatic=True,
            transfer_function=lambda f: 2.0 * np.ones_like(f),
        )(pts)
        np.testing.assert_allclose(p2, 2.0 * p1, rtol=1e-6)

    @pytest.mark.parametrize(
        "kw, expected",
        [
            (dict(monochromatic=True), "spectral"),
            (dict(), "temporal"),
            (dict(excitation="global", alpha0=0.5), "temporal"),
            (dict(excitation="global", alpha0=0.5, fast_attenuation=False), "spectral"),
            (dict(excitation="per_element"), "spectral"),
        ],
    )
    def test_default_picks_faster_method(self, small_linear_transducer, kw, expected):
        """method=None: spectral for monochromatic / per-element, else temporal."""
        tx = small_linear_transducer
        kw = dict(kw)
        pulse = _make_excitation()
        if kw.get("excitation") == "global":
            kw["excitation"] = pulse
        elif kw.get("excitation") == "per_element":
            kw["excitation"] = np.tile(pulse[:, None], (1, tx.n_elements))
        sim = _make_emission(tx, **kw)
        sim(self._PTS)
        assert sim._last_method == expected

    @pytest.mark.parametrize("method", ["spectral", "temporal"])
    def test_far_field_is_signed_rayleigh(self, method):
        """On axis, far from a small piston: p(t) = ρ·A/(2πz)·v'(t − z/c) — Rayleigh,
        signed, in pascals (no dependence on fs)."""
        from esdiva.transducers import LinearArrayTransducer

        fs = 100e6
        tx = LinearArrayTransducer(
            n_elements=1, element_width_mm=0.3, element_height_mm=0.3, kerf_mm=0.0,
            no_sub_x=1, no_sub_y=1, frequency_Hz=5e6,
        )  # fmt: skip
        v = _make_excitation(fs=fs)
        v = v * np.hanning(v.size).astype(np.float32)
        z = np.array([30.0, 60.0])
        pts = np.column_stack([np.zeros(2), np.zeros(2), z]).astype(np.float32)
        p, co = _make_emission(tx, fs=fs, excitation=v, method=method)(pts)
        t = co["t0"] + np.arange(p.shape[0]) / fs
        w, tw = 2 * np.pi * 5e6, (v.size - 1) / fs  # v = sin(ωt)·hann(t/tw), exactly

        def dv(s):  # analytic v'(s), zero outside the pulse
            win, dwin = (
                0.5 - 0.5 * np.cos(2 * np.pi * s / tw),
                np.pi / tw * np.sin(2 * np.pi * s / tw),
            )
            return np.where((s >= 0) & (s <= tw), w * np.cos(w * s) * win
                            + np.sin(w * s) * dwin, 0.0)  # fmt: skip

        for i, zi in enumerate(z * 1e-3):
            ref = dv(t - zi / 1540.0) * (0.3e-3) ** 2 / (2 * np.pi * zi)  # ρ = 1
            assert np.corrcoef(p[:, i], ref)[0, 1] > 0.99  # positive: same polarity
            assert abs(np.abs(p[:, i]).max() / np.abs(ref).max() - 1) < 0.03  # pascals
        # Spherical spreading: doubling the range halves the peak.
        assert abs(np.abs(p[:, 1]).max() / np.abs(p[:, 0]).max() - 0.5) < 0.02

    def test_unknown_method_raises(self, small_linear_transducer):
        with pytest.raises(ValueError, match="Unknown method"):
            _make_emission(small_linear_transducer, method="bogus")


# ---------------------------------------------------------------------------
# Impulse response wiring
# ---------------------------------------------------------------------------


class TestImpulseResponse:
    @pytest.mark.parametrize("ir_kind", ["delta", "burst"])
    def test_ir_equals_preconvolved_pulse(self, small_linear_transducer, ir_kind):
        """exc + tx.impulse_response must equal driving the full pulse exc ⊛ ir."""
        exc = _make_excitation()
        ir = np.zeros(32, np.float32)
        ir[0] = 1.0
        if ir_kind == "burst":
            ir = exc * np.hanning(exc.size).astype(np.float32)
        pts = np.array([[0.0, 0.0, 20.0]], dtype=np.float32)
        tx = small_linear_transducer
        tx.impulse_response = ir
        p_ir, _ = _make_emission(tx, excitation=exc)(pts)
        tx.impulse_response = None
        full = np.convolve(exc, ir).astype(np.float32)
        p_full, _ = _make_emission(tx, excitation=full)(pts)
        np.testing.assert_allclose(p_ir, p_full, atol=1e-6 * np.abs(p_full).max())
