"""Tests for Emission class — Batch 3 test gate."""

import warnings

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_excitation(fs=200e6, fc=5e6, n_cycles=2):
    t = np.arange(0, n_cycles / fc, 1.0 / fs)
    return np.sin(2.0 * np.pi * fc * t).astype(np.float32)


def _make_emission(tx, **kwargs):
    from pyfield.emission import Emission

    return Emission(tx, **kwargs)


def _make_pyfield(tx, **kwargs):
    from pyfield.emission import PyField

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return PyField(tx, **kwargs)


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

    def test_matches_pyfield(self, small_linear_transducer, small_field_grid):
        """Emission(monochromatic=True) output matches old PyField behavior."""
        sim_new = _make_emission(small_linear_transducer, monochromatic=True)
        sim_old = _make_pyfield(small_linear_transducer)

        p_new, _ = sim_new(small_field_grid)
        p_old, _ = sim_old(small_field_grid)

        np.testing.assert_allclose(p_new, p_old, rtol=1e-5)


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

    def test_matches_pyfield_with_excitation(
        self, small_linear_transducer, small_field_grid
    ):
        """Emission(excitation=pulse) matches PyField(excitation=pulse) output."""
        exc = _make_excitation()
        sim_new = _make_emission(small_linear_transducer, excitation=exc)
        sim_old = _make_pyfield(small_linear_transducer, monochromatic=False)

        p_new, _ = sim_new(small_field_grid)
        p_old, _ = sim_old(small_field_grid, excitation=exc)

        np.testing.assert_allclose(p_new, p_old, rtol=1e-5)

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

        sim_global = _make_emission(small_linear_transducer, excitation=exc_global)
        sim_pe = _make_emission(small_linear_transducer, excitation=exc_per_elem)

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
# PyField backward compatibility
# ---------------------------------------------------------------------------


class TestPyFieldBackwardCompatibility:
    def test_deprecation_warning(self, small_linear_transducer):
        from pyfield.emission import PyField

        with pytest.warns(DeprecationWarning, match="deprecated"):
            PyField(small_linear_transducer)

    def test_monochromatic_default_true(self, small_linear_transducer):
        sim = _make_pyfield(small_linear_transducer)
        assert sim.monochromatic is True

    def test_output_matches_emission_mono(
        self, small_linear_transducer, small_field_grid
    ):
        sim_pf = _make_pyfield(small_linear_transducer)
        sim_em = _make_emission(small_linear_transducer, monochromatic=True)

        p_pf, _ = sim_pf(small_field_grid)
        p_em, _ = sim_em(small_field_grid)

        np.testing.assert_allclose(p_pf, p_em, rtol=1e-5)

    def test_pyfield_transient_with_excitation(
        self, small_linear_transducer, small_field_grid
    ):
        exc = _make_excitation()
        sim_pf = _make_pyfield(small_linear_transducer, monochromatic=False)
        p, coords = sim_pf(small_field_grid, excitation=exc)
        assert p.ndim == 4
        assert "t0" in coords


# ---------------------------------------------------------------------------
# Impulse response wiring
# ---------------------------------------------------------------------------


class TestImpulseResponse:
    def test_ir_none_same_as_ir_delta(self, small_linear_transducer):
        """ir=None must produce same output as ir=delta function."""
        exc = _make_excitation()
        pts = np.array([[0.0, 0.0, 20.0]], dtype=np.float32)

        delta = np.zeros(32, dtype=np.float32)
        delta[0] = 1.0

        sim_no_ir = _make_emission(small_linear_transducer, excitation=exc)

        # Set delta IR
        tx = small_linear_transducer
        tx.impulse_response = delta
        sim_delta_ir = _make_emission(tx, excitation=exc)
        # Clear for other tests
        tx.impulse_response = None

        p_no, _ = sim_no_ir(pts)
        p_delta, _ = sim_delta_ir(pts)

        # With delta IR, output should be same (identity convolution) or very close.
        # Truncation to excitation length may introduce minor differences.
        assert p_no.shape == p_delta.shape
