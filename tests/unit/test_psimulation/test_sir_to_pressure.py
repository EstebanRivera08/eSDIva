"""Tests for sir_to_pressure — attenuation wiring (Batch 2)."""


import numpy as np

from pyfield.psimulation.sir_to_pressure import (
    from_sir_to_monochromatic_pressure,
    from_sir_to_pressure,
)


def _make_synthetic_sir(P=5, T=512):
    """Return a simple synthetic (T, P) h_sir array."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((T, P)).astype(np.float32)


def _make_excitation(fs=200e6, fc=5e6, n_cycles=2):
    t = np.arange(0, n_cycles / fc, 1.0 / fs)
    return np.sin(2.0 * np.pi * fc * t).astype(np.float32)


# ---------------------------------------------------------------------------
# from_sir_to_pressure — alpha0=None identity
# ---------------------------------------------------------------------------


class TestFromSirToPressureIdentity:
    def test_alpha0_none_identical_to_no_alpha0(self):
        """alpha0=None must produce bit-identical output to calling without alpha0."""
        h = _make_synthetic_sir(P=4, T=256)
        exc = _make_excitation()
        fs = 200e6

        out_old = from_sir_to_pressure(h, None, None, None, fs, excitation=exc)
        out_new = from_sir_to_pressure(h, None, None, None, fs, excitation=exc, alpha0=None)

        np.testing.assert_array_equal(
            out_old, out_new,
            err_msg="alpha0=None must yield bit-identical output.",
        )

    def test_no_excitation_no_attenuation_passthrough(self):
        """Without excitation and alpha0=None, output is h_sir itself."""
        h = _make_synthetic_sir(P=3, T=128)
        out = from_sir_to_pressure(h, None, None, None, 200e6)
        np.testing.assert_array_equal(out, h)


# ---------------------------------------------------------------------------
# from_sir_to_pressure — attenuation decreases with distance
# ---------------------------------------------------------------------------


class TestFromSirToPressureAttenuation:
    def test_amplitude_decreases_with_distance(self):
        """alpha0=0.5 must produce lower amplitude at larger distances."""
        P = 5
        T = 512
        h = _make_synthetic_sir(P=P, T=T)
        exc = _make_excitation()
        fs = 200e6

        # Distances: 1 cm to 5 cm (increasing)
        distances_m = np.linspace(0.01, 0.05, P)

        out_att = from_sir_to_pressure(
            h, None, None, None, fs,
            excitation=exc,
            alpha0=0.5,
            freq_power=1.0,
            f0_hz=5e6,
            distances_m=distances_m,
        )

        out_no_att = from_sir_to_pressure(
            h, None, None, None, fs, excitation=exc, alpha0=None
        )

        # With attenuation, RMS over time axis should be ≤ without attenuation
        # for every field point
        rms_att = np.sqrt(np.mean(out_att**2, axis=0))   # (P,)
        rms_no_att = np.sqrt(np.mean(out_no_att**2, axis=0))

        assert np.all(rms_att <= rms_no_att + 1e-10), (
            "Attenuation must reduce or preserve amplitude at all field points."
        )

    def test_more_distant_points_attenuated_more(self):
        """Larger distance → more attenuation."""
        P = 3
        T = 512
        # Identical SIR for all points so only distance differs
        rng = np.random.default_rng(0)
        sir_col = rng.standard_normal(T).astype(np.float32)
        h = np.tile(sir_col[:, np.newaxis], (1, P))
        exc = _make_excitation()
        fs = 200e6

        distances_m = np.array([0.01, 0.03, 0.06])

        out_att = from_sir_to_pressure(
            h, None, None, None, fs,
            excitation=exc,
            alpha0=0.5,
            freq_power=1.0,
            f0_hz=5e6,
            distances_m=distances_m,
        )

        rms = np.sqrt(np.mean(out_att**2, axis=0))
        assert rms[0] > rms[1] > rms[2], (
            "Amplitude must decrease monotonically with distance."
        )


# ---------------------------------------------------------------------------
# from_sir_to_monochromatic_pressure — attenuation
# ---------------------------------------------------------------------------


class TestMonochromaticPressureAttenuation:
    def test_alpha0_none_identical_to_no_alpha0(self):
        h = _make_synthetic_sir(P=4, T=256)
        fc = 5e6
        fs = 200e6
        out_old = from_sir_to_monochromatic_pressure(h, None, None, None, fc, fs)
        out_new = from_sir_to_monochromatic_pressure(
            h, None, None, None, fc, fs, alpha0=None
        )
        np.testing.assert_array_equal(out_old, out_new)

    def test_amplitude_decreases_with_distance(self):
        P = 4
        T = 256
        rng = np.random.default_rng(7)
        sir_col = rng.standard_normal(T).astype(np.float32)
        h = np.tile(sir_col[:, np.newaxis], (1, P))
        fc = 5e6
        fs = 200e6
        distances_m = np.array([0.01, 0.02, 0.04, 0.08])

        out_att = from_sir_to_monochromatic_pressure(
            h, None, None, None, fc, fs,
            alpha0=0.5, freq_power=1.0, f0_hz=fc, distances_m=distances_m,
        )
        out_no = from_sir_to_monochromatic_pressure(
            h, None, None, None, fc, fs, alpha0=None
        )

        assert np.all(out_att <= out_no + 1e-12), (
            "Attenuation must reduce monochromatic amplitude."
        )
        assert out_att[0] > out_att[1] > out_att[2] > out_att[3], (
            "Amplitude must decrease monotonically with distance."
        )
