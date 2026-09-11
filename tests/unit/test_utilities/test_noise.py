"""Additive receiver noise for simulated RF."""

import numpy as np
import pytest

from esdiva.utilities import add_noise, rf_rms


@pytest.fixture
def rf(rng):
    return rng.normal(0.0, 2.0, (3, 8, 512)).astype(np.float32)


class TestAddNoise:
    @pytest.mark.parametrize("snr_db", [0.0, 10.0, 30.0])
    def test_delivers_the_requested_snr(self, rf, snr_db):
        """The residual's RMS must sit `snr_db` below the signal's, as amplitude."""
        noisy = add_noise(rf, snr_db, rng=0)
        measured = 20 * np.log10(rf_rms(rf) / np.std(noisy - rf))
        assert measured == pytest.approx(snr_db, abs=0.3)

    def test_reference_gives_a_common_floor(self, rf):
        """A shared `reference` puts two different signals on ONE noise level.

        This is what makes two sequences comparable: without it, a weaker
        acquisition would derive a smaller sigma from its own RMS and be handed
        a quieter receiver.
        """
        weak = (rf * 0.01).astype(np.float32)
        ref = rf_rms(rf)
        n_strong = np.std(add_noise(rf, 12, reference=ref, rng=1) - rf)
        n_weak = np.std(add_noise(weak, 12, reference=ref, rng=2) - weak)
        assert n_weak == pytest.approx(n_strong, rel=0.05)
        # Without the shared reference the weak signal gets 100x less noise.
        n_self = np.std(add_noise(weak, 12, rng=3) - weak)
        assert n_self < 0.05 * n_strong

    def test_reproducible_and_shape_preserving(self, rf):
        a = add_noise(rf, 15, rng=7)
        b = add_noise(rf, 15, rng=7)
        np.testing.assert_array_equal(a, b)
        assert a.shape == rf.shape
        assert a.dtype == np.float32
        assert not np.array_equal(a, add_noise(rf, 15, rng=8))

    def test_rf_rms_matches_numpy(self, rf):
        assert rf_rms(rf) == pytest.approx(float(np.sqrt(np.mean(rf.astype(np.float64) ** 2))))
