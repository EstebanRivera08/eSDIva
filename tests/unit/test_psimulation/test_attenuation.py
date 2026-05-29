"""Tests for psimulation.attenuation — causal power-law TF and distance helpers."""

import numpy as np
import pytest

from pyfield.attenuation import (
    causal_attenuation_tf,
    compute_attenuation_distances,
    compute_reception_distances,
    convert_alpha0_to_nepers,
    reduce_patch_distances_to_element,
)


# ---------------------------------------------------------------------------
# convert_alpha0_to_nepers
# ---------------------------------------------------------------------------


class TestConvertAlpha0:
    def test_zero_input(self):
        assert convert_alpha0_to_nepers(0.0, 1.0) == pytest.approx(0.0)

    def test_known_value_y1(self):
        # 0.5 dB/(MHz·cm), y=1 → verify formula
        expected = 0.5 * 100.0 / (20.0 * np.log10(np.e) * 1e6)
        assert convert_alpha0_to_nepers(0.5, 1.0) == pytest.approx(expected, rel=1e-10)

    def test_known_value_y15(self):
        expected = 0.5 * 100.0 / (20.0 * np.log10(np.e) * (1e6**1.5))
        assert convert_alpha0_to_nepers(0.5, 1.5) == pytest.approx(expected, rel=1e-10)

    def test_result_positive(self):
        assert convert_alpha0_to_nepers(1.0, 1.2) > 0.0


# ---------------------------------------------------------------------------
# causal_attenuation_tf — identity paths
# ---------------------------------------------------------------------------


class TestCausalAttenuationTfIdentity:
    def test_alpha0_none_returns_ones(self):
        freqs = np.linspace(0, 10e6, 64)
        dist = np.array([0.01, 0.05, 0.10])
        H = causal_attenuation_tf(freqs, dist, alpha0_dB=None, y=1.0, f0_hz=5e6)
        assert H.shape == (3, 64)
        np.testing.assert_array_equal(H, np.ones_like(H))

    def test_alpha0_zero_returns_ones(self):
        freqs = np.linspace(0, 10e6, 64)
        dist = np.array([0.05])
        H = causal_attenuation_tf(freqs, dist, alpha0_dB=0, y=1.0, f0_hz=5e6)
        np.testing.assert_array_equal(H, np.ones_like(H))

    def test_zero_distance_no_attenuation(self):
        freqs = np.linspace(0, 10e6, 64)
        dist = np.array([0.0])
        H = causal_attenuation_tf(freqs, dist, alpha0_dB=0.5, y=1.0, f0_hz=5e6)
        np.testing.assert_allclose(np.abs(H), np.ones((1, 64)), atol=1e-12)

    def test_dc_is_one(self):
        """H(f=0, d) = 1 regardless of alpha0."""
        freqs = np.array([0.0, 1e6, 5e6])
        dist = np.array([0.05])
        H = causal_attenuation_tf(freqs, dist, alpha0_dB=0.5, y=1.0, f0_hz=5e6)
        assert abs(H[0, 0]) == pytest.approx(1.0, abs=1e-12)


# ---------------------------------------------------------------------------
# causal_attenuation_tf — general case (y != 1)
# ---------------------------------------------------------------------------


class TestCausalAttenuationTfGeneral:
    def test_amplitude_decay_monotone_with_distance(self):
        freqs = np.array([5e6])
        dist = np.array([0.01, 0.05, 0.10])
        H = causal_attenuation_tf(freqs, dist, alpha0_dB=0.5, y=1.2, f0_hz=5e6)
        amp = np.abs(H[:, 0])
        assert amp[0] > amp[1] > amp[2]

    def test_amplitude_le_one(self):
        freqs = np.linspace(0, 10e6, 128)
        dist = np.linspace(0.01, 0.10, 5)
        H = causal_attenuation_tf(freqs, dist, alpha0_dB=1.0, y=1.2, f0_hz=5e6)
        assert np.all(np.abs(H) <= 1.0 + 1e-12)

    def test_output_shape_broadcast(self):
        freqs = np.linspace(0, 10e6, 64)
        dist = np.ones((3, 4)) * 0.05
        H = causal_attenuation_tf(freqs, dist, alpha0_dB=0.5, y=1.2, f0_hz=5e6)
        assert H.shape == (3, 4, 64)

    def test_output_dtype_complex128(self):
        freqs = np.linspace(0, 10e6, 32)
        dist = np.array([0.05])
        H = causal_attenuation_tf(freqs, dist, alpha0_dB=0.5, y=1.2, f0_hz=5e6)
        assert H.dtype == np.complex128

    def test_known_absorption_at_1mhz_1cm(self):
        """At f=1 MHz, d=1 cm, alpha0=0.5 dB/(MHz·cm), y=1: absorption ≈ 0.5 dB."""
        f0 = 1e6
        freqs = np.array([f0])
        dist = np.array([0.01])  # 1 cm
        H = causal_attenuation_tf(freqs, dist, alpha0_dB=0.5, y=1.0, f0_hz=f0)
        att_dB = -20.0 * np.log10(np.abs(H[0, 0]))
        assert att_dB == pytest.approx(0.5, rel=1e-4)


# ---------------------------------------------------------------------------
# causal_attenuation_tf — y = 1 special case continuity
# ---------------------------------------------------------------------------


class TestCausalAttenuationTfYOne:
    def test_y1_amplitude_matches_formula(self):
        """y=1 absorption: |H| = exp(-alpha0 * |f| * d)."""
        f0 = 5e6
        freqs = np.array([f0])
        dist = np.array([0.05])
        alpha0_dB = 0.5
        H = causal_attenuation_tf(freqs, dist, alpha0_dB=alpha0_dB, y=1.0, f0_hz=f0)
        alpha0_nep = convert_alpha0_to_nepers(alpha0_dB, 1.0)
        expected_amp = np.exp(-alpha0_nep * f0 * 0.05)
        assert np.abs(H[0, 0]) == pytest.approx(expected_amp, rel=1e-8)


# ---------------------------------------------------------------------------
# compute_attenuation_distances
# ---------------------------------------------------------------------------


class TestComputeAttenuationDistances:
    @pytest.fixture
    def field_points(self):
        return np.array([[0.0, 0.0, 0.03], [0.01, 0.0, 0.04]], dtype=np.float64)

    @pytest.fixture
    def tx_center(self):
        return np.array([0.0, 0.0, 0.0], dtype=np.float64)

    @pytest.fixture
    def patch_centers(self):
        return np.array([[0.001, 0.0, 0.0], [-0.001, 0.0, 0.0]], dtype=np.float64)

    def test_per_point_shape(self, field_points, tx_center):
        d = compute_attenuation_distances(field_points, tx_center, mode="per_point")
        assert d.shape == (2,)

    def test_per_point_known_value(self, tx_center):
        pts = np.array([[0.0, 0.0, 0.03]])
        d = compute_attenuation_distances(pts, tx_center, mode="per_point")
        assert d[0] == pytest.approx(0.03, rel=1e-10)

    def test_per_patch_shape(self, field_points, tx_center, patch_centers):
        d = compute_attenuation_distances(
            field_points, tx_center, patch_centers_m=patch_centers, mode="per_patch"
        )
        assert d.shape == (2, 2)

    def test_per_patch_requires_patch_centers(self, field_points, tx_center):
        with pytest.raises(ValueError, match="patch_centers_m required"):
            compute_attenuation_distances(field_points, tx_center, mode="per_patch")

    def test_invalid_mode_raises(self, field_points, tx_center):
        with pytest.raises(ValueError, match="mode must be"):
            compute_attenuation_distances(field_points, tx_center, mode="bad_mode")


# ---------------------------------------------------------------------------
# reduce_patch_distances_to_element
# ---------------------------------------------------------------------------


class TestReducePatchDistancesToElement:
    def test_mean_reduction(self):
        # 2 points, 4 patches, 2 elements (patches 0,1 → elem 0; patches 2,3 → elem 1)
        dist_pm = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
        sub_el_idx = np.array([0, 0, 1, 1], dtype=np.int32)
        result = reduce_patch_distances_to_element(
            dist_pm, sub_el_idx, n_elements=2, reduce="mean"
        )
        assert result.shape == (2, 2)
        np.testing.assert_allclose(result[0], [1.5, 3.5])
        np.testing.assert_allclose(result[1], [5.5, 7.5])

    def test_min_reduction(self):
        dist_pm = np.array([[1.0, 3.0], [5.0, 7.0]])
        sub_el_idx = np.array([0, 0], dtype=np.int32)
        result = reduce_patch_distances_to_element(
            dist_pm, sub_el_idx, n_elements=1, reduce="min"
        )
        np.testing.assert_allclose(result[:, 0], [1.0, 5.0])

    def test_max_reduction(self):
        dist_pm = np.array([[1.0, 3.0]])
        sub_el_idx = np.array([0, 0], dtype=np.int32)
        result = reduce_patch_distances_to_element(
            dist_pm, sub_el_idx, n_elements=1, reduce="max"
        )
        assert result[0, 0] == pytest.approx(3.0)

    def test_invalid_reduce_raises(self):
        with pytest.raises(ValueError, match="reduce must be"):
            reduce_patch_distances_to_element(
                np.ones((2, 2)),
                np.array([0, 1], dtype=np.int32),
                n_elements=2,
                reduce="median",
            )


# ---------------------------------------------------------------------------
# compute_reception_distances
# ---------------------------------------------------------------------------


class TestComputeReceptionDistances:
    def test_shape(self):
        scatterers = np.array([[0.0, 0.0, 0.05], [0.01, 0.0, 0.04]])
        tx_center = np.array([0.0, 0.0, 0.0])
        rx_centers = np.array([[0.002, 0.0, 0.0], [-0.002, 0.0, 0.0], [0.0, 0.0, 0.0]])
        d = compute_reception_distances(scatterers, tx_center, rx_centers)
        assert d.shape == (2, 3)

    def test_round_trip_positive(self):
        scatterers = np.array([[0.0, 0.0, 0.05]])
        tx_center = np.array([0.0, 0.0, 0.0])
        rx_centers = np.array([[0.0, 0.0, 0.0]])
        d = compute_reception_distances(scatterers, tx_center, rx_centers)
        assert d[0, 0] == pytest.approx(0.10, rel=1e-10)

    def test_round_trip_greater_than_one_way(self):
        scatterers = np.array([[0.0, 0.0, 0.05]])
        tx_center = np.array([0.0, 0.0, 0.0])
        rx_centers = np.array([[0.01, 0.0, 0.0]])
        d = compute_reception_distances(scatterers, tx_center, rx_centers)
        one_way = np.linalg.norm(scatterers[0] - tx_center)
        assert d[0, 0] > one_way
