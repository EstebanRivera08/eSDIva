"""Tests for compute_delays — plane-wave steering via angle_steering_deg."""

import numpy as np
import pytest

from pyfield.transducers import LinearArrayTransducer, MatrixArrayTransducer


@pytest.fixture
def linear_8elem():
    """8-element linear array, elements centered on x-axis."""
    return LinearArrayTransducer(
        n_elements=8,
        element_width_mm=0.3,
        element_height_mm=10.0,
        kerf_mm=0.05,
        no_sub_x=1,
        no_sub_y=1,
        frequency_Hz=5e6,
    )


@pytest.fixture
def matrix_3x3():
    """3×3 matrix array for 2-D steering tests."""
    return MatrixArrayTransducer(
        n_elements_x=3,
        n_elements_y=3,
        element_width_mm=0.3,
        element_height_mm=0.3,
        kerf_x_mm=0.05,
        kerf_y_mm=0.05,
        no_sub_x=1,
        no_sub_y=1,
        frequency_Hz=5e6,
    )


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


class TestSteeringErrors:
    def test_both_focus_and_angle_raises(self, linear_8elem):
        with pytest.raises(ValueError, match="not both"):
            linear_8elem.compute_delays(focus_mm=[0, 0, 30], angle_steering_deg=15.0)

    def test_neither_raises(self, linear_8elem):
        with pytest.raises(ValueError, match="must be provided"):
            linear_8elem.compute_delays()

    def test_angle_exceeds_physical_limit_raises(self, linear_8elem):
        with pytest.raises(ValueError, match="physical limit"):
            linear_8elem.compute_delays(angle_steering_deg=(60.0, 60.0))

    def test_mono_element_returns_zeros_with_warning(self):
        from pyfield.transducers import FlatCircularTransducer

        tx = FlatCircularTransducer(
            diameter_mm=10.0, no_sub_diameter=4, frequency_Hz=1e6
        )
        with pytest.warns(UserWarning):
            d = tx.compute_delays(angle_steering_deg=0.0)
        np.testing.assert_array_equal(d, np.zeros(1))


# ---------------------------------------------------------------------------
# Zero steering — flat wavefront
# ---------------------------------------------------------------------------


class TestZeroSteering:
    def test_zero_angle_all_delays_zero(self, linear_8elem):
        d = linear_8elem.compute_delays(angle_steering_deg=0.0)
        np.testing.assert_allclose(d, 0.0, atol=1e-15)

    def test_zero_angle_tuple_all_delays_zero(self, matrix_3x3):
        d = matrix_3x3.compute_delays(angle_steering_deg=(0.0, 0.0))
        np.testing.assert_allclose(d, 0.0, atol=1e-15)

    def test_delays_non_negative(self, linear_8elem):
        d = linear_8elem.compute_delays(angle_steering_deg=20.0)
        assert np.all(d >= -1e-16)

    def test_minimum_delay_is_zero(self, linear_8elem):
        d = linear_8elem.compute_delays(angle_steering_deg=20.0)
        assert d.min() == pytest.approx(0.0, abs=1e-15)


# ---------------------------------------------------------------------------
# xz-plane steering — lateral array
# ---------------------------------------------------------------------------


class TestXzPlaneSteering:
    def test_delays_monotone_for_positive_angle(self, linear_8elem):
        """Positive θ_x: rightmost element fires first (max x → min delay)."""
        d = linear_8elem.compute_delays(angle_steering_deg=20.0)
        # Element centers sorted by x (linear array)
        x_centers = linear_8elem.element_centers[:, 0]
        sort_idx = np.argsort(x_centers)
        d_sorted = d[sort_idx]
        # Delays must be monotonically non-increasing as x increases
        assert np.all(np.diff(d_sorted) <= 1e-15)

    def test_delays_monotone_for_negative_angle(self, linear_8elem):
        """Negative θ_x: leftmost element fires first."""
        d = linear_8elem.compute_delays(angle_steering_deg=-20.0)
        x_centers = linear_8elem.element_centers[:, 0]
        sort_idx = np.argsort(x_centers)
        d_sorted = d[sort_idx]
        # Delays must be monotonically non-decreasing as x increases
        assert np.all(np.diff(d_sorted) >= -1e-15)

    def test_delays_proportional_to_x_position(self, linear_8elem):
        """For xz-plane only: delays ∝ (x_max - x_i) * sin(θ) / c."""
        theta_deg = 15.0
        c = linear_8elem.speed_of_sound_mps
        d = linear_8elem.compute_delays(angle_steering_deg=theta_deg)
        x = linear_8elem.element_centers[:, 0]
        sin_theta = np.sin(np.deg2rad(theta_deg))
        expected = (x.max() - x) * sin_theta / c
        np.testing.assert_allclose(d, expected, rtol=1e-8)

    def test_opposite_angles_mirror_delays(self, linear_8elem):
        """delays(+θ) reversed == delays(-θ) for symmetric array."""
        d_pos = linear_8elem.compute_delays(angle_steering_deg=20.0)
        d_neg = linear_8elem.compute_delays(angle_steering_deg=-20.0)
        np.testing.assert_allclose(d_pos[::-1], d_neg, rtol=1e-8)

    def test_inline_false_does_not_store(self, linear_8elem):
        old_delays = linear_8elem.delays.copy()
        linear_8elem.compute_delays(angle_steering_deg=20.0, inline=False)
        np.testing.assert_array_equal(linear_8elem.delays, old_delays)

    def test_inline_true_stores_delays(self, linear_8elem):
        d = linear_8elem.compute_delays(angle_steering_deg=10.0, inline=True)
        np.testing.assert_array_equal(linear_8elem.delays, d)

    def test_scalar_and_tuple_equal_for_1d_steer(self, linear_8elem):
        """angle_steering_deg=θ and (θ, 0) must produce identical delays."""
        d_scalar = linear_8elem.compute_delays(angle_steering_deg=15.0)
        d_tuple = linear_8elem.compute_delays(angle_steering_deg=(15.0, 0.0))
        np.testing.assert_allclose(d_scalar, d_tuple, rtol=1e-12)

    def test_custom_speed_of_sound(self, linear_8elem):
        """Faster medium → smaller delays at same angle."""
        d_default = linear_8elem.compute_delays(angle_steering_deg=20.0)
        d_fast = linear_8elem.compute_delays(angle_steering_deg=20.0, c=3000.0)
        assert np.all(d_fast <= d_default + 1e-15)
        # Max delay scales by c ratio
        c_default = linear_8elem.speed_of_sound_mps
        np.testing.assert_allclose(
            d_fast.max() / d_default.max(), c_default / 3000.0, rtol=1e-8
        )


# ---------------------------------------------------------------------------
# 2-D steering (matrix array)
# ---------------------------------------------------------------------------


class TestTwoDSteering:
    def test_xy_steer_delays_shape(self, matrix_3x3):
        d = matrix_3x3.compute_delays(angle_steering_deg=(10.0, 15.0))
        assert d.shape == (9,)

    def test_xy_steer_min_zero(self, matrix_3x3):
        d = matrix_3x3.compute_delays(angle_steering_deg=(10.0, 15.0))
        assert d.min() == pytest.approx(0.0, abs=1e-15)

    def test_x_steer_only_y_independent(self, matrix_3x3):
        """x-only steer: elements with same x have equal delays regardless of y."""
        d = matrix_3x3.compute_delays(angle_steering_deg=(15.0, 0.0))
        x = matrix_3x3.element_centers[:, 0]
        y = matrix_3x3.element_centers[:, 1]
        # Group by unique x and check delays equal within each group
        for xi in np.unique(x):
            mask = np.abs(x - xi) < 1e-9
            delays_at_xi = d[mask]
            np.testing.assert_allclose(delays_at_xi, delays_at_xi[0], atol=1e-15)
