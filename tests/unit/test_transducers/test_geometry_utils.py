"""Tests for pyfield.transducers.geometry_utils."""

import numpy as np
import pytest

from pyfield.transducers.geometry_utils import (
    compute_1d_element_centers,
    compute_distances_to_point,
    normalize_delays,
    rotation_matrix_z_to_normal,
)


class TestCompute1dElementCenters:
    def test_symmetry(self):
        """Centers should be symmetric around zero."""
        centers = compute_1d_element_centers(
            n_elements=8, element_size_m=0.25e-3, kerf_m=0.05e-3
        )
        assert abs(centers.sum()) < 1e-12

    def test_count(self):
        centers = compute_1d_element_centers(
            n_elements=4, element_size_m=0.3e-3, kerf_m=0.05e-3
        )
        assert centers.shape == (4,)

    def test_pitch_spacing(self):
        """Spacing between adjacent centers should equal pitch."""
        elem_size = 0.25e-3
        kerf = 0.05e-3
        centers = compute_1d_element_centers(
            n_elements=6, element_size_m=elem_size, kerf_m=kerf
        )
        diffs = np.diff(centers)
        expected_pitch = elem_size + kerf
        np.testing.assert_allclose(diffs, expected_pitch, atol=1e-15)

    def test_single_element_at_origin(self):
        centers = compute_1d_element_centers(
            n_elements=1, element_size_m=1e-3, kerf_m=0.0
        )
        assert centers.shape == (1,)
        assert centers[0] == pytest.approx(0.0)


class TestNormalizeDelays:
    def test_minimum_becomes_zero(self):
        delays = np.array([1.0, 2.0, 3.0])
        normed = normalize_delays(delays)
        assert normed.min() == pytest.approx(0.0)
        np.testing.assert_allclose(normed, [0.0, 1.0, 2.0])

    def test_already_zero_min(self):
        delays = np.array([0.0, 1.0, 2.0])
        normed = normalize_delays(delays)
        np.testing.assert_allclose(normed, delays)


class TestComputeDistancesToPoint:
    def test_known_distances(self):
        centers = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        focus = np.array([0.0, 0.0, 0.0])
        dists = compute_distances_to_point(centers, focus)
        np.testing.assert_allclose(dists, [0.0, 1.0])

    def test_3d_distance(self):
        centers = np.array([[1.0, 1.0, 1.0]])
        focus = np.array([0.0, 0.0, 0.0])
        dists = compute_distances_to_point(centers, focus)
        np.testing.assert_allclose(dists, [np.sqrt(3.0)])


class TestRotationMatrixZToNormal:
    def test_identity_for_z_axis(self):
        R = rotation_matrix_z_to_normal(np.array([0.0, 0.0, 1.0]))
        np.testing.assert_allclose(R, np.eye(3), atol=1e-10)

    def test_anti_parallel(self):
        R = rotation_matrix_z_to_normal(np.array([0.0, 0.0, -1.0]))
        result = R @ np.array([0.0, 0.0, 1.0])
        np.testing.assert_allclose(result, [0.0, 0.0, -1.0], atol=1e-10)

    def test_orthogonal(self):
        R = rotation_matrix_z_to_normal(np.array([1.0, 1.0, 0.0]))
        # R should be orthogonal: R @ R.T = I
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)

    def test_rotates_z_to_target(self):
        target = np.array([1.0, 0.0, 0.0])
        R = rotation_matrix_z_to_normal(target)
        result = R @ np.array([0.0, 0.0, 1.0])
        np.testing.assert_allclose(result, target, atol=1e-10)
