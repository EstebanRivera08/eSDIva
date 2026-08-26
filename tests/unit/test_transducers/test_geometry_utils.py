"""Tests for sondi.transducers.geometry_utils."""

import numpy as np

from sondi.transducers.geometry_utils import (
    rotation_matrix_z_to_normal,
)


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
