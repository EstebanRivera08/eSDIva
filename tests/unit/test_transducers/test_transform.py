"""Rigid-body motion of transducer geometry via ``TransducerBase.transform``."""

import numpy as np
import pytest

from sondi.emission import Emission


def _homogeneous(R, t_mm):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t_mm
    return T


def _rot_y(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


class TestTransformGeometry:
    def test_moves_all_computed_geometry(self, linear_4elem):
        tx = linear_4elem
        R = _rot_y(np.deg2rad(30.0))
        t_mm = np.array([1.0, -2.0, 5.0])
        ref_centers = tx.element_centers.copy()
        ref_quads = [q.copy() for q in tx.sub_quad_verts]
        ref_frames = {k: np.array(v).copy() for k, v in tx.sub_patch_frames.items()}

        tx.transform(_homogeneous(R, t_mm))
        t_m = t_mm * 1e-3

        np.testing.assert_allclose(
            tx.element_centers, ref_centers @ R.T + t_m, atol=1e-12
        )
        for q_new, q_old in zip(tx.sub_quad_verts, ref_quads):
            np.testing.assert_allclose(q_new, q_old @ R.T + t_m, atol=1e-12)
        frames = tx.sub_patch_frames
        np.testing.assert_allclose(
            frames["centers"], ref_frames["centers"] @ R.T + t_m, atol=1e-12
        )
        for key in ("normals", "tangents_u", "tangents_v"):
            np.testing.assert_allclose(frames[key], ref_frames[key] @ R.T, atol=1e-7)
        # Patch widths are rotation-invariant.
        np.testing.assert_allclose(frames["wu"], ref_frames["wu"], atol=0)
        np.testing.assert_allclose(frames["wv"], ref_frames["wv"], atol=0)

    def test_curved_transducer_frames_follow(self, concave_circular):
        tx = concave_circular
        ref_normals = np.array(tx.sub_patch_frames["normals"]).copy()
        R = _rot_y(np.deg2rad(90.0))
        tx.transform(_homogeneous(R, [0.0, 0.0, 0.0]))
        np.testing.assert_allclose(
            tx.sub_patch_frames["normals"], ref_normals @ R.T, atol=1e-7
        )
        # Vertex list and frames stay consistent (same corners feed both).
        np.testing.assert_allclose(
            np.mean(tx.sub_quad_verts[0], axis=0),
            tx.sub_patch_frames["centers"][0],
            atol=1e-9,
        )

    def test_rejects_scaling_and_bad_shape(self, linear_4elem):
        with pytest.raises(ValueError):
            linear_4elem.transform(np.diag([2.0, 2.0, 2.0, 1.0]))
        with pytest.raises(ValueError):
            linear_4elem.transform(np.eye(3))


class TestTransformPhysics:
    def test_field_moves_with_aperture(self, linear_4elem):
        """CW amplitude at p from the canonical pose equals the amplitude at
        T(p) from the transformed aperture — the field is rigidly carried
        along with the transducer."""
        point_mm = np.array([[1.5, 0.0, 20.0]])
        ref_tx = linear_4elem.copy()
        p_ref, _ = Emission(ref_tx, monochromatic=True, verbose=False)(point_mm)

        R = _rot_y(np.deg2rad(25.0))
        t_mm = np.array([2.0, 1.0, -3.0])
        moved_tx = linear_4elem
        moved_tx.transform(_homogeneous(R, t_mm))
        point_moved_mm = point_mm @ R.T + t_mm
        p_moved, _ = Emission(moved_tx, monochromatic=True, verbose=False)(
            point_moved_mm
        )

        np.testing.assert_allclose(p_moved, p_ref, rtol=1e-3)
