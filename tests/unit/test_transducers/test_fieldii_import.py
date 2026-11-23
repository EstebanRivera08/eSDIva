"""A Field II aperture export must rebuild the same patch mosaic as the native probe."""

from pathlib import Path

import numpy as np
import scipy.io

from esdiva.transducers import LinearArrayExample, from_fieldii_rect_data

MAT = (
    Path(__file__).parents[3]
    / "examples"
    / "applications"
    / "fieldii"
    / "domino_fieldii.mat"
)


def _sorted_patch_centres(t):
    c = np.array([q.mean(axis=0) for q in t.sub_quad_verts])
    return c[np.lexsort((c[:, 2], c[:, 1], c[:, 0]))]


def test_fieldii_rect_import_matches_native_linear_array():
    data = scipy.io.loadmat(MAT, simplify_cells=True)
    imported = from_fieldii_rect_data(
        data["rect"], frequency_hz=float(data["f0"]), elevation_focus_mm=8.0
    )
    native, ported = (
        _sorted_patch_centres(LinearArrayExample()),
        _sorted_patch_centres(imported),
    )
    assert native.shape == ported.shape
    np.testing.assert_allclose(ported, native, atol=0.2e-3)
