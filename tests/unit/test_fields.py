"""Field: canvas -> resample -> combine must preserve values and act point by point."""

import numpy as np
import pytest
from esdiva import Field

AXIS = np.linspace(-2, 2, 9)
GRID = {
    "x_extent": [-3, 5],
    "y_extent": [-2, 2],
    "z_extent": [-2, 2],
    "dx": 0.5,
    "dy": 0.5,
    "dz": 0.5,
}


@pytest.fixture
def box():
    return Field(np.ones((9, 9, 9)), {"x": AXIS, "y": AXIS, "z": AXIS})


def shift_x(mm):
    T = np.eye(4)
    T[0, 3] = mm
    return T


def test_canvas_sum_counts_overlap(box):
    canvas = Field.canvas(GRID)
    total = sum(f.resample(canvas) for f in (box, box.transform(shift_x(2.0))))
    # Grid-aligned shift: two boxes of 729 ones, overlapping where both reach.
    assert total.data.sum() == pytest.approx(2 * 729)
    assert total.data.max() == pytest.approx(2.0)


def test_pointwise_ops(box):
    assert np.allclose((box - box).data, 0.0)
    assert np.allclose((2 * box * box / 4).data, 0.5)


def test_numpy_interop_keeps_grid(box):
    shifted = box * -4.0
    root = np.sqrt(np.abs(shifted))  # ufuncs stay Fields on the same grid
    assert isinstance(root, Field) and root.coords is box.coords
    assert np.allclose(np.asarray(root), 2.0)
    assert np.allclose(np.maximum(box, shifted).data, 1.0)
    assert box.shape == (9, 9, 9) and box.ndim == 3


def test_time_reductions():
    t = np.arange(4.0)[:, None, None, None]
    f = Field(t * np.ones((4, 9, 9, 9)), {"x": AXIS, "y": AXIS, "z": AXIS})
    rms = np.sqrt((f**2).mean(axis=0))  # RMS over time of 0, 1, 2, 3
    assert isinstance(rms, Field) and rms.shape == (9, 9, 9)
    assert np.allclose(rms.data, np.sqrt(14 / 4))
    assert np.allclose(np.max(f, axis=0).data, 3.0)
    assert f.max() == 3.0
    with pytest.raises(ValueError, match="over time only"):
        f.mean(axis=1)


def test_beamformer_axes_accepted():
    f = Field(np.zeros((9, 9, 9)), {"x_mm": AXIS, "y_mm": AXIS, "z_mm": AXIS})
    assert np.array_equal(f.coords["x"], AXIS)


def test_missing_axis_raises():
    with pytest.raises(ValueError, match="missing 'z'"):
        Field(np.zeros((9, 9, 9)), {"x": AXIS, "y": AXIS})
