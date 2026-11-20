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


def test_beamformer_axes_accepted():
    f = Field(np.zeros((9, 9, 9)), {"x_mm": AXIS, "y_mm": AXIS, "z_mm": AXIS})
    assert np.array_equal(f.coords["x"], AXIS)


def test_missing_axis_raises():
    with pytest.raises(ValueError, match="missing 'z'"):
        Field(np.zeros((9, 9, 9)), {"x": AXIS, "y": AXIS})
