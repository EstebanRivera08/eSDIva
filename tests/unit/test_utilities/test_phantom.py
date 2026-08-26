"""Tests for sondi.utilities.phantom.make_phantom."""

import numpy as np
import pytest

from sondi.utilities import make_phantom

BOX = {"x_extent": [-10.0, 10.0], "y_extent": [-1.0, 1.0], "z_extent": [10.0, 40.0]}


def test_positions_inside_box_and_reproducible():
    pos, amps = make_phantom(BOX, 500, seed=0)
    assert pos.shape == (500, 3) and amps.shape == (500,)
    assert pos[:, 0].min() >= -10.0 and pos[:, 0].max() <= 10.0
    assert pos[:, 1].min() >= -1.0 and pos[:, 1].max() <= 1.0
    assert pos[:, 2].min() >= 10.0 and pos[:, 2].max() <= 40.0
    pos2, amps2 = make_phantom(BOX, 500, seed=0)
    np.testing.assert_array_equal(pos, pos2)
    np.testing.assert_array_equal(amps, amps2)


def test_map_sampling_zeroes_anechoic_half():
    # Map = 0 for z in the front half of the box, 1 in the back half: every
    # scatterer in the anechoic half must return (near-)zero amplitude.
    m = np.ones((64, 64))
    m[:, :32] = 0.0
    pos, amps = make_phantom(BOX, 2000, echogenicity_map=m, seed=1)
    z_mid = 0.5 * (BOX["z_extent"][0] + BOX["z_extent"][1])
    # Stay 1 px away from the 0/1 edge where linear interpolation blends.
    dz_px = (BOX["z_extent"][1] - BOX["z_extent"][0]) / 63
    front = pos[:, 2] < z_mid - dz_px
    back = pos[:, 2] > z_mid + dz_px
    assert np.all(amps[front] == 0.0)
    assert np.std(amps[back]) > 0.5  # Gaussian draws survive where map = 1.


def test_extents_as_array_and_bad_map():
    pos, _ = make_phantom([[-1, 1], [0, 0], [5, 6]], 100, seed=2)
    assert np.all(pos[:, 1] == 0.0)  # degenerate elevation axis
    with pytest.raises(ValueError, match="2-D"):
        make_phantom(BOX, 10, echogenicity_map=np.ones(8))
