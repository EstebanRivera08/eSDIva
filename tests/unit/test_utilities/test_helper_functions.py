"""Tests for esdiva.utilities.helper_functions."""

import time

from esdiva.utilities.helper_functions import (
    announce_eta,
    create_spatial_grid_from_dict,
    eta_progress,
)


class TestEta:
    def test_slow_first_unit_is_not_extrapolated(self, capsys):
        """A JIT-inflated first unit (0.4 s) must not project 100 fast units to 40 s."""
        for i in eta_progress(range(100), 100, label="bins"):
            if i == 0:
                time.sleep(0.4)
        assert capsys.readouterr().out == ""

    def test_long_run_announced_with_finish_time(self, capsys):
        """A representative unit × units above the threshold prints the ETA."""
        assert announce_eta(0.5, 100, "bins", n_done=2)
        assert "Estimated run time ~50.0 s (100 bins)" in capsys.readouterr().out
        assert not announce_eta(0.1, 100, "bins")  # 10 s: below threshold


class TestCreateSpatialGridFromDict:
    def test_output_shapes(self):
        grid_dict = {
            "x_extent": [-2, 2],
            "y_extent": [0, 0],
            "z_extent": [5, 15],
            "dx": 1.0,
            "dy": 0,
            "dz": 5.0,
        }
        x, y, z, pts = create_spatial_grid_from_dict(grid_dict)
        # x, y, z are 1D arrays; pts is (N, 3) array of grid points
        assert x.ndim == 1
        assert y.ndim == 1
        assert z.ndim == 1
        assert pts.ndim == 2
        assert pts.shape[1] == 3

    def test_grid_point_count(self):
        grid_dict = {
            "x_extent": [-1, 1],
            "y_extent": [0, 0],
            "z_extent": [5, 15],
            "dx": 1.0,
            "dy": 0,
            "dz": 5.0,
        }
        x, y, z, pts = create_spatial_grid_from_dict(grid_dict)
        expected_count = len(x) * len(y) * len(z)
        assert pts.shape[0] == expected_count
