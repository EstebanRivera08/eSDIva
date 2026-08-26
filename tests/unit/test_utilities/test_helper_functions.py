"""Tests for esdiva.utilities.helper_functions."""

from esdiva.utilities.helper_functions import (
    create_spatial_grid_from_dict,
)


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
