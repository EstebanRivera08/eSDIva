"""End-to-end integration tests: transducer -> simulation -> result check."""

import warnings

import numpy as np
import pytest

from pyfield.psimulation import PyField
from pyfield.transducers import LinearArrayTransducer


@pytest.fixture
def focused_transducer():
    """A small focused transducer for integration tests."""
    tx = LinearArrayTransducer(
        n_elements=15,
        element_width_mm=0.25,
        element_height_mm=5.0,
        kerf_mm=0.05,
        no_sub_x=1,
        no_sub_y=2,
        frequency_Hz=5e6,
    )
    tx.compute_delays(focus_mm=[0, 0, 20])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        tx.compute_apodization(focus_mm=[0, 0, 20], FoverD=1.0)
    return tx


@pytest.fixture
def xz_field_grid():
    """A 2D XZ plane grid for integration testing."""
    return {
        "x_extent": [-2, 2],
        "y_extent": [0, 0],
        "z_extent": [10, 30],
        "dx": 1.0,
        "dy": 0,
        "dz": 2.0,
    }


class TestMonochromaticSimulation:
    def test_runs_and_returns_correct_shapes(self, focused_transducer, xz_field_grid):
        """Full pipeline: transducer -> PyField -> monochromatic output."""
        sim = PyField(focused_transducer)
        p, coords = sim(xz_field_grid, method="auto")

        x, y, z = coords["x"], coords["y"], coords["z"]
        assert x.ndim == 1
        assert y.ndim == 1
        assert z.ndim == 1
        assert p.ndim == 3
        assert p.shape == (len(x), len(y), len(z))

    def test_pressure_is_nonzero(self, focused_transducer, xz_field_grid):
        """The pressure field should not be all zeros."""
        sim = PyField(focused_transducer)
        p, coords = sim(xz_field_grid, method="auto")
        assert np.max(np.abs(p)) > 0

    def test_pressure_nonzero_at_focus(self, focused_transducer):
        """Pressure near the focal point should be the global maximum."""
        grid = {
            "x_extent": [-3, 3],
            "y_extent": [0, 0],
            "z_extent": [10, 30],
            "dx": 0.5,
            "dy": 0,
            "dz": 1.0,
        }
        sim = PyField(focused_transducer)
        p, coords = sim(grid, method="auto")
        x, z = coords["x"], coords["z"]

        # Find the index of the maximum pressure
        max_idx = np.unravel_index(np.argmax(np.abs(p)), p.shape)
        x_max = x[max_idx[0]]
        z_max = z[max_idx[2]]

        # Focus was at [0, 0, 20] mm — max should be near x=0, z=20
        assert abs(x_max) <= 2.0, f"Max pressure x={x_max}, expected near 0"
        assert abs(z_max - 20) <= 5.0, f"Max pressure z={z_max}, expected near 20"

    def test_normalize_option(self, focused_transducer, xz_field_grid):
        """normalize=True should scale max to 1."""
        sim = PyField(focused_transducer)
        p, coords = sim(xz_field_grid, method="auto", normalize=True)
        assert np.max(np.abs(p)) == pytest.approx(1.0)
