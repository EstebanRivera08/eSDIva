"""Global test fixtures shared by all test modules."""

import warnings

import numpy as np
import pytest


@pytest.fixture
def rng():
    """Seeded random number generator for reproducible tests."""
    return np.random.default_rng(42)


@pytest.fixture
def small_linear_transducer():
    """A small 4-element LinearArrayTransducer for fast tests."""
    from sondi.transducers import LinearArrayTransducer

    tx = LinearArrayTransducer(
        n_elements=4,
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
def small_field_grid():
    """Minimal field-point dict for quick simulation tests."""
    return {
        "x_extent": [-1, 1],
        "y_extent": [0, 0],
        "z_extent": [5, 25],
        "dx": 1.0,
        "dy": 0,
        "dz": 5.0,
    }
