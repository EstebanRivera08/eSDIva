"""Fixtures specific to transducer tests."""

import pytest

from pyfield.transducers import (
    ConcaveCircularTransducer,
    ConvexArrayTransducer,
    ConvexCircularTransducer,
    FlatCircularTransducer,
    LinearArrayTransducer,
    MatrixArrayTransducer,
)


@pytest.fixture
def linear_4elem():
    """4-element linear array with minimal subdivisions."""
    return LinearArrayTransducer(
        n_elements=4,
        element_width_mm=0.25,
        element_height_mm=5.0,
        kerf_mm=0.05,
        no_sub_x=1,
        no_sub_y=2,
        frequency_Hz=5e6,
    )


@pytest.fixture
def linear_8elem():
    """8-element linear array for symmetry tests."""
    return LinearArrayTransducer(
        n_elements=8,
        element_width_mm=0.3,
        element_height_mm=10.0,
        kerf_mm=0.05,
        no_sub_x=2,
        no_sub_y=3,
        frequency_Hz=3e6,
    )


@pytest.fixture
def matrix_3x3():
    """Small 3x3 matrix array."""
    return MatrixArrayTransducer(
        n_elements_x=3,
        n_elements_y=3,
        element_width_mm=0.3,
        element_height_mm=0.3,
        kerf_x_mm=0.02,
        kerf_y_mm=0.02,
        no_sub_x=1,
        no_sub_y=1,
        frequency_Hz=5e6,
    )


@pytest.fixture
def convex_6elem():
    """6-element convex array with typical ROC."""
    return ConvexArrayTransducer(
        n_elements=6,
        element_width_mm=0.3,
        element_height_mm=5.0,
        kerf_mm=0.05,
        radius_of_curvature_mm=50.0,
        no_sub_x=1,
        no_sub_y=2,
        frequency_Hz=3e6,
    )


@pytest.fixture
def flat_circular():
    """Small flat circular transducer."""
    return FlatCircularTransducer(
        diameter_mm=10.0,
        no_sub_diameter=6,
        refine_factor=2,
        frequency_Hz=1e6,
    )


@pytest.fixture
def concave_circular():
    """Small concave (bowl) circular transducer."""
    return ConcaveCircularTransducer(
        diameter_mm=20.0,
        focus_mm=30.0,
        no_sub_diameter=6,
        frequency_Hz=0.5e6,
    )


@pytest.fixture
def convex_circular():
    """Small convex (dome) circular transducer."""
    return ConvexCircularTransducer(
        diameter_mm=20.0,
        focus_mm=30.0,
        no_sub_diameter=6,
        frequency_Hz=0.5e6,
    )
