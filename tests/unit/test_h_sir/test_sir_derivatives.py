"""Tests for sub-element attribute utilities and SIR index mapping."""

import warnings

import numpy as np
import pytest

from pyfield.transducers import LinearArrayTransducer
from pyfield.utilities.helper_functions import (
    compute_sub_elem_attributes,
)


@pytest.fixture
def simple_tx():
    """4-element linear array: small, fast, deterministic."""
    tx = LinearArrayTransducer(
        n_elements=4,
        element_width_mm=0.25,
        element_height_mm=5.0,
        kerf_mm=0.05,
        no_sub_x=1,
        no_sub_y=2,
        frequency_Hz=5e6,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        tx.compute_delays(focus_mm=[0, 0, 20])
    return tx


class TestSubElemAttributes:
    def test_sub_elem_returns_7_tuple(self, simple_tx):
        result = compute_sub_elem_attributes(simple_tx)
        assert len(result) == 7, "Expected 7-element return tuple."

    def test_sub_el_idx_arr_shape_and_dtype(self, simple_tx):
        centers, _, _, M, _, _, sub_el_idx_arr = compute_sub_elem_attributes(simple_tx)
        assert sub_el_idx_arr.dtype == np.int32
        assert sub_el_idx_arr.ndim == 1
        assert sub_el_idx_arr.shape[0] == M

    def test_sub_el_idx_arr_values_in_range(self, simple_tx):
        centers, apod, delays, M, wx, wy, sub_el_idx_arr = compute_sub_elem_attributes(
            simple_tx
        )
        assert sub_el_idx_arr.min() == 0
        assert sub_el_idx_arr.max() == simple_tx.n_elements - 1

    def test_sub_el_idx_arr_consistent_with_sub_el_idx(self, simple_tx):
        """sub_el_idx_arr must match the transducer's sub_el_idx list."""
        _, _, _, _, _, _, sub_el_idx_arr = compute_sub_elem_attributes(simple_tx)
        expected = np.array(simple_tx.sub_el_idx, dtype=np.int32)
        np.testing.assert_array_equal(sub_el_idx_arr, expected)
