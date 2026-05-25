"""Tests for sir_derivatives — compute_d2h, compute_dh, per-element variants."""

import warnings

import numpy as np
import pytest

from pyfield.transducers import LinearArrayTransducer
from pyfield.utilities.helper_functions import compute_sub_elem_attributes, compute_time_grid


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


@pytest.fixture
def sub_elem(simple_tx):
    """Unpacked sub-element attributes for simple_tx."""
    return compute_sub_elem_attributes(simple_tx)


# ---------------------------------------------------------------------------
# sub_elem_attributes return now includes sub_el_idx_arr (8-tuple)
# ---------------------------------------------------------------------------


class TestSubElemAttributes:
    def test_sub_elem_returns_8_tuple(self, simple_tx):
        result = compute_sub_elem_attributes(simple_tx)
        assert len(result) == 8, "Expected 8-element return tuple."

    def test_sub_el_idx_arr_shape_and_dtype(self, simple_tx):
        *_, sub_el_idx_arr = compute_sub_elem_attributes(simple_tx)
        centers, *_, M, _, _, _, sub_el_idx_arr = compute_sub_elem_attributes(simple_tx)
        assert sub_el_idx_arr.dtype == np.int32
        assert sub_el_idx_arr.ndim == 1
        assert sub_el_idx_arr.shape[0] == M

    def test_sub_el_idx_arr_values_in_range(self, simple_tx):
        centers, apod, delays, M, _, wx, wy, sub_el_idx_arr = compute_sub_elem_attributes(simple_tx)
        assert sub_el_idx_arr.min() == 0
        assert sub_el_idx_arr.max() == simple_tx.n_elements - 1

    def test_sub_el_idx_arr_consistent_with_sub_el_idx(self, simple_tx):
        """sub_el_idx_arr must match the transducer's sub_el_idx list."""
        _, _, _, _, _, _, _, sub_el_idx_arr = compute_sub_elem_attributes(simple_tx)
        expected = np.array(simple_tx.sub_el_idx, dtype=np.int32)
        np.testing.assert_array_equal(sub_el_idx_arr, expected)


# ---------------------------------------------------------------------------
# compute_d2h + 2 cumsums ≈ compute_h_sir (rtol=1e-4, float32)
# ---------------------------------------------------------------------------


def _build_time_grid(tx, points):
    """Helper: build time grid for given transducer + field points."""
    centers, apod, delays, M, _, wx_arr, wy_arr, sub_el_idx = compute_sub_elem_attributes(tx)
    c = 1540.0
    fs = 200e6
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        time_grid, t0, dt, T = compute_time_grid(
            points.shape[0], M, points, centers,
            float(wx_arr.max()), float(wy_arr.max()),
            c, fs, tx.delays, verbose=False,
        )
    return centers, apod, delays, M, wx_arr, wy_arr, sub_el_idx, time_grid, t0, dt, T


class TestSirDerivComputeD2h:
    """compute_d2h + double cumsum must reproduce compute_h_sir (SDI path)."""

    def test_d2h_double_cumsum_matches_h_sir(self, simple_tx):
        from pyfield.h_sir.farfield_rect_patch import compute_h_sir
        from pyfield.h_sir.sir_derivatives import (
            compute_d2h,
            integrate_d2h_to_dh,
            integrate_dh_to_h,
        )

        points = np.array([[0.0, 0.0, 20.0e-3]], dtype=np.float32)  # 1 point, on axis
        (
            centers, apod, delays, M, wx_arr, wy_arr, _,
            time_grid, t0, dt, T,
        ) = _build_time_grid(simple_tx, points)
        c = 1540.0
        fs = 200e6

        # Reference (existing SDI kernel)
        h_ref, _ = compute_h_sir(
            points.shape[0], M, T, dt, time_grid, points,
            centers, wx_arr, wy_arr, 1.0 / c, fs, apod, delays,
            method_flag=1,
        )  # (P, T)

        # New: d2h → dh → h
        d2h = compute_d2h(points, centers, wx_arr, wy_arr, 1.0 / c, apod, delays, t0, T, fs, dt)
        dh = integrate_d2h_to_dh(d2h, dt)
        h_new = integrate_dh_to_h(dh, dt)

        # atol = 0.5% of peak: tolerates float32 tail-ramp artifacts (~0.015 vs -0.004)
        # while body matches at ~7e-5 relative
        peak_tol = 0.005 * float(np.abs(h_ref).max())
        np.testing.assert_allclose(h_new, h_ref, rtol=0.005, atol=peak_tol,
                                   err_msg="compute_d2h + 2 cumsums must match compute_h_sir.")

    def test_d2h_multi_point(self, simple_tx):
        """Verify shape and sanity for multiple field points."""
        from pyfield.h_sir.sir_derivatives import compute_d2h

        points = np.array([
            [0.0, 0.0, 15.0e-3],
            [1.0e-3, 0.0, 20.0e-3],
            [-1.0e-3, 0.0, 25.0e-3],
        ], dtype=np.float32)
        P = points.shape[0]
        centers, apod, delays, M, wx_arr, wy_arr, _, _, t0, dt, T = _build_time_grid(simple_tx, points)
        d2h = compute_d2h(points, centers, wx_arr, wy_arr, 1.0 / 1540.0, apod, delays, t0, T, 200e6, dt)

        assert d2h.shape == (P, T)
        assert d2h.dtype == np.float32
        assert np.any(d2h != 0), "d2h should have non-zero events."

    def test_d2h_patches_parallel_matches_points_parallel(self, simple_tx):
        """patches axis must produce same result as points axis."""
        from pyfield.h_sir.sir_derivatives import compute_d2h

        points = np.array([[0.0, 0.0, 20.0e-3]], dtype=np.float32)
        centers, apod, delays, M, wx_arr, wy_arr, _, _, t0, dt, T = _build_time_grid(simple_tx, points)

        d2h_p = compute_d2h(points, centers, wx_arr, wy_arr, 1.0 / 1540.0, apod, delays,
                             t0, T, 200e6, dt, parallel_axis="points")
        d2h_m = compute_d2h(points, centers, wx_arr, wy_arr, 1.0 / 1540.0, apod, delays,
                             t0, T, 200e6, dt, parallel_axis="patches")

        np.testing.assert_allclose(d2h_p, d2h_m, rtol=1e-5, atol=1e-30,
                                   err_msg="points vs patches axis must agree.")


# ---------------------------------------------------------------------------
# Per-element grouping: sum over E == summed-all
# ---------------------------------------------------------------------------


class TestSirDerivPerElement:
    def test_sum_over_elements_matches_summed_all(self, simple_tx):
        from pyfield.h_sir.sir_derivatives import compute_d2h, compute_d2h_per_element

        points = np.array([[0.0, 0.0, 20.0e-3]], dtype=np.float32)
        centers, apod, delays, M, wx_arr, wy_arr, sub_el_idx, _, t0, dt, T = _build_time_grid(simple_tx, points)
        inv_c = 1.0 / 1540.0
        fs = 200e6

        d2h_all = compute_d2h(points, centers, wx_arr, wy_arr, inv_c, apod, delays, t0, T, fs, dt)
        d2h_per_e = compute_d2h_per_element(
            points, centers, wx_arr, wy_arr, inv_c, apod, delays, t0, T, fs, dt,
            sub_el_idx, simple_tx.n_elements,
        )

        # Sum over E (axis=1) should equal summed-all
        d2h_from_sum = d2h_per_e.sum(axis=1)  # (P, T)
        np.testing.assert_allclose(d2h_from_sum, d2h_all, rtol=1e-5, atol=1e-30,
                                   err_msg="sum over E of per-element d2h must equal summed d2h.")

    def test_per_element_shape(self, simple_tx):
        from pyfield.h_sir.sir_derivatives import compute_d2h_per_element

        points = np.array([[0.0, 0.0, 20.0e-3], [0.0, 0.0, 25.0e-3]], dtype=np.float32)
        P = points.shape[0]
        centers, apod, delays, M, wx_arr, wy_arr, sub_el_idx, _, t0, dt, T = _build_time_grid(simple_tx, points)

        d2h_pe = compute_d2h_per_element(
            points, centers, wx_arr, wy_arr, 1.0 / 1540.0, apod, delays, t0, T, 200e6, dt,
            sub_el_idx, simple_tx.n_elements,
        )
        assert d2h_pe.shape == (P, simple_tx.n_elements, T)
        assert d2h_pe.dtype == np.float32

    def test_compute_dh_per_element_sum_matches_compute_dh(self, simple_tx):
        from pyfield.h_sir.sir_derivatives import compute_dh, compute_dh_per_element

        points = np.array([[0.0, 0.0, 20.0e-3]], dtype=np.float32)
        centers, apod, delays, M, wx_arr, wy_arr, sub_el_idx, _, t0, dt, T = _build_time_grid(simple_tx, points)
        inv_c = 1.0 / 1540.0
        fs = 200e6

        dh_all = compute_dh(points, centers, wx_arr, wy_arr, inv_c, apod, delays, t0, T, fs, dt)
        dh_per_e = compute_dh_per_element(
            points, centers, wx_arr, wy_arr, inv_c, apod, delays, t0, T, fs, dt,
            sub_el_idx, simple_tx.n_elements,
        )
        # atol = 0.5% of peak: tolerates float32 DC-offset artifact in dh tail
        # (~2048 offset when element sums cancel differently); body matches at ~1e-7 relative
        peak_tol = 0.005 * float(np.abs(dh_all).max())
        np.testing.assert_allclose(
            dh_per_e.sum(axis=1), dh_all, rtol=0.005, atol=peak_tol,
            err_msg="sum of dh_per_element over E must equal dh all."
        )

    def test_compute_h_sir_patch_parallel(self, simple_tx):
        """Patch-parallel reference kernel must match points-parallel output."""
        from pyfield.h_sir.farfield_rect_patch import compute_h_sir
        from pyfield.h_sir.sir_derivatives import compute_h_sir_patch_parallel

        points = np.array([[0.0, 0.0, 20.0e-3]], dtype=np.float32)
        (
            centers, apod, delays, M, wx_arr, wy_arr, _,
            time_grid, t0, dt, T,
        ) = _build_time_grid(simple_tx, points)
        c = 1540.0
        fs = 200e6

        h_ref, _ = compute_h_sir(
            points.shape[0], M, T, dt, time_grid, points,
            centers, wx_arr, wy_arr, 1.0 / c, fs, apod, delays,
            method_flag=1,
        )
        h_mpar = compute_h_sir_patch_parallel(
            points, centers, wx_arr, wy_arr, 1.0 / c, apod, delays, t0, T, fs, dt
        )

        peak_tol = 0.005 * float(np.abs(h_ref).max())
        np.testing.assert_allclose(h_mpar, h_ref, rtol=0.005, atol=peak_tol,
                                   err_msg="patch-parallel h_sir must match reference kernel.")
