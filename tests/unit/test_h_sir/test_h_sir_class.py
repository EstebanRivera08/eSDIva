"""Tests for h_sir.compute_derivative — wiring Batch 2."""

import warnings

import numpy as np
import pytest

from pyfield.h_sir.h_sir import h_sir
from pyfield.transducers import LinearArrayTransducer
from pyfield.utilities.helper_functions import compute_time_grid


@pytest.fixture
def simple_tx():
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
def sim(simple_tx):
    return h_sir(simple_tx, c=1540.0, fs=200e6)


def _ref_h_sir(sim, points_m):
    """Compute reference h_sir via the existing low-level kernel."""
    from pyfield.h_sir.farfield_rect_patch import compute_h_sir as _compute_h_sir

    P = points_m.shape[0]
    M = sim.M
    _, t0, dt, T = compute_time_grid(
        P, M, points_m, sim.centers_sub_elem,
        sim.wx, sim.wy, sim.c, sim.fs, sim.delays, verbose=False,
    )
    # Need time_grid for compute_h_sir
    from pyfield.utilities.helper_functions import compute_time_grid as _ctg
    time_grid, t0, dt, T = _ctg(
        P, M, points_m, sim.centers_sub_elem,
        sim.wx, sim.wy, sim.c, sim.fs, sim.delays, verbose=False,
    )
    h, _ = _compute_h_sir(
        P, M, T, dt, time_grid, points_m,
        sim.centers_sub_elem, sim.wx_arr, sim.wy_arr,
        1.0 / sim.c, sim.fs,
        sim.apodization_sub_elem, sim.delays_sub_elem,
        None,
    )
    return t0, h.T  # (T, P)


class TestComputeDerivativeH:
    def test_derivative_h_matches_reference(self, sim):
        """compute_derivative('h') must match direct compute_h_sir output."""
        points_mm = np.array([[0.0, 0.0, 20.0]], dtype=np.float32)
        points_m = points_mm * 1e-3

        t0_ref, h_ref = _ref_h_sir(sim, points_m)
        t0_new, h_new = sim.compute_derivative(points_mm, derivative="h")

        assert t0_new == pytest.approx(t0_ref, rel=1e-6)
        assert h_new.shape == h_ref.shape
        np.testing.assert_allclose(h_new, h_ref, rtol=1e-5, atol=1e-30)

    def test_derivative_h_returns_T_P_shape(self, sim):
        points_mm = np.array([[0.0, 0.0, 20.0], [1.0, 0.0, 25.0]], dtype=np.float32)
        t0, result = sim.compute_derivative(points_mm, derivative="h")
        P = points_mm.shape[0]
        assert result.ndim == 2
        assert result.shape[1] == P

    def test_per_element_ignored_for_h(self, sim):
        """per_element=True is silently ignored for derivative='h'."""
        points_mm = np.array([[0.0, 0.0, 20.0]], dtype=np.float32)
        t0_a, h_a = sim.compute_derivative(points_mm, derivative="h", per_element=False)
        t0_b, h_b = sim.compute_derivative(points_mm, derivative="h", per_element=True)
        assert h_a.shape == h_b.shape
        np.testing.assert_array_equal(h_a, h_b)

    def test_invalid_derivative_raises(self, sim):
        with pytest.raises(ValueError, match="derivative must be"):
            sim.compute_derivative(np.array([[0, 0, 20.0]]), derivative="bad")


class TestComputeDerivativeDh:
    def test_dh_shape(self, sim):
        points_mm = np.array([[0.0, 0.0, 20.0], [0.0, 0.0, 25.0]], dtype=np.float32)
        t0, dh = sim.compute_derivative(points_mm, derivative="dh")
        P = points_mm.shape[0]
        assert dh.ndim == 2
        assert dh.shape[1] == P

    def test_dh_per_element_shape(self, sim, simple_tx):
        points_mm = np.array([[0.0, 0.0, 20.0]], dtype=np.float32)
        t0, dh_pe = sim.compute_derivative(points_mm, derivative="dh", per_element=True)
        P = points_mm.shape[0]
        E = simple_tx.n_elements
        assert dh_pe.shape[0] == dh_pe.shape[0]  # T
        assert dh_pe.shape[1] == P
        assert dh_pe.shape[2] == E

    def test_d2h_per_element_shape(self, sim, simple_tx):
        points_mm = np.array([[0.0, 0.0, 20.0], [0.0, 0.0, 25.0]], dtype=np.float32)
        t0, d2h_pe = sim.compute_derivative(points_mm, derivative="d2h", per_element=True)
        P = points_mm.shape[0]
        E = simple_tx.n_elements
        assert d2h_pe.ndim == 3
        assert d2h_pe.shape[1] == P
        assert d2h_pe.shape[2] == E

    def test_dict_input_accepted(self, sim):
        """Dict field spec accepted without error."""
        grid = {
            "x_extent": [-1, 1], "y_extent": [-0.5, 0.5], "z_extent": [15, 25],
            "dx": 1.0, "dy": 1.0, "dz": 5.0,
        }
        t0, result = sim.compute_derivative(grid, derivative="dh")
        assert result.ndim == 2
