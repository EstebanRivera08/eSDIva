"""Visual regression tests for plotting functions using pytest-mpl."""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

matplotlib.use("Agg")

from pyfield.utilities.plotting import plot_pressure_2D


@pytest.fixture
def synthetic_pressure_3d():
    """Create a simple synthetic 3D pressure field for plotting tests."""
    x = np.linspace(-5, 5, 21)
    y = np.linspace(-2, 2, 9)
    z = np.linspace(5, 25, 41)

    # Create a Gaussian-like pressure pattern
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    p = np.exp(-((X / 3) ** 2) - (Y / 1.5) ** 2 - ((Z - 15) / 5) ** 2)
    return x, y, z, p


@pytest.fixture
def synthetic_pressure_2d_plane():
    """Create a 2D pressure plane (single y-slice)."""
    x = np.linspace(-5, 5, 21)
    y = np.array([0.0])
    z = np.linspace(5, 25, 41)

    X, Z = np.meshgrid(x, z, indexing="ij")
    p = np.exp(-((X / 3) ** 2) - ((Z - 15) / 5) ** 2)
    return x, y, z, p[:, np.newaxis, :]


class TestPlotPressure2D:
    def test_returns_axes(self):
        x = np.linspace(-5, 5, 11)
        z = np.linspace(5, 25, 21)
        p = np.random.default_rng(0).random((11, 21))
        ax = plot_pressure_2D(x, z, p)
        assert isinstance(ax, plt.Axes)
        plt.close("all")

    def test_custom_ax(self):
        x = np.linspace(-5, 5, 11)
        z = np.linspace(5, 25, 21)
        p = np.random.default_rng(0).random((11, 21))
        fig, ax = plt.subplots()
        returned_ax = plot_pressure_2D(x, z, p, ax=ax)
        assert returned_ax is ax
        plt.close("all")


# [TO DO] : Perform better tests for plotting utilities
