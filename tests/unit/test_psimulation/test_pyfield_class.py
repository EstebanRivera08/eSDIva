"""Tests for pyfield.psimulation.PyField deprecated wrapper."""

import warnings

import pytest


def _make_pyfield(tx, **kwargs):
    from pyfield.psimulation import PyField

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return PyField(tx, **kwargs)


class TestPyFieldCreation:
    def test_instantiation(self, small_linear_transducer):
        sim = _make_pyfield(small_linear_transducer)
        assert sim.fc == small_linear_transducer.fc
        assert sim.M > 0
        assert sim.c == 1540.0
        assert sim.fs == 200e6

    def test_custom_medium_params(self, small_linear_transducer):
        sim = _make_pyfield(small_linear_transducer, c=1500.0, rho=1000.0, fs=100e6)
        assert sim.c == 1500.0
        assert sim.rho == 1000.0
        assert sim.fs == 100e6

    def test_deprecation_warning_on_init(self, small_linear_transducer):
        from pyfield.psimulation import PyField

        with pytest.warns(DeprecationWarning, match="deprecated"):
            PyField(small_linear_transducer)

    def test_repr(self, small_linear_transducer):
        sim = _make_pyfield(small_linear_transducer)
        r = repr(sim)
        assert "PyField" in r
        assert "1540" in r

    def test_set_field_valid(self, small_linear_transducer):
        sim = _make_pyfield(small_linear_transducer)
        sim.set_field("c", 1500.0)
        assert sim.c == 1500.0

    def test_set_field_invalid(self, small_linear_transducer):
        sim = _make_pyfield(small_linear_transducer)
        with pytest.raises(AttributeError, match="not a valid attribute"):
            sim.set_field("nonexistent_attr", 42)
