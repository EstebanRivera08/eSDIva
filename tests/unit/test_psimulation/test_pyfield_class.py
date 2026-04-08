"""Tests for pyfield.psimulation.PyField class instantiation and basic API."""

import pytest


class TestPyFieldCreation:
    def test_instantiation(self, small_linear_transducer):
        from pyfield.psimulation import PyField

        sim = PyField(small_linear_transducer)
        assert sim.fc == small_linear_transducer.fc
        assert sim.M > 0
        assert sim.c == 1540.0
        assert sim.fs == 200e6

    def test_custom_medium_params(self, small_linear_transducer):
        from pyfield.psimulation import PyField

        sim = PyField(
            small_linear_transducer,
            c=1500.0,
            rho=1000.0,
            fs=100e6,
        )
        assert sim.c == 1500.0
        assert sim.rho == 1000.0
        assert sim.fs == 100e6

    def test_repr(self, small_linear_transducer):
        from pyfield.psimulation import PyField

        sim = PyField(small_linear_transducer)
        r = repr(sim)
        assert "PyField" in r
        assert "1540" in r

    def test_set_field_valid(self, small_linear_transducer):
        from pyfield.psimulation import PyField

        sim = PyField(small_linear_transducer)
        sim.set_field("c", 1500.0)
        assert sim.c == 1500.0

    def test_set_field_invalid(self, small_linear_transducer):
        from pyfield.psimulation import PyField

        sim = PyField(small_linear_transducer)
        with pytest.raises(AttributeError, match="not a valid attribute"):
            sim.set_field("nonexistent_attr", 42)
