"""HDF5 interchange export round-trips the RF and its timing."""

import h5py
import numpy as np

from esdiva.io import save_rf_hdf5


def test_save_rf_hdf5_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    rf = rng.standard_normal((3, 8, 100)).astype(np.float32)  # (events, Erx, Nt)
    coords = {
        "dt": 5e-9,
        "t0": 1e-6,
        "t0_per_event": np.array([1e-6, 1.2e-6, 1.4e-6]),
        "pulse_center_lag_s": 2e-7,
    }
    geom = rng.standard_normal((8, 3))

    out = save_rf_hdf5(tmp_path / "rf.h5", rf, coords, probe_geometry_mm=geom)

    with h5py.File(out, "r") as f:
        np.testing.assert_array_equal(f["channel_data"][:], rf)
        np.testing.assert_allclose(f["t0_per_event"][:], coords["t0_per_event"])
        np.testing.assert_allclose(f["probe_geometry_mm"][:], geom)
        assert f.attrs["sampling_frequency"] == 1.0 / coords["dt"]
        assert f.attrs["initial_time"] == coords["t0_per_event"][0]
        assert f.attrs["pulse_center_lag_s"] == coords["pulse_center_lag_s"]


def test_single_event_2d_rf_is_promoted(tmp_path):
    rf = np.ones((4, 50), dtype=np.float32)  # (Erx, Nt) single event
    out = save_rf_hdf5(tmp_path / "one.h5", rf, {"dt": 1e-8, "t0": 0.0})
    with h5py.File(out, "r") as f:
        assert f["channel_data"].shape == (1, 4, 50)
