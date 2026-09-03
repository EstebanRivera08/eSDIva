"""Tests for RFDataset — checkpointed per-event RF storage with a contents file."""

import numpy as np
import pytest

from esdiva.io import RFDataset


def _config(**overrides):
    """Baseline simulation fingerprint config; override single keys per test."""
    cfg = {
        "fs": 1e8,
        "c": 1540.0,
        "excitation": np.arange(8, dtype=np.float32),
        "tx_events": [{"delays": np.zeros(3, dtype=np.float32)}],
    }
    cfg.update(overrides)
    return cfg


@pytest.fixture
def dataset(tmp_path):
    return RFDataset(tmp_path / "ds", config=_config())


class TestRoundtrip:
    def test_write_read_event(self, dataset):
        rf = np.random.default_rng(0).standard_normal((4, 100)).astype(np.float32)
        dataset.write_event(0, rf, t0=1.5e-6, dt=1e-8)
        rf_back, t0, dt = dataset.read_event(0, verify=True)
        np.testing.assert_array_equal(rf_back, rf)
        assert t0 == 1.5e-6 and dt == 1e-8

    def test_load_all_pads_to_longest(self, dataset):
        rng = np.random.default_rng(1)
        rf0 = rng.standard_normal((4, 100)).astype(np.float32)
        rf1 = rng.standard_normal((4, 80)).astype(np.float32)
        dataset.write_event(0, rf0, t0=1e-6, dt=1e-8)
        dataset.write_event(1, rf1, t0=2e-6, dt=1e-8)

        rf_all, coords = dataset.load_all()
        assert rf_all.shape == (2, 4, 100)
        np.testing.assert_array_equal(rf_all[0], rf0)
        np.testing.assert_array_equal(rf_all[1, :, :80], rf1)
        # Shorter events are zero-padded at the END (only t0 differs).
        np.testing.assert_array_equal(rf_all[1, :, 80:], 0.0)
        np.testing.assert_array_equal(coords["t0_per_event"], [1e-6, 2e-6])
        assert coords["t0"] == 1e-6 and coords["dt"] == 1e-8

    def test_load_all_sums_chunk_groups(self, tmp_path):
        """A chunked dataset collapses to per-event RF by summing chunks."""
        ds = RFDataset(
            tmp_path / "ds",
            config=_config(),
            meta={"n_events": 4, "checkpoint_chunks": 2},
        )
        rng = np.random.default_rng(2)
        parts = rng.standard_normal((4, 3, 20)).astype(np.float32)
        for i in range(4):
            # Chunks of one event share t0 (grid sentinels), events differ.
            ds.write_event(i, parts[i], t0=1e-6 * (i // 2), dt=1e-8)

        rf, coords = ds.load_all()
        assert rf.shape == (2, 3, 20)
        np.testing.assert_allclose(rf[0], parts[0] + parts[1], rtol=1e-6)
        np.testing.assert_allclose(rf[1], parts[2] + parts[3], rtol=1e-6)
        np.testing.assert_array_equal(coords["t0_per_event"], [0.0, 1e-6])

    def test_load_all_incomplete_chunked_refused(self, tmp_path):
        """Partial chunk groups must not be summed into a half-event RF."""
        ds = RFDataset(
            tmp_path / "ds",
            config=_config(),
            meta={"n_events": 4, "checkpoint_chunks": 2},
        )
        for i in range(3):  # last chunk of event 1 missing.
            ds.write_event(i, np.ones((3, 20), np.float32), t0=0.0, dt=1e-8)
        with pytest.raises(ValueError, match="chunked"):
            ds.load_all()

    def test_load_all_empty_raises(self, dataset):
        with pytest.raises(ValueError, match="no completed events"):
            dataset.load_all()


class TestResume:
    def test_reopen_same_config_resumes(self, tmp_path):
        ds = RFDataset(tmp_path / "ds", config=_config())
        ds.write_event(0, np.ones((2, 10), np.float32), t0=0.0, dt=1e-8)
        ds2 = RFDataset(tmp_path / "ds", config=_config())
        assert ds2.completed == [0]

    def test_changed_scalar_refused(self, tmp_path):
        RFDataset(tmp_path / "ds", config=_config())
        with pytest.raises(ValueError, match="fs"):
            RFDataset(tmp_path / "ds", config=_config(fs=2e8))

    def test_changed_array_values_refused(self, tmp_path):
        """Same shape/dtype but different values must be caught (byte hash)."""
        RFDataset(tmp_path / "ds", config=_config())
        with pytest.raises(ValueError, match="excitation"):
            RFDataset(
                tmp_path / "ds",
                config=_config(excitation=np.arange(8, dtype=np.float32) + 1),
            )

    def test_open_existing_without_config(self, tmp_path):
        ds = RFDataset(tmp_path / "ds", config=_config())
        ds.write_event(0, np.ones((2, 10), np.float32), t0=0.0, dt=1e-8)
        assert RFDataset(tmp_path / "ds").completed == [0]

    def test_new_dataset_requires_config(self, tmp_path):
        with pytest.raises(ValueError, match="config"):
            RFDataset(tmp_path / "empty")

    def test_missing_file_not_completed(self, dataset):
        dataset.write_event(0, np.ones((2, 10), np.float32), t0=0.0, dt=1e-8)
        dataset.write_event(1, np.ones((2, 10), np.float32), t0=0.0, dt=1e-8)
        (dataset.path / "rf_event_0000.npz").unlink()
        assert dataset.completed == [1]


class TestCrashSafety:
    def test_orphan_tmp_file_ignored(self, dataset):
        """A crash mid-write leaves only a *.tmp file — never a valid event."""
        dataset.write_event(0, np.ones((2, 10), np.float32), t0=0.0, dt=1e-8)
        (dataset.path / "leftover.tmp").write_bytes(b"garbage from a crash")
        assert dataset.completed == [0]
        rf_all, _ = dataset.load_all()
        assert rf_all.shape == (1, 2, 10)

    def test_verify_detects_corruption(self, dataset):
        dataset.write_event(0, np.ones((2, 10), np.float32), t0=0.0, dt=1e-8)
        fpath = dataset.path / "rf_event_0000.npz"
        raw = bytearray(fpath.read_bytes())
        raw[len(raw) // 2] ^= 0xFF  # flip one bit-pattern mid-file
        fpath.write_bytes(bytes(raw))
        with pytest.raises(ValueError, match="checksum"):
            dataset.read_event(0, verify=True)


class TestTimeReferenceMigration:
    """Version-1 stores kept a geometric `t0` with the pulse lag left over.

    `t0` now means the beamforming reference (lag already removed), so a store
    written under the old convention must be shifted when it is read back —
    otherwise an old acquisition beamforms `c·lag/2` deep.
    """

    def test_v1_store_shifts_t0_by_the_stored_lag(self, tmp_path):
        lag = 3e-7
        ds = RFDataset(tmp_path / "old", config=_config())
        ds.write_event(
            0,
            np.zeros((4, 50), dtype=np.float32),
            t0=1e-5,
            dt=1e-8,
            pulse_center_lag_s=lag,
        )
        # Rewrite the contents file as the old format would have left it.
        contents = tmp_path / "old" / "contents.json"
        contents.write_text(
            contents.read_text().replace('"version": 2', '"version": 1')
        )

        _, coords = RFDataset(tmp_path / "old").load_all()
        assert coords["t0"] == pytest.approx(1e-5 - lag)
        assert coords["pulse_center_lag_s"] == pytest.approx(lag)

    def test_current_store_keeps_t0(self, dataset):
        dataset.write_event(
            0,
            np.zeros((4, 50), dtype=np.float32),
            t0=1e-5,
            dt=1e-8,
            pulse_center_lag_s=3e-7,
        )
        _, coords = dataset.load_all()
        assert coords["t0"] == pytest.approx(1e-5)
