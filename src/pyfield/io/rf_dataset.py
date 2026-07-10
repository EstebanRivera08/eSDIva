"""Checkpointed on-disk RF dataset: one compressed file per TX event.

Long acquisition sequences (many TX events, each seconds-to-minutes of
simulation) must survive a crash without restarting from zero. `RFDataset`
stores each event's per-channel RF as an independent compressed ``.npz``
file next to a ``manifest.json`` that records what was simulated (a
fingerprint of the probe, medium, excitation, scatterers and TX events) and
which events completed. Re-opening the same folder with the same
configuration resumes: completed events are skipped, only missing ones are
recomputed. Opening it with a *different* configuration raises, listing
exactly which settings changed — a half-matching dataset is never silently
mixed with new data.

Every write is atomic (written to a temporary file, then renamed), so a
crash mid-event never corrupts an already-completed event or the manifest.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

_MANIFEST = "manifest.json"
_EVENT_FMT = "rf_event_{:04d}.npz"


def config_fingerprint(config: dict) -> dict:
    """Flatten a configuration dict into comparable strings.

    Scalars and strings are kept as ``repr``; numpy arrays are replaced by a
    SHA-256 digest of their raw bytes plus shape/dtype, so two runs match iff
    every array is bit-identical. The result is what ``manifest.json`` stores
    and what resume compares against.

    Parameters
    ----------
    config : dict
        Arbitrary (nested) dict of scalars, strings, lists and numpy arrays
        describing the simulation (probe geometry, fs, c, excitation,
        scatterers, TX events, ...).

    Returns
    -------
    dict
        Flat ``{dotted.key: string}`` mapping.
    """
    flat: dict[str, str] = {}

    def _walk(prefix: str, value) -> None:
        if isinstance(value, dict):
            for k in sorted(value):
                _walk(f"{prefix}.{k}" if prefix else str(k), value[k])
        elif isinstance(value, np.ndarray):
            digest = hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
            flat[prefix] = f"ndarray{value.shape}:{value.dtype}:sha256={digest[:16]}"
        elif isinstance(value, (list, tuple)):
            arr = np.asarray(value)
            if arr.dtype == object:  # heterogeneous list: recurse per item.
                for i, item in enumerate(value):
                    _walk(f"{prefix}[{i}]", item)
            else:
                _walk(prefix, arr)
        else:
            flat[prefix] = repr(value)

    _walk("", config)
    return flat


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write bytes to ``path`` via a temp file + rename, so a crash mid-write
    leaves either the old file or the new one — never a truncated file."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


class RFDataset:
    """Resumable folder of per-event RF files with a JSON manifest.

    Parameters
    ----------
    path : str or pathlib.Path
        Dataset folder. Created if missing.
    config : dict, optional
        Configuration identifying the simulation (see `config_fingerprint`).
        Required when creating a new dataset. When opening an existing one,
        it is compared against the stored fingerprint: any difference raises
        `ValueError` listing the changed keys. ``None`` opens read-only
        without checking (loading data for beamforming).
    meta : dict, optional
        Free-form JSON-serializable info stored once at creation
        (e.g. ``{"n_events": 18, "fs": 1e8, "note": "..."}``).

    Raises
    ------
    ValueError
        If ``config`` differs from the fingerprint stored in an existing
        manifest, or if a new dataset is created without ``config``.
    """

    _manifest: dict[str, Any]

    def __init__(self, path, config: dict | None = None, *, meta: dict | None = None):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self._manifest_path = self.path / _MANIFEST

        if self._manifest_path.exists():
            self._manifest = json.loads(self._manifest_path.read_text())
            if config is not None:
                self._check_fingerprint(config_fingerprint(config))
        else:
            if config is None:
                raise ValueError(
                    f"No manifest in {self.path} — pass `config` to create a "
                    f"new dataset (or point to an existing one)."
                )
            self._manifest = {
                "version": 1,
                "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "fingerprint": config_fingerprint(config),
                "meta": meta or {},
                "events": {},
            }
            self._save_manifest()

    # ------------------------------------------------------------------
    def _check_fingerprint(self, new: dict) -> None:
        old = self._manifest["fingerprint"]
        diffs = [
            f"  {k}: stored={old.get(k, '<missing>')}  now={new.get(k, '<missing>')}"
            for k in sorted(set(old) | set(new))
            if old.get(k) != new.get(k)
        ]
        if diffs:
            raise ValueError(
                f"Dataset {self.path} was simulated with a DIFFERENT "
                f"configuration ({len(diffs)} difference(s)):\n"
                + "\n".join(diffs)
                + "\nUse a new output folder (or delete this one) to re-simulate."
            )

    def _save_manifest(self) -> None:
        _atomic_write_bytes(
            self._manifest_path, json.dumps(self._manifest, indent=1).encode()
        )

    # ------------------------------------------------------------------
    @property
    def meta(self) -> dict:
        """Free-form info stored at creation."""
        return self._manifest["meta"]

    @property
    def completed(self) -> list[int]:
        """Sorted indices of events whose file exists and is recorded done."""
        return sorted(
            int(k)
            for k, ev in self._manifest["events"].items()
            if (self.path / ev["file"]).exists()
        )

    # ------------------------------------------------------------------
    def write_event(self, idx: int, rf, t0: float, dt: float, **info) -> None:
        """Atomically store one TX event's RF and mark it completed.

        Parameters
        ----------
        idx : int
            TX event index in the sequence.
        rf : (Erx, Nt) numpy.ndarray
            Per-receive-channel RF of this event.
        t0 : float
            Beam-axis time origin of the first sample (s).
        dt : float
            Sample period (s).
        **info
            Extra JSON-serializable fields recorded in the manifest
            (e.g. ``duration_s=12.3``).
        """
        rf = np.asarray(rf, dtype=np.float32)
        fname = _EVENT_FMT.format(idx)
        fd, tmp = tempfile.mkstemp(dir=self.path, suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                np.savez_compressed(f, rf=rf, t0=np.float64(t0), dt=np.float64(dt))
            os.replace(tmp, self.path / fname)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

        digest = hashlib.sha256((self.path / fname).read_bytes()).hexdigest()
        self._manifest["events"][str(idx)] = {
            "file": fname,
            "sha256": digest,
            "shape": list(rf.shape),
            "t0": float(t0),
            "dt": float(dt),
            "completed": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            **info,
        }
        self._save_manifest()

    def read_event(self, idx: int, *, verify: bool = False):
        """Load one event's RF.

        Parameters
        ----------
        idx : int
            TX event index.
        verify : bool, default False
            Re-hash the file and compare with the manifest checksum.

        Returns
        -------
        rf : (Erx, Nt) numpy.ndarray
        t0 : float
        dt : float

        Raises
        ------
        KeyError
            If the event is not in the manifest.
        ValueError
            If ``verify`` is True and the checksum does not match.
        """
        ev = self._manifest["events"][str(idx)]
        fpath = self.path / ev["file"]
        if verify:
            digest = hashlib.sha256(fpath.read_bytes()).hexdigest()
            if digest != ev["sha256"]:
                raise ValueError(f"{fpath} checksum mismatch — file corrupted.")
        with np.load(fpath) as z:
            return z["rf"], float(z["t0"]), float(z["dt"])

    def load_all(self):
        """Assemble all completed events into the `sequence_rf` return format.

        Traces are zero-padded at the end to the longest event (only the
        time origin differs between events; ``dt`` is shared).

        When the dataset was written with ``checkpoint_chunks > 1`` (each TX
        event split into scatterer chunks, one file per chunk), the chunk RFs
        of each event share one time grid and are summed here — the RF is
        linear in the scatterers, so the sum equals the unchunked event.

        Returns
        -------
        rf : (N_events, Erx, Nt) numpy.ndarray
            Per-event, per-channel RF (float32).
        coords : dict
            ``"t0"``/``"dt"`` of the first event plus ``"t0_per_event"``.

        Raises
        ------
        ValueError
            If the dataset has no completed events, or if it is chunked and
            not yet complete (partial chunk groups cannot be summed).
        """
        idxs = self.completed
        if not idxs:
            raise ValueError(f"Dataset {self.path} has no completed events.")
        chunks = int(self.meta.get("checkpoint_chunks") or 1)
        if chunks > 1:
            n_target = int(self.meta["n_events"])
            if set(idxs) != set(range(n_target)):
                raise ValueError(
                    f"Dataset {self.path} is chunked ({chunks} chunks/event) "
                    f"but only {len(idxs)}/{n_target} files are complete — "
                    "finish the acquisition before loading."
                )
        events = [self.read_event(i) for i in idxs]
        nt = max(rf.shape[1] for rf, _, _ in events)
        n_rx = events[0][0].shape[0]
        rf_all = np.zeros((len(events), n_rx, nt), dtype=np.float32)
        for k, (rf, _, _) in enumerate(events):
            rf_all[k, :, : rf.shape[1]] = rf
        t0s = np.array([t0 for _, t0, _ in events], dtype=np.float64)
        if chunks > 1:
            # Grid sentinels guarantee identical chunk time grids; verify
            # before collapsing so a mismatch can never silently smear echoes.
            t0g = t0s.reshape(-1, chunks)
            if not np.all(t0g == t0g[:, :1]):
                raise ValueError(
                    f"Dataset {self.path}: chunk time origins differ within an "
                    "event — files were not written by a chunked sequence_rf."
                )
            rf_all = rf_all.reshape(-1, chunks, n_rx, nt).sum(axis=1)
            t0s = t0g[:, 0]
        return rf_all, {
            "t0": t0s[0],
            "dt": events[0][2],
            "t0_per_event": t0s,
        }

    # ------------------------------------------------------------------
    def summary(self) -> str:
        """Human-readable status table (also returned as a string)."""
        n_target = self.meta.get("n_events")
        done = self.completed
        lines = [
            f"RFDataset {self.path}",
            f"  created  : {self._manifest['created']}",
            f"  completed: {len(done)}"
            + (f" / {n_target}" if n_target is not None else ""),
        ]
        for i in done:
            ev = self._manifest["events"][str(i)]
            dur = ev.get("duration_s")
            lines.append(
                f"    event {i:4d}  {ev['file']}  shape={tuple(ev['shape'])}"
                f"  t0={ev['t0'] * 1e6:+.2f} µs"
                + (f"  {dur:.1f} s" if dur is not None else "")
                + f"  [{ev['completed']}]"
            )
        text = "\n".join(lines)
        print(text)
        return text

    def __repr__(self) -> str:
        return f"RFDataset({self.path}, completed={len(self.completed)})"
