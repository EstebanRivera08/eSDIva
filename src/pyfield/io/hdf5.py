"""Export simulated RF to a single self-describing HDF5 file.

The checkpointed `RFDataset` (one compressed ``.npz`` per event) is PyField's
*internal* store — good for resuming a long run, but not meant for sharing.
For interchange, ultrasound tools read HDF5: MATLAB (``h5read``), USTB, and
Python (``h5py``) all open it natively. This writes one ``.h5`` holding the
channel data plus the acquisition parameters a beamformer needs, using the
field names USTB's Ultrasound File Format (UFF) uses for its ``channel_data``
object (``sampling_frequency``, ``initial_time``, ``sound_speed``, …) so the
result drops straight into a USTB workflow. It is a plain HDF5 file with
UFF-compatible naming, not a full serialized USTB ``uff`` object.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import numpy.typing as npt


def save_rf_hdf5(
    path,
    rf: npt.NDArray[np.floating],
    coords: dict,
    *,
    probe_geometry_mm: npt.NDArray[np.floating] | None = None,
    sound_speed: float = 1540.0,
    meta: dict | None = None,
) -> Path:
    """Write per-channel RF and its timing to a single HDF5 file.

    Parameters
    ----------
    path : str or pathlib.Path
        Output ``.h5`` file (created/overwritten).
    rf : (N_events, Erx, Nt) or (Erx, Nt) numpy.ndarray
        Per-event, per-receive-channel RF, as returned by ``sequence_rf`` (3-D)
        or ``pulse_echo_rf`` (2-D). A 2-D array is stored as a one-event
        acquisition.
    coords : dict
        Timing from the reception simulator / ``RFDataset.load_all``: ``"dt"``
        (sample period, s), ``"t0"`` (first-sample time, s), optionally
        ``"t0_per_event"`` (s) and ``"pulse_center_lag_s"`` (the two-way pulse
        lag a beamformer adds — stored so downstream tools reproduce PyField's
        depth referencing).
    probe_geometry_mm : (Erx, 3) numpy.ndarray, optional
        Receive-element centres in mm (``rx.element_centers * 1e3``), stored so
        the file is beamformable on its own. Omitted if ``None``.
    sound_speed : float, default 1540.0
        Speed of sound (m/s), stored as the ``sound_speed`` attribute.
    meta : dict, optional
        Any extra JSON-serializable acquisition info; stored as a JSON string
        in the ``meta`` attribute (e.g. probe name, sequence description).

    Returns
    -------
    pathlib.Path
        The written file path.
    """
    import h5py

    rf = np.ascontiguousarray(rf, dtype=np.float32)
    if rf.ndim == 2:  # single event (Erx, Nt) -> (1, Erx, Nt)
        rf = rf[None]
    if rf.ndim != 3:
        raise ValueError(f"rf must be 2-D or 3-D, got shape {rf.shape}.")

    dt = float(coords["dt"])
    n_ev = rf.shape[0]
    t0_per_event = np.asarray(
        coords["t0_per_event"] if "t0_per_event" in coords
        else np.full(n_ev, coords.get("t0", 0.0)),
        dtype=np.float64,
    )

    path = Path(path)
    with h5py.File(path, "w") as f:
        # (N_events, Erx, Nt) — the raw echo traces (UFF calls this channel_data).
        f.create_dataset("channel_data", data=rf, compression="gzip")
        f.create_dataset("t0_per_event", data=t0_per_event)
        if probe_geometry_mm is not None:
            f.create_dataset(
                "probe_geometry_mm",
                data=np.asarray(probe_geometry_mm, dtype=np.float64),
            )
        # Scalars a beamformer needs, named as in USTB's uff.channel_data.
        f.attrs["sampling_frequency"] = 1.0 / dt
        f.attrs["dt"] = dt
        f.attrs["initial_time"] = float(t0_per_event[0])
        f.attrs["sound_speed"] = float(sound_speed)
        if "pulse_center_lag_s" in coords:
            f.attrs["pulse_center_lag_s"] = float(coords["pulse_center_lag_s"])
        if meta:
            f.attrs["meta"] = json.dumps(meta, default=str)
    return path
