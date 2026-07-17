"""Tests for the general delay-and-sum beamformer `das_volume`.

Reference: for each transmission basis the textbook transmit-arrival time at
a point p is written out explicitly (independently of the beamformer's
internal time-origin recovery), a one-point echo is synthesized on every
channel at ``t_tx(p) + |p − r_e|/c``, and the beamformed volume must peak at
p. This verifies the whole delay bookkeeping: the recovery of the transmit
time origin from the event's own delays, the diverging/focused sign
convention, the plane-wave projection, and the sample interpolation.
"""

import numpy as np
import pytest

from pyfield.beamforming import das_volume

C = 1540.0
FS = 40e6
POINT_MM = np.array([1.0, 0.0, 20.0])
GRID = {
    "x_extent": [-2.0, 4.0],
    "y_extent": [-0.6, 0.61],
    "z_extent": [17.0, 23.0],
    "dx": 0.2,
    "dy": 0.3,
    "dz": 0.1,
}


class _Array:
    """Minimal stand-in for a transducer: an 8x8 matrix, 0.5 mm pitch."""

    def __init__(self):
        u = (np.arange(8) - 3.5) * 0.5e-3
        xx, yy = np.meshgrid(u, u, indexing="ij")
        self.element_centers = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(64)])


def _pulse():
    """2-cycle Hann-windowed 4 MHz burst whose CENTRE sample is the carrier
    maximum (cosine), so the RF argmax marks the arrival time exactly. An
    oscillating pulse is needed for lateral resolution — a smooth bump sums
    coherently over a wide plateau and the peak drifts."""
    t = np.arange(0, 2 / 4e6, 1.0 / FS)
    return (np.cos(2 * np.pi * 4e6 * (t - t[-1] / 2)) * np.hanning(t.size)).astype(
        np.float32
    )


def _synth_rf(el_m, t_tx_point, point_mm):
    """One event's RF: the echo of a unit point at ``point_mm`` STARTS on
    channel r at ``t_tx_point + |p − r_r|/c`` (causal pulse, like a real
    excitation), so its centre lags the geometric arrival by ``lag`` — the
    axial bias `das_volume`'s ``t_offset_s`` must remove."""
    p = point_mm * 1e-3
    t_back = np.linalg.norm(el_m - p, axis=1) / C  # (Erx,)
    pulse = _pulse()
    lag = (pulse.size - 1) / 2.0 / FS
    nt = 1600
    t = np.arange(nt) / FS
    rf = np.zeros((el_m.shape[0], nt), dtype=np.float32)
    for r in range(el_m.shape[0]):
        rf[r] = np.interp(
            t - (t_tx_point + t_back[r]),
            np.arange(pulse.size) / FS,
            pulse,
            left=0.0,
            right=0.0,
        )
    return rf[None], {"t0": 0.0, "dt": 1.0 / FS}, lag


def _peak_mm(vol, axes):
    """Envelope peak (detected along z): the RF argmax can jump one carrier
    cycle axially, the envelope maximum cannot."""
    from scipy.signal import hilbert

    env = np.abs(hilbert(vol, axis=2))
    i = np.unravel_index(np.argmax(env), env.shape)
    return np.array([axes["x_mm"][i[0]], axes["y_mm"][i[1]], axes["z_mm"][i[2]]])


@pytest.mark.parametrize(
    "basis",
    ["plane_wave", "diverging_wave", "focused", "synthetic_aperture"],
)
def test_point_reconstructs_at_true_position(basis):
    arr = _Array()
    el = arr.element_centers
    p = POINT_MM * 1e-3
    apod = np.ones(64)
    if basis == "plane_wave":
        # Steered plane wave: element e fires at (ξ_e − ξ_min)/c with
        # ξ = r_e·n. In the data frame (bulk delay removed) the wavefront
        # crosses p at (p·n − ξ_max)/c.
        n = np.array([np.sin(np.deg2rad(8.0)), 0.0, np.cos(np.deg2rad(8.0))])
        xi = el @ n
        delays = (xi - xi.min()) / C
        t_tx = (p @ n - xi.max()) / C
        event = {"delays": delays, "apodization": apod, "angles_deg": (8.0, 0.0)}
    elif basis == "diverging_wave":
        # Source behind the array: delays (d_e − d_min)/c; the spherical
        # wavefront crosses p at (|p − r_vs| − d_max)/c.
        vs = np.array([2.0, 0.0, -10.0]) * 1e-3
        d = np.linalg.norm(el - vs, axis=1)
        delays = (d - d.min()) / C
        t_tx = (np.linalg.norm(p - vs) - d.max()) / C
        event = {"delays": delays, "apodization": apod, "virtual_source_mm": vs * 1e3}
    elif basis == "focused":
        # Focus above the point: delays (d_max − d_e)/c put the focus at
        # d_min/c in the data frame; past the focus the wave diverges from it.
        vs = np.array([1.0, 0.0, 18.0]) * 1e-3
        d = np.linalg.norm(el - vs, axis=1)
        delays = (d.max() - d) / C
        t_tx = (d.min() + np.sign(p[2] - vs[2]) * np.linalg.norm(p - vs)) / C
        event = {"delays": delays, "apodization": apod, "virtual_source_mm": vs * 1e3}
    else:
        # Synthetic aperture: one corner element fires with zero delay; the
        # wave diverges from the element itself.
        apod = np.zeros(64)
        apod[0] = 1.0
        t_tx = np.linalg.norm(p - el[0]) / C
        event = {
            "delays": np.zeros(64),
            "apodization": apod,
            "virtual_source_mm": el[0] * 1e3,
        }

    rf, coords, lag = _synth_rf(el, t_tx, POINT_MM)
    vol, axes = das_volume(
        rf,
        coords,
        [event],
        arr,
        GRID,
        c=C,
        fnum=0.5,
        rx_apodization="rect",
        t_offset_s=lag,
    )
    err = np.abs(_peak_mm(vol, axes) - POINT_MM)
    tol = np.array([GRID["dx"], GRID["dy"], GRID["dz"]]) + 1e-9  # one voxel
    assert np.all(err <= tol), f"{basis}: peak off by {err} mm"


def test_event_requires_exactly_one_geometry_key():
    arr = _Array()
    rf = np.zeros((1, 64, 100), dtype=np.float32)
    coords = {"t0": 0.0, "dt": 1.0 / FS}
    with pytest.raises(ValueError, match="exactly one"):
        das_volume(rf, coords, [{"delays": np.zeros(64)}], arr, GRID)
    with pytest.raises(ValueError, match="exactly one"):
        das_volume(
            rf,
            coords,
            [
                {
                    "delays": np.zeros(64),
                    "angles_deg": 0.0,
                    "virtual_source_mm": [0, 0, -10],
                }
            ],
            arr,
            GRID,
        )
