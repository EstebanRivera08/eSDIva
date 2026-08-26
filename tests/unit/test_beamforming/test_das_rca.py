"""Tests for das_rca_volume — 3-D row-column plane-wave delay-and-sum.

Two levels: (1) synthetic RF with a delta placed at the analytic two-way
delay of a known point — verifies the delay model, axis inference, t0
handling and interpolation localize the point exactly; (2) end-to-end
against Reception — verifies the DAS time convention matches the
simulator's beam-axis-referenced ``t0`` (the real regression risk).
"""

import warnings

import numpy as np
import pytest
from scipy.signal import hilbert

from esdiva.beamforming import das_rca_volume
from esdiva.reception import Reception
from esdiva.transducers import LinearArrayTransducer

C = 1540.0


def _crossed_centers(n, pitch_mm, tx_axis):
    """Element centres (mm) for crossed TX/RX linear arrays at the origin."""
    idx = (np.arange(n) - (n - 1) / 2) * pitch_mm
    tx = np.zeros((n, 3))
    tx[:, tx_axis] = idx
    rx = np.zeros((n, 3))
    rx[:, 1 - tx_axis] = idx
    return tx, rx, n * pitch_mm  # row length = aperture extent


def _delta_rf(point_mm, angles_deg, tx_mm, rx_mm, row_len_mm, t0_ev, dt, nt=4000):
    """RF with a split-sample delta at each channel's analytic two-way delay.

    Uses the same physics as the beamformer's docstring: plane-wave arrival
    ``(u·sinα + z·cosα − ξ_max)/c`` plus nearest-point-of-row return path.
    """
    p = np.asarray(point_mm, float) * 1e-3
    tx, rx = tx_mm * 1e-3, rx_mm * 1e-3
    u_ax = int(np.argmax(np.ptp(tx[:, :2], axis=0)))
    half = row_len_mm * 1e-3 / 2
    rf = np.zeros((len(angles_deg), rx.shape[0], nt), np.float32)
    for e, a in enumerate(np.deg2rad(angles_deg)):
        xi_max = (tx[:, u_ax] * np.sin(a) + tx[:, 2] * np.cos(a)).max()
        t_tx = (p[u_ax] * np.sin(a) + p[2] * np.cos(a) - xi_max) / C
        for r in range(rx.shape[0]):
            du = max(0.0, abs(p[u_ax] - rx[r, u_ax]) - half)
            dv = p[1 - u_ax] - rx[r, 1 - u_ax]
            t_rx = np.hypot(np.hypot(du, dv), p[2] - rx[r, 2]) / C
            s = (t_tx + t_rx - t0_ev[e]) / dt
            i0 = int(np.floor(s))
            rf[e, r, i0] += 1.0 - (s - i0)
            rf[e, r, i0 + 1] += s - i0
    return rf


def _peak_mm(volume, axes):
    ix, iy, iz = np.unravel_index(np.argmax(np.abs(volume)), volume.shape)
    return axes["x_mm"][ix], axes["y_mm"][iy], axes["z_mm"][iz]


@pytest.mark.parametrize("tx_axis", [0, 1])
def test_point_localized_synthetic(tx_axis):
    """Delta at the analytic delay must beamform to the true voxel (both
    orientations: columns-TX along x and rows-TX along y)."""
    point = np.array([1.2, -0.9, 12.0])
    angles = [-6.0, 0.0, 6.0]
    dt = 1.0 / 50e6
    tx_mm, rx_mm, row_len = _crossed_centers(17, 0.6, tx_axis)
    # Distinct nonzero per-event t0 exercises the t0_per_event handling.
    t0_ev = np.array([2e-6, 2.5e-6, 3e-6])

    rf = _delta_rf(point, angles, tx_mm, rx_mm, row_len, t0_ev, dt)
    grid = {
        "x_extent": [point[0] - 1.5, point[0] + 1.5],
        "y_extent": [point[1] - 1.5, point[1] + 1.5],
        "z_extent": [point[2] - 2.0, point[2] + 2.0],
        "dx": 0.1,
        "dy": 0.1,
        "dz": 0.05,
    }
    vol, axes = das_rca_volume(
        rf,
        {"t0_per_event": t0_ev, "t0": t0_ev[0], "dt": dt},
        angles_deg=angles,
        tx_centers_mm=tx_mm,
        rx_centers_mm=rx_mm,
        rx_length_mm=row_len,
        grid_mm=grid,
        c=C,
        fnum=0.8,
    )
    peak = _peak_mm(vol, axes)
    np.testing.assert_allclose(peak, point, atol=0.11)  # within one voxel


def test_invalid_apodization_raises():
    rf = np.zeros((1, 2, 10), np.float32)
    with pytest.raises(ValueError, match="rx_apodization"):
        das_rca_volume(
            rf,
            {"t0": 0.0, "dt": 1e-8},
            angles_deg=[0.0],
            tx_centers_mm=np.zeros((2, 3)),
            rx_centers_mm=np.zeros((2, 3)),
            rx_length_mm=1.0,
            grid_mm={
                "x_extent": [0, 1],
                "y_extent": [0, 1],
                "z_extent": [1, 2],
                "dx": 0.5,
                "dy": 0.5,
                "dz": 0.5,
            },
            rx_apodization="blackman",
        )


def test_point_localized_end_to_end():
    """Full chain: Reception RCA sequence → DAS → peak at the scatterer.

    This is the convention guard: it fails if the beamformer's time origin
    disagrees with the simulator's beam-axis-referenced ``t0`` or if the
    plane-wave delay sign flips.
    """
    fc, fs = 3e6, 50e6
    n = 8
    tx = LinearArrayTransducer(
        n_elements=n,
        element_width_mm=0.55,
        element_height_mm=4.75,
        kerf_mm=0.05,
        no_sub_x=1,
        no_sub_y=8,
        frequency_Hz=fc,
    )
    rx = LinearArrayTransducer(
        n_elements=n,
        element_width_mm=0.55,
        element_height_mm=4.75,
        kerf_mm=0.05,
        no_sub_x=1,
        no_sub_y=8,
        frequency_Hz=fc,
    )
    # Rotate the RX array 90° about z: its elements become the "rows".
    rot_z = np.array(
        [[0.0, -1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0, 0, 1, 0], [0, 0, 0, 1]]
    )
    rx.transform(rot_z)

    t = np.arange(0, 2 / fc, 1 / fs)
    exc = (np.sin(2 * np.pi * fc * t) * np.hanning(t.size)).astype(np.float32)
    sim = Reception(tx, rx, c=C, fs=fs, excitation=exc, verbose=False)

    point = np.array([[0.8, -0.6, 12.0]], dtype=np.float32)
    angles = [-5.0, 0.0, 5.0]
    events = []
    for a in angles:
        nvec = np.array([np.sin(np.deg2rad(a)), 0.0, np.cos(np.deg2rad(a))])
        d = tx.element_centers @ nvec / C
        events.append(
            {
                "delays": (d - d.min()).astype(np.float32),
                "apodization": np.ones(n, np.float32),
            }
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        rf, coords = sim.sequence_rf(point, np.array([1.0], np.float32), events)

    grid = {
        "x_extent": [-1.2, 2.8],
        "y_extent": [-2.6, 1.4],
        "z_extent": [10.0, 14.0],
        "dx": 0.1,
        "dy": 0.1,
        "dz": 0.05,
    }
    vol, axes = das_rca_volume(
        rf,
        coords,
        angles_deg=angles,
        tx_centers_mm=tx.element_centers * 1e3,
        rx_centers_mm=rx.element_centers * 1e3,
        rx_length_mm=4.75,
        grid_mm=grid,
        c=C,
        fnum=0.7,
        # Remove the axial bias of the band-limited pulse: the two-way
        # waveform peaks ~half the excitation length after the geometric
        # arrival (symmetric Hann-windowed burst).
        t_offset_s=(t.size - 1) / 2 / fs,
    )
    env = np.abs(hilbert(vol, axis=2))
    peak = _peak_mm(env, axes)
    # Half a wavelength (λ = 0.51 mm) of tolerance on each axis.
    np.testing.assert_allclose(peak, point[0], atol=0.3)
