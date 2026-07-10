"""Tests for das_dw_volume — 3-D diverging-wave (virtual source) delay-and-sum.

Two levels: (1) synthetic RF with a delta placed at the analytic two-way
delay of a known point — verifies the spherical-wavefront delay model, t0
handling and interpolation localize the point exactly; (2) end-to-end
against ReceptionSDI — verifies the DAS time convention matches the
simulator's beam-axis-referenced ``t0`` (the real regression risk).
"""

import warnings

import numpy as np
import pytest
from scipy.signal import hilbert

from pyfield.beamforming import das_dw_volume
from pyfield.reception import ReceptionSDI
from pyfield.transducers import MatrixArrayTransducer

C = 1540.0


def _matrix_centers(n, pitch_mm):
    """Element centres (mm) of an n×n matrix centred at the origin."""
    idx = (np.arange(n) - (n - 1) / 2) * pitch_mm
    gx, gy = np.meshgrid(idx, idx, indexing="ij")
    return np.column_stack([gx.ravel(), gy.ravel(), np.zeros(n * n)])


def _delta_rf(point_mm, vs_mm, el_mm, t0_ev, dt, nt=4000):
    """RF with a split-sample delta at each channel's analytic two-way delay.

    Uses the same physics as the beamformer's docstring: spherical-wavefront
    arrival ``(|r − r_vs| − max_e|r_e − r_vs|)/c`` plus the direct return
    path ``|r − r_e|/c``.
    """
    p = np.asarray(point_mm, float) * 1e-3
    vs = np.asarray(vs_mm, float) * 1e-3
    el = el_mm * 1e-3
    rf = np.zeros((vs.shape[0], el.shape[0], nt), np.float32)
    for e in range(vs.shape[0]):
        d_max = np.linalg.norm(el - vs[e], axis=1).max()
        t_tx = (np.linalg.norm(p - vs[e]) - d_max) / C
        for r in range(el.shape[0]):
            t_rx = np.linalg.norm(p - el[r]) / C
            s = (t_tx + t_rx - t0_ev[e]) / dt
            i0 = int(np.floor(s))
            rf[e, r, i0] += 1.0 - (s - i0)
            rf[e, r, i0 + 1] += s - i0
    return rf


def _peak_mm(volume, axes):
    ix, iy, iz = np.unravel_index(np.argmax(np.abs(volume)), volume.shape)
    return axes["x_mm"][ix], axes["y_mm"][iy], axes["z_mm"][iz]


def test_point_localized_synthetic():
    """Delta at the analytic delay must beamform to the true voxel."""
    point = np.array([1.1, -0.7, 12.0])
    vs = np.array([[1.5, 0.0, -20.0], [0.0, 0.0, -20.0], [-1.5, 1.5, -20.0]])
    dt = 1.0 / 50e6
    el_mm = _matrix_centers(11, 0.9)
    # Distinct nonzero per-event t0 exercises the t0_per_event handling.
    t0_ev = np.array([2e-6, 2.5e-6, 3e-6])

    rf = _delta_rf(point, vs, el_mm, t0_ev, dt)
    grid = {
        "x_extent": [point[0] - 1.5, point[0] + 1.5],
        "y_extent": [point[1] - 1.5, point[1] + 1.5],
        "z_extent": [point[2] - 2.0, point[2] + 2.0],
        "dx": 0.1,
        "dy": 0.1,
        "dz": 0.05,
    }
    vol, axes = das_dw_volume(
        rf,
        {"t0_per_event": t0_ev, "t0": t0_ev[0], "dt": dt},
        virtual_sources_mm=vs,
        elem_centers_mm=el_mm,
        grid_mm=grid,
        c=C,
        fnum=0.8,
    )
    peak = _peak_mm(vol, axes)
    np.testing.assert_allclose(peak, point, atol=0.11)  # within one voxel


def test_coherence_weight_keeps_peak_suppresses_offpeak():
    """CF weighting must not move the point but must lower off-peak clutter."""
    point = np.array([0.0, 0.0, 12.0])
    vs = np.array([[0.0, 0.0, -20.0], [1.5, 0.0, -20.0]])
    dt = 1.0 / 50e6
    el_mm = _matrix_centers(11, 0.9)
    t0_ev = np.zeros(2)
    rf = _delta_rf(point, vs, el_mm, t0_ev, dt)
    grid = {
        "x_extent": [-2.0, 2.0],
        "y_extent": [-0.1, 0.1],
        "z_extent": [11.0, 13.0],
        "dx": 0.05,
        "dy": 0.2,
        "dz": 0.05,
    }
    kw = dict(
        virtual_sources_mm=vs,
        elem_centers_mm=el_mm,
        grid_mm=grid,
        c=C,
        fnum=0.8,
    )
    coords = {"t0_per_event": t0_ev, "t0": 0.0, "dt": dt}
    plain, axes = das_dw_volume(rf, coords, **kw)
    cf, _ = das_dw_volume(rf, coords, coherence_weight=True, **kw)

    assert _peak_mm(cf, axes) == _peak_mm(plain, axes)
    x = axes["x_mm"]
    iz = np.argmin(np.abs(axes["z_mm"] - 12.0))
    off = np.argmin(np.abs(x - 1.0))  # 1 mm off-axis: sidelobe region.
    rel_plain = abs(plain[off, 0, iz]) / np.abs(plain).max()
    rel_cf = abs(cf[off, 0, iz]) / np.abs(cf).max()
    assert rel_cf < rel_plain


def test_event_mismatch_raises():
    rf = np.zeros((2, 4, 10), np.float32)
    with pytest.raises(ValueError, match="virtual sources"):
        das_dw_volume(
            rf,
            {"t0": 0.0, "dt": 1e-8},
            virtual_sources_mm=[[0.0, 0.0, -20.0]],
            elem_centers_mm=np.zeros((4, 3)),
            grid_mm={
                "x_extent": [0, 1],
                "y_extent": [0, 1],
                "z_extent": [1, 2],
                "dx": 0.5,
                "dy": 0.5,
                "dz": 0.5,
            },
        )


def test_point_localized_end_to_end():
    """Full chain: ReceptionSDI diverging-wave sequence → DAS → peak at the
    scatterer.

    This is the convention guard: it fails if the beamformer's time origin
    disagrees with the simulator's beam-axis-referenced ``t0`` or if the
    virtual-source delay sign flips.
    """
    fc, fs = 3e6, 50e6
    n = 8

    def _matrix():
        return MatrixArrayTransducer(
            n_elements_x=n,
            n_elements_y=n,
            element_width_mm=0.55,
            element_height_mm=0.55,
            kerf_x_mm=0.05,
            kerf_y_mm=0.05,
            no_sub_x=1,
            no_sub_y=1,
            frequency_Hz=fc,
        )

    # Separate TX/RX instances: sequence_rf refuses a shared object because
    # the per-event TX delays would also time-shift every receive channel.
    mat = _matrix()
    t = np.arange(0, 2 / fc, 1 / fs)
    exc = (np.sin(2 * np.pi * fc * t) * np.hanning(t.size)).astype(np.float32)
    sim = ReceptionSDI(mat, _matrix(), c=C, fs=fs, excitation=exc, verbose=False)

    point = np.array([[0.8, -0.6, 12.0]], dtype=np.float32)
    vs = np.array([[1.0, 0.0, -15.0], [0.0, 0.0, -15.0], [-0.5, 1.0, -15.0]])
    events = []
    for v in vs * 1e-3:
        d = np.linalg.norm(mat.element_centers - v, axis=1) / C
        events.append(
            {
                "delays": (d - d.min()).astype(np.float32),
                "apodization": np.ones(n * n, np.float32),
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
    vol, axes = das_dw_volume(
        rf,
        coords,
        virtual_sources_mm=vs,
        elem_centers_mm=mat.element_centers * 1e3,
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
    # Within ~1 voxel: a looser 0.3 mm once masked a per-channel RX delay
    # corruption (shared TX/RX object), so keep this tight.
    np.testing.assert_allclose(peak, point[0], atol=0.15)
