"""Shared helpers for the far-field rectangular-patch SIR kernels.

Holds the geometry and array-prep pieces every SIR engine in this package needs, so the
one-way (`farfield_rect_patch.py`) and pulse-echo (`transducer_sir_pe_sdi.py`) kernels
do not each carry their own copy:

- `_compute_rectangle_SIR_params` — the trapezoidal SIR of a single rectangular patch
  (corner times + plateau height), the atom every kernel evaluates per patch.
- `identity_tangents` / `pack_tangents` — build and pack the per-patch local frame
  (in-plane unit vectors) the kernels project the field-point direction onto.
- `_prep_pe_arrays` — cast the pulse-echo TX/RX patch arrays to the float32 layout the
  Numba kernels expect.
"""

import numpy as np
from numba import njit

# 1/2π as float32: the trapezoid plateau scales the patch area by 1/2π (the SIR
# normalisation). float32 keeps the constant in the same precision as the patch
# coordinates the kernels run in, so no silent float64 promotion mid-kernel.
_inv_2pi = np.float32(1.0 / (2.0 * np.pi))


@njit(inline="always")
def _compute_rectangle_SIR_params(wx, wy, ux, uy, dist, inv_c, apod, delay, dt):
    """Trapezoidal SIR corner times and plateau height for one rectangular patch.

    The spatial impulse response of a small rectangle, seen from a field point in the
    far field, is a trapezoid in time: it rises from `t1`, plateaus between `t2` and
    `t3`, and falls to zero at `t4`. The plateau height `h_max` carries the patch area.

    Parameters
    ----------
    wx, wy : float
        Patch widths in the two in-plane directions (metres).
    ux, uy : float
        Direction cosines from patch centre to field point along the patch in-plane axes
        (dimensionless, already divided by distance) — the `u`/`v` projections of the
        unit vector, not position deltas.
    dist : float
        Distance from patch centre to field point (metres).
    inv_c : float
        Inverse speed of sound 1/c (s/m).
    apod : float
        Apodization weight for this patch.
    delay : float
        Time delay applied to this patch (seconds).
    dt : float
        Time step 1/fs (seconds); the two edge-crossing durations are clamped to it so a
        sub-sample patch still spans at least one bin.

    Returns
    -------
    t1, t2, t3, t4 : float
        Trapezoid corner times (seconds): onset, plateau start, plateau end, offset.
    h_max : float
        Trapezoid plateau height (the patch-area-weighted SIR amplitude).
    """
    xp_abs = abs(ux) * wx * inv_c
    yp_abs = abs(uy) * wy * inv_c
    Dt1 = min(xp_abs, yp_abs)  # shorter edge crossing.
    Dt2 = max(xp_abs, yp_abs)  # longer edge crossing.
    if Dt1 < dt:
        Dt1 = dt
    if Dt2 < dt:
        Dt2 = dt
    area = (wx * wy * _inv_2pi) / dist
    t1 = dist * inv_c - 0.5 * (Dt1 + Dt2) + delay  # first corner time-of-flight.
    t2 = t1 + Dt1
    t3 = t1 + Dt2
    t4 = t1 + Dt1 + Dt2
    h_max = area * apod / Dt2
    return t1, t2, t3, t4, h_max


def identity_tangents(M):
    """Flat-patch local frames for M patches: u = (1,0,0), v = (0,1,0).

    Used when a transducer surface is planar and its patches share the global x/y axes
    as their in-plane directions, so no per-patch tilt has to be supplied.

    Parameters
    ----------
    M : int
        Number of patches.

    Returns
    -------
    eu : (M, 3) numpy.ndarray
        Per-patch u unit vectors, all ``(1, 0, 0)`` (float32).
    ev : (M, 3) numpy.ndarray
        Per-patch v unit vectors, all ``(0, 1, 0)`` (float32).
    """
    eu = np.zeros((M, 3), dtype=np.float32)
    ev = np.zeros((M, 3), dtype=np.float32)
    eu[:, 0] = 1.0
    ev[:, 1] = 1.0
    return eu, ev


def pack_tangents(eu, ev):
    """Pack (M,3) in-plane unit-vector pairs into one contiguous (M, 6) float32 array.

    The kernels read the patch frame as a single contiguous row per patch (columns 0-2
    the u-tangent, 3-5 the v-tangent); fewer array arguments also eases Numba's parfor
    alias analysis.

    Parameters
    ----------
    eu : (M, 3) numpy.ndarray
        Per-patch u (in-plane width direction) unit vectors.
    ev : (M, 3) numpy.ndarray
        Per-patch v (in-plane height direction) unit vectors.

    Returns
    -------
    (M, 6) numpy.ndarray
        Contiguous float32 patch frames, u-tangent then v-tangent per row.
    """
    tangents = np.empty((eu.shape[0], 6), dtype=np.float32)
    tangents[:, :3] = eu
    tangents[:, 3:] = ev
    return np.ascontiguousarray(tangents)


def _prep_pe_arrays(
    points,
    tx_centers,
    tx_wx,
    tx_wy,
    tx_apod,
    tx_delays,
    rx_centers,
    rx_wx,
    rx_wy,
    rx_apod,
    rx_delays,
    inv_c,
    t0,
    T,
    fs,
    dt,
    tx_eu,
    tx_ev,
    rx_eu,
    rx_ev,
):
    """Cast pulse-echo TX/RX patch arrays to float32 and pack their tangent frames.

    The Numba pulse-echo kernels run in float32; this returns every patch array in that
    dtype, with TX and RX local frames packed to (M, 6) (flat-patch identity frames when
    none are supplied). Scalars are returned as plain Python floats/ints.

    Returns the kernel-ready tuple ``(points, tx_centers, tx_wx, tx_wy, tx_apod,
    tx_delays, tx_tangents, rx_centers, rx_wx, rx_wy, rx_apod, rx_delays, rx_tangents,
    inv_c, t0, fs, dt, T)``.
    """
    points = np.asarray(points, dtype=np.float32)
    tx_centers = np.asarray(tx_centers, dtype=np.float32)
    tx_wx = np.asarray(tx_wx, dtype=np.float32)
    tx_wy = np.asarray(tx_wy, dtype=np.float32)
    tx_apod = np.asarray(tx_apod, dtype=np.float32)
    tx_delays = np.asarray(tx_delays, dtype=np.float32)
    rx_centers = np.asarray(rx_centers, dtype=np.float32)
    rx_wx = np.asarray(rx_wx, dtype=np.float32)
    rx_wy = np.asarray(rx_wy, dtype=np.float32)
    rx_apod = np.asarray(rx_apod, dtype=np.float32)
    rx_delays = np.asarray(rx_delays, dtype=np.float32)
    inv_c, t0, fs, dt, T = float(inv_c), float(t0), float(fs), float(dt), int(T)
    M_e = tx_centers.shape[0]
    M_r = rx_centers.shape[0]
    if tx_eu is None or tx_ev is None:
        tx_eu, tx_ev = identity_tangents(M_e)
    if rx_eu is None or rx_ev is None:
        rx_eu, rx_ev = identity_tangents(M_r)
    tx_tangents = pack_tangents(
        np.asarray(tx_eu, dtype=np.float32), np.asarray(tx_ev, dtype=np.float32)
    )
    rx_tangents = pack_tangents(
        np.asarray(rx_eu, dtype=np.float32), np.asarray(rx_ev, dtype=np.float32)
    )
    return (
        points,
        tx_centers,
        tx_wx,
        tx_wy,
        tx_apod,
        tx_delays,
        tx_tangents,
        rx_centers,
        rx_wx,
        rx_wy,
        rx_apod,
        rx_delays,
        rx_tangents,
        inv_c,
        t0,
        fs,
        dt,
        T,
    )
