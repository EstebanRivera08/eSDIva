"""Paired pulse-echo SDI kernel — pedagogic reference, cost ∝ M_tx·M_rx.

Convolving the TX and RX corner-delta trains analytically gives 16 deltas per
(TX patch, RX patch) pair, ``Δδ_pe = D²h_tx ⊛ D²h_rx``. `compute_pe_complete` pushes the
four integrations onto the drive once (``w = I⁴ v_pe``) and splats a shifted, scaled
copy of ``w`` at every corner event: no FFT, no cumulative sum, the output is the RF.
Exact but quadratic in the patch count; used by `ReceptionPaired` for teaching and
cross-checks. The spectral kernels evaluate the same RF in linear cost.
"""

import numpy as np
from numba import get_num_threads, njit, prange

from .helpers import _compute_rectangle_SIR_params, _prep_pe_arrays


@njit(inline="always")
def _patch_corner_times(
    px, py, pz, cx, cy, cz, eu0, eu1, eu2, ev0, ev1, ev2, wx, wy, inv_c, apod, delay, dt
):
    """Trapezoid corner times + slope of one patch seen from one field point.

    Projects the patch-to-point direction onto the patch local frame, then returns the
    trapezoidal SIR corners and its rising slope (= plateau height / rise time). The
    second derivative of that trapezoid is the delta train the SDI kernels place, scaled
    by this slope. ``slope == 0.0`` flags a degenerate patch (point on the patch, or a
    sub-threshold plateau) the caller should skip.
    """
    dx = px - cx
    dy = py - cy
    dz = pz - cz
    dist = np.sqrt(dx * dx + dy * dy + dz * dz)
    if dist < np.float32(1e-12):
        return np.float32(0.0), np.float32(0.0), np.float32(0.0), np.float32(0.0), 0.0
    inv_dist = np.float32(1.0) / dist
    xp = (dx * eu0 + dy * eu1 + dz * eu2) * inv_dist
    yp = (dx * ev0 + dy * ev1 + dz * ev2) * inv_dist
    t1, t2, t3, t4, h_max = _compute_rectangle_SIR_params(
        wx, wy, xp, yp, dist, inv_c, apod, delay, dt
    )
    if h_max < np.float32(1e-6):
        return t1, t2, t3, t4, 0.0
    return t1, t2, t3, t4, h_max / (t2 - t1)


# ---------------------------------------------------------------------------
# Complete SDI PE: splat w = I⁴ v_pe per pair (no FFT, no cumsum).
# Each of the 16 corner events convolves a 2-bin linear-interp delta with w, i.e. adds a
# shifted, scaled copy of w. Output is the final per-element RF trace (still ×scale and
# amplitude-weighted by the caller).
# ---------------------------------------------------------------------------


@njit(inline="always")
def _add_shifted_w(out, row, kf, gain, w, nfft):
    """Add ``gain``-scaled, 2-bin-interpolated copy of ``w`` at index ``kf``, wrapped mod nfft.

    Circular placement: the event at continuous index ``kf`` deposits ``w`` rolled to
    ``kf`` (plus the fractional ceil tap). ``w`` is the full-length (``nfft``) integrated
    exc/IR kernel — the zero-phase I⁴ filter is delocalized, so the convolution must be
    circular (then sliced to ``pe_T``) to match the FFT path exactly.
    """
    kf_floor = int(np.floor(kf))
    w_ceil = kf - kf_floor
    g_floor = gain * (np.float32(1.0) - w_ceil)
    g_ceil = gain * w_ceil
    base = kf_floor % nfft
    base1 = base + 1
    if base1 >= nfft:
        base1 -= nfft
    for li in range(nfft):
        wl = w[li]
        k = base + li
        if k >= nfft:
            k -= nfft
        out[row, k] += g_floor * wl
        k1 = base1 + li
        if k1 >= nfft:
            k1 -= nfft
        out[row, k1] += g_ceil * wl


@njit(inline="always")
def _place_pe_complete(
    out, row, t0, fs, t1e, t2e, t3e, t4e, t1r, t2r, t3r, t4r, weight, w, nfft
):
    """Splat w for the 16 PE corner events of one (m_e, m_r) pair into out[row, :]."""
    signp = np.float32(1.0)
    signn = np.float32(-1.0)
    for i_r in range(4):
        if i_r == 0:
            sr = signp
            tr = t1r
        elif i_r == 1:
            sr = signn
            tr = t2r
        elif i_r == 2:
            sr = signn
            tr = t3r
        else:
            sr = signp
            tr = t4r
        gp = signp * sr * weight
        gn = signn * sr * weight
        _add_shifted_w(out, row, (t1e + tr - t0) * fs, gp, w, nfft)
        _add_shifted_w(out, row, (t2e + tr - t0) * fs, gn, w, nfft)
        _add_shifted_w(out, row, (t3e + tr - t0) * fs, gn, w, nfft)
        _add_shifted_w(out, row, (t4e + tr - t0) * fs, gp, w, nfft)


@njit(parallel=True, fastmath=True, cache=True)
def _pe_complete_points(
    points,
    tx_centers,
    tx_wx,
    tx_wy,
    tx_tangents,
    tx_apod,
    tx_delays,
    rx_centers,
    rx_wx,
    rx_wy,
    rx_tangents,
    rx_apod,
    rx_delays,
    w,
    inv_c,
    t0,
    T,
    fs,
    dt,
):
    """Complete-SDI PE RF per scatterer → (P, T) float64. prange over scatterers.

    Splats ``w`` (full-length ``nfft`` integrated exc/IR kernel) circularly per pair,
    then returns the ``[:T]`` window — the FFT path's circular convolution done by hand.
    Exact but O(nfft) per pair, hence the slow reference path. Each scatterer owns its
    own output row.
    """
    P = points.shape[0]
    M_e = tx_centers.shape[0]
    M_r = rx_centers.shape[0]
    nfft = w.shape[0]
    out = np.zeros((P, nfft), dtype=np.float64)
    for p in prange(P):  # ty: ignore[not-iterable]
        px = points[p, 0]
        py = points[p, 1]
        pz = points[p, 2]
        for m_r in range(M_r):
            t1r, t2r, t3r, t4r, slope_r = _patch_corner_times(
                px,
                py,
                pz,
                rx_centers[m_r, 0],
                rx_centers[m_r, 1],
                rx_centers[m_r, 2],
                rx_tangents[m_r, 0],
                rx_tangents[m_r, 1],
                rx_tangents[m_r, 2],
                rx_tangents[m_r, 3],
                rx_tangents[m_r, 4],
                rx_tangents[m_r, 5],
                rx_wx[m_r],
                rx_wy[m_r],
                inv_c,
                rx_apod[m_r],
                rx_delays[m_r],
                dt,
            )
            if slope_r == 0.0:
                continue
            for m_e in range(M_e):
                t1e, t2e, t3e, t4e, slope_e = _patch_corner_times(
                    px,
                    py,
                    pz,
                    tx_centers[m_e, 0],
                    tx_centers[m_e, 1],
                    tx_centers[m_e, 2],
                    tx_tangents[m_e, 0],
                    tx_tangents[m_e, 1],
                    tx_tangents[m_e, 2],
                    tx_tangents[m_e, 3],
                    tx_tangents[m_e, 4],
                    tx_tangents[m_e, 5],
                    tx_wx[m_e],
                    tx_wy[m_e],
                    inv_c,
                    tx_apod[m_e],
                    tx_delays[m_e],
                    dt,
                )
                if slope_e == 0.0:
                    continue
                _place_pe_complete(
                    out,
                    p,
                    t0,
                    fs,
                    t1e,
                    t2e,
                    t3e,
                    t4e,
                    t1r,
                    t2r,
                    t3r,
                    t4r,
                    slope_r * slope_e,
                    w,
                    nfft,
                )
    return out[:, :T]


@njit(parallel=True, fastmath=True, cache=True)
def _pe_complete_patches(
    point,
    tx_centers,
    tx_wx,
    tx_wy,
    tx_tangents,
    tx_apod,
    tx_delays,
    rx_centers,
    rx_wx,
    rx_wy,
    rx_tangents,
    rx_apod,
    rx_delays,
    w,
    inv_c,
    t0,
    T,
    fs,
    dt,
):
    """Complete-SDI PE RF for one field point → (T,) float64. prange over TX patches.

    With few scatterers `prange` over scatterers starves the cores, yet the cost per
    point is the full ``16·M_e·M_r`` pair sweep, each pair splatting the length-``nfft``
    kernel ``w`` — the analytic wall the complete form pays. Here each TX patch writes
    its own row of an ``(M_e, nfft)`` buffer (race-free), so one point saturates the box.
    Each pair splats ``w`` circularly (the I⁴ filter is zero-phase, delocalized); the
    summed buffer is sliced to ``[:T]``.
    """
    M_e = tx_centers.shape[0]
    M_r = rx_centers.shape[0]
    nfft = w.shape[0]
    px = point[0]
    py = point[1]
    pz = point[2]
    buf = np.zeros((M_e, nfft), dtype=np.float64)  # one row per TX patch, race-free.
    for m_e in prange(M_e):  # ty: ignore[not-iterable]
        t1e, t2e, t3e, t4e, slope_e = _patch_corner_times(
            px,
            py,
            pz,
            tx_centers[m_e, 0],
            tx_centers[m_e, 1],
            tx_centers[m_e, 2],
            tx_tangents[m_e, 0],
            tx_tangents[m_e, 1],
            tx_tangents[m_e, 2],
            tx_tangents[m_e, 3],
            tx_tangents[m_e, 4],
            tx_tangents[m_e, 5],
            tx_wx[m_e],
            tx_wy[m_e],
            inv_c,
            tx_apod[m_e],
            tx_delays[m_e],
            dt,
        )
        if slope_e == 0.0:
            continue
        for m_r in range(M_r):
            t1r, t2r, t3r, t4r, slope_r = _patch_corner_times(
                px,
                py,
                pz,
                rx_centers[m_r, 0],
                rx_centers[m_r, 1],
                rx_centers[m_r, 2],
                rx_tangents[m_r, 0],
                rx_tangents[m_r, 1],
                rx_tangents[m_r, 2],
                rx_tangents[m_r, 3],
                rx_tangents[m_r, 4],
                rx_tangents[m_r, 5],
                rx_wx[m_r],
                rx_wy[m_r],
                inv_c,
                rx_apod[m_r],
                rx_delays[m_r],
                dt,
            )
            if slope_r == 0.0:
                continue
            _place_pe_complete(
                buf,
                m_e,
                t0,
                fs,
                t1e,
                t2e,
                t3e,
                t4e,
                t1r,
                t2r,
                t3r,
                t4r,
                slope_r * slope_e,
                w,
                nfft,
            )
    return buf.sum(axis=0)[:T]


def compute_pe_complete(
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
    w,
    inv_c,
    t0,
    T,
    fs,
    dt,
    *,
    tx_eu=None,
    tx_ev=None,
    rx_eu=None,
    rx_ev=None,
):
    """Complete SDI PE: splat ``w = I⁴ v_pe`` per pair → ``(P, T)`` float32.

    Cumsum-free evaluation of ``p_pe = Σ_i Σ_j a_i a_j w(t − τ_i − τ_j)``: each of the
    16 corner events per (m_e, m_r) pair adds a 2-bin-interpolated, slope-weighted copy
    of ``w``, wrapped mod ``len(w)``. ``w`` is the FULL-length (``nfft``) integrated
    exc/IR kernel ``I⁴(e ⊛ ir_tx ⊛ ir_rx)``; the circular convolution is sliced to ``T``
    (= ``pe_T``), reproducing the FFT path exactly. The caller applies ``scale`` +
    amplitude weighting. Exact but O(nfft) per pair (slow reference path).

    Parallelizes over scatterers when ``P ≥ n_threads``; with fewer points it loops the
    points and parallelizes each over TX patches instead, so even a single point-spread
    scatterer keeps every core busy on the ``16·M_e·M_r`` sweep.

    Parameters
    ----------
    points : (P, 3) numpy.ndarray
        Scatterer positions in metres.
    tx_centers : (M_e, 3) numpy.ndarray
        TX patch centres in metres.
    tx_wx : (M_e,) numpy.ndarray
        TX patch width in x (metres).
    tx_wy : (M_e,) numpy.ndarray
        TX patch width in y (metres).
    tx_apod : (M_e,) numpy.ndarray
        TX apodization weight per patch.
    tx_delays : (M_e,) numpy.ndarray
        TX delay per patch (seconds).
    rx_centers : (M_r, 3) numpy.ndarray
        RX patch centres in metres (one element's patches).
    rx_wx : (M_r,) numpy.ndarray
        RX patch width in x (metres).
    rx_wy : (M_r,) numpy.ndarray
        RX patch width in y (metres).
    rx_apod : (M_r,) numpy.ndarray
        RX apodization weight per patch.
    rx_delays : (M_r,) numpy.ndarray
        RX delay per patch (seconds).
    w : (nfft,) numpy.ndarray
        Integrated exc/IR kernel ``I⁴(e ⊛ ir_tx ⊛ ir_rx)``; ``len(w)`` sets the
        circular-convolution period.
    inv_c : float
        Inverse speed of sound 1/c (s/m).
    t0 : float
        Start of the time grid (seconds).
    T : int
        Number of output time samples (``pe_T``); the circular result is sliced to it.
    fs : float
        Sampling frequency (Hz).
    dt : float
        Time step 1/fs (seconds).
    tx_eu, tx_ev : (M_e, 3) numpy.ndarray or None, default None
        TX patch tangent vectors; None → flat-patch identity tangents.
    rx_eu, rx_ev : (M_r, 3) numpy.ndarray or None, default None
        RX patch tangent vectors; None → flat-patch identity tangents.

    Returns
    -------
    (P, T) numpy.ndarray
        Per-scatterer complete-SDI pulse-echo RF (float32), before ``scale``/amps.
    """
    (
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
    ) = _prep_pe_arrays(
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
    )
    w = np.ascontiguousarray(np.asarray(w, dtype=np.float64))
    P = points.shape[0]

    # Few scatterers: prange over P starves the cores (only P run), but each point
    # carries the full 16·M_e·M_r pair sweep. Point-parallel wall ≈ W (one point's work,
    # P cores busy); patch-parallel looped ≈ P·W/n_threads — cheaper exactly when
    # P < n_threads. Below that crossover, loop the points and parallelize each over TX
    # patches so even a single scatterer (PSF) saturates the box.
    if P < get_num_threads():
        out = np.zeros((P, T), dtype=np.float64)
        for p in range(P):
            out[p] = _pe_complete_patches(
                points[p],
                tx_centers,
                tx_wx,
                tx_wy,
                tx_tangents,
                tx_apod,
                tx_delays,
                rx_centers,
                rx_wx,
                rx_wy,
                rx_tangents,
                rx_apod,
                rx_delays,
                w,
                inv_c,
                t0,
                T,
                fs,
                dt,
            )
        return out.astype(np.float32)

    return _pe_complete_points(
        points,
        tx_centers,
        tx_wx,
        tx_wy,
        tx_tangents,
        tx_apod,
        tx_delays,
        rx_centers,
        rx_wx,
        rx_wy,
        rx_tangents,
        rx_apod,
        rx_delays,
        w,
        inv_c,
        t0,
        T,
        fs,
        dt,
    ).astype(np.float32)
