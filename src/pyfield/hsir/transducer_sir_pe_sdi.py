"""Pulse-echo SDI kernels: combined two-way delta placement (Δδ_pe) and variants.

The pulse-echo response of a (TX patch, RX patch) pair is the time convolution of their
one-way spatial impulse responses, `h_tx ⊛ h_rx`. Each one-way SIR is a trapezoid whose
second time-derivative is four Dirac deltas, so the *product* of the two delta trains is
the raw two-way kernel `Δδ_pe = D²h_tx ⊛ D²h_rx` — 16 deltas per patch pair. The smooth
two-way SIR is recovered by integrating four times, `h_tx ⊛ h_rx = I⁴ Δδ_pe`; here that
I⁴ is **not** done by cumulative sums but applied downstream in Fourier as ÷(jω)⁴, so
these kernels return the *raw* delta train.

Three kernels, all sharing the patch geometry:
- `compute_pe_sdi` — Δδ_pe at every scatterer, `(P, T)`.
- `compute_pe_sdi_summed` — amplitude-weighted sum over scatterers, `(T,)`, for the
  no-attenuation single-trace fast path (skips the `(P, T)` buffer).
- `compute_pe_complete` — the FFT-free reference: splats the already-integrated kernel
  `w = I⁴ v_pe` directly, so the output is the final RF trace.

Each kernel parallelizes over scatterers (`prange` over P); a single field point (a PSF,
`P == 1`) instead parallelizes over TX patches so one point still saturates every core.
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
# PE SDI (combined pulse-echo SDI) — 16 deltas per (m_e, m_r) pair
# ---------------------------------------------------------------------------


@njit(inline="always")
def _place_pe_sdi_deltas(
    out, p, t0, fs, t1e, t2e, t3e, t4e, t1r, t2r, t3r, t4r, weight
):
    """Place the 16 Δδ_pe deltas of one (m_e, m_r) pair into out[p, :].

    The two-way delta train is the product of the TX and RX second-derivative trains:
    each of the 4 TX corners (signs +,−,−,+) times each of the 4 RX corners lands one
    delta at time ``t_e + t_r`` with the product sign. Linear interpolation splits every
    delta across the two adjacent bins (32 writes total). The events are placed at the
    naive ``h_tx ⊛ h_rx`` onset — no extra sample shift (verified against the naive SIR).
    """
    signpos1 = np.float32(1.0)
    signneg1 = np.float32(-1.0)
    k_shift = np.float32(0.0)

    # Unrolled 4×4 loop for Numba performance.
    for i_r in range(4):
        if i_r == 0:
            sign_i = signpos1
            t_r = t1r
        elif i_r == 1:
            sign_i = signneg1
            t_r = t2r
        elif i_r == 2:
            sign_i = signneg1
            t_r = t3r
        else:
            sign_i = signpos1
            t_r = t4r

        # TX corner 0 (+1)
        w1 = signpos1 * sign_i * weight
        t_event = t1e + t_r
        kf = (t_event - t0) * fs + k_shift
        kf_floor = int(np.floor(kf))
        w_ceil = kf - kf_floor
        w_floor = signpos1 - w_ceil  # reuse +1 as one
        out[p, kf_floor] += w1 * w_floor
        out[p, kf_floor + 1] += w1 * w_ceil

        # TX corner 1 (-1)
        w2 = signneg1 * sign_i * weight
        t_event = t2e + t_r
        kf = (t_event - t0) * fs + k_shift
        kf_floor = int(np.floor(kf))
        w_ceil = kf - kf_floor
        w_floor = signpos1 - w_ceil
        out[p, kf_floor] += w2 * w_floor
        out[p, kf_floor + 1] += w2 * w_ceil

        # TX corner 2 (-1)
        w3 = signneg1 * sign_i * weight
        t_event = t3e + t_r
        kf = (t_event - t0) * fs + k_shift
        kf_floor = int(np.floor(kf))
        w_ceil = kf - kf_floor
        w_floor = signpos1 - w_ceil
        out[p, kf_floor] += w3 * w_floor
        out[p, kf_floor + 1] += w3 * w_ceil

        # TX corner 3 (+1)
        w4 = signpos1 * sign_i * weight
        t_event = t4e + t_r
        kf = (t_event - t0) * fs + k_shift
        kf_floor = int(np.floor(kf))
        w_ceil = kf - kf_floor
        w_floor = signpos1 - w_ceil
        out[p, kf_floor] += w4 * w_floor
        out[p, kf_floor + 1] += w4 * w_ceil


@njit(parallel=True, fastmath=True, cache=True)
def _pe_sdi_points(
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
    inv_c,
    t0,
    T,
    fs,
    dt,
):
    """Raw Δδ_pe at each scatterer → (P, T) float64. prange over scatterers.

    Each scatterer owns its own output row, so the parallel loop writes race-free.
    """
    P = points.shape[0]
    M_e = tx_centers.shape[0]
    M_r = rx_centers.shape[0]
    # float64 buffer: the PE weight slope_e·slope_r can reach ~1e20 for sub-sample
    # patches (Dt clamped to dt → slope ~ area/dt², squared in the product). A float32
    # buffer would lose ~1e13 per write to cancellation, leaking DC/ramp (vertical
    # streaks). float64 ULP at 1e20 is ~2e4 (negligible).
    out = np.zeros((P, T), dtype=np.float64)
    for p in prange(P):  # ty: ignore[not-iterable]
        for m_r in range(M_r):
            t1r, t2r, t3r, t4r, slope_r = _patch_corner_times(
                points[p, 0],
                points[p, 1],
                points[p, 2],
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
                    points[p, 0],
                    points[p, 1],
                    points[p, 2],
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

                kstart = (t1e + t1r - t0) * fs
                kend = (t4e + t4r - t0) * fs + 2
                if kstart < 1 or kend > T:
                    print(
                        f"Warning: event outside time grid in point {p}"
                        f"and patch pair (m_e, m_r) = ({m_e}, {m_r})"
                    )
                    print("k_start:", kstart, "k_end:", kend, "T:", T)
                    continue

                _place_pe_sdi_deltas(
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
                )

    # Raw Δδ_pe (= D²h_tx ⊛ D²h_rx); I⁴ applied downstream as ÷(jω)⁴.
    return out


@njit(parallel=True, fastmath=True, cache=True)
def _pe_sdi_patches(
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
    inv_c,
    t0,
    T,
    fs,
    dt,
):
    """Raw Δδ_pe for one field point → (T,) float64. prange over TX patches.

    For a single point (a PSF) `prange` over scatterers would leave every core but one
    idle, so we parallelize over TX patches instead. Each patch writes its own row of an
    ``(M_e, T)`` buffer (race-free), summed at the end.
    """
    M_e = tx_centers.shape[0]
    M_r = rx_centers.shape[0]
    px = point[0]
    py = point[1]
    pz = point[2]
    buf = np.zeros((M_e, T), dtype=np.float64)  # one row per TX patch, race-free.
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
            kstart = (t1e + t1r - t0) * fs
            kend = (t4e + t4r - t0) * fs + 2
            if kstart < 1 or kend > T:
                continue
            _place_pe_sdi_deltas(
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
            )
    return buf.sum(axis=0)


# No cache=True: get_num_threads() is a dynamic global, which numba refuses to cache
# (and the warning would trip pytest's filterwarnings=error). Recompiles per session.
@njit(parallel=True, fastmath=True)
def _pe_sdi_summed(
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
    amps,
    inv_c,
    t0,
    T,
    fs,
    dt,
):
    """Amplitude-weighted sum over scatterers of Δδ_pe → (T,) float64.

    Same deltas as `_pe_sdi_points` but accumulated (× ``amps``) into one trace, so the
    caller skips the ``amps @ (P, T)`` matvec and the ``(P, T)`` buffer. The output is a
    single shared ``(T,)`` trace, so — unlike the other kernels — we cannot give each
    parallel iteration its own output row; instead each thread owns one row of a small
    ``(n_threads, T)`` buffer (a contiguous chunk of scatterers), race-free, reduced
    after the loop. No-attenuation summed path only.
    """
    P = points.shape[0]
    M_e = tx_centers.shape[0]
    M_r = rx_centers.shape[0]
    n_threads = get_num_threads()
    buf = np.zeros((n_threads, T), dtype=np.float64)
    chunk = (P + n_threads - 1) // n_threads
    for c in prange(n_threads):  # ty: ignore[not-iterable]
        start = c * chunk
        end = min(start + chunk, P)
        for p in range(start, end):
            ap = amps[p]
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
                    kstart = (t1e + t1r - t0) * fs
                    kend = (t4e + t4r - t0) * fs + 2
                    if kstart < 1 or kend > T:
                        continue
                    _place_pe_sdi_deltas(
                        buf,
                        c,
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
                        slope_r * slope_e * ap,
                    )
    return buf.sum(axis=0)


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


def compute_pe_sdi(
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
    *,
    tx_eu=None,
    tx_ev=None,
    rx_eu=None,
    rx_ev=None,
    batch_size_points=None,
):
    """Pulse-echo SDI: raw Δδ_pe = D²h_tx ⊛ D²h_rx via combined delta placement.

    Computes the RAW two-way delta train for one RX element (rx patches = patches of that
    element). For a full reception: call once per RX element with the element's rx
    patches. I⁴ = ÷(jω)⁴ is applied downstream in Fourier — there is no cumsum here.

    16 deltas per (m_e, m_r) pair → 32 sample writes. Parallelized over field points,
    except a single point (``P == 1``, e.g. a PSF) which parallelizes over TX patches
    instead (`_pe_sdi_patches`).

    Parameters
    ----------
    points : (P, 3) float32
        Scatterer positions in metres.
    tx_centers : (M_e, 3) float32
        TX patch centre coordinates in metres.
    tx_wx : (M_e,) float32
        TX patch width in x (metres).
    tx_wy : (M_e,) float32
        TX patch width in y (metres).
    tx_apod : (M_e,) float32
        TX apodization weight per patch.
    tx_delays : (M_e,) float32
        TX delay per patch (seconds).
    rx_centers : (M_r, 3) float32
        RX patch centre coordinates in metres (one element's patches).
    rx_wx : (M_r,) float32
        RX patch width in x (metres).
    rx_wy : (M_r,) float32
        RX patch width in y (metres).
    rx_apod : (M_r,) float32
        RX apodization weight per patch.
    rx_delays : (M_r,) float32
        RX delay per patch (seconds).
    inv_c : float
        Inverse speed of sound 1/c (s/m).
    t0 : float
        Start of time grid (seconds).
    T : int
        Number of time samples.
    fs : float
        Sampling frequency (Hz).
    dt : float
        Time step 1/fs (seconds).
    tx_eu, tx_ev : (M_e, 3) numpy.ndarray or None, default None
        TX patch tangent vectors; None → flat-patch identity tangents.
    rx_eu, rx_ev : (M_r, 3) numpy.ndarray or None, default None
        RX patch tangent vectors; None → flat-patch identity tangents.
    batch_size_points : int or None
        Chunk P into batches at Python level. None = no batching.

    Returns
    -------
    (P, T) float32 ndarray
        Raw two-way pulse-echo delta train Δδ_pe at each scatterer.
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
    P = points.shape[0]

    # Single point (e.g. PSF): prange over P is useless — parallelize over patches.
    if P == 1:
        row = _pe_sdi_patches(
            points[0],
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
            inv_c,
            t0,
            T,
            fs,
            dt,
        )
        return row.reshape(1, T).astype(np.float32)

    if batch_size_points is None or batch_size_points >= P:
        return _pe_sdi_points(
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
            inv_c,
            t0,
            T,
            fs,
            dt,
        ).astype(np.float32)

    out = np.zeros((P, T), dtype=np.float32)
    for start in range(0, P, batch_size_points):
        end = min(start + batch_size_points, P)
        out[start:end] = _pe_sdi_points(
            points[start:end],
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
            inv_c,
            t0,
            T,
            fs,
            dt,
        )
    return out


def compute_pe_sdi_summed(
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
    amps,
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
    """Amplitude-weighted sum over scatterers of Δδ_pe → ``(T,)`` float32.

    Equivalent to ``amps @ compute_pe_sdi(...)`` but accumulates inside the kernel,
    avoiding the ``(P, T)`` buffer and the matvec. Summed, no-attenuation fast path.

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
    amps : (P,) numpy.ndarray
        Scattering amplitude per scatterer.
    inv_c : float
        Inverse speed of sound 1/c (s/m).
    t0 : float
        Start of the time grid (seconds).
    T : int
        Number of time samples.
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
    (T,) numpy.ndarray
        Amplitude-summed Δδ_pe delta train (float32).
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
    amps = np.asarray(amps, dtype=np.float32)
    return _pe_sdi_summed(
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
        amps,
        inv_c,
        t0,
        T,
        fs,
        dt,
    ).astype(np.float32)


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
