"""Pulse-echo SDI kernels: combined two-way delta placement (Δδ_pe) and variants.

Builds the pulse-echo SIR from the product of the TX/RX second-derivative delta
trains (16 deltas per patch pair). Provides the base kernel (`compute_pe_sdi`), an
amplitude-summed variant (`compute_pe_sdi_summed`) that accumulates over scatterers
inside the kernel, and the FFT-free complete-SDI reference (`compute_pe_complete`).
"""

import numpy as np
from numba import get_num_threads, njit, prange

_inv_2pi = np.float32(1.0 / (2.0 * np.pi))


# ---------------------------------------------------------------------------
# Rectangle SIR params — copied from farfield_rect_patch.py to keep this
# module independent of the main SIR engine.
# ---------------------------------------------------------------------------


@njit(inline="always")
def _compute_rectangle_SIR_params(wx, wy, dx, dy, dist, inv_c, apod, delay, dt):
    """Trapezoidal SIR corner times and plateau height for one rectangular patch."""
    xp_abs = abs(dx) * wx * inv_c
    yp_abs = abs(dy) * wy * inv_c
    Dt1 = min(xp_abs, yp_abs)
    Dt2 = max(xp_abs, yp_abs)
    if Dt1 < dt:
        Dt1 = dt
    if Dt2 < dt:
        Dt2 = dt
    area = (wx * wy * _inv_2pi) / dist
    t1 = dist * inv_c - 0.5 * (Dt1 + Dt2) + delay
    t2 = t1 + Dt1
    t3 = t1 + Dt2
    t4 = t1 + Dt1 + Dt2
    h_max = area * apod / Dt2
    return t1, t2, t3, t4, h_max


@njit(inline="always")
def _patch_corner_times(
    px, py, pz, cx, cy, cz, eu0, eu1, eu2, ev0, ev1, ev2, wx, wy, inv_c, apod, delay, dt
):
    """Trapezoid corner times + slope of one patch seen from one field point.

    Returns ``(t1, t2, t3, t4, slope)``; ``slope == 0.0`` flags a degenerate patch
    (point on the patch, or sub-threshold plateau) the caller should skip. Shared
    geometry path for the summed / patch-parallel / complete PE kernels below.
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


def _pack_tangents(eu, ev):
    """Pack (M,3) tangent pairs into contiguous (M, 6) float32 for Numba kernels."""
    tangents = np.empty((eu.shape[0], 6), dtype=np.float32)
    tangents[:, :3] = eu
    tangents[:, 3:] = ev
    return np.ascontiguousarray(tangents)


def _identity_tangents(M):
    """Return flat-patch tangent tangents: eu=(1,0,0), ev=(0,1,0) for M patches."""
    eu = np.zeros((M, 3), dtype=np.float32)
    ev = np.zeros((M, 3), dtype=np.float32)
    eu[:, 0] = 1.0
    ev[:, 1] = 1.0
    return eu, ev


def _prepare_arrays(points, centers, wx, wy, apod, delays):
    """Cast all patch arrays to float32 for Numba kernels."""
    return (
        np.asarray(points, dtype=np.float32),
        np.asarray(centers, dtype=np.float32),
        np.asarray(wx, dtype=np.float32),
        np.asarray(wy, dtype=np.float32),
        np.asarray(apod, dtype=np.float32),
        np.asarray(delays, dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# PE SDI (combined pulse-echo SDI) — 16 deltas per (m_e, m_r) pair
# ---------------------------------------------------------------------------


@njit(inline="always")
def _place_pe_sdi_deltas(
    out, p, t0, fs, t1e, t2e, t3e, t4e, t1r, t2r, t3r, t4r, weight
):
    """Place 16 PE SDI deltas for one (m_e, m_r) pair into out[p, :].

    Each of 4 TX corners × 4 RX corners produces one delta at t_e + t_r.
    Linear interpolation splits each delta across two adjacent bins (32 writes).
    """
    # Static sign arrays: +1, -1, -1, +1 for corners 1–4.
    signpos1 = np.float32(1.0)
    signneg1 = np.float32(-1.0)
    # Combined-kernel placement: zeta = d2h_e ⊛ d2h_r, one cumsum → Dh_pe.
    # Discrete-conv index adds: event lands at floor((t_e+t_r-pe_t0)*fs), matching
    # naive h_tx⊛h_rx onset. No extra shift — single-SDI h is already correctly
    # timed (verified: Emission/Reception sdi-h_sir agree with naive at lag 0).
    k_shift = np.float32(0.0)

    # Unrolled 4×4 loop for Numba performance.
    # TX corner 0 (sign +1) × all RX corners
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
def _compute_Dh_pe_parallel_points(
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
    """PE SDI: 16 deltas per (m_e, m_r) pair + 1 cumsum. prange over P.

    Returns (P, T) float32 — Dh_pe at each scatterer position.
    """
    P = points.shape[0]
    M_e = tx_centers.shape[0]
    M_r = rx_centers.shape[0]
    # float64 buffer: PE weight = slope_e*slope_r can reach ~1e20 for sub-sample
    # patches (Dt clamped to dt → slope ~ area/dt², squared in the product).
    # A float32 buffer loses ~1e13 per write to cancellation → DC/ramp leak
    # (vertical streaks) after the cumsum. float64 ULP at 1e20 is ~2e4 (negligible).
    out = np.zeros((P, T), dtype=np.float64)
    for p in prange(P):  # ty: ignore[not-iterable]
        for m_r in range(M_r):
            # RX patch → scatterer direction (projected onto patch local frame).
            dx_r = points[p, 0] - rx_centers[m_r, 0]
            dy_r = points[p, 1] - rx_centers[m_r, 1]
            dz_r = points[p, 2] - rx_centers[m_r, 2]
            dist_r = np.sqrt(dx_r * dx_r + dy_r * dy_r + dz_r * dz_r)
            if dist_r < np.float32(1e-12):
                continue
            inv_dist_r = np.float32(1.0) / dist_r

            # Project onto RX patch local frame (xp, yp) and normalise by distance.
            xp_r = (
                dx_r * rx_tangents[m_r, 0]
                + dy_r * rx_tangents[m_r, 1]
                + dz_r * rx_tangents[m_r, 2]
            ) * inv_dist_r
            yp_r = (
                dx_r * rx_tangents[m_r, 3]
                + dy_r * rx_tangents[m_r, 4]
                + dz_r * rx_tangents[m_r, 5]
            ) * inv_dist_r
            t1r, t2r, t3r, t4r, h_max_r = _compute_rectangle_SIR_params(
                rx_wx[m_r],
                rx_wy[m_r],
                xp_r,
                yp_r,
                dist_r,
                inv_c,
                rx_apod[m_r],
                rx_delays[m_r],
                dt,
            )
            if h_max_r < np.float32(1e-6):
                continue
            slope_r = h_max_r / (t2r - t1r)

            for m_e in range(M_e):
                # TX patch → scatterer direction (projected onto patch local frame).
                dx_e = points[p, 0] - tx_centers[m_e, 0]
                dy_e = points[p, 1] - tx_centers[m_e, 1]
                dz_e = points[p, 2] - tx_centers[m_e, 2]
                dist_e = np.sqrt(dx_e * dx_e + dy_e * dy_e + dz_e * dz_e)
                if dist_e < np.float32(1e-12):
                    continue
                inv_dist_e = np.float32(1.0) / dist_e
                xp_e = (
                    dx_e * tx_tangents[m_e, 0]
                    + dy_e * tx_tangents[m_e, 1]
                    + dz_e * tx_tangents[m_e, 2]
                ) * inv_dist_e
                yp_e = (
                    dx_e * tx_tangents[m_e, 3]
                    + dy_e * tx_tangents[m_e, 4]
                    + dz_e * tx_tangents[m_e, 5]
                ) * inv_dist_e
                t1e, t2e, t3e, t4e, h_max_e = _compute_rectangle_SIR_params(
                    tx_wx[m_e],
                    tx_wy[m_e],
                    xp_e,
                    yp_e,
                    dist_e,
                    inv_c,
                    tx_apod[m_e],
                    tx_delays[m_e],
                    dt,
                )

                kstart = (t1e + t1r - t0) * fs
                kend = (t4e + t4r - t0) * fs + 2

                if kstart < 1 or kend > T:
                    print(
                        f"Warning: event outside time grid in point {p}"
                        f"and patch pair (m_e, m_r) = ({m_e}, {m_r})"
                    )
                    print(
                        "k_start:",
                        kstart,
                        "k_end:",
                        kend,
                        "T:",
                        T,
                    )
                    continue

                if h_max_e < np.float32(1e-6):
                    continue

                slope_e = h_max_e / (t2e - t1e)

                weight = slope_r * slope_e
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
                    weight,
                )

    # Raw Δδ_pe (= D²h_tx ⊛ D²h_rx); I⁴ applied downstream as ÷(jω)⁴.
    return out


# ---------------------------------------------------------------------------
# Accumulate-in-kernel summed variant — returns (T,), no (P, T) buffer.
# Threads each own one row of a (n_threads, T) buffer (chunked P range), so the
# shared accumulation is race-free without atomics; reduce after the loop.
# ---------------------------------------------------------------------------


# No cache=True: get_num_threads() is a dynamic global, which numba refuses to cache
# (and the warning would trip pytest's filterwarnings=error). Recompiles per session.
@njit(parallel=True, fastmath=True)
def _compute_Dh_pe_summed_points(
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

    Same deltas as `_compute_Dh_pe_parallel_points` but accumulated (× ``amps``)
    into one trace, so the caller skips the ``amps @ (P, T)`` matvec and the
    ``(P, T)`` buffer. Summed, no-attenuation path only.
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
    out = np.zeros(T, dtype=np.float64)
    for c in range(n_threads):
        for k in range(T):
            out[k] += buf[c, k]
    return out


# ---------------------------------------------------------------------------
# Patch-parallel single-point PSF — prange over TX patches, not P.
# For P == 1 the point-parallel kernel leaves the whole M_e loop serial; here
# threads split the M_e range (one (n_threads, T) row each) and reduce.
# ---------------------------------------------------------------------------


# No cache=True: get_num_threads() is a dynamic global (see _compute_Dh_pe_summed_points).
@njit(parallel=True, fastmath=True)
def _compute_Dh_pe_parallel_patches(
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
    """Δδ_pe for one field point, parallelized over TX patches → (T,) float64."""
    M_e = tx_centers.shape[0]
    M_r = rx_centers.shape[0]
    px = point[0]
    py = point[1]
    pz = point[2]
    n_threads = get_num_threads()
    buf = np.zeros((n_threads, T), dtype=np.float64)
    chunk = (M_e + n_threads - 1) // n_threads
    for c in prange(n_threads):  # ty: ignore[not-iterable]
        estart = c * chunk
        eend = min(estart + chunk, M_e)
        for m_e in range(estart, eend):
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
                    slope_r * slope_e,
                )
    out = np.zeros(T, dtype=np.float64)
    for c in range(n_threads):
        for k in range(T):
            out[k] += buf[c, k]
    return out


# ---------------------------------------------------------------------------
# Method 3 — Complete SDI PE: splat w = I⁴ v_pe per pair (no FFT, no cumsum).
# Each of the 16 corner events convolves a 2-bin linear-interp delta with w,
# i.e. adds a shifted, scaled copy of w. Output is the final per-element RF
# trace (still ×scale and amplitude-weighted by the caller).
# ---------------------------------------------------------------------------


@njit(inline="always")
def _add_shifted_w(out, row, kf, gain, w, nfft):
    """Add ``gain``-scaled, 2-bin-interpolated copy of ``w`` at index ``kf``, wrapped mod nfft.

    Circular placement: the event at continuous index ``kf`` deposits ``w`` rolled to
    ``kf`` (plus the fractional ceil tap). ``w`` is the full-length (``nfft``) integrated
    exc/IR kernel — the zero-phase I⁴ filter is delocalized, so the convolution must be
    circular (then sliced to ``pe_T``) to match the truncated path's FFT exactly.
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
def _compute_pe_complete_parallel_points(
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
    """Complete-SDI PE RF per scatterer → (P, T) float64. prange over P.

    Splats ``w`` (full-length ``nfft`` integrated exc/IR kernel) circularly per pair,
    then returns the ``[:T]`` window — the truncated path's circular convolution done
    by hand. Exact (≡ truncated) but O(nfft) per pair, hence the slow reference path.
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
    """Cast patch arrays to float32 and pack tangents for the PE Numba kernels."""
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
        tx_eu, tx_ev = _identity_tangents(M_e)
    if rx_eu is None or rx_ev is None:
        rx_eu, rx_ev = _identity_tangents(M_r)
    tx_tangents = _pack_tangents(
        np.asarray(tx_eu, dtype=np.float32), np.asarray(tx_ev, dtype=np.float32)
    )
    rx_tangents = _pack_tangents(
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
    """Pulse-echo SDI: Dh_pe = dh^e *_t d2h^r via combined delta placement.

    Computes the RAW Δδ_pe (= D²h_tx ⊛ D²h_rx) for one RX element (rx patches =
    patches of that element). For full Reception: call once per RX element with
    element-filtered rx patches. I⁴ is applied downstream in Fourier as ÷(jω)⁴ —
    there is no cumsum here.

    16 deltas per (m_e, m_r) pair → 32 sample writes. Parallelized over field
    points (prange over P), except a single point (``P == 1``, e.g. PSF) which is
    parallelized over TX patches instead (`_compute_Dh_pe_parallel_patches`).

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
        Differentiated modified pulse-echo SIR at each scatterer.
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
        row = _compute_Dh_pe_parallel_patches(
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
        return _compute_Dh_pe_parallel_points(
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
        out[start:end] = _compute_Dh_pe_parallel_points(
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
    return _compute_Dh_pe_summed_points(
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
    """Method 3 (complete SDI PE): splat ``w = I⁴ v_pe`` per pair → ``(P, T)`` float32.

    Cumsum-free evaluation of ``p_pe = Σ_i Σ_j a_i a_j w(t − τ_i − τ_j)``: each of the
    16 corner events per (m_e, m_r) pair adds a 2-bin-interpolated, slope-weighted copy
    of ``w``, wrapped mod ``len(w)``. ``w`` is the FULL-length (``nfft``) integrated
    exc/IR kernel ``I⁴(e ⊛ ir_tx ⊛ ir_rx)``; the circular convolution is sliced to
    ``T`` (= ``pe_T``), reproducing the truncated path exactly. The caller applies
    ``scale`` + amplitude weighting. Exact but O(nfft) per pair (slow reference path).

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
    return _compute_pe_complete_parallel_points(
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
