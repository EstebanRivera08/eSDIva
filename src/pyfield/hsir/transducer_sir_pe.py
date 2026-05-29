import numpy as np
from numba import njit, prange

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
    k_shift = np.float32(2.0)

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
        w_floor = signpos1 - w_ceil  # re-use +1 as one
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
    out = np.zeros((P, T), dtype=np.float32)
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

        # Integrate: 1 cumsum → Dh_pe (delta distribution, no dt scaling needed).
        acc = np.float64(0.0)
        for k in range(T):
            acc += np.float64(out[p, k])
            out[p, k] = np.float32(acc)
    return out


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

    Computes Dh_pe for one RX element (rx patches = patches of that element).
    For full Reception: call once per RX element with element-filtered rx patches.

    16 deltas per (m_e, m_r) pair → 32 sample writes → 1 cumsum.
    Parallelized over field points (prange over P).

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
    batch_size_points : int or None
        Chunk P into batches at Python level. None = no batching.

    Returns
    -------
    (P, T) float32 ndarray
        Differentiated modified pulse-echo SIR at each scatterer.
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
    P = points.shape[0]
    if tx_eu is None or tx_ev is None:
        tx_eu, tx_ev = _identity_tangents(M_e)
    if rx_eu is None or rx_ev is None:
        rx_eu, rx_ev = _identity_tangents(M_r)
    tx_eu = np.asarray(tx_eu, dtype=np.float32)
    tx_ev = np.asarray(tx_ev, dtype=np.float32)
    rx_eu = np.asarray(rx_eu, dtype=np.float32)
    rx_ev = np.asarray(rx_ev, dtype=np.float32)

    tx_tangents = _pack_tangents(tx_eu, tx_ev)
    rx_tangents = _pack_tangents(rx_eu, rx_ev)

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
        )

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
