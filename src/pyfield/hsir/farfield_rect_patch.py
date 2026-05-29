"""Far-field rectangular patch SIR computation kernels."""

import numpy as np
from numba import njit, prange

inv_2pi = 1 / (2 * np.pi)


# ---------- small helper (njit) for rectangle SIR parameters ----------


@njit(inline="always")
def compute_rectangle_SIR_params(wx, wy, dx, dy, dist, inv_c, apod, delay, dt):
    """Compute trapezoidal SIR parameters for a rectangular patch.

    Parameters
    ----------
    wx : float
        Patch half-width in the x-direction.
    wy : float
        Patch half-width in the y-direction.
    dx : float
        Direction component (xp) from patch centre to field point.
    dy : float
        Direction component (yp) from patch centre to field point.
    dist : float
        Distance from patch centre to field point.
    inv_c : float
        Inverse speed of sound (1/c).
    apod : float
        Apodization weight for this patch.
    delay : float
        Time delay for this patch in seconds.
    dt : float
        Time step size.

    Returns
    -------
    tuple of float
        ``(t1, t2, t3, t4, h_max)`` trapezoidal SIR timing parameters.
    """
    xp_abs = abs(dx) * wx * inv_c
    yp_abs = abs(dy) * wy * inv_c
    Dt1 = min(xp_abs, yp_abs)
    Dt2 = max(xp_abs, yp_abs)

    # enforce minimum to avoid zero width
    if Dt1 < dt:
        Dt1 = dt
    if Dt2 < dt:
        Dt2 = dt

    area = (wx * wy * inv_2pi) / dist
    # time-of-flight
    t1 = dist * inv_c - 0.5 * (Dt1 + Dt2) + delay
    t2 = t1 + Dt1
    t3 = t1 + Dt2
    t4 = t1 + Dt1 + Dt2

    # max height of trapezoid
    h_max = area * apod / Dt2

    return t1, t2, t3, t4, h_max


# ---------- main SIR computation function (njit, parallel) ----------
@njit(parallel=True, fastmath=True, cache=True)
def compute_parallelized_sir_optimized(
    P,
    M,
    T,
    points,
    center,
    wx,
    wy,
    patch_frames,
    inv_c,
    apodization,
    delays,
    time_grid,
    fs,
    dt,
    method_flag,  # 0 -> naive, 1 -> sdi, 2 -> auto
):
    """Compute SIR in parallel over field points.

    Parameters
    ----------
    P : int
        Number of field points.
    M : int
        Number of sub-patches.
    T : int
        Number of time samples.
    points : ndarray, shape (P, 3)
        Field point coordinates.
    center : ndarray, shape (M, 3)
        Patch centre coordinates.
    wx : ndarray, shape (M,)
        Per-patch width in x.
    wy : ndarray, shape (M,)
        Per-patch width in y.
    patch_frames : ndarray, shape (M, 6)
        Packed local frame: columns 0-2 are u-tangent (eu0, eu1, eu2),
        columns 3-5 are v-tangent (ev0, ev1, ev2).
    inv_c : float
        Inverse speed of sound.
    apodization : ndarray, shape (M,)
        Apodization weights per patch.
    delays : ndarray, shape (M,)
        Delays per patch in seconds.
    time_grid : ndarray
        Array of time samples.
    fs : float
        Sampling frequency in Hz.
    dt : float
        Time step size.
    method_flag : int
        0 for naive, 1 for SDI, 2 for auto.

    Returns
    -------
    tuple
        ``(h_out, range_k_matrix, min_time, max_time)``.
    """
    h_out = np.zeros((P, T), dtype=np.float32)
    d2h = np.zeros((P, T), dtype=np.float32)  # used if SDI path chosen
    range_k_matrix = np.zeros((P, M), dtype=np.int32)
    t0 = time_grid[0]

    # Per-point min/max to avoid race conditions in prange
    local_min_time = np.empty(P, dtype=np.float32)
    local_max_time = np.empty(P, dtype=np.float32)

    # precompute threshold term for auto decision (8 + 2*T/M)
    threshold_term = 8.0 + 2.0 * (T / M)

    for p in prange(P):  # ty: ignore[not-iterable]
        # per-point local event buffers for SDI (max 8*M entries)
        idxs = np.empty(8 * M, dtype=np.int32)
        vals = np.empty(8 * M, dtype=np.float32)

        p_min_time = np.float32(1e30)
        p_max_time = np.float32(-1e30)

        for m in range(M):
            dx = points[p, 0] - center[m, 0]
            dy = points[p, 1] - center[m, 1]
            dz = points[p, 2] - center[m, 2]

            distance = np.sqrt(dx * dx + dy * dy + dz * dz)
            inv_dist = np.float32(1.0) / distance
            # project direction onto local patch frame (u, v axes)
            xp = (dx * patch_frames[m, 0] + dy * patch_frames[m, 1] + dz * patch_frames[m, 2]) * inv_dist
            yp = (dx * patch_frames[m, 3] + dy * patch_frames[m, 4] + dz * patch_frames[m, 5]) * inv_dist

            t1, t2, t3, t4, h_max = compute_rectangle_SIR_params(
                wx[m],
                wy[m],
                xp,
                yp,
                distance,
                inv_c,
                apodization[m],
                delays[m],
                dt,
            )
            # skip if h_max negligible
            if t1 < p_min_time:
                p_min_time = t1
            if t4 > p_max_time:
                p_max_time = t4
            if h_max < 1e-6:
                range_k_matrix[p, m] = 0
                continue

            # compute discrete indices (floats)
            # find the first/last sample indices that could possibly overlap
            k_start = int(np.floor((t1 - t0) * fs))
            k_end = int(np.ceil((t4 - t0) * fs) + 1)

            # If out of time range skip point
            if k_end < 0 or k_start >= T:
                print("Warning: event outside time grid in point ", p)
                print(
                    "t1 (us):",
                    t1 * 1e6,
                    "t4 (us):",
                    t4 * 1e6,
                    "k_start:",
                    k_start,
                    "k_end:",
                    k_end,
                    "T:",
                    T,
                )
                continue

            range_k = k_end - k_start
            range_k_matrix[p, m] = range_k

            # decide method for this patch
            use_naive = True

            if method_flag == 0:  # naive
                use_naive = True
            elif method_flag == 1:  # sdi
                use_naive = False
            else:  # auto
                if range_k > threshold_term:
                    use_naive = False  # sdi

            # compute slope and basic values
            slope = h_max / (t2 - t1)

            if use_naive:
                # naive: fill h_out[p,k_start:k_end] with trapezoid values
                # note: convert t grid index to times on the fly
                for k in range(k_start, k_end):
                    t = time_grid[k]
                    # evaluate continuous trapezoid h(t)
                    if t < t1 or t >= t4:
                        continue
                    elif t < t2:
                        h_val = slope * (t - t1)
                    elif t < t3:
                        h_val = h_max
                    else:
                        h_val = slope * (t4 - t)
                    # accumulate
                    h_out[p, k] += h_val
            else:
                # SDI: accumulate eight events (floor+ceil weights per time) to d2h[p, ...]
                evt = 0
                # t1 (+)
                k1f = (t1 - t0) * fs + 1
                k4f = (t4 - t0) * fs + 1

                kf = k1f
                kf_floor = int(np.floor(kf))
                w_ceil = kf - kf_floor
                w_floor = 1.0 - w_ceil

                idxs[evt] = kf_floor
                vals[evt] = slope * w_floor
                evt += 1
                kf_ceil = kf_floor + 1

                idxs[evt] = kf_ceil
                vals[evt] = slope * w_ceil
                evt += 1

                # t2 (-)
                kf = (t2 - t0) * fs + 1
                kf_floor = int(np.floor(kf))
                w_ceil = kf - kf_floor

                w_floor = 1.0 - w_ceil
                idxs[evt] = kf_floor
                vals[evt] = -slope * w_floor
                evt += 1

                kf_ceil = kf_floor + 1
                idxs[evt] = kf_ceil
                vals[evt] = -slope * w_ceil
                evt += 1

                # t3 (-)
                kf = (t3 - t0) * fs + 1
                kf_floor = int(np.floor(kf))
                w_ceil = kf - kf_floor

                w_floor = 1.0 - w_ceil
                idxs[evt] = kf_floor
                vals[evt] = -slope * w_floor
                evt += 1

                kf_ceil = kf_floor + 1
                idxs[evt] = kf_ceil
                vals[evt] = -slope * w_ceil
                evt += 1

                # t4 (+)
                kf = k4f
                kf_floor = int(np.floor(kf))
                w_ceil = kf - kf_floor

                w_floor = 1.0 - w_ceil
                idxs[evt] = kf_floor
                vals[evt] = slope * w_floor
                evt += 1

                kf_ceil = kf_floor + 1
                idxs[evt] = kf_ceil
                vals[evt] = slope * w_ceil
                evt += 1

                # apply events to d2h
                for j in range(evt):
                    k_idx = idxs[j]
                    d2h[p, k_idx] += vals[j]

        # After all patches for point p processed, if any SDI events were added, integrate
        # We must integrate d2h -> dh -> h and add to h_out
        # (Even if some patches used naive, we still need to add integrated d2h)
        # First cumulative sum (in-place on a temp)
        if method_flag != 0:
            acc = 0.0
            for k in range(T):
                acc += d2h[p, k]
                d2h[p, k] = acc
                # we dont multiply by dt here, because delta width is 1 sample in discrete sum
            acc2 = 0.0
            for k in range(T):
                acc2 += d2h[p, k]
                # multiply by dt to match continuous integral scaling
                h_out[p, k] += acc2 * dt

        local_min_time[p] = p_min_time
        local_max_time[p] = p_max_time

    # Serial reduction over P to get global min/max (race-safe)
    min_time = local_min_time[0]
    max_time = local_max_time[0]
    for p in range(1, P):
        if local_min_time[p] < min_time:
            min_time = local_min_time[p]
        if local_max_time[p] > max_time:
            max_time = local_max_time[p]

    return h_out, range_k_matrix, min_time, max_time


# ---------- computes h_sir ----------
def compute_h_sir(
    P,
    M,
    T,
    dt,
    time_grid,
    points,
    centers,
    wx,
    wy,
    inv_c,
    fs,
    apodization_sub_elem,
    delays_sub_elem,
    method_flag=1,
    eu=None,
    ev=None,
):
    """Compute the SIR impulse response for field points and patches.

    Parameters
    ----------
    P : int
        Number of field points.
    M : int
        Number of transducer sub-patches.
    T : int
        Number of time samples.
    dt : float
        Time step size.
    time_grid : ndarray
        Array of time samples.
    points : ndarray, shape (P, 3)
        Coordinates of field points.
    centers : ndarray, shape (M, 3)
        Coordinates of patch centres.
    wx : ndarray, shape (M,)
        Per-patch width in x-direction.
    wy : ndarray, shape (M,)
        Per-patch width in y-direction.
    inv_c : float
        Inverse of the speed of sound (1/c).
    fs : float
        Sampling frequency in Hz.
    apodization_sub_elem : ndarray, shape (M,)
        Apodization weights per patch.
    delays_sub_elem : ndarray, shape (M,)
        Delay values per patch in seconds.
    method_flag : int, optional
        Computation method: 0 naive, 1 SDI, 2 auto. Default 1.
    eu : ndarray, shape (M, 3), optional
        Local u-tangent unit vectors per patch. None = global x-axis.
    ev : ndarray, shape (M, 3), optional
        Local v-tangent unit vectors per patch. None = global y-axis.

    Returns
    -------
    tuple
        ``(h_out, info_struct)`` where ``h_out`` is shape ``(P, T)`` and
        ``info_struct`` is a dict with ``min_time``, ``max_time``, and
        ``range_k_matrix``.
    """
    if eu is None or ev is None:
        eu_k = np.zeros((M, 3), dtype=np.float32)
        ev_k = np.zeros((M, 3), dtype=np.float32)
        eu_k[:, 0] = 1.0
        ev_k[:, 1] = 1.0
    else:
        eu_k = np.asarray(eu, dtype=np.float32)
        ev_k = np.asarray(ev, dtype=np.float32)

    # Pack (M,3) tangent pairs into one (M,6) array — fewer params reduces parfors alias analysis work.
    patch_frames = np.ascontiguousarray(np.concatenate([eu_k, ev_k], axis=1))
    h_out, range_k_matrix, min_time, max_time = compute_parallelized_sir_optimized(
        P,
        M,
        T,
        points,
        centers,
        wx,
        wy,
        patch_frames,
        inv_c,
        apodization_sub_elem,
        delays_sub_elem,
        time_grid,
        fs,
        dt,
        method_flag,
    )

    info_data = {
        "min_time": min_time,
        "max_time": max_time,
        "range_k_matrix": range_k_matrix,
    }
    return h_out, info_data
