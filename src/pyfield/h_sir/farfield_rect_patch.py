import numpy as np
from numba import njit, prange

inv_2pi = 1 / (2 * np.pi)


# ---------- small helper (njit) for rectangle SIR parameters ----------
@njit(inline="always")
def compute_rectangle_SIR_params(wx, wy, dx, dy, dist, inv_c, apod, delay, dt):
    """
    Return t1,t2,t3,t4,h_max (float32).
    dx,dy are direction cosines (xp, yp) used in your original compute.
    dist is distance from patch center to field point (float).
    inv_c is 1/c (float).
    """
    xp_abs = abs(dx) * wx * inv_c
    yp_abs = abs(dy) * wy * inv_c
    # enforce minimum to avoid zero width
    Dt1 = min(xp_abs, yp_abs)
    Dt2 = max(xp_abs, yp_abs)
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
@njit(parallel=True, fastmath=True)
def compute_parallelized_sir_optimized(
    P,
    M,
    T,
    points,
    center,
    wx,
    wy,
    inv_c,
    apodization,
    delays,
    time_grid,
    fs,
    dt,
    method_flag,  # 0 -> naive, 1 -> sdi, 2 -> auto
):
    """
    Returns h_out (P, T) and range_k_matrix (P, M)
    method_flag: 0 naive, 1 sdi, 2 auto
    """
    h_out = np.zeros((P, T), dtype=np.float32)
    d2h = np.zeros((P, T), dtype=np.float32)  # used if SDI path chosen
    range_k_matrix = np.zeros((P, M), dtype=np.int32)
    t0 = time_grid[0]

    # precompute threshold term for auto decision (8 + 2*T/M)
    threshold_term = 8.0 + 2.0 * (T / M)

    for p in prange(P):
        # per-point local event buffers for SDI (max 8*M entries)
        idxs = np.empty(8 * M, dtype=np.int32)
        vals = np.empty(8 * M, dtype=np.float32)

        for m in range(M):
            dx = points[p, 0] - center[m, 0]
            dy = points[p, 1] - center[m, 1]
            dz = points[p, 2] - center[m, 2]

            distance = np.sqrt(dx * dx + dy * dy + dz * dz)
            xp = dx / distance
            yp = dy / distance

            t1, t2, t3, t4, h_max = compute_rectangle_SIR_params(
                wx,
                wy,
                xp,
                yp,
                distance,
                inv_c,
                apodization[m],
                delays[m],
                dt,
            )
            if h_max < 1e-6:
                range_k_matrix[p, m] = np.nan
                continue

            # compute discrete indices (floats)
            # find the first/last sample indices that could possibly overlap
            k_start = int(np.floor((t1 - t0) * fs))
            k_end = int(np.ceil((t4 - t0) * fs) + 1)

            # clamp to valid range
            if k_end < 0 or k_start >= T:
                continue
            if k_start < 0:
                k_start = 0
            if k_end > T:
                k_end = T

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
                if k1f < 0.0 or k1f > T - 1.0 or k4f > T - 1.0:
                    print("Warning: event outside time grid in point ", p)
                    continue
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

    return h_out, range_k_matrix


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
):
    return compute_parallelized_sir_optimized(
        P,
        M,
        T,
        points,
        centers,
        wx,
        wy,
        inv_c,
        apodization_sub_elem,
        delays_sub_elem,
        time_grid,
        fs,
        dt,
        method_flag,  # 0 -> naive, 1 -> sdi, 2 -> auto
    )
