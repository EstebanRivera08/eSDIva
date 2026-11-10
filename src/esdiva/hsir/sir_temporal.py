"""Temporal SIR kernels: the far-field trapezoidal SIR h(r, t) sampled at fs.

Each rectangular patch contributes a trapezoid in time; `compute_h_sir` sums them per
field point, either fully sampled (FST) or as sparse corner deltas double-integrated (SDI).
Its frequency-domain twin lives in `sir_spectral`.

`place_twoway_rf` turns those sampled SIRs into a pulse-echo RF the time-domain way: a
scatterer's echo is short, so it convolves the two one-way SIRs over their non-zero span,
then the two-way pulse, and adds the result at the echo's own sample offset — no transform
over the whole depth window. This is the fst-like time-domain scattering solver, as used
by Field II; the name Field II is not repeated below.
"""

import warnings

import numpy as np
from numba import get_num_threads, njit, prange

from .helpers import (
    _compute_rectangle_SIR_params,
    identity_tangents,
    pack_tangents,
)


@njit(inline="always")
def _fully_sampled_trapezoid(
    h_out, p, k_start, k_end, time_grid, t1, t2, t3, t4, slope, h_max
):
    """Fill h_out[p, k_start:k_end] with the continuous trapezoid SIR (FST method).

    FST = Fully-Sampled Trapezoid: the patch SIR trapezoid is evaluated directly at
    every time sample it spans (as opposed to SDI, which writes only the corner deltas
    and recovers the trapezoid by a double cumulative sum).

    Evaluates the rising / plateau / falling segments of the patch SIR at each time
    sample and accumulates them — the exact band-unlimited SIR, used as the reference
    when the patch spans only a few samples.
    """
    for k in range(k_start, k_end):
        t = time_grid[k]
        if t < t1 or t >= t4:
            continue
        elif t < t2:
            h_val = slope * (t - t1)
        elif t < t3:
            h_val = h_max
        else:
            h_val = slope * (t4 - t)
        h_out[p, k] += h_val


@njit(inline="always")
def _place_sir_sdi_deltas(d2h, p, t0, fs, t1, t2, t3, t4, slope):
    """Place the 4 second-derivative deltas of one trapezoid into d2h[p, :] (SDI method).

    d²h/dt² of a trapezoid is four signed Diracs at its corners with signs (+,−,−,+),
    each scaled by the rising slope. Each delta is split across its two neighbouring
    sample bins by linear interpolation (the +1 keeps the same lattice the double
    cumulative sum integrates back on); the caller recovers h by that double cumsum.
    Cheaper than the FST fill when the patch spans many samples.
    """
    for tc, sign in ((t1, 1.0), (t2, -1.0), (t3, -1.0), (t4, 1.0)):
        kf = (tc - t0) * fs + 1.0
        k_floor = int(np.floor(kf))
        w_ceil = kf - k_floor
        signed_slope = sign * slope
        d2h[p, k_floor] += np.float32(signed_slope * (1.0 - w_ceil))
        d2h[p, k_floor + 1] += np.float32(signed_slope * w_ceil)


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
    method_flag,  # 0 -> FST, 1 -> sdi, 2 -> auto
    return_deltak,  # if True, fill range_k_matrix (per-patch trapezoid width in samples)
    soft,  # soft baffle: weight each patch by cosθ = max(0, n·u)
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
        0 for FST, 1 for SDI, 2 for auto.
    return_deltak : bool
        If True, fill and return the per-patch trapezoid width Δk (samples) in
        ``range_k_matrix``. If False, skip that ``(P, M)`` allocation entirely
        (a ``(1, 1)`` placeholder is returned) — the width is still computed as
        a scalar for the auto method decision.
    soft : bool
        Soft baffle: scale each trapezoid by ``cosθ = max(0, n·u)``, the angle between
        the patch normal ``n = u-tangent × v-tangent`` and the direction to the point.

    Returns
    -------
    tuple
        ``(h_out, range_k_matrix, min_time, max_time)``.
    """
    h_out = np.zeros((P, T), dtype=np.float32)

    # SDI pays off only for patches wider than threshold_term = 8 + 2T/M samples (the
    # 8 deltas + the double-cumsum break even there). A patch's trapezoid width is
    # bounded by range_k <= fs*(wx+wy)/c (|u| <= 1). So if the WIDEST patch still can't
    # reach the threshold, no patch ever will: auto then behaves as pure FST, and we
    # downgrade it here to skip the (P, T) d2h buffer entirely. This is what makes auto
    # the best of both worlds — e.g. a single receive element (M=1) has a huge threshold
    # and narrow patches, so auto costs exactly FST, not SDI.
    threshold_term = 8.0 + 2.0 * (T / M)
    if method_flag == 2:
        dk_max = np.float32(0.0)
        for m in range(M):
            dk = fs * (wx[m] + wy[m]) * inv_c
            if dk > dk_max:
                dk_max = dk
        if (
            dk_max + 2.0 <= threshold_term
        ):  # +2 covers the floor/ceil rounding of range_k
            method_flag = 0

    # SDI scratch for the 2nd-derivative delta train (double-integrated into h_out
    # below). Only the SDI and auto paths ever write it; the FST path
    # (method_flag == 0) leaves it untouched, so skip the full (P, T) allocation there
    # and keep a (1, 1) placeholder — halves the big-buffer footprint of FST runs.
    if method_flag != 0:
        d2h = np.zeros((P, T), dtype=np.float32)
    else:
        d2h = np.zeros((1, 1), dtype=np.float32)
    # Only allocate the full Δk matrix when the caller wants it; the scalar
    # range_k below still drives the auto FST/SDI decision either way.
    if return_deltak:
        range_k_matrix = np.zeros((P, M), dtype=np.int32)
    else:
        range_k_matrix = np.zeros((1, 1), dtype=np.int32)
    t0 = time_grid[0]

    # Per-point min/max to avoid race conditions in prange
    local_min_time = np.empty(P, dtype=np.float32)
    local_max_time = np.empty(P, dtype=np.float32)
    # Per-point SIR support [k_lo, k_hi) in samples — the first/last sample index any
    # patch actually writes for this point. This is the EXACT non-zero span (for SDI the
    # trapezoid ends at k_hi; its double-cumsum leaves only numerical drift beyond, which
    # this bound excludes), so the pulse-echo placement reads no spurious tail.
    local_k_lo = np.full(P, T, dtype=np.int32)
    local_k_hi = np.zeros(P, dtype=np.int32)
    # Out-of-grid flag per point (race-safe: each p writes its own slot). The caller
    # warns once if any patch event fell outside the time grid — never print inside
    # prange (garbled and slow under parallel threads).
    oob = np.zeros(P, dtype=np.int8)

    for p in prange(P):  # ty: ignore[not-iterable]
        p_min_time = np.float32(1e30)
        p_max_time = np.float32(-1e30)
        p_k_lo = T  # tightest written sample bounds for this point (support)
        p_k_hi = 0
        any_sdi = False  # did ANY patch place deltas? (auto: narrow patches use FST)

        for m in range(M):
            dx = points[p, 0] - center[m, 0]
            dy = points[p, 1] - center[m, 1]
            dz = points[p, 2] - center[m, 2]

            distance = np.sqrt(dx * dx + dy * dy + dz * dz)
            inv_dist = np.float32(1.0) / distance
            # project direction onto local patch frame (u, v axes)
            xp = (
                dx * patch_frames[m, 0]
                + dy * patch_frames[m, 1]
                + dz * patch_frames[m, 2]
            ) * inv_dist
            yp = (
                dx * patch_frames[m, 3]
                + dy * patch_frames[m, 4]
                + dz * patch_frames[m, 5]
            ) * inv_dist

            t1, t2, t3, t4, h_max = _compute_rectangle_SIR_params(
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
            if soft:
                f = patch_frames[m]
                zp = (
                    dx * (f[1] * f[5] - f[2] * f[4])
                    + dy * (f[2] * f[3] - f[0] * f[5])
                    + dz * (f[0] * f[4] - f[1] * f[3])
                ) * inv_dist
                h_max *= max(np.float32(0.0), zp)
            # skip if h_max negligible
            if t1 < p_min_time:
                p_min_time = t1
            if t4 > p_max_time:
                p_max_time = t4
            if h_max < 1e-6:
                continue

            # compute discrete indices (floats)
            # find the first/last sample indices that could possibly overlap
            k_start = int(np.floor((t1 - t0) * fs))
            k_end = int(np.ceil((t4 - t0) * fs) + 1)

            # If out of time range, flag the point and skip this patch (warn in caller).
            if k_end < 0 or k_start >= T:
                oob[p] = 1
                continue

            range_k = k_end - k_start
            if return_deltak:
                range_k_matrix[p, m] = range_k

            # Grow this point's written-sample support (used by the PE placement).
            if k_start < p_k_lo:
                p_k_lo = k_start
            if k_end > p_k_hi:
                p_k_hi = k_end

            # decide method for this patch
            use_fst = True

            if method_flag == 0:  # FST
                use_fst = True
            elif method_flag == 1:  # sdi
                use_fst = False
            else:  # auto
                if range_k > threshold_term:
                    use_fst = False  # sdi

            # compute slope and basic values
            slope = h_max / (t2 - t1)

            if use_fst:
                _fully_sampled_trapezoid(
                    h_out, p, k_start, k_end, time_grid, t1, t2, t3, t4, slope, h_max
                )
            else:
                _place_sir_sdi_deltas(d2h, p, t0, fs, t1, t2, t3, t4, slope)
                any_sdi = True

        # Recover h from the SDI delta train by double cumulative sum (d2h → dh → h).
        # The deltas live only inside this point's support [p_k_lo, p_k_hi], and the train
        # sums to zero, so integrating OUTSIDE that span adds nothing but float32 drift.
        # Summing only over the support is therefore both faster (O(span), not O(T)) and
        # cleaner (no drift tail) — the win the recorded support bounds unlock. acc/acc2 are
        # float64 to limit cancellation; the +1 includes the final corner delta at p_k_hi.
        if any_sdi and p_k_hi > p_k_lo:
            lo = max(0, p_k_lo)
            hi = min(T, p_k_hi + 1)
            acc = 0.0
            for k in range(lo, hi):
                acc += d2h[p, k]
                d2h[p, k] = acc  # delta width is 1 sample, so no dt here
            acc2 = 0.0
            for k in range(lo, hi):
                acc2 += d2h[p, k]
                h_out[p, k] += acc2 * dt  # dt matches the continuous integral scaling

        local_min_time[p] = p_min_time
        local_max_time[p] = p_max_time
        # Clamp the support to the grid; empty (no patch written) → [0, 0).
        if p_k_hi > p_k_lo:
            local_k_lo[p] = max(0, p_k_lo)
            local_k_hi[p] = min(T, p_k_hi)

    # Serial reduction over P to get global min/max (race-safe)
    min_time = local_min_time[0]
    max_time = local_max_time[0]
    for p in range(1, P):
        if local_min_time[p] < min_time:
            min_time = local_min_time[p]
        if local_max_time[p] > max_time:
            max_time = local_max_time[p]

    return h_out, range_k_matrix, min_time, max_time, oob, local_k_lo, local_k_hi


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
    return_deltak=False,
    soft_baffle=False,
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
        Computation method: 0 FST, 1 SDI, 2 auto. Default 1.
    eu : ndarray, shape (M, 3), optional
        Local u-tangent unit vectors per patch. None = global x-axis.
    ev : ndarray, shape (M, 3), optional
        Local v-tangent unit vectors per patch. None = global y-axis.
    return_deltak : bool, optional
        If True, include ``range_k_matrix`` (per-patch trapezoid width in
        samples, shape ``(P, M)``) in the returned info dict. Default False
        skips that allocation — most callers never read it.
    soft_baffle : bool, optional
        Soft (pressure-release) baffle: weight each patch by ``cosθ``. Default False
        is the rigid baffle.

    Returns
    -------
    tuple
        ``(h_out, info_struct)`` where ``h_out`` is shape ``(P, T)`` and
        ``info_struct`` is a dict with ``min_time``, ``max_time``, the per-point SIR
        support bounds ``k_lo``/``k_hi`` (``(P,)`` int32, the first/last written sample),
        and (only when ``return_deltak=True``) ``range_k_matrix``.
    """
    if eu is None or ev is None:
        eu, ev = identity_tangents(M)
    patch_frames = pack_tangents(
        np.asarray(eu, dtype=np.float32), np.asarray(ev, dtype=np.float32)
    )
    h_out, range_k_matrix, min_time, max_time, oob, k_lo, k_hi = (
        compute_parallelized_sir_optimized(
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
            return_deltak,
            bool(soft_baffle),
        )
    )

    n_oob = int(oob.sum())
    if n_oob:
        warnings.warn(
            f"{n_oob} field point(s) had patch SIR events outside the time grid; "
            "those contributions were dropped. Widen the time grid or check the "
            "geometry (points coincident with the aperture).",
            UserWarning,
            stacklevel=2,
        )

    info_data = {"min_time": min_time, "max_time": max_time, "k_lo": k_lo, "k_hi": k_hi}
    if return_deltak:
        info_data["range_k_matrix"] = range_k_matrix
    return h_out, info_data


# ---------- pulse-echo RF by time-domain placement ----------
@njit(parallel=True, fastmath=True, cache=True)
def _place_twoway_rf(
    h_tx, h_rx, k_lo_tx, k_hi_tx, k_lo_rx, k_hi_rx, pulse, amps, pe_T, n_threads
):
    """Accumulate one receive channel's pulse-echo RF by placing each scatterer's echo.

    A scatterer's echo occupies only a short window around its round-trip arrival, namely
    the SIR support ``[k_lo, k_hi)`` the SIR kernel recorded per point. For scatterer p this
    convolves its two one-way SIRs over those exact spans (``h_tx[p] ⊛ h_rx[p]``), then the
    short two-way pulse ``exc ⊛ ir_tx ⊛ ir_rx``, and adds the amplitude-weighted result at
    the integer sample where the echo starts — no transform over the whole depth. Using the
    recorded support (not a magnitude threshold) keeps SDI's drift tail out exactly. Threads
    own private row buffers (summed at the end), so the accumulation is race-free.
    """
    P = h_tx.shape[0]
    Lp = pulse.shape[0]
    buf = np.zeros((n_threads, pe_T), dtype=np.float32)
    for ti in prange(n_threads):  # ty: ignore[not-iterable]
        for p in range(ti * P // n_threads, (ti + 1) * P // n_threads):
            a = amps[p]
            if a == 0.0:
                continue
            a_tx, b_tx = k_lo_tx[p], k_hi_tx[p]
            a_rx, b_rx = k_lo_rx[p], k_hi_rx[p]
            if a_tx >= b_tx or a_rx >= b_rx:
                continue
            n_tx, n_rx = b_tx - a_tx, b_rx - a_rx
            # Two-way SIR h_pe = h_tx ⊛ h_rx over the two recorded support spans.
            h_pe = np.zeros(n_tx + n_rx - 1, dtype=np.float32)
            for i in range(n_tx):
                v = h_tx[p, a_tx + i]
                if v == 0.0:
                    continue
                for j in range(n_rx):
                    h_pe[i + j] += v * h_rx[p, a_rx + j]
            # Convolve with the pulse and add at the echo's sample offset a_tx + a_rx.
            off = a_tx + a_rx
            for i in range(h_pe.shape[0]):
                v = h_pe[i]
                if v == 0.0:
                    continue
                av = a * v
                base = off + i
                for k in range(Lp):
                    buf[ti, base + k] += av * pulse[k]
    out = buf[0].copy()
    for ti in range(1, n_threads):
        out += buf[ti]
    return out


def place_twoway_rf(h_tx, h_rx, k_tx, k_rx, pulse, amps, pe_T):
    """Pulse-echo RF of one receive channel by time-domain placement.

    Resolves the thread count outside the cached kernel (so it never touches the Numba
    runtime) and calls `_place_twoway_rf`, which does the per-scatterer short-convolution
    and splat. The caller applies the ``ρ₀/2c₀²·dt`` scale.

    Parameters
    ----------
    h_tx : (P, T) numpy.ndarray
        Sampled transmit one-way SIR per scatterer, float32.
    h_rx : (P, T) numpy.ndarray
        Sampled receive one-way SIR per scatterer (this channel), float32.
    k_tx : tuple[numpy.ndarray, numpy.ndarray]
        Transmit support bounds ``(k_lo, k_hi)`` (first/last written sample per
        scatterer, ``(P,)`` int), from `compute_h_sir`'s info dict.
    k_rx : tuple[numpy.ndarray, numpy.ndarray]
        Receive support bounds ``(k_lo, k_hi)``, same convention as ``k_tx``.
    pulse : (L,) numpy.ndarray
        Two-way pulse ``exc ⊛ ir_tx ⊛ ir_rx``, float32.
    amps : (P,) numpy.ndarray
        Scattering amplitude per scatterer, float32.
    pe_T : int
        Length of the output RF (two-way SIR span + pulse length).

    Returns
    -------
    (pe_T,) numpy.ndarray
        Pulse-echo RF for this receive channel (unscaled), float32.
    """
    nthr = max(1, min(get_num_threads(), h_tx.shape[0]))
    return _place_twoway_rf(
        np.ascontiguousarray(h_tx, dtype=np.float32),
        np.ascontiguousarray(h_rx, dtype=np.float32),
        np.ascontiguousarray(k_tx[0], dtype=np.int32),
        np.ascontiguousarray(k_tx[1], dtype=np.int32),
        np.ascontiguousarray(k_rx[0], dtype=np.int32),
        np.ascontiguousarray(k_rx[1], dtype=np.int32),
        np.ascontiguousarray(pulse, dtype=np.float32),
        np.ascontiguousarray(amps, dtype=np.float32),
        int(pe_T),
        nthr,
    )
