"""SIR derivative kernels — d2h, dh, and per-element variants (SDI-only).

All Numba kernels operate on float32 arrays and use the SDI (Sparse Delta
Integration) method exclusively.  Two parallelism axes are supported:

* ``"points"`` (default) — ``prange`` over P field points (no race conditions).
* ``"patches"`` — ``prange`` over M patches with thread-local reduction.
  Use when P < n_threads (e.g. single scatterer in Reception).

Integration helpers convert between derivative levels:
  d2h (raw SDI events) → dh (1 cumsum) → h (2nd cumsum × dt)
"""

import numba
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


# ---------------------------------------------------------------------------
# SDI event placement helpers (inlined into kernel loops)
# ---------------------------------------------------------------------------


@njit(inline="always")
def _place_sdi_2d(out, p, t0, T, fs, t1, t2, t3, t4, slope):
    """Place 8 SDI delta events for one patch into out[p, :] (P, T)."""
    # t1: +slope
    kf = (t1 - t0) * fs + 1.0
    kf_floor = int(np.floor(kf))
    w_ceil = kf - kf_floor
    w_floor = 1.0 - w_ceil
    if 0 <= kf_floor < T:
        out[p, kf_floor] += slope * w_floor
    if 0 <= kf_floor + 1 < T:
        out[p, kf_floor + 1] += slope * w_ceil
    # t2: -slope
    kf = (t2 - t0) * fs + 1.0
    kf_floor = int(np.floor(kf))
    w_ceil = kf - kf_floor
    w_floor = 1.0 - w_ceil
    if 0 <= kf_floor < T:
        out[p, kf_floor] -= slope * w_floor
    if 0 <= kf_floor + 1 < T:
        out[p, kf_floor + 1] -= slope * w_ceil
    # t3: -slope
    kf = (t3 - t0) * fs + 1.0
    kf_floor = int(np.floor(kf))
    w_ceil = kf - kf_floor
    w_floor = 1.0 - w_ceil
    if 0 <= kf_floor < T:
        out[p, kf_floor] -= slope * w_floor
    if 0 <= kf_floor + 1 < T:
        out[p, kf_floor + 1] -= slope * w_ceil
    # t4: +slope
    kf = (t4 - t0) * fs + 1.0
    kf_floor = int(np.floor(kf))
    w_ceil = kf - kf_floor
    w_floor = 1.0 - w_ceil
    if 0 <= kf_floor < T:
        out[p, kf_floor] += slope * w_floor
    if 0 <= kf_floor + 1 < T:
        out[p, kf_floor + 1] += slope * w_ceil


@njit(inline="always")
def _place_sdi_3d(out, p, e_idx, t0, T, fs, t1, t2, t3, t4, slope):
    """Place 8 SDI delta events for one patch into out[p, e_idx, :] (P, E, T)."""
    # t1: +slope
    kf = (t1 - t0) * fs + 1.0
    kf_floor = int(np.floor(kf))
    w_ceil = kf - kf_floor
    w_floor = 1.0 - w_ceil
    if 0 <= kf_floor < T:
        out[p, e_idx, kf_floor] += slope * w_floor
    if 0 <= kf_floor + 1 < T:
        out[p, e_idx, kf_floor + 1] += slope * w_ceil
    # t2: -slope
    kf = (t2 - t0) * fs + 1.0
    kf_floor = int(np.floor(kf))
    w_ceil = kf - kf_floor
    w_floor = 1.0 - w_ceil
    if 0 <= kf_floor < T:
        out[p, e_idx, kf_floor] -= slope * w_floor
    if 0 <= kf_floor + 1 < T:
        out[p, e_idx, kf_floor + 1] -= slope * w_ceil
    # t3: -slope
    kf = (t3 - t0) * fs + 1.0
    kf_floor = int(np.floor(kf))
    w_ceil = kf - kf_floor
    w_floor = 1.0 - w_ceil
    if 0 <= kf_floor < T:
        out[p, e_idx, kf_floor] -= slope * w_floor
    if 0 <= kf_floor + 1 < T:
        out[p, e_idx, kf_floor + 1] -= slope * w_ceil
    # t4: +slope
    kf = (t4 - t0) * fs + 1.0
    kf_floor = int(np.floor(kf))
    w_ceil = kf - kf_floor
    w_floor = 1.0 - w_ceil
    if 0 <= kf_floor < T:
        out[p, e_idx, kf_floor] += slope * w_floor
    if 0 <= kf_floor + 1 < T:
        out[p, e_idx, kf_floor + 1] += slope * w_ceil


# ---------------------------------------------------------------------------
# Points-parallel kernels — prange over P (no race conditions)
# ---------------------------------------------------------------------------


@njit(parallel=True, fastmath=True)
def _compute_d2h_ppar(points, centers, wx, wy, inv_c, apod, delays, t0, T, fs, dt):
    """d2h summed over all patches, prange over P. Returns (P, T) float32."""
    P = points.shape[0]
    M = centers.shape[0]
    out = np.zeros((P, T), dtype=np.float32)
    for p in prange(P):  # ty: ignore[not-iterable]
        for m in range(M):
            dx = points[p, 0] - centers[m, 0]
            dy = points[p, 1] - centers[m, 1]
            dz = points[p, 2] - centers[m, 2]
            dist = np.sqrt(dx * dx + dy * dy + dz * dz)
            if dist < np.float32(1e-12):
                continue
            xp = dx / dist
            yp = dy / dist
            t1, t2, t3, t4, h_max = _compute_rectangle_SIR_params(
                wx[m], wy[m], xp, yp, dist, inv_c, apod[m], delays[m], dt
            )
            if h_max < np.float32(1e-6):
                continue
            slope = h_max / (t2 - t1)
            _place_sdi_2d(out, p, t0, T, fs, t1, t2, t3, t4, slope)
    return out


@njit(parallel=True, fastmath=True)
def _compute_d2h_per_element_ppar(
    points, centers, wx, wy, inv_c, apod, delays, t0, T, fs, dt, sub_el_idx, n_elements
):
    """d2h per element, prange over P. Returns (P, E, T) float32."""
    P = points.shape[0]
    M = centers.shape[0]
    out = np.zeros((P, n_elements, T), dtype=np.float32)
    for p in prange(P):  # ty: ignore[not-iterable]
        for m in range(M):
            dx = points[p, 0] - centers[m, 0]
            dy = points[p, 1] - centers[m, 1]
            dz = points[p, 2] - centers[m, 2]
            dist = np.sqrt(dx * dx + dy * dy + dz * dz)
            if dist < np.float32(1e-12):
                continue
            xp = dx / dist
            yp = dy / dist
            t1, t2, t3, t4, h_max = _compute_rectangle_SIR_params(
                wx[m], wy[m], xp, yp, dist, inv_c, apod[m], delays[m], dt
            )
            if h_max < np.float32(1e-6):
                continue
            slope = h_max / (t2 - t1)
            e_idx = sub_el_idx[m]
            _place_sdi_3d(out, p, e_idx, t0, T, fs, t1, t2, t3, t4, slope)
    return out


# ---------------------------------------------------------------------------
# Patches-parallel kernels — prange over M, thread-local reduction
# ---------------------------------------------------------------------------


@njit(parallel=True, fastmath=True)
def _compute_d2h_mpar(
    points, centers, wx, wy, inv_c, apod, delays, t0, T, fs, dt, n_threads
):
    """d2h summed over all patches, prange over M. Thread-local reduction.

    Returns (P, T) float32. Use when P < n_threads (few scatterers).
    WARNING: allocates (n_threads, P, T) intermediate buffer.
    """
    P = points.shape[0]
    M = centers.shape[0]
    thread_out = np.zeros((n_threads, P, T), dtype=np.float32)
    for m in prange(M):  # ty: ignore[not-iterable]
        tid = numba.get_thread_id()
        out_t = thread_out[tid]
        for p in range(P):
            dx = points[p, 0] - centers[m, 0]
            dy = points[p, 1] - centers[m, 1]
            dz = points[p, 2] - centers[m, 2]
            dist = np.sqrt(dx * dx + dy * dy + dz * dz)
            if dist < np.float32(1e-12):
                continue
            xp = dx / dist
            yp = dy / dist
            t1, t2, t3, t4, h_max = _compute_rectangle_SIR_params(
                wx[m], wy[m], xp, yp, dist, inv_c, apod[m], delays[m], dt
            )
            if h_max < np.float32(1e-6):
                continue
            slope = h_max / (t2 - t1)
            _place_sdi_2d(out_t, p, t0, T, fs, t1, t2, t3, t4, slope)
    # Serial reduction over threads
    out = np.zeros((P, T), dtype=np.float32)
    for t_idx in range(n_threads):
        for p in range(P):
            for k in range(T):
                out[p, k] += thread_out[t_idx, p, k]
    return out


@njit(parallel=True, fastmath=True)
def _compute_d2h_per_element_mpar(
    points,
    centers,
    wx,
    wy,
    inv_c,
    apod,
    delays,
    t0,
    T,
    fs,
    dt,
    sub_el_idx,
    n_elements,
    n_threads,
):
    """d2h per element, prange over M. Thread-local reduction.

    Returns (P, E, T) float32.
    WARNING: allocates (n_threads, P, E, T) intermediate buffer.
    """
    P = points.shape[0]
    M = centers.shape[0]
    thread_out = np.zeros((n_threads, P, n_elements, T), dtype=np.float32)
    for m in prange(M):  # ty: ignore[not-iterable]
        tid = numba.get_thread_id()
        out_t = thread_out[tid]
        e_idx = sub_el_idx[m]
        for p in range(P):
            dx = points[p, 0] - centers[m, 0]
            dy = points[p, 1] - centers[m, 1]
            dz = points[p, 2] - centers[m, 2]
            dist = np.sqrt(dx * dx + dy * dy + dz * dz)
            if dist < np.float32(1e-12):
                continue
            xp = dx / dist
            yp = dy / dist
            t1, t2, t3, t4, h_max = _compute_rectangle_SIR_params(
                wx[m], wy[m], xp, yp, dist, inv_c, apod[m], delays[m], dt
            )
            if h_max < np.float32(1e-6):
                continue
            slope = h_max / (t2 - t1)
            _place_sdi_3d(out_t, p, e_idx, t0, T, fs, t1, t2, t3, t4, slope)
    # Serial reduction
    out = np.zeros((P, n_elements, T), dtype=np.float32)
    for t_idx in range(n_threads):
        for p in range(P):
            for e in range(n_elements):
                for k in range(T):
                    out[p, e, k] += thread_out[t_idx, p, e, k]
    return out


# ---------------------------------------------------------------------------
# Integration helpers (Numba inner loops + Python dispatchers)
# ---------------------------------------------------------------------------


@njit(parallel=True, fastmath=True)
def _cumsum_2d(arr):
    """Cumulative sum along last axis for 2D (N, T) float32 array (no dt scaling).

    Uses float64 accumulator with float32 write-back to match the reference
    integration in farfield_rect_patch.py (acc = 0.0 → d2h[p,k] = acc).
    Rows are independent — prange over N is safe (no race conditions).
    """
    N, T = arr.shape
    out = np.empty_like(arr)
    for i in prange(N):  # ty: ignore[not-iterable]
        acc = np.float64(0.0)
        for k in range(T):
            acc += np.float64(arr[i, k])
            out[i, k] = np.float32(acc)
    return out


@njit(parallel=True, fastmath=True)
def _cumsum_3d(arr):
    """Cumulative sum along last axis for 3D (P, E, T) float32 array (no dt).

    Uses float64 accumulator with float32 write-back (matches reference kernel).
    Flattens (P, E) into a single prange — each row is independent.
    """
    P, E, T = arr.shape
    out = np.empty_like(arr)
    for i in prange(P * E):  # ty: ignore[not-iterable]
        p = i // E
        e = i % E
        acc = np.float64(0.0)
        for k in range(T):
            acc += np.float64(arr[p, e, k])
            out[p, e, k] = np.float32(acc)
    return out


def integrate_d2h_to_dh(d2h, dt):
    """Single cumulative sum along last axis (no dt scaling).

    Converts raw SDI events (d2h) to first-derivative SIR (dh).
    No dt factor: delta width is 1 sample in the discrete SDI convention.

    Parameters
    ----------
    d2h : float32 ndarray, shape (P, T) or (P, E, T)
        Second-derivative SIR — raw SDI event array from ``compute_d2h`` or
        ``compute_d2h_per_element``.
    dt : float
        Time step 1/fs (seconds). Accepted for API symmetry; not applied here.

    Returns
    -------
    dh : float32 ndarray, same shape as d2h
        First-derivative SIR.

    Raises
    ------
    ValueError
        If ``d2h`` is not 2-D or 3-D.
    """
    d2h = np.asarray(d2h, dtype=np.float32)
    if d2h.ndim == 2:
        return _cumsum_2d(d2h)
    if d2h.ndim == 3:
        return _cumsum_3d(d2h)
    raise ValueError(f"Expected 2D or 3D array, got {d2h.ndim}D.")


def integrate_dh_to_h(dh, dt):
    """Single cumulative sum along last axis, scaled by dt.

    Converts first-derivative SIR (dh) to spatial impulse response (h).

    Parameters
    ----------
    dh : float32 ndarray, shape (P, T) or (P, E, T)
        First-derivative SIR from ``integrate_d2h_to_dh`` or ``compute_dh``.
    dt : float
        Time step 1/fs (seconds). Applied as multiplicative scaling after cumsum.

    Returns
    -------
    h : float32 ndarray, same shape as dh
        Spatial impulse response.

    Raises
    ------
    ValueError
        If ``dh`` is not 2-D or 3-D.
    """
    dh = np.asarray(dh, dtype=np.float32)
    dt32 = np.float32(dt)
    if dh.ndim == 2:
        return _cumsum_2d(dh) * dt32
    if dh.ndim == 3:
        return _cumsum_3d(dh) * dt32
    raise ValueError(f"Expected 2D or 3D array, got {dh.ndim}D.")


# ---------------------------------------------------------------------------
# Public Python wrappers
# ---------------------------------------------------------------------------


def _get_n_threads():
    """Return current Numba thread count."""
    try:
        return numba.get_num_threads()
    except AttributeError:
        return int(numba.config.NUMBA_NUM_THREADS)  # type: ignore[attr-defined]


def _run_d2h(
    points, centers, wx, wy, inv_c, apod, delays, t0, T, fs, dt, parallel_axis
):
    if parallel_axis == "points":
        return _compute_d2h_ppar(
            points, centers, wx, wy, inv_c, apod, delays, t0, T, fs, dt
        )
    return _compute_d2h_mpar(
        points, centers, wx, wy, inv_c, apod, delays, t0, T, fs, dt, _get_n_threads()
    )


def _run_d2h_per_element(
    points,
    centers,
    wx,
    wy,
    inv_c,
    apod,
    delays,
    t0,
    T,
    fs,
    dt,
    sub_el_idx,
    n_elements,
    parallel_axis,
):
    if parallel_axis == "points":
        return _compute_d2h_per_element_ppar(
            points,
            centers,
            wx,
            wy,
            inv_c,
            apod,
            delays,
            t0,
            T,
            fs,
            dt,
            sub_el_idx,
            n_elements,
        )
    return _compute_d2h_per_element_mpar(
        points,
        centers,
        wx,
        wy,
        inv_c,
        apod,
        delays,
        t0,
        T,
        fs,
        dt,
        sub_el_idx,
        n_elements,
        _get_n_threads(),
    )


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


def compute_d2h(
    points,
    centers,
    wx,
    wy,
    inv_c,
    apod,
    delays,
    t0,
    T,
    fs,
    dt,
    *,
    parallel_axis="points",
    batch_size_points=None,
):
    """Second derivative of SIR summed over all patches (raw SDI events).

    Apply ``integrate_d2h_to_dh`` then ``integrate_dh_to_h`` to recover h_sir.

    Parameters
    ----------
    points : float32 ndarray, shape (P, 3)
        Field point coordinates in metres.
    centers : float32 ndarray, shape (M, 3)
        Patch centre coordinates in metres.
    wx : float32 ndarray, shape (M,)
        Patch width in x (metres).
    wy : float32 ndarray, shape (M,)
        Patch width in y (metres).
    inv_c : float
        Inverse speed of sound 1/c (s/m).
    apod : float32 ndarray, shape (M,)
        Apodization weight per patch.
    delays : float32 ndarray, shape (M,)
        Transmit delay per patch (seconds).
    t0 : float
        Start of time grid (seconds).
    T : int
        Number of time samples.
    fs : float
        Sampling frequency (Hz).
    dt : float
        Time step 1/fs (seconds).
    parallel_axis : {"points", "patches"}
        Parallelism axis. Default ``"points"`` (prange over P).
        Use ``"patches"`` when P < n_threads.
    batch_size_points : int or None
        Chunk P into batches at Python level. None = no batching.

    Returns
    -------
    d2h : float32 ndarray, shape (P, T)
        Second derivative of SIR.
    """
    points, centers, wx, wy, apod, delays = _prepare_arrays(
        points, centers, wx, wy, apod, delays
    )
    inv_c, t0, fs, dt, T = float(inv_c), float(t0), float(fs), float(dt), int(T)
    P = points.shape[0]

    if batch_size_points is None or batch_size_points >= P:
        return _run_d2h(
            points, centers, wx, wy, inv_c, apod, delays, t0, T, fs, dt, parallel_axis
        )

    out = np.zeros((P, T), dtype=np.float32)
    for start in range(0, P, batch_size_points):
        end = min(start + batch_size_points, P)
        out[start:end] = _run_d2h(
            points[start:end],
            centers,
            wx,
            wy,
            inv_c,
            apod,
            delays,
            t0,
            T,
            fs,
            dt,
            parallel_axis,
        )
    return out


def compute_dh(
    points,
    centers,
    wx,
    wy,
    inv_c,
    apod,
    delays,
    t0,
    T,
    fs,
    dt,
    *,
    parallel_axis="points",
    batch_size_points=None,
):
    """First derivative of SIR summed over all patches (d2h integrated once).

    Parameters
    ----------
    (same as ``compute_d2h``)

    Returns
    -------
    dh : float32 ndarray, shape (P, T)
        First derivative of SIR.
    """
    d2h = compute_d2h(
        points,
        centers,
        wx,
        wy,
        inv_c,
        apod,
        delays,
        t0,
        T,
        fs,
        dt,
        parallel_axis=parallel_axis,
        batch_size_points=batch_size_points,
    )
    return integrate_d2h_to_dh(d2h, dt)


def compute_d2h_per_element(
    points,
    centers,
    wx,
    wy,
    inv_c,
    apod,
    delays,
    t0,
    T,
    fs,
    dt,
    sub_el_idx,
    n_elements,
    *,
    parallel_axis="points",
    batch_size_points=None,
):
    """Second derivative of SIR per element (raw SDI events).

    Parameters
    ----------
    points : float32 ndarray, shape (P, 3)
    centers : float32 ndarray, shape (M, 3)
    wx : float32 ndarray, shape (M,)
    wy : float32 ndarray, shape (M,)
    inv_c : float
    apod : float32 ndarray, shape (M,)
    delays : float32 ndarray, shape (M,)
    t0 : float
    T : int
    fs : float
    dt : float
    sub_el_idx : int32 ndarray, shape (M,)
        Element index for each patch (from ``compute_sub_elem_attributes``).
    n_elements : int
        Total number of elements E.
    parallel_axis : {"points", "patches"}
    batch_size_points : int or None

    Returns
    -------
    d2h : float32 ndarray, shape (P, E, T)
        Second derivative of SIR grouped by element.
    """
    points, centers, wx, wy, apod, delays = _prepare_arrays(
        points, centers, wx, wy, apod, delays
    )
    sub_el_idx = np.asarray(sub_el_idx, dtype=np.int32)
    inv_c, t0, fs, dt, T = float(inv_c), float(t0), float(fs), float(dt), int(T)
    n_elements = int(n_elements)
    P = points.shape[0]

    if batch_size_points is None or batch_size_points >= P:
        return _run_d2h_per_element(
            points,
            centers,
            wx,
            wy,
            inv_c,
            apod,
            delays,
            t0,
            T,
            fs,
            dt,
            sub_el_idx,
            n_elements,
            parallel_axis,
        )

    out = np.zeros((P, n_elements, T), dtype=np.float32)
    for start in range(0, P, batch_size_points):
        end = min(start + batch_size_points, P)
        out[start:end] = _run_d2h_per_element(
            points[start:end],
            centers,
            wx,
            wy,
            inv_c,
            apod,
            delays,
            t0,
            T,
            fs,
            dt,
            sub_el_idx,
            n_elements,
            parallel_axis,
        )
    return out


def compute_dh_per_element(
    points,
    centers,
    wx,
    wy,
    inv_c,
    apod,
    delays,
    t0,
    T,
    fs,
    dt,
    sub_el_idx,
    n_elements,
    *,
    parallel_axis="points",
    batch_size_points=None,
):
    """First derivative of SIR per element (d2h integrated once).

    Parameters
    ----------
    (same as ``compute_d2h_per_element``)

    Returns
    -------
    dh : float32 ndarray, shape (P, E, T)
        First derivative of SIR grouped by element.
    """
    d2h = compute_d2h_per_element(
        points,
        centers,
        wx,
        wy,
        inv_c,
        apod,
        delays,
        t0,
        T,
        fs,
        dt,
        sub_el_idx,
        n_elements,
        parallel_axis=parallel_axis,
        batch_size_points=batch_size_points,
    )
    return integrate_d2h_to_dh(d2h, dt)


def compute_h_sir_patch_parallel(
    points,
    centers,
    wx,
    wy,
    inv_c,
    apod,
    delays,
    t0,
    T,
    fs,
    dt,
    *,
    n_threads=None,
):
    """Compute h_sir with prange over patches M (reference / testing only).

    Equivalent to the SDI path of ``compute_h_sir`` in ``farfield_rect_patch.py``
    but parallelised over M patches instead of P field points.  Uses thread-local
    reduction to avoid race conditions.

    Intended for validating ``compute_d2h`` via the identity:
    ``compute_h_sir_patch_parallel == integrate_dh_to_h(integrate_d2h_to_dh(compute_d2h(...)))``.

    Parameters
    ----------
    points : float32 ndarray, shape (P, 3)
    centers : float32 ndarray, shape (M, 3)
    wx : float32 ndarray, shape (M,)
    wy : float32 ndarray, shape (M,)
    inv_c : float
    apod : float32 ndarray, shape (M,)
    delays : float32 ndarray, shape (M,)
    t0 : float
    T : int
    fs : float
    dt : float
    n_threads : int or None
        Number of Numba threads. None = ``numba.get_num_threads()``.

    Returns
    -------
    h : float32 ndarray, shape (P, T)
        Spatial impulse response.
    """
    points, centers, wx, wy, apod, delays = _prepare_arrays(
        points, centers, wx, wy, apod, delays
    )
    inv_c, t0, fs, dt, T = float(inv_c), float(t0), float(fs), float(dt), int(T)
    if n_threads is None:
        n_threads = _get_n_threads()

    d2h = _compute_d2h_mpar(
        points, centers, wx, wy, inv_c, apod, delays, t0, T, fs, dt, n_threads
    )
    dh = integrate_d2h_to_dh(d2h, dt)
    return integrate_dh_to_h(dh, dt)


# ---------------------------------------------------------------------------
# PE SDI (combined pulse-echo SDI) — 16 deltas per (m_e, m_r) pair
# ---------------------------------------------------------------------------


@njit(inline="always")
def _place_pe_sdi_2d(out, p, t0, T, fs, t1e, t2e, t3e, t4e, t1r, t2r, t3r, t4r, weight):
    """Place 16 PE SDI deltas for one (m_e, m_r) pair into out[p, :].

    Each of 4 TX corners × 4 RX corners produces one delta at t_e + t_r.
    Linear interpolation splits each delta across two adjacent bins (32 writes).
    """
    # Static sign arrays: +1, -1, -1, +1 for corners 1–4.
    signs_e_0 = np.float32(1.0)
    signs_e_1 = np.float32(-1.0)
    signs_e_2 = np.float32(-1.0)
    signs_e_3 = np.float32(1.0)
    signs_r_0 = np.float32(1.0)
    signs_r_1 = np.float32(-1.0)
    signs_r_2 = np.float32(-1.0)
    signs_r_3 = np.float32(1.0)

    # Unrolled 4×4 loop for Numba performance.
    # TX corner 0 (sign +1) × all RX corners
    for i_r in range(4):
        if i_r == 0:
            s_r = signs_r_0
            t_r = t1r
        elif i_r == 1:
            s_r = signs_r_1
            t_r = t2r
        elif i_r == 2:
            s_r = signs_r_2
            t_r = t3r
        else:
            s_r = signs_r_3
            t_r = t4r

        # TX corner 0
        w0 = signs_e_0 * s_r * weight
        t_event = t1e + t_r
        kf = (t_event - t0) * fs + np.float32(1.0)
        kf_floor = int(np.floor(kf))
        w_ceil = kf - kf_floor
        w_floor = np.float32(1.0) - w_ceil
        if 0 <= kf_floor < T:
            out[p, kf_floor] += w0 * w_floor
        if 0 <= kf_floor + 1 < T:
            out[p, kf_floor + 1] += w0 * w_ceil

        # TX corner 1
        w1 = signs_e_1 * s_r * weight
        t_event = t2e + t_r
        kf = (t_event - t0) * fs + np.float32(1.0)
        kf_floor = int(np.floor(kf))
        w_ceil = kf - kf_floor
        w_floor = np.float32(1.0) - w_ceil
        if 0 <= kf_floor < T:
            out[p, kf_floor] += w1 * w_floor
        if 0 <= kf_floor + 1 < T:
            out[p, kf_floor + 1] += w1 * w_ceil

        # TX corner 2
        w2 = signs_e_2 * s_r * weight
        t_event = t3e + t_r
        kf = (t_event - t0) * fs + np.float32(1.0)
        kf_floor = int(np.floor(kf))
        w_ceil = kf - kf_floor
        w_floor = np.float32(1.0) - w_ceil
        if 0 <= kf_floor < T:
            out[p, kf_floor] += w2 * w_floor
        if 0 <= kf_floor + 1 < T:
            out[p, kf_floor + 1] += w2 * w_ceil

        # TX corner 3
        w3 = signs_e_3 * s_r * weight
        t_event = t4e + t_r
        kf = (t_event - t0) * fs + np.float32(1.0)
        kf_floor = int(np.floor(kf))
        w_ceil = kf - kf_floor
        w_floor = np.float32(1.0) - w_ceil
        if 0 <= kf_floor < T:
            out[p, kf_floor] += w3 * w_floor
        if 0 <= kf_floor + 1 < T:
            out[p, kf_floor + 1] += w3 * w_ceil


@njit(parallel=True, fastmath=True)
def _compute_pe_sdi_ppar(
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
            # RX patch → scatterer direction.
            dx_r = points[p, 0] - rx_centers[m_r, 0]
            dy_r = points[p, 1] - rx_centers[m_r, 1]
            dz_r = points[p, 2] - rx_centers[m_r, 2]
            dist_r = np.sqrt(dx_r * dx_r + dy_r * dy_r + dz_r * dz_r)
            if dist_r < np.float32(1e-12):
                continue
            xp_r = dx_r / dist_r
            yp_r = dy_r / dist_r
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
                # TX patch → scatterer direction.
                dx_e = points[p, 0] - tx_centers[m_e, 0]
                dy_e = points[p, 1] - tx_centers[m_e, 1]
                dz_e = points[p, 2] - tx_centers[m_e, 2]
                dist_e = np.sqrt(dx_e * dx_e + dy_e * dy_e + dz_e * dz_e)
                if dist_e < np.float32(1e-12):
                    continue
                xp_e = dx_e / dist_e
                yp_e = dy_e / dist_e
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
                if h_max_e < np.float32(1e-6):
                    continue
                slope_e = h_max_e / (t2e - t1e)

                weight = slope_r * slope_e
                _place_pe_sdi_2d(
                    out,
                    p,
                    t0,
                    T,
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


@njit(parallel=True, fastmath=True)
def _compute_pe_sdi_mpar(
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
    n_threads,
):
    """PE SDI: prange over M_r × M_e product, thread-local reduction.

    Use when P < n_threads (e.g. single scatterer). Cumsum after reduction.
    Returns (P, T) float32.
    """
    P = points.shape[0]
    M_e = tx_centers.shape[0]
    M_r = rx_centers.shape[0]
    total_pairs = M_r * M_e
    thread_out = np.zeros((n_threads, P, T), dtype=np.float32)

    for pair_idx in prange(total_pairs):  # ty: ignore[not-iterable]
        tid = numba.get_thread_id()
        out_t = thread_out[tid]
        m_r = pair_idx // M_e
        m_e = pair_idx % M_e

        for p in range(P):
            # RX patch → scatterer.
            dx_r = points[p, 0] - rx_centers[m_r, 0]
            dy_r = points[p, 1] - rx_centers[m_r, 1]
            dz_r = points[p, 2] - rx_centers[m_r, 2]
            dist_r = np.sqrt(dx_r * dx_r + dy_r * dy_r + dz_r * dz_r)
            if dist_r < np.float32(1e-12):
                continue
            xp_r = dx_r / dist_r
            yp_r = dy_r / dist_r
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

            # TX patch → scatterer.
            dx_e = points[p, 0] - tx_centers[m_e, 0]
            dy_e = points[p, 1] - tx_centers[m_e, 1]
            dz_e = points[p, 2] - tx_centers[m_e, 2]
            dist_e = np.sqrt(dx_e * dx_e + dy_e * dy_e + dz_e * dz_e)
            if dist_e < np.float32(1e-12):
                continue
            xp_e = dx_e / dist_e
            yp_e = dy_e / dist_e
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
            if h_max_e < np.float32(1e-6):
                continue
            slope_e = h_max_e / (t2e - t1e)

            weight = slope_r * slope_e
            _place_pe_sdi_2d(
                out_t,
                p,
                t0,
                T,
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

    # Reduce threads.
    out = np.zeros((P, T), dtype=np.float32)
    for t_idx in range(n_threads):
        for p in range(P):
            for k in range(T):
                out[p, k] += thread_out[t_idx, p, k]

    # Integrate: 1 cumsum → Dh_pe (delta distribution, no dt scaling needed).
    for p in range(P):
        acc = np.float64(0.0)
        for k in range(T):
            acc += np.float64(out[p, k])
            out[p, k] = np.float32(acc)
    return out


def _run_pe_sdi(
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
    parallel_axis,
):
    """Dispatch PE SDI to points-parallel or patches-parallel kernel."""
    if parallel_axis == "points":
        return _compute_pe_sdi_ppar(
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
        )
    return _compute_pe_sdi_mpar(
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
        _get_n_threads(),
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
    parallel_axis="points",
    batch_size_points=None,
):
    """Pulse-echo SDI: Dh_pe = dh^e *_t d2h^r via combined delta placement.

    Computes Dh_pe for one RX element (rx patches = patches of that element).
    For full Reception: call once per RX element with element-filtered rx patches.

    16 deltas per (m_e, m_r) pair → 32 sample writes → 1 cumsum.

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
    parallel_axis : {"points", "patches"}
        Parallelism axis. Default ``"points"`` (prange over P).
        Use ``"patches"`` when P < n_threads.
    batch_size_points : int or None
        Chunk P into batches at Python level. None = no batching.

    Returns
    -------
    Dh_pe : (P, T) float32
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
    P = points.shape[0]

    if batch_size_points is None or batch_size_points >= P:
        return _run_pe_sdi(
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
            parallel_axis,
        )

    out = np.zeros((P, T), dtype=np.float32)
    for start in range(0, P, batch_size_points):
        end = min(start + batch_size_points, P)
        out[start:end] = _run_pe_sdi(
            points[start:end],
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
            parallel_axis,
        )
    return out


# TODO: Future — compute_pe_sdi_per_element variant
# Would output (P, E_rx, T) directly using sub_el_idx routing,
# avoiding E_rx separate kernel calls. Trades memory for fewer calls.
# See PROMPT2.md §4 Option B.
