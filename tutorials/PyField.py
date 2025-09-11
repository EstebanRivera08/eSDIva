import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from numba import njit, prange
from tqdm import tqdm

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
    # time-of-flight base
    base = dist * inv_c - 0.5 * (Dt1 + Dt2) + delay
    t1 = base
    t2 = base + Dt1
    t3 = base + Dt2
    t4 = base + Dt1 + Dt2
    h_max = area * apod / Dt2

    return t1, t2, t3, t4, h_max


# ---------- main SIR computation function (njit, parallel) ----------
@njit(parallel=True, fastmath=True)
def compute_parallelized_sir_optimized(
    P,
    M,
    T,
    wx,
    wy,
    xp,
    yp,
    dist,
    inv_c,
    apodization,
    delays,
    time_grid,
    t0,
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

    # precompute threshold term for auto decision (8 + 2*T/M)
    threshold_term = 8.0 + 2.0 * (T / M)

    for p in prange(P):
        # per-point local event buffers for SDI (max 4*M entries)
        idxs = np.empty(8 * M, dtype=np.int32)
        vals = np.empty(8 * M, dtype=np.float32)

        for m in range(M):
            t1, t2, t3, t4, h_max = compute_rectangle_SIR_params(
                wx,
                wy,
                xp[p, m],
                yp[p, m],
                dist[p, m],
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
            if k_start < 0:
                k_start = 0
            if k_end > T:
                k_end = T

            range_k = k_end - k_start
            range_k_matrix[p, m] = range_k

            # decide method for this patch
            use_naive = True

            if method_flag == 1:  # sdi
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
                k1f = (t1 - t0) * fs
                k4f = (t4 - t0) * fs
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
                kf = (t2 - t0) * fs
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
                kf = (t3 - t0) * fs
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
        acc = 0.0
        for k in range(T):
            acc += d2h[p, k]
            d2h[p, k] = acc
        acc2 = 0.0
        for k in range(T):
            acc2 += d2h[p, k]
            # multiply by dt once to match continuous integral scaling
            h_out[p, k] += acc2 * dt

    return h_out, range_k_matrix


@njit(parallel=True)
def compute_distance_patch_to_point(P, M, pts, center):
    dist = np.zeros((P, M), dtype=np.float32)
    xp = np.zeros((P, M), dtype=np.float32)
    yp = np.zeros((P, M), dtype=np.float32)
    for p in prange(P):
        for m in range(M):
            dx = pts[p, 0] - center[m, 0]
            dy = pts[p, 1] - center[m, 1]
            dz = pts[p, 2] - center[m, 2]

            dist_value = np.sqrt(dx * dx + dy * dy + dz * dz)
            inv_dist = 1 / dist_value
            xp[p, m] = dx * inv_dist
            yp[p, m] = dy * inv_dist
            dist[p, m] = dist_value
    return dist, xp, yp


class PyField:
    def __init__(self, transducer):
        self.tx = transducer
        self.c = 1540.0
        self.fs = 200e6  # Hz
        self.fc = transducer.fc  # Hz
        self.lambda_mm = self.c / self.fc
        # compute patch centers/apodization/delays once
        el_h = self.tx.el_h / self.tx.no_sub_y
        el_w = self.tx.el_w / self.tx.no_sub_x
        self.wx = el_w
        self.wy = el_h
        centers, apodization, delays = [], [], []
        for elem in range(self.tx.n_elements):
            for sub_elem in range(self.tx.no_sub_x * self.tx.no_sub_y):
                verts = self.tx.sub_quad_verts[
                    elem * (self.tx.no_sub_x * self.tx.no_sub_y) + sub_elem
                ]
                centers.append(verts.mean(axis=0))
                apodization.append(self.tx.apodization[elem])
                delays.append(self.tx.delays[elem])
        self.centers = np.array(centers, dtype=np.float32)
        self.apodization = np.array(apodization, dtype=np.float32)
        self.delays = np.array(delays, dtype=np.float32)
        self.M = len(centers)
        self.range_k = None
        self.mean_range_k_log = []
        self.T_log = []
        self.P_log = []
        self.sir_running_time_log = []

    def compute_sir(self, points, *, method="auto"):
        if method not in ["auto", "naive", "sdi", None]:
            raise ValueError("method must be None or 'auto', 'naive', or 'sdi'.")
        if method == "naive":
            method = 0
        elif method == "sdi":
            method = 1
        else:
            method = None

        P, M = points.shape[0], self.M

        print(f"Computing SIR for {P} points and {M} patches...")
        dist, xp, yp = compute_distance_patch_to_point(P, M, points, self.centers)
        time_grid, t0, dt, T = self._compute_time_grid(dist)

        startSIR = time.time()

        h_sir, self.range_k = compute_parallelized_sir_optimized(
            P,
            M,
            T,
            self.wx,
            self.wy,
            xp,
            yp,
            dist,
            1 / self.c,
            self.apodization,
            self.delays,
            time_grid,
            t0,
            self.fs,
            dt,
            method_flag=method,  # 0 -> naive, 1 -> sdi, 2 -> auto
        )

        runtime_sir = time.time() - startSIR
        # Store information
        self.P_log.append(P)
        self.T_log.append(T)
        self.mean_range_k_log.append(np.mean(self.range_k))
        self.sir_running_time_log.append(runtime_sir)
        print(f"SIR computed in {runtime_sir:.2f} seconds...")
        return t0, h_sir.T

    def from_sir_to_pressure(self, h_sir, x, y, z, batch_size=8192, max_workers=None):
        """
        Compute the pressure field from the Spatial Impulse Response (SIR) in parallel.
        """
        start_time = time.time()
        n_points = h_sir.shape[1]
        # Frequency vector
        freq_vect = np.linspace(0, self.fs, h_sir.shape[0])
        idx = np.argmin((freq_vect - self.fc) ** 2)

        fft_results = np.zeros(n_points, dtype=np.float32)

        def process_batch(start):
            end = min(start + batch_size, n_points)
            fft_batch = np.fft.fft(h_sir[:, start:end], axis=0)
            return start, end, np.abs(fft_batch[idx, :])

        # Parallel loop
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for start, end, vals in executor.map(
                process_batch, range(0, n_points, batch_size)
            ):
                fft_results[start:end] = vals

        # Reshape back to 3D grid
        amp_sir_at_tx_freq = fft_results.reshape(len(z), len(x), len(y)).transpose(
            1, 2, 0
        )
        print(
            f"Pressure computed from SIR in {time.time() - start_time:.2f} seconds..."
        )
        return amp_sir_at_tx_freq

    def __call__(self, field_points_mm, *, method="auto"):
        start = time.time()
        x, y, z, points = self._check_points(field_points_mm)
        t0, h_sir = self.compute_sir(points, method=method)
        pressure_field = self.from_sir_to_pressure(h_sir, x, y, z)
        print(f"Pressure field computed in {time.time() - start:.2f} seconds...")
        return x, y, z, pressure_field

    def _check_points(self, field_points_mm):
        start = time.time()
        if isinstance(field_points_mm, list):
            field_points_mm = np.array(field_points_mm)
        elif isinstance(field_points_mm, np.ndarray):
            pass
        else:
            raise ValueError("field_points_mm must be a list or numpy array")

        pts = np.atleast_2d(field_points_mm).astype(np.float32)
        # Check
        x = np.sort(np.unique(pts[:, 0]))
        y = np.sort(np.unique(pts[:, 1]))
        z = np.sort(np.unique(pts[:, 2]))
        spatial_grid = np.array(np.meshgrid(x, y, z)).T.reshape(-1, 3)

        print(
            f"Points checked and spatial grid created in {time.time() - start:.2f} seconds..."
        )
        return x, y, z, spatial_grid * 1e-3

    def _compute_time_grid(self, dist):
        start = time.time()
        min_dist = dist.min()
        max_dist = dist.max()
        max_delay = self.delays.max()
        size_patch = self.wx + self.wy

        # Compute min and max time
        # t1 = min_l/c - 0.5*(max_Dt1 + max_Dt2) + min_delay
        # t4 = t1 + Dt1 + Dt2 = min_/c + 0.5*(max_Dt1 + max_Dt2) + max_delay
        # Dt1 and Dt2 max are wx/c and wy/c respectively
        # So:
        min_time = (min_dist - 0.5 * size_patch) / self.c  # us (or unit)
        min_time = max(min_time, 0.0)

        max_time = (max_dist + 0.5 * size_patch) / self.c + max_delay  # us (or unit)

        dt = 1.0 / self.fs
        T = int(np.ceil((max_time - min_time) * self.fs))
        # next power of two
        t_grid = min_time + np.arange(T, dtype=np.float32) * dt
        print(
            f"Computed time grid from {min_time * 1e6:.2f} us to {max_time * 1e6:.2f} us, with {T} samples in {time.time() - start:.2f} seconds."
        )
        return t_grid, min_time, dt, T
