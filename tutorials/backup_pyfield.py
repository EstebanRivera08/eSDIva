from time import time as time

import numpy as np
from tqdm import tqdm

pi = np.pi


def compute_rectangle_SIR(wx, wy, xp, yp, l, c0, apod, delay, dt):
    xp_abs = abs(xp) * wx / c0  # us
    yp_abs = abs(yp) * wy / c0
    Dt1 = max(min(xp_abs, yp_abs), dt)
    Dt2 = max(max(xp_abs, yp_abs), dt)

    area = wx * wy / (2 * pi * l)
    t1 = l / c0 - 0.5 * (Dt1 + Dt2) + delay
    t2 = t1 + Dt1
    t3 = t1 + Dt2
    t4 = t1 + Dt1 + Dt2

    h_max = area / Dt2 * apod

    return t1, t2, t3, t4, h_max


def compute_all_events(P, M, pts, centers, wx, wy, c, apodization, delays, events, dt):
    for p in range(P):
        for i in range(M):
            dx = pts[p, 0] - centers[i, 0]
            dy = pts[p, 1] - centers[i, 1]
            dz = pts[p, 2] - centers[i, 2]
            dist = np.sqrt(dx * dx + dy * dy + dz * dz)
            xp, yp = dx / (dist), dy / (dist)
            t1, t2, t3, t4, h_max = compute_rectangle_SIR(
                wx,
                wy,
                xp,
                yp,
                dist,
                c,
                apodization[i],
                delays[i],
                dt,
            )
            events[p, i, 0] = t1
            events[p, i, 1] = t2
            events[p, i, 2] = t3
            events[p, i, 3] = t4
            events[p, i, 4] = h_max


def naive_accumulation_of_events(P, M, T, events, fs, time_grid):
    t0 = time_grid[0]
    h_out = np.zeros((P, T), dtype=np.float32)
    range_k = np.zeros((P, M), dtype=np.int32)
    for p in range(P):
        for i in range(M):
            t1, t2, t3, t4, h_max = (
                events[p, i, 0],
                events[p, i, 1],
                events[p, i, 2],
                events[p, i, 3],
                events[p, i, 4],
            )

            # find the first/last sample indices that could possibly overlap
            s1 = h_max / (t2 - t1)  # area under the rising edge
            k_start = int(np.floor((t1 - t0) * fs))
            k_end = int(np.ceil((t4 - t0) * fs) + 1)

            range_k[p, i] = k_end - k_start

            # clamp to valid range
            if k_end < 0 or k_start >= T - 1:
                continue
            if k_start < 0:
                k_start = 0
            if k_end > T - 1:
                k_end = T - 1

            # loop over every sample that might see part of this trapezoid
            for k in range(k_start, k_end):
                t = time_grid[k]
                # evaluate continuous trapezoid h(t)
                if t < t1 or t >= t4 or h_max < 1e-3:
                    continue
                elif t < t2:
                    h = s1 * (t - t1)
                elif t < t3:
                    h = h_max
                else:
                    h = s1 * (t4 - t)

                # accumulate
                h_out[p, k] += h

    return h_out, range_k


def sdi_accumulation_of_events(P, M, T, events, fs, t_grid):
    """
    Parallel accumulation of trapezoidal SIR contributions for all patches.
    events shape: (P, M, 5) storing t1, t2, t3, t4, h_max.
    h_out: (P, n2) output array, t0: start time, fs: sampling rate.
    """
    t0, tN = t_grid[0], t_grid[-1]
    dt = 1.0 / fs
    d2h_sir = np.zeros(
        (P, T), dtype=np.float32
    )  # extra padding to avoid boundary issues
    for p in range(P):
        for i in range(M):
            t1, t2, t3, t4, h_max = (
                events[p, i, 0],
                events[p, i, 1],
                events[p, i, 2],
                events[p, i, 3],
                events[p, i, 4],
            )
            s1 = h_max / (t2 - t1)
            for ti, sign in zip([t1, t2, t3, t4], [1, -1, -1, 1]):
                if ti < t0 or ti >= tN or h_max < 1e-3:
                    continue
                k_t = (ti - t0) * fs + 1
                k_floor = int(np.floor(k_t))
                k_ceil = k_floor + 1
                k_floor = max(min(k_floor, T - 1), 0)
                k_ceil = max(min(k_ceil, T - 1), 0)
                w_ceil = k_t - k_floor
                w_floor = 1 - w_ceil
                d2h_sir[p, k_floor] += sign * s1 * w_floor
                d2h_sir[p, k_ceil] += sign * s1 * w_ceil

    dh_sir = np.cumsum(d2h_sir, axis=1)
    h_out = np.cumsum(dh_sir, axis=1) * dt
    return h_out


class PyFieldv1:
    def __init__(self, transducer, method="Naive"):
        self.method = method
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
        self.range_k = None
        self.T_M = None

    def compute_sir(self, pts, *, method="Naive", return_all=False):
        start = time()

        P, M = pts.shape[0], self.centers.shape[0]

        print(f"Computing Pressure field for {P} points and {M} patches...")
        # allocate events
        events = np.zeros((P, M, 5), dtype=np.float32)
        # compute all event times and amplitudes
        compute_all_events(
            P,
            M,
            pts,
            self.centers,
            self.wx,
            self.wy,
            self.c,
            self.apodization,
            self.delays,
            events,
            1 / self.fs,
        )
        end_event = time()
        print("Events computed in {:.2f} seconds".format(end_event - start))
        # build global time vector from real event times
        t_grid, t0, T = self._compute_time_grid(events)
        if method == "Naive":
            h_out, range_k = naive_accumulation_of_events(
                P, M, T, events, self.fs, t_grid
            )
            self.range_k = range_k
            self.T_M = T / M
        elif method == "SDI":
            h_out = sdi_accumulation_of_events(P, M, T, events, self.fs, t_grid)
        else:
            raise ValueError("Method must be either 'Naive' or 'SDI'.")

        print(f"SIR computed in {time() - end_event:.2f} seconds...")

        if return_all:
            return t_grid, h_out.T, events
        return t0, h_out.T

    def from_sir_to_pressure(self, h_sir, x, y, z):
        start = time()
        # Perform FFT along the first axis
        h_sir_FT = np.fft.fft(h_sir, axis=0)
        # Generate the frequency vector
        freq_vect = np.linspace(0, self.fs, h_sir_FT.shape[0])

        # Find the index of the desired frequency
        idx_freq = np.argmin((freq_vect - self.fc) ** 2)

        # Amplitude for the given frequency
        monochrom_pressure = (
            np.abs(h_sir_FT[idx_freq, :])
            .reshape(z.shape[0], x.shape[0], y.shape[0])
            .transpose(1, 2, 0)
        )
        print(f"SIR transformed to pressure field in {time() - start:.2f} seconds...")

        return monochrom_pressure

    def __call__(self, field_points_mm):
        start = time()
        x, y, z, points = self._check_points(field_points_mm)
        t0, h_sir = self.compute_sir(points, method=self.method)
        pressure_field = self.from_sir_to_pressure(h_sir, x, y, z)
        print(f"Field computed in {time() - start:.2f} seconds...")
        return x, y, z, pressure_field

    def _check_points(self, field_points_mm):
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
        return x, y, z, spatial_grid * 1e-3

    def _compute_time_grid(self, events):
        all_times = events[:, :, 0:4]
        t0, tN = all_times.min(), all_times.max()
        # create sampling grid
        dt = 1.0 / self.fs
        num_samples = int(np.ceil((tN - t0) * self.fs))
        # next power of two
        T = 2 ** max(int(np.ceil(np.log2(num_samples))), 5)
        t_grid = t0 + np.arange(T, dtype=np.float32) * dt
        print(
            f"Time grid from {t0 * 1e6:.2f} us to {tN * 1e6:.2f} us, with {T} samples."
        )
        return t_grid, t0, T
