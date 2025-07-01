import math
import time
from time import time as TIME

import numpy as np
import pyvista as pv
import torch
import torch.nn as nn
from numba import njit, prange
from tqdm import tqdm

import pysonogen


def create_simulation_grid(simulation_struct, device="cpu"):
    x0, xf = simulation_struct["x_extent"]
    y0, yf = simulation_struct["y_extent"]
    z0, zf = simulation_struct["z_extent"]
    dx, dy, dz = (
        simulation_struct["dx"],
        simulation_struct["dy"],
        simulation_struct["dz"],
    )

    Nx = int((xf - x0) / dx) if (dx != 0 and abs(xf - x0) > 1e-10) else 1
    Ny = int((yf - y0) / dy) if (dy != 0 and abs(yf - y0) > 1e-10) else 1
    Nz = int((zf - z0) / dz) if (dz != 0 and abs(zf - z0) > 1e-10) else 1
    if Nx % 2 == 0:
        Nx += 1
    if Ny % 2 == 0:
        Ny += 1
    if Nz % 2 == 0:
        Nz += 1
    x = torch.linspace(x0, xf, Nx, device=device)
    y = torch.linspace(y0, yf, Ny, device=device)
    z = torch.linspace(z0, zf, Nz, device=device)
    grid = torch.stack(torch.meshgrid(x, y, z, indexing="ij"), dim=-1)
    return x, y, z, grid.reshape(-1, 3) * 1e-3  # Convert to meters


# JIT-compiled core event computation
# @torch.jit.script

tolerance_apod = 1e-3


def compute_patch_events(
    wx: float,
    wy: float,
    xp: torch.Tensor,
    yp: torch.Tensor,
    dist: torch.Tensor,
    delays: torch.Tensor,
    apods: torch.Tensor,
    c: float,
    fs: float,
) -> torch.Tensor:
    Dt1 = torch.min((wx * xp / c).abs(), (wy * yp / c).abs())
    Dt2 = torch.max((wx * xp / c).abs(), (wy * yp / c).abs()).clamp(min=1.0 / fs)
    area = (wx * wy) / (2 * np.pi * dist)
    t1 = dist / c - 0.5 * (Dt1 + Dt2) + delays
    t2 = t1 + Dt1
    t3 = t1 + Dt2
    t4 = t1 + Dt1 + Dt2
    hmax = area * apods / Dt2
    t1[apods < tolerance_apod] = 0
    t2[apods < tolerance_apod] = 0
    t3[apods < tolerance_apod] = 0
    t4[apods < tolerance_apod] = 0
    hmax[apods < tolerance_apod] = 0
    return torch.stack((t1, t2, t3, t4, hmax), dim=2)


# @torch.jit.script
def accumulate_events_batch(events: torch.Tensor, t_global: torch.Tensor):
    """
    JIT-compiled trapezoidal accumulation over patches and time.
    Batches over points to control memory footprint.

    Args:
        events: Tensor[P, M, 5] holding t1, t2, t3, t4, h_max per patch
        fs: sampling frequency
        t0: global start time
        n2: number of time samples
        batch_size: chunk size for batching over P
    Returns:
        h_out: Tensor[P, n2]
        t_global: Tensor[n2]
    """

    # batch over points
    t1 = events.select(2, 0)  # [B, M]
    t2 = events.select(2, 1)
    t3 = events.select(2, 2)
    t4 = events.select(2, 3)
    h_max = events.select(2, 4)
    # expand for broadcast over time axis
    t1 = t1.unsqueeze(-1)
    t2 = t2.unsqueeze(-1)
    t3 = t3.unsqueeze(-1)
    t4 = t4.unsqueeze(-1)
    h_max = h_max.unsqueeze(-1)
    t = t_global.view(1, 1, -1)

    # piecewise trapezoid
    rising = ((t >= t1) & (t < t2)) * h_max * ((t - t1) / (t2 - t1 + 1e-12))
    plateau = ((t >= t2) & (t < t3)) * h_max
    falling = ((t >= t3) & (t < t4)) * h_max * ((t4 - t) / (t4 - t3 + 1e-12))

    # sum over M patches
    h_batch = rising + plateau + falling

    return h_batch.sum(dim=1)


class TorchField(nn.Module):
    def __init__(self, transducer, device="cpu"):
        super().__init__()
        """ Initializes a TorchField object for ultrasound simulations.
        Args:
            transducer: Transducer object containing transducer parameters.
            device: Device to run the computations on ('cpu' or 'cuda').
        """
        self.device = device
        self.tx = transducer
        self.c = 1540.0
        self.fs = 300e6
        self.fc = transducer.fc
        self.lambda_mm = self.c / self.fc

        el_h = self.tx.el_h / self.tx.no_sub_y
        el_w = self.tx.el_w / self.tx.no_sub_x
        centers, apods, delays = [], [], []

        for elem in range(self.tx.n_elements):
            for sub_elem in range(self.tx.no_sub_x * self.tx.no_sub_y):
                verts = self.tx.sub_quad_verts[
                    elem * (self.tx.no_sub_x * self.tx.no_sub_y) + sub_elem
                ]
                centers.append(verts.mean(axis=0))
                apods.append(self.tx.apodization[elem])
                delays.append(self.tx.delays[elem])

        self.centers = torch.tensor(centers, dtype=torch.float32, device=device)
        self.apods = nn.Parameter(
            torch.tensor(apods, dtype=torch.float32, device=device, requires_grad=True)
        )
        self.delays = nn.Parameter(
            torch.tensor(delays, dtype=torch.float32, device=device, requires_grad=True)
        )
        self.wx = el_w
        self.wy = el_h

        self.field = None
        self.x = self.y = self.z = None
        print(f"Initialized TorchField on {device}")

    def spatial_impulse_response(self, field_points, batch_size=100, return_all=False):
        if not isinstance(field_points, torch.Tensor):
            try:
                # Only use the grid_points (last element of the tuple)
                *_, field_points = create_simulation_grid(
                    field_points, device=self.device
                )
            except Exception as e:
                raise ValueError(
                    "Invalid field_points input. It should be a numpy array or a dictionary with simulation parameters."
                ) from e
        start_time = TIME()
        pts = torch.atleast_2d(
            torch.tensor(field_points, device=self.device, dtype=torch.float32)
        )
        P, M = pts.shape[0], self.centers.shape[0]

        # Vectorized distance and direction calculations
        diff = pts.unsqueeze(1) - self.centers.unsqueeze(0)  # (P, M, 3)
        dist = torch.norm(diff, dim=-1)

        xp = diff[..., 0] / dist
        yp = diff[..., 1] / dist

        # Vectorized SIR computation
        print(f"Computing SIR for {P} points and {M} patches...")
        events = compute_patch_events(
            self.wx,
            self.wy,
            xp,
            yp,
            dist,
            self.delays.unsqueeze(0).expand(P, M),
            self.apods.unsqueeze(0).expand(P, M),
            self.c,
            self.fs,
        )
        events_time = TIME()

        print(
            f"Events patch - field points computed in: {events_time - start_time:.4f} seconds."
        )

        # Time vector setup
        all_times = events[..., :4].contiguous().view(-1)
        t0 = all_times.min()
        tN = all_times.max()
        print(f"Time range: {t0} to {tN} seconds.")
        num_samples = int(torch.ceil((tN - t0) * self.fs).item())
        # n2 = 2 ** max(int(math.ceil(math.log2(num_samples))), 5)
        P, M, _ = events.shape
        dt = 1.0 / self.fs
        # global time axis
        t_global = t0 + torch.arange(num_samples, device=events.device) * dt
        h_out = torch.zeros(P, num_samples, device=events.device)

        # Optimized accumulation
        print(f"Accumulating events for {P} points over {num_samples} time samples.")
        for start in tqdm(range(0, P, batch_size), unit="batch"):
            # Process in batches to avoid memory issues
            end = min(start + batch_size, P)
            batch_events = events[start:end]
            if batch_events.numel() == 0:
                continue
            # Accumulate contributions for this batch
            h_batch = accumulate_events_batch(batch_events, t_global)
            h_out[start:end] = h_batch
            torch.cuda.empty_cache()  # Clear cache to manage memory
        print(f"Accumulation of events elapsed in: {TIME() - events_time:.4f} seconds.")
        print(f"SIR computed in {time.time() - events_time:.2f} seconds")

        if return_all:
            # Return both the time vector and the impulse response
            return t_global, h_out.T, events

        return t0, h_out.T

    def compute_pr_from_sir(self, h_sir, x, y, z):
        # Reshape to spatial dimensions
        print(f"Computing pressure field from SIR with shape: {h_sir.shape}")
        n_time, n_points = h_sir.shape
        h_sir_4d = h_sir.T.view(-1, len(y), len(x), len(z)).permute(1, 2, 3, 0)

        # FFT processing
        h_sir_fft = torch.fft.fft(h_sir_4d, dim=-1)
        freqs = torch.fft.fftfreq(n_time, 1 / self.fs, device=self.device)
        idx_fc = torch.argmin(torch.abs(freqs - self.fc))
        pressure = torch.abs(h_sir_fft[..., idx_fc])
        return pressure

    def forward(self, field_info, normalize=False, inplace=False):
        x, y, z, grid_points = create_simulation_grid(field_info, self.device)
        print(
            f"Grid created with shape: {grid_points.shape}, x: {len(x)}, y: {len(y)}, z: {len(z)}"
        )
        _, h_sir = self.spatial_impulse_response(grid_points)
        pressure_field = self.compute_pr_from_sir(h_sir, x, y, z)

        if normalize:
            pressure_field /= pressure_field.max()

        if inplace:
            self.field = pressure_field
            self.x, self.y, self.z = x, y, z
        print("Pressure field computed succesfully")
        return pressure_field, x, y, z


# --------------------------------------


def create_simulation_grid(simulation_struct):
    """
    Create a simulation mesh for the ultrasound field.

    Parameters
    ----------
    simulation_grid_dict : dict
        Dictionary containing the simulation parameters:
        - x_extent : list
            The extent of the simulation in the x direction (in mm).
        - y_extent : list
            The extent of the simulation in the y direction (in mm).
        - z_extent : list
            The extent of the simulation in the z direction (in mm).
        - dx : float
            The grid spacing in the x direction (in mm).
        - dy : float
            The grid spacing in the y direction (in mm).
        - dz : float
            The grid spacing in the z direction (in mm).

    Returns
    -------
    grid_points : ndarray
        Array of points in the simulation space.
    """
    # Create a grid of points in the simulation space
    [x0, xf], [y0, yf], [z0, zf] = (
        simulation_struct["x_extent"],
        simulation_struct["y_extent"],
        simulation_struct["z_extent"],
    )
    dx, dy, dz = (
        simulation_struct["dx"],
        simulation_struct["dy"],
        simulation_struct["dz"],
    )

    Nx = int((xf - x0) / dx) if (dx != 0 and abs(xf - x0) > 1e-10) else 1
    Ny = int((yf - y0) / dy) if (dy != 0 and abs(yf - y0) > 1e-10) else 1
    Nz = int((zf - z0) / dz) if (dz != 0 and abs(zf - z0) > 1e-10) else 1
    if Nx % 2 == 0:
        Nx += 1
    if Ny % 2 == 0:
        Ny += 1
    if Nz % 2 == 0:
        Nz += 1

    x = np.linspace(x0, xf, Nx)
    y = np.linspace(y0, yf, Ny)
    z = np.linspace(z0, zf, Nz)
    # Create a meshgrid of points
    grid_points = np.array(np.meshgrid(x, y, z)).T.reshape(-1, 3)

    return x, y, z, grid_points * 1e-3


pi = np.pi
tolerance_apod = 1e-3


@njit
def compute_patch_sir(wx, wy, xp, yp, l, c0, apod, delay, sampling_rate_Hz, lambda_mm):
    # Common sampling rate is 100 MHz
    # Then minimum time step is 0.01 us,
    # if apod < tolerance_apod:
    #     return 0, 0, 0, 0, 0
    epsilon = 1 / (sampling_rate_Hz)  # 1 ns
    Dt1 = min(abs(wx * xp / c0), abs(wy * yp / c0))
    Dt2 = max(abs(wx * xp / c0), abs(wy * yp / c0))
    area = wx * wy / (2 * pi * l)

    t1 = l / c0 - 0.5 * (Dt1 + Dt2) + delay
    t2 = t1 + Dt1
    t3 = t1 + Dt2
    t4 = t1 + Dt1 + Dt2

    if 2 * Dt2 < epsilon:
        Dt2 = epsilon
    # trapezoid area
    h_max = area * apod / Dt2
    return t1, t2, t3, t4, h_max


@njit(parallel=True)
def compute_all_events(
    P, M, pts, centers, wx, wy, c, apods, delays, events, sampling_rate_Hz, lambda_mm
):
    for p in prange(P):
        for i in range(M):
            dx = pts[p, 0] - centers[i, 0]
            dy = pts[p, 1] - centers[i, 1]
            dz = pts[p, 2] - centers[i, 2]
            dist = np.sqrt(dx * dx + dy * dy + dz * dz)

            xp, yp = dx / (dist), dy / (dist)
            t1, t2, t3, t4, h_max = compute_patch_sir(
                wx,
                wy,
                xp,
                yp,
                dist,
                c,
                apods[i],
                delays[i],
                sampling_rate_Hz,
                lambda_mm,
            )
            events[p, i, 0] = t1
            events[p, i, 1] = t2
            events[p, i, 2] = t3
            events[p, i, 3] = t4
            events[p, i, 4] = h_max


@njit(parallel=True)
def accumulate_from_events(P, M, events, fs, t0, h_out):
    """
    Parallel accumulation of trapezoidal SIR contributions for all patches.
    events shape: (P, M, 5) storing t1, t2, t3, t4, h_max.
    h_out: (P, n2) output array, t0: start time, fs: sampling rate.
    """
    dt = 1.0 / fs
    n2 = h_out.shape[1]
    for p in prange(P):
        for i in range(M):
            t1, t2, t3, t4, h_max = (
                events[p, i, 0],
                events[p, i, 1],
                events[p, i, 2],
                events[p, i, 3],
                events[p, i, 4],
            )

            # find the first/last sample indices that could possibly overlap
            k_start = int(np.floor((t1 - t0) * fs) + 1)
            k_end = int(np.ceil((t4 - t0) * fs) + 1)

            # clamp to valid range
            if k_end < 0 or k_start >= n2:
                continue
            if k_start < 0:
                k_start = 0
            if k_end > n2:
                k_end = n2

            # loop over every sample that might see part of this trapezoid
            for k in range(k_start, k_end):
                t = t0 + k * dt
                # evaluate continuous trapezoid h(t)
                if t < t1 or t >= t4:
                    continue
                elif t < t2:
                    h = h_max * ((t - t1) / (t2 - t1))
                elif t < t3:
                    h = h_max
                else:
                    h = h_max * ((t4 - t) / (t4 - t3))

                # accumulate
                h_out[p, k] += h


def spatial_impulse_response(self, field_points, return_all=False):
    start_comput_time = TIME()
    if not isinstance(field_points, np.ndarray):
        try:
            # Only use the grid_points (last element of the tuple)
            *_, field_points = create_simulation_grid(field_points)
        except Exception as e:
            raise ValueError(
                "Invalid field_points input. It should be a numpy array or a dictionary with simulation parameters."
            ) from e

    pts = np.atleast_2d(field_points).astype(np.float32)
    P, M = pts.shape[0], self.centers.shape[0]

    print(f"Computing SIR for {P} points and {M} patches...")
    # allocate events
    events = np.zeros((P, M, 5), dtype=np.float32)
    # tqdm.write("Computing all patch events...")
    compute_all_events(
        P,
        M,
        pts,
        self.centers,
        self.wx,
        self.wy,
        self.c,
        self.apods,
        self.delays,
        events,
        self.fs,
        self.lambda_mm,
    )
    events_time = TIME()
    print(
        f"Events patch - field points computed in: {events_time - start_comput_time:.4f} seconds."
    )
    # build global time vector from real event times
    all_times = np.unique(events[:, :, 0:4].ravel())
    all_times.sort()
    t0, tN = all_times[0], all_times[-1]
    # create sampling grid
    dt = 1.0 / self.fs
    num_samples = int(np.ceil((tN - t0) * self.fs))
    # next power of two
    n2 = 2 ** max(int(np.ceil(np.log2(num_samples))) - 1, 5)
    t_global = t0 + np.arange(n2, dtype=np.float32) * dt
    h_out = np.zeros((P, n2), dtype=np.float32)
    # tqdm.write("Accumulating SIR from events...")
    accumulate_from_events(P, M, events, self.fs, t0, h_out)
    print(f"Accumulation of events elapsed in: {TIME() - events_time:.4f} seconds.")

    print(f"SIR computed in {TIME() - start_comput_time:.4f} seconds.")
    if return_all:
        return t_global, h_out.T, events
    return t0, h_out.T


# -----------------------------------------------------


# JIT-compiled core event computation (unchanged)
@torch.jit.script
def compute_patch_events_batch(
    wx: float,
    wy: float,
    diff: torch.Tensor,
    dist: torch.Tensor,
    delays: torch.Tensor,
    apods: torch.Tensor,
    c: float,
    fs: float,
) -> torch.Tensor:
    xp = diff[..., 0] / dist
    yp = diff[..., 1] / dist
    Dt1 = torch.min((wx * xp / c).abs(), (wy * yp / c).abs())
    Dt2 = torch.max((wx * xp / c).abs(), (wy * yp / c).abs()).clamp(min=1.0 / fs)
    area = (wx * wy) / (2 * math.pi * dist)
    t1 = dist / c - 0.5 * (Dt1 + Dt2) + delays
    t2 = t1 + Dt1
    t3 = t1 + Dt2
    t4 = t1 + Dt1 + Dt2
    hmax = area * apods / Dt2
    # mask = apods < 1e-3
    # t1.masked_fill_(mask, 0)
    # t2.masked_fill_(mask, 0)
    # t3.masked_fill_(mask, 0)
    # t4.masked_fill_(mask, 0)
    # hmax.masked_fill_(mask, 0)
    return torch.stack((t1, t2, t3, t4, hmax), dim=-1)


# Vectorized accumulation via derivative + prefix-sum
@torch.jit.script
def accumulate_events_derivative(
    events: torch.Tensor,  # [B, M, 5]
    t0: float,
    dt: float,
    T: int,
) -> torch.Tensor:
    B, M, _ = events.shape
    # Extract and flatten
    t1 = events[..., 0].reshape(B, M)
    t2 = events[..., 1].reshape(B, M)
    t3 = events[..., 2].reshape(B, M)
    t4 = events[..., 3].reshape(B, M)
    hmax = events[..., 4].reshape(B, M)

    # Compute slopes
    s1 = hmax / (t2 - t1 + 1e-12)
    s2 = hmax / (t4 - t3 + 1e-12)

    # Compute sample indices
    fs = 1.0 / dt
    k1 = torch.floor((t1 - t0) * fs).long() + 1
    k2 = torch.floor((t2 - t0) * fs).long() + 1
    k3 = torch.ceil((t3 - t0) * fs).long() + 1
    k4 = torch.ceil((t4 - t0) * fs).long() + 1

    # Clamp indices to valid range [0, T]
    if t4.min() < t0 or t4.max() > T * dt:
        raise ValueError("Event times are out of bounds for the given time range.")

    # Slopes
    s1 = hmax / (t2 - t1 + 1e-12)
    s2 = hmax / (t4 - t3 + 1e-12)

    dH = torch.zeros(B, T, device=events.device)
    # Only include valid events
    # Mask invalid events
    valid = t4 > t1
    mask_flat = valid.reshape(-1)
    idx1 = k1.reshape(-1)[mask_flat]
    idx2 = k2.reshape(-1)[mask_flat]
    idx3 = k3.reshape(-1)[mask_flat]
    idx4 = k4.reshape(-1)[mask_flat]
    s1f = s1.reshape(-1)[mask_flat]
    s2f = s2.reshape(-1)[mask_flat]

    # Up-ramp: +s1 at k1, -s1 at k2
    dH.view(-1).index_add_(0, idx1, s1f)
    dH.view(-1).index_add_(0, idx2, -s1f)
    # Down-ramp: -s2 at k3, +s2 at k4
    dH.view(-1).index_add_(0, idx3, s2f)
    dH.view(-1).index_add_(0, idx4, -s2f)
    H = torch.cumsum(dH, dim=1) * dt

    return H


def spatial_impulse_response(self, field_points, batch_size=1024, return_all=False):
    if not isinstance(field_points, torch.Tensor):
        try:
            # Only use the grid_points (last element of the tuple)
            *_, field_points = create_simulation_grid(field_points, device=self.device)
        except Exception as e:
            raise ValueError(
                "Invalid field_points input. It should be a numpy array or a dictionary with simulation parameters."
            ) from e
    pts = torch.tensor(field_points, dtype=torch.float32, device=self.device)
    P = pts.shape[0]
    # Precompute global time
    max_d = (pts.max(0).values - self.centers.min(0).values).norm()
    max_time = (
        max_d / self.c + self.delays.max() + 0.5 * (self.wx + self.wy) / self.c
    ).item()
    T = int(math.ceil(max_time * self.fs))
    dt = 1.0 / self.fs
    t_global = torch.arange(T, device=self.device) * dt
    H = torch.zeros(P, T, device=self.device)
    print(
        f"Computing spatial impulse response for {P} points with {self.centers.shape[0]} patches..."
    )
    with torch.no_grad():
        for i in range(0, P, batch_size):
            print(
                f"Processing batch {i // batch_size + 1} of {math.ceil(P / batch_size)}"
            )
            j = min(i + batch_size, P)
            batch = pts[i:j]
            diff = batch.unsqueeze(1) - self.centers.unsqueeze(0)
            dist = diff.norm(dim=-1)
            events = compute_patch_events_batch(
                self.wx,
                self.wy,
                diff,
                dist,
                self.delays.unsqueeze(0).expand(j - i, -1),
                self.apods.unsqueeze(0).expand(j - i, -1),
                self.c,
                self.fs,
            )
            H[i:j] = accumulate_events_derivative(events, 0.0, dt, T)

    if return_all:
        return t_global, H.T, events
    return t_global, H.T
