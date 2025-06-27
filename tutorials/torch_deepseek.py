import math
import time
from time import time as TIME

import numpy as np
import torch
import torch.nn as nn
from rich.progress import Progress
from tqdm import tqdm


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
        start_time = TIME()
        pts = torch.atleast_2d(
            torch.tensor(field_points, device=self.device, dtype=torch.float32)
        )
        P, M = pts.shape[0], self.centers.shape[0]

        with Progress() as progress:
            for start in progress.track(range(0, P, batch_size), unit="batch"):
                end = min(start + batch_size, P)
                # Vectorized SIR computation
                task = progress.add_task(
                    f"Computing SIR for {P} points and {M} patches...", total=P
                )

                # Vectorized distance and direction calculations
                diff = (
                    pts.unsqueeze(1)[start:end] - self.centers.unsqueeze(0)[start:end]
                )  # (P, M, 3)
                dist = torch.norm(diff, dim=-1)

                xp = diff[..., 0] / dist
                yp = diff[..., 1] / dist

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

                progress.update(
                    task,
                    description=f"Events patch - field points computed in: {events_time - start_time:.4f} seconds.",
                )

                # Time vector setup
                all_times = events[..., :4].contiguous().view(-1)
                t0 = all_times.min()
                tN = all_times.max()
                num_samples = int(torch.ceil((tN - t0) * self.fs).item())
                n2 = 2 ** max(int(math.ceil(math.log2(num_samples))), 5)
                P, M, _ = events.shape
                dt = 1.0 / self.fs
                # global time axis
                t_global = t0 + torch.arange(n2, device=events.device) * dt
                h_out = torch.zeros(P, n2, device=events.device)

                # Optimized accumulation
                progress.update(
                    task,
                    description=f"Accumulating events for {P} points over {n2} time samples.",
                )
                # Process in batches to avoid memory issues
                end = min(start + batch_size, P)
                batch_events = events[start:end]
                if batch_events.numel() == 0:
                    continue
                # Accumulate contributions for this batch
                h_batch = accumulate_events_batch(batch_events, t_global)
                h_out[start:end] = h_batch
                torch.cuda.empty_cache()  # Clear cache to manage memory
                print(
                    f"Accumulation of events elapsed in: {TIME() - events_time:.4f} seconds."
                )
                print(f"SIR computed in {time.time() - events_time:.2f} seconds")

        if return_all:
            # Return both the time vector and the impulse response
            return t_global, h_out.T, events

        return t0, h_out.T


def create_simulation_grid(simulation_struct, device="cpu"):
    x0, xf = simulation_struct["x_extent"]
    y0, yf = simulation_struct["y_extent"]
    z0, zf = simulation_struct["z_extent"]
    dx, dy, dz = (
        simulation_struct["dx"],
        simulation_struct["dy"],
        simulation_struct["dz"],
    )

    Nx = int((xf - x0) / dx)
    Ny = int((yf - y0) / dy)
    Nz = int((zf - z0) / dz)

    if Nx % 2 == 0:
        Nx += 1
    if Ny % 2 == 0:
        Ny += 1
    if Nz % 2 == 0:
        Nz += 1

    x = torch.linspace(x0, xf, Nx, device=device) * 1e-3
    y = torch.linspace(y0, yf, Ny, device=device) * 1e-3
    z = torch.linspace(z0, zf, Nz, device=device) * 1e-3

    grid = torch.stack(torch.meshgrid(x, y, z, indexing="ij"), dim=-1)
    grid_points = grid.reshape(-1, 3)
    return x, y, z, grid_points


# JIT-compiled core event computation
@torch.jit.script
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
    return torch.stack((t1, t2, t3, t4, hmax), dim=2)


@torch.jit.script
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
        start_time = TIME()
        pts = torch.atleast_2d(
            torch.tensor(field_points, device=self.device, dtype=torch.float32)
        )
        P, M = pts.shape[0], self.centers.shape[0]

        # Vectorized distance and direction calculations
        diff = pts.unsqueeze(1) - self.centers.unsqueeze(0)  # (P, M, 3)
        dist = torch.norm(diff, dim=-1)

        max_time = dist.max() / self.c

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
        print(f"Time range: {t0} to {tN}, {max_time}, {max_time > tN} seconds.")
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
        _, h_sir = self.spatial_impulse_response(grid_points.cpu().numpy())
        pressure_field = self.compute_pr_from_sir(h_sir, x, y, z)

        if normalize:
            pressure_field /= pressure_field.max()

        if inplace:
            self.field = pressure_field
            self.x, self.y, self.z = x, y, z
        print("Pressure field computed succesfully")
        return pressure_field, x, y, z
