import math
from time import time as TIME

import torch
import torch.nn as nn
from torch import Tensor
from tqdm import tqdm


# Is working with 1 points, but once we have more points the indexation is not working properly. If
# interested in solving of this, this might be a good place to start:
# --- JIT-compiled core event computation (unchanged, but output in μs) ---
# @torch.jit.script
def compute_patch_events_batch(
    wx: float,
    wy: float,
    diff: Tensor,
    dist: Tensor,
    delays: Tensor,
    apods: Tensor,
    c: float,
    fs: float,
) -> Tensor:
    # Compute times in seconds, then convert to microseconds
    xp = diff[..., 0] / dist
    yp = diff[..., 1] / dist
    Dt1 = torch.min((wx * xp / c).abs(), (wy * yp / c).abs())
    Dt2 = torch.max((wx * xp / c).abs(), (wy * yp / c).abs()).clamp(min=1.0 / fs)
    area = (wx * wy) / (2 * math.pi * dist)
    t1 = (dist / c - 0.5 * (Dt1 + Dt2) + delays) * 1e6
    t2 = t1 + Dt1 * 1e6
    t3 = t1 + Dt2 * 1e6
    t4 = t1 + (Dt1 + Dt2) * 1e6
    hmax = area * apods / Dt2
    return torch.stack((t1, t2, t3, t4, hmax), dim=-1)


# --- Improved accumulation using direct masked updates ---
# @torch.jit.script
def accumulate_events_derivative(
    events: Tensor,  # [B, M, 5]
    t_us: float,
) -> Tensor:
    # events times already in μs
    B, M, _ = events.shape
    # time axis: 0..T-1, each representing t = idx * dt_us in μs
    device = events.device
    T = len(t_us)
    H = torch.zeros(B, T, device=device)

    # loop over patches vectorized by batch and patch
    t1 = events[..., 0]
    t2 = events[..., 1]
    t3 = events[..., 2]
    t4 = events[..., 3]
    hmax = events[..., 4]

    s1 = hmax / ((t2 - t1) + 1e-12)
    s2 = hmax / ((t4 - t3) + 1e-12)

    # masks for each segment [B,M,T]
    mask_rise = (t_us >= t1.unsqueeze(-1)) & (t_us < t2.unsqueeze(-1))
    mask_flat = (t_us >= t2.unsqueeze(-1)) & (t_us < t3.unsqueeze(-1))
    mask_fall = (t_us >= t3.unsqueeze(-1)) & (t_us < t4.unsqueeze(-1))

    # accumulate H via broadcasting
    H = (
        mask_rise * ((t_us - t1.unsqueeze(-1)) * (s1.unsqueeze(-1)))
        + mask_flat * (hmax.unsqueeze(-1))
        + mask_fall * ((t4.unsqueeze(-1) - t_us) * (s2.unsqueeze(-1)))
    )

    # sum over patches
    return H.sum(dim=1)


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


class PyFieldTorch(nn.Module):
    def __init__(self, transducer, device="cpu"):
        super().__init__()
        self.device = torch.device(device)
        self.tx = transducer
        self.c = 1540.0
        self.fs = 300e6
        self.fc = transducer.fc
        self.wx = transducer.el_w / transducer.no_sub_x
        self.wy = transducer.el_h / transducer.no_sub_y
        centers, apods, delays = [], [], []
        for elem in range(transducer.n_elements):
            for sub in range(transducer.no_sub_x * transducer.no_sub_y):
                verts = transducer.sub_quad_verts[
                    elem * (transducer.no_sub_x * transducer.no_sub_y) + sub
                ]
                centers.append(verts.mean(axis=0))
                apods.append(transducer.apodization[elem])
                delays.append(transducer.delays[elem])
        self.centers = torch.tensor(centers, dtype=torch.float32, device=self.device)
        self.apods = torch.tensor(apods, dtype=torch.float32, device=self.device)
        self.delays = torch.tensor(delays, dtype=torch.float32, device=self.device)

    # --- Full spatial_impulse_response using μs units ---
    def spatial_impulse_response(self, field_points, batch_size=100, return_all=False):
        if not isinstance(field_points, torch.Tensor):
            *_, field_points = create_simulation_grid(field_points, device=self.device)

        start_time = TIME()
        pts = field_points.to(torch.float32).to(self.device)
        P = pts.shape[0]

        # Precompute time range
        with torch.no_grad():
            dists = (pts.unsqueeze(1) - self.centers.unsqueeze(0)).norm(dim=-1)
            max_d = dists.max().item()
            min_d = dists.min().item()
            del dists
        max_time_s = (
            max_d / self.c + self.delays.max() + 0.5 * (self.wx + self.wy) / self.c
        )
        min_time_s = (
            min_d / self.c + self.delays.min() - 0.5 * (self.wx + self.wy) / self.c
        )
        max_time_us = max_time_s * 1e6
        min_time_us = min_time_s * 1e6
        # min_time = 0
        print(f"Time range: {min_time_us:.6f} to {max_time_us:.6f} microseconds.")
        dt_us = (1.0 / self.fs) * 1e6
        T = int(math.ceil((max_time_us - min_time_us) / dt_us))
        t_global = min_time_us + torch.arange(T, device=self.device) * dt_us  # in μs

        H = torch.zeros(P, T, device=self.device)

        print(
            f"Computing spatial impulse response for {P} points with {self.centers.shape[0]} patches..."
        )
        with torch.no_grad():
            for i in tqdm(range(0, P, batch_size), desc="Computing SIR", unit="batch"):
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
                # returns [B, T] already summed over patches
                H[i:j] = accumulate_events_derivative(events, t_global)
        print(f"SIR computed in {TIME() - start_time:.2f} seconds.")
        if return_all:
            return t_global * 1e-6, H.T, events
        return t_global * 1e-6, H.T

    def compute_pr_from_sir(self, h_sir, x, y, z):
        n_time = h_sir.shape[0]
        grid_shape = (len(y), len(x), len(z))
        h4d = h_sir.view(-1, *grid_shape).permute(1, 2, 3, 0)
        fft = torch.fft.fft(h4d, dim=-1)
        freqs = torch.fft.fftfreq(n_time, d=1 / self.fs, device=self.device)
        idx = torch.argmin((freqs - self.fc).abs())
        return fft.abs()[..., idx]

    def forward(self, field_info, normalize=False, batch_size=500):
        x, y, z, pts = create_simulation_grid(field_info, self.device)
        print("Simulation grid created with shape:", pts.shape)
        t, h = self.spatial_impulse_response(pts, batch_size=batch_size)
        pr = self.compute_pr_from_sir(h, x, y, z)
        if normalize:
            pr = pr / pr.max()
        return pr, x, y, z
