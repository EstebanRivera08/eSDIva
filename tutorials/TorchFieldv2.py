import math
from time import time as TIME

import torch
import torch.nn as nn


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
    # # Initialize derivative array
    # dH = torch.zeros(B, T, device=events.device)

    # # Create batch indices
    # batch_idx = torch.arange(B, device=events.device)[:, None].expand(B, M)

    # # Up-ramp contributions
    # dH.index_put_((batch_idx, k1), s1, accumulate=True)
    # dH.index_put_((batch_idx, k2), -s1, accumulate=True)

    # # Down-ramp contributions
    # dH.index_put_((batch_idx, k3), -s2, accumulate=True)  # Negative slope
    # dH.index_put_((batch_idx, k4), s2, accumulate=True)  # Positive correction

    # # Integrate to get impulse response
    # H = torch.cumsum(dH, dim=1) * dt

    return H


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


class TorchFieldv2(nn.Module):
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

    def spatial_impulse_response(self, field_points, batch_size=1024, return_all=False):
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

    def compute_pr_from_sir(self, h_sir, x, y, z):
        n_time = h_sir.shape[0]
        grid_shape = (len(y), len(x), len(z))
        h4d = h_sir.T.view(-1, *grid_shape).permute(1, 2, 3, 0)
        fft = torch.fft.fft(h4d, dim=-1)
        freqs = torch.fft.fftfreq(n_time, d=1 / self.fs, device=self.device)
        idx = torch.argmin((freqs - self.fc).abs())
        return fft.abs()[..., idx]

    def forward(self, field_info, normalize=False):
        x, y, z, pts = create_simulation_grid(field_info, self.device)
        t, h = self.spatial_impulse_response(pts)
        pr = self.compute_pr_from_sir(h, x, y, z)
        if normalize:
            pr = pr / pr.max()
        return pr, x, y, z
