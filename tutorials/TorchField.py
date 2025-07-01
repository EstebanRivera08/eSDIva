# -*- coding: utf-8 -*-
import math
from time import time as TIME

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


# --- JIT-compiled core event computation (unchanged, but output in μs) ---
# @torch.jit.script
def compute_patch_events_batch(
    wx: float,  # um (or unit)
    wy: float,  # um (or unit)
    diff: Tensor,  # [B, M, 3] in um (or unit)
    dist: Tensor,  # [B, M] in um (or unit)
    delays: Tensor,  # [B, M] in us (or unit)
    apods: Tensor,  # [B, M] in unitless
    c: float,  # Speed of sound in m/s = um/us
    fs: float,  # Sampling frequency in MHz for computation
) -> Tensor:
    # Compute times in seconds, then convert to microseconds
    xp = diff[..., 0] / dist  # unitless
    yp = diff[..., 1] / dist  # unitless
    print(f"distances: {dist}, diff: {diff}, delays: {delays}, apods: {apods}")
    Dt1 = torch.min((wx * xp / c).abs(), (wy * yp / c).abs()).clamp(min=1 / fs)  # us
    Dt2 = torch.max((wx * xp / c).abs(), (wy * yp / c).abs()).clamp(min=1 / fs)  # us
    print(f"Dt1: {Dt1}, Dt2: {Dt2}, wx: {wx}, wy: {wy}, c: {c}, fs: {fs}")
    area = (wx * wy) / (2 * math.pi * dist)  # um (or unit)
    # Build event times in μs relative to t0
    t1 = (dist / c) - 0.5 * (Dt1 + Dt2) + delays  # us (or unit)
    t2 = t1 + Dt1  # us (or unit)
    t3 = t1 + Dt2  # us (or unit)
    t4 = t1 + (Dt1 + Dt2)  # us (or unit)
    hmax = area * apods / Dt2  # space_unit/time_unit (m/s)
    print(f"dt = {1 / fs}, t1: {t1}, t2: {t2}, t3: {t3}, t4: {t4}, hmax: {hmax}")
    return torch.stack((t1, t2, t3, t4, hmax), dim=-1)


# --- Improved accumulation via index_put to ensure correct spatial placement ---
# @torch.jit.script
# @torch.jit.script
def accumulate_events_derivative(
    events: Tensor,  # [B, M, 5]
    t0_us: float,  # Global start time in μs
    dt_us: float,  # sample interval in μs
    T: int,  # number of time bins
) -> Tensor:
    B, M, _ = events.shape
    device = events.device

    # flatten
    t1 = events[..., 0].reshape(-1)
    t2 = events[..., 1].reshape(-1)
    t3 = events[..., 2].reshape(-1)
    t4 = events[..., 3].reshape(-1)
    hmax = events[..., 4].reshape(-1)

    # compute slopes
    s1 = hmax / (t2 - t1)
    s2 = hmax / (t4 - t3)

    # bin indices WITHOUT the +1 shift
    idx_t1 = torch.ceil((t1 - t0_us) / dt_us).long().clamp(0, T - 1)
    idx_t2 = torch.ceil((t2 - t0_us) / dt_us).long().clamp(0, T - 1)
    idx_t3 = torch.floor((t3 - t0_us) / dt_us).long().clamp(0, T - 1)
    idx_t4 = torch.floor((t4 - t0_us) / dt_us).long().clamp(0, T - 1)
    # Debugging output printing to check indices
    print("idx_t1:", idx_t1)
    print("idx_t2:", idx_t2)
    print("idx_t3:", idx_t3)
    print("idx_t4:", idx_t4)
    critere1 = idx_t2 >= idx_t1
    critere2 = idx_t3 >= idx_t2
    critere3 = idx_t4 >= idx_t3
    if not critere1.any():
        print("critere1:", critere1)
    elif not critere2.any():
        print("critere2:", critere2)
    elif not critere3.any():
        print("critere3:", critere3)

    # batch indices
    batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, M).reshape(-1)

    # build derivative accumulator
    dH = torch.zeros(B, T, device=device)

    # CORRECTED signs for the four jumps:
    dH.index_put_((batch_idx, idx_t1), +s1, accumulate=True)
    dH.index_put_((batch_idx, idx_t2), -s1, accumulate=True)
    dH.index_put_((batch_idx, idx_t3), -s2, accumulate=True)
    dH.index_put_((batch_idx, idx_t4), +s2, accumulate=True)

    # integrate by cumsum
    H = torch.cumsum(dH, dim=1) * dt_us
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


class TorchField(nn.Module):
    def __init__(self, transducer, device="cpu"):
        super().__init__()
        self.device = torch.device(device)
        self.tx = transducer
        self.c = 1540.0  # Speed of sound in m/s
        self.fs = 300e6  # Sampling frequency in Hz
        self.fc = transducer.fc
        self.time_sec_to_unit = 1e9  # Convert seconds to microseconds
        self.space_m_to_unit = 1e6  # Convert meters to micrometers

        self.wx = transducer.el_w / transducer.no_sub_x * self.space_m_to_unit  # um
        self.wy = transducer.el_h / transducer.no_sub_y * self.space_m_to_unit  # um
        centers, apods, delays = [], [], []
        for elem in range(transducer.n_elements):
            for sub in range(transducer.no_sub_x * transducer.no_sub_y):
                verts = transducer.sub_quad_verts[
                    elem * (transducer.no_sub_x * transducer.no_sub_y) + sub
                ]
                centers.append(verts.mean(axis=0))
                apods.append(transducer.apodization[elem])
                delays.append(transducer.delays[elem])

        self.centers = torch.tensor(
            np.array(centers) * self.space_m_to_unit,
            dtype=torch.float32,
            device=self.device,
        )  # um
        self.apods = torch.tensor(apods, dtype=torch.float32, device=self.device)
        self.delays = torch.tensor(
            np.array(delays) * self.time_sec_to_unit,
            dtype=torch.float32,
            device=self.device,
        )  # us

    # --- Full spatial_impulse_response using μs units ---
    def spatial_impulse_response(
        self, field_points_m, batch_size=1024, return_all=False
    ):
        if not isinstance(field_points_m, torch.Tensor):
            *_, field_points_m = create_simulation_grid(
                field_points_m, device=self.device
            )
        pts = field_points_m * self.space_m_to_unit  # Convert to um (or unit)
        pts = pts.to(torch.float32).to(self.device)
        P = pts.shape[0]

        # Precompute time range
        with torch.no_grad():
            dists = (pts.unsqueeze(1) - self.centers.unsqueeze(0)).norm(
                dim=-1
            )  # um - um (or unit)
            max_d = dists.max().item()  # um (or unit)
            min_d = dists.min().item()  # um (or unit)
        print("centers:", self.centers)
        print(f"Max distance (um): {max_d}, Min distance (um): {min_d}")
        max_time_us = (
            max_d / self.space_m_to_unit / self.c
            + self.delays.max() / self.time_sec_to_unit
            + 0.5 * (self.wx + self.wy) / self.space_m_to_unit / self.c
        ) * self.time_sec_to_unit  # us (or unit)
        min_time_us = (
            min_d / self.space_m_to_unit / self.c
            + self.delays.min() / self.time_sec_to_unit
            - 0.5 * (self.wx + self.wy) / self.space_m_to_unit / self.c
        ) * self.time_sec_to_unit  # us (or unit)
        print(f"Max time (us): {max_time_us}, Min time (us): {min_time_us}")
        dt_us = (1.0 / self.fs) * self.time_sec_to_unit
        print(f"dt (us): {dt_us}")
        T = int(math.ceil((max_time_us - min_time_us) / dt_us))
        t_global = min_time_us + torch.arange(T, device=self.device) * dt_us
        print(f"wx: {self.wx}, wy: {self.wy}, c: {self.c}, fs: {self.fs}")
        H = torch.zeros(P, T, device=self.device)
        with torch.no_grad():
            for i in range(0, P, batch_size):
                j = min(i + batch_size, P)
                batch = pts[i:j]  # [j-i, 3] in um (or unit)
                diff = batch.unsqueeze(1) - self.centers.unsqueeze(
                    0
                )  # [j-i, n_elements, 3] in um (or unit)
                # Compute distances and events
                dist = diff.norm(dim=-1)  # [j-i, n_elements] in um (or unit)
                events = compute_patch_events_batch(
                    self.wx,  # um (or unit)
                    self.wy,  # um (or unit)
                    diff,  # um (or unit)
                    dist,  # um (or unit)
                    self.delays.unsqueeze(0).expand(
                        j - i, -1
                    ),  # Delays in us (or unit)
                    self.apods.unsqueeze(0).expand(j - i, -1),
                    self.c
                    * (
                        self.space_m_to_unit / self.time_sec_to_unit
                    ),  # Speed of sound in m/s = um/us
                    self.fs  # Sampling frequency in Hz
                    / self.time_sec_to_unit,  # Convert fs to MHz for computation
                )
                # Correct call with t0_us
                H[i:j] = accumulate_events_derivative(events, min_time_us, dt_us, T)

        if return_all:
            return (
                t_global / self.time_sec_to_unit,
                H.T * self.time_sec_to_unit / self.space_m_to_unit,
                events,
            )
        return (
            t_global / self.time_sec_to_unit,
            H.T * self.time_sec_to_unit / self.space_m_to_unit,
        )

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
