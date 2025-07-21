# -*- coding: utf-8 -*-
import math
from time import time as TIME

import numpy as np
import torch
import torch.nn as nn
import torch.profiler
from torch import Tensor
from tqdm import tqdm


# --- JIT-compiled core event computation (unchanged, but output in μs) ---
@torch.jit.script
def compute_patch_events_batch(
    wx: float,  # um (or unit)
    wy: float,  # um (or unit)
    diff: Tensor,  # [B, M, 3] in um (or unit)
    dist: Tensor,  # [B, M] in um (or unit)
    delays: Tensor,  # [B, M] in us (or unit)
    apods: Tensor,  # [B, M] in unitless
    inv_c: float,  # Speed of sound in m/s = um/us
    inv_fs: float,  # Sampling frequency in MHz for computation
) -> Tensor:
    # Compute times in seconds, then convert to microseconds
    xp = diff[..., 0] / dist  # unitless
    yp = diff[..., 1] / dist  # unitless
    xp_abs = torch.abs(xp) * wx * inv_c
    yp_abs = torch.abs(yp) * wy * inv_c
    Dt1 = torch.min(xp_abs, yp_abs)  # us
    Dt2 = torch.max(xp_abs, yp_abs)  # us
    area = (wx * wy) / (2 * math.pi * dist)  # um (or unit)
    # Build event times in μs relative to t0
    t1 = (dist * inv_c) - 0.5 * (Dt1 + Dt2) + delays  # us (or unit)
    t2 = t1 + Dt1  # us (or unit)
    t3 = t1 + Dt2  # us (or unit)
    t4 = t1 + (Dt1 + Dt2)  # us (or unit)
    hmax = area * apods / Dt2.clamp(min=inv_fs)  # space_unit/time_unit (m/s)

    return torch.stack((t1, t2, t3, t4, hmax), dim=-1)


def accumulate_events_derivative(
    events: Tensor,  # [B, M, 5]
    t0_us: float,  # Global start time in μs
    dt_us: float,  # sample interval in μs
    T: int,  # number of time bins
) -> Tensor:
    B, M, _ = events.shape
    device = events.device

    # Flatten tensors
    t1 = events[..., 0].reshape(-1)
    t2 = events[..., 1].reshape(-1)
    t3 = events[..., 2].reshape(-1)
    t4 = events[..., 3].reshape(-1)
    hmax = events[..., 4].reshape(-1)

    inv_dt = 1.0 / dt_us
    s1 = hmax / (t2 - t1).clamp(min=dt_us)  # Use actual time differences

    # Batch indices for flattened events
    batch_idx_full = (
        torch.arange(B, device=device).unsqueeze(1).expand(B, M).reshape(-1)
    )

    # Case masks
    Dt2 = t4 - t2
    case3 = Dt2**2 < (2 * dt_us) ** 2
    case1 = ~case3

    d2H = torch.zeros(B, T, device=device)

    # Helper for linear-interpolated accumulation
    def accumulate_impulse(t_event, value, sign, mask):
        if not mask.any():
            return
        t_event_masked = t_event[mask]
        value_masked = value[mask]
        batch_idx_masked = batch_idx_full[mask]

        f_idx = (t_event_masked - t0_us) * inv_dt
        idx_floor = torch.floor(f_idx).long().clamp(0, T - 1)
        idx_ceil = (idx_floor + 1).clamp(0, T - 1)
        w_floor = 1 - (f_idx - idx_floor)
        w_ceil = f_idx - idx_floor

        d2H.index_put_(
            (batch_idx_masked, idx_floor),
            sign * value_masked * w_floor,
            accumulate=True,
        )
        d2H.index_put_(
            (batch_idx_masked, idx_ceil), sign * value_masked * w_ceil, accumulate=True
        )

    # Case 1: Trapezoid events
    accumulate_impulse(t1, s1, +1, case1)
    accumulate_impulse(t2, s1, -1, case1)
    accumulate_impulse(t3, s1, -1, case1)
    accumulate_impulse(t4, s1, +1, case1)

    # Integrate to get H
    dH = torch.cumsum(d2H, dim=1)
    H = torch.cumsum(dH, dim=1) * dt_us

    # Case 3: Delta events
    def accumulate_delta(t_event, value, mask):
        if not mask.any():
            return
        t_event_masked = t_event[mask]
        value_masked = value[mask]
        batch_idx_masked = batch_idx_full[mask]

        f_idx = (t_event_masked - t0_us) * inv_dt
        idx_floor = torch.floor(f_idx).long().clamp(0, T - 1)
        idx_ceil = (idx_floor + 1).clamp(0, T - 1)
        w_floor = 1 - (f_idx - idx_floor)
        w_ceil = f_idx - idx_floor

        H.index_put_(
            (batch_idx_masked, idx_floor), value_masked * w_floor, accumulate=True
        )
        H.index_put_(
            (batch_idx_masked, idx_ceil), value_masked * w_ceil, accumulate=True
        )

    accumulate_delta(t2, hmax, case3)

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

    # print(
    #     f"Creating grid with {Nx} x {Ny} x {Nz} points in x, y, z directions respectively."
    # )
    # print(f"Grid extents: x: [{x0}, {xf}], y: [{y0}, {yf}], z: [{z0}, {zf}]")
    x = torch.linspace(x0, xf, Nx, device=device)
    y = torch.linspace(y0, yf, Ny, device=device)
    z = torch.linspace(z0, zf, Nz, device=device)
    grid = torch.stack(torch.meshgrid(x, y, z, indexing="ij"), dim=-1)
    return x, y, z, grid.reshape(-1, 3) * 1e-3


class TorchFieldv2(nn.Module):
    def __init__(self, transducer, z_plane_mm=None, device="cpu"):
        super(TorchFieldv2, self).__init__()

        self.device = torch.device(device)
        self.tx = transducer
        self.c = 1540.0  # Speed of sound in m/s
        self.fs = 400e6  # Sampling frequency in Hz
        self.fc = transducer.fc
        self.time_sec_to_unit = 1e6  # Convert seconds to microseconds
        self.space_m_to_unit = 1e6  # Convert meters to micrometers
        self.z_plane_mm = z_plane_mm  # Plane where the field is computed
        self.wx = transducer.el_w / transducer.no_sub_x * self.space_m_to_unit  # um
        self.wy = transducer.el_h / transducer.no_sub_y * self.space_m_to_unit  # um
        self.n_elements = transducer.n_elements
        self.no_sub_x = transducer.no_sub_x
        self.no_sub_y = transducer.no_sub_y
        centers = []
        for elem in range(transducer.n_elements):
            for sub in range(transducer.no_sub_x * transducer.no_sub_y):
                verts = transducer.sub_quad_verts[
                    elem * (transducer.no_sub_x * transducer.no_sub_y) + sub
                ]
                centers.append(verts.mean(axis=0))

        apods = transducer.apodization
        delays = transducer.delays

        self.centers = torch.tensor(
            np.array(centers) * self.space_m_to_unit,
            dtype=torch.float32,
            device=self.device,
        )  # um (or unit)
        self.apods = nn.Parameter(
            torch.tensor(apods, dtype=torch.float32, device=device, requires_grad=True),
        )
        self.delays = nn.Parameter(
            torch.tensor(
                np.array(delays) * self.time_sec_to_unit,
                dtype=torch.float32,
                device=device,
                requires_grad=True,
            ),  # s,
        )

    # --- Full spatial_impulse_response using μs units ---
    def spatial_impulse_response(self, field_points_m, batch_size=1024):
        start_time = TIME()
        if not isinstance(field_points_m, torch.Tensor):
            *_, field_points_m = create_simulation_grid(
                field_points_m, device=self.device
            )
        pts = field_points_m * self.space_m_to_unit  # Convert to um (or unit)
        pts = pts.to(torch.float32).to(self.device)
        P = pts.shape[0]

        max_d = float("-inf")  # Initialize max distance
        min_d = float("inf")  # Initialize min distance

        # Precompute time range
        with torch.no_grad():
            for i in range(0, P, batch_size):
                batch_pts = pts[i : i + batch_size]  # Get a batch of points
                dists_batch = (batch_pts.unsqueeze(1) - self.centers.unsqueeze(0)).norm(
                    dim=-1
                )  # [batch_size, n_elements]
                max_d = max(max_d, dists_batch.max().item())  # Update max distance
                min_d = min(min_d, dists_batch.min().item())  # Update min distance

        focal = torch.tensor(
            [0, 0, self.z_plane_mm * self.space_m_to_unit * 1e-3],
            dtype=torch.float32,
            device=self.device,
        )
        delays_to_focal_plane = torch.norm(self.centers - focal, dim=-1) / self.c
        max_delay = (-delays_to_focal_plane + delays_to_focal_plane.max()).max()

        min_time_us = (
            min_d / self.space_m_to_unit / self.c
            - min(self.wx, self.wy) / self.space_m_to_unit / self.c
        ) * self.time_sec_to_unit  # us (or unit)
        max_time_us = (
            max_d / self.space_m_to_unit / self.c
            + max_delay / self.time_sec_to_unit
            + max(self.wx, self.wy) / self.space_m_to_unit / self.c
        ) * self.time_sec_to_unit  # us (or unit)

        print(f"Time range: {min_time_us:.2f} us to {max_time_us:.2f} us")
        print(f"max delay: {max_delay:.2f} us, vs real {self.delays.max():.2f} us")
        dt_us = (1.0 / self.fs) * self.time_sec_to_unit
        T = int(math.ceil((max_time_us - min_time_us) / dt_us))

        del max_d, min_d  # Free memory

        # Create the apodization and delays tensors of the patches
        # self.apods and self.delays are of shape [n_elements]
        # Then we expand them to [n_elements * no_sub_x * no_sub_y]
        # where each element corresponds to a sub-element has the value of the
        # corresponding element in self.apods and self.delays.
        # Expand apodization and delays tensors to match the number of sub-elements

        expanded_delays = self.delays.repeat_interleave(self.no_sub_x * self.no_sub_y)
        expanded_apods = torch.sigmoid(10 * self.apods).repeat_interleave(
            self.no_sub_x * self.no_sub_y
        )

        H = torch.zeros(P, T, device=self.device)
        for i in tqdm(range(0, P, batch_size), desc="Computing SIR", unit="batch"):
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
                expanded_delays.unsqueeze(0).expand(
                    j - i, -1
                ),  # Delays in us (or unit)
                expanded_apods.unsqueeze(0).expand(j - i, -1),
                inv_c=1 / self.c * (self.time_sec_to_unit / self.space_m_to_unit),
                # Speed of sound in s/m = time unit / space unit
                inv_fs=self.time_sec_to_unit / self.fs,  # 1/fs (time unit)
            )
            # Correct call with t0_us
            H[i:j] = accumulate_events_derivative(events, min_time_us, dt_us, T)
        print(
            f"Spatial impulse response computed in {TIME() - start_time:.2f} seconds with batch size {batch_size}."
        )

        return (
            min_time_us / self.time_sec_to_unit,
            H.T * self.time_sec_to_unit / self.space_m_to_unit,
        )

    def compute_pr_from_sir(self, h_sir, x, y, z):
        start_time = TIME()
        n_time = h_sir.shape[0]
        grid_shape = (len(x), len(y), len(z))
        # print(f"Grid shape: {grid_shape}, n_time: {n_time}")
        # print(f"Shape of h_sir: {h_sir.shape}")
        h4d = h_sir.view(-1, *grid_shape).permute(1, 2, 3, 0)
        # print(f"Shape of h4d: {h4d.shape}")
        fft = torch.fft.fft(h4d, dim=-1)
        freqs = torch.fft.fftfreq(n_time, d=1 / self.fs, device=self.device)
        idx = torch.argmin((freqs - self.fc) ** 2)
        print(
            f"Looking for fc: {self.fc} Hz, found : {freqs[idx]} Hz, in {TIME() - start_time:.2f} seconds."
        )
        return fft[..., idx].abs()

    def examine_bottleneck(
        self, field_info, batch_size=1024, normalize=True, training=False
    ):
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            on_trace_ready=torch.profiler.tensorboard_trace_handler("./log"),
            record_shapes=True,
            with_stack=True,
        ) as prof:
            pr, x, y, z = self.forward(
                field_info,
                batch_size=batch_size,
                normalize=normalize,
                training=training,
            )

        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        return pr, x, y, z

    def forward(self, field_info, batch_size=1024, normalize=False, training=False):
        # Best batch size is ~ 8e6/M, where M = # patches.
        start_time = TIME()
        x, y, z, pts = create_simulation_grid(field_info, self.device)
        if self.z_plane_mm is None:
            self.z_plane_mm = z[int(len(z) / 2)]  # Default to the middle of the z-axis

        if training:
            print(
                f"Computing field for {len(pts)} points and {self.centers.shape[0]} patches, WITH gradients in {self.device}."
            )
            t, h = self.spatial_impulse_response(pts, batch_size=batch_size)
            pr = self.compute_pr_from_sir(h, x, y, z)
        else:
            print(
                f"Computing field for {len(pts)} points and {self.centers.shape[0]} patches, WITHOUT gradients in {self.device}."
            )
            with torch.no_grad():
                t, h = self.spatial_impulse_response(pts, batch_size=batch_size)
                pr = self.compute_pr_from_sir(h, x, y, z)

        print(
            f"Pressure field computed in: {TIME() - start_time:.4f} seconds, using {self.device}."
        )
        if normalize:
            pr = pr / pr.max()
        return pr, x, y, z
