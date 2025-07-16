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


# --- Improved accumulation via index_put to ensure correct spatial placement ---
# @torch.jit.script
def accumulate_events_derivative(
    events: Tensor,  # [B, M, 5]
    t0_us: float,  # Global start time in μs
    dt_us: float,  # sample interval in μs
    T: int,  # number of time bins
    t: Tensor,
) -> Tensor:
    B, M, _ = events.shape
    device = events.device

    # flatten
    t1 = events[..., 0].reshape(-1)  # us (or unit)  [B]
    t2 = events[..., 1].reshape(-1)
    t3 = events[..., 2].reshape(-1)
    t4 = events[..., 3].reshape(-1)
    hmax = events[..., 4].reshape(-1)

    # We have 3 different cases when integrating from the deltas to compute the
    # trapezoidal impulse response.
    # 1. Dt2 > Dt1 >= 1/fs => Dt2 >= 2/fs : This is a trapezoid that could be created
    # with the temporal resolution of the system and the derivative method. Thus, we
    # can create the second derivative setting the impulses at t1, t2, t3, t4.
    # [d2H/dt2 = s1*dirac(t-t1) - s1*dirac(t-t2) - s1*dirac(t-t3) + s1*dirac(t-t4)],
    # thus after integrating we get dH/dt = s1*u(t-t1) -s1*u(t-t2) - s2*u(t-t3)
    # + s2*u(t-t4). Which after integrating gives us the trapezoid.
    # 2. Dt1 < 1/fs and Dt2 >= 2/fs: This means that t1 = t2 and t3 = t4, but t3 and
    # t2 are spaced by at least 1/fs, so the final expected trapezoid is equivalent
    # to the result one would have if t1 = t2 - 1/fs and t3 = t4 + 1/fs.
    # 3. Dt1 < 1/fs and Dt2 < 2/fs: In this case, we have a delta event, which means
    # that td = t1 = t2 = t3 = t4, and we can just set the value at t2
    # (for instance) to hmax.

    # Since we created the events taking this min(Dt1, Dt2) = 1/fs into account,
    # we can counter this artificial artificial increase (to avoid numerical issues)
    # by computing the indexes of t1 and t3 with ceil and t2 and t4 with floor.
    # This ensures that we are accumulating the events in the correct time bins.
    # t = torch.arange(T, device=device) * dt_us + t0_us  # [T] in μs (or unit)
    inv_dt = 1.0 / dt_us  # 1/us (or unit)
    idx_t1 = torch.floor((t1 - t0_us) * inv_dt + 1).long().clamp(0, T - 1)
    idx_t2 = torch.ceil((t2 - t0_us) * inv_dt + 1).long().clamp(0, T - 1)
    idx_t3 = torch.floor((t3 - t0_us) * inv_dt + 1).long().clamp(0, T - 1)
    idx_t4 = torch.ceil((t4 - t0_us) * inv_dt + 1).long().clamp(0, T - 1)

    # Ensure that the rampdown and rampup are the same length
    diff_rampdown_rampup = (idx_t4 - idx_t3) - (idx_t2 - idx_t1)  # [B, M]

    idx_t4 = idx_t4 - (diff_rampdown_rampup)

    # Create boolean masks for the different cases
    # Dt1 = t2 - t1  # us (or unit)
    Dt2 = t4 - t2  # us (or unit)
    case3 = Dt2**2 < (2 * dt_us) ** 2  # Check if Dt2 is too small (close to zero)
    case2 = (idx_t2 == idx_t1) | (
        idx_t4 == idx_t3
    )  # Check if t2-t1 and t4-t3 is too small (close to zero)
    case1 = torch.logical_not(case2) & torch.logical_not(
        case3
    )  # Case 1: trapezoid events

    # We make sure that the up and down ramps are the same length
    # If the samples of the rampdown are shorter than the rampup,
    # the difference is negative, and then will be positive.
    s1 = hmax / (t[idx_t2] - t[idx_t1]).clamp(min=dt_us)  # us (or unit)

    # batch indices
    batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, M).reshape(-1)

    # build derivative accumulator
    d2H = torch.zeros(B, T, device=device)

    # Case 1: trapezoid events
    if case1.any():
        d2H.index_put_((batch_idx[case1], idx_t1[case1]), +s1[case1], accumulate=True)
        d2H.index_put_((batch_idx[case1], idx_t2[case1]), -s1[case1], accumulate=True)
        d2H.index_put_((batch_idx[case1], idx_t3[case1]), -s1[case1], accumulate=True)
        d2H.index_put_((batch_idx[case1], idx_t4[case1]), +s1[case1], accumulate=True)

    if torch.any(d2H.isnan()):
        print("Warning: NaN values found in case1and2.")
    # Case 2: trapezoid events with Dt1 < 1/fs
    if case2.any():
        d2H.index_put_(
            (batch_idx[case2], idx_t2[case2] - 1), +s1[case2], accumulate=True
        )
        d2H.index_put_((batch_idx[case2], idx_t2[case2]), -s1[case2], accumulate=True)
        d2H.index_put_((batch_idx[case2], idx_t3[case2]), -s1[case2], accumulate=True)
        d2H.index_put_(
            (batch_idx[case2], idx_t3[case2] + 1), +s1[case2], accumulate=True
        )

    if torch.any(d2H.isnan()):
        print("Warning: NaN values found in case2.")
    dH = torch.cumsum(d2H, dim=1)  # [B, T]
    # No need to multiply by dt_us here because diracs have no width.

    H = torch.cumsum(dH, dim=1) * dt_us  # [B, T]

    # Case 3: delta events
    if case3.any():
        H.index_put_(
            (batch_idx[case3], idx_t2[case3]), hmax[case3], accumulate=True
        )  # For delta events, we just add the hmax value at t2

    if torch.any(H.isnan()):
        print("Warning: NaN values found in case3.")

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


class TorchField(nn.Module):
    def __init__(self, transducer, device="cpu"):
        super(TorchField, self).__init__()
        self.device = torch.device(device)
        self.tx = transducer
        self.c = 1540.0  # Speed of sound in m/s
        self.fs = 400e6  # Sampling frequency in Hz
        self.fc = transducer.fc
        self.time_sec_to_unit = 1e6  # Convert seconds to microseconds
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
        )  # um (or unit)
        self.apods = nn.Parameter(
            torch.tensor(apods, dtype=torch.float32, device=device, requires_grad=True)
        )
        self.delays = nn.Parameter(
            torch.tensor(
                np.array(delays) * self.time_sec_to_unit,
                dtype=torch.float32,
                device=device,
                requires_grad=True,
            )  # us (or unit)
        )

    # --- Full spatial_impulse_response using μs units ---
    def spatial_impulse_response(
        self, field_points_m, batch_size=1024, return_all=False
    ):
        start_time = TIME()
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

        max_time_us = (
            max_d / self.space_m_to_unit / self.c
            + self.delays.max() / self.time_sec_to_unit
            + max(self.wx, self.wy) / self.space_m_to_unit / self.c
        ) * self.time_sec_to_unit  # us (or unit)
        min_time_us = (
            min_d / self.space_m_to_unit / self.c
            + self.delays.min() / self.time_sec_to_unit
            - min(self.wx, self.wy) / self.space_m_to_unit / self.c
        ) * self.time_sec_to_unit  # us (or unit)
        # print(f"Time range: {min_time_us:.2f} us to {max_time_us:.2f} us")
        dt_us = (1.0 / self.fs) * self.time_sec_to_unit
        T = int(math.ceil((max_time_us - min_time_us) / dt_us))
        t_global = min_time_us + torch.arange(T, device=self.device) * dt_us

        del max_d, min_d  # Free memory

        # Ensure that the centers are in the correct unit (um or unit)
        self.centers = self.centers.to(self.device)
        # For debugging purposes, print the centers and distances
        # print("centers:", self.centers)
        # print(f"Max distance (um): {max_d}, Min distance (um): {min_d}")
        # print(f"wx: {self.wx}, wy: {self.wy}, c: {self.c}, fs: {self.fs}")

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
                self.delays.unsqueeze(0).expand(j - i, -1),  # Delays in us (or unit)
                self.apods.unsqueeze(0).expand(j - i, -1),
                inv_c=1 / self.c * (self.time_sec_to_unit / self.space_m_to_unit),
                # Speed of sound in s/m = time unit / space unit
                inv_fs=self.time_sec_to_unit / self.fs,  # 1/fs (time unit)
            )
            # Correct call with t0_us
            H[i:j] = accumulate_events_derivative(
                events, min_time_us, dt_us, T, t_global
            )
        print(
            f"Spatial impulse response computed in {TIME() - start_time:.2f} seconds with batch size {batch_size}."
        )
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

    def forward(self, field_info, batch_size=1024, normalize=True, training=False):
        start_time = TIME()
        x, y, z, pts = create_simulation_grid(field_info, self.device)
        if training:
            t, h = self.spatial_impulse_response(pts, batch_size=batch_size)
            pr = self.compute_pr_from_sir(h, x, y, z)
        else:
            with torch.no_grad():
                print(
                    f"Computing field for {len(pts)} points with no gradients (No Training)."
                )
                t, h = self.spatial_impulse_response(pts, batch_size=batch_size)
                pr = self.compute_pr_from_sir(h, x, y, z)

        print(f"Pressure field computed in: {TIME() - start_time:.4f} seconds.")
        if normalize:
            pr = pr / pr.max()
        return pr, x, y, z
