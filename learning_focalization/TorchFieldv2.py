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
    apodization: Tensor,  # [B, M] in unitless
    inv_c: float,  # Speed of sound in m/s = um/us
    inv_fs: float,  # Sampling frequency in MHz for computation
) -> Tensor:
    # Compute times in seconds, then convert to microseconds
    xp = diff[..., 0] / dist  # unitless
    yp = diff[..., 1] / dist  # unitless
    xp_abs = torch.abs(xp) * wx * inv_c
    yp_abs = torch.abs(yp) * wy * inv_c
    Dt1 = torch.min(xp_abs, yp_abs)  # .clamp(min=0.1 * inv_fs)  # us
    Dt2 = torch.max(xp_abs, yp_abs).clamp(min=0.15 * inv_fs)  # us
    area = (wx * wy) / (2 * math.pi * dist)  # um (or unit)
    # Build event times in μs relative to t0
    t1 = (dist * inv_c) - 0.5 * (Dt1 + Dt2) + delays  # us (or unit)
    t2 = t1 + Dt1  # us (or unit)
    t3 = t1 + Dt2  # us (or unit)
    t4 = t1 + (Dt1 + Dt2)  # us (or unit)
    hmax = area * apodization / Dt2  # space_unit/time_unit (m/s)

    return torch.stack((t1, t2, t3, t4, hmax), dim=-1)  # [B, M, 5]


# Helper for linear-interpolated accumulation
def accumulate_d2H_interpolation(
    d2H, batch_idx_full, t_event, value, sign, t0_us, inv_dt, T, shift=0
):
    f_idx = (t_event - t0_us) * inv_dt + 1 + shift
    idx_floor = torch.floor(f_idx).long().clamp(0, T - 1)
    w_floor = 1 - (f_idx - idx_floor)

    idx_ceil = (idx_floor + 1).clamp(0, T - 1)
    w_ceil = f_idx - idx_floor

    d2H.index_put_(
        (batch_idx_full, idx_ceil),
        sign * value * w_ceil,
        accumulate=True,
    )

    d2H.index_put_(
        (batch_idx_full, idx_floor),
        sign * value * w_floor,
        accumulate=True,
    )


def accumulate_events_derivative(
    events: Tensor,  # [B, M, 5]
    t0_us: float,  # Global start time in μs
    dt_us: float,  # sample interval in μs
    T: int,  # number of time bins
    tolerance: float = 0.5,  # tolerance for numerical
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
    s1 = hmax / (t2 - t1)  # .clamp(min=dt_us)  # Use actual time differences

    # Batch indices for flattened events
    batch_idx_full = (
        torch.arange(B, device=device).unsqueeze(1).expand(B, M).reshape(-1)
    )

    d2H = torch.zeros(B, T, device=device)

    accumulate_d2H_interpolation(d2H, batch_idx_full, t1, s1, +1, t0_us, inv_dt, T)
    accumulate_d2H_interpolation(d2H, batch_idx_full, t2, s1, -1, t0_us, inv_dt, T)
    accumulate_d2H_interpolation(d2H, batch_idx_full, t3, s1, -1, t0_us, inv_dt, T)
    accumulate_d2H_interpolation(d2H, batch_idx_full, t4, s1, +1, t0_us, inv_dt, T)

    # Integrate to get H
    dH = torch.cumsum(d2H, dim=1)
    H = torch.cumsum(dH, dim=1) * dt_us

    return H


def create_simulation_grid_from_dict(simulation_struct, device="cpu"):
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
    return x, y, z, grid.reshape(-1, 3)  # [P, 3]


class TorchFieldv2(nn.Module):
    def __init__(self, transducer, z_plane_mm=None, device="cpu"):
        super(TorchFieldv2, self).__init__()

        # -------------- Medium and TX characteristics ----------------
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

        apodization = transducer.apodization
        delays = transducer.delays

        self.centers = torch.tensor(
            np.array(centers) * self.space_m_to_unit,
            dtype=torch.float32,
            device=self.device,
        )  # um (or unit)

        # self.softplus = nn.Softplus(beta=20, threshold=0.5)

        # ------------------ Define parameters -----------------------
        self.apodization = nn.Parameter(
            torch.tensor(
                apodization, dtype=torch.float32, device=device, requires_grad=True
            ),
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
    def spatial_impulse_response(
        self,
        pts,
        batch_size=1024,
        *,
        P=None,
        delays=None,
        apodization=None,
    ):
        """
        Compute the spatial impulse response (SIR) for the given field points.

        Parameters :
        -----------
        field_points_m : Tensor
            The field points in meters.
        batch_size : int
            The batch size for processing.
        delays : Tensor, optional
            The delays for the field points.
        apodization : Tensor, optional
            The apodization for the field points.
        """
        if delays is None:
            delays = self.delays
        if apodization is None:
            apodization = self.apodization

        start_time = TIME()
        if P is None:
            P = pts.shape[0]

        min_time_us, dt_us, T = self._compute_temporal_grid(pts, P)

        # Create the apodization and delays tensors of the patches
        # self.apodization and self.delays are of shape [n_elements]
        # Then we expand them to [n_elements * no_sub_x * no_sub_y]
        # where each element corresponds to a sub-element has the value of the
        # corresponding element in self.apodization and self.delays.
        # Expand apodization and delays tensors to match the number of sub-elements

        expanded_delays = delays.repeat_interleave(self.no_sub_x * self.no_sub_y)
        expanded_apodization = apodization.repeat_interleave(
            self.no_sub_x * self.no_sub_y
        )

        # Initialize the SIR tensor
        H = torch.zeros(P, T, device=self.device)

        # Compute and accumulate the events
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
                expanded_apodization.unsqueeze(0).expand(j - i, -1),
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

    def compute_pr_from_sir(self, h_sir, x, y, z, batch_size=1024):
        """
        Compute the pressure field from the spatial impulse response (SIR) in batches.

        Parameters
        ----------
        h_sir : Tensor
            The spatial impulse response tensor of shape [n_time, n_points].
        x : Tensor
            The x-coordinates of the grid.
        y : Tensor
            The y-coordinates of the grid.
        z : Tensor
            The z-coordinates of the grid.
        batch_size : int, optional
            The number of points to process in each batch. Default is 1024.

        Returns
        -------
        Tensor
            The computed pressure field of shape [len(x), len(y), len(z)].
        """
        start_time = TIME()
        n_time = h_sir.shape[0]
        n_points = h_sir.shape[1]
        grid_shape = (len(x), len(y), len(z))

        # Initialize the output tensor
        pr_field = torch.zeros(grid_shape, device=self.device)

        # Compute frequencies
        freqs = torch.fft.fftfreq(n_time, d=1 / self.fs, device=self.device)
        idx = torch.argmin((freqs - self.fc) ** 2)
        print(
            f"Looking for fc: {self.fc} Hz, found : {freqs[idx]} Hz, in {TIME() - start_time:.2f} seconds."
        )

        # Process the FFT in batches
        fft_results = []
        for i in range(0, n_points, batch_size):
            batch_start = i
            batch_end = min(i + batch_size, n_points)

            # Compute FFT for the batch
            fft_batch = torch.fft.fft(h_sir[:, batch_start:batch_end], dim=0)

            # Extract the frequency component corresponding to fc
            fft_results.append(fft_batch[idx].abs())

        # Concatenate the FFT results for all batches
        fft_results = torch.cat(fft_results, dim=0)

        # Reshape and permute the data after FFT
        h4d = fft_results.view(len(x), len(y), len(z)).permute(0, 1, 2)

        # Add the reshaped data to the output tensor
        pr_field += h4d

        print(f"Transformed from SIR to Pressure in {TIME() - start_time:.2f} seconds.")
        return pr_field

    def forward(self, field_info_mm, batch_size=1024, normalize=False, training=False):
        """
        Forward pass for the model: 1) Compute of the SIR. 2) Compute Pr from SIR.

        Parameters :

        """
        # Best batch size is ~ 8e6/M, where M = # patches.
        start_time = TIME()
        x, y, z, pts, P = self._compute_spatial_grid(field_info_mm)

        if training:
            print(
                f"Computing field for {len(pts)} points and {self.centers.shape[0]} patches, WITH gradients in {self.device}."
            )
            t, h = self.spatial_impulse_response(
                pts,
                batch_size=batch_size,
                delays=self._process_delays(self.delays),
                apodization=self._process_apodization(self.apodization),
                P=P,
            )
            pr = self.compute_pr_from_sir(h, x, y, z)
        else:
            print(
                f"Computing field for {len(pts)} points and {self.centers.shape[0]} patches, WITHOUT gradients in {self.device}."
            )
            with torch.no_grad():
                t, h = self.spatial_impulse_response(
                    pts,
                    batch_size=batch_size,
                    P=P,
                )
                pr = self.compute_pr_from_sir(h, x, y, z)

        print(
            f"Pressure field computed in: {TIME() - start_time:.4f} seconds, using {self.device}."
        )
        if normalize:
            pr = pr / pr.max()
        return pr, x, y, z

    def examine_bottleneck(
        self, field_info, batch_size=1024, normalize=False, training=False
    ):
        """
        Examine the bottleneck of the model by profiling the forward pass.
        It will print a table showing the operations that consume the most time and ressources.

        Note: This requires the PyTorch profiler to be enabled.

        Parameters:
        The same as the forward pass:
        - field_info : Tensor
            The input field information tensor.
        - batch_size : int
            The batch size for processing.
        - normalize : bool
            Whether to normalize the output.
        - training : bool
            Whether the model is in training mode.

        Output :
        The same as the forward pass + profiling information.
        - pr : Tensor
            The pressure field tensor.
        - x : Tensor
            The x coordinates of the field points.
        - y : Tensor
            The y coordinates of the field points.
        - z : Tensor
            The z coordinates of the field points.
        """

        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            on_trace_ready=None,  # Disable tensorboard trace handler for now
            record_shapes=True,
            with_stack=False,  # Disable stack tracing to simplify
        ) as prof:
            pr, x, y, z = self.forward(
                field_info,
                batch_size=batch_size,
                normalize=normalize,
                training=training,
            )

        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        return pr, x, y, z

    # ----------------------------- helper functions --------------------------------------
    def _compute_spatial_grid(self, field_points_mm):
        # Check field inputs
        if isinstance(field_points_mm, torch.Tensor):
            # Check dimentsions [P, 3]
            if field_points_mm.ndim > 2 or field_points_mm.shape[1] != 3:
                raise ValueError("field_points_mm must be a 2D tensor of shape [P, 3]")
            else:
                field_points_mm.to(self.device)
                x, y, z = (
                    field_points_mm[:, 0],
                    field_points_mm[:, 1],
                    field_points_mm[:, 2],
                )
        elif isinstance(field_points_mm, dict):
            x, y, z, field_points_mm = create_simulation_grid_from_dict(
                field_points_mm, device=self.device
            )
        elif isinstance(field_points_mm, np.ndarray):
            field_points_mm = torch.tensor(
                field_points_mm, dtype=torch.float32, device=self.device
            )
            x, y, z = (
                field_points_mm[:, 0],
                field_points_mm[:, 1],
                field_points_mm[:, 2],
            )
        else:
            raise ValueError(
                "field_points_mm must be a 2D tensor/np.array of shape [P, 3], or \n"
                " a dict with keys 'x_entent', 'y_entent', 'z_entent', 'dx', 'dy', 'dz'."
            )

        # Convert to unit
        pts = field_points_mm * 1e-3 * self.space_m_to_unit  # Convert to um (or unit)
        pts = pts.to(torch.float32).to(self.device)
        return x, y, z, pts, pts.shape[0]

    def _compute_temporal_grid(self, pts, P, batch_size=1024):
        """
        Compute the spatial and temporal grid for the given field points.

        Parameters
        ----------
        pts : Tensor
            The Tensor [P,3] of field points in meters.
        batch_size : int
            The batch size for processing.
        """
        max_d = float("-inf")  # Initialize max distance
        min_d = float("inf")  # Initialize min distance

        # Precompute time range
        # - We look for the nearest and farthest points
        with torch.no_grad():
            for i in range(0, P, batch_size):
                batch_pts = pts[i : i + batch_size]  # Get a batch of points
                dists_batch = (batch_pts.unsqueeze(1) - self.centers.unsqueeze(0)).norm(
                    dim=-1
                )  # [batch_size, n_elements]
                max_d = max(max_d, dists_batch.max().item())  # Update max distance
                min_d = min(min_d, dists_batch.min().item())  # Update min distance

        # Assuming focalization, we set a virtual focal point to compute a max delay
        max_z = pts[:, 2].max().item()

        focal = torch.tensor(
            [0, 0, max_z * self.space_m_to_unit * 1e-3],
            dtype=torch.float32,
            device=self.device,
        )  # Focal point in um (or unit)
        delays_to_focal_plane = torch.norm(self.centers - focal, dim=-1) / self.c
        max_delay = (-delays_to_focal_plane + delays_to_focal_plane.max()).max()

        # Compute min and max time
        min_time_us = (
            min_d / self.space_m_to_unit / self.c
            - min(self.wx, self.wy) / self.space_m_to_unit / self.c
        ) * self.time_sec_to_unit  # us (or unit)
        max_time_us = (
            max_d / self.space_m_to_unit / self.c
            + max_delay / self.time_sec_to_unit
            + max(self.wx, self.wy) / self.space_m_to_unit / self.c
        ) * self.time_sec_to_unit  # us (or unit)

        del max_d, min_d, dists_batch, batch_pts, delays_to_focal_plane  # Free memory

        dt_us = (1.0 / self.fs) * self.time_sec_to_unit
        T = int(math.ceil((max_time_us - min_time_us) / dt_us))
        print(
            f"Time range: {min_time_us:.2f} us to {max_time_us:.2f} us, with {T} samples."
        )

        return min_time_us, dt_us, T

    def _process_apodization(
        self,
        apodization=None,
        *,
        sigmoid_width=1,
        sigmoid_center=0.5,
    ):
        """
        Functions to process apodization during training and inference.
        For example, applying a sigmoid function to the apodization tensor to ensure it is [0, 1].

        Parameters
        ----------
        apodization : Tensor
            The input apodization tensor.
        sigmoid_width : float
            The width of the sigmoid function.
        sigmoid_center : float
            The center of the sigmoid function.

        Returns
        -------
        Tensor
            The processed apodization tensor.
        """

        apodization = torch.sigmoid(10 / sigmoid_width * (apodization - sigmoid_center))

        return apodization

    def _process_delays(
        self,
        delays=None,
    ):
        """
        Functions to process delays during training and inference.
        Note: After some tries it seems that no processing is needed

        Parameters
        ----------

        Returns
        -------
        """

        return delays

    def __repr__(self):
        """
        String representation of the PyField object.

        Returns
        -------
        str
            A string representation of the PyField object.
        """
        return (
            f"TorchField(transducer={self.tx}, c={self.c}, fs={self.fs}, fc={self.fc})"
        )
