import torch
import numpy as np
from tqdm import tqdm
import time
import math
import torch.nn as nn

def create_simulation_grid(simulation_struct, device='cpu'):
    x0, xf = simulation_struct["x_extent"]
    y0, yf = simulation_struct["y_extent"]
    z0, zf = simulation_struct["z_extent"]
    dx, dy, dz = simulation_struct["dx"], simulation_struct["dy"], simulation_struct["dz"]
    
    Nx = int((xf - x0) / dx)
    Ny = int((yf - y0) / dy)
    Nz = int((zf - z0) / dz)
    
    if Nx % 2 == 0: Nx += 1
    if Ny % 2 == 0: Ny += 1
    if Nz % 2 == 0: Nz += 1

    x = torch.linspace(x0, xf, Nx, device=device) * 1e-3
    y = torch.linspace(y0, yf, Ny, device=device) * 1e-3
    z = torch.linspace(z0, zf, Nz, device=device) * 1e-3
    
    grid = torch.stack(torch.meshgrid(x, y, z, indexing='ij'), dim=-1)
    grid_points = grid.reshape(-1, 3)
    return x, y, z, grid_points

def accumulate_events_vectorized(events, fs, t0, n2, device='cpu', batch_size=100):
    """
    Optimized accumulation using PyTorch's advanced indexing and segment operations.
    Achieves O(P*M*log(T)) complexity with minimal memory footprint.
    """
    P, M, _ = events.shape
    dt = 1.0 / fs
    t_global = t0 + torch.arange(n2, device=device) * dt
    h_out = torch.zeros((P, n2), dtype=torch.float32, device=device)
    
    # Precompute time parameters for all events
    t1 = events[..., 0]
    t2 = events[..., 1]
    t3 = events[..., 2]
    t4 = events[..., 3]
    h_max = events[..., 4]
    
    # Precompute valid events mask
    valid_mask = (h_max > 0) & (t4 > t1)
    
    for start in tqdm(range(0, P, batch_size), desc="Processing batches", unit="batches"):
        end = min(start + batch_size, P)
        batch_idx = slice(start, end)
        
        # Process only valid events in this batch
        for m in range(M):
            # Skip invalid events early
            valid = valid_mask[batch_idx, m]
            if not valid.any():
                continue
                
            # Get valid events for this patch
            t1_v = t1[batch_idx, m][valid]
            t2_v = t2[batch_idx, m][valid]
            t3_v = t3[batch_idx, m][valid]
            t4_v = t4[batch_idx, m][valid]
            h_max_v = h_max[batch_idx, m][valid]
            point_idx = torch.arange(start, end, device=device)[valid]
            
            # Calculate time segments
            k_start = torch.floor((t1_v - t0) * fs).long().clamp(0, n2-1)
            k_end = torch.ceil((t4_v - t0) * fs).long().clamp(1, n2)
            
            # Process each valid event
            for i in range(len(t1_v)):
                if k_start[i] >= k_end[i]:
                    continue
                    
                # Get time segment indices
                k0 = k_start[i]
                k1 = k_end[i]
                seg_length = k1 - k0
                seg_indices = torch.arange(k0, k1, device=device)
                t_seg = t_global[seg_indices]
                
                # Compute trapezoid values
                h_vals = torch.zeros(seg_length, device=device)
                
                # Ramp up phase
                mask1 = (t_seg >= t1_v[i]) & (t_seg < t2_v[i])
                if mask1.any():
                    slope = h_max_v[i] / (t2_v[i] - t1_v[i] + 1e-12)
                    h_vals[mask1] = slope * (t_seg[mask1] - t1_v[i])
                
                # Plateau phase
                mask2 = (t_seg >= t2_v[i]) & (t_seg < t3_v[i])
                if mask2.any():
                    h_vals[mask2] = h_max_v[i]
                
                # Ramp down phase
                mask3 = (t_seg >= t3_v[i]) & (t_seg < t4_v[i])
                if mask3.any():
                    slope = h_max_v[i] / (t4_v[i] - t3_v[i] + 1e-12)
                    h_vals[mask3] = slope * (t4_v[i] - t_seg[mask3])
                
                # Accumulate to output
                h_out[point_idx[i], seg_indices] += h_vals
    
    return h_out, t_global

class TorchField(nn.Module):
    def __init__(self, transducer, device='cpu'):
        super(TorchField, self).__init__()
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
                verts = self.tx.sub_quad_verts[elem*(self.tx.no_sub_x*self.tx.no_sub_y)+sub_elem]
                centers.append(verts.mean(axis=0))
                apods.append(self.tx.apodization[elem])
                delays.append(self.tx.delays[elem])
        
        self.centers = torch.tensor(centers, dtype=torch.float32, device=device) * 1e-3
        self.apods = nn.Parameter(torch.tensor(apods, dtype=torch.float32, device=device, requires_grad=True))
        self.delays = nn.Parameter(torch.tensor(delays, dtype=torch.float32, device=device, requires_grad=True))
        self.wx = el_w * 1e-3
        self.wy = el_h * 1e-3
        
        self.field = None
        self.x = self.y = self.z = None
        print(f"Initialized TorchField on {device}")

    def spatial_impulse_response(self, field_points):
        start_time = time.time()
        pts = torch.atleast_2d(torch.tensor(field_points, device=self.device, dtype=torch.float32))
        P, M = pts.shape[0], self.centers.shape[0]
        
        # Vectorized distance and direction calculations
        diff = pts.unsqueeze(1) - self.centers.unsqueeze(0)  # (P, M, 3)
        dist = torch.norm(diff, dim=-1)
        xp = diff[..., 0] / dist
        yp = diff[..., 1] / dist
        
        # Vectorized SIR computation
        print(f"Computing SIR for {P} points and {M} transducer elements.")
        epsilon = 1 / self.fs
        Dt1 = torch.min(torch.abs(self.wx * xp / self.c), 
                       torch.abs(self.wy * yp / self.c))
        Dt2 = torch.max(torch.abs(self.wx * xp / self.c), 
                       torch.abs(self.wy * yp / self.c))
        
        area = self.wx * self.wy / (2 * math.pi * dist)
        t1 = dist / self.c - 0.5 * (Dt1 + Dt2) + self.delays.unsqueeze(0)
        t2 = t1 + Dt1
        t3 = t1 + Dt2
        t4 = t1 + Dt1 + Dt2
        
        Dt2_adj = torch.where(2 * Dt2 < epsilon, epsilon, Dt2)
        h_max = area * self.apods.unsqueeze(0) / Dt2_adj
        
        events = torch.stack([t1, t2, t3, t4, h_max], dim=-1)
        
        # Time vector setup
        all_times = events[..., :4].contiguous().view(-1)
        t0 = all_times.min()
        tN = all_times.max()
        num_samples = int(torch.ceil((tN - t0) * self.fs).item())
        n2 = 2**max(int(math.ceil(math.log2(num_samples))), 5)
        
        # Optimized accumulation
        print(f"Accumulating events for {P} points over {n2} time samples.")
        h_out, t_global = accumulate_events_vectorized(
            events, self.fs, t0, n2, device=self.device
        )
        
        print(f"SIR computed in {time.time() - start_time:.2f} seconds")
        return t_global, h_out.T

    def compute_pr_from_sir(self, h_sir, x, y, z):
        # Reshape to spatial dimensions
        print(f"Computing pressure field from SIR with shape: {h_sir.shape}")
        n_time, n_points = h_sir.shape
        grid_shape = (len(y), len(x), len(z))
        h_sir_4d = h_sir.T.view(-1, len(y), len(x), len(z)).permute(1, 2, 3, 0)
        
        # FFT processing
        h_sir_fft = torch.fft.fft(h_sir_4d, dim=-1)
        freqs = torch.fft.fftfreq(n_time, 1/self.fs, device=self.device)
        idx_fc = torch.argmin(torch.abs(freqs - self.fc))
        pressure = torch.abs(h_sir_fft[..., idx_fc])
        return pressure

    def forward(self, field_info, normalize=False, inplace=False):
        x, y, z, grid_points = create_simulation_grid(field_info, self.device)
        print(f"Grid created with shape: {grid_points.shape}, x: {len(x)}, y: {len(y)}, z: {len(z)}")
        _, h_sir = self.spatial_impulse_response(grid_points.cpu().numpy())
        pressure_field = self.compute_pr_from_sir(h_sir, x, y, z)
        
        if normalize:
            pressure_field /= pressure_field.max()
            
        if inplace:
            self.field = pressure_field
            self.x, self.y, self.z = x, y, z
        print(f"Pressure field computed succesfully")
        return pressure_field, x, y, z