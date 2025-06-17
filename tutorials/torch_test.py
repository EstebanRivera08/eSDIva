import torch
import torch.nn as nn
import numpy as np

def create_simulation_grid(simulation_struct):
    [x0, xf], [y0, yf], [z0, zf] = (
        simulation_struct["x_extent"], simulation_struct["y_extent"], simulation_struct["z_extent"])
    dx, dy, dz = simulation_struct["dx"], simulation_struct["dy"], simulation_struct["dz"]
    # Ensure odd counts via bitwise OR with 1
    Nx = int(np.ceil((xf-x0)/dx)) | 1
    Ny = int(np.ceil((yf-y0)/dy)) | 1
    Nz = int(np.ceil((zf-z0)/dz)) | 1

    x = np.linspace(x0, xf, Nx)
    y = np.linspace(y0, yf, Ny)
    z = np.linspace(z0, zf, Nz)
    pts = np.stack(np.meshgrid(x, y, z), axis=-1).reshape(-1, 3) * 1e-3
    return x, y, z, pts

class TorchField(nn.Module):
    def __init__(self, transducer, device='cuda'):
        super().__init__()
        self.device = device
        self.tx = transducer
        self.c = 1540.0
        self.fs = 300e6
        self.fc = transducer.fc
        self.lambda_mm = self.c / self.fc

        # Patch dimensions (m)
        self.wx = transducer.el_w / transducer.no_sub_x
        self.wy = transducer.el_h / transducer.no_sub_y

        # Build patch centers once
        centers = []
        for elem in range(transducer.n_elements):
            for sub in range(transducer.no_sub_x * transducer.no_sub_y):
                verts = transducer.sub_quad_verts[
                    elem * (transducer.no_sub_x * transducer.no_sub_y) + sub]
                centers.append(verts.mean(axis=0))
        C = np.array(centers, dtype=np.float32)

        # Learnable apodization & delays per patch
        init_apod = torch.tensor(
            np.repeat(transducer.apodization, transducer.no_sub_x * transducer.no_sub_y),
            dtype=torch.float32, device=device)
        init_delay = torch.tensor(
            np.repeat(transducer.delays, transducer.no_sub_x * transducer.no_sub_y),
            dtype=torch.float32, device=device)

        self.apods = nn.Parameter(init_apod)       # (M,)
        self.delays = nn.Parameter(init_delay)     # (M,)
        self.centers = torch.tensor(C, dtype=torch.float32,
                                    device=device)  # (M,3) static

    def forward(self, field_points, pt_batch_size=2048, p_batch_size=16):
        """
        Args:
            field_points: (P,3) numpy or torch tensor in meters
            pt_batch_size: number of spatial points per sub-batch
            p_batch_size: number of patches per sub-batch
        Returns:
            amp: (P,) tensor of pressure amplitudes
        """
        # Ensure tensor
        if not isinstance(field_points, torch.Tensor):
            pts_all = torch.tensor(field_points, dtype=torch.float32,
                                   device=self.device)
        else:
            pts_all = field_points.to(self.device)
        P_total = pts_all.size(0)
        M = self.centers.size(0)

        results = []
        # Loop over spatial point batches
        print(f"Computing pressure field for {P_total} points with {M} patches...")
        for i0 in range(0, P_total, pt_batch_size):
            pts = pts_all[i0:i0+pt_batch_size]        # (P,3)
            P = pts.size(0)

            # Compute events: (P, M, 5)
            vec = pts.unsqueeze(1) - self.centers.unsqueeze(0)
            dist = vec.norm(dim=2).clamp(min=1e-6)
            xp, yp = vec[:,:,0]/dist, vec[:,:,1]/dist

            Dt1 = torch.min(self.wx*xp/self.c, self.wy*yp/self.c).abs()
            Dt2 = torch.max(self.wx*xp/self.c, self.wy*yp/self.c).abs().clamp(min=1.0/self.fs)
            area = (self.wx*self.wy)/(2*np.pi*dist)

            t1 = dist/self.c - 0.5*(Dt1 + Dt2) + self.delays.unsqueeze(0)
            t2 = t1 + Dt1;  t3 = t1 + Dt2;  t4 = t1 + Dt1 + Dt2
            h_max = area * self.apods.unsqueeze(0) / Dt2     # (P, M)

            # Global time grid from events
            t0 = t1.min();  tN = t4.max()
            n2 = int(2**np.ceil(np.log2((tN - t0).item()*self.fs)))
            n2 = max(n2, 32)
            dt = 1.0/self.fs
            k = t0 + torch.arange(n2, device=self.device) * dt  # (n2,)

            # Allocate output
            h_out = torch.zeros(P, n2, device=self.device)

            # Batch over patches to limit memory
            for j0 in range(0, M, p_batch_size):
                # slice events for these patches
                t1_b = t1[:, j0:j0+p_batch_size].unsqueeze(-1)  # (P,b,1)
                t2_b = t2[:, j0:j0+p_batch_size].unsqueeze(-1)
                t3_b = t3[:, j0:j0+p_batch_size].unsqueeze(-1)
                t4_b = t4[:, j0:j0+p_batch_size].unsqueeze(-1)
                h_b  = h_max[:, j0:j0+p_batch_size].unsqueeze(-1)

                # Broadcast k: (1,1,n2)
                k_b = k.view(1,1,-1)
                # Vectorized trapezoid for this block: (P,b,n2)
                h_block = torch.where(k_b < t1_b, 0.0,
                    torch.where(k_b < t2_b, h_b * (k_b - t1_b)/(t2_b - t1_b),
                    torch.where(k_b < t3_b, h_b,
                    torch.where(k_b < t4_b, h_b * (t4_b - k_b)/(t4_b - t3_b), 0.0))))
                # Sum over patches in block and accumulate
                h_out += h_block.sum(dim=1)

            # FFT & select transmit freq
            H = torch.fft.rfft(h_out.T.unsqueeze(0), dim=1)  # (1, P, n_freq)
            freqs = torch.linspace(0, self.fs/2, H.size(-1), device=self.device)
            idx = (freqs - self.fc).abs().argmin()
            amp_batch = H[0, :, idx].abs()                   # (P,)
            results.append(amp_batch)

        return torch.cat(results, dim=0)