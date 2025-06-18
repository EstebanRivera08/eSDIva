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

# JIT-compiled core event computation
@torch.jit.script
def compute_patch_events(
    wx: float, wy: float,
    xp: torch.Tensor, yp: torch.Tensor,
    dist: torch.Tensor,
    delays: torch.Tensor, apods: torch.Tensor,
    c: float, fs: float
) -> torch.Tensor:
    Dt1 = torch.min((wx * xp / c).abs(), (wy * yp / c).abs())
    Dt2 = torch.max((wx * xp / c).abs(), (wy * yp / c).abs()).clamp(min=1.0/fs)
    area = (wx * wy) / (2 * np.pi * dist)
    t1 = dist / c - 0.5 * (Dt1 + Dt2) + delays
    t2 = t1 + Dt1
    t3 = t1 + Dt2
    t4 = t1 + Dt1 + Dt2
    hmax = area * apods / Dt2
    return torch.stack((t1, t2, t3, t4, hmax), dim=2)

class TorchField(nn.Module):
    def __init__(self, transducer, device='cuda', dtype=torch.float32):
        super().__init__()
        self.device = device; self.dtype = dtype
        self.c = 1540.0; self.fs = 300e6; self.fc = transducer.fc
        self.wx = transducer.el_w / transducer.no_sub_x
        self.wy = transducer.el_h / transducer.no_sub_y
        # Precompute centers
        centers = []
        for e in range(transducer.n_elements):
            for s in range(transducer.no_sub_x * transducer.no_sub_y):
                verts = transducer.sub_quad_verts[e*(transducer.no_sub_x*transducer.no_sub_y)+s]
                centers.append(verts.mean(axis=0))
        C = np.array(centers, dtype=np.float32)
        init_a = np.repeat(transducer.apodization, transducer.no_sub_x*transducer.no_sub_y)
        init_d = np.repeat(transducer.delays,     transducer.no_sub_x*transducer.no_sub_y)
        self.apods   = nn.Parameter(torch.tensor(init_a, dtype=dtype, device=device))
        self.delays  = nn.Parameter(torch.tensor(init_d, dtype=dtype, device=device))
        self.centers = nn.Parameter(torch.tensor(C,    dtype=dtype, device=device), requires_grad=False)

    def forward(self, field_points, pt_batch=128, p_batch=8, max_n2=512):
        pts_all = torch.as_tensor(field_points, dtype=self.dtype, device=self.device)
        P_tot = pts_all.size(0); M = self.centers.size(0)
        amps = []

        for i in range(0, P_tot, pt_batch):
            pts = pts_all[i:i+pt_batch]  # (p,3)
            # Phase 1: compute global time bounds
            t0 = torch.tensor(float('inf'), device=self.device)
            tN = torch.tensor(0.0, device=self.device)
            for j in range(0, M, p_batch):
                ctr = self.centers[j:j+p_batch]
                ap = self.apods[j:j+p_batch].unsqueeze(0)
                de = self.delays[j:j+p_batch].unsqueeze(0)
                vec = pts.unsqueeze(1) - ctr.unsqueeze(0)
                dist = vec.norm(dim=2).clamp(min=1e-6)
                xp, yp = vec[...,0]/dist, vec[...,1]/dist
                ev = compute_patch_events(self.wx, self.wy, xp, yp, dist, de, ap, self.c, self.fs)
                t0 = min(t0, ev[:,:,0].min())
                tN = max(tN, ev[:,:,3].max())

            # Build time grid
            dt = 1.0/self.fs
            n2 = int(2**np.ceil(np.log2((tN-t0).item()*self.fs)))
            n2 = min(max(n2,32), max_n2)
            k = t0 + torch.arange(n2, device=self.device, dtype=self.dtype)*dt
            h_out = torch.zeros(pts.size(0), n2, dtype=self.dtype, device=self.device)

            # Phase 2: accumulate contributions
            for j in range(0, M, p_batch):
                ctr = self.centers[j:j+p_batch]
                ap = self.apods[j:j+p_batch].unsqueeze(0)
                de = self.delays[j:j+p_batch].unsqueeze(0)
                vec = pts.unsqueeze(1) - ctr.unsqueeze(0)
                dist = vec.norm(dim=2).clamp(min=1e-6)
                xp, yp = vec[...,0]/dist, vec[...,1]/dist
                ev = compute_patch_events(self.wx, self.wy, xp, yp, dist, de, ap, self.c, self.fs)
                t1b, t2b, t3b, t4b, hmb = [ev[:,:,k].unsqueeze(-1) for k in range(5)]
                kb = k.view(1,1,-1)
                hb = torch.where(kb<t1b, 0.0,
                     torch.where(kb<t2b, hmb*(kb-t1b)/(t2b-t1b),
                     torch.where(kb<t3b, hmb,
                     torch.where(kb<t4b, hmb*(t4b-kb)/(t4b-t3b), 0.0))))
                h_out += hb.sum(dim=1)

            # FFT and pick freq
            H = torch.fft.rfft(h_out.T.unsqueeze(0), dim=1)
            freqs = torch.linspace(0, self.fs/2, H.size(-1), device=self.device)
            idx = (freqs-self.fc).abs().argmin()
            amps.append(H[0,:,idx].abs())

        return torch.cat(amps)
