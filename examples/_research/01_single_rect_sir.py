"""Single flat rectangle: PyField far-field trapezoid SIR vs ground-truth.

Ground truth = Rayleigh integral h(t) = integral_S delta(t - R/c)/(2*pi*R) dS,
computed by binning a very fine point-source grid into the time axis.

Tests:
  (a) 1 patch trapezoid vs ground truth
  (b) trapezoid summed over N x N subdivisions vs ground truth (convergence?)
  (c) effect of dt-clamp by sweeping fs
"""

import numpy as np

from pyfield.hsir.farfield_rect_patch import compute_h_sir

C = 1540.0


def ground_truth_sir(Wx, Wy, fp, fs, t_axis, n=2000):
    """Numerical Rayleigh SIR for a flat rectangle (normal +z) centred at origin.

    Wx, Wy : full widths (m). fp : (3,) field point (m). Returns h on t_axis.
    """
    xs = (np.arange(n) + 0.5) / n * Wx - Wx / 2
    ys = (np.arange(n) + 0.5) / n * Wy - Wy / 2
    X, Y = np.meshgrid(xs, ys)
    dA = (Wx / n) * (Wy / n)
    R = np.sqrt((fp[0] - X) ** 2 + (fp[1] - Y) ** 2 + fp[2] ** 2)
    t_arr = R / C
    w = dA / (2.0 * np.pi * R)
    h = np.zeros_like(t_axis)
    dt = 1.0 / fs
    idx = np.floor((t_arr.ravel() - t_axis[0]) / dt).astype(int)
    wr = w.ravel()
    valid = (idx >= 0) & (idx < len(t_axis))
    np.add.at(h, idx[valid], wr[valid])
    h /= dt  # histogram -> density (SIR has units of velocity, 1/dt scaling)
    return h


def trapezoid_sir(Wx, Wy, fp, fs, n_sub):
    """PyField far-field trapezoid SIR, rectangle split into n_sub x n_sub patches."""
    sx = Wx / n_sub
    sy = Wy / n_sub
    cx = (np.arange(n_sub) + 0.5) / n_sub * Wx - Wx / 2
    cy = (np.arange(n_sub) + 0.5) / n_sub * Wy - Wy / 2
    CX, CY = np.meshgrid(cx, cy)
    M = n_sub * n_sub
    centers = np.column_stack(
        [CX.ravel(), CY.ravel(), np.zeros(M)]
    ).astype(np.float32)
    wx = np.full(M, sx, np.float32)
    wy = np.full(M, sy, np.float32)
    apod = np.ones(M, np.float32)
    delays = np.zeros(M, np.float32)
    eu = np.zeros((M, 3), np.float32)
    ev = np.zeros((M, 3), np.float32)
    eu[:, 0] = 1.0
    ev[:, 1] = 1.0

    pts = fp.reshape(1, 3).astype(np.float32)
    dt = 1.0 / fs
    dist = np.linalg.norm(fp)
    # time window: cover all patch responses generously
    t0 = dist / C - (Wx + Wy) / C - 5 * dt
    T = int(((Wx + Wy) * 2 / C) / dt) + 40
    time_grid = (t0 + np.arange(T) * dt).astype(np.float32)
    h, _ = compute_h_sir(
        1, M, T, np.float32(dt), time_grid, pts, centers, wx, wy,
        np.float32(1.0 / C), np.float32(fs), apod, delays, 0, eu, ev,
    )
    return time_grid, h[0]


def stats(h, t, ref_h, ref_t):
    """Resample ref onto t, return correlation + peak/centroid."""
    ref_i = np.interp(t, ref_t, ref_h, left=0, right=0)
    a, b = h - h.mean(), ref_i - ref_i.mean()
    corr = float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-30))
    pk_t = t[np.argmax(h)]
    cen = (t * h).sum() / (h.sum() + 1e-30)
    return corr, pk_t, cen


if __name__ == "__main__":
    Wx = Wy = 1e-3  # 1 mm element (matches Field II concave math element)
    for fp_mm in ([10.0, 0.0, 30.0], [0.5, 0.0, 30.0]):
        fp = np.array(fp_mm) * 1e-3
        print(f"\n=== field point {fp_mm} mm, dist={np.linalg.norm(fp)*1e3:.2f} mm ===")
        for fs in (100e6, 400e6):
            dt = 1.0 / fs
            t_gt, h1 = trapezoid_sir(Wx, Wy, fp, fs, 1)
            gt = ground_truth_sir(Wx, Wy, fp, fs, t_gt)
            c1, pk1, cen1 = stats(h1, t_gt, gt, t_gt)
            cg, pkg, ceng = stats(gt, t_gt, gt, t_gt)
            line = f"  fs={fs/1e6:.0f}MHz  1patch corr={c1:.4f}"
            for nsub in (2, 4, 8, 16):
                _, hn = trapezoid_sir(Wx, Wy, fp, fs, nsub)
                cn, pkn, cenn = stats(hn, t_gt, gt, t_gt)
                line += f"  {nsub}^2 corr={cn:.4f}"
            print(line)
            print(f"      peak t: 1patch={pk1*1e6:.4f} gt={pkg*1e6:.4f}us  "
                  f"centroid 1patch={cen1*1e6:.4f} gt={ceng*1e6:.4f}us  "
                  f"support_gt={(t_gt[gt>gt.max()*0.01][[0,-1]]*1e6)}")
