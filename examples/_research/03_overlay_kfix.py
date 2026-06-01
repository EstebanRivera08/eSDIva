"""Overlay on-axis + lateral envelopes for (jw)^1 vs (jw)^3 vs Field II calc_hhp.

Confirms: Field II reference (calc_hhp, 1 derivative) matches PyField with
k=1, while PyField Reception default (k=3, ~ calc_scat) is shifted/reshaped by
the missing 2 scattering derivatives.
"""

import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from scipy.fft import irfft, rfft, rfftfreq
from scipy.signal import hilbert

from pyfield.hsir.transducer_sir import compute_h_sir
from pyfield.transducers import ConcaveCircularTransducer
from pyfield.utilities.helper_functions import (
    compute_sub_elem_attributes, compute_time_grid,
)

F0, FS, C, ZS = 3e6, 100e6, 1540.0, 30e-3
X_MM = np.linspace(-10, 10, 101)
m = scipy.io.loadmat("examples/rf_concave_psf.mat", simplify_cells=True)["save_struct"]
fii_db = m["rf_env_dB"]; fii_t0 = float(m["t0"]); fii_dt = 5.0 / FS
fii_t = fii_t0 + np.arange(fii_db.shape[0]) * fii_dt
fii_lin = 10 ** (fii_db / 20)

t_ir = np.arange(0, 2.0 / F0, 1.0 / FS)
EXC = np.sin(2 * np.pi * F0 * t_ir).astype(np.float64)
IR = (np.sin(2 * np.pi * F0 * t_ir) * np.hanning(len(t_ir))).astype(np.float64)
pts = np.column_stack([X_MM * 1e-3, np.zeros(101), np.full(101, ZS)]).astype(np.float32)
P = 101

tx = ConcaveCircularTransducer(diameter_mm=16.0, focus_mm=80.0, frequency_Hz=F0,
                               refine_factor=1, no_sub_diameter=16)
c_, a_, d_, M, _, wx, wy, idx = compute_sub_elem_attributes(tx)
fr = tx.sub_patch_frames
eu = np.asarray(fr["tangents_u"], np.float32); ev = np.asarray(fr["tangents_v"], np.float32)
tg, t0, dt, T = compute_time_grid(P, M, pts, c_, float(wx.max()), float(wy.max()),
                                  C, FS, tx.delays, verbose=False)
h_tx, _ = compute_h_sir(P, M, T, dt, tg, pts, c_, wx, wy, np.float32(1/C), FS,
                        a_, d_, 0, eu, ev)


def env_for_k(k):
    pe_T = 2 * T - 1
    nfft = 1 << int((pe_T + 3 * len(EXC) - 1)).bit_length()
    fq = rfftfreq(nfft, d=1.0 / FS); jw = 1j * 2 * np.pi * fq
    H = rfft(h_tx.astype(np.float64), n=nfft, axis=1)
    chain_f = rfft(EXC, nfft) * (jw**k) * rfft(IR, nfft) * rfft(IR, nfft)
    rf = irfft(H * H * chain_f[None, :], n=nfft, axis=1)[:, :pe_T]
    t = 2 * t0 + np.arange(pe_T) * dt
    return np.abs(hilbert(rf, axis=1)), t


e1, t1 = env_for_k(1)
e3, t3 = env_for_k(3)
ci = 50
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
ax[0].plot((t1) * 1e6, e1[ci] / e1[ci].max(), label="PyField (jw)^1  ~calc_hhp")
ax[0].plot((t3) * 1e6, e3[ci] / e3[ci].max(), "--", label="PyField (jw)^3  ~calc_scat (default)")
ax[0].plot(fii_t * 1e6, fii_lin[:, ci] / fii_lin[:, ci].max(), ":k", lw=2, label="Field II calc_hhp")
ax[0].set_xlim(39.8, 41.2); ax[0].set_xlabel("Time (us)"); ax[0].set_ylabel("norm env")
ax[0].set_title("On-axis envelope"); ax[0].legend(); ax[0].grid(alpha=0.3)

# lateral at each one's own peak row
def lat_db(e, t):
    r = int(np.argmax(e[ci])); v = e[:, r]; return 20 * np.log10(v / v.max() + 1e-9)
fr_row = int(np.argmax(fii_lin[:, ci]))
ax[1].plot(X_MM, lat_db(e1, t1), label="(jw)^1")
ax[1].plot(X_MM, lat_db(e3, t3), "--", label="(jw)^3")
ax[1].plot(X_MM, fii_db[fr_row], ":k", lw=2, label="Field II")
ax[1].set_ylim(-60, 2); ax[1].set_xlabel("Lateral (mm)"); ax[1].set_ylabel("dB")
ax[1].set_title("Lateral profile at peak"); ax[1].legend(); ax[1].grid(alpha=0.3)
fig.suptitle("Reception derivative-count: calc_hhp (k=1) vs calc_scat (k=3) vs Field II")
plt.tight_layout()
out = "examples/_research/overlay_kfix.png"
plt.savefig(out, dpi=130)
print("saved", out)
