"""Quantify the *remaining* PyField vs Field II difference after the k=1 fix.

Key question: is the residual a WIDTH difference (true smoothing → SIR shape) or
a POSITION/phase difference (group delay)? Measures on-axis envelope FWHM and
peak time, lateral -6 dB width, and renders overlays + difference map.
"""

import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from scipy.signal import hilbert

from pyfield.reception import ReceptionSDI
from pyfield.transducers import ConcaveCircularTransducer

F0, FS, C, ZS = 3e6, 100e6, 1540.0, 30.0
X = np.linspace(-10, 10, 101)
m = scipy.io.loadmat("examples/rf_concave_psf.mat", simplify_cells=True)["save_struct"]
fii_db = m["rf_env_dB"]; fii_t0 = float(m["t0"]); fii_dt = 5.0 / FS
fii_t = (fii_t0 + np.arange(fii_db.shape[0]) * fii_dt) * 1e6
fii_lin = 10 ** (fii_db / 20)


def mk(exc=False):
    tx = ConcaveCircularTransducer(diameter_mm=16.0, focus_mm=80.0, frequency_Hz=F0,
                                   refine_factor=1, no_sub_diameter=16)
    t = np.arange(0, 2 / F0, 1 / FS)
    tx.impulse_response = np.sin(2 * np.pi * F0 * t) * np.hanning(len(t))
    if exc:
        tx.excitation = np.sin(2 * np.pi * F0 * t)
    return tx


pts = np.column_stack([X, np.zeros(101), np.full(101, ZS)]).astype(np.float32)
sim = ReceptionSDI(mk(True), mk(), fs=FS, c=C, verbose=False)
rf, co = sim.pulse_echo_response(pts, per_scatterer=True)
env = np.abs(hilbert(rf[:, :, 0].T.astype(float), axis=0))
t_us = (co["t0"] + np.arange(env.shape[0]) * co["dt"]) * 1e6
ci = 50


def fwhm(y, x):
    y = y / y.max(); above = np.where(y >= 0.5)[0]
    return x[above[-1]] - x[above[0]] if len(above) > 1 else 0.0


# on-axis temporal FWHM (us)
py_ax = env[:, ci]
fii_ax = fii_lin[:, ci]
print("=== on-axis temporal envelope ===")
print(f"  PyField FWHM = {fwhm(py_ax, t_us)*1e3:.1f} ns   peak t = {t_us[np.argmax(py_ax)]:.3f} us")
print(f"  Field II FWHM = {fwhm(fii_ax, fii_t)*1e3:.1f} ns   peak t = {fii_t[np.argmax(fii_ax)]:.3f} us")

# lateral -6 dB width at each peak row
def lat6(env2d, taxis, x):
    r = int(np.argmax(env2d[:, ci])); v = env2d[r] / env2d[r].max()
    a = np.where(v >= 0.5)[0]
    return x[a[-1]] - x[a[0]] if len(a) > 1 else 0.0
print("\n=== lateral -6 dB width ===")
print(f"  PyField = {lat6(env, t_us, X):.2f} mm   Field II = {lat6(fii_lin, fii_t, X):.2f} mm")

fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
ax[0].plot(t_us, py_ax / py_ax.max(), label="PyField SDI (calc_hhp)")
ax[0].plot(fii_t, fii_ax / fii_ax.max(), ":k", lw=2, label="Field II")
ax[0].set_xlim(39.6, 41.2); ax[0].set_title("On-axis envelope"); ax[0].legend(); ax[0].grid(alpha=.3)
ax[0].set_xlabel("Time (us)")

# align peaks to compare SHAPE independent of position
shift = fii_t[np.argmax(fii_ax)] - t_us[np.argmax(py_ax)]
ax[1].plot(t_us + shift, py_ax / py_ax.max(), label=f"PyField (shifted {shift*1e3:.0f}ns)")
ax[1].plot(fii_t, fii_ax / fii_ax.max(), ":k", lw=2, label="Field II")
ax[1].set_xlim(39.6, 41.2); ax[1].set_title("Peak-aligned (shape only)"); ax[1].legend(); ax[1].grid(alpha=.3)
ax[1].set_xlabel("Time (us)")

rr = int(np.argmax(env[:, ci])); fr = int(np.argmax(fii_lin[:, ci]))
ax[2].plot(X, 20 * np.log10(env[rr] / env[rr].max() + 1e-9), label="PyField")
ax[2].plot(X, fii_db[fr], ":k", lw=2, label="Field II")
ax[2].set_ylim(-60, 2); ax[2].set_title("Lateral profile"); ax[2].legend(); ax[2].grid(alpha=.3)
ax[2].set_xlabel("Lateral (mm)")
plt.tight_layout()
plt.savefig("examples/_research/remaining_diff.png", dpi=130)
print("\nsaved examples/_research/remaining_diff.png")
