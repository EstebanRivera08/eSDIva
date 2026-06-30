"""Rayleigh-Sommerfeld gold standard for validating PyField's SIR / pulse-echo.

The direct numerical Rayleigh integral is the exact continuous reference both
PyField (far-field-trapezoid sum) and Field II (exact-rect sum) must converge to:

    h(r_p, t) = integral_S  delta(t - |r_p - r_s| / c) / (2*pi*|r_p - r_s|)  dS

Implementation: tile the aperture into K tiny surface elements (area dA_k,
distance R_k); each drops weight dA_k/(2*pi*R_k) at delay R_k/c, linearly split
across 2 time bins and scaled by fs (unit-area delta -> sampled density). As
K -> inf this is exact.

Findings for the concave Ø16 mm, focus 80 mm bowl, scatterer at z=30 mm, fs=100 MHz:
  * PyField h_sir converges to the Rayleigh gold SIR (no_sub=64 overlays it).
  * On-axis pulse-echo null depth: Rayleigh gold = PyField = ~6.1 dB
    (invariant to surface density, bowl faceting, and signal-chain variant).
  * Field II reports ~7.8 dB — DEEPER than the exact physics. That extra depth is
    internal to Field II's calc_hhp (its time-bin SIR integration / derivative
    model); it is not the ground truth. PyField matches Rayleigh-Sommerfeld.

Run:  uv run python examples/_research/rayleigh_gold_standard.py
"""

import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from scipy.fft import irfft, rfft, rfftfreq
from scipy.signal import hilbert

from pyfield.emission import Emission
from pyfield.reception import Reception, ReceptionSDI
from pyfield.transducers import ConcaveCircularTransducer

D_MM, FOCUS_MM, F0, FS, C, ZS = 16.0, 80.0, 3e6, 100e6, 1540.0, 30.0
R_CURV, R_AP = FOCUS_MM * 1e-3, D_MM / 2 * 1e-3
X = np.linspace(-10, 10, 101)
CI = 50  # on-axis lateral index

t_ir = np.arange(0, 2 / F0, 1 / FS)
IR = np.sin(2 * np.pi * F0 * t_ir) * np.hanning(len(t_ir))
EXC = np.sin(2 * np.pi * F0 * t_ir)


# ---------------------------------------------------------------------------
# Rayleigh gold standard
# ---------------------------------------------------------------------------
def sample_bowl(n_theta, n_phi):
    """Dense spherical-cap surface samples (apex z=0, sphere centre z=+R)."""
    th_max = np.arcsin(R_AP / R_CURV)
    th = (np.arange(n_theta) + 0.5) / n_theta * th_max
    ph = (np.arange(n_phi) + 0.5) / n_phi * 2 * np.pi
    TH, PH = np.meshgrid(th, ph, indexing="ij")
    x = R_CURV * np.sin(TH) * np.cos(PH)
    y = R_CURV * np.sin(TH) * np.sin(PH)
    z = R_CURV * (1 - np.cos(TH))
    dA = (R_CURV**2 * np.sin(TH) * (th_max / n_theta) * (2 * np.pi / n_phi)).ravel()
    return np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1), dA


def rayleigh_sir(field_pt_m, surf, dA, t0, T):
    """One-way SIR via the binned Rayleigh integral. Returns (T,) float64."""
    diff = surf - field_pt_m[None, :]
    R = np.sqrt((diff * diff).sum(axis=1))
    w = dA / (2 * np.pi * R)
    kf = (R / C - t0) * FS
    k0 = np.floor(kf).astype(np.int64)
    frac = kf - k0
    h = np.zeros(T + 2)
    inb = (k0 >= 0) & (k0 < T)
    np.add.at(h, k0[inb], w[inb] * (1 - frac[inb]))
    np.add.at(h, k0[inb] + 1, w[inb] * frac[inb])
    return h[:T] * FS


def pyfield_hsir(no_sub, point_mm):
    """PyField raw one-way SIR (Emission pulsed-pure, no excitation)."""
    tx = ConcaveCircularTransducer(diameter_mm=D_MM, focus_mm=FOCUS_MM,
                                   frequency_Hz=F0, refine_factor=1,
                                   no_sub_diameter=no_sub)
    sim = Emission(tx, fs=FS, c=C, verbose=False)
    p, co = sim(np.asarray([point_mm], dtype=np.float32))
    h = np.asarray(p).ravel()
    return h, (co["t0"] + np.arange(h.shape[0]) / FS) * 1e6


def on_axis_psf(sir_fn, t0, T, n_ir=2):
    """Pulse-echo PSF envelope on-axis: (jw)^1 * exc * ir^n_ir * (h conv h).

    n_ir = 2 matches Field II calc_hhp(Th, Th) (emit + receive IR). n_ir = 1
    provided only to demonstrate it gives the wrong FWHM (see console check)."""
    nfft = 1 << int((2 * T + len(EXC)) - 1).bit_length()
    chain = rfft(EXC, nfft) * rfft(IR, nfft) ** n_ir * (1j * 2 * np.pi * rfftfreq(nfft, 1 / FS))
    h = sir_fn(np.array([0.0, 0.0, ZS * 1e-3]), t0, T)
    H = rfft(h, nfft)
    pe = irfft(H * H * chain, nfft)[:T]
    return np.abs(hilbert(pe))


def dip_db(env, t):
    e = env / env.max()
    pk = t[np.argmax(e)]
    seg = e[(t >= pk - 0.35) & (t <= pk + 0.35)]
    return -20 * np.log10(seg.min() + 1e-9)


def fwhm_ns(env, t):
    e = env / env.max()
    a = np.where(e >= 0.5)[0]
    return (t[a[-1]] - t[a[0]]) * 1e3 if len(a) > 1 else 0.0


def pyfield_psf_onaxis(cls):
    """PyField on-axis pulse-echo envelope + time axis (us). cls = Reception or
    ReceptionSDI. Same IR + excitation as the Rayleigh chain for a fair match."""
    def mk(exc=False):
        tx = ConcaveCircularTransducer(diameter_mm=D_MM, focus_mm=FOCUS_MM,
                                       frequency_Hz=F0, refine_factor=1,
                                       no_sub_diameter=16)
        tx.impulse_response = IR
        if exc:
            tx.excitation = EXC
        return tx
    kw = dict(fs=FS, c=C, verbose=False)
    sim = cls(mk(True), mk(), method="FST", **kw) if cls is Reception else cls(mk(True), mk(), **kw)
    rf, co = sim.pulse_echo_response(np.array([[0.0, 0.0, ZS]], np.float32),
                                     per_scatterer=True)
    env = np.abs(hilbert(rf[0, :, 0].astype(float)))
    return env, (co["t0"] + np.arange(env.shape[0]) / FS) * 1e6


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
surf, dA = sample_bowl(400, 720)
print(f"Rayleigh surface: {len(dA):,} elements "
      f"(area {dA.sum() * 1e6:.3f} / {np.pi * R_AP**2 * 1e6:.3f} mm^2)")

# True min one-way surface distance on-axis (apex, not rim) — anchors t0 so the
# binned SIR is not truncated on the left.
fp_axis = np.array([0.0, 0.0, ZS * 1e-3])
R_min = np.sqrt(((surf - fp_axis[None, :]) ** 2).sum(1)).min()

# --- A. one-way SIR overlay ---
t0a = R_min / C - 0.3e-6
Ta = 1200
ta = (t0a + np.arange(Ta) / FS) * 1e6
h_gold = rayleigh_sir(fp_axis, surf, dA, t0a, Ta)
h16, t16 = pyfield_hsir(16, [0.0, 0.0, ZS])
h64, t64 = pyfield_hsir(64, [0.0, 0.0, ZS])

# --- B. on-axis pulse-echo PSF: gold vs PyField vs Field II ---
t0b = R_min / C - 0.5e-6
Tb = 1600
# h is referenced to t0b; the pulse-echo autoconvolution (h conv h) is therefore
# referenced to 2*t0b.
tb = (2 * t0b + np.arange(Tb) / FS) * 1e6
env_gold = on_axis_psf(lambda fp, t0, T: rayleigh_sir(fp, surf, dA, t0, T), t0b, Tb)

m = scipy.io.loadmat("rf_concave_psf.mat", simplify_cells=True)["save_struct"]
fii_ax = 10 ** (m["rf_env_dB"][:, CI] / 20)
fii_t = (float(m["t0"]) + np.arange(len(fii_ax)) * 5.0 / FS) * 1e6

# PyField pulse-echo envelopes (both backends), real time axes.
env_sdi, t_sdi = pyfield_psf_onaxis(ReceptionSDI)
env_FST, t_FST = pyfield_psf_onaxis(Reception)

print("\n=== On-axis pulse-echo: null depth + FWHM ===")
print(f"  {'source':30s} {'dip(dB)':>8s} {'FWHM(ns)':>9s}")
print(f"  {'Rayleigh gold (exact)':30s} {dip_db(env_gold, tb):8.1f} {fwhm_ns(env_gold, tb):9.0f}")
print(f"  {'PyField FST':30s} {dip_db(env_FST, t_FST):8.1f} {fwhm_ns(env_FST, t_FST):9.0f}")
print(f"  {'PyField SDI':30s} {dip_db(env_sdi, t_sdi):8.1f} {fwhm_ns(env_sdi, t_sdi):9.0f}")
print(f"  {'Field II reference':30s} {dip_db(fii_ax, fii_t):8.1f} {fwhm_ns(fii_ax, fii_t):9.0f}")

# IR-count hypothesis: does using the impulse response ONCE (not twice) match
# Field II? It does not — one IR makes the PSF too narrow (wrong FWHM). Field II
# calc_hhp(Th, Th) convolves BOTH apertures' IRs, so ir^2 is correct.
gold_ir1 = on_axis_psf(lambda fp, t0, T: rayleigh_sir(fp, surf, dA, t0, T), t0b, Tb, n_ir=1)
print("\n  -- IR-count check on the exact gold SIR --")
print(f"  {'gold, ir^2 (= calc_hhp)':30s} {dip_db(env_gold, tb):8.1f} {fwhm_ns(env_gold, tb):9.0f}")
print(f"  {'gold, ir^1 (one IR)':30s} {dip_db(gold_ir1, tb):8.1f} {fwhm_ns(gold_ir1, tb):9.0f}")

# Absolute t0 conventions differ across the 3 engines + the manual gold chain
# (exc/IR group delay, SIR-only t0). Peak-align to compare SHAPE + null depth.
def rel(t, e):
    return t - t[np.argmax(e)]

fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
ax[0].plot(ta, h_gold, "k", lw=2, label="Rayleigh gold")
ax[0].plot(t16, h16, label="PyField no_sub=16")
ax[0].plot(t64, h64, "--", label="PyField no_sub=64")
ax[0].set_xlim(19.4, 20.1)
ax[0].set_title("One-way SIR, on-axis z=30 mm")
ax[0].set_xlabel("Time (us)"); ax[0].legend(); ax[0].grid(alpha=.3)

ax[1].plot(rel(tb, env_gold), env_gold / env_gold.max(), "k", lw=2.5, label="Rayleigh gold")
ax[1].plot(rel(t_FST, env_FST), env_FST / env_FST.max(), label="PyField FST")
ax[1].plot(rel(t_sdi, env_sdi), env_sdi / env_sdi.max(), "--", label="PyField SDI")
ax[1].plot(rel(fii_t, fii_ax), fii_ax / fii_ax.max(), ":", lw=2, label="Field II")
ax[1].set_xlim(-1.0, 1.0)
ax[1].set_title("On-axis pulse-echo envelope (peak-aligned)")
ax[1].set_xlabel("Time relative to peak (us)"); ax[1].legend(); ax[1].grid(alpha=.3)
plt.tight_layout()
plt.savefig("_research/rayleigh_gold_standard.png", dpi=130)
print("\nsaved _research/rayleigh_gold_standard.png")
