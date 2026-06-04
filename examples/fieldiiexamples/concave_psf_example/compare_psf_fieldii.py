"""PSF comparison: PyField (naive + SDI) vs Field II ``calc_hhp``.

Three stages, top to bottom:

1. Waveform comparison    — on-axis raw RF traces overlaid (normalised).
2. dB envelope comparison — naive / SDI / Field II envelope images + difference.
3. Peak profile comparison — lateral profile at envelope peak, axial profile
   on-axis (both in dB).

Field II reference geometry (``example_concave_psf.m``):
  - Concave circular, Ø 16 mm, focal depth 80 mm, 3 MHz.
  - 2-cycle Hanning-windowed sine impulse response, plain sine excitation.
  - Scatterers: 101 lateral positions, z = 30 mm, fs = 100 MHz.

The raw Field II RF is loaded from ``psf_concave_RFdata.mat`` (full fs, signed),
giving a true waveform comparison rather than only the down-sampled dB envelope.

Note: PyField and Field II carry different absolute-amplitude conventions
(ratio ~1e24) and opposite polarity for ``calc_hhp``; every panel here is
normalised, so only shape, timing, and dB structure are compared.

Run with:
    uv run examples/fieldiiexamples/concave_psf_example/compare_psf_fieldii.py
"""

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from scipy.signal import correlate, hilbert

sys.path.insert(0, str(Path(__file__).parents[2]))
from config import FIG_FOLDER, SAVE_FIG

from pyfield.reception import Reception, ReceptionSDI
from pyfield.transducers import ConcaveCircularTransducer
from pyfield.transducers.base import TransducerBase
from pyfield.utilities import to_dB

# ---------------------------------------------------------------------------
# Geometry / simulation parameters (must match Field II reference)
# ---------------------------------------------------------------------------
DIAMETER_MM = 16.0
FOCUS_MM = 80.0
FREQUENCY_HZ = 3e6
FS = 100e6
C = 1540.0
SCATTERER_Z_MM = 30.0
DB_FLOOR = -60.0

X_SCAT_MM = np.linspace(-10.0, 10.0, 101)
CENTER_IDX = len(X_SCAT_MM) // 2

# When True, build TX/RX from the actual Field II aperture (concave_data.mat =
# xdc_get(Th,'all')) so both simulators share identical patch sampling. When
# False, use PyField's own ConcaveCircularTransducer discretisation.
TX_FROM_FIELDII = True

# When True, use Field II's own impulse_response + excitation (stored in
# psf_concave_RFdata.mat) instead of regenerating them. Removes the
# np.hanning vs Matlab hanning window mismatch as a confound.
USE_FIELDII_PULSES = True

HERE = Path(__file__).parent

# ---------------------------------------------------------------------------
# Load Field II reference (raw, full-fs, signed RF from calc_hhp)
# ---------------------------------------------------------------------------
_fii = scipy.io.loadmat(str(HERE / "psf_concave_RFdata.mat"), simplify_cells=True)
fii_rf = np.asarray(_fii["RF_data"])  # (Nt_fii, 101), signed
fii_t0 = float(_fii["start_time"])
fii_t_us = (fii_t0 + np.arange(fii_rf.shape[0]) / FS) * 1e6
# Field II's own pulses (added to the mat file) — exact, no window mismatch.
FII_IR = np.ravel(_fii["impulse_response"]).astype(np.float32)
FII_EXC = np.ravel(_fii["excitation"]).astype(np.float32)
print(
    f"Field II: Nt={fii_rf.shape[0]}, t0={fii_t0 * 1e6:.3f} us, maxabs={np.abs(fii_rf).max():.3e}"
)

# IR comparison: Matlab hanning (no zero endpoints) vs np.hanning (with them).
_t_ir = np.arange(0, 2.0 / FREQUENCY_HZ, 1.0 / FS)
_ir_py = np.sin(2 * np.pi * FREQUENCY_HZ * _t_ir) * np.hanning(len(_t_ir))
print(
    f"Impulse response Field II vs np.hanning: maxabs {np.abs(FII_IR).max():.4f} vs "
    f"{np.abs(_ir_py).max():.4f}, max|diff| {np.abs(FII_IR - _ir_py).max():.3e} "
    f"(window-shape mismatch)"
)


# ---------------------------------------------------------------------------
# Field II aperture wrapper: the exact mathematical-element patches Field II
# used, loaded from concave_data.mat = xdc_get(Th, 'all'). Row layout (one
# column per rectangle): rows 2,3 = width/height, row 4 = apodization,
# rows 7-9 = centre (x,y,z), rows 10-21 = the four corner (x,y,z) — all metres.
# Feeding these makes PyField and Field II share identical aperture sampling.
# ---------------------------------------------------------------------------
class FieldIIConcaveTransducer(TransducerBase):
    """Mono-element transducer whose patches are Field II's own rectangles."""

    def __init__(self, data_all: np.ndarray, frequency_Hz: float):
        super().__init__()
        self.type = "fieldii"
        self.name = "FieldIIConcaveTransducer"
        self.n_elements = 1
        self.fc = float(frequency_Hz)
        corners = data_all[10:22, :].T.reshape(-1, 4, 3)  # (M, 4, 3) metres
        self._quads = [c.astype(np.float64) for c in corners]
        edge = np.linalg.norm(corners[:, 1] - corners[:, 0], axis=1)
        self.elem_width = float(np.median(edge))
        self.elem_height = float(np.median(edge))
        self.no_sub_x = self.no_sub_y = 1

    def _compute_element_centers(self) -> np.ndarray:
        return np.array([np.mean([q.mean(axis=0) for q in self._quads], axis=0)])

    def _build_subdivisions(self):
        areas = [
            np.linalg.norm(q[1] - q[0]) * np.linalg.norm(q[3] - q[0])
            for q in self._quads
        ]
        return self._quads, float(np.mean(areas)), [0] * len(self._quads)


def _set_pulses(tx: TransducerBase, with_excitation: bool) -> TransducerBase:
    if USE_FIELDII_PULSES:
        tx.impulse_response = FII_IR
        if with_excitation:
            tx.excitation = FII_EXC
        return tx
    t_ir = np.arange(0, 2.0 / FREQUENCY_HZ, 1.0 / FS)
    # Field II: xdc_impulse uses Hanning-windowed sine on the shared TX/RX handle.
    tx.impulse_response = np.sin(2 * np.pi * FREQUENCY_HZ * t_ir) * np.hanning(
        len(t_ir)
    )
    if with_excitation:
        # Field II: xdc_excitation uses plain sine (no window).
        tx.excitation = np.sin(2 * np.pi * FREQUENCY_HZ * t_ir)
    return tx


def _make_transducer_pyfield(with_excitation: bool = False) -> TransducerBase:
    tx = ConcaveCircularTransducer(
        diameter_mm=DIAMETER_MM,
        focus_mm=FOCUS_MM,
        frequency_Hz=FREQUENCY_HZ,
        refine_factor=1,
        no_sub_diameter=16,
    )
    return _set_pulses(tx, with_excitation)


_FII_DATA = scipy.io.loadmat(str(HERE / "concave_data.mat"), simplify_cells=True)[
    "data"
]


def _make_transducer_from_fieldii(with_excitation: bool = False) -> TransducerBase:
    tx = FieldIIConcaveTransducer(_FII_DATA, FREQUENCY_HZ)
    return _set_pulses(tx, with_excitation)


make_tx = _make_transducer_from_fieldii if TX_FROM_FIELDII else _make_transducer_pyfield


field_points_mm = np.column_stack(
    [
        X_SCAT_MM,
        np.zeros_like(X_SCAT_MM),
        np.full_like(X_SCAT_MM, SCATTERER_Z_MM),
    ]
).astype(np.float32)

# ---------------------------------------------------------------------------
# Run both PyField backends (calc_hhp = pulse_echo_rf, 0 derivatives)
# ---------------------------------------------------------------------------
print(f"\nSimulating {len(X_SCAT_MM)} lateral positions at z={SCATTERER_Z_MM} mm ...")

print("\n  [1/2] Reception(method='naive') ...")
t_start = time.time()
sim_naive = Reception(
    make_tx(with_excitation=True),
    make_tx(),
    fs=FS,
    c=C,
    method="naive",
    verbose=False,
)
rf_naive, coords_naive = sim_naive.pulse_echo_rf(field_points_mm, per_scatterer=True)
t_naive = time.time() - t_start
print(f"  Done in {t_naive:.2f} s")

print("\n  [2/2] ReceptionSDI() ...")
t_start = time.time()
sim_sdi = ReceptionSDI(
    make_tx(with_excitation=True),
    make_tx(),
    fs=FS,
    c=C,
    verbose=False,
)
rf_sdi, coords_sdi = sim_sdi.pulse_echo_rf(field_points_mm, per_scatterer=True)
t_sdi = time.time() - t_start
print(f"  Done in {t_sdi:.2f} s")

# ---------------------------------------------------------------------------
# Reshape to (Nt, N_lat) and build time axes (µs)
# ---------------------------------------------------------------------------
# per_scatterer → (P, Erx, Nt); mono-element → channel 0 → (P, Nt) → .T = (Nt, P).
rf_naive_img = rf_naive[:, 0, :].T
rf_sdi_img = rf_sdi[:, 0, :].T
t_us_naive = (
    coords_naive["t0"] + np.arange(rf_naive_img.shape[0]) * coords_naive["dt"]
) * 1e6
t_us_sdi = (coords_sdi["t0"] + np.arange(rf_sdi_img.shape[0]) * coords_sdi["dt"]) * 1e6

# Envelopes (per-array peak normalisation — absolute scales are non-comparable).
env_naive = np.abs(hilbert(rf_naive_img, axis=0))
env_sdi = np.abs(hilbert(rf_sdi_img, axis=0))
env_fii = np.abs(hilbert(fii_rf, axis=0))

env_naive_db = to_dB(env_naive / env_naive.max(), vmin=10 ** (DB_FLOOR / 20))
env_sdi_db = to_dB(env_sdi / env_sdi.max(), vmin=10 ** (DB_FLOOR / 20))
env_fii_db = to_dB(env_fii / env_fii.max(), vmin=10 ** (DB_FLOOR / 20))

# Diagnostics: peak times and polarity.
pk = lambda e, t: t[int(np.argmax(e[:, CENTER_IDX]))]
print("\nOn-axis envelope peak time:")
print(f"  Field II : {pk(env_fii, fii_t_us):.3f} us")
print(f"  Naive    : {pk(env_naive, t_us_naive):.3f} us")
print(f"  SDI      : {pk(env_sdi, t_us_sdi):.3f} us")
sign = lambda x: np.sign(x[np.argmax(np.abs(x))])
print(
    f"On-axis polarity (largest-sample sign): "
    f"naive={sign(rf_naive_img[:, CENTER_IDX]):.0f}  "
    f"SDI={sign(rf_sdi_img[:, CENTER_IDX]):.0f}  "
    f"FieldII={sign(fii_rf[:, CENTER_IDX]):.0f}"
)

# ===========================================================================
# FIGURE: 3 stages stacked as rows
# ===========================================================================
fig = plt.figure(figsize=(16, 11))
gs = fig.add_gridspec(
    3,
    4,
    height_ratios=[1, 1.2, 1],
    hspace=0.4,
    wspace=0.3,
    width_ratios=[1, 1, 1, 0.05],  # Last column narrow for colorbars
)

# Common time window centred on the PSF (focus ~1 µs wide; full traces are
# mostly empty tail). Tightening makes the envelope structure visible.
_pk_t = pk(env_fii, fii_t_us)
t_lo, t_hi = _pk_t - 1.0, _pk_t + 1.5

# ---------------------------------------------------------------------------
# STAGE 1 — Raw RF maps (signed, per-array max-abs normalised → sign preserved)
# ---------------------------------------------------------------------------
ext_naive = [X_SCAT_MM[0], X_SCAT_MM[-1], t_us_naive[-1], t_us_naive[0]]
ext_sdi = [X_SCAT_MM[0], X_SCAT_MM[-1], t_us_sdi[-1], t_us_sdi[0]]
ext_fii = [X_SCAT_MM[0], X_SCAT_MM[-1], fii_t_us[-1], fii_t_us[0]]
kw_rf = dict(aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
norm = lambda x: x / (np.abs(x).max() + 1e-30)

axes_rf = []
for ax, img, ext, title in [
    (fig.add_subplot(gs[0, 0]), norm(rf_naive_img), ext_naive, "Reception naive — RF"),
    (fig.add_subplot(gs[0, 1]), norm(rf_sdi_img), ext_sdi, "ReceptionSDI — RF"),
    (fig.add_subplot(gs[0, 2]), norm(fii_rf), ext_fii, "Field II — RF"),
]:
    im = ax.imshow(img, extent=ext, **kw_rf)
    ax.set_ylim(t_hi, t_lo)  # time increases downward, common window
    ax.set_xlabel("Lateral (mm)")
    ax.set_ylabel("Time (µs)")
    ax.set_title(title)
    axes_rf.append((ax, im))

# Shared colorbar for RF row
cax_rf = fig.add_subplot(gs[0, 3])
cbar_rf = plt.colorbar(axes_rf[0][1], cax=cax_rf, label="norm. RF")


# ---------------------------------------------------------------------------
# STAGE 2 — dB envelope contours spaced 5 dB (naive | SDI | Field II)
# ---------------------------------------------------------------------------
# Contour levels spaced 5 dB from floor to peak
contour_levels = np.arange(DB_FLOOR, 0, 6)

axes_contour = []
for ax, img, ext, x_dat, t_dat, title in [
    (
        fig.add_subplot(gs[1, 0]),
        env_naive_db,
        ext_naive,
        X_SCAT_MM,
        t_us_naive,
        "Reception naive — envelope (dB)",
    ),
    (
        fig.add_subplot(gs[1, 1]),
        env_sdi_db,
        ext_sdi,
        X_SCAT_MM,
        t_us_sdi,
        "ReceptionSDI — envelope (dB)",
    ),
    (
        fig.add_subplot(gs[1, 2]),
        env_fii_db,
        ext_fii,
        X_SCAT_MM,
        fii_t_us,
        "Field II — envelope (dB)",
    ),
]:
    # Create mesh grid for contour plot
    CMAP = "jet"
    T, X = np.meshgrid(t_dat, x_dat, indexing="ij")
    cs = ax.contour(X, T, img, levels=contour_levels, cmap=CMAP, linewidths=0.8)
    ax.set_ylim(t_hi, t_lo)  # time increases downward, common window
    ax.set_xlabel("Lateral (mm)")
    ax.set_ylabel("Time (µs)")
    ax.set_title(title)
    axes_contour.append((ax, cs, img))

# Shared colorbar for contour row using ScalarMappable
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

sm = ScalarMappable(cmap=CMAP, norm=Normalize(vmin=DB_FLOOR, vmax=0))
sm.set_array([])
cax_contour = fig.add_subplot(gs[1, 3])
cbar_contour = plt.colorbar(sm, cax=cax_contour, label="dB")

# ---------------------------------------------------------------------------
# STAGE 3 — Peak profiles (lateral at peak | axial on-axis), dB
# ---------------------------------------------------------------------------
row_naive = int(np.argmax(env_naive.max(axis=1)))
row_sdi = int(np.argmax(env_sdi.max(axis=1)))
row_fii = int(np.argmax(env_fii.max(axis=1)))

ax = fig.add_subplot(gs[2, 0:2])
ax.plot(X_SCAT_MM, env_naive_db[row_naive, :], label="naive", lw=1.2)
ax.plot(X_SCAT_MM, env_sdi_db[row_sdi, :], "--", label="SDI", lw=1.2)
ax.plot(X_SCAT_MM, env_fii_db[row_fii, :], ":", label="Field II", lw=1.6)
ax.axhline(-6, color="grey", ls=":", lw=0.8)
ax.set_ylim(DB_FLOOR, 2)
ax.set_xlabel("Lateral (mm)")
ax.set_ylabel("Envelope (dB)")
ax.set_title("Lateral profile at envelope peak")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = fig.add_subplot(gs[2, 2])
ax.plot(t_us_naive, env_naive_db[:, CENTER_IDX], label="naive", lw=1.2)
ax.plot(t_us_sdi, env_sdi_db[:, CENTER_IDX], "--", label="SDI", lw=1.2)
ax.plot(fii_t_us, env_fii_db[:, CENTER_IDX], ":", label="Field II", lw=1.6)
ax.set_xlim(t_lo, t_hi)
ax.set_ylim(DB_FLOOR, 2)
ax.set_xlabel("Time (µs)")
ax.set_ylabel("Envelope (dB)")
ax.set_title("On-axis axial profile")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

_src = "Field II aperture" if TX_FROM_FIELDII else "PyField aperture"
fig.suptitle(
    f"PSF comparison — concave Ø{DIAMETER_MM:.0f} mm, focus {FOCUS_MM:.0f} mm, "
    f"scatterer z = {SCATTERER_Z_MM} mm   [{_src}]   "
    f"(naive {t_naive:.1f}s · SDI {t_sdi:.1f}s)",
    fontsize=13,
)

# ===========================================================================
# FIGURE 2 — Waveform match at 5 lateral positions, with lag compensation
#   Row 1: naive / SDI / Field II overlaid (normalised, signed).
#   Row 2: naive & SDI shifted to max-correlation lag vs Field II + Field II.
#   Row 3: SDI − Field II residual, before vs after lag compensation.
# ===========================================================================
X_PROBE_MM = [0.0, 2.0, 4.0, 6.0, 8.0]
norm1 = lambda x: x / (np.abs(x).max() + 1e-30)


def _best_lag(py_trace, py_t, fii_trace, fii_t):
    """Integer-sample lag maximising the *signed* (in-phase) correlation.

    Both traces are placed on the same PyField time grid, then a full
    cross-correlation scans every lag (negative and positive). We take the lag
    of the maximum signed correlation, not ``|corr|`` — otherwise the adjacent
    anti-phase peak one half-period away (the pulse is oscillatory) can win and
    roll the signal the wrong way. With lag L, ``roll(a, -L) ≈ b``.
    """
    a = norm1(py_trace)
    b = np.interp(py_t, fii_t, norm1(fii_trace), left=0.0, right=0.0)
    denom = np.sqrt((a**2).sum() * (b**2).sum()) + 1e-30
    xc = correlate(a, b, mode="full") / denom
    lags = np.arange(-len(a) + 1, len(a))
    i = int(np.argmax(xc))  # signed → in-phase alignment
    return int(lags[i]), float(xc[i]), b  # b = Field II on PyField grid


fii_t_s = fii_t0 + np.arange(fii_rf.shape[0]) / FS  # seconds, full grid

fig2, ax2 = plt.subplots(3, 5, figsize=(18, 9), sharex=True)
print(f"\n5-position lag/corr ({_src}):")
print(f"{'x[mm]':>6} {'SDI r':>8} {'SDI lag':>8} {'naive r':>8} {'naive lag':>9}")
for j, xp in enumerate(X_PROBE_MM):
    col = int(np.argmin(np.abs(X_SCAT_MM - xp)))
    a_sdi = rf_sdi_img[:, col]
    a_nai = rf_naive_img[:, col]
    b = fii_rf[:, col]

    lag_s, r_s, b_on = _best_lag(
        a_sdi, py_t := (coords_sdi["t0"] + np.arange(len(a_sdi)) / FS), b, fii_t_s
    )
    lag_n, r_n, _ = _best_lag(
        a_nai, coords_naive["t0"] + np.arange(len(a_nai)) / FS, b, fii_t_s
    )
    print(
        f"{xp:6.1f} {r_s:+8.3f} {lag_s / FS * 1e9:7.0f}n {r_n:+8.3f} {lag_n / FS * 1e9:8.0f}n"
    )

    an_s, an_n = norm1(a_sdi), norm1(a_nai)
    # Row 1: raw overlay.
    ax2[0, j].plot(t_us_sdi, an_s, label="SDI", lw=1.0)
    ax2[0, j].plot(t_us_naive, an_n, "--", label="naive", lw=1.0)
    ax2[0, j].plot(fii_t_us, norm1(b), ":", label="Field II", lw=1.4)
    ax2[0, j].set_title(f"x = {xp:.1f} mm")
    # Row 2: lag-aligned overlay (roll PyField by -lag onto Field II grid).
    ax2[1, j].plot(t_us_sdi, np.roll(an_s, -lag_s), label="SDI (lagged)", lw=1.0)
    ax2[1, j].plot(
        t_us_naive, np.roll(an_n, -lag_n), "--", label="naive (lagged)", lw=1.0
    )
    ax2[1, j].plot(fii_t_us, norm1(b), ":", label="Field II", lw=1.4)
    # Row 3: SDI − Field II residual, before vs after lag.
    ax2[2, j].plot(t_us_sdi, an_s - b_on, label="before lag", lw=1.0, color="C3")
    ax2[2, j].plot(
        t_us_sdi, np.roll(an_s, -lag_s) - b_on, label="after lag", lw=1.0, color="C2"
    )
    ax2[2, j].set_xlabel("Time (µs)")
    for r in range(3):
        ax2[r, j].set_xlim(t_lo, t_hi)
        ax2[r, j].set_ylim(-1.2, 1.2)
        ax2[r, j].grid(alpha=0.3)

ax2[0, 0].set_ylabel("RF (norm)")
ax2[1, 0].set_ylabel("RF lag-aligned")
ax2[2, 0].set_ylabel("SDI − Field II")
for r, c in [(0, 0), (1, 0), (2, 0)]:
    ax2[r, c].legend(fontsize=7, loc="upper right")
fig2.suptitle(
    f"RF waveform match vs Field II at 5 positions   [{_src}]   "
    "(row1 raw · row2 lag-aligned · row3 residual before/after lag)",
    fontsize=13,
)
fig2.tight_layout()

if SAVE_FIG:
    FIG_FOLDER.mkdir(exist_ok=True)
    fig.savefig(str(FIG_FOLDER / "compare_psf_fieldii.png"), dpi=150)
    fig2.savefig(str(FIG_FOLDER / "compare_psf_fieldii_waveforms.png"), dpi=150)
    print(f"\nSaved to {FIG_FOLDER / 'compare_psf_fieldii.png'}")
    print(f"Saved to {FIG_FOLDER / 'compare_psf_fieldii_waveforms.png'}")

plt.show()
