"""Patch-size convergence: PyField vs Field II ``calc_hhp`` across ele_size.

Field II was run at several mathematical-element sizes (2.0 → 0.125 mm); each
``concave_es_<N>um.mat`` stores ``RF_data``, ``start_time``, ``geom``
(= xdc_get(Th,'all')), ``impulse_response`` and ``excitation``. For every level
this script feeds Field II's *own* rectangles to PyField (so the only remaining
difference is the per-rectangle SIR model + method), runs both backends, and:

1. Saves one 3-stage PSF figure per level (same layout as
   ``compare_psf_fieldii.py``) into ``figures/``.
2. Tracks metrics vs ele_size — peak correlation, group/phase lag, -6 dB lateral
   beamwidth — and saves a convergence figure ``figures/convergence_metrics.png``.

Question answered: as both grids refine, does PyField↔Field II agreement → 1
(pure discretisation, same physics) or plateau (per-element SIR-model gap)?

Run with:
    uv run examples/fieldiiexamples/concave_psf_example/comparison_patches_fieldii.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from scipy.signal import correlate, hilbert

sys.path.insert(0, str(Path(__file__).parents[2]))

from pyfield.reception import Reception, ReceptionSDI
from pyfield.transducers.base import TransducerBase
from pyfield.utilities import to_dB

# ---------------------------------------------------------------------------
FREQUENCY_HZ = 3e6
FS = 100e6
C = 1540.0
SCATTERER_Z_MM = 30.0
DB_FLOOR = -60.0
X_SCAT_MM = np.linspace(-10.0, 10.0, 101)
CENTER_IDX = len(X_SCAT_MM) // 2
ELE_SIZES_UM = [2000, 1000, 500, 250, 125]

HERE = Path(__file__).parent
FIG_DIR = HERE / "figures"
FIG_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Field II aperture wrapper (rows 10-21 of xdc_get('all') = 4 corners x,y,z, m)
# ---------------------------------------------------------------------------
class FieldIIConcaveTransducer(TransducerBase):
    """Mono-element transducer whose patches are Field II's own rectangles."""

    def __init__(self, geom: np.ndarray, frequency_Hz: float):
        super().__init__()
        self.type = "fieldii"
        self.name = "FieldIIConcaveTransducer"
        self.n_elements = 1
        self.fc = float(frequency_Hz)
        corners = geom[10:22, :].T.reshape(-1, 4, 3)  # (M, 4, 3) metres
        self._quads = [c.astype(np.float64) for c in corners]
        edge = np.linalg.norm(corners[:, 1] - corners[:, 0], axis=1)
        self.elem_width = self.elem_height = float(np.median(edge))
        self.no_sub_x = self.no_sub_y = 1

    def _compute_element_centers(self):
        return np.array([np.mean([q.mean(axis=0) for q in self._quads], axis=0)])

    def _build_subdivisions(self):
        areas = [
            np.linalg.norm(q[1] - q[0]) * np.linalg.norm(q[3] - q[0])
            for q in self._quads
        ]
        return self._quads, float(np.mean(areas)), [0] * len(self._quads)


def _make(geom, ir, exc=None):
    tx = FieldIIConcaveTransducer(geom, FREQUENCY_HZ)
    tx.impulse_response = ir
    if exc is not None:
        tx.excitation = exc
    return tx


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------
def _env_peak_time(sig, t):
    """Parabolic-refined absolute time of the envelope maximum."""
    e = np.abs(hilbert(sig))
    i = int(np.argmax(e))
    if 0 < i < len(e) - 1:
        y0, y1, y2 = e[i - 1], e[i], e[i + 1]
        d = (y0 - y2) / (2 * (y0 - 2 * y1 + y2) + 1e-30)
    else:
        d = 0.0
    return t[i] + d / FS


def _windowed_corr_lag(a, at, b, bt, t_lo, t_hi, max_ns=200, osr=8):
    """Peak signed correlation + carrier lag (ns) over [t_lo,t_hi], 8x oversampled.

    Args: t_lo, t_hi in MICROSECONDS (from pkt - 1.5 etc).
    """
    # Convert µs → s for step calculation
    t_lo_s = t_lo * 1e-6
    t_hi_s = t_hi * 1e-6
    g = np.arange(t_lo_s, t_hi_s, 1.0 / (FS * osr))
    A = np.interp(g, at, a, left=0, right=0)
    B = np.interp(g, bt, b, left=0, right=0)
    A /= np.abs(A).max() + 1e-30
    B /= np.abs(B).max() + 1e-30
    xc = correlate(A, B, mode="full")
    lags = np.arange(-len(A) + 1, len(A)) / (FS * osr)
    win = np.abs(lags) <= max_ns * 1e-9
    i = int(np.argmax(np.where(win, xc, -np.inf)))
    den = np.sqrt((A**2).sum() * (B**2).sum()) + 1e-30
    return xc[i] / den, lags[i] * 1e9


def _beamwidth_6db(env2d, xax):
    """-6 dB lateral beamwidth (mm) of the contiguous main lobe.

    Span the contiguous run of columns around the peak that stay above -6 dB, so
    far sidelobes that also poke above the threshold do not inflate the width.
    """
    row = env2d[int(np.argmax(env2d.max(axis=1))), :]
    row = row / (row.max() + 1e-30)
    thr = 10 ** (-6 / 20)
    pk = int(np.argmax(row))
    lo = hi = pk
    while lo > 0 and row[lo - 1] >= thr:
        lo -= 1
    while hi < row.size - 1 and row[hi + 1] >= thr:
        hi += 1
    return (xax[hi] - xax[lo]) if hi > lo else np.nan


# ---------------------------------------------------------------------------
# Per-configuration PSF figure (same 3-stage layout as compare_psf_fieldii.py)
# ---------------------------------------------------------------------------
def _save_psf_figure(
    es_um, n_patches, peak_corr, rf_nai, t_nai, rf_sdi, t_sdi, fii_rf, fii_t_us
):
    env_nai = np.abs(hilbert(rf_nai, axis=0))
    env_sdi = np.abs(hilbert(rf_sdi, axis=0))
    env_fii = np.abs(hilbert(fii_rf, axis=0))
    env_nai_db = to_dB(env_nai / env_nai.max(), vmin=10 ** (DB_FLOOR / 20))
    env_sdi_db = to_dB(env_sdi / env_sdi.max(), vmin=10 ** (DB_FLOOR / 20))
    env_fii_db = to_dB(env_fii / env_fii.max(), vmin=10 ** (DB_FLOOR / 20))

    pk = lambda e, t: t[int(np.argmax(e[:, CENTER_IDX]))]
    t_lo, t_hi = pk(env_fii, fii_t_us) - 1.0, pk(env_fii, fii_t_us) + 1.5

    ext_n = [X_SCAT_MM[0], X_SCAT_MM[-1], t_nai[-1], t_nai[0]]
    ext_s = [X_SCAT_MM[0], X_SCAT_MM[-1], t_sdi[-1], t_sdi[0]]
    ext_f = [X_SCAT_MM[0], X_SCAT_MM[-1], fii_t_us[-1], fii_t_us[0]]
    norm = lambda x: x / (np.abs(x).max() + 1e-30)

    fig = plt.figure(figsize=(15, 11))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1.2, 1], hspace=0.4, wspace=0.3)

    # Stage 1 — signed RF maps.
    kw_rf = dict(aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    for c, (img, ext, ttl) in enumerate(
        [
            (norm(rf_nai), ext_n, "Reception naive — RF"),
            (norm(rf_sdi), ext_s, "ReceptionSDI — RF"),
            (norm(fii_rf), ext_f, "Field II — RF"),
        ]
    ):
        ax = fig.add_subplot(gs[0, c])
        im = ax.imshow(img, extent=ext, **kw_rf)
        plt.colorbar(im, ax=ax, label="norm. RF")
        ax.set_ylim(t_hi, t_lo)
        ax.set_xlabel("Lateral (mm)")
        ax.set_ylabel("Time (µs)")
        ax.set_title(ttl)

    # Stage 2 — dB envelope (6dB-spaced contours, jet colormap, shared colorbar).
    levels = np.arange(DB_FLOOR, 0 + 6, 6)  # -60, -54, -48, ..., 0 dB
    fig_axes = [fig.add_subplot(gs[1, c]) for c in range(3)]

    for ax, (img, ext, ttl) in zip(
        fig_axes,
        [
            (env_nai_db, ext_n, "naive — envelope (dB)"),
            (env_sdi_db, ext_s, "SDI — envelope (dB)"),
            (env_fii_db, ext_f, "Field II — envelope (dB)"),
        ],
    ):
        # Create meshgrid for contour
        y_idx = np.arange(img.shape[0])
        x_idx = np.arange(img.shape[1])
        x_mm = np.linspace(ext[0], ext[1], img.shape[1])
        t_us = np.linspace(ext[3], ext[2], img.shape[0])
        X, T = np.meshgrid(x_mm, t_us)

        # Filled contours with 6 dB spacing, jet colormap
        cf = ax.contourf(X, T, img, levels=levels, cmap="jet", extend="both")
        ax.contour(X, T, img, levels=levels, colors="black", linewidths=0.3, alpha=0.3)

        ax.set_ylim(t_hi, t_lo)
        ax.set_xlabel("Lateral (mm)")
        ax.set_ylabel("Time (µs)")
        ax.set_title(ttl)

    # Shared colorbar for all Stage 2 plots
    cbar_ax = fig.add_axes([0.92, 0.35, 0.015, 0.25])
    cbar = fig.colorbar(cf, cax=cbar_ax, label="dB")
    cbar.set_ticks(levels)

    # Stage 3 — lateral profile + axial profile.
    rn = int(np.argmax(env_nai.max(axis=1)))
    rs = int(np.argmax(env_sdi.max(axis=1)))
    rfi = int(np.argmax(env_fii.max(axis=1)))
    ax = fig.add_subplot(gs[2, 0:2])
    ax.plot(X_SCAT_MM, env_nai_db[rn, :], label="naive", lw=1.2)
    ax.plot(X_SCAT_MM, env_sdi_db[rs, :], "--", label="SDI", lw=1.2)
    ax.plot(X_SCAT_MM, env_fii_db[rfi, :], ":", label="Field II", lw=1.6)
    ax.axhline(-6, color="grey", ls=":", lw=0.8)
    ax.set_ylim(DB_FLOOR, 2)
    ax.set_xlabel("Lateral (mm)")
    ax.set_ylabel("Envelope (dB)")
    ax.set_title("Lateral profile at envelope peak")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[2, 2])
    ax.plot(t_nai, env_nai_db[:, CENTER_IDX], label="naive", lw=1.2)
    ax.plot(t_sdi, env_sdi_db[:, CENTER_IDX], "--", label="SDI", lw=1.2)
    ax.plot(fii_t_us, env_fii_db[:, CENTER_IDX], ":", label="Field II", lw=1.6)
    ax.set_xlim(t_lo, t_hi)
    ax.set_ylim(DB_FLOOR, 2)
    ax.set_xlabel("Time (µs)")
    ax.set_ylabel("Envelope (dB)")
    ax.set_title("On-axis axial profile")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle(
        f"PSF — Field II aperture, ele_size {es_um / 1000:.3f} mm "
        f"({n_patches} patches) · on-axis peak corr {peak_corr:+.4f}",
        fontsize=13,
    )
    out = FIG_DIR / f"patches_es_{es_um}um.png"
    fig.savefig(str(out), dpi=140)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------
field_points_mm = np.column_stack(
    [X_SCAT_MM, np.zeros_like(X_SCAT_MM), np.full_like(X_SCAT_MM, SCATTERER_Z_MM)]
).astype(np.float32)

rows = []
print(
    f"{'ele[mm]':>8} {'patches':>8} {'corr':>8} {'grp[ns]':>8} {'phs[ns]':>8} "
    f"{'BW_py':>7} {'BW_fii':>7}"
)
for es in ELE_SIZES_UM:
    print("es", es)
    d = scipy.io.loadmat(str(HERE / f"concave_es_{es}um.mat"), simplify_cells=True)
    geom = np.asarray(d["geom"])
    fii_rf = np.asarray(d["RF_data"])
    fii_t = float(d["start_time"]) + np.arange(fii_rf.shape[0]) / FS
    ir = np.ravel(d["impulse_response"]).astype(np.float32)
    exc = np.ravel(d["excitation"]).astype(np.float32)
    n_patches = geom.shape[1]

    sim_n = Reception(
        _make(geom, ir, exc), _make(geom, ir), fs=FS, c=C, method="naive", verbose=False
    )
    rf_n, cn = sim_n.pulse_echo_rf(field_points_mm, per_scatterer=True)
    sim_s = ReceptionSDI(
        _make(geom, ir, exc), _make(geom, ir), fs=FS, c=C, verbose=False
    )
    rf_s, cs = sim_s.pulse_echo_rf(field_points_mm, per_scatterer=True)

    # per_scatterer now returns (P, Erx, Nt); mono-element → take channel 0 → (P, Nt).
    rf_n_img = rf_n[:, 0, :].T
    rf_s_img = rf_s[:, 0, :].T
    t_n = cn["t0"] + np.arange(rf_n_img.shape[0]) / FS
    t_s = cs["t0"] + np.arange(rf_s_img.shape[0]) / FS

    # Metrics (SDI vs Field II, on-axis).
    a = rf_s_img[:, CENTER_IDX]
    b = fii_rf[:, CENTER_IDX]
    tb_pk = _env_peak_time(b, fii_t)
    grp = (_env_peak_time(a, t_s) - tb_pk) * 1e9
    pkt = tb_pk * 1e6
    corr, phs = _windowed_corr_lag(a, t_s, b, fii_t, pkt - 1.5, pkt + 1.5)
    bw_py = _beamwidth_6db(np.abs(hilbert(rf_s_img, axis=0)), X_SCAT_MM)
    bw_fii = _beamwidth_6db(np.abs(hilbert(fii_rf, axis=0)), X_SCAT_MM)
    print("entering to save psf")
    _save_psf_figure(
        es,
        n_patches,
        corr,
        rf_n_img,
        t_n * 1e6,
        rf_s_img,
        t_s * 1e6,
        fii_rf,
        fii_t * 1e6,
    )
    rows.append((es, n_patches, corr, grp, phs, bw_py, bw_fii))
    print(
        f"{es / 1000:8.3f} {n_patches:8d} {corr:+8.4f} {grp:+8.1f} {phs:+8.1f} "
        f"{bw_py:7.3f} {bw_fii:7.3f}"
    )

# ---------------------------------------------------------------------------
# Metrics table (CSV) — durable numbers for inspection.
# ---------------------------------------------------------------------------
import csv

csv_path = FIG_DIR / "convergence_metrics.csv"
try:
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["ele_um", "n_patches", "peak_corr", "group_lag_ns", "phase_lag_ns",
             "bw6_py_mm", "bw6_fii_mm"]
        )
        for es, n_patches, corr_v, grp_v, phs_v, bwp, bwf in rows:
            w.writerow([es, n_patches, f"{corr_v:.6f}", f"{grp_v:.3f}",
                        f"{phs_v:.3f}", f"{bwp:.4f}", f"{bwf:.4f}"])
    print(f"Wrote {csv_path}")
except PermissionError:
    print(f"WARNING: {csv_path} is locked (open elsewhere?); skipped CSV write.")

# ---------------------------------------------------------------------------
# Convergence figure
# ---------------------------------------------------------------------------
es_mm = np.array([r[0] for r in rows]) / 1000.0
corr = np.array([r[2] for r in rows])
grp = np.array([r[3] for r in rows])
phs = np.array([r[4] for r in rows])
bw_py = np.array([r[5] for r in rows])
bw_fii = np.array([r[6] for r in rows])

fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
ax[0].plot(es_mm, corr, "o-")
ax[0].set_xscale("log")
ax[0].invert_xaxis()
ax[0].set_xlabel("ele_size (mm) — refine →")
ax[0].set_ylabel("on-axis peak corr")
ax[0].set_title("Correlation vs patch size")
ax[0].grid(alpha=0.3)

ax[1].plot(es_mm, grp, "o-", label="group (envelope)")
ax[1].plot(es_mm, phs, "s--", label="phase (carrier)")
ax[1].set_xscale("log")
ax[1].invert_xaxis()
ax[1].axhline(0, color="grey", lw=0.8)
ax[1].set_xlabel("ele_size (mm) — refine →")
ax[1].set_ylabel("lag (ns)")
ax[1].set_title("Lag vs patch size")
ax[1].legend(fontsize=8)
ax[1].grid(alpha=0.3)

ax[2].plot(es_mm, bw_py, "o-", label="PyField SDI")
ax[2].plot(es_mm, bw_fii, "s--", label="Field II")
ax[2].set_xscale("log")
ax[2].invert_xaxis()
ax[2].set_xlabel("ele_size (mm) — refine →")
ax[2].set_ylabel("-6 dB beamwidth (mm)")
ax[2].set_title("Lateral beamwidth vs patch size")
ax[2].legend(fontsize=8)
ax[2].grid(alpha=0.3)

fig.suptitle(
    "PyField vs Field II — patch-size convergence (matched apertures)", fontsize=13
)
fig.tight_layout()
fig.savefig(str(FIG_DIR / "convergence_metrics.png"), dpi=150)
print(f"\nSaved {len(rows)} PSF figures + convergence_metrics.png to {FIG_DIR}")
plt.close(fig)
