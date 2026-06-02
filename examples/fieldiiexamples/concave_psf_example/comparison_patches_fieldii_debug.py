"""Debug version — same as comparison_patches_fieldii.py but SKIPS figure generation."""

import sys
from pathlib import Path

import numpy as np
import scipy.io
from scipy.signal import correlate, hilbert

sys.path.insert(0, str(Path(__file__).parents[2]))

from pyfield.reception import Reception, ReceptionSDI
from pyfield.transducers.base import TransducerBase

# ---------------------------------------------------------------------------
FREQUENCY_HZ = 3e6
FS = 100e6
C = 1540.0
SCATTERER_Z_MM = 30.0
DB_FLOOR = -60.0
X_SCAT_MM = np.linspace(-10.0, 10.0, 101)
CENTER_IDX = len(X_SCAT_MM) // 2
ELE_SIZES_UM = [500, 250, 125]

HERE = Path(__file__).parent


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
    print(f"    [_windowed_corr_lag] Start: t_lo={t_lo}µs, t_hi={t_hi}µs", flush=True)
    
    # Convert µs → s for step calculation
    t_lo_s = t_lo * 1e-6
    t_hi_s = t_hi * 1e-6
    print(f"    [_windowed_corr_lag] In seconds: t_lo={t_lo_s}, t_hi={t_hi_s}", flush=True)
    
    print(f"    [_windowed_corr_lag] Creating grid...", flush=True)
    g = np.arange(t_lo_s, t_hi_s, 1.0 / (FS * osr))
    print(f"    [_windowed_corr_lag] Grid size: {len(g)}", flush=True)
    
    print(f"    [_windowed_corr_lag] Interpolating A...", flush=True)
    A = np.interp(g, at, a, left=0, right=0)
    print(f"    [_windowed_corr_lag] Interpolating B...", flush=True)
    B = np.interp(g, bt, b, left=0, right=0)
    
    print(f"    [_windowed_corr_lag] Normalizing A, B...", flush=True)
    A /= np.abs(A).max() + 1e-30
    B /= np.abs(B).max() + 1e-30
    
    print(f"    [_windowed_corr_lag] Computing correlate (SLOW PART)...", flush=True)
    xc = correlate(A, B, mode="full")
    print(f"    [_windowed_corr_lag] Correlate done, xc size: {len(xc)}", flush=True)
    
    print(f"    [_windowed_corr_lag] Creating lags...", flush=True)
    lags = np.arange(-len(A) + 1, len(A)) / (FS * osr)
    
    print(f"    [_windowed_corr_lag] Finding window...", flush=True)
    win = np.abs(lags) <= max_ns * 1e-9
    n_win = win.sum()
    print(f"    [_windowed_corr_lag] Window has {n_win} elements", flush=True)
    
    print(f"    [_windowed_corr_lag] Finding argmax...", flush=True)
    i = int(np.argmax(np.where(win, xc, -np.inf)))
    print(f"    [_windowed_corr_lag] argmax index: {i}", flush=True)
    
    print(f"    [_windowed_corr_lag] Computing result...", flush=True)
    den = np.sqrt((A**2).sum() * (B**2).sum()) + 1e-30
    result = xc[i] / den, lags[i] * 1e9
    print(f"    [_windowed_corr_lag] Done! result={result}", flush=True)
    return result


# ---------------------------------------------------------------------------
# Sweep (NO FIGURE GENERATION)
# ---------------------------------------------------------------------------
field_points_mm = np.column_stack(
    [X_SCAT_MM, np.zeros_like(X_SCAT_MM), np.full_like(X_SCAT_MM, SCATTERER_Z_MM)]
).astype(np.float32)

rows = []
print(
    f"{'ele[mm]':>8} {'patches':>8} {'corr':>8} {'grp[ns]':>8} {'phs[ns]':>8}"
)
for es in ELE_SIZES_UM:
    print(f"Processing es={es}um...", flush=True)
    d = scipy.io.loadmat(str(HERE / f"concave_es_{es}um.mat"), simplify_cells=True)
    geom = np.asarray(d["geom"])
    fii_rf = np.asarray(d["RF_data"])
    fii_t = float(d["start_time"]) + np.arange(fii_rf.shape[0]) / FS
    ir = np.ravel(d["impulse_response"]).astype(np.float32)
    exc = np.ravel(d["excitation"]).astype(np.float32)
    n_patches = geom.shape[1]

    print(f"  Running Reception naive...", flush=True)
    sim_n = Reception(
        _make(geom, ir, exc), _make(geom, ir), fs=FS, c=C, method="naive", verbose=False
    )
    rf_n, cn = sim_n.pulse_echo_response(field_points_mm, per_scatterer=True)
    print(f"    ✓ naive done in ~0.06s", flush=True)

    print(f"  Running ReceptionSDI...", flush=True)
    sim_s = ReceptionSDI(
        _make(geom, ir, exc), _make(geom, ir), fs=FS, c=C, verbose=False
    )
    rf_s, cs = sim_s.pulse_echo_response(field_points_mm, per_scatterer=True)
    print(f"    ✓ SDI done", flush=True)

    rf_n_img = rf_n[:, :, 0].T
    rf_s_img = rf_s[:, :, 0].T
    t_n = cn["t0"] + np.arange(rf_n_img.shape[0]) / FS
    t_s = cs["t0"] + np.arange(rf_s_img.shape[0]) / FS

    print(f"  Computing metrics...", flush=True)
    # Metrics (SDI vs Field II, on-axis).
    a = rf_s_img[:, CENTER_IDX]
    b = fii_rf[:, CENTER_IDX]
    tb_pk = _env_peak_time(b, fii_t)
    grp = (_env_peak_time(a, t_s) - tb_pk) * 1e9
    pkt = tb_pk * 1e6
    corr, phs = _windowed_corr_lag(a, t_s * 1e6, b, fii_t * 1e6, pkt - 1.5, pkt + 1.5)
    print(f"    ✓ metrics done", flush=True)

    rows.append((es, n_patches, corr, grp, phs))
    print(
        f"{es / 1000:8.3f} {n_patches:8d} {corr:+8.4f} {grp:+8.1f} {phs:+8.1f}"
    )
    print(f"  ✓ Completed es={es}um\n", flush=True)

print(f"\n✓ All {len(rows)} iterations completed successfully!")
print(rows)
