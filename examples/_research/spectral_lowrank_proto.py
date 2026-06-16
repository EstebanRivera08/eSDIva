"""Prototype: can the spectral one-way SIR build be made low-rank over the scatterer grid?

Why. The ``spectral`` reception core builds each scatterer's closed-form SIR spectrum
``Σ(ω) = Σ_m slope_m Σ_i σ_i e^{-jω t_i}`` independently, so its cost is ``P · M · N_band``
(scatterers × patches × in-band bins). That ``P × M`` coupling is exactly why ``spectral``
ties — and at large aperture loses to — the depth-binned ``conventional`` method, which
amortizes the SIR build across all scatterers in a depth bin. The proposed fix (idea #1):
bin scatterers by depth, build ``Σ`` at only ``r ≪ P`` anchor points per bin, and
INTERPOLATE the spectrum for every real scatterer. That would turn ``P · M · N_band`` into
``B · r · M · N_band + P · r · N_band`` — recovering the depth-bin amortization conventional
gets, while keeping the spectral form's exactness and band-limiting.

The make-or-break question, tested here. Across a depth bin spanning the full lateral
field, neighbouring scatterers' spectra differ mostly by a fast propagation phase
``e^{-jω · d_p/c}`` (d_p = distance from the aperture to the scatterer). Interpolating
``Σ`` directly fights that oscillation; interpolating the **carrier-factored** envelope
``Σ̃ = Σ · e^{+jω · d_p/c}`` (which varies slowly in space) should be far more accurate at
the same anchor count. This script measures the reconstruction correlation of both — direct
vs carrier-factored — against the exact spectrum, sweeping the anchor count and bin depth.
If carrier-factored reconstructs to corr ≈ 1 at small ``r``, idea #1 is viable; if not, it
is not worth implementing. Run with ``uv run examples/_research/spectral_lowrank_proto.py``.

This tests only the spectrum reconstruction (the crux); it does not build the full RF.
"""

import numpy as np
from scipy.interpolate import RBFInterpolator

from pyfield.hsir.transducer_sir_pe_sdi import compute_oneway_spectrum_band
from pyfield.transducers import LinearArrayTransducer
from pyfield.utilities.helper_functions import compute_sub_elem_attributes

FS, FC, C = 100e6, 5e6, 1540.0
RNG = np.random.RandomState(0)


def _tx_aperture():
    """A 128-element array's transmit patch set (the SIR-building aperture)."""
    tx = LinearArrayTransducer(
        n_elements=128,
        element_width_mm=0.25,
        element_height_mm=10.0,
        kerf_mm=0.05,
        no_sub_x=3,
        no_sub_y=6,
        frequency_Hz=FC,
    )
    centers, apod, delays, _, _, wx, wy, _ = compute_sub_elem_attributes(tx)
    return centers, wx, wy, apod, delays


def _band_omega(nfft=4096):
    """Uniform in-band angular-frequency grid over the 2-8 MHz pass-band."""
    freqs = np.fft.rfftfreq(nfft, 1.0 / FS)
    band = (freqs > 2e6) & (freqs < 8e6)
    return (2.0 * np.pi * freqs[band]).astype(np.float64)


def _cheb_anchors(x_lim, y_lim, z_lim, nx, ny, nz):
    """Tensor-product Chebyshev grid filling the bin's [x, y, z] box (metres)."""

    def cheb(a, b, n):
        if n == 1:
            return np.array([0.5 * (a + b)])
        k = np.arange(n)
        nodes = np.cos(np.pi * k / (n - 1))  # [-1, 1]
        return 0.5 * (a + b) + 0.5 * (b - a) * nodes

    gx, gy, gz = (
        cheb(*x_lim, nx),
        cheb(*y_lim, ny),
        cheb(*z_lim, nz),
    )
    grid = np.stack(np.meshgrid(gx, gy, gz, indexing="ij"), axis=-1).reshape(-1, 3)
    return grid.astype(np.float32)


def _spectrum(points_m, aperture, omega):
    centers, wx, wy, apod, delays = aperture
    return compute_oneway_spectrum_band(
        points_m, centers, wx, wy, apod, delays, 1.0 / C, 0.0, omega, 1.0 / FS
    )  # (P, N_band), referenced to t0 = 0


def _carrier(points_m, aperture, omega):
    """Bulk propagation phase e^{-jω d/c}, d = aperture-centroid → scatterer distance.

    This is the fast, spatially-coherent part of each scatterer's spectrum; factoring it out
    leaves a slowly-varying envelope that interpolates well. Shape (P, N_band).
    """
    centroid = aperture[0].mean(axis=0)  # patch-centre centroid (m)
    d = np.linalg.norm(points_m - centroid[None, :], axis=1)  # (P,)
    return np.exp(-1j * omega[None, :] * (d[:, None] / C))  # (P, N_band)


def _interp_complex(anchors_mm, values, query_mm):
    """RBF-interpolate complex (r, N) anchor values to the query points → (P, N)."""
    stacked = np.concatenate([values.real, values.imag], axis=1)  # (r, 2N)
    rbf = RBFInterpolator(anchors_mm, stacked, kernel="thin_plate_spline")
    out = rbf(query_mm)  # (P, 2N)
    n = values.shape[1]
    return out[:, :n] + 1j * out[:, n:]


def _corr(a, b):
    a, b = a.ravel(), b.ravel()
    return float(np.abs(np.vdot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b)))


def _run_bin(aperture, omega, z0_mm, dz_mm, n_axis, x_half_mm=8.0):
    """Reconstruct a cell's spectrum from a Chebyshev anchor grid.

    ``x_half_mm`` sets the lateral half-width of the cell: 8 mm = a full-FOV depth bin,
    ~1 mm = a small local cell. Returns ``(r, corr_direct, corr_carrier)``.
    """
    x_lim, y_lim = (-x_half_mm * 1e-3, x_half_mm * 1e-3), (-0.5e-3, 0.5e-3)
    z_lim = (z0_mm * 1e-3, (z0_mm + dz_mm) * 1e-3)

    # Real scatterers: random within the bin box.
    P = 300
    pts = np.empty((P, 3), dtype=np.float32)
    pts[:, 0] = RNG.uniform(*x_lim, P)
    pts[:, 1] = RNG.uniform(*y_lim, P)
    pts[:, 2] = RNG.uniform(*z_lim, P)

    # Anchors: Chebyshev tensor grid (nz = 2: a thin depth bin needs few axial nodes).
    anchors = _cheb_anchors(x_lim, y_lim, z_lim, n_axis, max(2, n_axis // 2), 2)

    sig_exact = _spectrum(pts, aperture, omega)
    sig_anch = _spectrum(anchors, aperture, omega)

    # Direct interpolation of the raw spectrum.
    rec_direct = _interp_complex(anchors[:, :3] * 1e3, sig_anch, pts[:, :3] * 1e3)

    # Carrier-factored: interpolate the slow envelope, then re-apply the bulk phase.
    env_anch = sig_anch * _carrier(anchors, aperture, omega).conj()  # remove carrier
    rec_env = _interp_complex(anchors[:, :3] * 1e3, env_anch, pts[:, :3] * 1e3)
    rec_carrier = rec_env * _carrier(pts, aperture, omega)  # re-apply

    return anchors.shape[0], _corr(sig_exact, rec_direct), _corr(sig_exact, rec_carrier)


def main():
    aperture = _tx_aperture()
    omega = _band_omega()
    print(f"TX patches: {aperture[0].shape[0]}   in-band bins: {omega.shape[0]}")
    print("\nSpectrum reconstruction corr vs exact (P=300 scatterers per cell):")

    for label, x_half in (("FULL-FOV depth bin (x: +-8 mm)", 8.0),
                          ("LOCAL cell        (x: +-1 mm)", 1.0)):
        print(f"\n  {label}")
        print(f"    {'dz [mm]':>8s} {'anchors r':>10s} {'direct':>10s} {'carrier-factored':>18s}")
        for dz in (1.0, 2.0, 4.0):
            for n_axis in (3, 5, 7):
                r, c_dir, c_car = _run_bin(aperture, omega, 29.0, dz, n_axis, x_half_mm=x_half)
                print(f"    {dz:8.1f} {r:10d} {c_dir:10.4f} {c_car:18.5f}")

    print(
        "\nRead: if the full-FOV depth bin stays low-corr but the local cell reaches "
        "corr ~1.0 at small r, the spectrum is smooth only LOCALLY — idea #1 needs 3D cells "
        "(x,y,z), not depth bins, so the cell count B explodes and the B*r << P advantage "
        "shrinks. If even the local cell fails, the spectral build is not interpolable at all."
    )


if __name__ == "__main__":
    main()
