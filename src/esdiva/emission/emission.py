"""Emission: the pressure field radiated by a transducer.

Emitted pressure at a field point r (Tupholme–Stepanishen):

    p(r, t) = ρ₀ · ∂v/∂t ⊛ h(r, t)   ⇔   P(r, ω) = ρ₀ · jω·V(ω) · H(r, ω)

with v the surface-velocity pulse (excitation ⊛ impulse response) and h the spatial
impulse response (SIR) of the aperture. Everything is assembled in the frequency domain,
where the transfer functions simply multiply: attenuation ``H_att(ω, d)``, a user
``transfer_function``, and per-element drives. Only the SIR source differs by ``method``:

* ``"spectral"`` — H(ω) in closed form (`compute_h_sir_spectrum`), exact and
  evaluated only on the frequencies the pulse occupies (one bin in monochromatic mode).
* ``"temporal"`` (= ``"sdi"``), ``"fst"``, ``"auto"`` — sample h(t) (`compute_h_sir`),
  then Fourier-transform it; the string names the trapezoid-sampling kernel.

``rfft(h[n]) ≈ fs·H(ω)`` links the two, so both give the same field; spectral is exact
(no sub-sample clamp). ``method=None`` (default) picks the faster one, measured on a
64-element array, 12.5k points: spectral for monochromatic (2×; 23× with per-element
attenuation) and for per-element drives or attenuation (2.3×, one FFT for all elements
instead of one per element); temporal otherwise (1.3–2.5×, including with attenuation,
a transfer function or a soft baffle — each is one multiply per bin either way).
"""

import time
from contextlib import contextmanager

import numpy as np
from scipy.fft import irfft, rfft, rfftfreq

from esdiva.hsir.sir_spectral import compute_h_sir_spectrum
from esdiva.hsir.sir_temporal import compute_h_sir
from esdiva.utilities.helper_functions import (
    compute_sub_elem_attributes,
    compute_time_grid,
    create_3D_spatial_grid_from_points,
    method_to_flag as _method_to_flag,
    next_pow2 as _next_pow2,
    reshape_to_mapped_points,
)

from ..attenuation import causal_attenuation_tf
from ..simulation_base import SimulationBase

_METHODS = ("spectral", "temporal", "sdi", "fst", "FST", "auto")

_POINTS_PER_BIN = 1500
"""Field points per depth bin: each bin gets a time window just long enough for its own
arrivals, so its FFT (and, for ``spectral``, its in-band bin count) stays short. Measured
on a 50k-point linear-array grid: ~1500 points per bin was the fastest setting."""


class Emission(SimulationBase):
    """Compute emitted acoustic pressure fields (monochromatic or transient).

    Parameters
    ----------
    transducer : TransducerBase
        Transducer with geometry, delays, apodization and optional impulse response.
    c : float, default 1540.0
        Speed of sound (m/s).
    rho : float, default 1.0
        Medium density (kg/m^3).
    fs : float, default 100e6
        Sampling frequency (Hz).
    alpha0 : float or None, default None
        Attenuation in dB/(MHz^y·cm). None = no attenuation.
    freq_power : float, default 1.0
        Power-law exponent y.
    excitation : numpy.ndarray or None, default None
        Drive pulse: ``None`` (the output is the SIR itself), ``(L,)`` global, or
        ``(L, E)`` one pulse per element. Convolved with ``transducer.impulse_response``.
    transfer_function : callable or None, default None
        Extra frequency response ``TF(freq_hz) -> array``, multiplied in the frequency
        domain (applied at fc in monochromatic mode).
    monochromatic : bool, default False
        True → pressure amplitude ``|P(r, fc)|`` at the centre frequency (no pulse).
    fast_attenuation : bool, default True
        Attenuation path origin: True → transducer centre (one path per point);
        False → each element's centre (per-element paths, accurate near field).
    method : str or None, default None
        SIR source: ``"spectral"``, ``"temporal"`` (= ``"sdi"``), ``"fst"``, ``"auto"``;
        None picks the faster for the mode (see the module docstring).
    verbose : bool, default True
        Print diagnostic information during simulation.
    """

    _SETTABLE: dict = {
        "c": (float, "Speed of sound (m/s)"),
        "rho": (float, "Density (kg/m^3)"),
        "fs": (float, "Sampling frequency (Hz)"),
        "alpha0": ((float, type(None)), "Attenuation dB/(MHz^y cm) or None"),
        "freq_power": (float, "Attenuation exponent"),
        "excitation": ((np.ndarray, type(None)), "Excitation pulse or None"),
        "monochromatic": (bool, "Monochromatic (single-frequency) mode"),
        "fast_attenuation": (bool, "Attenuation from the transducer centre"),
        "method": (
            (str, type(None)),
            "SIR source: spectral/temporal/sdi/fst/auto/None",
        ),
        "verbose": (bool, "Print diagnostics"),
    }

    def __init__(
        self,
        transducer,
        *,
        c=1540.0,
        rho=1.0,
        fs=100e6,
        alpha0=None,
        freq_power=1.0,
        excitation=None,
        transfer_function=None,
        monochromatic=False,
        fast_attenuation=True,
        method=None,
        verbose=True,
    ):
        self.tx = transducer
        self.fc = transducer.fc
        self.c = c
        self.rho = rho
        self.fs = fs
        self.alpha0 = alpha0
        self.freq_power = freq_power
        self.excitation = (
            np.asarray(excitation, dtype=np.float32) if excitation is not None else None
        )
        if transfer_function is not None and not callable(transfer_function):
            raise TypeError(
                f"'transfer_function' must be callable or None, got {type(transfer_function)}"
            )
        self.transfer_function = transfer_function
        self.monochromatic = monochromatic
        self.fast_attenuation = fast_attenuation
        self.method = self._validate_method(method)
        self.verbose = verbose
        # Wall-clock per phase of the last call: time grids, SIR kernel, FFT work.
        self.time_log: dict = {"time_grid_s": 0.0, "hsir_s": 0.0, "fft_s": 0.0}
        self._refresh_sub_elem_attributes()

        if self.verbose:
            lambda_m = c / self.fc
            print(
                f"Min distance must be >> w^2/(4*lambda): "
                f"{max(self.wx, self.wy) ** 2 / 4 / lambda_m * 1e3:.4f} mm"
            )

    @staticmethod
    def _validate_method(method):
        if method is not None and method not in _METHODS:
            raise ValueError(f"Unknown method {method!r}. Valid: {list(_METHODS)}")
        return method

    def _refresh_sub_elem_attributes(self):
        (
            self.centers_sub_elem,
            self.apodization_sub_elem,
            self.delays_sub_elem,
            self.M,
            self.wx_arr,
            self.wy_arr,
            self.sub_el_idx_arr,
        ) = compute_sub_elem_attributes(self.tx)
        self.wx = float(self.wx_arr.max())
        self.wy = float(self.wy_arr.max())
        self.delays = self.tx.delays
        self.apodization = self.tx.apodization
        frames = self.tx.sub_patch_frames
        self.eu_arr = np.asarray(frames["tangents_u"], dtype=np.float32)
        self.ev_arr = np.asarray(frames["tangents_v"], dtype=np.float32)

    def set(self, name: str, value):
        """Update a simulation parameter at runtime.

        Parameters
        ----------
        name : str
            A key of ``_SETTABLE``, ``"transfer_function"`` or ``"transducer"``.
        value : object
            New value for the parameter.

        Raises
        ------
        ValueError
            If name is not a recognized parameter, or an unknown method.
        TypeError
            If value has the wrong type.
        """
        if name == "transducer":
            self.tx = value
            self.fc = value.fc
            self._refresh_sub_elem_attributes()
            return
        if name == "transfer_function":
            if value is not None and not callable(value):
                raise TypeError(
                    f"'transfer_function' must be callable or None, got {type(value)}"
                )
            self.transfer_function = value
            return
        if name == "method":
            self._validate_method(value)
        self._apply_settable(name, value)

    @contextmanager
    def _timer(self, key):
        """Add the wall-clock time of the enclosed block to ``self.time_log[key]``."""
        t = time.perf_counter()
        try:
            yield
        finally:
            self.time_log[key] = self.time_log.get(key, 0.0) + (time.perf_counter() - t)

    @staticmethod
    def _apply_ir_to_excitation(excitation, ir):
        """Transmitted pulse ``exc ⊛ ir`` at full length ``L + L_ir − 1`` (float32).

        Truncating to ``len(exc)`` would drop the pulse tail (over half its energy for a
        2-cycle excitation and a 2-cycle impulse response).
        """
        if ir is None:
            return excitation
        conv = np.convolve(excitation.astype(np.float64), ir.astype(np.float64))
        return conv.astype(np.float32)

    @staticmethod
    def _points_from_field(field_points_mm):
        """Parse field points (grid dict or raw mm array) → ``(x, y, z, points_m)``."""
        if isinstance(field_points_mm, dict):
            return create_3D_spatial_grid_from_points(field_points_mm)
        pts = np.asarray(field_points_mm, dtype=np.float32)
        if pts.ndim == 1 and pts.shape[0] == 3:
            pts = pts.reshape(1, 3)
        return None, None, None, pts * np.float32(1e-3)

    def _time_grid(self, points_m):
        """SIR time window ``(time_grid, t0, dt, T)`` covering every point."""
        with self._timer("time_grid_s"):
            return compute_time_grid(
                points_m.shape[0],
                self.M,
                points_m,
                self.centers_sub_elem,
                self.wx,
                self.wy,
                self.c,
                self.fs,
                self.delays,
                verbose=False,
            )

    def compute_deltak(self, field_points_mm, *, method="auto"):
        """Per-patch trapezoid width Δk (in samples) for every field point.

        SIR-accuracy diagnostic: Δk is how many time samples each patch's
        trapezoidal SIR spans at this geometry. The far-field/sampling
        approximation degrades when Δk is too small (a patch barely resolved in
        time), and the auto method switches FST→SDI above ``8 + 2T/M``. Inspect
        this to check a chosen ``no_sub_x``/``no_sub_y`` resolves every patch.

        Parameters
        ----------
        field_points_mm : dict or (N, 3) numpy.ndarray
            Grid spec dict (mm) or raw point array (mm), as in ``__call__``.
        method : str, default "auto"
            SIR method ("auto", "FST", "sdi") — only affects which patches the
            kernel would take the SDI path for; Δk itself is method-independent.

        Returns
        -------
        (P, M) numpy.ndarray
            Trapezoid width in samples for each field point P and patch M.
        """
        *_, points_m = self._points_from_field(field_points_mm)
        time_grid, _t0, dt, T = self._time_grid(points_m)
        _, info = compute_h_sir(
            points_m.shape[0],
            self.M,
            T,
            dt,
            time_grid,
            points_m,
            self.centers_sub_elem,
            self.wx_arr,
            self.wy_arr,
            float(1.0 / self.c),
            self.fs,
            self.apodization_sub_elem,
            self.delays_sub_elem,
            _method_to_flag(method),
            self.eu_arr,
            self.ev_arr,
            return_deltak=True,
        )
        return info["range_k_matrix"]

    # ------------------------------------------------------------------
    # Building blocks: radiating groups, SIR spectrum, attenuation, bins
    # ------------------------------------------------------------------

    def _groups(self, per_element):
        """Radiating groups ``(patch arrays, path origin)``.

        One group = the whole aperture with the transducer centre as attenuation origin;
        per element = one group per element, each with its own centre (needed for
        per-element drives or per-element attenuation paths).
        """
        arrays = (
            self.centers_sub_elem,
            self.wx_arr,
            self.wy_arr,
            self.apodization_sub_elem,
            self.delays_sub_elem,
            self.eu_arr,
            self.ev_arr,
        )
        centres = np.asarray(self.tx.element_centers, dtype=np.float64)
        if not per_element:
            return [(arrays, centres.mean(axis=0))]
        per = self._group_patches_by_element(len(centres), self.sub_el_idx_arr, arrays)
        return list(zip(per, centres))

    def _sir_spectrum(self, pts, patches, t0, T, omega, method, nfft=0, b0=0):
        """``fs·H(ω)`` of one patch group at ``pts``, phase-referenced to ``t0`` → (P, N_ω).

        Spectral: the closed form. Temporal: sample h(t) on ``t0 + n/fs`` (``T`` samples)
        and take its DFT — ``rfft`` over ``nfft`` sliced at ``b0`` for a band, or a
        direct sum at the single frequency of monochromatic mode.
        """
        c, wx, wy, ap, dl, eu, ev = patches
        soft = self.tx.baffle == "soft"
        with self._timer("hsir_s"):
            if method == "spectral":
                return self.fs * compute_h_sir_spectrum(
                    pts, c, wx, wy, ap, dl, 1.0 / self.c, t0, omega,
                    eu=eu, ev=ev, soft_baffle=soft,
                )  # fmt: skip
            grid = (t0 + np.arange(T) / self.fs).astype(np.float32)
            h, info = compute_h_sir(
                pts.shape[0], c.shape[0], T, 1.0 / self.fs, grid, pts, c, wx, wy,
                1.0 / self.c, self.fs, ap, dl, _method_to_flag(method), eu, ev,
                soft_baffle=soft,
            )  # fmt: skip
        # Zero the SDI double-cumsum drift after the last arrival.
        h[:, max(0, min(T, int((info["max_time"] - t0) * self.fs) + 2)) :] = 0.0
        with self._timer("fft_s"):
            if omega.size == 1:
                return (h @ np.exp(-1j * omega[0] * np.arange(T) / self.fs))[:, None]
            return rfft(h, n=nfft, axis=1, workers=-1)[:, b0 : b0 + omega.size]

    def _atten(self, freqs, pts, origin):
        """Causal attenuation ``H_att(f, |r − origin|)`` → (P, N_f), or 1 if disabled."""
        if self.alpha0 is None:
            return 1.0
        d = np.linalg.norm(pts.astype(np.float64) - origin, axis=1)
        return causal_attenuation_tf(freqs, d, self.alpha0, self.freq_power, self.fc)

    def _tf(self, freqs):
        """User transfer function on ``freqs`` (1 if unset)."""
        if self.transfer_function is None:
            return 1.0
        return np.asarray(
            self.transfer_function(freqs.astype(np.float32)), np.complex128
        )

    def _bins(self, pts):
        """Field-point groups sorted by distance to the aperture (see `_POINTS_PER_BIN`)."""
        centre = np.asarray(self.tx.element_centers, dtype=np.float64).mean(axis=0)
        order = np.argsort(np.linalg.norm(pts - centre, axis=1))
        return np.array_split(order, max(1, pts.shape[0] // _POINTS_PER_BIN))

    # ------------------------------------------------------------------
    # The two modes
    # ------------------------------------------------------------------

    def _monochromatic(self, pts, groups, method):
        """``|Σ_g fs·H_g(ωc)·H_att,g(fc)| · |TF(fc)|`` → (P,)."""
        fc = np.array([self.fc])
        omega = 2.0 * np.pi * fc
        acc = np.zeros(pts.shape[0], dtype=np.complex128)
        for idx in self._bins(pts):
            p = pts[idx]
            _, t0, _, T = self._time_grid(p) if method != "spectral" else (0, 0.0, 0, 0)
            for patches, origin in groups:
                H = self._sir_spectrum(p, patches, t0, T, omega, method)
                acc[idx] += (H * self._atten(fc, p, origin))[:, 0]
        return (np.abs(acc * self._tf(fc))).astype(np.float32)

    def _transient(self, pts, groups, drives, method):
        """Signed ``irfft(Σ_g fs·H_g · D_g · TF · H_att,g)`` on one shared time axis → (T, P).

        ``D_g = jω·DFT(pulse_g)`` is the drive of group g (1 for the raw SIR). Points are
        processed in depth bins, each on a short window snapped onto the global sample
        lattice, so its samples drop into place with no resampling.
        """
        _, t0, dt, T = self._time_grid(pts)
        L = max((len(d) for d in drives if d is not None), default=1)
        out = np.zeros((pts.shape[0], T), dtype=np.float32)  # rows: contiguous writes
        for idx in self._bins(pts):
            p = pts[idx]
            _, t0_nat, _, T_nat = self._time_grid(p)
            n0, t0_b, _ = self._snap_to_lattice(t0_nat, t0, dt)
            T_b = min(T_nat + 1, T - n0)
            nfft = _next_pow2(T_b + L - 1)
            f = rfftfreq(nfft, dt)
            jw = 2j * np.pi * f
            D = [np.ones_like(jw) if d is None else jw * rfft(d, nfft) for d in drives]
            b0, b1 = 0, f.size
            if method == "spectral":
                b0, b1 = _band(np.abs(D).max(axis=0) * np.abs(self._tf(f) + 0 * f))
            X = np.zeros((p.shape[0], f.size), dtype=np.complex128)
            fb = f[b0:b1]
            for g, (patches, origin) in enumerate(groups):
                H = self._sir_spectrum(
                    p, patches, t0_b, T_b, 2 * np.pi * fb, method, nfft, b0
                )
                X[:, b0:b1] += H * D[g % len(D)][b0:b1] * self._atten(fb, p, origin)
            with self._timer("fft_s"):
                X[:, b0:b1] *= self._tf(fb)
                out[idx, n0 : n0 + T_b] = irfft(X, nfft, axis=1)[:, :T_b]
        return out.T, t0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __call__(self, field_points_mm, *, method=None) -> tuple[np.ndarray, dict]:
        """Compute the pressure field at given field points.

        Modes (instance state): ``monochromatic=True`` → ``|P(r, fc)|``; otherwise the
        transient field ``ρ₀·∂v/∂t ⊛ h`` for the global ``(L,)`` or per-element ``(L, E)``
        excitation, or the SIR itself when ``excitation`` is None. Field II
        equivalents: ``calc_h`` (no excitation), ``calc_hp`` (excitation + impulse).

        Parameters
        ----------
        field_points_mm : dict or (N, 3) ndarray
            Grid spec dict (mm) or raw point array (mm).
        method : str or None, default None
            Override ``self.method`` for this call.

        Returns
        -------
        pressure : ndarray
            Monochromatic: amplitude ``(Nx, Ny, Nz)`` / ``(N,)``. Transient: signed
            pressure ``(Nt, Nx, Ny, Nz)`` / ``(Nt, N)`` (compression > 0, rarefaction < 0;
            take ``abs`` or an envelope for field maps).
        coords : dict
            ``"x"``, ``"y"``, ``"z"`` for a grid; ``"t0"``, ``"dt"`` when transient.

        Raises
        ------
        ValueError
            If a per-element excitation does not have one column per element.
        """
        method = self._validate_method(method or self.method)
        x, y, z, pts = self._points_from_field(field_points_mm)
        exc = self._resolve_excitation()
        n_el = int(self.delays.shape[0])
        per_elem_exc = exc is not None and exc.ndim == 2
        if per_elem_exc and exc.shape[1] != n_el:
            raise ValueError(
                f"Per-element excitation must have shape (L, E={n_el}), got {exc.shape}."
            )
        per_element = (self.alpha0 is not None and not self.fast_attenuation) or (
            per_elem_exc and not self.monochromatic
        )
        groups = self._groups(per_element)
        if method is None:  # the faster SIR source for this mode (module docstring)
            method = "spectral" if self.monochromatic or per_element else "temporal"
        self._last_method = method
        self.time_log = {"time_grid_s": 0.0, "hsir_s": 0.0, "fft_s": 0.0}
        t_wall = time.time()

        coords: dict = {}
        if self.monochromatic:
            flat = self._monochromatic(pts, groups, method)[np.newaxis, :]
        else:
            ir = self.tx.impulse_response
            if exc is None:
                drives = [None]
            elif per_elem_exc:
                drives = [
                    self._apply_ir_to_excitation(exc[:, e], ir) for e in range(n_el)
                ]
            else:
                drives = [self._apply_ir_to_excitation(exc, ir)]
            flat, coords["t0"] = self._transient(pts, groups, drives, method)
            coords["dt"] = 1.0 / self.fs

        if x is not None:
            pressure = reshape_to_mapped_points(x, y, z, flat) * self.rho
            if self.monochromatic:
                pressure = pressure[0]
            coords.update(x=x, y=y, z=z)
        else:
            pressure = (flat[0] if self.monochromatic else flat) * self.rho
        if self.verbose:
            tl = self.time_log
            print(
                f"Emission [{'monochromatic' if self.monochromatic else 'transient'}, "
                f"{method}] {pts.shape[0]} points in {time.time() - t_wall:.2f} s "
                f"(time_grid {tl['time_grid_s']:.2f}s, hsir {tl['hsir_s']:.2f}s, "
                f"fft {tl['fft_s']:.2f}s)"
            )
        return pressure, coords

    def __repr__(self) -> str:
        return (
            f"Emission(transducer={self.tx}, c={self.c} m/s, fs={self.fs} Hz, "
            f"fc={self.fc} Hz, alpha0={self.alpha0} dB/(MHz^y cm), "
            f"freq_power={self.freq_power}, monochromatic={self.monochromatic}, "
            f"fast_attenuation={self.fast_attenuation}, method='{self.method}')"
        )


def _band(mag, tol=1e-4):
    """Contiguous bin span ``[b0, b1)`` where ``mag`` exceeds ``tol`` of its peak."""
    sig = np.nonzero(mag >= tol * mag.max())[0] if mag.max() > 0 else []
    return (int(sig[0]), int(sig[-1]) + 1) if len(sig) else (0, mag.size)
