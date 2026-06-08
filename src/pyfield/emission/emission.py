"""Emission: compute emitted acoustic pressure fields."""

import time

import numpy as np
from scipy.fft import irfft, rfft, rfftfreq

from pyfield.hsir.farfield_rect_patch import compute_h_sir
from pyfield.utilities.helper_functions import (
    compute_sub_elem_attributes,
    compute_time_grid,
    create_3D_spatial_grid_from_points,
    reshape_to_mapped_points,
)

from ..attenuation import causal_attenuation_tf, compute_attenuation_distances
from .sir_to_pressure import (
    from_sir_to_monochromatic_pressure,
    from_sir_to_pressure,
)


def _next_pow2(n):
    return 1 << (int(n - 1).bit_length())


def _method_to_flag(method):
    if method == "naive":
        return 0
    if method in ("sdi", "SDI"):
        return 1
    return 2  # auto


def _wrap_tqdm(iterable, **kwargs):
    """Wrap with tqdm if importable, else return plain iterable."""
    try:
        from tqdm import tqdm

        return tqdm(iterable, **kwargs)
    except ImportError:
        return iterable


class Emission:
    """Compute emitted acoustic pressure fields.

    Parameters
    ----------
    transducer : TransducerBase
        Transducer with geometry, delays, and apodization.
    c : float, default 1540.0
        Speed of sound (m/s).
    rho : float, default 1.0
        Medium density (kg/m^3).
    fs : float, default 200e6
        Sampling frequency (Hz).
    alpha0 : float or None, default None
        Attenuation in dB/(MHz^y·cm). None = no attenuation.
    freq_power : float, default 1.0
        Power-law exponent y.
    excitation : numpy.ndarray or None, default None
        Excitation pulse. Shape determines dispatch:

        * ``None`` — pulsed (raw SIR / attenuated SIR if alpha0 set).
        * ``(L,)`` — global excitation convolved with h_sir.
        * ``(L, E)`` — per-element excitation; each element's SIR computed
          separately to avoid a full (P, E, T) allocation.
    transfer_function : callable or None, default None
        Global frequency-domain transfer function ``TF(freq) -> array``.
        Applied multiplicatively in frequency domain alongside the excitation
        convolution (modes 3 and 4).  ``freq`` is ``(N_freq,)`` float32 from
        ``scipy.fft.rfftfreq``.
    monochromatic : bool, default False
        If True, return CW amplitude at fc.
    fast_attenuation : bool, default False
        If True and ``alpha0`` is set, use transducer-center distance for all
        field points (fast approximation, ignores element spatial spread).
        If False (default), run the per-element loop using each element's
        center as the propagation origin (accurate near-field attenuation).
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
        "monochromatic": (bool, "CW mode flag"),
        "fast_attenuation": (bool, "Use TX-center distance for attenuation"),
        "verbose": (bool, "Print diagnostics"),
    }

    def __init__(
        self,
        transducer,
        *,
        c=1540.0,
        rho=1.0,
        fs=200e6,
        alpha0=None,
        freq_power=1.0,
        excitation=None,
        transfer_function=None,
        monochromatic=False,
        fast_attenuation=False,
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
        self.verbose = verbose
        self._refresh_sub_elem_attributes()

        lambda_m = c / self.fc
        print(
            f"Min distance must be >> w^2/(4*lambda): "
            f"{max(self.wx, self.wy) ** 2 / 4 / lambda_m * 1e3:.4f} mm"
        )

    # ------------------------------------------------------------------
    # Sub-element state management
    # ------------------------------------------------------------------

    def _refresh_sub_elem_attributes(self):
        (
            self.centers_sub_elem,
            self.apodization_sub_elem,
            self.delays_sub_elem,
            self.M,
            self.sub_elem_delta_k,
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

    # ------------------------------------------------------------------
    # Runtime parameter update
    # ------------------------------------------------------------------

    def set(self, name: str, value):
        """Update a simulation parameter at runtime.

        Parameters
        ----------
        name : str
            One of: "c", "rho", "fs", "alpha0", "freq_power", "excitation",
            "transfer_function", "monochromatic", "fast_attenuation", "verbose",
            "transducer".
        value : object
            New value for the parameter.

        Raises
        ------
        ValueError
            If name is not a recognized parameter.
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
        if name not in self._SETTABLE:
            raise ValueError(
                f"Unknown parameter '{name}'. "
                f"Valid: {['transducer', 'transfer_function'] + list(self._SETTABLE)}"
            )
        expected = self._SETTABLE[name][0]
        if not isinstance(value, expected):
            raise TypeError(f"'{name}' expects {expected}, got {type(value)}")
        if name == "excitation" and value is not None:
            value = np.asarray(value, dtype=np.float32)
        setattr(self, name, value)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_ir_to_excitation(excitation, ir):
        """Convolve excitation with impulse response (if not None).

        Returns float32 array truncated to the original excitation length.
        """
        if ir is None:
            return excitation
        conv = np.convolve(excitation.astype(np.float64), ir.astype(np.float64))
        return conv[: len(excitation)].astype(np.float32)

    def _compute_sir(self, points_m, *, method="auto", time_grid_params=None):
        """Compute h_sir summed over all patches, returns (T, P) float32.

        Parameters
        ----------
        points_m : (P, 3) float32
        method : str
        time_grid_params : tuple or None
            Pre-computed ``(time_grid, t0, dt, T)`` to avoid recomputing.

        Returns
        -------
        h : (T, P) float32
        t0 : float
        info : dict
            Keys ``"min_time"``, ``"max_time"``, ``"range_k_matrix"`` from
            ``compute_h_sir``.
        """
        method_flag = _method_to_flag(method)
        P = points_m.shape[0]

        if self.verbose:
            print(
                f"\nComputing SIR for {P} points, {self.M} patches, method={method}..."
            )

        t_wall = time.time()

        if time_grid_params is not None:
            time_grid, t0, dt, T = time_grid_params
        else:
            time_grid, t0, dt, T = compute_time_grid(
                P,
                self.M,
                points_m,
                self.centers_sub_elem,
                self.wx,
                self.wy,
                self.c,
                self.fs,
                self.delays,
                verbose=self.verbose,
            )

        h, info = compute_h_sir(
            P,
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
            method_flag,
            self.eu_arr,
            self.ev_arr,
        )

        if self.verbose:
            print(f"SIR computed in {time.time() - t_wall:.3f} s")

        return h.T, t0, info  # (T, P), float, dict

    def _extract_patch_slices(self):
        """Pre-extract per-element patch arrays (outside E-loop for efficiency)."""
        n_elements = int(self.delays.shape[0])
        slices = []
        for e in range(n_elements):
            mask = self.sub_el_idx_arr == e
            slices.append(
                (
                    self.centers_sub_elem[mask],
                    self.wx_arr[mask],
                    self.wy_arr[mask],
                    self.apodization_sub_elem[mask],
                    self.delays_sub_elem[mask],
                    self.eu_arr[mask],
                    self.ev_arr[mask],
                )
            )
        return slices

    def _compute_h_sir_batch(
        self, pts_batch, T, dt, time_grid, method_flag, patch_arrays=None
    ):
        """Compute h_sir for a batch, returns (cols, T) float32.

        Parameters
        ----------
        pts_batch : (cols, 3) float32
        T, dt, time_grid : global time grid values
        method_flag : int
        patch_arrays : tuple or None
            ``(centers, wx, wy, apod, delays)`` for a specific element.
            None uses all patches (global h_sir).
        """
        if patch_arrays is None:
            centers = self.centers_sub_elem
            wx_arr = self.wx_arr
            wy_arr = self.wy_arr
            apod_arr = self.apodization_sub_elem
            delays_arr = self.delays_sub_elem
            eu_arr = self.eu_arr
            ev_arr = self.ev_arr
        else:
            centers, wx_arr, wy_arr, apod_arr, delays_arr, eu_arr, ev_arr = patch_arrays

        cols = pts_batch.shape[0]
        M_e = centers.shape[0]

        h_out, _ = compute_h_sir(
            cols,
            M_e,
            T,
            dt,
            time_grid,
            pts_batch,
            centers,
            wx_arr,
            wy_arr,
            float(1.0 / self.c),
            self.fs,
            apod_arr,
            delays_arr,
            method_flag,
            eu_arr,
            ev_arr,
        )
        return h_out  # (cols, T) float32

    def _batch_P(self, nfft):
        """Batch size for P-loop: 400 MB budget (float32 h_pad + 2× complex64 arrays)."""
        N_freq = nfft // 2 + 1
        bytes_per_point = nfft * 4 + 2 * N_freq * 8
        return max(1, int(400 * 1024**2 // bytes_per_point))

    def _causal_tf_at_fc(self, dist_e):
        """Evaluate causal attenuation TF at fc only. Returns (P,) complex64."""
        H = causal_attenuation_tf(
            np.array([self.fc], dtype=np.float64),
            np.asarray(dist_e, dtype=np.float64),
            self.alpha0,
            self.freq_power,
            self.fc,
        )  # (P, 1) complex128
        return H[:, 0].astype(np.complex64)

    def _announce_mode(self, exc, use_per_element):
        """Print mode summary before heavy computation."""
        if not self.verbose:
            return

        if self.monochromatic:
            mode_str = "Monochromatic (CW)"
        elif exc is None:
            mode_str = "Pulsed (raw SIR)"
        elif exc.ndim == 1:
            mode_str = "Global excitation"
        else:
            mode_str = "Per-element excitation"

        if self.alpha0 is not None:
            att_label = "[per-element]" if use_per_element else "[TX-center, fast]"
            att_str = f"alpha0={self.alpha0} dB/(MHz^{self.freq_power} cm) {att_label}"
        else:
            att_str = "None"

        n_el = int(self.delays.shape[0])
        loop_str = f"Yes (E={n_el} elements)" if use_per_element else "No"

        print("\n--- Emission ---")
        print(f"  Mode       : {mode_str}")
        print(f"  Attenuation: {att_str}")
        print(f"  Per-element: {loop_str}")

    # ------------------------------------------------------------------
    # Global processing paths (no E-loop)
    # ------------------------------------------------------------------

    def _mono_global(self, points_m, distances_m, method, time_grid_params=None):
        """Monochromatic, global path: full h_sir → monochromatic pressure."""
        h, t0, info = self._compute_sir(
            points_m, method=method, time_grid_params=time_grid_params
        )
        T = h.shape[0]
        dt = 1.0 / self.fs
        idx_e = min(T, int(np.floor((info["max_time"] - t0) / dt)) + 2)
        h[idx_e:, :] = 0.0
        return from_sir_to_monochromatic_pressure(
            h,
            None,
            None,
            None,
            self.fc,
            self.fs,
            alpha0=self.alpha0,
            freq_power=self.freq_power,
            f0_hz=self.fc,
            distances_m=distances_m,
        )  # (P,) flat

    def _transient_global(
        self, points_m, t0, T, dt, time_grid, distances_m, method, exc_1d
    ):
        """Global path for pulsed and global-excitation modes.

        Parameters
        ----------
        exc_1d : (L,) float32 or None
            None = pulsed (no excitation multiply, H_att only if set).
        """
        P = points_m.shape[0]
        method_flag = _method_to_flag(method)

        if exc_1d is not None:
            L = len(exc_1d)
            nfft = _next_pow2(T + L - 1)
        else:
            nfft = _next_pow2(T)
        # float32 → complex64 throughout (half memory vs float64 → complex128).
        freqs = rfftfreq(nfft, d=1.0 / self.fs).astype(np.float32)

        # Excitation FFT: j2πf × FFT(exc) = freq-domain derivative of excitation.
        fft_exc = None
        if exc_1d is not None:
            j2pif = (2j * np.pi * freqs).astype(np.complex64)
            fft_exc = (j2pif * rfft(exc_1d, n=nfft, workers=-1)).astype(np.complex64)

        TF = None
        if self.transfer_function is not None:
            TF = np.asarray(self.transfer_function(freqs), dtype=np.complex64)

        H_att = None
        if self.alpha0 is not None and distances_m is not None:
            H_att = causal_attenuation_tf(
                freqs.astype(np.float64),
                np.asarray(distances_m, dtype=np.float64),
                self.alpha0,
                self.freq_power,
                self.fc,
            ).astype(np.complex64)  # (P, N_freq)

        batch_P = self._batch_P(nfft)
        n_batches = (P + batch_P - 1) // batch_P
        pressure_flat = np.zeros((T, P), dtype=np.float32)

        t_wall = time.time()
        if self.verbose:
            print(
                f"\nFFT processing: {P} points, nfft={nfft}, batch_P={batch_P} ({n_batches} batches)"
            )

        for p_start in range(0, P, batch_P):
            p_end = min(p_start + batch_P, P)
            pts_batch = points_m[p_start:p_end]

            h_b = self._compute_h_sir_batch(pts_batch, T, dt, time_grid, method_flag)
            # (cols, T) float32 — zero-padding handled by rfft(n=nfft), no h_pad needed.
            H = rfft(h_b, n=nfft, axis=1, workers=-1)  # (cols, N_freq) complex64
            del h_b

            if fft_exc is not None:
                H *= fft_exc[np.newaxis, :]
            if TF is not None:
                H *= TF[np.newaxis, :]
            if H_att is not None:
                H *= H_att[p_start:p_end]

            pressure_flat[:, p_start:p_end] = np.abs(
                irfft(H, n=nfft, axis=1, workers=-1)[:, :T]
            ).T.astype(np.float32)
            del H

        if self.verbose:
            print(f"FFT processing done in {time.time() - t_wall:.3f} s")

        return pressure_flat  # (T, P)

    # ------------------------------------------------------------------
    # Per-element processing paths (E-loop)
    # ------------------------------------------------------------------

    def _mono_per_element(self, points_m, T, dt, time_grid, method_flag):
        """Monochromatic, per-element: dot(h_e, exp(-j2πfc·t)) × H_att_e, accumulate.

        E-outer: one Numba call per element (full P parallelism, 128 calls total).
        P-batch inner: dot product (memory management).
        """
        P = points_m.shape[0]
        n_elements = int(self.delays.shape[0])
        patch_slices = self._extract_patch_slices()
        elem_centers = np.asarray(self.tx.element_centers, dtype=np.float64)  # (E, 3)

        t_vec = time_grid.astype(np.float64)  # (T,)
        exp_vec = np.exp(-2j * np.pi * self.fc * t_vec).astype(np.complex64)  # (T,)

        # Budget: (cols, T) float32 h_e_b + (cols,) complex64 result
        batch_P = max(1, int(400 * 1024**2 // (T * 4 + 8)))
        acc_flat = np.zeros(P, dtype=np.complex64)

        el_iter = (
            _wrap_tqdm(
                range(n_elements),
                desc="Monochromatic elements",
                total=n_elements,
                leave=True,
            )
            if self.verbose
            else range(n_elements)
        )

        for e in el_iter:
            if not self.verbose and (
                e % max(1, n_elements // 10) == 0 or e == n_elements - 1
            ):
                print(f"\r  Element {e + 1}/{n_elements}", end="", flush=True)

            # ONE Numba call for all P — maximizes parallel utilization.
            h_e = self._compute_h_sir_batch(
                points_m, T, dt, time_grid, method_flag, patch_slices[e]
            )  # (P, T) float32

            for p_start in range(0, P, batch_P):
                p_end = min(p_start + batch_P, P)
                # DFT at fc via dot product: (cols,) — slice is a view, no copy.
                H_e_fc = h_e[p_start:p_end].astype(np.complex64) @ exp_vec

                if self.alpha0 is not None:
                    dist_e_b = np.linalg.norm(
                        points_m[p_start:p_end].astype(np.float64) - elem_centers[e],
                        axis=1,
                    )
                    H_att_e_b = self._causal_tf_at_fc(dist_e_b)
                    acc_flat[p_start:p_end] += H_e_fc * H_att_e_b
                else:
                    acc_flat[p_start:p_end] += H_e_fc

            del h_e

        if not self.verbose:
            print()

        return np.abs(acc_flat).astype(np.float32)  # (P,)

    def _transient_per_element(self, points_m, t0, T, dt, time_grid, method_flag, exc):
        """Per-element path for pulsed/global/per-element excitation with attenuation.

        P-batch outer, E-element inner: freq-domain accumulation per batch, then one
        IRFFT per batch.  This limits total IRFFT calls to n_batches (not E×n_batches).
        scipy rfft with n=nfft zero-pads internally — no h_pad allocation needed.

        Parameters
        ----------
        exc : None, (L,) float32, or (L, E) float32
        """
        P = points_m.shape[0]
        n_elements = int(self.delays.shape[0])
        patch_slices = self._extract_patch_slices()
        elem_centers = np.asarray(self.tx.element_centers, dtype=np.float64)  # (E, 3)

        if exc is not None:
            L = exc.shape[0]
            nfft = _next_pow2(T + L - 1)
        else:
            nfft = _next_pow2(T)
        N_freq = nfft // 2 + 1

        freqs = rfftfreq(nfft, d=1.0 / self.fs).astype(np.float32)
        j2pif = (2j * np.pi * freqs).astype(np.complex64)

        # Pre-compute per-element excitation FFTs (outside both loops).
        fft_exc_list = None  # None = pulsed (no excitation multiply)
        if exc is not None:
            if exc.ndim == 1:
                fft_e = (j2pif * rfft(exc, n=nfft, workers=-1)).astype(np.complex64)
                fft_exc_list = [
                    fft_e
                ] * n_elements  # shared reference — no extra memory
            else:
                fft_exc_list = [
                    (j2pif * rfft(exc[:, e], n=nfft, workers=-1)).astype(np.complex64)
                    for e in range(n_elements)
                ]

        TF = None
        if self.transfer_function is not None:
            TF = np.asarray(self.transfer_function(freqs), dtype=np.complex64)

        batch_P = self._batch_P(nfft)
        n_batches = (P + batch_P - 1) // batch_P
        pressure_flat = np.zeros((T, P), dtype=np.float32)

        if self.verbose:
            print(
                f"\nPer-element processing: E={n_elements}, {P} points, "
                f"nfft={nfft}, batch_P={batch_P} ({n_batches} batches)"
            )

        batch_iter = (
            _wrap_tqdm(
                range(0, P, batch_P),
                desc="P-batches",
                total=n_batches,
                leave=True,
            )
            if self.verbose
            else range(0, P, batch_P)
        )

        # Pre-allocate one zero-padded h_pad buffer reused for every element call.
        # scipy.fft receives an already-nfft-length input → no internal zero-padding
        # buffer created per call, eliminating E×n_batches × 140 MB allocations.
        # Tail columns [:, T:] are zeroed once here and never modified.
        h_pad_buf = np.zeros((batch_P, nfft), dtype=np.float32)

        _t_batch0 = None  # wall time of first batch start (used for ETA estimate)

        for ib, p_start in enumerate(batch_iter):
            if ib == 0:
                _t_batch0 = time.time()

            p_end = min(p_start + batch_P, P)
            cols = p_end - p_start
            pts_batch = points_m[p_start:p_end]
            h_pad = h_pad_buf[:cols]  # view into pre-allocated buffer, no copy

            # Freq-domain accumulator for this batch — ONE irfft after the E loop.
            acc_H = np.zeros((cols, N_freq), dtype=np.complex64)

            for e in range(n_elements):
                h_e_b = self._compute_h_sir_batch(
                    pts_batch, T, dt, time_grid, method_flag, patch_slices[e]
                )  # (cols, T) float32
                # Write SIR into pre-allocated buffer; tail [:, T:] stays zero.
                h_pad[:, :T] = h_e_b
                del h_e_b

                H_e = rfft(h_pad, axis=1, workers=-1)  # (cols, N_freq) complex64
                if fft_exc_list is not None:
                    H_e *= fft_exc_list[e][np.newaxis, :]
                if TF is not None:
                    H_e *= TF[np.newaxis, :]
                if self.alpha0 is not None:
                    dist_e_b = np.linalg.norm(
                        pts_batch.astype(np.float64) - elem_centers[e], axis=1
                    )  # (cols,)
                    H_att_e_b = causal_attenuation_tf(
                        freqs.astype(np.float64),
                        dist_e_b,
                        self.alpha0,
                        self.freq_power,
                        self.fc,
                    ).astype(np.complex64)  # (cols, N_freq)
                    H_e *= H_att_e_b

                acc_H += H_e
                del H_e

            # ONE irfft + abs per P-batch — preserves inter-element interference.
            pressure_flat[:, p_start:p_end] = np.abs(
                irfft(acc_H, n=nfft, axis=1, workers=-1)[:, :T]
            ).T.astype(np.float32)
            del acc_H

            # After first batch: print ETA based on measured batch time.
            if ib == 0 and self.verbose and _t_batch0 is not None:
                t_first = time.time() - _t_batch0
                est_s = t_first * n_batches
                unit = "min" if est_s >= 60 else "s"
                est_val = est_s / 60 if est_s >= 60 else est_s
                print(
                    f"  First batch: {t_first:.1f}s → "
                    f"estimated total: ~{est_val:.1f} {unit} "
                    f"(FFT-bound: {n_elements}×{n_batches} batches)"
                )

        return pressure_flat  # (T, P)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __call__(self, field_points_mm, *, method="auto") -> tuple[np.ndarray, dict]:
        """Compute the pressure field at given field points.

        Behavior is determined by instance state:

        1. ``monochromatic=True`` → CW amplitude at fc.
        2. ``excitation=None`` → pulsed transient (raw / attenuated SIR).
        3. ``excitation=(L,)`` → transient with global excitation convolution.
        4. ``excitation=(L, E)`` → transient with per-element excitation.

        **FieldII equivalences** (same transducer geometry):

        * Mode 2 → ``ρ₀ · h(r, t)``.  Equivalent to FieldII ``calc_h``;
          with ``rho=1.0`` (PyField default) the arrays are numerically
          identical up to floating-point precision.
        * Mode 1 → ``|H(r, ω_c)|`` (SIR Fourier magnitude at fc).
          Equivalent to FieldII ``calc_h`` → FFT → extract the fc bin.
        * Mode 3 → ``ρ₀ · d(e ⊛ ir_tx)/dt ⊛ h(r, t)`` (pulsed acoustic
          pressure), where ``ir_tx = tx.impulse_response`` if set, else the
          bare excitation ``e`` is used directly (``dv_n/dt = de/dt``).
          Equivalent to FieldII ``calc_hp`` with ``xdc_excitation(Th, e)``
          and ``xdc_impulse(Th, ir_tx)`` set accordingly.
        * Mode 4 → same as mode 3, one loop iteration per element.

        For modes 1–3, the per-element loop is triggered when
        ``alpha0 is not None and not fast_attenuation``.  When triggered,
        each element's h_sir is computed separately and attenuation uses the
        element center as propagation origin (accurate near-field model).
        Set ``fast_attenuation=True`` to skip the loop and use the TX center
        distance (faster approximation).

        Mode 4 always uses the per-element loop; attenuation (if set)
        automatically uses element-center distances at no extra cost.

        Parameters
        ----------
        field_points_mm : dict or (N, 3) ndarray
            Grid spec dict (mm) or raw point array (mm).
        method : str, default "auto"
            SIR computation method: "auto", "naive", or "sdi".

        Returns
        -------
        pressure : ndarray
            Monochromatic: shape ``(Nx, Ny, Nz)`` / ``(N_points,)``.
            Transient: shape ``(Nt, Nx, Ny, Nz)`` / ``(Nt, N_points)``.
        coords : dict
            Keys "x", "y", "z" for structured grid; "t0", "dt" for transient.
        """
        is_structured = isinstance(field_points_mm, dict)
        if is_structured:
            x, y, z, points_m = create_3D_spatial_grid_from_points(field_points_mm)
        else:
            x, y, z = None, None, None
            pts = np.asarray(field_points_mm, dtype=np.float32)
            if pts.ndim == 1 and pts.shape[0] == 3:
                pts = pts.reshape(1, 3)
            points_m = pts * np.float32(1e-3)

        # Dispatch flags: per_elem_exc = mode 4 (excitation shape (L, E)).
        # use_per_element = E-loop needed (mode 4 always, modes 1-3 when
        # attenuation requires element-center distances).
        exc = self.excitation
        per_elem_exc = exc is not None and exc.ndim == 2
        use_per_element = (
            self.alpha0 is not None and not self.fast_attenuation
        ) or per_elem_exc

        self._announce_mode(exc, use_per_element)

        t_wall = time.time()

        # Validate per-element excitation shape
        if per_elem_exc:
            assert exc is not None
            n_elements = int(self.delays.shape[0])
            if exc.shape[1] != n_elements:
                raise ValueError(
                    f"Per-element excitation must have shape (L, E={n_elements}), "
                    f"got {exc.shape}."
                )

        method_flag = _method_to_flag(method)

        # --- Compute global time grid (used by all paths) ---
        P = points_m.shape[0]
        time_grid, t0, dt, T = compute_time_grid(
            P,
            self.M,
            points_m,
            self.centers_sub_elem,
            self.wx,
            self.wy,
            self.c,
            self.fs,
            self.delays,
            verbose=self.verbose,
        )

        # --- Attenuation distances (global TX-center path only) ---
        distances_m = None
        if self.alpha0 is not None and (not use_per_element or self.fast_attenuation):
            tx_center_m = np.asarray(self.tx.element_centers, dtype=np.float64).mean(
                axis=0
            )
            distances_m = compute_attenuation_distances(
                np.asarray(points_m, dtype=np.float64), tx_center_m
            )

        # --- Apply impulse response to excitation ---
        if exc is not None:
            ir = self.tx.impulse_response
            if ir is not None:
                if exc.ndim == 1:
                    exc = self._apply_ir_to_excitation(exc, ir)
                else:
                    exc = np.stack(
                        [
                            self._apply_ir_to_excitation(exc[:, e], ir)
                            for e in range(exc.shape[1])
                        ],
                        axis=1,
                    )

        # -------------------------------------------------------
        # MONOCHROMATIC
        # -------------------------------------------------------
        if self.monochromatic:
            if use_per_element:
                pressure_flat = self._mono_per_element(
                    points_m, T, dt, time_grid, method_flag
                )
            else:
                pressure_flat = self._mono_global(
                    points_m,
                    distances_m,
                    method,
                    time_grid_params=(time_grid, t0, dt, T),
                )

        # -------------------------------------------------------
        # PULSED / GLOBAL EXC / PER-ELEMENT EXC — TRANSIENT
        # -------------------------------------------------------
        else:
            if use_per_element:
                # Per-element loop: pulsed (exc=None), global (L,), per-element (L, E).
                pressure_flat = self._transient_per_element(
                    points_m, t0, T, dt, time_grid, method_flag, exc
                )
            elif exc is None and self.alpha0 is None:
                # Mode 2 — pure pulsed: no excitation, no attenuation.
                h, _t0, info = self._compute_sir(
                    points_m,
                    method=method,
                    time_grid_params=(time_grid, t0, dt, T),
                )
                T_h = h.shape[0]
                idx_e_h = min(T_h, int(np.floor((info["max_time"] - t0) / dt)) + 2)
                h[idx_e_h:, :] = 0.0
                pressure_flat = from_sir_to_pressure(
                    h, None, None, None, self.fs, rho=self.rho
                )
            else:
                # Mode 3 — global excitation (L,) or fast attenuation.
                pressure_flat = self._transient_global(
                    points_m, t0, T, dt, time_grid, distances_m, method, exc
                )

        # -------------------------------------------------------
        # Common exit: reshape + rho scaling + coords
        # -------------------------------------------------------
        coords: dict = {}
        if is_structured:
            pressure = reshape_to_mapped_points(x, y, z, pressure_flat) * self.rho
            if self.monochromatic:
                pressure = pressure[0]  # (1, Nx, Ny, Nz) → (Nx, Ny, Nz)
            coords["x"] = x
            coords["y"] = y
            coords["z"] = z
        else:
            pressure = pressure_flat * self.rho
        if not self.monochromatic:
            coords["t0"] = t0
            coords["dt"] = 1.0 / self.fs

        print(f"Pressure field computed in {time.time() - t_wall:.2f} s\n")
        return pressure, coords

    def __repr__(self) -> str:
        tf_str = (
            "None" if self.transfer_function is None else repr(self.transfer_function)
        )
        return (
            f"Emission(transducer={self.tx}, c={self.c} m/s, fs={self.fs} Hz, "
            f"fc={self.fc} Hz, alpha0={self.alpha0} dB/(MHz^y cm), "
            f"freq_power={self.freq_power}, monochromatic={self.monochromatic}, "
            f"fast_attenuation={self.fast_attenuation}, "
            f"transfer_function={tf_str})"
        )
