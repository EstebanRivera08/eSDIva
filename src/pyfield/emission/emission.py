"""Emission: compute emitted acoustic pressure fields."""

import time
from contextlib import contextmanager

import numpy as np
from scipy.fft import irfft, rfft, rfftfreq

from pyfield.hsir.farfield_rect_patch import compute_h_sir
from pyfield.utilities.helper_functions import (
    compute_sub_elem_attributes,
    compute_time_grid,
    create_3D_spatial_grid_from_points,
    eta_progress as _eta_progress,
    method_to_flag as _method_to_flag,
    next_pow2 as _next_pow2,
    reshape_to_mapped_points,
    wrap_tqdm as _wrap_tqdm,
)

from ..attenuation import causal_attenuation_tf, compute_attenuation_distances
from ..simulation_base import SimulationBase
from .sir_to_pressure import (
    from_sir_to_monochromatic_pressure,
    from_sir_to_pressure,
)


class Emission(SimulationBase):
    """Compute emitted acoustic pressure fields.

    Parameters
    ----------
    transducer : TransducerBase
        Transducer with geometry, delays, and apodization.
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
    fast_attenuation : bool, default True
        If True (default) and ``alpha0`` is set, use transducer-center distance
        for all field points (fast approximation, ignores element spatial spread).
        Set False to run the per-element loop using each element's center as the
        propagation origin (accurate near-field attenuation, much slower).
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
        fs=100e6,
        alpha0=None,
        freq_power=1.0,
        excitation=None,
        transfer_function=None,
        monochromatic=False,
        fast_attenuation=True,
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
        # Wall-clock time (s) per phase of the last __call__, so the cost of the
        # physics (the geometric SIR) is separated from the signal processing
        # (FFT-domain excitation convolution / attenuation / DFT-at-fc):
        #   "time_grid_s" — building the common output time axis,
        #   "hsir_s"      — the h_sir kernel, summed over every call,
        #   "fft_s"       — all FFT/iFFT convolution + monochromatic DFT work.
        # Reset at the start of each __call__.
        self.time_log: dict = {"time_grid_s": 0.0, "hsir_s": 0.0, "fft_s": 0.0}
        self._refresh_sub_elem_attributes()

        if self.verbose:
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
        self._apply_settable(name, value)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

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
            Keys ``"min_time"``, ``"max_time"`` from ``compute_h_sir``.
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

        with self._timer("hsir_s"):
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

    @staticmethod
    def _points_from_field(field_points_mm):
        """Parse field points (grid dict or raw mm array) → ``(x, y, z, points_m)``.

        ``x``/``y``/``z`` are the unique axis coordinates for a structured grid
        (None for a raw point array); ``points_m`` is the ``(P, 3)`` array in metres.
        """
        if isinstance(field_points_mm, dict):
            x, y, z, points_m = create_3D_spatial_grid_from_points(field_points_mm)
            return x, y, z, points_m
        pts = np.asarray(field_points_mm, dtype=np.float32)
        if pts.ndim == 1 and pts.shape[0] == 3:
            pts = pts.reshape(1, 3)
        return None, None, None, pts * np.float32(1e-3)

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
        _x, _y, _z, points_m = self._points_from_field(field_points_mm)
        P = points_m.shape[0]
        time_grid, _t0, dt, T = compute_time_grid(
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
        _, info = compute_h_sir(
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
            _method_to_flag(method),
            self.eu_arr,
            self.ev_arr,
            return_deltak=True,
        )
        return info["range_k_matrix"]

    def _extract_patch_slices(self):
        """Pre-extract per-element patch arrays (outside E-loop for efficiency)."""
        return self._group_patches_by_element(
            int(self.delays.shape[0]),
            self.sub_el_idx_arr,
            (
                self.centers_sub_elem,
                self.wx_arr,
                self.wy_arr,
                self.apodization_sub_elem,
                self.delays_sub_elem,
                self.eu_arr,
                self.ev_arr,
            ),
        )

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

        with self._timer("hsir_s"):
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

    def _build_freq_filters(self, T, exc_len):
        """Shared rfft length + frequency-domain filters for the transient paths.

        Both the global and per-element transient paths zero-pad the SIR (length ``T``)
        and excitation (length ``exc_len``) to a common power-of-two ``nfft`` for linear
        convolution, then work on the rfft frequency axis. This returns the pieces they
        share: ``freqs`` (Hz), ``j2pif = j2πf`` (the freq-domain ∂/∂t applied to the
        excitation), and the optional user transfer function ``TF`` sampled on ``freqs``.
        Per-path pieces (the excitation FFT itself, attenuation ``H_att``) stay in the
        callers because their shapes differ (single pulse vs per-element list).

        Parameters
        ----------
        T : int
            SIR length in samples.
        exc_len : int or None
            Excitation length in samples; None for the pulsed path (no excitation).

        Returns
        -------
        nfft : int
        freqs : (nfft//2+1,) float32
        j2pif : (nfft//2+1,) complex64
        TF : (nfft//2+1,) complex64 or None
        """
        nfft = _next_pow2(T + exc_len - 1) if exc_len else _next_pow2(T)
        # float32 → complex64 throughout (half memory vs float64 → complex128).
        freqs = rfftfreq(nfft, d=1.0 / self.fs).astype(np.float32)
        j2pif = (2j * np.pi * freqs).astype(np.complex64)
        TF = None
        if self.transfer_function is not None:
            TF = np.asarray(self.transfer_function(freqs), dtype=np.complex64)
        return nfft, freqs, j2pif, TF

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
        with self._timer("fft_s"):
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

        exc_len = len(exc_1d) if exc_1d is not None else None
        nfft, freqs, j2pif, TF = self._build_freq_filters(T, exc_len)

        # Excitation FFT: j2πf × FFT(exc) = freq-domain derivative of excitation.
        fft_exc = None
        if exc_1d is not None:
            fft_exc = (j2pif * rfft(exc_1d, n=nfft, workers=-1)).astype(np.complex64)

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

        for p_start in _eta_progress(range(0, P, batch_P), n_batches, label="batches"):
            p_end = min(p_start + batch_P, P)
            pts_batch = points_m[p_start:p_end]

            h_b = self._compute_h_sir_batch(pts_batch, T, dt, time_grid, method_flag)
            # (cols, T) float32 — zero-padding handled by rfft(n=nfft), no h_pad needed.
            with self._timer("fft_s"):
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
        # ETA + in-place progress only when the projected run exceeds ~30 s
        # (tqdm already shows progress in verbose mode).
        el_iter = _eta_progress(
            el_iter, n_elements, label="elements", progress=not self.verbose
        )

        for e in el_iter:
            # ONE Numba call for all P — maximizes parallel utilization.
            h_e = self._compute_h_sir_batch(
                points_m, T, dt, time_grid, method_flag, patch_slices[e]
            )  # (P, T) float32

            with self._timer("fft_s"):
                for p_start in range(0, P, batch_P):
                    p_end = min(p_start + batch_P, P)
                    # DFT at fc via dot product: (cols,) — slice is a view, no copy.
                    H_e_fc = h_e[p_start:p_end].astype(np.complex64) @ exp_vec

                    if self.alpha0 is not None:
                        dist_e_b = np.linalg.norm(
                            points_m[p_start:p_end].astype(np.float64)
                            - elem_centers[e],
                            axis=1,
                        )
                        H_att_e_b = self._causal_tf_at_fc(dist_e_b)
                        acc_flat[p_start:p_end] += H_e_fc * H_att_e_b
                    else:
                        acc_flat[p_start:p_end] += H_e_fc

            del h_e

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

        exc_len = exc.shape[0] if exc is not None else None
        nfft, freqs, j2pif, TF = self._build_freq_filters(T, exc_len)
        N_freq = nfft // 2 + 1

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
        # ETA + in-place progress only when the projected run exceeds ~30 s
        # (tqdm already shows progress in verbose mode).
        batch_iter = _eta_progress(
            batch_iter, n_batches, label="batches", progress=not self.verbose
        )

        # Pre-allocate one zero-padded h_pad buffer reused for every element call.
        # scipy.fft receives an already-nfft-length input → no internal zero-padding
        # buffer created per call, eliminating E×n_batches × 140 MB allocations.
        # Tail columns [:, T:] are zeroed once here and never modified.
        h_pad_buf = np.zeros((batch_P, nfft), dtype=np.float32)

        for p_start in batch_iter:
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

                with self._timer("fft_s"):
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
            with self._timer("fft_s"):
                pressure_flat[:, p_start:p_end] = np.abs(
                    irfft(acc_H, n=nfft, axis=1, workers=-1)[:, :T]
                ).T.astype(np.float32)
            del acc_H

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
            SIR computation method: "auto", "FST", or "sdi".

        Returns
        -------
        pressure : ndarray
            Monochromatic: shape ``(Nx, Ny, Nz)`` / ``(N_points,)``.
            Transient: shape ``(Nt, Nx, Ny, Nz)`` / ``(Nt, N_points)``.
        coords : dict
            Keys "x", "y", "z" for structured grid; "t0", "dt" for transient.
        """
        is_structured = isinstance(field_points_mm, dict)
        x, y, z, points_m = self._points_from_field(field_points_mm)

        # Dispatch flags: per_elem_exc = mode 4 (excitation shape (L, E)).
        # use_per_element = E-loop needed (mode 4 always, modes 1-3 when
        # attenuation requires element-center distances).
        # _resolve_excitation falls back to tx.excitation (set_excitation), so a
        # pulse set on the transducer drives emission just like the ctor arg.
        exc = self._resolve_excitation()
        per_elem_exc = exc is not None and exc.ndim == 2
        use_per_element = (
            self.alpha0 is not None and not self.fast_attenuation
        ) or per_elem_exc

        self._announce_mode(exc, use_per_element)

        # Reset the per-phase timing log for this call (see __init__).
        self.time_log = {"time_grid_s": 0.0, "hsir_s": 0.0, "fft_s": 0.0}
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
        with self._timer("time_grid_s"):
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

        total_s = time.time() - t_wall
        if self.verbose:
            tl = self.time_log
            print(
                f"Pressure field computed in {total_s:.2f} s "
                f"(time_grid {tl['time_grid_s']:.2f}s, hsir {tl['hsir_s']:.2f}s, "
                f"fft {tl['fft_s']:.2f}s)\n"
            )
        else:
            print(f"Pressure field computed in {total_s:.2f} s\n")
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
