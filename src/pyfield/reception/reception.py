"""Reception: conventional FieldII-style pulse-echo RF simulation.

Implements the standard Tupholme-Stepanishen formulation. The two-way SIR is
``h_pe = h_tx(r₁→r₅) ⊛_t h_rx(r₅→r₁)`` (a plain SIR convolution); the number of
temporal derivatives applied in the frequency domain selects the quantity:

    scattered_rf        (Field II calc_scat) : v_pe = (ρ₀/2c₀²) E_m ⊛_t ∂³v/∂t³
    pulse_echo_response (Field II calc_hhp)  : ∂/∂t(exc) ⊛ ir_tx ⊛ ir_rx ⊛ h_pe

i.e. 3 derivatives for the recorded echo, 1 for the pulse-echo response / PSF.

Supports ``"naive"``, ``"sdi"``, and ``"auto"`` methods for SIR computation.
See `ReceptionSDI` for the faster formulation where the 3 scattering derivatives
are baked onto the SIR side (scattered_rf only).
"""

import time

import numpy as np
from scipy.fft import irfft, rfft, rfftfreq

from pyfield.hsir.transducer_sir import compute_h_sir
from pyfield.utilities.helper_functions import (
    compute_sub_elem_attributes,
    compute_time_grid,
)

from ..attenuation import causal_attenuation_tf, compute_reception_distances


def _next_pow2(n):
    return 1 << (int(n - 1).bit_length())


def _method_to_flag(method):
    if method == "naive":
        return 0
    if method in ("sdi", "SDI"):
        return 1
    return None  # auto


def _wrap_tqdm(iterable, **kwargs):
    """Wrap with tqdm if importable, else return plain iterable."""
    try:
        from tqdm import tqdm

        return tqdm(iterable, **kwargs)
    except ImportError:
        return iterable


class Reception:
    """Conventional FieldII-style pulse-echo RF simulation.

    Computes received signals using the standard Tupholme-Stepanishen SIR
    formulation. The SIR kernels h_tx and h_rx are computed without
    differentiation; temporal derivatives are applied in the frequency domain,
    and their count selects the physical quantity returned:

    - `scattered_rf` (Field II ``calc_scat``) — the recorded echo, 3 derivatives:
      ``rf = (ρ₀/2c₀²) E_m ⊛_t ∂³v/∂t³ ⊛_t h_pe ⊛_r f_m``
    - `pulse_echo_response` (Field II ``calc_hhp``) — the pulse-echo response /
      PSF, 1 derivative: ``∂/∂t(exc) ⊛ ir_tx ⊛ ir_rx ⊛ h_pe``

    with ``h_pe = h_tx(r₁→r₅) ⊛_t h_rx(r₅→r₁)`` the plain two-way SIR.

    Parameters
    ----------
    tx : TransducerBase
        Transmit transducer (with delays, apodization, optional
        impulse_response and excitation).
    rx : TransducerBase
        Receive transducer (with apodization, optional impulse_response).
        Can be the same object as tx for monostatic pulse-echo.
    c : float, default 1540.0
        Speed of sound (m/s).
    rho : float, default 1.0
        Medium density (kg/m^3).
    fs : float, default 200e6
        Sampling frequency (Hz).
    alpha0 : float or None, default None
        Attenuation in dB/(MHz^y·cm). None = no attenuation.
    freq_power : float, default 1.0
        Attenuation power-law exponent.
    excitation : numpy.ndarray or None, default None
        TX excitation pulse ``(L,)``. If None, uses tx.excitation or delta.
    method : str, default "auto"
        SIR computation method: ``"naive"``, ``"sdi"``, or ``"auto"``.
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
        "method": (str, "SIR method: naive / sdi / auto"),
        "verbose": (bool, "Print diagnostics"),
    }

    def __init__(
        self,
        tx,
        rx,
        *,
        c=1540.0,
        rho=1.0,
        fs=200e6,
        alpha0=None,
        freq_power=1.0,
        excitation=None,
        method="auto",
        verbose=True,
    ):
        self.tx = tx
        self.rx = rx
        self.c = c
        self.rho = rho
        self.fs = fs
        self.alpha0 = alpha0
        self.freq_power = freq_power
        self.excitation = (
            np.asarray(excitation, dtype=np.float32) if excitation is not None else None
        )
        self.method = method
        self.verbose = verbose
        self._refresh_sub_elem_attributes()

    # ------------------------------------------------------------------
    # Sub-element state management
    # ------------------------------------------------------------------

    def _refresh_sub_elem_attributes(self):
        """Extract patch arrays from both TX and RX transducers."""
        (
            self._tx_centers,
            self._tx_apod,
            self._tx_delays,
            self._tx_M,
            _,
            self._tx_wx,
            self._tx_wy,
            self._tx_sub_el_idx,
        ) = compute_sub_elem_attributes(self.tx)
        self._tx_wx_max = float(self._tx_wx.max())
        self._tx_wy_max = float(self._tx_wy.max())
        tx_frames = self.tx.sub_patch_frames
        self._tx_eu = np.asarray(tx_frames["tangents_u"], dtype=np.float32)
        self._tx_ev = np.asarray(tx_frames["tangents_v"], dtype=np.float32)

        (
            self._rx_centers,
            self._rx_apod,
            self._rx_delays,
            self._rx_M,
            _,
            self._rx_wx,
            self._rx_wy,
            self._rx_sub_el_idx,
        ) = compute_sub_elem_attributes(self.rx)
        self._rx_wx_max = float(self._rx_wx.max())
        self._rx_wy_max = float(self._rx_wy.max())
        rx_frames = self.rx.sub_patch_frames
        self._rx_eu = np.asarray(rx_frames["tangents_u"], dtype=np.float32)
        self._rx_ev = np.asarray(rx_frames["tangents_v"], dtype=np.float32)

    # ------------------------------------------------------------------
    # Runtime parameter update
    # ------------------------------------------------------------------

    def set(self, name: str, value):
        """Update a simulation parameter at runtime.

        Parameters
        ----------
        name : str
            One of: "c", "rho", "fs", "alpha0", "freq_power", "excitation",
            "method", "verbose", "tx", "rx".
        value : object
            New value for the parameter.

        Raises
        ------
        ValueError
            If name is not a recognized parameter.
        TypeError
            If value has the wrong type.
        """
        if name in ("tx", "rx"):
            setattr(self, name, value)
            self._refresh_sub_elem_attributes()
            return
        if name not in self._SETTABLE:
            raise ValueError(
                f"Unknown parameter '{name}'. "
                f"Valid: {['tx', 'rx'] + list(self._SETTABLE)}"
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

    def _resolve_excitation(self):
        """Return effective excitation: self.excitation or tx.excitation."""
        exc = self.excitation
        if exc is None:
            tx_exc = getattr(self.tx, "excitation", None)
            if tx_exc is not None:
                exc = np.asarray(tx_exc, dtype=np.float32).ravel()
        return exc

    def _extract_rx_element_patches(self):
        """Pre-extract per-RX-element patch arrays."""
        n_rx = int(self.rx.delays.shape[0])
        slices = []
        for e in range(n_rx):
            mask = self._rx_sub_el_idx == e
            slices.append(
                (
                    self._rx_centers[mask],
                    self._rx_wx[mask],
                    self._rx_wy[mask],
                    self._rx_apod[mask],
                    self._rx_delays[mask],
                    self._rx_eu[mask],
                    self._rx_ev[mask],
                )
            )
        return slices

    def _validate_scatterer_inputs(self, positions_mm, amplitudes):
        """Normalise and validate positions + amplitudes, return (points_m, amps)."""
        pts_mm = np.asarray(positions_mm, dtype=np.float32)
        if pts_mm.ndim == 1 and pts_mm.shape[0] == 3:
            pts_mm = pts_mm.reshape(1, 3)
        points_m = pts_mm * np.float32(1e-3)
        P = points_m.shape[0]
        if amplitudes is None:
            amps = np.ones(P, dtype=np.float32)
        else:
            amps = np.asarray(amplitudes, dtype=np.float32)
            if amps.shape[0] != P:
                raise ValueError(
                    f"amplitudes length ({amps.shape[0]}) must match "
                    f"number of positions ({P})."
                )
        return points_m, amps

    def _compute_sir_time_grids(self, points_m):
        """Compute separate TX and RX time grids.

        Returns (time_grid_tx, t0_tx, dt, T_tx, time_grid_rx, t0_rx, T_rx).
        """
        P = points_m.shape[0]
        time_grid_tx, t0_tx, dt, T_tx = compute_time_grid(
            P,
            self._tx_M,
            points_m,
            self._tx_centers,
            self._tx_wx_max,
            self._tx_wy_max,
            self.c,
            self.fs,
            self.tx.delays,
            verbose=False,
        )
        time_grid_rx, t0_rx, _, T_rx = compute_time_grid(
            P,
            self._rx_M,
            points_m,
            self._rx_centers,
            self._rx_wx_max,
            self._rx_wy_max,
            self.c,
            self.fs,
            self.rx.delays,
            verbose=False,
        )
        return time_grid_tx, t0_tx, dt, T_tx, time_grid_rx, t0_rx, T_rx

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _compute_rf_inner(
        self, points_m, amps, *, n_derivatives=3, downsampling=None, per_scatterer=False
    ):
        """Shared computation core for scattered_rf and pulse_echo_response.

        Parameters
        ----------
        points_m : (P, 3) numpy.ndarray
            Scatterer positions in metres.
        amps : (P,) numpy.ndarray
            Scattering amplitudes (float32).
        n_derivatives : int, default 3
            Number of temporal derivatives applied in the frequency domain via
            ``(jω)^n``. 3 = scattered RF (Field II ``calc_scat``); 1 = pulse-echo
            response / point-spread function (Field II ``calc_hhp``).
        downsampling : int or None, default None
            Downsample output by this factor.
        per_scatterer : bool, default False
            If True return ``(P, Nt, E_rx)``. If False return ``(Nt, E_rx)``.

        Returns
        -------
        rf : (Nt, E_rx) or (P, Nt, E_rx) numpy.ndarray
        coords : dict
            Keys ``"t0"`` and ``"dt"`` (seconds).
        """
        P = points_m.shape[0]
        n_rx = int(self.rx.delays.shape[0])
        rx_slices = self._extract_rx_element_patches()
        method_flag = _method_to_flag(self.method)

        time_grid_tx, t0_tx, dt, T_tx, time_grid_rx, t0_rx, T_rx = (
            self._compute_sir_time_grids(points_m)
        )
        inv_c = np.float32(1.0 / self.c)

        # PE time axis: convolution of h_tx (T_tx) and h_rx (T_rx).
        pe_t0 = t0_tx + t0_rx
        pe_T = T_tx + T_rx - 1

        exc = self._resolve_excitation()
        ir_tx = getattr(self.tx, "impulse_response", None)
        ir_rx = getattr(self.rx, "impulse_response", None)

        L = len(exc) if exc is not None else 0
        nfft = _next_pow2(pe_T + L)
        freqs = rfftfreq(nfft, d=1.0 / self.fs)  # (N_freq,) float64 for precision

        # Temporal derivative factor in freq domain: (j*2*pi*f)^n.
        # n=3 → scattered RF (calc_scat); n=1 → pulse-echo response (calc_hhp).
        jw_pow = (1j * 2.0 * np.pi * freqs) ** n_derivatives

        # Pre-compute excitation * (jω)^n FFT.
        if exc is not None:
            fft_v = rfft(exc.astype(np.float64), n=nfft, workers=-1)
            fft_v_pe = (fft_v * jw_pow).astype(np.complex64)
        else:
            # Delta excitation: FFT(delta) = 1 → V_pe(f) = (j*2*pi*f)^n.
            fft_v_pe = jw_pow.astype(np.complex64)

        fft_ir_tx = (
            rfft(np.asarray(ir_tx, dtype=np.float64), n=nfft, workers=-1).astype(
                np.complex64
            )
            if ir_tx is not None
            else None
        )
        fft_ir_rx = (
            rfft(np.asarray(ir_rx, dtype=np.float64), n=nfft, workers=-1).astype(
                np.complex64
            )
            if ir_rx is not None
            else None
        )

        do_attenuation = self.alpha0 is not None
        distances_pe = None
        if do_attenuation:
            tx_center_m = np.asarray(self.tx.element_centers, dtype=np.float64).mean(
                axis=0
            )
            rx_elem_centers_m = np.asarray(self.rx.element_centers, dtype=np.float64)
            distances_pe = compute_reception_distances(
                points_m.astype(np.float64), tx_center_m, rx_elem_centers_m
            )  # (P, E_rx)

        scale = np.float32(self.rho / (2.0 * self.c**2))

        if self.verbose:
            mode = (
                "scattered RF (calc_scat)"
                if n_derivatives == 3
                else (
                    "pulse-echo response (calc_hhp)"
                    if n_derivatives == 1
                    else f"custom (jω)^{n_derivatives}"
                )
            )
            print("\n--- Reception (conventional) ---")
            print(f"  Quantity   : {mode}")
            print(f"  Scatterers : {P}")
            print(f"  TX patches : {self._tx_M}")
            print(f"  RX elements: {n_rx} ({self._rx_M} patches total)")
            print(f"  T_tx / T_rx: {T_tx} / {T_rx}  →  PE T = {pe_T}")
            print(f"  nfft       : {nfft}")
            print(f"  method     : {self.method}")
            att_str = (
                f"alpha0={self.alpha0} dB/(MHz^{self.freq_power} cm)"
                if do_attenuation
                else "None"
            )
            print(f"  Attenuation: {att_str}")

        t_wall = time.time()

        # Compute h_tx once for all scatterers: (P, T_tx).
        h_tx, _ = compute_h_sir(
            P,
            self._tx_M,
            T_tx,
            dt,
            time_grid_tx,
            points_m,
            self._tx_centers,
            self._tx_wx,
            self._tx_wy,
            inv_c,
            self.fs,
            self._tx_apod,
            self._tx_delays,
            method_flag,
            self._tx_eu,
            self._tx_ev,
        )  # (P, T_tx) float32

        H_tx = rfft(h_tx.astype(np.float64), n=nfft, axis=1, workers=-1)  # (P, N_freq)
        del h_tx

        rf = np.zeros(
            (P, pe_T, n_rx) if per_scatterer else (pe_T, n_rx), dtype=np.float32
        )

        el_iter = (
            _wrap_tqdm(range(n_rx), desc="RX elements", total=n_rx, leave=True)
            if self.verbose
            else range(n_rx)
        )

        for e_rx in el_iter:
            rx_c, rx_wx, rx_wy, rx_ap, rx_dl, rx_eu, rx_ev = rx_slices[e_rx]
            M_e = rx_c.shape[0]

            # Compute h_rx for this element's patches: (P, T_rx).
            h_rx_e, _ = compute_h_sir(
                P,
                M_e,
                T_rx,
                dt,
                time_grid_rx,
                points_m,
                rx_c,
                rx_wx,
                rx_wy,
                inv_c,
                self.fs,
                rx_ap,
                rx_dl,
                method_flag,
                rx_eu,
                rx_ev,
            )  # (P, T_rx) float32

            # H_pe = FFT(h_tx) * FFT(h_rx_e) — convolve SIRs in freq domain.
            H_rx_e = rfft(h_rx_e.astype(np.float64), n=nfft, axis=1, workers=-1)
            del h_rx_e
            H_pe = (H_tx * H_rx_e).astype(np.complex64)
            del H_rx_e

            # Apply 3rd derivative on excitation, IR, attenuation.
            H_pe *= fft_v_pe[np.newaxis, :]
            if fft_ir_tx is not None:
                H_pe *= fft_ir_tx[np.newaxis, :]
            if fft_ir_rx is not None:
                H_pe *= fft_ir_rx[np.newaxis, :]
            if do_attenuation and distances_pe is not None:
                H_att = causal_attenuation_tf(
                    freqs,
                    distances_pe[:, e_rx],
                    self.alpha0,
                    self.freq_power,
                    self.tx.fc,
                ).astype(np.complex64)
                H_pe *= H_att

            rf_pe = irfft(H_pe, n=nfft, axis=1, workers=-1)[:, :pe_T]  # (P, pe_T)
            del H_pe

            if per_scatterer:
                rf[:, :, e_rx] = (rf_pe * amps[:, np.newaxis] * scale).astype(
                    np.float32
                )
            else:
                rf_channel = (rf_pe * amps[:, np.newaxis]).sum(axis=0)
                rf[:, e_rx] = (rf_channel * scale).astype(np.float32)
            del rf_pe

        if self.verbose:
            print(f"Reception computed in {time.time() - t_wall:.2f} s\n")

        if downsampling is not None and downsampling > 1:
            step = int(downsampling)
            rf = rf[:, ::step, :] if per_scatterer else rf[::step, :]

        coords = {"t0": pe_t0, "dt": dt}
        if downsampling is not None and downsampling > 1:
            coords["dt"] = dt * step

        return rf, coords

    def scattered_rf(
        self,
        scatterer_positions_mm,
        scattering_amplitudes=None,
        *,
        per_scatterer=False,
        downsampling=None,
    ):
        """Scattered RF echo from point scatterers (Field II ``calc_scat``).

        Full pulse-echo scattered signal with the three temporal derivatives of
        the Born scattering model: ``v_pe = (ρ₀/2c₀²) E_m ⊛ ∂³v/∂t³``. This is
        the signal a transducer would record.

        Parameters
        ----------
        scatterer_positions_mm : (N_scat, 3) numpy.ndarray
            Scatterer positions in mm.
        scattering_amplitudes : (N_scat,) numpy.ndarray or None, default None
            Scattering coefficient at each position. None defaults to ones.
        per_scatterer : bool, default False
            If False, sum scatterer contributions per RX channel and return
            ``(Nt, E_rx)``. If True, keep each scatterer separate and return
            ``(N_scat, Nt, E_rx)``.
        downsampling : int or None, default None
            If set, downsample the time axis by this factor.

        Returns
        -------
        rf : (Nt, E_rx) or (N_scat, Nt, E_rx) numpy.ndarray
            Scattered RF per receive element.
        coords : dict
            Keys ``"t0"`` and ``"dt"`` (seconds).
        """
        pts_mm, amps = self._validate_scatterer_inputs(
            scatterer_positions_mm, scattering_amplitudes
        )
        return self._compute_rf_inner(
            pts_mm,
            amps,
            n_derivatives=3,
            downsampling=downsampling,
            per_scatterer=per_scatterer,
        )

    # ``__call__`` is the ergonomic alias for the most common operation.
    __call__ = scattered_rf

    def pulse_echo_response(
        self,
        points_mm,
        amplitudes=None,
        *,
        per_scatterer=False,
        downsampling=None,
    ):
        """Pulse-echo response / point-spread function (Field II ``calc_hhp``).

        The two-way spatial impulse response convolved with excitation and both
        impulse responses, with a single emission derivative::

            pe = ∂/∂t(exc) ⊛ ir_tx ⊛ ir_rx ⊛ (h_tx ⊛ h_rx)

        Unlike `scattered_rf` (3 derivatives, the recorded echo), this is the
        system's pulse-echo response itself — the quantity Field II ``calc_hhp``
        returns. Useful for inspecting the PSF at chosen field points.

        Parameters
        ----------
        points_mm : (N_points, 3) numpy.ndarray
            Field-point positions in mm.
        amplitudes : (N_points,) numpy.ndarray or None, default None
            Per-point scaling. None defaults to ones.
        per_scatterer : bool, default False
            If False return ``(Nt, E_rx)`` (summed); if True return
            ``(N_points, Nt, E_rx)`` — one PSF trace per point.
        downsampling : int or None, default None
            If set, downsample the time axis by this factor.

        Returns
        -------
        rf : (Nt, E_rx) or (N_points, Nt, E_rx) numpy.ndarray
        coords : dict
            Keys ``"t0"`` and ``"dt"`` (seconds).
        """
        pts_mm, amps = self._validate_scatterer_inputs(points_mm, amplitudes)
        return self._compute_rf_inner(
            pts_mm,
            amps,
            n_derivatives=1,
            downsampling=downsampling,
            per_scatterer=per_scatterer,
        )

    def rf_sequence(
        self,
        scatterer_positions_mm,
        scattering_amplitudes,
        tx_events,
        *,
        downsampling=None,
    ):
        """Scattered RF for a sequence of TX events (e.g. a scan-line sweep).

        Parameters
        ----------
        scatterer_positions_mm : (N_scat, 3) numpy.ndarray
        scattering_amplitudes : (N_scat,) numpy.ndarray
        tx_events : list of dict
            Each dict has ``"delays"`` and/or ``"apodization"`` arrays.
        downsampling : int or None, default None

        Returns
        -------
        rf : (N_events, Nt, E_rx) numpy.ndarray
        coords : dict
        """
        orig_delays = self.tx.delays.copy()
        orig_apod = self.tx.apodization.copy()

        results = []
        coords_out = None
        try:
            for i, event in enumerate(tx_events):
                if "delays" in event:
                    self.tx.delays = np.asarray(event["delays"], dtype=np.float32)
                if "apodization" in event:
                    self.tx.apodization = np.asarray(
                        event["apodization"], dtype=np.float32
                    )
                self._refresh_sub_elem_attributes()

                if self.verbose:
                    print(f"\n=== TX event {i + 1}/{len(tx_events)} ===")

                rf_i, coords_i = self(
                    scatterer_positions_mm,
                    scattering_amplitudes,
                    downsampling=downsampling,
                )
                results.append(rf_i)
                if coords_out is None:
                    coords_out = coords_i
        finally:
            self.tx.delays = orig_delays
            self.tx.apodization = orig_apod
            self._refresh_sub_elem_attributes()

        max_Nt = max(r.shape[0] for r in results)
        n_rx = results[0].shape[1]
        rf_all = np.zeros((len(results), max_Nt, n_rx), dtype=np.float32)
        for i, r in enumerate(results):
            rf_all[i, : r.shape[0], :] = r

        return rf_all, coords_out

    def rf_matrix(
        self,
        scatterer_positions_mm,
        scattering_amplitudes,
        *,
        downsampling=None,
    ):
        """Full-matrix capture (FMC): each TX element fires alone, all RX receive.

        Parameters
        ----------
        scatterer_positions_mm : (N_scat, 3) numpy.ndarray
        scattering_amplitudes : (N_scat,) numpy.ndarray
        downsampling : int or None, default None

        Returns
        -------
        rf : (E_tx, Nt, E_rx) numpy.ndarray
        coords : dict
        """
        n_tx = int(self.tx.delays.shape[0])
        tx_events = []
        for e_tx in range(n_tx):
            apod = np.zeros(n_tx, dtype=np.float32)
            apod[e_tx] = 1.0
            delays = np.zeros(n_tx, dtype=np.float32)
            tx_events.append({"delays": delays, "apodization": apod})

        return self.rf_sequence(
            scatterer_positions_mm,
            scattering_amplitudes,
            tx_events,
            downsampling=downsampling,
        )

    def __repr__(self) -> str:
        return (
            f"Reception(tx={self.tx}, rx={self.rx}, c={self.c} m/s, "
            f"fs={self.fs} Hz, alpha0={self.alpha0}, method='{self.method}')"
        )
