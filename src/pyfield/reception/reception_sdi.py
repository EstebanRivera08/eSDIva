"""ReceptionSDI: pulse-echo RF via the combined PE SDI kernel.

Redistributes all 3 derivatives onto the SIR side via:

    Dh_pe = dh^e *_t d2h^r = integral(zeta_pe dt)

where zeta_pe = d2h_tx/dt2 *_t d2h_rx/dt2 (16 deltas per patch pair).
Excitation enters without any derivative: v'_pe = E_m * v.
"""

import time
import warnings

import numpy as np
from scipy.fft import irfft, rfft, rfftfreq

from pyfield.hsir.transducer_sir_pe import compute_pe_sdi
from pyfield.utilities.helper_functions import (
    compute_sub_elem_attributes,
    compute_time_grid,
)

from ..attenuation import causal_attenuation_tf, compute_reception_distances


def _next_pow2(n):
    return 1 << (int(n - 1).bit_length())


def _wrap_tqdm(iterable, **kwargs):
    """Wrap with tqdm if importable, else return plain iterable."""
    try:
        from tqdm import tqdm

        return tqdm(iterable, **kwargs)
    except ImportError:
        return iterable


class ReceptionSDI:
    """Compute received RF signals via the PE SDI formulation.

    Redistributes all 3 temporal derivatives onto the SIR side, enabling
    efficient computation via combined delta placement (16 deltas per TX-RX
    patch pair, 1 cumsum). Excitation enters without derivatives.

    Physics:

        rf = v_pe' ⊛_t Dh_pe ⊛_r f_m

        v_pe' = (ρ₀/2c₀²) × E_m × v             ← no derivatives on excitation
        Dh_pe = dh^e *_t d²h^r  (= ∫ zeta_pe dt)  ← 3 derivatives on SIR

    Only the ``"sdi"`` method is supported (SDI is intrinsic to the formulation).

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
            "verbose", "tx", "rx".
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

    def _batch_P(self, nfft):
        """Batch size for scatterer loop: 400 MB budget."""
        N_freq = nfft // 2 + 1
        bytes_per_point = nfft * 4 + 2 * N_freq * 8
        return max(1, int(400 * 1024**2 // bytes_per_point))

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

    def _compute_pe_time_grid(self, points_m):
        """Compute time grid covering both TX and RX propagation paths."""
        _, tx_t0, tx_dt, tx_T = compute_time_grid(
            points_m.shape[0],
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
        _, rx_t0, rx_dt, rx_T = compute_time_grid(
            points_m.shape[0],
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
        dt = 1.0 / self.fs
        pe_t0 = tx_t0 + rx_t0
        pe_T = tx_T + rx_T - 1
        return pe_t0, dt, pe_T

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _compute_rf_inner(
        self, points_m, amps, *, downsampling=None, per_scatterer=False
    ):
        """Shared computation core for __call__ and compute_point_rf.

        Parameters
        ----------
        points_m : (P, 3) numpy.ndarray
            Scatterer positions in metres.
        amps : (P,) numpy.ndarray
            Scattering amplitudes (float32).
        downsampling : int or None, default None
            Downsample output by this factor.
        per_scatterer : bool, default False
            If True return ``(P, Nt, E_rx)`` without summing over scatterers.
            If False return ``(Nt, E_rx)`` — scatterer contributions summed.

        Returns
        -------
        rf : (Nt, E_rx) or (P, Nt, E_rx) numpy.ndarray
        coords : dict
            Keys ``"t0"`` and ``"dt"`` (seconds).
        """
        P = points_m.shape[0]
        n_rx = int(self.rx.delays.shape[0])
        rx_slices = self._extract_rx_element_patches()

        pe_t0, dt, pe_T = self._compute_pe_time_grid(points_m)
        inv_c = np.float32(1.0 / self.c)

        exc = self._resolve_excitation()
        ir_tx = getattr(self.tx, "impulse_response", None)
        ir_rx = getattr(self.rx, "impulse_response", None)

        nfft = _next_pow2(pe_T + len(exc) - 1) if exc is not None else _next_pow2(pe_T)
        freqs = rfftfreq(nfft, d=1.0 / self.fs).astype(np.float32)

        # Pre-compute excitation and IR FFTs (no jw — derivatives in Dh_pe).
        fft_v = (
            rfft(exc, n=nfft, workers=-1).astype(np.complex64)
            if exc is not None
            else None
        )
        fft_ir_tx = (
            rfft(np.asarray(ir_tx, dtype=np.float32), n=nfft, workers=-1).astype(
                np.complex64
            )
            if ir_tx is not None
            else None
        )
        fft_ir_rx = (
            rfft(np.asarray(ir_rx, dtype=np.float32), n=nfft, workers=-1).astype(
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
            print("\n--- ReceptionSDI ---")
            print(f"  Scatterers : {P}")
            print(f"  TX patches : {self._tx_M}")
            print(f"  RX elements: {n_rx} ({self._rx_M} patches total)")
            print(f"  PE T       : {pe_T} samples")
            print(f"  nfft       : {nfft}")
            att_str = (
                f"alpha0={self.alpha0} dB/(MHz^{self.freq_power} cm)"
                if do_attenuation
                else "None"
            )
            print(f"  Attenuation: {att_str}")

        t_wall = time.time()
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

            Dh_pe = compute_pe_sdi(
                points_m,
                self._tx_centers,
                self._tx_wx,
                self._tx_wy,
                self._tx_apod,
                self._tx_delays,
                rx_c,
                rx_wx,
                rx_wy,
                rx_ap,
                rx_dl,
                inv_c,
                pe_t0,
                pe_T,
                self.fs,
                dt,
                tx_eu=self._tx_eu,
                tx_ev=self._tx_ev,
                rx_eu=rx_eu,
                rx_ev=rx_ev,
            )  # (P, pe_T) float32

            H_pe = rfft(Dh_pe, n=nfft, axis=1, workers=-1)  # (P, N_freq)
            del Dh_pe

            if fft_v is not None:
                H_pe *= fft_v[np.newaxis, :]
            if fft_ir_tx is not None:
                H_pe *= fft_ir_tx[np.newaxis, :]
            if fft_ir_rx is not None:
                H_pe *= fft_ir_rx[np.newaxis, :]
            if do_attenuation and distances_pe is not None:
                H_att = causal_attenuation_tf(
                    freqs.astype(np.float64),
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
                rf_channel = (rf_pe * amps[:, np.newaxis]).sum(axis=0)  # (pe_T,)
                rf[:, e_rx] = (rf_channel * scale).astype(np.float32)
            del rf_pe

        if self.verbose:
            print(f"ReceptionSDI computed in {time.time() - t_wall:.2f} s\n")

        if downsampling is not None and downsampling > 1:
            step = int(downsampling)
            rf = rf[:, ::step, :] if per_scatterer else rf[::step, :]

        coords = {"t0": pe_t0, "dt": dt}
        if downsampling is not None and downsampling > 1:
            coords["dt"] = dt * step

        return rf, coords

    def __call__(
        self,
        scatterer_positions_mm,
        scattering_amplitudes=None,
        *,
        downsampling=None,
    ):
        """Compute RF signal for all receive elements.

        Parameters
        ----------
        scatterer_positions_mm : (N_scat, 3) numpy.ndarray
            Scatterer positions in mm.
        scattering_amplitudes : (N_scat,) numpy.ndarray or None, default None
            Scattering coefficient at each position. None defaults to ones.
        downsampling : int or None, default None
            If set, downsample output by this factor.

        Returns
        -------
        rf : (Nt, E_rx) numpy.ndarray
            RF signal per receive element.
        coords : dict
            Keys ``"t0"`` and ``"dt"`` (seconds).
        """
        pts_mm, amps = self._validate_scatterer_inputs(
            scatterer_positions_mm, scattering_amplitudes
        )
        return self._compute_rf_inner(
            pts_mm, amps, downsampling=downsampling, per_scatterer=False
        )

    def compute_point_rf(
        self,
        positions_mm,
        amplitudes=None,
        *,
        downsampling=None,
    ):
        """Compute the pulse-echo RF response at each position independently.

        Parameters
        ----------
        positions_mm : (N_points, 3) numpy.ndarray
            Field-point positions in mm.
        amplitudes : (N_points,) numpy.ndarray or None, default None
            Per-point scaling. None defaults to ones.
        downsampling : int or None, default None
            If set, downsample time axis by this factor.

        Returns
        -------
        rf : (N_points, Nt, E_rx) numpy.ndarray
            Pulse-echo RF response at each position.
        coords : dict
            Keys ``"t0"`` and ``"dt"`` (seconds).
        """
        pts_mm, amps = self._validate_scatterer_inputs(positions_mm, amplitudes)
        return self._compute_rf_inner(
            pts_mm, amps, downsampling=downsampling, per_scatterer=True
        )

    def compute_sequence(
        self,
        scatterer_positions_mm,
        scattering_amplitudes,
        tx_events,
        *,
        downsampling=None,
    ):
        """Compute RF for multiple TX events (e.g., scan line sweep).

        Parameters
        ----------
        scatterer_positions_mm : (N_scat, 3) numpy.ndarray
            Scatterer positions in mm.
        scattering_amplitudes : (N_scat,) numpy.ndarray
            Scattering coefficient at each position.
        tx_events : list of dict
            Each dict has ``"delays"`` and/or ``"apodization"`` arrays.
        downsampling : int or None, default None
            Temporal downsampling factor.

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

    def compute_all(
        self,
        scatterer_positions_mm,
        scattering_amplitudes,
        *,
        downsampling=None,
    ):
        """Full matrix capture: each TX element transmits, all RX receive.

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

        return self.compute_sequence(
            scatterer_positions_mm,
            scattering_amplitudes,
            tx_events,
            downsampling=downsampling,
        )

    def __repr__(self) -> str:
        return (
            f"ReceptionSDI(tx={self.tx}, rx={self.rx}, c={self.c} m/s, "
            f"fs={self.fs} Hz, alpha0={self.alpha0}, "
            f"freq_power={self.freq_power})"
        )
