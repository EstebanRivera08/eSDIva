"""ReceptionSDI: pulse-echo RF via the combined PE SDI kernel.

Physics. The pulse-echo signal carries the THIRD time-derivative of the
excitation::

    v_pe(t) = (ρ₀/2c₀²) · E_m(t) ⊛_t ∂³v/∂t³

Those three derivatives can sit on either factor of the convolution:

  * On the excitation (the equation above). In a real system ∂³v/∂t³ is never
    formed explicitly — it is absorbed into the band-limited excitation e(t) and
    the TX/RX impulse responses, so in practice
    ``E_m ⊛ ∂³v/∂t³  ∝  e ⊛ h_e ⊛ h_r``. The physical derivatives live in the
    pulse + IR shapes.
  * On the SIR (what this kernel does, for speed). The combined PE-SDI kernel
    moves all three derivatives onto the two-way SIR::

        Dh_pe = dh^e *_t d²h^r = ∫ zeta_pe dt
        zeta_pe = d²h_tx/dt² *_t d²h_rx/dt²   (16 deltas per patch pair)

    and lets the excitation enter at zero derivative (v'_pe = E_m · v).

Because the practical excitation/IR chain ``e ⊛ h_e ⊛ h_r`` already embodies the
three physical derivatives, while ``Dh_pe`` baked three more onto the SIR, the two
together would differentiate six times. The public RF therefore INTEGRATES three
times (``÷(jω)³``) to strip the kernel's three derivatives, relocating the single
physical set back onto the excitation/IR chain where it belongs.

Field II parallel (for adoption): Field II uses this same practical convention —
its ``calc_scat``/``calc_hhp`` apply no explicit ∂³, the derivatives living in the
pulse and impulse responses — so the integrated result coincides with Field II.
"""

import time

import numpy as np
from scipy.fft import irfft, rfft, rfftfreq

from pyfield.hsir.transducer_sir_pe import compute_pe_sdi
from pyfield.utilities.helper_functions import compute_time_grid

from ..attenuation import causal_attenuation_tf, compute_reception_distances
from .base import (
    ReceptionBase,
    _anti_alias_decimate,
    _next_pow2,
    _warn_if_rx_delays_apods_not_default,
    _wrap_tqdm,
)


class ReceptionSDI(ReceptionBase):
    """Compute received RF signals via the PE SDI formulation.

    The pulse-echo signal physically carries the third time-derivative of the
    excitation, ``v_pe = (ρ₀/2c₀²) E_m ⊛ ∂³v/∂t³``. For efficiency the combined
    PE-SDI kernel moves those three derivatives onto the two-way SIR::

        rf = v_pe' ⊛_t Dh_pe ⊛_r f_m

        v_pe' = (ρ₀/2c₀²) × E_m × v               ← excitation at zero derivative
        Dh_pe = dh^e *_t d²h^r  (= ∫ zeta_pe dt)   ← the 3 derivatives, on the SIR

    (combined delta placement: 16 deltas per TX-RX patch pair, 1 cumsum). In a
    real system those three derivatives instead live in the band-limited
    excitation and impulse responses (``E_m ⊛ ∂³v/∂t³ ∝ e ⊛ h_e ⊛ h_r``). Since
    that practical chain already supplies them, the public RF integrates three
    times (``÷(jω)³``) to remove the redundant kernel derivatives, leaving the
    single physical set on the excitation/IR chain. Field II uses the same
    convention (no explicit ∂³ in ``calc_scat``/``calc_hhp``), so the result
    coincides with it — see the module docstring for the full argument.

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
        _warn_if_rx_delays_apods_not_default(self.rx)

    # ------------------------------------------------------------------
    # Backend-specific helpers
    # ------------------------------------------------------------------

    def _batch_P(self, nfft):
        """Batch size for scatterer loop: 400 MB budget."""
        N_freq = nfft // 2 + 1
        bytes_per_point = nfft * 4 + 2 * N_freq * 8
        return max(1, int(400 * 1024**2 // bytes_per_point))

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
        self,
        points_m,
        amps,
        *,
        n_integrations=0,
        downsampling=None,
        per_scatterer=False,
        focused_sum=False,
    ):
        """Shared computation core for pulse_echo_rf (and the mixin wrappers).

        Parameters
        ----------
        points_m : (P, 3) numpy.ndarray
            Scatterer positions in metres.
        amps : (P,) numpy.ndarray
            Scattering amplitudes (float32).
        n_integrations : int, default 0
            Number of frequency-domain integrations (÷``(jω)``) applied to the
            PE SIR before the exc/ir multiplies. The PE-SDI kernel carries the
            three physical excitation derivatives on the SIR; the practical
            excitation/IR chain (``e ⊛ h_e ⊛ h_r``) already carries them too, so
            3 removes the kernel's copies and relocates the single physical set
            onto the excitation/IR chain — leaving the bare exc⊛ir⊛ir⊛h
            convolution (Field II's convention; ``calc_scat`` ≡ ``calc_hhp``).
            Done in the frequency domain (not cumsum) so it adds no group delay
            and stays sample-aligned with conventional `Reception`. (Lower values
            leave extra derivatives on the SIR — e.g. 0 is the raw kernel output
            with both sets present.)
        downsampling : int or None, default None
            Downsample output by this factor.
        per_scatterer : bool, default False
            If True return ``(P, Nt, E_rx)`` without summing over scatterers.
            If False return ``(Nt, E_rx)`` — scatterer contributions summed.
        focused_sum : bool, default False
            If True, beamform on receive INSIDE the SIR kernel: all RX patches
            (across every element, carrying their focusing ``rx.delays`` +
            ``rx.apodization``) are passed to a single ``compute_pe_sdi`` call,
            so the kernel sums them into one focused, apodized line — Field II
            ``calc_scat``'s internal receive beamforming. Output collapses the
            RX axis to 1 (``(1, Nt)``); one FFT pair instead of ``E_rx``, no
            external DAS, and the focus is applied at corner-time resolution
            (no sample-interpolation loss). Mutually exclusive with
            ``per_scatterer`` (cannot keep both scatterers and a beamformed sum).

        Returns
        -------
        rf : (Nt, E_rx) or (P, Nt, E_rx) numpy.ndarray
        coords : dict
            Keys ``"t0"`` and ``"dt"`` (seconds).
        """
        if focused_sum and per_scatterer:
            raise ValueError("focused_sum and per_scatterer are mutually exclusive.")
        P = points_m.shape[0]
        n_rx = int(self.rx.delays.shape[0])
        # focused_sum: one group = all RX patches (kernel sums → beamformed line).
        # otherwise: one group per RX element → per-channel RF.
        if focused_sum:
            rx_groups = [
                (
                    self._rx_centers,
                    self._rx_wx,
                    self._rx_wy,
                    self._rx_apod,
                    self._rx_delays,
                    self._rx_eu,
                    self._rx_ev,
                )
            ]
        else:
            rx_groups = self._extract_rx_element_patches()
        n_out = len(rx_groups)
        show = (
            self.verbose and not focused_sum
        )  # focused_sum is the quiet loop primitive

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
            # focused_sum collapses the RX axis, so use one aperture-centroid path.
            if focused_sum:
                rx_elem_centers_m = rx_elem_centers_m.mean(axis=0, keepdims=True)
            distances_pe = compute_reception_distances(
                points_m.astype(np.float64), tx_center_m, rx_elem_centers_m
            )  # (P, n_out)

        scale = np.float32(self.rho / (2.0 * self.c**2))

        if show:
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
            (P, n_out, pe_T) if per_scatterer else (n_out, pe_T), dtype=np.float32
        )

        # pulse_echo_rf integration is done in the frequency domain
        # (÷(jω) per integration) rather than by time-domain cumsum. This is the
        # exact inverse of the analytic (jω) derivative, so it carries ZERO group
        # delay and stays sample-aligned with conventional `Reception` — whereas a
        # forward-Euler cumsum adds ½ sample of delay per integration. The f=0 bin
        # is zeroed (the exc/ir band-pass carries no DC anyway).
        inv_jw_pow = None
        if n_integrations > 0:
            jw = 1j * 2.0 * np.pi * freqs.astype(np.float64)
            inv = np.zeros_like(jw)
            nz = freqs > 0
            inv[nz] = (1.0 / jw[nz]) ** n_integrations
            inv_jw_pow = inv.astype(np.complex64)

        el_iter = (
            _wrap_tqdm(range(n_out), desc="RX elements", total=n_out, leave=True)
            if show
            else range(n_out)
        )

        for e_rx in el_iter:
            # rx_ap / rx_dl carry this group's RX apodization + delays. Per element
            # (focused_sum=False) they shift/scale that element's trace and are NOT
            # summed across elements (raw per-channel RF — see the warning). With
            # focused_sum the single group holds every RX patch, so the kernel sums
            # them into one focusing-delayed, apodized beamformed line.
            rx_c, rx_wx, rx_wy, rx_ap, rx_dl, rx_eu, rx_ev = rx_groups[e_rx]

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

            # Strip the 3 derivatives the kernel placed on Dh_pe by integrating in
            # the frequency domain; the physical set is supplied by exc ⊛ ir below.
            if inv_jw_pow is not None:
                H_pe *= inv_jw_pow[np.newaxis, :]

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
                rf[:, e_rx, :] = (rf_pe * amps[:, np.newaxis] * scale).astype(
                    np.float32
                )
            else:
                rf_channel = (rf_pe * amps[:, np.newaxis]).sum(axis=0)  # (pe_T,)
                rf[e_rx, :] = (rf_channel * scale).astype(np.float32)
            del rf_pe

        if show:
            print(f"ReceptionSDI computed in {time.time() - t_wall:.2f} s\n")

        # t0 referenced to the beam axis (subtract the TX focusing bulk) so
        # downstream beamforming needs no per-line correction. See pulse_echo_rf.
        # focused_sum also bakes the RX focusing delays into the line, so subtract
        # the RX bulk too (mirror of TX) to keep the depth mapping correct.
        t0 = pe_t0 - float(np.max(self.tx.delays))
        if focused_sum:
            t0 -= float(np.max(self.rx.delays))
        coords = {"t0": t0, "dt": dt}
        if downsampling is not None and int(downsampling) > 1:
            step = int(downsampling)
            rf = _anti_alias_decimate(rf, step)  # anti-aliased along last (time) axis
            coords["dt"] = dt * step

        return rf, coords

    def pulse_echo_rf(
        self,
        scatterer_positions_mm,
        amplitudes=None,
        *,
        per_scatterer=False,
        downsampling=None,
    ):
        """Pulse-echo RF from point scatterers.

        The core reception primitive: amplitude-weighted superposition of each
        scatterer's pulse-echo response. The three physical excitation derivatives
        are carried by the band-limited excitation and impulse responses
        (``e ⊛ h_e ⊛ h_r``), so the three the PE-SDI kernel placed on the SIR are
        removed by three frequency-domain integrations (``÷(jω)³``) — leaving the
        single physical set on the excitation/IR chain. Frequency-domain
        integration carries no group delay, so the SDI result stays sample-aligned
        with conventional `Reception`. Field II uses the same convention, so this
        equals Field II ``calc_scat`` (≡ ``calc_hhp`` for a unit point, corr 1.0000).

        ``per_scatterer=True`` keeps each scatterer separate (PSF); ``False`` sums
        them. ``coords["t0"]`` is beam-axis referenced (TX bulk ``delays.max()``
        subtracted) so downstream beamforming needs no per-line correction.

        Parameters
        ----------
        scatterer_positions_mm : (N_scat, 3) numpy.ndarray
            Scatterer positions in mm.
        amplitudes : (N_scat,) numpy.ndarray or None, default None
            Scattering coefficient at each position. None defaults to ones.
        per_scatterer : bool, default False
            If False, sum over scatterers → ``(Erx, Nt)``. If True →
            ``(N_scat, Erx, Nt)`` (PSF per point).
        downsampling : int or None, default None
            Anti-aliased time decimation factor.

        Returns
        -------
        rf : (Erx, Nt) or (N_scat, Erx, Nt) numpy.ndarray
            Pulse-echo RF per receive element (channels before time).
        coords : dict
            Keys ``"t0"`` and ``"dt"`` (seconds).
        """
        pts_mm, amps = self._validate_scatterer_inputs(
            scatterer_positions_mm, amplitudes
        )
        return self._compute_rf_inner(
            pts_mm,
            amps,
            n_integrations=3,
            downsampling=downsampling,
            per_scatterer=per_scatterer,
        )

    def _focused_sum_rf(self, points_m, amps, *, downsampling=None):
        """Receive-beamformed line via in-kernel focused sum.

        Backend hook for `ReceptionBase.scan_focusline`: the pulse-echo RF
        (``n_integrations=3`` — the 3 SIR derivatives relocated to the
        excitation/IR chain) with ``focused_sum=True``. Returns the single line
        ``(Nt,)`` and its coords. Field II's ``calc_scat`` builds its line the
        same way (focused, apodized, summed on receive).
        """
        rf, coords = self._compute_rf_inner(
            points_m,
            amps,
            n_integrations=3,
            downsampling=downsampling,
            focused_sum=True,
        )
        return rf[0], coords

    def __repr__(self) -> str:
        return (
            f"ReceptionSDI(tx={self.tx}, rx={self.rx}, c={self.c} m/s, "
            f"fs={self.fs} Hz, alpha0={self.alpha0}, "
            f"freq_power={self.freq_power})"
        )
