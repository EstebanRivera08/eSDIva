"""Reception: pulse-echo RF from point scatterers, via the combined PE-SDI kernel.

Pulse-echo RF model. eSDIva computes the received radio-frequency (RF) echo from a
field of point scatterers using the spatial-impulse-response scattering model introduced
by Jensen (J. A. Jensen, "A model for the propagation and scattering of ultrasound in
tissue", J. Acoust. Soc. Am. 89(1), 182-190, 1991), the same model implemented by Field
II. For a field of P scatterers at positions r_p with scattering amplitudes σ_p (the
``amplitudes`` argument), the recorded RF is the amplitude-weighted sum

    rf(t) = Σ_p σ_p · [ v_pe ⊛_t h_tx(·; r_p) ⊛_t h_rx(·; r_p) ](t) ,   (the RF equation)

where h_tx(·; r_p), h_rx(·; r_p) are the transmit and receive spatial impulse responses
(SIRs) seen by scatterer p and v_pe is the pulse-echo excitation waveform. Physically
v_pe carries the THIRD time-derivative of the excitation,
``v_pe = (ρ₀/2c₀²) E_m ⊛ ∂³v/∂t³``, but that ∂³ is
never formed explicitly: it is absorbed into the band-limited excitation e and the TX/RX
impulse responses, so in practice ``v_pe ∝ e ⊛ h_e ⊛ h_r`` (Field II's convention — its
``calc_scat``/``calc_hhp`` apply no explicit ∂³).

SIR assumption (where SDI is valid). Everything here rests on the Tupholme-Stepanishen
SIR formulation (G. E. Tupholme, Mathematika 16, 209-224, 1969; P. R. Stepanishen, J.
Acoust. Soc. Am. 49, 1629-1638, 1971): eSDIva evaluates each aperture as a sum of small
rectangular patches whose SIR, in the far field, is a trapezoid with four corner times.
The SDI ("sparse delta integration") development below is purely a fast way to evaluate
the RF equation under that assumption — it introduces NO new physics and is valid only
where the far-field trapezoidal rectangular-patch SIR is.

Evaluations of the SAME RF equation (``p_pe`` = one scatterer's bracket):

    p_pe = v_pe ⊛ (h_tx ⊛ h_rx)                          ← conventional (fst/sdi/auto)
         = F⁻¹{ fs · H_TX(ω) · H_RX(ω) · V_pe(ω) }         ← spectral (default)

* ``fst`` / ``sdi`` / ``auto`` — sample each one-way SIR and FFT-convolve them; delegated
  to `ReceptionConventional` (the string names its trapezoid-sampling kernel).
* ``spectral`` — each one-way SIR spectrum H is closed form (`compute_h_sir_spectrum`: per
  patch, area × two sincs × a delay phasor), so the two-way spectrum is the product
  ``H_TX·H_RX`` evaluated only on the in-band bins of the excitation/IR chain. No time
  sampling, no forward FFT, cost linear in the patch count, exact. ``fs`` converts the
  continuous spectra to the sampled-signal scale (``rfft(h[n]) ≈ fs·H``).

The paired SDI form (16 corner deltas per patch pair, O(M²)) is the separate pedagogic
class `ReceptionPaired`. All agree to correlation ~1.0 with each other and Field II.
Attenuation (causal power law) uses the round trip TX-centre → scatterer → RX element
centre in every method; a soft baffle (``transducer.baffle = "soft"``) is spectral-only.
"""

import time
import warnings

import numpy as np
from scipy.fft import irfft, rfft, rfftfreq

from esdiva.hsir.sir_spectral import (
    compute_h_sir_spectrum,
    compute_twoway_spectrum_summed,
)

from esdiva.utilities.helper_functions import (
    eta_progress as _eta_progress,
    next_pow2 as _next_pow2,
    wrap_tqdm as _wrap_tqdm,
)

from ..attenuation import causal_attenuation_tf, convert_alpha0_to_nepers
from .base import (
    ReceptionBase,
    _warn_if_rx_delays_apods_not_default,
)

# Formulation selector values (see Reception.method). "fst"/"sdi"/"auto" delegate to the
# conventional backend (that string is the SIR-sampling kernel it uses).
_VALID_METHODS = ("spectral", "fst", "sdi", "auto")

# Values routed to the conventional `ReceptionConventional` delegate.
_CONVENTIONAL_METHODS = ("fst", "sdi", "auto")


class Reception(ReceptionBase):
    """Compute pulse-echo RF from point scatterers (Jensen's SIR scattering model).

    ``p_pe = v_pe ⊛ h_tx ⊛ h_rx`` under the Tupholme-Stepanishen far-field trapezoidal
    rectangular-patch SIR. All methods evaluate the same RF equation and agree to
    correlation ~1.0 with each other and with Field II; ``method`` trades speed only:

    * ``"spectral"`` (default) — multiply the closed-form one-way SIR spectra
      ``H_TX·H_RX`` on the in-band bins only. No forward FFT, cost linear in patches,
      exact; the only method that models ``baffle="soft"``.
    * ``"fst"`` / ``"sdi"`` / ``"auto"`` — sample ``h_tx``/``h_rx`` and convolve, delegated
      to `ReceptionConventional`. The string names its SIR-sampling kernel: ``"fst"`` fully
      samples each trapezoid, ``"sdi"`` places sparse corner deltas, ``"auto"`` lets the
      delegate choose per grid.

    The paired SDI form (O(M²), pedagogic) is the separate class `ReceptionPaired`.

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
    fs : float, default 100e6
        Sampling frequency (Hz).
    alpha0 : float or None, default None
        Attenuation in dB/(MHz^y·cm). None = no attenuation.
    freq_power : float, default 1.0
        Attenuation power-law exponent.
    excitation : numpy.ndarray or None, default None
        TX excitation pulse ``(L,)``. If None, uses tx.excitation or delta.
    method : str, default "spectral"
        Pulse-echo formulation: ``"spectral"`` (default) / ``"fst"`` / ``"sdi"`` /
        ``"auto"``. All produce the same RF; they trade speed only.
        ``"fst"``/``"sdi"``/``"auto"`` delegate to `ReceptionConventional`.
    n_depth_bins : "auto" or int, default "auto"
        Spectral speed knob. Scatterers are grouped into this many depth bins so each bin
        uses a short time window — a small ``nfft`` and hence few in-band frequency bins,
        the spectral form's dominant cost factor — with the per-bin results added back on a
        shared sample lattice (big speedup at high scatterer counts). ``"auto"`` sizes it
        from the arrival-time spread; ``1`` disables binning. Applies to the summed RF only.
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
        "method": (str, "Formulation: spectral/fst/sdi/auto"),
        "n_depth_bins": ((int, str), "Spectral depth bins: 'auto' or int"),
        "verbose": (bool, "Print diagnostics"),
    }

    def __init__(
        self,
        tx,
        rx,
        *,
        c=1540.0,
        rho=1.0,
        fs=100e6,
        alpha0=None,
        freq_power=1.0,
        excitation=None,
        method="spectral",
        n_depth_bins="auto",
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
        self.method = self._validate_method(method)
        self.verbose = verbose
        self.n_depth_bins = n_depth_bins
        self._conv = None  # lazily-built Reception for the conventional branch
        self._refresh_sub_elem_attributes()
        _warn_if_rx_delays_apods_not_default(self.rx)

    @staticmethod
    def _validate_method(method):
        if method == "paired":
            raise ValueError(
                "method='paired' moved to its own class: "
                "esdiva.reception.ReceptionPaired(tx, rx, ...)."
            )
        if method not in _VALID_METHODS:
            raise ValueError(
                f"Unknown method {method!r}. Valid: {list(_VALID_METHODS)}"
            )
        return method

    def set(self, name, value):
        """Update a parameter at runtime, then invalidate the conventional delegate.

        Extends `ReceptionBase.set` with validation of the ``"method"`` selector and by
        dropping the cached `ReceptionConventional` (rebuilt on the next conventional call,
        picking up the new tx/rx/medium/excitation state).

        Parameters
        ----------
        name : str
            Parameter name (a key of ``_SETTABLE``, ``"tx"``, or ``"rx"``).
        value : object
            New value; for ``"method"`` it must be a valid selector.
        """
        if name == "method":
            value = self._validate_method(value)
        super().set(name, value)
        # tx/rx/medium/excitation/method changes must rebuild the delegate.
        self._conv = None

    # ------------------------------------------------------------------
    # Backend-specific helpers
    # ------------------------------------------------------------------

    def _compute_pe_time_grid(self, points_m):
        """Time grid covering both TX and RX propagation paths.

        Returns the combined pulse-echo window (``pe_t0``, ``dt``, ``pe_T``) and the two
        one-way windows (``tx_t0``, ``tx_T``, ``rx_t0``, ``rx_T``). The spectral form
        references its TX and RX spectra to ``tx_t0`` / ``rx_t0`` so their product lands at
        ``pe_t0 = tx_t0 + rx_t0``.
        """
        _, tx_t0, dt, tx_T = self._oneway_time_grid(points_m, "tx")
        _, rx_t0, _, rx_T = self._oneway_time_grid(points_m, "rx")
        pe_t0 = tx_t0 + rx_t0
        pe_T = tx_T + rx_T - 1
        return pe_t0, dt, pe_T, tx_t0, tx_T, rx_t0, rx_T

    # ------------------------------------------------------------------
    # Formulation router + conventional delegate
    # ------------------------------------------------------------------

    def _ensure_conv(self, sir_method):
        """Lazily build (and cache) the `ReceptionConventional` used for the conventional
        branch, with its SIR-sampling kernel set to ``sir_method`` (``"fst"``/``"sdi"``/
        ``"auto"``).

        Shares the same ``tx``/``rx`` objects (so focusing state set by
        `sequence_rf`/`scan_focusline`/`synthetic_aperture_rf` is seen) and mirrors
        the medium params. The RX-non-default warning already fired at this object's
        construction, so suppress the duplicate here.
        """
        if self._conv is None:
            from .conventional import ReceptionConventional

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                self._conv = ReceptionConventional(
                    self.tx,
                    self.rx,
                    c=self.c,
                    rho=self.rho,
                    fs=self.fs,
                    alpha0=self.alpha0,
                    freq_power=self.freq_power,
                    excitation=self.excitation,
                    method=sir_method,
                    n_depth_bins="auto",
                    verbose=self.verbose,
                )
        else:
            self._conv.method = sir_method
        return self._conv

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _compute_rf_inner(
        self,
        points_m,
        amps,
        *,
        downsampling=None,
        per_scatterer=False,
        focused_sum=False,
    ):
        """Dispatch on ``self.method`` (spectral / conventional family).

        ``"spectral"`` runs the PE-SDI core here; ``"fst"``/``"sdi"``/
        ``"auto"`` delegate to `ReceptionConventional` with that SIR-sampling kernel.
        Signature is the one `pulse_echo_rf` / `_focused_sum_rf` and the
        `ReceptionBase` wrappers rely on.

        Parameters
        ----------
        points_m : (P, 3) numpy.ndarray
            Scatterer positions in metres.
        amps : (P,) numpy.ndarray
            Scattering amplitudes (float32).
        downsampling : int or None, default None
            Downsample output by this factor.
        per_scatterer : bool, default False
            If True return ``(P, E_rx, Nt)`` without summing over scatterers.
        focused_sum : bool, default False
            If True, beamform on receive inside the kernel → one ``(1, Nt)`` line.
            Mutually exclusive with ``per_scatterer``.

        Returns
        -------
        rf : (E_rx, Nt) or (P, E_rx, Nt) numpy.ndarray
        coords : dict
            Keys ``"t0"`` and ``"dt"`` (seconds).
        """
        if focused_sum and per_scatterer:
            raise ValueError("focused_sum and per_scatterer are mutually exclusive.")
        self._reset_time_log()
        method = self.method

        # Per-element excitation (L, E) needs each TX element's pulse folded into its own
        # partial SIR. The spectral core builds one fused TX spectrum, so it
        # cannot; the conventional core (a per-element SIR loop) can — route there.
        exc = self._resolve_excitation()
        if exc is not None and exc.ndim == 2 and method not in _CONVENTIONAL_METHODS:
            warnings.warn(
                "Per-element excitation (L, E) is only supported by the conventional "
                f"core; falling back from method='{method}' to 'auto'.",
                UserWarning,
                stacklevel=2,
            )
            method = "auto"

        self._last_method = method  # introspection hook (tests / diagnostics)

        if method in _CONVENTIONAL_METHODS:
            conv = self._ensure_conv(method)
            conv._refresh_sub_elem_attributes()  # resync from shared tx/rx state
            out = conv._compute_rf_inner(
                points_m,
                amps,
                downsampling=downsampling,
                per_scatterer=per_scatterer,
                focused_sum=focused_sum,
            )
            self.time_log = conv.time_log  # surface the delegate's phase timings
            return out
        return self._rf_spectral(
            points_m,
            amps,
            downsampling=downsampling,
            per_scatterer=per_scatterer,
            focused_sum=focused_sum,
        )

    # ------------------------------------------------------------------
    # Shared setup + finalisation for the SDI cores
    # ------------------------------------------------------------------

    @staticmethod
    def _band_range(band_mag, tol=1e-4):
        """In-band frequency slice ``[b0, b1)`` where the pulse filter is non-negligible.

        The received signal is shaped by the excitation × impulse-response magnitude
        ``band_mag``; outside the band where it exceeds ``tol`` of its peak the result is
        ~0, so the spectral form's spectra need not be evaluated there. Returns the contiguous
        span covering every significant bin (the whole range if the drive is wideband).
        """
        peak = float(band_mag.max()) if band_mag.size else 0.0
        if peak <= 0.0:
            return 0, band_mag.size
        sig = np.nonzero(band_mag >= tol * peak)[0]
        if sig.size == 0:
            return 0, band_mag.size
        return int(sig[0]), int(sig[-1]) + 1

    def _pe_setup(
        self,
        points_m,
        *,
        per_scatterer,
        focused_sum,
        label,
        grid_override=None,
    ):
        """Common time-grid, FFT-filter and band-range setup for the spectral core.

        Returns a dict of everything the spectral (and `ReceptionPaired`) cores share.

        ``grid_override`` (a ``(pe_t0, dt, pe_T, tx_t0, rx_t0)`` tuple) bypasses the
        natural time grid — used by the depth-binned spectral path, where each bin's
        window is snapped to the global sample lattice so per-bin results add back at an
        integer offset. ``pe_T`` (and hence ``nfft``, the band-bin count) is then the
        SHORT per-bin window, not the whole-field span.
        """
        P = points_m.shape[0]
        n_rx = int(self.rx.delays.shape[0])
        rx_groups = self._rx_groups(focused_sum)
        n_out = len(rx_groups)
        show = self.verbose and not focused_sum  # focused_sum is the quiet primitive

        if grid_override is not None:
            pe_t0, dt, pe_T, tx_t0, rx_t0 = grid_override
        else:
            with self._timer("time_grid_s"):
                pe_t0, dt, pe_T, tx_t0, _tx_T, rx_t0, _rx_T = (
                    self._compute_pe_time_grid(points_m)
                )
        exc = self._resolve_excitation()
        ir_tx = getattr(self.tx, "impulse_response", None)
        ir_rx = getattr(self.rx, "impulse_response", None)
        nfft = _next_pow2(pe_T + len(exc) - 1) if exc is not None else _next_pow2(pe_T)
        freqs = rfftfreq(nfft, d=1.0 / self.fs).astype(np.float32)

        # Excitation and IR spectra (no jω: the physical ∂³ lives in the band-limited chain).
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

        # Band-limiting (spectral): the excitation × impulse-response magnitude bounds the
        # frequency support, so the analytic SIR spectra are evaluated only on this slice.
        band_mag = np.ones(freqs.shape[0], dtype=np.float64)
        for filt in (fft_v, fft_ir_tx, fft_ir_rx):
            if filt is not None:
                band_mag *= np.abs(filt).astype(np.float64)
        b0, b1 = self._band_range(band_mag)
        omega_band = (2.0 * np.pi * freqs[b0:b1]).astype(np.float64)

        if show:
            att = (
                f"alpha0={self.alpha0} dB/(MHz^{self.freq_power} cm)"
                if do_attenuation
                else "None"
            )
            print(f"\n--- {label} ---")
            print(f"  Scatterers : {P}")
            print(f"  TX patches : {self._tx_M}")
            print(f"  RX elements: {n_rx} ({self._rx_M} patches total)")
            print(f"  PE T       : {pe_T} samples   nfft: {nfft}")
            print(f"  Band bins  : {b1 - b0} / {freqs.shape[0]}")
            print(f"  Attenuation: {att}")

        return {
            "rx_groups": rx_groups,
            "n_out": n_out,
            "pe_t0": pe_t0,
            "tx_t0": tx_t0,
            "rx_t0": rx_t0,
            "dt": dt,
            "pe_T": pe_T,
            "inv_c": np.float32(1.0 / self.c),
            "nfft": nfft,
            "freqs": freqs,
            "fft_v": fft_v,
            "fft_ir_tx": fft_ir_tx,
            "fft_ir_rx": fft_ir_rx,
            "b0": b0,
            "b1": b1,
            "omega_band": omega_band,
            "do_attenuation": do_attenuation,
            "scale": np.float32(self.rho / (2.0 * self.c**2)),
            "show": show,
        }

    # ------------------------------------------------------------------
    # spectral PE: F⁻¹{ fs · H_TX(ω) · H_RX(ω) · exc · IR · H_att } on the in-band slice
    # ------------------------------------------------------------------

    def _rf_spectral(
        self,
        points_m,
        amps,
        *,
        downsampling,
        per_scatterer,
        focused_sum,
    ):
        """Spectral PE core: closed-form one-way SIR spectra, multiplied — no forward FFT.

        ``H_TX`` and each receive element's ``H_RX`` are evaluated only on the in-band
        frequencies; their product is the two-way SIR spectrum. For the summed RF the
        scatterers are amplitude-summed in the frequency domain (``amps @ H_TX·H_RX``), the
        shared filter ``G = fs·exc·IR`` is applied per bin, and one inverse FFT per element
        returns the RF. Attenuation multiplies each (scatterer, element) product by
        ``H_att`` over the round trip TX centre → scatterer → element centre.

        Depth binning. The in-band bin count is ``N_band = (BW/fs)·nfft`` and
        ``nfft ≈ next_pow2(pe_T)`` spans the arrival window of ALL scatterers, so a deep
        or wide field inflates ``N_band`` — and the spectrum build costs ``P·M·N_band``,
        the dominant term. Grouping scatterers by depth lets each bin use a SHORT window
        (small ``nfft`` → small ``N_band``), cutting both the build and the per-element
        product-sum by ``nfft_full/nfft_bin``. Each bin shares one global sample lattice,
        so its RF adds back at an integer sample offset (no resampling). Used only for the
        summed RF (``per_scatterer=False``); the per-scatterer PSF keeps one window.
        """
        n_bins = self.n_depth_bins
        if not per_scatterer:
            # n_out<2 (e.g. focused_sum) still benefits — binning shrinks nfft regardless
            # of channel count, so bypass the auto rule's n_out>=2 gate.
            n_bins = (
                self._auto_depth_bins(
                    points_m, max(self._n_spectral_out(focused_sum), 2)
                )
                if n_bins == "auto"
                else int(n_bins)
            )
            if n_bins > 1:
                return self._rf_spectral_binned(
                    points_m,
                    amps,
                    downsampling=downsampling,
                    focused_sum=focused_sum,
                    n_bins=n_bins,
                )

        s = self._pe_setup(
            points_m,
            per_scatterer=per_scatterer,
            focused_sum=focused_sum,
            label="Reception [spectral]",
        )
        pe_t0, dt = s["pe_t0"], s["dt"]

        if not per_scatterer:
            t_wall = time.time()
            rx_csr = self._build_rx_csr(s["rx_groups"])
            rf = self._spectral_summed_from_setup(s, points_m, amps, rx_csr)
            if s["show"]:
                print(
                    f"Reception [spectral] computed in {time.time() - t_wall:.2f} s "
                    f"({self._fmt_time_log()})\n"
                )
            return self._finalize(rf, pe_t0, dt, focused_sum, downsampling)

        # Per-scatterer (PSF): keep each scatterer's trace separate, no depth binning.
        P = points_m.shape[0]
        nfft, pe_T = s["nfft"], s["pe_T"]
        b0, b1, omega_band = s["b0"], s["b1"], s["omega_band"]
        n_freq = s["freqs"].shape[0]
        scale, inv_c = s["scale"], s["inv_c"]
        g_band, atten = self._spectral_filters(s)
        t_wall = time.time()
        with self._timer("sir_s"):
            h_tx = compute_h_sir_spectrum(
                points_m,
                self._tx_centers,
                self._tx_wx,
                self._tx_wy,
                self._tx_apod,
                self._tx_delays,
                inv_c,
                s["tx_t0"],
                omega_band,
                eu=self._tx_eu,
                ev=self._tx_ev,
                soft_baffle=self.tx.baffle == "soft",
            )  # (P, N_band) complex64
        el_iter = (
            _wrap_tqdm(
                range(s["n_out"]), desc="RX elements", total=s["n_out"], leave=True
            )
            if s["show"]
            else range(s["n_out"])
        )
        # ETA + in-place progress only when the projected run exceeds ~30 s
        # (tqdm already shows progress in verbose mode).
        el_iter = _eta_progress(
            el_iter, s["n_out"], label="RX elements", progress=not s["show"]
        )
        rf = np.zeros((P, s["n_out"], pe_T), dtype=np.float32)
        for e_rx in el_iter:
            h_rx = self._spectral_h_rx(s, e_rx, points_m, omega_band)
            with self._timer("fft_s"):
                sp_band = (h_tx * h_rx) * g_band[np.newaxis, :]  # (P, N_band)
                if atten is not None:
                    d = np.linalg.norm(
                        points_m - atten["tx_ref"], axis=1
                    ) + np.linalg.norm(points_m - atten["rx_ref"][e_rx], axis=1)
                    sp_band *= causal_attenuation_tf(
                        s["freqs"][b0:b1], d, self.alpha0, self.freq_power, self.tx.fc
                    )
                full = np.zeros((P, n_freq), dtype=np.complex64)
                full[:, b0:b1] = sp_band
                rf_pe = irfft(full, n=nfft, axis=1)[:, :pe_T]  # (P, pe_T)
            rf[:, e_rx, :] = (rf_pe * amps[:, np.newaxis] * scale).astype(np.float32)
        if s["show"]:
            print(
                f"Reception [spectral] computed in {time.time() - t_wall:.2f} s "
                f"({self._fmt_time_log()})\n"
            )
        return self._finalize(rf, pe_t0, dt, focused_sum, downsampling)

    def _n_spectral_out(self, focused_sum):
        """Number of output channels the spectral core produces (1 if focused_sum)."""
        return 1 if focused_sum else int(self.rx.delays.shape[0])

    @staticmethod
    def _build_rx_csr(rx_groups):
        """Lay the per-element RX patch arrays out element-by-element, CSR-style.

        The fused two-way kernel reads the whole receive aperture as one set of patch
        arrays in which receive element ``e`` occupies the contiguous block
        ``ptr[e]:ptr[e+1]``. This concatenates the per-element groups into that layout and
        returns the patch arrays, their tangent frames, and the offsets ``ptr``.
        """

        def cat(i):
            return np.concatenate([g[i] for g in rx_groups])

        counts = [g[0].shape[0] for g in rx_groups]
        return {
            "centers": cat(0).astype(np.float32),
            "wx": cat(1).astype(np.float32),
            "wy": cat(2).astype(np.float32),
            "apod": cat(3).astype(np.float32),
            "delays": cat(4).astype(np.float32),
            "eu": cat(5).astype(np.float32),
            "ev": cat(6).astype(np.float32),
            "ptr": np.concatenate([[0], np.cumsum(counts)]).astype(np.int64),
        }

    def _spectral_filters(self, s):
        """In-band filter ``G = fs·exc·ir_tx·ir_rx`` and the round-trip attenuation spec.

        ``fs`` turns the product of two continuous SIR spectra into the sampled-RF scale
        (``rfft(h[n]) ≈ fs·H`` per SIR, and the discrete convolution carries one ``dt``).
        Attenuation (None when ``alpha0`` is unset) is referenced to the TX aperture centre
        and each receive element centre (the aperture centroid for a focused sum).
        """
        b0, b1 = s["b0"], s["b1"]
        g_band = np.full(b1 - b0, self.fs, dtype=np.complex64)
        for filt in (s["fft_v"], s["fft_ir_tx"], s["fft_ir_rx"]):
            if filt is not None:
                g_band = g_band * filt[b0:b1]
        if self.alpha0 is None:
            return g_band, None
        ec = np.asarray(self.rx.element_centers, dtype=np.float64)
        return g_band, {
            "alpha0_np": convert_alpha0_to_nepers(self.alpha0, self.freq_power),
            "freq_power": self.freq_power,
            "f0_hz": self.tx.fc,
            "tx_ref": np.asarray(self.tx.element_centers, dtype=np.float64).mean(0),
            "rx_ref": ec if ec.shape[0] == s["n_out"] else ec.mean(0, keepdims=True),
        }

    def _spectral_h_rx(self, s, e_rx, points_m, omega_band):
        """One receive element's closed-form SIR spectrum ``H_RX`` → (P, N_band)."""
        rx_c, rx_wx, rx_wy, rx_ap, rx_dl, rx_eu, rx_ev = s["rx_groups"][e_rx]
        with self._timer("sir_s"):
            return compute_h_sir_spectrum(
                points_m,
                rx_c,
                rx_wx,
                rx_wy,
                rx_ap,
                rx_dl,
                s["inv_c"],
                s["rx_t0"],
                omega_band,
                eu=rx_eu,
                ev=rx_ev,
                soft_baffle=self.rx.baffle == "soft",
            )

    def _spectral_summed_from_setup(self, s, points_m, amps, rx_csr):
        """Summed spectral RF for one (sub-field, window) → ``(n_out, pe_T)`` float32.

        Forms the two-way SIR spectrum ``H_TX·H_RX`` for every receive element and
        amplitude-sums it over scatterers in a single fused parallel pass (the analogue of
        conventional's ``amps @ H_pe``): the transmit one-way spectrum is built once per
        scatterer and reused across all receive elements, with nothing of size
        ``(P, N_band)`` materialised. The shared filter ``G = fs·exc·IR`` is then applied
        per bin and one inverse FFT per element returns the RF. Reused for the single-window
        path and for each depth bin (the window — hence ``nfft`` and the band-bin count —
        comes from ``s``); ``rx_csr`` is the receive aperture laid out element-by-element.

        The fused kernel accumulates the scatterer sum in complex128 internally — cheap
        because ``N_band`` is the small in-band bin count.
        """
        nfft, pe_T = s["nfft"], s["pe_T"]
        b0, b1, omega_band = s["b0"], s["b1"], s["omega_band"]
        n_freq = s["freqs"].shape[0]
        scale = s["scale"]
        g_band, atten = self._spectral_filters(s)
        tx = {
            "centers": self._tx_centers,
            "wx": self._tx_wx,
            "wy": self._tx_wy,
            "apod": self._tx_apod,
            "delays": self._tx_delays,
            "eu": self._tx_eu,
            "ev": self._tx_ev,
            "t0": s["tx_t0"],
            "soft": self.tx.baffle == "soft",
        }
        rx = {**rx_csr, "t0": s["rx_t0"], "soft": self.rx.baffle == "soft"}
        with self._timer("sir_s"):
            s_all = compute_twoway_spectrum_summed(
                points_m,
                amps,
                tx,
                rx,
                rx_csr["ptr"],
                s["inv_c"],
                omega_band,
                atten=atten,
            )  # (n_out, N_band) complex128
        with self._timer("fft_s"):
            full = np.zeros((s["n_out"], n_freq), dtype=np.complex64)
            full[:, b0:b1] = (s_all * g_band[np.newaxis, :]).astype(np.complex64)
            return (irfft(full, n=nfft, axis=1)[:, :pe_T] * scale).astype(np.float32)

    def _rf_spectral_binned(
        self,
        points_m,
        amps,
        *,
        downsampling,
        focused_sum,
        n_bins,
    ):
        """Summed spectral RF, split into depth bins for short per-bin windows.

        Same physical result as the single-window spectral path, but scatterers are grouped
        by depth so each bin spans a tight arrival window → small ``nfft`` → small in-band
        bin count ``N_band`` (the spectral form's dominant cost factor). All bins share one
        global sample lattice (``t0_g``, the reported ``t0``): each bin's window is snapped
        to that lattice and its RF added back at the integer sample offset ``n0`` — no
        resampling. Only the summed RF uses this; per-scatterer does not.
        """
        # Global lattice origin (also the reported t0); every bin snaps to it.
        t0_g, dt, _pe_T, _txt0, _txT, _rxt0, _rxT = self._compute_pe_time_grid(points_m)
        # RX aperture layout is bin-independent → build the element-CSR once for all bins.
        rx_csr = self._build_rx_csr(self._rx_groups(focused_sum))

        def per_bin(idx):
            pts, am = points_m[idx], amps[idx]
            pe_t0_nat, _dt, pe_T_nat, tx_t0_b, _txTb, rx_t0_b, _rxTb = (
                self._compute_pe_time_grid(pts)
            )
            # Snap the bin's pe window to the global lattice: round t0 down to a sample,
            # shift the TX reference by the (sub-sample) remainder so the product still
            # lands at the snapped origin. +1 sample covers the snap.
            n0, pe_t0_snap, shift = self._snap_to_lattice(pe_t0_nat, t0_g, dt)
            s = self._pe_setup(
                pts,
                per_scatterer=False,
                focused_sum=focused_sum,
                label="Reception [spectral]",
                grid_override=(
                    pe_t0_snap,
                    dt,
                    pe_T_nat + 1,
                    tx_t0_b - shift,  # tx_t0_eff + rx_t0_b = pe_t0_snap
                    rx_t0_b,
                ),
            )
            return self._spectral_summed_from_setup(s, pts, am, rx_csr), n0

        # Per-bin setup prints would spam; silence the bin loop, print one summary.
        verbose = self.verbose
        self.verbose = False
        try:
            rf = self._accumulate_depth_bins(points_m, n_bins, per_bin)
        finally:
            self.verbose = verbose

        if verbose and not focused_sum:
            print(
                f"\n--- Reception [spectral, {n_bins} depth bins] ---\n"
                f"  Scatterers : {points_m.shape[0]}   "
                f"RX elements: {self._n_spectral_out(focused_sum)}\n"
                f"  Nt         : {rf.shape[1]}   ({self._fmt_time_log()})"
            )
        return self._finalize(rf, t0_g, dt, focused_sum, downsampling)

    def pulse_echo_rf(
        self,
        scatterer_positions_mm,
        amplitudes=None,
        *,
        per_scatterer=False,
        downsampling=None,
        out_path=None,
        checkpoint_chunks=1,
    ):
        """Pulse-echo RF from point scatterers.

        The core reception primitive: amplitude-weighted superposition of each
        scatterer's pulse-echo response ``v_pe ⊛ h_tx ⊛ h_rx``; the single physical ∂³
        stays on the band-limited excitation / impulse-response chain (``e ⊛ h_e ⊛ h_r``).
        Field II uses the same convention, so this equals Field II ``calc_scat``
        (≡ ``calc_hhp`` for a unit point, corr 1.0000).

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
        out_path : str or pathlib.Path or None, default None
            Checkpoint folder (an ``RFDataset``): the acquisition is written
            to disk as it progresses and a re-run resumes instead of starting
            over — use with ``checkpoint_chunks`` for hours-long phantoms.
        checkpoint_chunks : int, default 1
            Scatterer chunks checkpointed separately (requires ``out_path``);
            the RF is linear in the scatterers, so a crash costs at most one
            chunk. Incompatible with ``per_scatterer=True``.

        Returns
        -------
        rf : (Erx, Nt) or (N_scat, Erx, Nt) numpy.ndarray
            Pulse-echo RF per receive element (channels before time).
        coords : dict
            Keys ``"t0"`` and ``"dt"`` (seconds).

        Raises
        ------
        ValueError
            If ``out_path`` is combined with ``per_scatterer=True``.
        """
        if out_path is not None or checkpoint_chunks != 1:
            return self._checkpointed_pulse_echo(
                scatterer_positions_mm,
                amplitudes,
                per_scatterer=per_scatterer,
                downsampling=downsampling,
                out_path=out_path,
                checkpoint_chunks=checkpoint_chunks,
            )
        points_m, amps = self._validate_scatterer_inputs(
            scatterer_positions_mm, amplitudes
        )
        return self._compute_rf_inner(
            points_m,
            amps,
            downsampling=downsampling,
            per_scatterer=per_scatterer,
        )

    def _focused_sum_rf(self, points_m, amps, *, downsampling=None):
        """Receive-beamformed line via in-kernel focused sum.

        Backend hook for `ReceptionBase.scan_focusline`: the pulse-echo RF with
        ``focused_sum=True``. Returns the single line ``(Nt,)`` and its coords. Field II's ``calc_scat`` builds
        its line the same way (focused, apodized, summed on receive).
        """
        rf, coords = self._compute_rf_inner(
            points_m,
            amps,
            downsampling=downsampling,
            focused_sum=True,
        )
        return rf[0], coords

    def __repr__(self) -> str:
        return (
            f"Reception(tx={self.tx}, rx={self.rx}, c={self.c} m/s, "
            f"fs={self.fs} Hz, alpha0={self.alpha0}, "
            f"freq_power={self.freq_power}, method='{self.method}')"
        )
