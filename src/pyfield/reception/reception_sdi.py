"""ReceptionSDI: pulse-echo RF from point scatterers, via the combined PE-SDI kernel.

Pulse-echo RF model. PyField computes the received radio-frequency (RF) echo from a
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
Acoust. Soc. Am. 49, 1629-1638, 1971): PyField evaluates each aperture as a sum of small
rectangular patches whose SIR, in the far field, is a trapezoid with four corner times.
The SDI ("sparse delta integration") development below is purely a fast way to evaluate
the RF equation under that assumption — it introduces NO new physics and is valid only
where the far-field trapezoidal rectangular-patch SIR is.

Three evaluations of the SAME RF equation. Write ``p_pe`` for one scatterer's bracket
``v_pe ⊛ h_tx ⊛ h_rx`` (the full RF is then ``Σ_p σ_p · p_pe`` as above). Each one-way SIR
is the double time-integral of its piecewise-constant second derivative, a sparse train of
trapezoid-corner deltas (``h_tx = I² D²h_tx``); ``D²h`` of a patch is four signed Diracs at
its corners. Two ways to recombine the TX and RX corner trains give the two SDI methods,
alongside the conventional sampled convolution::

    p_pe = v_pe ⊛ (h_tx ⊛ h_rx)                                    ← conventional
         = (I⁴ v_pe) ⊛ (D²h_tx ⊛ D²h_rx) = w ⊛ Δδ_pe              ← paired
         = F⁻¹{ V_pe/(i*omega)^4 · Σ_TX(ω) · Σ_RX(ω) }                      ← spectral

    Δδ_pe ≡ D²h_tx ⊛ D²h_rx   (16·M_tx·M_Erx deltas: 4 TX corners × 4 RX corners per pair)
    Σ_TX(ω) ≡ F{D²h_tx} = Σ_{m,i} slope σ_i e^{-jω t_i}   (closed form, 4 corners per patch)
    I⁴ in fourier domain is ÷(jω)⁴ four time-integrations; ``w = I⁴ v_pe`` is the integrated drive.

The methods differ only in HOW the TX and RX corner trains are combined:

* ``conventional`` — sample each one-way SIR (place its corner deltas, double-integrate by
  cumulative sum) and FFT-convolve them. The convolution is independent of the patch count
  M, but the SIR *build* is linear in M. Delegated to `Reception`.
* ``paired`` — convolve the two corner-delta trains analytically (deltas ⊛ deltas = deltas)
  into the 16-delta two-way train ``Δδ_pe``, enumerating all ``M_tx·M_Erx`` TX–RX patch
  pairs. The four integrations are pushed onto the drive ONCE, forming the integrated
  pulse-echo waveform ``w = I⁴ v_pe``; each of the 16 corner events of a pair then lays down
  a shifted, scaled copy of ``w`` (``Σ_i Σ_j a_i a_j w(t − τ_i − τ_j)``) — no FFT, no
  cumulative sum, the output is the RF directly. Cost is quadratic in the patch count and
  carries the full kernel length per event, so this is the exact reference path, used for
  tiny apertures (a point-spread function, a monoelement) and cross-checks.
* ``spectral`` — never form the pairs. Each one-way SIR spectrum is the closed-form sum of
  four corner phasors per patch (``Σ_TX``, ``Σ_RX``), built independently and multiplied
  (convolution ⇒ product), so the cost is linear in the patch count (``M_tx + M_Erx``) with
  NO forward FFT at all. Because the received signal is band-limited by the excitation and
  impulse responses, the spectra are evaluated only on the in-band frequencies (the rest is
  multiplied by a near-zero filter); the TX spectrum is built once and each receive
  element's RX spectrum is built in turn. This removes the
  per-scatterer forward FFT that dominates the conventional cost, so ``spectral`` is the fast
  default for both compact apertures and large arrays — and it is exact (no time sampling,
  no interpolation). Per-patch one-way attenuation is folded into each phasor for free.

All three give the same RF (correlation ~1.0 with each other and with Field II). The
SDI forms place ∂² on each one-way SIR (six derivatives via D² ⊛ D²), so the public RF
applies ``I⁴`` to leave exactly the single physical ∂³ on the excitation/IR chain — matching
the conventional form and Field II. ``method="auto"`` picks ``spectral`` whenever the
excitation is band-limited (the usual case) and falls back to ``conventional`` for a
near-delta / wideband excitation where band-limiting gives no benefit; ``paired`` is reserved
for tiny apertures where its quadratic placement is cheap.
"""

import time
import warnings

import numpy as np
from scipy.fft import irfft, rfft, rfftfreq

from pyfield.hsir.transducer_sir_pe_sdi import (
    compute_oneway_spectrum_band,
    compute_pe_complete,
    compute_twoway_spectrum_summed,
)

from ..attenuation import convert_alpha0_to_nepers
from .base import (
    ReceptionBase,
    _next_pow2,
    _warn_if_rx_delays_apods_not_default,
    _wrap_tqdm,
)

# Formulation selector values (see ReceptionSDI.method).
_VALID_METHODS = ("auto", "conventional", "paired", "spectral")

# Router crossover constant (see `_regime_select`): compares paired's pair count
# `16·M_tx·M_Erx` against the patch-independent transform cost `k·log₂T`, so paired wins
# only for a handful of patches. Approximate — re-tune against measured timings.
_PE_FFT_CONST = 4.0


class ReceptionSDI(ReceptionBase):
    """Compute pulse-echo RF from point scatterers via sparse delta integration (SDI).

    Implements Jensen's spatial-impulse-response pulse-echo model
    ``p_pe = v_pe ⊛ h_tx ⊛ h_rx`` under the Tupholme-Stepanishen far-field trapezoidal
    rectangular-patch SIR (full derivation of the three formulations in the module
    docstring above). Each one-way SIR is the double integral of a sparse train of
    trapezoid-corner deltas; the two-way SIR is recovered with the four integrations
    ``I⁴ = ÷(jω)⁴``. All formulations evaluate the same RF equation and agree to
    correlation ~1.0 with each other and with Field II. ``method`` picks how the TX/RX
    corner trains are combined (speed trade-off only):

    * ``"spectral"`` (fast default) — multiply the two closed-form one-way SIR spectra on
      the in-band bins only. No forward FFT, cost linear in patches, exact, and folds
      per-patch one-way attenuation in for free.
    * ``"conventional"`` — sample ``h_tx``/``h_rx`` and convolve (delegated to `Reception`).
      Chosen by ``auto`` for a near-delta / wideband excitation where band-limiting helps not.
    * ``"paired"`` — splat the integrated drive ``w = I⁴ v_pe`` at the 16 corner events of
      every TX–RX patch pair (no FFT, no cumsum). Exact but quadratic in patches, so the
      reference path for tiny apertures (PSF, monoelement). No attenuation.
    * ``"auto"`` (default) — ``paired`` for a handful of patches, else ``spectral`` (band-
      limited drive) or ``conventional`` (wideband).

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
        Pulse-echo formulation: ``"auto"`` / ``"conventional"`` / ``"spectral"`` /
        ``"paired"``. All produce the same RF; they trade speed by regime.
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
        "method": (str, "Formulation: auto/conventional/spectral/paired"),
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
        fs=200e6,
        alpha0=None,
        freq_power=1.0,
        excitation=None,
        method="auto",
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
        if method not in _VALID_METHODS:
            raise ValueError(
                f"Unknown method {method!r}. Valid: {list(_VALID_METHODS)}"
            )
        return method

    def set(self, name, value):
        """Update a parameter at runtime, then invalidate the conventional delegate.

        Extends `ReceptionBase.set` with validation of the ``"method"`` selector and by
        dropping the cached `Reception` (rebuilt on the next conventional call, picking up
        the new tx/rx/medium/excitation state).

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

    def _ensure_conv(self):
        """Lazily build (and cache) the `Reception` used for the conventional branch.

        Shares the same ``tx``/``rx`` objects (so focusing state set by
        `sequence_rf`/`scan_focusline`/`synthetic_aperture_rf` is seen) and mirrors
        the medium params. The RX-non-default warning already fired at this object's
        construction, so suppress the duplicate here.
        """
        if self._conv is None:
            from .reception import Reception

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                self._conv = Reception(
                    self.tx,
                    self.rx,
                    c=self.c,
                    rho=self.rho,
                    fs=self.fs,
                    alpha0=self.alpha0,
                    freq_power=self.freq_power,
                    excitation=self.excitation,
                    method="auto",
                    n_depth_bins="auto",
                    verbose=self.verbose,
                )
        return self._conv

    def _regime_select(self, points_m, focused_sum):
        """Pick the formulation for ``method="auto"`` by estimated cost.

        ``paired`` splats the full integrated drive ``w`` for each of the
        ``16·M_tx·M_Erx`` corner events, so its cost grows with the patch count squared (and
        carries the kernel length per event) — only cheap for a handful of patches (a
        monoelement, a point-spread function), prohibitive otherwise. The patch-independent
        alternative costs one ``~T·log₂T`` transform: ``spectral`` when the drive is
        band-limited (the usual case — it builds the SIR spectra on the in-band bins only and
        is exact), else ``conventional`` (samples the SIR, bandwidth-agnostic). So a few-patch
        aperture goes to ``paired``; everything else goes to ``spectral`` (or ``conventional``
        for a near-delta / wideband drive).
        """
        exc = self._resolve_excitation()
        ir = getattr(self.tx, "impulse_response", None)
        large_aperture_method = (
            "spectral" if (exc is not None or ir is not None) else "conventional"
        )
        n_rx = int(self.rx.delays.shape[0])
        # focused_sum sums every RX patch in one kernel call → M_Erx = all RX patches.
        m_erx = self._rx_M if focused_sum else max(1, self._rx_M // max(1, n_rx))
        _, _, pe_T = self._compute_pe_time_grid(points_m)[:3]
        # Both sides carry a ~pe_T factor (paired's w length ≈ the transform length), so it
        # cancels: paired wins only while its pair count beats k·log₂T (i.e. very few patches).
        pair_cost = 16.0 * self._tx_M * m_erx
        fft_cost = _PE_FFT_CONST * np.log2(max(pe_T, 2))
        return "paired" if pair_cost < fft_cost else large_aperture_method

    def _resolve_method(self, points_m, focused_sum):
        if self.method != "auto":
            return self.method
        return self._regime_select(points_m, focused_sum)

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
        """Dispatch to the resolved formulation (conventional / spectral / paired).

        Signature is the one `pulse_echo_rf` / `_focused_sum_rf` and the
        `ReceptionBase` wrappers rely on. ``n_integrations`` is the PE-SDI integration
        count (4 = full I⁴) used by the spectral/paired cores; the conventional
        branch ignores it (it builds ``h_tx ⊛ h_rx`` directly, Field II convention).

        Parameters
        ----------
        points_m : (P, 3) numpy.ndarray
            Scatterer positions in metres.
        amps : (P,) numpy.ndarray
            Scattering amplitudes (float32).
        n_integrations : int, default 0
            Frequency-domain integrations (÷``(jω)``) for the SDI cores; 4 = full I⁴.
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
        resolved = self._resolve_method(points_m, focused_sum)
        self._last_method = resolved  # introspection hook (tests / diagnostics)

        if resolved == "conventional":
            conv = self._ensure_conv()
            conv._refresh_sub_elem_attributes()  # resync from shared tx/rx state
            return conv._compute_rf_inner(
                points_m,
                amps,
                downsampling=downsampling,
                per_scatterer=per_scatterer,
                focused_sum=focused_sum,
            )
        core = self._rf_paired if resolved == "paired" else self._rf_spectral
        return core(
            points_m,
            amps,
            n_integrations=n_integrations,
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
        n_integrations,
        per_scatterer,
        focused_sum,
        label,
        grid_override=None,
    ):
        """Common time-grid, FFT-filter, band-range and ``inv_jw_pow`` setup.

        Returns a dict of everything the spectral/paired cores share. ``do_attenuation`` is a
        bare flag: the spectral core folds per-patch attenuation into each phasor itself
        (using the patch-to-point distance already in hand), and the paired core does not
        support attenuation — so no combined round-trip distance is precomputed here.

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
            pe_t0, dt, pe_T, tx_t0, _tx_T, rx_t0, _rx_T = self._compute_pe_time_grid(
                points_m
            )
        exc = self._resolve_excitation()
        ir_tx = getattr(self.tx, "impulse_response", None)
        ir_rx = getattr(self.rx, "impulse_response", None)
        nfft = _next_pow2(pe_T + len(exc) - 1) if exc is not None else _next_pow2(pe_T)
        freqs = rfftfreq(nfft, d=1.0 / self.fs).astype(np.float32)

        # Pre-compute excitation and IR FFTs (no jw — derivatives in Δδ_pe).
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

        # Frequency-domain integration ÷(jω)^n (÷(jω)⁴ for n_integrations=4): exact
        # inverse of the analytic (jω) derivative, so ZERO group delay (a cumsum would
        # add ½ sample each). The f=0 bin is zeroed (exc/ir band-pass carries no DC).
        # ×fs: Δδ_pe holds delta *areas*; ÷(jω) integrates with a dt weight, so it
        # under-counts by fs=1/dt — one ×fs restores it.
        inv_jw_pow = None
        if n_integrations > 0:
            jw = 1j * 2.0 * np.pi * freqs.astype(np.float64)
            inv = np.zeros_like(jw)
            nz = freqs > 0
            inv[nz] = (1.0 / jw[nz]) ** n_integrations
            inv *= self.fs
            inv_jw_pow = inv.astype(np.complex64)

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
            "inv_jw_pow": inv_jw_pow,
            "b0": b0,
            "b1": b1,
            "omega_band": omega_band,
            "do_attenuation": do_attenuation,
            "scale": np.float32(self.rho / (2.0 * self.c**2)),
            "show": show,
        }

    # ------------------------------------------------------------------
    # spectral SDI PE: F⁻¹{ Σ_TX(ω) · Σ_RX(ω) · I⁴ · exc · IR } on the in-band slice
    # ------------------------------------------------------------------

    def _rf_spectral(
        self,
        points_m,
        amps,
        *,
        n_integrations,
        downsampling,
        per_scatterer,
        focused_sum,
    ):
        """Spectral SDI PE core: closed-form one-way spectra, multiplied — no forward FFT.

        Builds the TX one-way SIR spectrum ``Σ_TX`` once (a sum of corner phasors per
        patch, evaluated only on the in-band frequencies), then loops over receive elements
        building each ``Σ_RX`` the same way. The two-way SIR spectrum is the TX×RX product;
        for the summed RF the scatterers are amplitude-summed in the frequency domain
        (``amps @ Σ_TX·Σ_RX``, the exact analogue of conventional's ``amps @ H_pe``), the
        shared filter ``G = I⁴ · exc · IR`` is applied per bin, and one inverse FFT per
        element returns the RF. Per-patch one-way attenuation is folded into ``Σ_TX`` and
        ``Σ_RX`` so their product carries the true round-trip loss. Exact (no time sampling).

        Depth binning. The in-band bin count is ``N_band = (BW/fs)·nfft`` and
        ``nfft ≈ next_pow2(pe_T)`` spans the arrival window of ALL scatterers, so a deep
        or wide field inflates ``N_band`` — and the spectrum build costs ``P·M·N_band``,
        the dominant term. Grouping scatterers by depth lets each bin use a SHORT window
        (small ``nfft`` → small ``N_band``), cutting both the build and the per-element
        product-sum by ``nfft_full/nfft_bin``. Each bin shares one global sample lattice,
        so its RF adds back at an integer sample offset (no resampling). Used only for the
        summed RF (``per_scatterer=False``); per-scatterer / attenuated paths keep the
        single-window path below.
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
                    n_integrations=n_integrations,
                    downsampling=downsampling,
                    focused_sum=focused_sum,
                    n_bins=n_bins,
                )

        s = self._pe_setup(
            points_m,
            n_integrations=n_integrations,
            per_scatterer=per_scatterer,
            focused_sum=focused_sum,
            label="ReceptionSDI [spectral]",
        )
        if s["inv_jw_pow"] is None:
            raise ValueError("method='spectral' requires n_integrations > 0 (full I⁴).")
        pe_t0, dt = s["pe_t0"], s["dt"]

        if not per_scatterer:
            t_wall = time.time()
            rx_csr = self._build_rx_csr(s["rx_groups"])
            rf = self._spectral_summed_from_setup(s, points_m, amps, rx_csr)
            if s["show"]:
                print(
                    f"ReceptionSDI [spectral] computed in {time.time() - t_wall:.2f} s\n"
                )
            return self._finalize(rf, pe_t0, dt, focused_sum, downsampling)

        # Per-scatterer (PSF): keep each scatterer's trace separate, no depth binning.
        P = points_m.shape[0]
        nfft, pe_T = s["nfft"], s["pe_T"]
        b0, b1, omega_band = s["b0"], s["b1"], s["omega_band"]
        n_freq = s["freqs"].shape[0]
        scale, inv_c = s["scale"], s["inv_c"]
        g_band, atten_kw = self._spectral_filters(s)
        t_wall = time.time()
        h_tx = compute_oneway_spectrum_band(
            points_m,
            self._tx_centers,
            self._tx_wx,
            self._tx_wy,
            self._tx_apod,
            self._tx_delays,
            inv_c,
            s["tx_t0"],
            omega_band,
            dt,
            eu=self._tx_eu,
            ev=self._tx_ev,
            **atten_kw,
        )  # (P, N_band) complex64
        el_iter = (
            _wrap_tqdm(
                range(s["n_out"]), desc="RX elements", total=s["n_out"], leave=True
            )
            if s["show"]
            else range(s["n_out"])
        )
        rf = np.zeros((P, s["n_out"], pe_T), dtype=np.float32)
        for e_rx in el_iter:
            h_rx = self._spectral_h_rx(s, e_rx, points_m, omega_band, dt, atten_kw)
            sp_band = (h_tx * h_rx) * g_band[np.newaxis, :]  # (P, N_band)
            full = np.zeros((P, n_freq), dtype=np.complex64)
            full[:, b0:b1] = sp_band
            rf_pe = irfft(full, n=nfft, axis=1)[:, :pe_T]  # (P, pe_T)
            rf[:, e_rx, :] = (rf_pe * amps[:, np.newaxis] * scale).astype(np.float32)
        if s["show"]:
            print(f"ReceptionSDI [spectral] computed in {time.time() - t_wall:.2f} s\n")
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
        """Shared in-band filter ``G = ÷(jω)⁴·exc·ir_tx·ir_rx`` and attenuation kwargs.

        ``G`` is applied once per frequency bin to the summed two-way spectrum; the
        attenuation kwargs are folded per-patch into each one-way spectrum by the kernel.
        """
        b0, b1 = s["b0"], s["b1"]
        g_band = s["inv_jw_pow"][b0:b1].astype(np.complex64)
        for filt in (s["fft_v"], s["fft_ir_tx"], s["fft_ir_rx"]):
            if filt is not None:
                g_band = g_band * filt[b0:b1]
        a0_np = (
            convert_alpha0_to_nepers(self.alpha0, self.freq_power)
            if self.alpha0 is not None
            else None
        )
        atten_kw = {
            "alpha0_np": a0_np,
            "freq_power": self.freq_power,
            "f0_hz": self.tx.fc,
        }
        return g_band, atten_kw

    def _spectral_h_rx(self, s, e_rx, points_m, omega_band, dt, atten_kw):
        """One receive element's closed-form one-way SIR spectrum ``Σ_RX`` → (P, N_band)."""
        rx_c, rx_wx, rx_wy, rx_ap, rx_dl, rx_eu, rx_ev = s["rx_groups"][e_rx]
        return compute_oneway_spectrum_band(
            points_m,
            rx_c,
            rx_wx,
            rx_wy,
            rx_ap,
            rx_dl,
            s["inv_c"],
            s["rx_t0"],
            omega_band,
            dt,
            eu=rx_eu,
            ev=rx_ev,
            **atten_kw,
        )

    def _spectral_summed_from_setup(self, s, points_m, amps, rx_csr):
        """Summed spectral RF for one (sub-field, window) → ``(n_out, pe_T)`` float32.

        Forms the two-way SIR spectrum ``Σ_TX·Σ_RX`` for every receive element and
        amplitude-sums it over scatterers in a single fused parallel pass (the analogue of
        conventional's ``amps @ H_pe``): the transmit one-way spectrum is built once per
        scatterer and reused across all receive elements, with nothing of size
        ``(P, N_band)`` materialised. The shared filter ``G = I⁴·exc·IR`` is then applied
        per bin and one inverse FFT per element returns the RF. Reused for the single-window
        path and for each depth bin (the window — hence ``nfft`` and the band-bin count —
        comes from ``s``); ``rx_csr`` is the receive aperture laid out element-by-element.

        The fused kernel accumulates the scatterer sum in complex128 internally — cheap
        because ``N_band`` is the small in-band bin count, and well conditioned because the
        one-way spectrum is built in its cancellation-free factored form.
        """
        nfft, pe_T = s["nfft"], s["pe_T"]
        b0, b1, omega_band = s["b0"], s["b1"], s["omega_band"]
        n_freq = s["freqs"].shape[0]
        scale, inv_c, dt = s["scale"], s["inv_c"], s["dt"]
        g_band, atten_kw = self._spectral_filters(s)

        s_all = compute_twoway_spectrum_summed(
            points_m,
            amps,
            self._tx_centers,
            self._tx_wx,
            self._tx_wy,
            self._tx_apod,
            self._tx_delays,
            s["tx_t0"],
            rx_csr["centers"],
            rx_csr["wx"],
            rx_csr["wy"],
            rx_csr["apod"],
            rx_csr["delays"],
            s["rx_t0"],
            rx_csr["ptr"],
            inv_c,
            omega_band,
            dt,
            tx_eu=self._tx_eu,
            tx_ev=self._tx_ev,
            rx_eu=rx_csr["eu"],
            rx_ev=rx_csr["ev"],
            **atten_kw,
        )  # (n_out, N_band) complex128
        full = np.zeros((s["n_out"], n_freq), dtype=np.complex64)
        full[:, b0:b1] = (s_all * g_band[np.newaxis, :]).astype(np.complex64)
        return (irfft(full, n=nfft, axis=1)[:, :pe_T] * scale).astype(np.float32)

    def _rf_spectral_binned(
        self,
        points_m,
        amps,
        *,
        n_integrations,
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
        resampling. Only the summed RF uses this; attenuation/per-scatterer do not.
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
            n0 = int(np.floor((pe_t0_nat - t0_g) / dt))
            pe_t0_snap = t0_g + n0 * dt
            shift = pe_t0_nat - pe_t0_snap  # in [0, dt)
            s = self._pe_setup(
                pts,
                n_integrations=n_integrations,
                per_scatterer=False,
                focused_sum=focused_sum,
                label="ReceptionSDI [spectral]",
                grid_override=(
                    pe_t0_snap,
                    dt,
                    pe_T_nat + 1,
                    tx_t0_b - shift,  # tx_t0_eff + rx_t0_b = pe_t0_snap
                    rx_t0_b,
                ),
            )
            if s["inv_jw_pow"] is None:
                raise ValueError(
                    "method='spectral' requires n_integrations > 0 (full I⁴)."
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
                f"\n--- ReceptionSDI [spectral, {n_bins} depth bins] ---\n"
                f"  Scatterers : {points_m.shape[0]}   "
                f"RX elements: {self._n_spectral_out(focused_sum)}\n"
                f"  Nt         : {rf.shape[1]}"
            )
        return self._finalize(rf, t0_g, dt, focused_sum, downsampling)

    # ------------------------------------------------------------------
    # paired SDI PE: Σ a_i a_j w(t − τ_i − τ_j),  w = I⁴ v_pe (no FFT, no cumsum)
    # ------------------------------------------------------------------

    def _rf_paired(
        self,
        points_m,
        amps,
        *,
        n_integrations,
        downsampling,
        per_scatterer,
        focused_sum,
    ):
        """Paired SDI PE core: precompute ``w = I⁴ v_pe`` once, splat it per patch pair.

        Pushes the four integrations onto the drive once (``w = I⁴ v_pe``), then for each of
        the 16 corner events of every TX–RX patch pair lays down a shifted, scaled copy of
        ``w`` — no FFT and no cumulative sum, the output is the RF directly. Exact (it
        reproduces the Fourier convolution), but O(len(w)) per pair, so it is the small-
        aperture / cross-check path. Attenuation is not supported here (it would need a
        separate integrated kernel per depth); use ``method='spectral'`` or ``'conventional'``.
        """
        s = self._pe_setup(
            points_m,
            n_integrations=n_integrations,
            per_scatterer=per_scatterer,
            focused_sum=focused_sum,
            label="ReceptionSDI [paired]",
        )
        if s["do_attenuation"]:
            raise NotImplementedError(
                "method='paired' does not support attenuation; "
                "use method='spectral' or 'conventional'."
            )
        if s["inv_jw_pow"] is None:
            raise ValueError("method='paired' requires n_integrations > 0 (full I⁴).")

        pe_t0, pe_T, dt = s["pe_t0"], s["pe_T"], s["dt"]
        # w = I⁴ v_pe = irfft( ÷(jω)⁴ · fft_v · fft_ir_tx · fft_ir_rx ) on the pe_T grid.
        # ÷(jω)⁴ is zero-phase and delocalized, so the per-pair splat must be circular over
        # the full nfft (sliced to pe_T) to match the FFT convolution; hence the full-length
        # kernel.
        filt = s["inv_jw_pow"].astype(np.complex128)
        for f in (s["fft_v"], s["fft_ir_tx"], s["fft_ir_rx"]):
            if f is not None:
                filt = filt * f
        w = np.ascontiguousarray(irfft(filt, n=s["nfft"]))  # length nfft

        P = points_m.shape[0]
        t_wall = time.time()
        rf = np.zeros(
            (P, s["n_out"], pe_T) if per_scatterer else (s["n_out"], pe_T),
            dtype=np.float32,
        )
        el_iter = (
            _wrap_tqdm(
                range(s["n_out"]), desc="RX elements", total=s["n_out"], leave=True
            )
            if s["show"]
            else range(s["n_out"])
        )
        for e_rx in el_iter:
            rx_c, rx_wx, rx_wy, rx_ap, rx_dl, rx_eu, rx_ev = s["rx_groups"][e_rx]
            rf_pe = compute_pe_complete(
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
                w,
                s["inv_c"],
                pe_t0,
                pe_T,
                self.fs,
                dt,
                tx_eu=self._tx_eu,
                tx_ev=self._tx_ev,
                rx_eu=rx_eu,
                rx_ev=rx_ev,
            )  # (P, pe_T) float32 — already the RF (w convolved in)
            if per_scatterer:
                rf[:, e_rx, :] = (rf_pe * amps[:, np.newaxis] * s["scale"]).astype(
                    np.float32
                )
            else:
                rf[e_rx, :] = ((amps @ rf_pe) * s["scale"]).astype(np.float32)

        if s["show"]:
            print(f"ReceptionSDI [paired] computed in {time.time() - t_wall:.2f} s\n")
        return self._finalize(rf, pe_t0, dt, focused_sum, downsampling)

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
        scatterer's pulse-echo response. The SDI cores carry a second derivative on
        each one-way SIR (``D²h_tx``, ``D²h_rx``), so the public RF applies the four
        integrations ``I⁴ = ÷(jω)⁴`` in the frequency domain to recover the two-way SIR
        ``h_tx ⊛ h_rx``; the single physical ∂³ stays on the band-limited excitation /
        impulse-response chain (``e ⊛ h_e ⊛ h_r``). Frequency-domain integration carries
        no group delay, so the result stays sample-aligned with conventional `Reception`.
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
            n_integrations=4,
            downsampling=downsampling,
            per_scatterer=per_scatterer,
        )

    def _focused_sum_rf(self, points_m, amps, *, downsampling=None):
        """Receive-beamformed line via in-kernel focused sum.

        Backend hook for `ReceptionBase.scan_focusline`: the pulse-echo RF
        (``n_integrations=4`` — full I⁴ in Fourier; the 3 SIR derivatives are
        relocated to the excitation/IR chain) with ``focused_sum=True``. Returns
        the single line ``(Nt,)`` and its coords. Field II's ``calc_scat`` builds
        its line the same way (focused, apodized, summed on receive).
        """
        rf, coords = self._compute_rf_inner(
            points_m,
            amps,
            n_integrations=4,
            downsampling=downsampling,
            focused_sum=True,
        )
        return rf[0], coords

    def __repr__(self) -> str:
        return (
            f"ReceptionSDI(tx={self.tx}, rx={self.rx}, c={self.c} m/s, "
            f"fs={self.fs} Hz, alpha0={self.alpha0}, "
            f"freq_power={self.freq_power}, method='{self.method}')"
        )
