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
is the double time-integral of its piecewise-constant second derivative, which is a
sparse train of corner deltas (h_tx = I² D²h_tx). Substituting into ``p_pe`` and moving
the four integrations through the convolutions yields one identity chain; each equality
is one ``method``::

    p_pe = v_pe ⊛ (h_tx ⊛ h_rx)                                    ← (1) Conventional
         = v_pe ⊛ I⁴(D²h_tx ⊛ D²h_rx) = v_pe ⊛ (I⁴ Δδ_pe)          ← (2) Truncated SDI PE
         = (I⁴ v_pe) ⊛ Δδ_pe = w ⊛ Δδ_pe
         = Σ_i Σ_j a_i a_j · w(t − τ_i − τ_j)                       ← (3) Complete SDI PE

    Δδ_pe ≡ D²h_tx ⊛ D²h_rx   (analytic; deltas ⊛ deltas = deltas, 16·M_tx·M_Erx of them)
    w     ≡ I⁴ v_pe           (analytic; precomputed once)
    I⁴ = ÷(jω)⁴               four time-integrations, applied in the Fourier domain.
    a_i, a_j / τ_i, τ_j       TX-patch i and RX-patch j apodization weights / two-way
                              delays — these build one scatterer's SIR and are NOT the
                              scatterer amplitude σ_p, which multiplies the whole bracket.

The three differ only in HOW ``p_pe`` is evaluated — at which stage the four integrations
are applied, and whether the two-way SIR is kept factored or expanded:

1. Conventional — ``p_pe = v_pe ⊛ (h_tx ⊛ h_rx)``. Build each one-way SIR by placing its
   second-derivative corner deltas and double-integrating (cumulative sum) to recover
   ``h_tx`` and ``h_rx``, then two FFT convolutions. The convolution is independent of the
   patch count M, but the SIR *build* is linear in M.
2. Truncated (this class's default) — ``p_pe = v_pe ⊛ (I⁴ Δδ_pe)``. Form ``Δδ_pe``
   directly as the analytic convolution of the two corner-delta trains (deltas ⊛ deltas =
   deltas → 16 deltas per TX-RX patch pair, the 4 TX corners × 4 RX corners), then realize
   ``I⁴`` entirely in Fourier as ``÷(jω)⁴`` folded into the single spectral multiply with
   ``v_pe`` and the impulse responses — no time-domain cumulative sum (which also avoids
   float32 cancellation when sub-sample patches give very large trapezoid slopes). One FFT
   convolution.
3. Complete — push all four integrations onto the excitation, ``w = I⁴ v_pe`` (precomputed
   once), so the convolution collapses to a closed sum of shifted, scaled kernels,
   ``p_pe = Σ_i Σ_j a_i a_j w(t − τ_i − τ_j)``. No FFT and no cumulative sum — just
   ``16·M_tx·M_Erx`` scaled, shifted copies of ``w`` (M_tx, M_Erx = TX and per-element RX
   patch counts). All three give the same RF (correlation ~1.0 with each other and Field II).

Why a router, not one method. Separability sets the convolution cost. The pair weights
``a_i a_j`` are rank-1 (an outer product ``a_tx ⊗ a_rx``) and the delays ``τ_i + τ_j`` are
an outer sum, so the double sum is a separable bilinear form. FFT convolution (conventional,
truncated) exploits that — cost ``~T·log T``, independent of patch count. Pair enumeration
(complete) discards it and pays the ``M²`` patch-pair count to rediscover it. Convolving two
sparse delta trains is fundamentally ``min(M², T·log T)``: enumeration wins only when the
patch count is tiny (a point-spread function or monoelement), or when the physics breaks
separability — e.g. per-path attenuation, where each patch-pair path has its own distance so
the weights are no longer rank-1 and enumeration is forced. ``method="auto"`` picks the
cheaper side per call.

Derivative bookkeeping: the ``Δδ_pe``-based forms place ∂² on each one-way SIR (six
derivatives total via D² ⊛ D²), so the public RF applies ``I⁴`` to leave exactly the single
physical ∂³ on the excitation/IR chain — matching the conventional form and Field II.
"""

import time
import warnings

import numpy as np
from scipy.fft import irfft, rfft, rfftfreq

from pyfield.hsir.transducer_sir_pe_sdi import (
    compute_pe_complete,
    compute_pe_sdi,
    compute_pe_sdi_summed,
)
from pyfield.utilities.helper_functions import compute_time_grid

from ..attenuation import causal_attenuation_tf, compute_reception_distances
from .base import (
    ReceptionBase,
    _anti_alias_decimate,
    _next_pow2,
    _warn_if_rx_delays_apods_not_default,
    _wrap_tqdm,
)

# Formulation selector values (see ReceptionSDI.method).
_VALID_METHODS = ("auto", "conventional", "truncated", "complete")

# Router constant scaling the FFT-cost side of `16·M_tx·M_Erx ⋛ k·T·log₂T` (truncated
# pair-placement cost vs conventional FFT cost). A delta write is far cheaper than an FFT
# op, so k > 1 keeps mid-size apertures on the (measured-faster) truncated path while
# still routing large arrays to conventional. Approximate — calibrated from a couple of
# points (mid-size M_tx≈512, pair/fft≈1.2 → truncated; large M_tx≈1280, ≈20 → conventional);
# re-tune against measured timings on a different CPU.
_PE_FFT_CONST = 4.0


class ReceptionSDI(ReceptionBase):
    """Compute pulse-echo RF from point scatterers via sparse delta integration (SDI).

    Implements Jensen's spatial-impulse-response pulse-echo model
    ``p_pe = v_pe ⊛ h_tx ⊛ h_rx`` under the Tupholme-Stepanishen far-field trapezoidal
    rectangular-patch SIR (full model and citations in the module docstring above). The
    SIR convolution is evaluated by SDI: each aperture's one-way SIR is the double
    integral of a sparse train of trapezoid-corner deltas, so the two-way SIR follows
    from the corner-delta product ``Δδ_pe = D²h_tx ⊛ D²h_rx`` (16 deltas per TX-RX patch
    pair) with the four integrations ``I⁴ = ÷(jω)⁴`` applied in the Fourier domain — no
    time-domain cumsum. This default ("truncated") form, the direct-convolution
    ("conventional") form, and the FFT-free ("complete") form are the same RF equation
    evaluated at different stages; they agree to correlation ~1.0 with each other and
    with Field II.

    The formulation is chosen by ``method`` (see the module docstring for the full
    identity chain relating them):

    * ``"truncated"`` — this class's PE-SDI kernel: place ``Δδ_pe`` then apply ``I⁴`` and
      the excitation/IR filters in one Fourier multiply. Best for small ``M`` (point-
      spread function, monoelement, few patches); the FFT is amortized over scatterers.
    * ``"conventional"`` — build ``h_tx`` and ``h_rx`` and convolve directly (delegated
      to `Reception`, including its depth-bin fast path). Best for arrays / many
      scatterers, where the patch-pair count ``16·M_tx·M_Erx`` exceeds the FFT cost.
    * ``"complete"`` — move ``I⁴`` onto the excitation (``w = I⁴ v_pe``) and accumulate
      shifted, scaled copies of ``w`` per patch pair. FFT-free and exact, but slowest at
      scale — a reference / single-scatterer backend. No attenuation support yet.
    * ``"auto"`` (default) — pick by regime: arrays → conventional, small ``M`` / point-
      spread function → truncated. Makes the fast formulation the default.

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
        Pulse-echo formulation: ``"auto"`` / ``"conventional"`` / ``"truncated"`` /
        ``"complete"``. All produce the same RF; they trade speed by regime.
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
        "method": (str, "Formulation: auto/conventional/truncated/complete"),
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
        self.method = self._validate_method(method)
        self.verbose = verbose
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

        Extends `ReceptionBase.set` with validation of the ``"method"`` selector and
        by dropping the cached `Reception` (rebuilt on the next conventional call,
        picking up the new tx/rx/medium/excitation state).

        Parameters
        ----------
        name : str
            Parameter name (a key of ``_SETTABLE``, ``"tx"``, or ``"rx"``).
        value : object
            New value; for ``"method"`` it must be one of ``_VALID_METHODS``.
        """
        if name == "method":
            value = self._validate_method(value)
        super().set(name, value)
        # tx/rx/medium/excitation/method changes must rebuild the delegate.
        self._conv = None

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

    def _regime_select(self, points_m, per_scatterer, focused_sum):
        """Pick the formulation for ``method="auto"`` by estimated cost.

        Truncated places ``16·M_tx·M_Erx`` delta pairs per scatterer; conventional
        instead pays one FFT of length ``~T`` (independent of patch count). So a
        point-spread / single-scatterer run (``per_scatterer``) and small apertures go
        to truncated (its FFT cost is amortized over scatterers), while large arrays —
        where the pair count ``16·M_tx·M_Erx`` exceeds the FFT cost ``~T·log₂T`` — go to
        conventional. Attenuated arrays also route to conventional, which applies the
        per-scatterer attenuation transfer function ``H_att``.
        """
        if per_scatterer:
            return "truncated"
        n_rx = int(self.rx.delays.shape[0])
        # focused_sum sums every RX patch in one kernel call → M_Erx = all RX patches.
        m_erx = self._rx_M if focused_sum else max(1, self._rx_M // max(1, n_rx))
        _, _, pe_T = self._compute_pe_time_grid(points_m)
        pair_cost = 16.0 * self._tx_M * m_erx
        fft_cost = _PE_FFT_CONST * pe_T * np.log2(max(pe_T, 2))
        return "truncated" if pair_cost < fft_cost else "conventional"

    def _resolve_method(self, points_m, per_scatterer, focused_sum):
        if self.method != "auto":
            return self.method
        return self._regime_select(points_m, per_scatterer, focused_sum)

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
        """Dispatch to the resolved formulation (conventional / truncated / complete).

        Signature is the one `pulse_echo_rf` / `_focused_sum_rf` and the
        `ReceptionBase` wrappers rely on. ``n_integrations`` is the PE-SDI integration
        count (4 = full I⁴) used by the truncated/complete cores; the conventional
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
        resolved = self._resolve_method(points_m, per_scatterer, focused_sum)
        self._last_method = resolved  # introspection hook (tests / diagnostics)

        if resolved == "conventional":
            conv = self._ensure_conv()
            conv._refresh_sub_elem_attributes()  # resync from shared tx/rx state
            return conv._compute_rf_inner(
                points_m,
                amps,
                n_derivatives=0,  # Field II convention; derivatives in exc/IR chain
                downsampling=downsampling,
                per_scatterer=per_scatterer,
                focused_sum=focused_sum,
            )
        core = self._rf_complete if resolved == "complete" else self._rf_truncated
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

    def _pe_setup(self, points_m, *, n_integrations, per_scatterer, focused_sum, label):
        """Common time-grid, FFT-filter, attenuation and ``inv_jw_pow`` setup."""
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
        show = self.verbose and not focused_sum  # focused_sum is the quiet primitive

        pe_t0, dt, pe_T = self._compute_pe_time_grid(points_m)
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
            print(f"  Attenuation: {att}")

        return {
            "rx_groups": rx_groups,
            "n_out": n_out,
            "pe_t0": pe_t0,
            "dt": dt,
            "pe_T": pe_T,
            "inv_c": np.float32(1.0 / self.c),
            "nfft": nfft,
            "freqs": freqs,
            "fft_v": fft_v,
            "fft_ir_tx": fft_ir_tx,
            "fft_ir_rx": fft_ir_rx,
            "inv_jw_pow": inv_jw_pow,
            "do_attenuation": do_attenuation,
            "distances_pe": distances_pe,
            "scale": np.float32(self.rho / (2.0 * self.c**2)),
            "show": show,
        }

    def _finalize(self, rf, pe_t0, dt, focused_sum, downsampling):
        """Beam-axis ``t0``, coords, and optional anti-aliased decimation."""
        # Subtract the TX focusing bulk so downstream beamforming needs no per-line
        # correction; focused_sum also bakes the RX focus into the line → subtract it too.
        t0 = pe_t0 - float(np.max(self.tx.delays))
        if focused_sum:
            t0 -= float(np.max(self.rx.delays))
        coords = {"t0": t0, "dt": dt}
        if downsampling is not None and int(downsampling) > 1:
            step = int(downsampling)
            rf = _anti_alias_decimate(rf, step)  # anti-aliased along last (time) axis
            coords["dt"] = dt * step
        return rf, coords

    # ------------------------------------------------------------------
    # Method 2 — truncated SDI PE: v_pe ⊛ (I⁴ Δδ_pe)
    # ------------------------------------------------------------------

    def _rf_truncated(
        self,
        points_m,
        amps,
        *,
        n_integrations,
        downsampling,
        per_scatterer,
        focused_sum,
    ):
        """Truncated SDI PE core: place Δδ_pe, apply I⁴ + exc/IR filters in Fourier."""
        s = self._pe_setup(
            points_m,
            n_integrations=n_integrations,
            per_scatterer=per_scatterer,
            focused_sum=focused_sum,
            label="ReceptionSDI [truncated]",
        )
        P = points_m.shape[0]
        nfft, pe_t0, pe_T, dt = s["nfft"], s["pe_t0"], s["pe_T"], s["dt"]
        fft_v, fft_ir_tx, fft_ir_rx = s["fft_v"], s["fft_ir_tx"], s["fft_ir_rx"]
        inv_jw_pow, scale = s["inv_jw_pow"], s["scale"]
        do_attenuation, distances_pe = s["do_attenuation"], s["distances_pe"]
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

            # Fast path: amplitude-accumulate the deltas IN the kernel → one
            # (pe_T,) trace, skipping the (P, pe_T) buffer and the amps@Dh matvec; then
            # one FFT pair with the shared I⁴/exc/IR filters. Attenuation differs per
            # scatterer, so it keeps the per-scatterer path below.
            if not per_scatterer and not do_attenuation:
                delta_sum = compute_pe_sdi_summed(
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
                    amps,
                    s["inv_c"],
                    pe_t0,
                    pe_T,
                    self.fs,
                    dt,
                    tx_eu=self._tx_eu,
                    tx_ev=self._tx_ev,
                    rx_eu=rx_eu,
                    rx_ev=rx_ev,
                )
                H = rfft(delta_sum, n=nfft, workers=-1)
                if inv_jw_pow is not None:
                    H *= inv_jw_pow
                if fft_v is not None:
                    H *= fft_v
                if fft_ir_tx is not None:
                    H *= fft_ir_tx
                if fft_ir_rx is not None:
                    H *= fft_ir_rx
                rf[e_rx, :] = (irfft(H, n=nfft)[:pe_T] * scale).astype(np.float32)
                continue

            delta_pe = compute_pe_sdi(
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
                s["inv_c"],
                pe_t0,
                pe_T,
                self.fs,
                dt,
                tx_eu=self._tx_eu,
                tx_ev=self._tx_ev,
                rx_eu=rx_eu,
                rx_ev=rx_ev,
            )  # (P, pe_T) float32
            H_pe = rfft(delta_pe, n=nfft, axis=1, workers=-1)  # (P, N_freq)
            del delta_pe
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
                    s["freqs"].astype(np.float64),
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
                rf[e_rx, :] = (
                    (rf_pe * amps[:, np.newaxis]).sum(axis=0) * scale
                ).astype(np.float32)
            del rf_pe

        if s["show"]:
            print(
                f"ReceptionSDI [truncated] computed in {time.time() - t_wall:.2f} s\n"
            )
        return self._finalize(rf, pe_t0, dt, focused_sum, downsampling)

    # ------------------------------------------------------------------
    # Method 3 — complete SDI PE: Σ a_i a_j w(t − τ_i − τ_j),  w = I⁴ v_pe
    # ------------------------------------------------------------------

    def _rf_complete(
        self,
        points_m,
        amps,
        *,
        n_integrations,
        downsampling,
        per_scatterer,
        focused_sum,
    ):
        """Complete SDI PE core: precompute ``w = I⁴ v_pe`` once, splat it per pair.

        Exact reference (≡ truncated ≡ conventional): the per-pair circular splat of
        the full-length kernel ``w`` reproduces the truncated path's FFT convolution.
        O(nfft) per pair, so the slowest path — never auto-selected, reference / single
        scatterer use. Attenuation is not supported here (it would need a separate
        integrated kernel per depth); use ``conventional`` or ``truncated`` instead.
        """
        s = self._pe_setup(
            points_m,
            n_integrations=n_integrations,
            per_scatterer=per_scatterer,
            focused_sum=focused_sum,
            label="ReceptionSDI [complete]",
        )
        if s["do_attenuation"]:
            raise NotImplementedError(
                "method='complete' does not support attenuation yet; "
                "use method='conventional' or 'truncated'."
            )
        if s["inv_jw_pow"] is None:
            raise ValueError("method='complete' requires n_integrations > 0 (full I⁴).")

        pe_t0, pe_T, dt = s["pe_t0"], s["pe_T"], s["dt"]
        # w = I⁴ v_pe = irfft( ÷(jω)⁴ · fft_v · fft_ir_tx · fft_ir_rx ) on the pe_T grid
        # — exactly the spectrum the truncated path convolves with Δδ_pe. ÷(jω)⁴ is
        # zero-phase and delocalized, so the per-pair splat must be circular over the
        # full nfft (sliced to pe_T) to match truncated; hence the full-length kernel.
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
            print(f"ReceptionSDI [complete] computed in {time.time() - t_wall:.2f} s\n")
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
            f"freq_power={self.freq_power})"
        )
