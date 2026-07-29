"""ReceptionConventional: pulse-echo RF by direct spatial-impulse-response convolution.

Pulse-echo RF model. Computes the received RF echo from point scatterers with Jensen's
spatial-impulse-response model (J. A. Jensen, "A model for the propagation and scattering
of ultrasound in tissue", J. Acoust. Soc. Am. 89(1), 182-190, 1991), as in Field II. The
SIRs themselves use the Tupholme-Stepanishen formulation (Tupholme, Mathematika 16, 1969;
Stepanishen, J. Acoust. Soc. Am. 49, 1971) with PyField's far-field trapezoidal SIR of
rectangular patches.

This class evaluates the RF equation the direct ("conventional") way — it builds the two
one-way SIRs and convolves them — in contrast to `Reception`, which evaluates the same
equation by sparse delta integration (see that class for the unifying identity chain).
It is the backend `Reception` delegates to for its ``method="fst"`` / ``method="sdi"``
selectors; end users normally reach it through `Reception`, not directly.

Physics. The pulse-echo signal carries the third time-derivative of the excitation,
``v_pe = (ρ₀/2c₀²) E_m ⊛ ∂³v/∂t³``. In a real system that ∂³ is not formed explicitly —
it is absorbed into the band-limited excitation and impulse responses, so the practical
chain is ``e ⊛ h_e ⊛ h_r``. The two-way SIR is the plain convolution
``h_pe = h_tx(r₁→r₅) ⊛_t h_rx(r₅→r₁)`` (no derivatives) and the recorded RF is
``exc ⊛ ir_tx ⊛ ir_rx ⊛ h_pe``. Because the excitation/IR chain already supplies the
three physical derivatives, NO extra explicit ∂/∂t is applied. Field II uses the same
convention — ``calc_scat`` for a unit point equals ``calc_hhp``, the derivatives living
in the pulse and impulse responses — so the RF coincides with Field II (corr≈1.0000).

Public API (output axis ``[emission, reception, Nt]``, channels before time;
``coords["t0"]`` beam-axis referenced):
    pulse_echo_rf          core; (Erx,Nt) summed / (P,Erx,Nt) per_scatterer (PSF)
    sequence_rf            per TX event → (Nev,Erx,Nt)            [in ReceptionBase]
    synthetic_aperture_rf  per element/group DW basis → (Ntx,Erx,Nt)   [base]
    scan_focusline         one focused line, RX summed in-kernel    [base]
    __call__ = pulse_echo_rf

Supports ``"FST"``, ``"sdi"``, and ``"auto"`` SIR methods. See `Reception`
for the faster formulation (PE-SDI kernel carries the 3 derivatives on the SIR for
speed, then integrates them back onto the excitation/IR chain — same result).
"""

import time

import numpy as np
from scipy.fft import irfft, rfft, rfftfreq

from pyfield.utilities.helper_functions import (
    eta_progress as _eta_progress,
    method_to_flag as _method_to_flag,
    next_pow2 as _next_pow2,
    wrap_tqdm as _wrap_tqdm,
)

from ..attenuation import causal_attenuation_tf, compute_reception_distances
from .base import (
    ReceptionBase,
    _warn_if_rx_delays_apods_not_default,
)


class ReceptionConventional(ReceptionBase):
    """Pulse-echo RF by direct two-way SIR convolution (Jensen's model).

    Computes the received echo from point scatterers using the standard SIR
    formulation (model and citations in the module docstring above): the one-way SIRs
    ``h_tx`` and ``h_rx`` are built without differentiation, and the recorded RF is the
    amplitude-weighted sum over scatterers

        rf(t) = (ρ₀/2c₀²) Σ_p σ_p · (exc ⊛ ir_tx ⊛ ir_rx ⊛ h_pe)|_{r_p}

    where the sum runs over scatterers ``p`` at positions ``r_p`` with scattering
    amplitudes ``σ_p`` (the ``amplitudes`` argument), and
    ``h_pe = h_tx(r₁→r_p) ⊛_t h_rx(r_p→r₁)`` is the plain two-way SIR for that
    scatterer. The three physical excitation derivatives are carried by the
    band-limited exc/IR chain, so no extra explicit ∂/∂t is applied (Field II's
    ``calc_scat`` ≡ ``calc_hhp`` convention). Public API via `pulse_echo_rf` (core)
    plus `sequence_rf` / `synthetic_aperture_rf` / `scan_focusline` from `ReceptionBase`.

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
    method : str, default "auto"
        SIR computation method: ``"FST"``, ``"sdi"``, or ``"auto"``.
    n_depth_bins : "auto" or int, default "auto"
        Pulse-echo speed knob (result is unchanged). Scatterers at different depths
        echo at different times, so one FFT over all of them must span the whole
        depth range; grouping them into depth bins lets each bin use an FFT only as
        long as its own arrival window — a big speedup at high scatterer counts.
        Bins recombine sample-exactly on a shared time axis. ``"auto"`` picks the
        count from the scatterer number and depth spread (~100 scatterers per bin);
        an int forces it; ``1`` disables binning.
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
        "method": (str, "SIR method: FST / sdi / auto"),
        "n_depth_bins": ((int, str), "Pulse-echo depth bins: 'auto' or int"),
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
        self.method = method
        self.n_depth_bins = n_depth_bins
        self.verbose = verbose
        self._refresh_sub_elem_attributes()
        _warn_if_rx_delays_apods_not_default(self.rx)

    # ------------------------------------------------------------------
    # Backend-specific helpers
    # ------------------------------------------------------------------

    def _compute_sir_time_grids(self, points_m):
        """Separate TX and RX SIR sample grids.

        Returns ``(time_grid_tx, t0_tx, dt, T_tx, time_grid_rx, t0_rx, T_rx)``.
        """
        time_grid_tx, t0_tx, dt, T_tx = self._oneway_time_grid(points_m, "tx")
        time_grid_rx, t0_rx, _, T_rx = self._oneway_time_grid(points_m, "rx")
        return time_grid_tx, t0_tx, dt, T_tx, time_grid_rx, t0_rx, T_rx

    # ------------------------------------------------------------------
    # Depth-binned fast path (Field II-style bounded windows)
    # ------------------------------------------------------------------

    def _lattice_grid(self, pts, aperture, t0_global):
        """SIR sample grid for ``pts`` on ``aperture`` snapped to the global lattice.

        Returns ``(grid, n0, T)``: ``grid`` starts at ``t0_global + n0·dt`` so every
        bin shares one lattice and per-bin results add at integer offset ``n0``.
        """
        dt = 1.0 / self.fs
        _, t0, _, T = self._oneway_time_grid(pts, aperture)
        # Snap the bin's start onto the global lattice; +1 sample covers the snap
        # (the sub-sample remainder is < dt, so it never adds a second sample).
        n0, t0_snap, _shift = self._snap_to_lattice(t0, t0_global, dt)
        T += 1
        grid = (t0_snap + np.arange(T) * dt).astype(np.float32)
        return grid, n0, T

    def _fast_rf_binned(
        self, points_m, amps, rx_groups, n_out, method_flag, *, n_bins, downsampling
    ):
        """Pulse-echo RF, split into depth bins for short per-bin FFTs.

        Same physical result as the inline fast path (the full RF summed over
        scatterers), but scatterers are grouped by depth so each bin spans a tight
        arrival-time window and uses a smaller FFT. All bins share one global sample
        lattice (`_lattice_grid`), so each bin's RF simply adds back at an integer sample
        offset — no resampling. Used only when scatterers are summed
        (``per_scatterer=False``) and attenuation is off.
        """
        dt = 1.0 / self.fs
        inv_c = np.float32(1.0 / self.c)
        scale = np.float32(self.rho / (2.0 * self.c**2) * dt)
        exc = self._resolve_excitation()
        ir_tx = getattr(self.tx, "impulse_response", None)
        ir_rx = getattr(self.rx, "impulse_response", None)
        L = len(exc) if exc is not None else 0
        # Global lattice origin (also the reported t0); shared by every bin.
        with self._timer("time_grid_s"):
            t0_tx_g = self._oneway_time_grid(points_m, "tx")[1]
            t0_rx_g = self._oneway_time_grid(points_m, "rx")[1]

        def per_bin(idx):
            pts, am = points_m[idx], amps[idx]
            Pb = pts.shape[0]
            with self._timer("time_grid_s"):
                grid_t, n0t, Tt = self._lattice_grid(pts, "tx", t0_tx_g)
                grid_r, n0r, Tr = self._lattice_grid(pts, "rx", t0_rx_g)
            peTb = Tt + Tr - 1
            nfft = _next_pow2(peTb + L)
            with self._timer("fft_s"):
                fft_v = (
                    rfft(exc.astype(np.float64), n=nfft, workers=-1).astype(
                        np.complex64
                    )
                    if exc is not None
                    else 1
                )
                fft_ir = [
                    rfft(np.asarray(ir, np.float64), n=nfft, workers=-1).astype(
                        np.complex64
                    )
                    for ir in (ir_tx, ir_rx)
                    if ir is not None
                ]
            h_tx, _ = self._timed_h_sir(
                Pb,
                self._tx_M,
                Tt,
                dt,
                grid_t,
                pts,
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
            )
            with self._timer("fft_s"):
                H_tx = rfft(h_tx, n=nfft, axis=1, workers=-1)
            del h_tx
            rf_bin = np.zeros((n_out, peTb), dtype=np.float32)
            for e_rx in range(n_out):
                rx_c, rx_wx, rx_wy, rx_ap, rx_dl, rx_eu, rx_ev = rx_groups[e_rx]
                h_rx_e, _ = self._timed_h_sir(
                    Pb,
                    rx_c.shape[0],
                    Tr,
                    dt,
                    grid_r,
                    pts,
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
                )
                with self._timer("fft_s"):
                    H_rx_e = rfft(h_rx_e, n=nfft, axis=1, workers=-1)
                    del h_rx_e
                    H = (
                        am @ (H_tx * H_rx_e)
                    ) * fft_v  # sum scatterers, then exc filter
                    for f in fft_ir:
                        H *= f
                    rf_bin[e_rx] = (irfft(H, n=nfft)[:peTb] * scale).astype(np.float32)
            return rf_bin, n0t + n0r

        rf = self._accumulate_depth_bins(points_m, n_bins, per_bin)
        if self.verbose:
            print(
                f"Reception [{n_bins} depth bins] computed ({self._fmt_time_log()})\n"
            )
        # Beam-axis t0 = global pe origin minus the TX focusing bulk (no RX focus here).
        return self._finalize(rf, t0_tx_g + t0_rx_g, dt, False, downsampling)

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
        """Shared computation core for pulse_echo_rf (and the mixin wrappers).

        Builds the two one-way SIRs ``h_tx``/``h_rx`` and convolves them (no extra
        explicit ∂/∂t — the three physical derivatives are carried by the band-limited
        ``exc ⊛ ir_tx ⊛ ir_rx`` chain, Field II's ``calc_scat`` ≡ ``calc_hhp``).

        Parameters
        ----------
        points_m : (P, 3) numpy.ndarray
            Scatterer positions in metres.
        amps : (P,) numpy.ndarray
            Scattering amplitudes (float32).
        downsampling : int or None, default None
            Downsample output by this factor.
        per_scatterer : bool, default False
            If True return ``(P, E_rx, Nt)``. If False return ``(E_rx, Nt)``.
        focused_sum : bool, default False
            If True, beamform on receive INSIDE the SIR: ``h_rx`` is computed over
            ALL RX patches at once (carrying their focusing ``rx.delays`` +
            ``rx.apodization``), so the patch sum is the focused, apodized receive
            line — Field II ``calc_scat``'s internal receive beamforming. Output
            collapses the RX axis to 1 (``(1, Nt)``), needing no external DAS.
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
        P = points_m.shape[0]
        n_rx = int(self.rx.delays.shape[0])
        rx_groups = self._rx_groups(focused_sum)
        n_out = len(rx_groups)
        show = (
            self.verbose and not focused_sum
        )  # focused_sum is the quiet loop primitive
        method_flag = _method_to_flag(self.method)

        # Per-element excitation (L, E): one pulse per TX element. It must be folded into
        # each element's partial SIR, which the inline path below does; the depth-binned
        # fast path builds one combined h_tx, so it is skipped for this mode.
        exc = self._resolve_excitation()
        n_tx = int(self.tx.delays.shape[0])
        per_elem_exc = exc is not None and exc.ndim == 2
        if per_elem_exc and exc.shape[1] != n_tx:
            raise ValueError(
                f"Per-element excitation must have shape (L, E={n_tx}), got {exc.shape}."
            )

        # Depth-binned fast path: per-element RF, no attenuation. Bounded per-bin
        # time grids → smaller nfft → big speedup at high scatterer counts. Bin count
        # from self.n_depth_bins ("auto" or int).
        if (
            not per_scatterer
            and not focused_sum
            and self.alpha0 is None
            and not per_elem_exc
        ):
            n_bins = self.n_depth_bins
            n_bins = (
                self._auto_depth_bins(points_m, n_out)
                if n_bins == "auto"
                else int(n_bins)
            )
            if n_bins > 1:
                return self._fast_rf_binned(
                    points_m,
                    amps,
                    rx_groups,
                    n_out,
                    method_flag,
                    n_bins=n_bins,
                    downsampling=downsampling,
                )

        with self._timer("time_grid_s"):
            time_grid_tx, t0_tx, dt, T_tx, time_grid_rx, t0_rx, T_rx = (
                self._compute_sir_time_grids(points_m)
            )
        inv_c = np.float32(1.0 / self.c)

        # PE time axis: convolution of h_tx (T_tx) and h_rx (T_rx).
        pe_t0 = t0_tx + t0_rx
        pe_T = T_tx + T_rx - 1

        ir_tx = getattr(self.tx, "impulse_response", None)
        ir_rx = getattr(self.rx, "impulse_response", None)
        # exc / per_elem_exc resolved above (needed for the fast-path guard). Per-element
        # exc is folded into each TX element's partial SIR in the H_tx build below; a
        # global pulse joins the shared IRs in fft_v.

        L = exc.shape[0] if exc is not None else 0
        nfft = _next_pow2(pe_T + L)
        freqs = rfftfreq(nfft, d=1.0 / self.fs)  # (N_freq,) float64 for precision

        def _rfft64(sig):
            return rfft(np.asarray(sig, dtype=np.float64), n=nfft, workers=-1).astype(
                np.complex64
            )

        # ir_tx and ir_rx are shared by every patch, so they stay in the post-sum filter
        # for both excitation modes; only a global excitation joins them in fft_v.
        fft_ir_tx = _rfft64(ir_tx) if ir_tx is not None else 1
        fft_ir_rx = _rfft64(ir_rx) if ir_rx is not None else 1
        fft_v = _rfft64(exc) if (exc is not None and not per_elem_exc) else 1
        post_filter = fft_v * fft_ir_tx * fft_ir_rx  # (N_freq,) complex64 or scalar 1

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

        # The dt restores the continuous SIR convolution: irfft(H_tx·H_rx) is the discrete
        # sum Σ_k h_tx[k]h_rx[n-k], but (h_tx ⊛ h_rx)(t) = ∫h_tx(τ)h_rx(t-τ)dτ ≈ dt·Σ_k —
        # the two differ by one dt. (The SDI delta-train form is intrinsically the
        # continuous convolution, so without this dt the two would differ by fs = 1/dt.)
        scale = np.float32(self.rho / (2.0 * self.c**2) * dt)

        if show:
            print("\n--- Reception (conventional) ---")
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

        # Build the transmit SIR spectrum H_tx (P, N_freq). FFT in float32 (→complex64):
        # the SIR is float32 already and the exc/IR band-pass tolerates it (matches
        # the SDI cores). Upcasting to float64 here doubled the dominant FFT cost for
        # negligible accuracy gain.
        if per_elem_exc:
            # One pulse per TX element: sum each element's SIR spectrum weighted by its
            # own excitation FFT, H_tx = Σ_e FFT(h_tx_e)·FFT(exc[:, e]).
            tx_groups = self._group_patches_by_element(
                n_tx,
                self._tx_sub_el_idx,
                (
                    self._tx_centers,
                    self._tx_wx,
                    self._tx_wy,
                    self._tx_apod,
                    self._tx_delays,
                    self._tx_eu,
                    self._tx_ev,
                ),
            )
            H_tx = np.zeros((P, freqs.shape[0]), dtype=np.complex64)
            for e in range(n_tx):
                c_e, wx_e, wy_e, ap_e, dl_e, eu_e, ev_e = tx_groups[e]
                if c_e.shape[0] == 0:
                    continue
                h_e, _ = self._timed_h_sir(
                    P,
                    c_e.shape[0],
                    T_tx,
                    dt,
                    time_grid_tx,
                    points_m,
                    c_e,
                    wx_e,
                    wy_e,
                    inv_c,
                    self.fs,
                    ap_e,
                    dl_e,
                    method_flag,
                    eu_e,
                    ev_e,
                )  # (P, T_tx) float32
                with self._timer("fft_s"):
                    H_tx += rfft(h_e, n=nfft, axis=1, workers=-1) * _rfft64(exc[:, e])
                del h_e
        else:
            # Compute h_tx once for all scatterers from every TX patch: (P, T_tx).
            h_tx, _ = self._timed_h_sir(
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
            with self._timer("fft_s"):
                H_tx = rfft(h_tx, n=nfft, axis=1, workers=-1)  # (P, N_freq) complex64
            del h_tx

        rf = np.zeros(
            (P, n_out, pe_T) if per_scatterer else (n_out, pe_T), dtype=np.float32
        )

        el_iter = (
            _wrap_tqdm(range(n_out), desc="RX elements", total=n_out, leave=True)
            if show
            else range(n_out)
        )
        # ETA + in-place progress only when the projected run exceeds ~30 s
        # (tqdm already shows progress in verbose mode).
        el_iter = _eta_progress(el_iter, n_out, label="RX elements", progress=not show)

        for e_rx in el_iter:
            # rx_ap / rx_dl carry this group's RX apodization + delays. Per element
            # (focused_sum=False) they shift/scale that element's trace and are NOT
            # summed across elements (raw per-channel RF — see the warning). With
            # focused_sum the single group holds every RX patch, so the patch sum
            # is one focusing-delayed, apodized beamformed line.
            rx_c, rx_wx, rx_wy, rx_ap, rx_dl, rx_eu, rx_ev = rx_groups[e_rx]
            M_e = rx_c.shape[0]

            # Compute h_rx for this element's patches: (P, T_rx).
            h_rx_e, _ = self._timed_h_sir(
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
            # float32 FFT (→complex64); see the h_tx note above.
            with self._timer("fft_s"):
                H_rx_e = rfft(h_rx_e, n=nfft, axis=1, workers=-1)
                del h_rx_e
                H_pe = H_tx * H_rx_e
                del H_rx_e

                # Fast path: the shared exc/IR filter and irfft linearity let us sum the P
                # scatterers in the frequency domain FIRST, then run one irfft instead of
                # P: Σ_p a_p·irfft(H_p·F) = irfft((Σ_p a_p·H_p)·F). Attenuation differs per
                # scatterer, so it stays on the per-scatterer path below.
                if not per_scatterer and not do_attenuation:
                    H_sum = amps @ H_pe  # (N_freq,) — matvec
                    del H_pe
                    H_sum = H_sum * post_filter
                    rf[e_rx, :] = (irfft(H_sum, n=nfft)[:pe_T] * scale).astype(
                        np.float32
                    )
                    continue

                # Apply the post-sum filter (global exc and/or IRs) and attenuation.
                # post_filter is scalar 1 when none are set; it broadcasts against H_pe's
                # (P, N_freq) either way, so no explicit newaxis is needed.
                H_pe *= post_filter
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
                rf[:, e_rx, :] = (rf_pe * amps[:, np.newaxis] * scale).astype(
                    np.float32
                )
            else:
                rf_channel = (rf_pe * amps[:, np.newaxis]).sum(axis=0)
                rf[e_rx, :] = (rf_channel * scale).astype(np.float32)
            del rf_pe

        if show:
            print(
                f"Reception computed in {time.time() - t_wall:.2f} s "
                f"({self._fmt_time_log()})\n"
            )

        return self._finalize(rf, pe_t0, dt, focused_sum, downsampling)

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

        The core reception primitive. The recorded pulse-echo signal is the
        amplitude-weighted superposition of each scatterer's pulse-echo response::

            rf = (ρ₀/2c₀²) Σ_i a_i (exc ⊛ ir_tx ⊛ ir_rx ⊛ h_pe)|_{r_i}

        The three physical excitation derivatives are carried by the band-limited
        ``exc ⊛ ir_tx ⊛ ir_rx`` chain, so NO extra explicit ∂/∂t is applied. Field II
        uses the same convention, so this equals its ``calc_scat`` (≡ ``calc_hhp`` for a
        unit point, corr 1.0000).

        ``per_scatterer=True`` keeps each scatterer separate — the point-spread
        function (PSF) use; ``False`` sums them (the recorded echo).

        ``coords["t0"]`` is referenced to the beam axis (the TX focusing bulk
        ``tx.delays.max()`` is subtracted), so downstream beamforming needs no
        per-line bulk correction.

        Parameters
        ----------
        scatterer_positions_mm : (N_scat, 3) numpy.ndarray
            Scatterer positions in mm.
        amplitudes : (N_scat,) numpy.ndarray or None, default None
            Scattering coefficient at each position. None defaults to ones.
        per_scatterer : bool, default False
            If False, sum over scatterers → ``(Erx, Nt)``. If True, keep each
            scatterer separate → ``(N_scat, Erx, Nt)`` (PSF per point).
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
        ``focused_sum=True`` (the exc/IR chain carries the physical derivatives).
        Returns the single line ``(Nt,)`` and its coords. Field II's ``calc_scat``
        builds its line the same way (focused, apodized, summed on receive).
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
            f"fs={self.fs} Hz, alpha0={self.alpha0}, method='{self.method}')"
        )
