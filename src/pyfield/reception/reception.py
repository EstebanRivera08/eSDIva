"""Reception: pulse-echo RF simulation by direct spatial-impulse-response convolution.

Pulse-echo RF model. Computes the received RF echo from point scatterers with Jensen's
spatial-impulse-response model (J. A. Jensen, "A model for the propagation and scattering
of ultrasound in tissue", J. Acoust. Soc. Am. 89(1), 182-190, 1991), as in Field II. The
SIRs themselves use the Tupholme-Stepanishen formulation (Tupholme, Mathematika 16, 1969;
Stepanishen, J. Acoust. Soc. Am. 49, 1971) with PyField's far-field trapezoidal SIR of
rectangular patches.

This class evaluates the RF equation the direct ("conventional") way — it builds the two
one-way SIRs and convolves them — in contrast to `ReceptionSDI`, which evaluates the same
equation by sparse delta integration (see that class for the unifying identity chain).

Physics. The pulse-echo signal carries the third time-derivative of the excitation,
``v_pe = (ρ₀/2c₀²) E_m ⊛ ∂³v/∂t³``. In a real system that ∂³ is not formed explicitly —
it is absorbed into the band-limited excitation and impulse responses, so the practical
chain is ``e ⊛ h_e ⊛ h_r``. The two-way SIR is the plain convolution
``h_pe = h_tx(r₁→r₅) ⊛_t h_rx(r₅→r₁)`` (no derivatives) and the recorded RF is
``exc ⊛ ir_tx ⊛ ir_rx ⊛ h_pe``. Because the excitation/IR chain already supplies the
three physical derivatives, NO extra explicit ∂/∂t is applied (``n_derivatives=0``);
applying the textbook ``∂³`` on top would double them. The optional ``n_derivatives``
knob exposes that explicit form (e.g. 3 = the textbook Born ``∂³`` for a
derivative-free excitation).

Field II parallel (for adoption): Field II uses the same convention — ``calc_hhp``
and ``calc_scat`` apply no round-trip ∂/∂t (``calc_scat`` for a unit point equals
``calc_hhp``), the derivatives living in the pulse and impulse responses — so the
``n_derivatives=0`` RF coincides with Field II (corr≈1.0000).

Public API (output axis ``[emission, reception, Nt]``, channels before time;
``coords["t0"]`` beam-axis referenced):
    pulse_echo_rf          core; (Erx,Nt) summed / (P,Erx,Nt) per_scatterer (PSF)
    sequence_rf            per TX event → (Nev,Erx,Nt)            [in ReceptionBase]
    synthetic_aperture_rf  per element/group DW basis → (Ntx,Erx,Nt)   [base]
    scan_focusline         one focused line, RX summed in-kernel    [base]
    __call__ = pulse_echo_rf

Supports ``"naive"``, ``"sdi"``, and ``"auto"`` SIR methods. See `ReceptionSDI`
for the faster formulation (PE-SDI kernel carries the 3 derivatives on the SIR for
speed, then integrates them back onto the excitation/IR chain — same result).
"""

import time

import numpy as np
from scipy.fft import irfft, rfft, rfftfreq

from pyfield.hsir.transducer_sir import compute_h_sir
from pyfield.utilities.helper_functions import compute_time_grid

from ..attenuation import causal_attenuation_tf, compute_reception_distances
from .base import (
    ReceptionBase,
    _anti_alias_decimate,
    _next_pow2,
    _warn_if_rx_delays_apods_not_default,
    _wrap_tqdm,
)


def _method_to_flag(method):
    if method == "naive":
        return 0
    if method in ("sdi", "SDI"):
        return 1
    return None  # auto


# Pulse-echo fast path splits scatterers into depth bins (see _auto_depth_bins).
# Best ≈ this many scatterers per bin; override per instance with n_depth_bins.
_SCATTERERS_PER_BIN = 200


class Reception(ReceptionBase):
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
    band-limited exc/IR chain, so no extra explicit ∂/∂t is applied
    (``n_derivatives=0``); this is Field II's convention (``calc_scat`` ≡ ``calc_hhp``
    for a point). Public API via `pulse_echo_rf` (core) plus `sequence_rf` /
    `synthetic_aperture_rf` / `scan_focusline` from `ReceptionBase`.

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
    n_depth_bins : "auto" or int, default "auto"
        Pulse-echo speed knob. Scatterers are grouped into this many depth bins so
        each bin uses a short FFT (big speedup at high scatterer counts). ``"auto"``
        sizes it automatically; pass an int to tune for your CPU/scatterer count.
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
        self.method = method
        self.n_depth_bins = n_depth_bins
        self.verbose = verbose
        self._refresh_sub_elem_attributes()
        _warn_if_rx_delays_apods_not_default(self.rx)

    # ------------------------------------------------------------------
    # Backend-specific helpers
    # ------------------------------------------------------------------

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
    # Depth-binned fast path (Field II-style bounded windows)
    # ------------------------------------------------------------------

    def _auto_depth_bins(self, points_m, n_out):
        """Number of depth bins for the fast path (1 = no binning).

        Scatterers at very different depths echo at very different times, so a single
        FFT must span the whole arrival spread (a long, mostly-empty time grid). Grouping
        them by depth lets each bin use a short grid (small FFT), which dominates the
        cost. The count balances two needs: keep each bin's grid short (≈ arrival-time
        spread in samples / 128) and keep its scatterer batch cache-resident
        (≈ ``_SCATTERERS_PER_BIN`` scatterers/bin).
        """
        P = points_m.shape[0]
        if P < 128 or n_out < 2:
            return 1
        center = np.asarray(self._tx_centers, dtype=np.float64).mean(axis=0)
        arrival = 2.0 * np.linalg.norm(points_m - center, axis=1) / self.c
        spread = float(arrival.max() - arrival.min()) * self.fs  # samples
        n_bins = max(round(spread / 128), round(P / _SCATTERERS_PER_BIN))
        return max(1, min(n_bins, P // 128))  # keep ≥128 scatterers/bin

    def _lattice_grid(self, pts, M, centers, wx_max, wy_max, delays, t0_global):
        """Time grid for ``pts`` snapped to the global sample lattice.

        Returns ``(grid, n0, T)``: ``grid`` starts at ``t0_global + n0·dt`` so every
        bin shares one lattice and per-bin results add at integer offset ``n0``.
        """
        dt = 1.0 / self.fs
        _, t0, _, T = compute_time_grid(
            pts.shape[0],
            M,
            pts,
            centers,
            wx_max,
            wy_max,
            self.c,
            self.fs,
            delays,
            verbose=False,
        )
        n0 = int(np.floor((t0 - t0_global) / dt))
        T += int(round((t0 - (t0_global + n0 * dt)) / dt)) + 1
        grid = (t0_global + (n0 + np.arange(T)) * dt).astype(np.float32)
        return grid, n0, T

    def _fast_rf_binned(
        self,
        points_m,
        amps,
        rx_groups,
        n_out,
        method_flag,
        *,
        n_derivatives,
        n_bins,
        downsampling,
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
        _, t0_tx_g, _, _ = compute_time_grid(
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
        _, t0_rx_g, _, _ = compute_time_grid(
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

        center = np.asarray(self._tx_centers, dtype=np.float64).mean(axis=0)
        order = np.argsort(np.linalg.norm(points_m - center, axis=1))  # by depth

        results = []  # (rf_bin, offset)
        for idx in np.array_split(order, n_bins):
            if idx.size == 0:
                continue
            pts, am = points_m[idx], amps[idx]
            Pb = pts.shape[0]

            grid_t, n0t, Tt = self._lattice_grid(
                pts,
                self._tx_M,
                self._tx_centers,
                self._tx_wx_max,
                self._tx_wy_max,
                self.tx.delays,
                t0_tx_g,
            )
            grid_r, n0r, Tr = self._lattice_grid(
                pts,
                self._rx_M,
                self._rx_centers,
                self._rx_wx_max,
                self._rx_wy_max,
                self.rx.delays,
                t0_rx_g,
            )

            peTb = Tt + Tr - 1
            nfft = _next_pow2(peTb + L)
            jw_pow = (1j * 2.0 * np.pi * rfftfreq(nfft, d=dt)) ** n_derivatives
            fft_v = (
                rfft(exc.astype(np.float64), n=nfft, workers=-1) * jw_pow
                if exc is not None
                else jw_pow
            ).astype(np.complex64)
            fft_ir = [
                rfft(np.asarray(ir, np.float64), n=nfft, workers=-1).astype(
                    np.complex64
                )
                for ir in (ir_tx, ir_rx)
                if ir is not None
            ]

            h_tx, _ = compute_h_sir(
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
            H_tx = rfft(h_tx, n=nfft, axis=1, workers=-1)
            del h_tx

            rf_bin = np.zeros((n_out, peTb), dtype=np.float32)
            for e_rx in range(n_out):
                rx_c, rx_wx, rx_wy, rx_ap, rx_dl, rx_eu, rx_ev = rx_groups[e_rx]
                h_rx_e, _ = compute_h_sir(
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
                H_rx_e = rfft(h_rx_e, n=nfft, axis=1, workers=-1)
                del h_rx_e
                H = (am @ (H_tx * H_rx_e)) * fft_v  # sum scatterers, then exc filter
                for f in fft_ir:
                    H *= f
                rf_bin[e_rx] = (irfft(H, n=nfft)[:peTb] * scale).astype(np.float32)
            results.append((rf_bin, n0t + n0r))

        nt_total = max(off + r.shape[1] for r, off in results)
        rf = np.zeros((n_out, nt_total), dtype=np.float32)
        for r, off in results:
            rf[:, off : off + r.shape[1]] += r

        t0 = (t0_tx_g + t0_rx_g) - float(np.max(self.tx.delays))  # beam-axis ref
        coords = {"t0": t0, "dt": dt}
        if downsampling is not None and int(downsampling) > 1:
            step = int(downsampling)
            rf = _anti_alias_decimate(rf, step)
            coords["dt"] = dt * step
        return rf, coords

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _compute_rf_inner(
        self,
        points_m,
        amps,
        *,
        n_derivatives=3,
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
        n_derivatives : int, default 3
            Number of EXTRA explicit temporal derivatives applied in the frequency
            domain via ``(jω)^n``, on top of the exc/IR chain. 0 = the practical
            pulse-echo RF: the three physical derivatives are already carried by
            ``exc ⊛ ir_tx ⊛ ir_rx``, so the bare convolution is taken (Field II
            ``calc_scat`` ≡ ``calc_hhp``). 3 = the textbook Born ``∂³`` form, valid
            only for a derivative-free excitation (would double-differentiate the
            band-limited exc/IR used here).
        downsampling : int or None, default None
            Downsample output by this factor.
        per_scatterer : bool, default False
            If True return ``(P, Nt, E_rx)``. If False return ``(Nt, E_rx)``.
        focused_sum : bool, default False
            If True, beamform on receive INSIDE the SIR: ``h_rx`` is computed over
            ALL RX patches at once (carrying their focusing ``rx.delays`` +
            ``rx.apodization``), so the patch sum is the focused, apodized receive
            line — Field II ``calc_scat``'s internal receive beamforming. Output
            collapses the RX axis to 1 (``(1, Nt)``), needing no external DAS.
            Mutually exclusive with ``per_scatterer``.

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
        # focused_sum: one group = all RX patches (their sum is the beamformed
        # line). otherwise: one group per RX element → per-channel RF.
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
        method_flag = _method_to_flag(self.method)

        # Depth-binned fast path: per-element RF, no attenuation. Bounded per-bin
        # time grids → smaller nfft → big speedup at high scatterer counts. Bin count
        # from self.n_depth_bins ("auto" or int).
        if not per_scatterer and not focused_sum and self.alpha0 is None:
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
                    n_derivatives=n_derivatives,
                    n_bins=n_bins,
                    downsampling=downsampling,
                )

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

        # Extra explicit derivative factor in freq domain: (j*2*pi*f)^n.
        # n=0 → exc/IR chain already carries the 3 physical derivatives ((jω)^0=1);
        # n=3 → textbook Born ∂³ (derivative-free excitation only).
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
            # focused_sum collapses the RX axis, so use one aperture-centroid path.
            if focused_sum:
                rx_elem_centers_m = rx_elem_centers_m.mean(axis=0, keepdims=True)
            distances_pe = compute_reception_distances(
                points_m.astype(np.float64), tx_center_m, rx_elem_centers_m
            )  # (P, n_out)

        # Continuous-convolution dt factor for h_tx ⊛ h_rx.
        # H_pe = irfft(H_tx · H_rx) is a *discrete* convolution Σ_k h_tx[k] h_rx[n-k],
        # whereas the physical SIR convolution is the continuous integral
        # (h_tx ⊛ h_rx)(t) = ∫ h_tx(τ) h_rx(t-τ) dτ ≈ dt · Σ_k. The two differ by
        # one factor of dt. The excitation/IR convolutions are also freq-domain
        # products here, but `ReceptionSDI` forms them the same way, so they cancel
        # in any cross-check; only the SIR-SIR convolution differs — `ReceptionSDI`
        # builds it from a Dirac-delta train (δ⊛δ → weights multiply, intrinsically
        # the continuous convolution). Without this dt, conventional `Reception` is
        # larger than `ReceptionSDI` by exactly fs = 1/dt (verified fc-independent).
        scale = np.float32(self.rho / (2.0 * self.c**2) * dt)

        if show:
            mode = (
                "pulse-echo RF (exc/IR carry ∂³; calc_scat ≡ calc_hhp)"
                if n_derivatives == 0
                else f"+{n_derivatives} explicit ∂/∂t (textbook Born = 3)"
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

        # FFT the SIR in float32 (→complex64): the SIR is float32 already and the
        # exc/IR band-pass tolerates it (matches ReceptionSDI, which also FFTs in
        # float32). Upcasting to float64 here doubled the FFT cost — the dominant
        # term — for negligible accuracy gain.
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

        for e_rx in el_iter:
            # rx_ap / rx_dl carry this group's RX apodization + delays. Per element
            # (focused_sum=False) they shift/scale that element's trace and are NOT
            # summed across elements (raw per-channel RF — see the warning). With
            # focused_sum the single group holds every RX patch, so the patch sum
            # is one focusing-delayed, apodized beamformed line.
            rx_c, rx_wx, rx_wy, rx_ap, rx_dl, rx_eu, rx_ev = rx_groups[e_rx]
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
            # float32 FFT (→complex64); see the h_tx note above.
            H_rx_e = rfft(h_rx_e, n=nfft, axis=1, workers=-1)
            del h_rx_e
            H_pe = H_tx * H_rx_e
            del H_rx_e

            # Fast path: only a per-element sum over scatterers is returned and the
            # exc/IR filters are identical across scatterers, so collapse the P
            # scatterers FIRST — weighted sum of the SIR-product spectra — then run
            # ONE irfft instead of P. irfft is linear and the filters are shared, so
            # Σ_p a_p·irfft(H_p·F) = irfft((Σ_p a_p·H_p)·F) exactly (float reordering
            # only). Attenuation differs per scatterer, so it keeps the per-scatterer
            # path below. (The forward SIR FFTs stay per-scatterer: each scatterer has
            # its own h_tx/h_rx.)
            if not per_scatterer and not do_attenuation:
                H_sum = amps @ H_pe  # (N_freq,) — BLAS matvec
                del H_pe
                H_sum = H_sum * fft_v_pe
                if fft_ir_tx is not None:
                    H_sum *= fft_ir_tx
                if fft_ir_rx is not None:
                    H_sum *= fft_ir_rx
                rf[e_rx, :] = (irfft(H_sum, n=nfft)[:pe_T] * scale).astype(np.float32)
                continue

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
                rf[:, e_rx, :] = (rf_pe * amps[:, np.newaxis] * scale).astype(
                    np.float32
                )
            else:
                rf_channel = (rf_pe * amps[:, np.newaxis]).sum(axis=0)
                rf[e_rx, :] = (rf_channel * scale).astype(np.float32)
            del rf_pe

        if show:
            print(f"Reception computed in {time.time() - t_wall:.2f} s\n")

        # t0 referenced to the beam axis: subtract the TX focusing bulk
        # (delays.max(), the centre/last-firing element delay) so downstream
        # beamforming needs no per-line bulk correction. See pulse_echo_rf.
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

        The core reception primitive. The recorded pulse-echo signal is the
        amplitude-weighted superposition of each scatterer's pulse-echo response::

            rf = (ρ₀/2c₀²) Σ_i a_i (exc ⊛ ir_tx ⊛ ir_rx ⊛ h_pe)|_{r_i}

        The three physical excitation derivatives are carried by the band-limited
        ``exc ⊛ ir_tx ⊛ ir_rx`` chain, so NO extra explicit ∂/∂t is applied
        (``n_derivatives=0``); applying the textbook ``∂³`` on top would
        double-differentiate (the discarded "3 Born derivatives" convention
        inflated the RF by ~(jω)³≈10²²). Field II uses the same convention, so this
        equals its ``calc_scat`` (≡ ``calc_hhp`` for a unit point, corr 1.0000).

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
            n_derivatives=0,
            downsampling=downsampling,
            per_scatterer=per_scatterer,
        )

    def _focused_sum_rf(self, points_m, amps, *, downsampling=None):
        """Receive-beamformed line via in-kernel focused sum.

        Backend hook for `ReceptionBase.scan_focusline`: the pulse-echo RF
        (``n_derivatives=0`` — no extra explicit ∂/∂t, the exc/IR chain carries
        the physical derivatives) with ``focused_sum=True``. Returns the single
        line ``(Nt,)`` and its coords. Field II's ``calc_scat`` builds its line
        the same way (focused, apodized, summed on receive).
        """
        rf, coords = self._compute_rf_inner(
            points_m,
            amps,
            n_derivatives=0,
            downsampling=downsampling,
            focused_sum=True,
        )
        return rf[0], coords

    def __repr__(self) -> str:
        return (
            f"Reception(tx={self.tx}, rx={self.rx}, c={self.c} m/s, "
            f"fs={self.fs} Hz, alpha0={self.alpha0}, method='{self.method}')"
        )
