"""ReceptionPaired: pulse-echo RF by the paired SDI form — pedagogic reference.

The two-way kernel of one (TX patch, RX patch) pair is the convolution of their corner-delta
trains, ``Δδ_pe = D²h_tx ⊛ D²h_rx`` (16 deltas). Pushing the four integrations onto the
drive once, ``w = I⁴ v_pe``, the RF is ``Σ_pairs Σ_16 a_i a_j w(t − τ_i − τ_j)``: no FFT, no
cumulative sum. Exact but quadratic in the patch count, so `Reception` (spectral, linear
cost) is the production path; this class exists for teaching and cross-checks.
"""

import time
import warnings

import numpy as np
from scipy.fft import irfft

from esdiva.hsir.sir_paired import compute_pe_complete
from esdiva.utilities.helper_functions import (
    eta_progress as _eta_progress,
    wrap_tqdm as _wrap_tqdm,
)

from .reception import Reception


class ReceptionPaired(Reception):
    """Pulse-echo RF via the paired SDI form (exact, O(M_tx·M_rx), no attenuation).

    Same constructor and public methods as `Reception`; only the RF core differs.
    Warns on construction because it is far slower than `Reception`.
    """

    def __init__(self, tx, rx, **kwargs):
        warnings.warn(
            "ReceptionPaired is a pedagogic reference: exact but O(M²) per patch pair, "
            "far slower than Reception. Use it only for teaching or cross-checks.",
            UserWarning,
            stacklevel=2,
        )
        super().__init__(tx, rx, **kwargs)
        self.method = "paired"  # informational: the RF core is fixed

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
        if focused_sum and per_scatterer:
            raise ValueError("focused_sum and per_scatterer are mutually exclusive.")
        exc = self._resolve_excitation()
        if exc is not None and exc.ndim == 2:
            raise NotImplementedError(
                "Per-element excitation (L, E) is not supported by ReceptionPaired; "
                "use Reception."
            )
        self._reset_time_log()
        self._last_method = "paired"
        return self._rf_paired(
            points_m,
            amps,
            n_integrations=n_integrations,
            downsampling=downsampling,
            per_scatterer=per_scatterer,
            focused_sum=focused_sum,
        )

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
            label="Reception [paired]",
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
        with self._timer("fft_s"):
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
        # ETA + in-place progress only when the projected run exceeds ~30 s
        # (tqdm already shows progress in verbose mode).
        el_iter = _eta_progress(
            el_iter, s["n_out"], label="RX elements", progress=not s["show"]
        )
        for e_rx in el_iter:
            rx_c, rx_wx, rx_wy, rx_ap, rx_dl, rx_eu, rx_ev = s["rx_groups"][e_rx]
            with self._timer("sir_s"):
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
            print(
                f"Reception [paired] computed in {time.time() - t_wall:.2f} s "
                f"({self._fmt_time_log()})\n"
            )
        return self._finalize(rf, pe_t0, dt, focused_sum, downsampling)

    def __repr__(self) -> str:
        return super().__repr__().replace("Reception(", "ReceptionPaired(", 1)
