"""Backend-agnostic public-API wrappers shared by Reception and ReceptionSDI.

These build on each class's own ``pulse_echo_rf`` core (the only backend-specific
method). Output axis convention is ``[emission, reception, Nt]`` (channels before
time): ``pulse_echo_rf`` → ``(Erx, Nt)`` / ``(P, Erx, Nt)``; ``sequence_rf`` →
``(Nev, Erx, Nt)``; ``synthetic_aperture_rf`` → ``(Ntx_grp, Erx, Nt)``.
"""

import sys
import time
import warnings

import numpy as np
from scipy.signal import decimate, hilbert

from pyfield.beamforming import DAS_focused_scanline

# Output-size threshold above which the heavy methods warn / auto-decimate.
_SIZE_WARN_BYTES = 2 * 1024**3  # 2 GiB


def _anti_alias_decimate(rf, q):
    """Anti-aliased decimation along the last (time) axis (linear-phase FIR)."""
    q = int(q)
    if q <= 1:
        return rf
    return decimate(rf, q, axis=-1, ftype="fir").astype(np.float32)


def _countdown(est_bytes, label, enabled):
    """Print a 10 s abortable countdown before a heavy computation."""
    if not enabled or not sys.stdout.isatty():
        return
    print(
        f"{label}: estimated ~{est_bytes / 1024**3:.2f} GiB. Starting in 10 s — "
        f"Ctrl-C now to abort and fix inputs."
    )
    for s in range(10, 0, -1):
        print(f"  {s:2d} ...", end="\r", flush=True)
        time.sleep(1)
    print(" " * 20, end="\r")


class ReceptionMixin:
    """Shared public methods (composed on top of per-class ``pulse_echo_rf``)."""

    # ``sim(...)`` is the ergonomic alias for the most common operation.
    def __call__(self, *args, **kwargs):
        return self.pulse_echo_rf(*args, **kwargs)

    # ------------------------------------------------------------------
    def sequence_rf(
        self, scatterer_positions_mm, amplitudes, tx_events, *, downsampling=None
    ):
        """Pulse-echo RF for a sequence of TX events (emission basis: PW/DW/...).

        Each event sets the TX delays/apodization, then ``pulse_echo_rf`` is run
        (summed over scatterers). Useful as the emission basis for matrix imaging.

        Parameters
        ----------
        scatterer_positions_mm : (N_scat, 3) numpy.ndarray
        amplitudes : (N_scat,) numpy.ndarray
        tx_events : list of dict
            Each dict has ``"delays"`` and/or ``"apodization"`` ``(E,)`` arrays.
        downsampling : int or None, default None
            Anti-aliased time decimation factor applied per event.

        Returns
        -------
        rf : (N_events, Erx, Nt) numpy.ndarray
        coords : dict
            ``"t0"``/``"dt"`` of the first event (events with differing focus have
            differing ``t0``; only the first is returned — beamform per event if
            that matters).
        """
        n_ev = len(tx_events)
        orig_delays = self.tx.delays.copy()
        orig_apod = self.tx.apodization.copy()
        results, coords_out = [], None
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
                    print(f"\n=== TX event {i + 1}/{n_ev} ===")
                rf_i, coords_i = self.pulse_echo_rf(
                    scatterer_positions_mm, amplitudes, downsampling=downsampling
                )
                if i == 0:
                    n_rx, nt = rf_i.shape
                    est = n_ev * n_rx * nt * 4
                    if est > _SIZE_WARN_BYTES:
                        warnings.warn(
                            f"sequence_rf output is ~{est / 1024**3:.1f} GiB "
                            f"({n_ev}×{n_rx}×{nt}); pass downsampling= to shrink it.",
                            UserWarning,
                            stacklevel=2,
                        )
                    coords_out = coords_i
                results.append(rf_i)
        finally:
            self.tx.delays = orig_delays
            self.tx.apodization = orig_apod
            self._refresh_sub_elem_attributes()

        max_nt = max(r.shape[1] for r in results)
        n_rx = results[0].shape[0]
        rf_all = np.zeros((n_ev, n_rx, max_nt), dtype=np.float32)
        for i, r in enumerate(results):
            rf_all[i, :, : r.shape[1]] = r
        return rf_all, coords_out

    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_tx_groups(tx_groups, n_el):
        """Build the list of TX element-index groups for synthetic_aperture_rf."""
        if tx_groups == "element":
            return [[e] for e in range(n_el)]
        if isinstance(tx_groups, int):
            return [
                list(range(i, min(i + tx_groups, n_el)))
                for i in range(0, n_el, tx_groups)
            ]
        return [list(g) for g in tx_groups]

    def synthetic_aperture_rf(
        self,
        scatterer_positions_mm,
        amplitudes=None,
        *,
        tx_groups="element",
        decimation=10,
        out_path=None,
        countdown=True,
    ):
        """Synthetic-aperture RF: each TX element/group fires alone, all RX record.

        Canonical diverging-wave emission basis for matrix imaging: every group
        fires FLAT (zero delay, unit apodization) — this OVERRIDES any TX delays/
        apodization set on the transducer. ``tx_groups="element"`` is full FMC
        (= Field II ``calc_scat_all``); ``int N`` uses N-element sub-apertures;
        a ``list[list[int]]`` gives custom groups.

        Output ``Ntx_grp·Erx·Nt`` can be huge: the result is decimated (anti-
        aliased, default 10×) and, if it would still exceed ~2 GiB and no
        ``out_path`` is given, the decimation is auto-raised to fit RAM (with a
        warning to instead pass coarser ``tx_groups`` and/or ``out_path``). A 10 s
        countdown precedes the computation so it can be aborted.

        Parameters
        ----------
        scatterer_positions_mm : (N_scat, 3) numpy.ndarray
        amplitudes : (N_scat,) numpy.ndarray or None, default None
        tx_groups : str or int or list[list[int]], default "element"
        decimation : int, default 10
            Anti-aliased time decimation factor.
        out_path : str or pathlib.Path or None, default None
            If given, write the result to a ``.npy`` memmap on disk (per group)
            instead of returning it in RAM.
        countdown : bool, default True
            Print a 10 s abortable countdown before computing (skipped if stdout
            is not a TTY).

        Returns
        -------
        rf : (Ntx_grp, Erx, Nt) numpy.ndarray (or numpy.memmap if out_path given)
        coords : dict
        """
        n_el = int(self.tx.delays.shape[0])
        groups = self._resolve_tx_groups(tx_groups, n_el)
        n_grp = len(groups)
        n_rx = int(self.rx.delays.shape[0])

        orig_delays = self.tx.delays.copy()
        orig_apod = self.tx.apodization.copy()

        def _fire(group):
            self.tx.delays = np.zeros(n_el, dtype=np.float32)
            apod = np.zeros(n_el, dtype=np.float32)
            apod[list(group)] = 1.0
            self.tx.apodization = apod
            self._refresh_sub_elem_attributes()
            return self.pulse_echo_rf(scatterer_positions_mm, amplitudes)

        try:
            # Probe the first group to learn Nt, then size-guard before the rest.
            rf0, coords = _fire(groups[0])  # (Erx, Nt_raw)
            nt_raw = rf0.shape[1]
            q = int(decimation)
            nt_dec = -(-nt_raw // max(q, 1))  # ceil
            est = n_grp * n_rx * nt_dec * 4
            if est > _SIZE_WARN_BYTES and out_path is None:
                if tx_groups == "element":
                    warnings.warn(
                        f"synthetic_aperture_rf would be ~{est / 1024**3:.1f} GiB "
                        f"in RAM. Pass a coarser tx_groups (int N) and/or out_path "
                        f"to stream to disk; auto-raising decimation meanwhile.",
                        UserWarning,
                        stacklevel=2,
                    )
                while est > _SIZE_WARN_BYTES and q < nt_raw:
                    q += 1
                    nt_dec = -(-nt_raw // q)
                    est = n_grp * n_rx * nt_dec * 4
            _countdown(est, "synthetic_aperture_rf", countdown)

            rf0d = _anti_alias_decimate(rf0, q)
            nt_dec = rf0d.shape[1]
            coords = {"t0": coords["t0"], "dt": coords["dt"] * q}

            if out_path is not None:
                rf = np.lib.format.open_memmap(
                    str(out_path), mode="w+", dtype=np.float32,
                    shape=(n_grp, n_rx, nt_dec),
                )
            else:
                rf = np.zeros((n_grp, n_rx, nt_dec), dtype=np.float32)
            rf[0] = rf0d
            for gi in range(1, n_grp):
                rfg, _ = _fire(groups[gi])
                rf[gi] = _anti_alias_decimate(rfg, q)
            if out_path is not None:
                rf.flush()
        finally:
            self.tx.delays = orig_delays
            self.tx.apodization = orig_apod
            self._refresh_sub_elem_attributes()
        return rf, coords

    # ------------------------------------------------------------------
    def scan_focusline(
        self,
        focus_mm,
        scatterer_positions_mm,
        amplitudes=None,
        *,
        FoverD=None,
        apodization_type=None,
        output="envelope",
        downsampling=None,
    ):
        """One conventional focused scan line (Field II ``calc_scat``-like).

        Recomputes the TX focus + apodization FROM ``focus_mm`` (reusing the
        transducer's ``compute_delays`` / ``compute_apodization``), runs
        ``pulse_echo_rf``, beamforms the line with
        ``pyfield.beamforming.DAS_focused_scanline`` (RX focus = same point), and
        returns its Hilbert envelope. Loop ``focus_mm`` externally to build a
        conventional line-by-line B-mode, e.g. ``img[i] = sim.scan_focusline(...)[0]``.

        Parameters
        ----------
        focus_mm : (3,) array_like
            Focal point ``[x, y, z]`` in mm (the per-line looped variable).
        scatterer_positions_mm : (N_scat, 3) numpy.ndarray
        amplitudes : (N_scat,) numpy.ndarray or None, default None
        FoverD : float or None, default None
            F-number for the active aperture. Used only when ``apodization_type``
            is set (defaults to 2.0 there if None).
        apodization_type : str or None, default None
            ``"hanning"`` / ``"rect"`` (F/D=1) / ``"gaussian"`` / custom. None →
            uniform full-aperture (no taper).
        output : {"envelope", "rf"}, default "envelope"
            Return the Hilbert envelope (default) or the raw beamformed RF line.
        downsampling : int or None, default None
            Anti-aliased time decimation factor.

        Returns
        -------
        out : (Nt,) numpy.ndarray
            Envelope (or RF line) of the beamformed scan line at lateral focus[0].
        coords : dict
        """
        focus = [float(v) for v in focus_mm]
        orig_delays = self.tx.delays.copy()
        orig_apod = self.tx.apodization.copy()
        try:
            self.tx.compute_delays(focus_mm=focus)
            if apodization_type is not None:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    self.tx.compute_apodization(
                        focus_mm=focus,
                        FoverD=2.0 if FoverD is None else FoverD,
                        apodization_type=apodization_type,
                    )
            else:
                self.tx.apodization = np.ones(self.tx.n_elements, dtype=np.float32)
            self._refresh_sub_elem_attributes()
            rf, coords = self.pulse_echo_rf(
                scatterer_positions_mm, amplitudes, downsampling=downsampling
            )  # (Erx, Nt)
            line = DAS_focused_scanline(rf, coords, self.rx, focus_mm=focus, c=self.c)
        finally:
            self.tx.delays = orig_delays
            self.tx.apodization = orig_apod
            self._refresh_sub_elem_attributes()

        if output == "rf":
            return line.astype(np.float32), coords
        return np.abs(hilbert(line.astype(np.float64))).astype(np.float32), coords
