"""Shared base class for Reception and ReceptionSDI.

`ReceptionBase` holds everything common to both backends: the patch-state
extraction (`_refresh_sub_elem_attributes`), runtime parameter update (`set`),
scatterer/excitation helpers, and the backend-agnostic public API built on each
subclass's own ``pulse_echo_rf`` / ``_focused_sum_rf`` core — ``sequence_rf``,
``synthetic_aperture_rf`` and ``scan_focusline``. Each subclass only adds its
constructor, time-grid helper, SIR algorithm (``_compute_rf_inner``) and the two
convention-defining wrappers, so the algorithmic flow stays readable.

Output axis convention is ``[emission, reception, Nt]`` (channels before time):
``pulse_echo_rf`` → ``(Erx, Nt)`` / ``(P, Erx, Nt)``; ``sequence_rf`` →
``(Nev, Erx, Nt)``; ``synthetic_aperture_rf`` → ``(Ntx_grp, Erx, Nt)``.
"""

import sys
import time
import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.signal import decimate, hilbert

from pyfield.utilities.helper_functions import compute_sub_elem_attributes

# Output-size threshold above which the heavy methods warn / auto-decimate.
_SIZE_WARN_BYTES = 2 * 1024**3  # 2 GiB

# Pulse-echo fast path splits scatterers into depth bins (see _auto_depth_bins).
# Keep at least this many scatterers per bin so each batch stays cache-resident.
_MIN_SCATTERERS_PER_BIN = 128


def _next_pow2(n):
    return 1 << (int(n - 1).bit_length())


def _wrap_tqdm(iterable, **kwargs):
    """Wrap with tqdm if importable, else return plain iterable."""
    try:
        from tqdm import tqdm

        return tqdm(iterable, **kwargs)
    except ImportError:
        return iterable


def _warn_if_rx_delays_apods_not_default(rx):
    """Warn when RX delays are nonzero or apodization is non-uniform.

    Reception applies the RX delays and apodization PER RECEIVE ELEMENT inside the
    SIR (each element's trace is time-shifted by its delay and scaled by its
    apodization weight); it does NOT sum over receive elements. That is intentional
    so receive-side weighting can be modelled, but it means a weighted/focused RX
    transducer changes the raw per-element RF. Detection is value-based: only fires
    when the weights would actually affect the signal (so an unfocused RX, or
    ``rx = tx.copy()`` of an unfocused TX, stays silent).
    """
    if np.any(np.asarray(rx.delays) != 0.0) or not np.allclose(rx.apodization, 1.0):
        warnings.warn(
            "RX apodization/delays are non-default and ARE applied per receive "
            "element (each element's RF is time-shifted and scaled in the SIR; no "
            "receive sum is performed). Intentional for receive-side weighting; for "
            "raw per-element RF leave the RX transducer unfocused (zero delays, unit "
            "apodization).",
            UserWarning,
            stacklevel=3,
        )


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


class ReceptionBase:
    """Shared state + public API for `Reception` and `ReceptionSDI`.

    Subclasses provide the constructor (setting ``tx``/``rx``/``c``/``fs``/…),
    a per-backend ``_SETTABLE`` map, the SIR core ``_compute_rf_inner`` and its
    convention wrappers ``pulse_echo_rf`` / ``_focused_sum_rf``. Everything else
    (patch extraction, ``set``, input validation, ``sequence_rf`` /
    ``synthetic_aperture_rf`` / ``scan_focusline``) lives here.
    """

    if TYPE_CHECKING:
        # Provided by the concrete Reception / ReceptionSDI classes.
        tx: Any
        rx: Any
        c: float
        fs: float
        verbose: bool
        excitation: Any
        _SETTABLE: dict

        def pulse_echo_rf(
            self, *args: Any, **kwargs: Any
        ) -> tuple[np.ndarray, dict]: ...
        def _focused_sum_rf(
            self, points_m: Any, amps: Any, *, downsampling: Any = None
        ) -> tuple[np.ndarray, dict]: ...

    # ``sim(...)`` is the ergonomic alias for the most common operation.
    def __call__(self, *args, **kwargs):
        return self.pulse_echo_rf(*args, **kwargs)

    # ------------------------------------------------------------------
    # Shared state management + runtime config
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

    def set(self, name: str, value):
        """Update a simulation parameter at runtime.

        Parameters
        ----------
        name : str
            A key of the subclass ``_SETTABLE`` map, or ``"tx"`` / ``"rx"``.
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

    def _auto_depth_bins(self, points_m, n_out):
        """Number of depth bins for the fast path (1 = no binning).

        Scatterers at very different depths echo at very different times, so a single
        FFT (or spectral inverse transform) must span the whole arrival spread — a long,
        mostly-empty time grid whose ``nfft`` (and therefore the spectral form's in-band
        bin count ``N_band = BW/fs·nfft``) grows with the depth span. Grouping scatterers
        by depth lets each bin use a short grid, which dominates the cost.

        The count is driven by the arrival-time spread, but capped where shrinking stops
        paying: ``nfft = next_pow2(pe_T + L)`` cannot drop below ``next_pow2(L)`` (the
        excitation length floor), so once a bin's window is ≈ ``max(128, L)`` samples,
        more bins only multiply the fixed per-bin transform/setup overhead — pure waste
        at very high scatterer counts. Bins are then capped to keep each batch
        cache-resident (≥ ``_MIN_SCATTERERS_PER_BIN`` scatterers/bin).
        """
        P = points_m.shape[0]
        if P < _MIN_SCATTERERS_PER_BIN or n_out < 2:
            return 1
        center = np.asarray(self._tx_centers, dtype=np.float64).mean(axis=0)
        arrival = 2.0 * np.linalg.norm(points_m - center, axis=1) / self.c
        spread = float(arrival.max() - arrival.min()) * self.fs  # samples
        exc = self._resolve_excitation()
        # Window length below which nfft stops shrinking (the next_pow2(L) floor).
        win_floor = max(128.0, float(len(exc)) if exc is not None else 0.0)
        n_bins = max(
            1, round(spread / win_floor)
        )  # windows down to the nfft floor only
        return max(1, min(n_bins, P // _MIN_SCATTERERS_PER_BIN))

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
                    str(out_path),
                    mode="w+",
                    dtype=np.float32,
                    shape=(n_grp, n_rx, nt_dec),
                )
            else:
                rf = np.zeros((n_grp, n_rx, nt_dec), dtype=np.float32)
            rf[0] = rf0d
            for gi in range(1, n_grp):
                rfg, _ = _fire(groups[gi])
                rf[gi] = _anti_alias_decimate(rfg, q)
            if out_path is not None:
                rf.flush()  # ty: ignore[unresolved-attribute]  # rf is a memmap here
        finally:
            self.tx.delays = orig_delays
            self.tx.apodization = orig_apod
            self._refresh_sub_elem_attributes()
        return rf, coords

    # ------------------------------------------------------------------
    @staticmethod
    def _apply_focus_apod(trans, focus, FoverD, apodization_type):
        """Set focusing delays + apodization on ``trans`` from ``focus`` (mm)."""
        trans.compute_delays(focus_mm=focus)
        if apodization_type is not None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                trans.compute_apodization(
                    focus_mm=focus,
                    FoverD=2.0 if FoverD is None else FoverD,
                    apodization_type=apodization_type,
                )
        else:
            trans.apodization = np.ones(trans.n_elements, dtype=np.float32)

    def scan_focusline(
        self,
        focus_mm,
        scatterer_positions_mm,
        amplitudes=None,
        *,
        FoverD=None,
        apodization_type=None,
        rx_FoverD=None,
        rx_apodization_type=None,
        output="envelope",
        downsampling=None,
    ):
        """One conventional focused scan line (Field II ``calc_scat`` match).

        Recomputes the TX **and** RX focus + apodization FROM ``focus_mm`` (reusing
        each transducer's ``compute_delays`` / ``compute_apodization``), then
        beamforms ON RECEIVE INSIDE THE SIR KERNEL via ``_focused_sum_rf``
        (``focused_sum=True``): every RX patch carries its focusing delay +
        apodization and the kernel sums them into one line — exactly Field II
        ``calc_scat``'s internal receive beamforming. This replaces the old
        per-element-RF + external ``DAS_focused_scanline`` path, so it is ~``E_rx``×
        cheaper (one FFT pair, no per-channel loop) and applies the receive focus at
        corner-time resolution (no sample-interpolation loss). Loop ``focus_mm``
        externally to build a B-mode, e.g. ``img[i] = sim.scan_focusline(...)[0]``.

        By default the RX aperture **mirrors** the TX focus + apodization (the usual
        focused-imaging case, and what Field II's ``sesr.m`` does: identical
        ``xdc_apodization`` on emit and receive). Pass ``rx_FoverD`` /
        ``rx_apodization_type`` to give the receive aperture a different profile.

        Parameters
        ----------
        focus_mm : (3,) array_like
            Focal point ``[x, y, z]`` in mm (the per-line looped variable).
        scatterer_positions_mm : (N_scat, 3) numpy.ndarray
        amplitudes : (N_scat,) numpy.ndarray or None, default None
        FoverD : float or None, default None
            TX F-number for the active aperture. Used only when
            ``apodization_type`` is set (defaults to 2.0 there if None).
        apodization_type : str or None, default None
            TX apodization: ``"hanning"`` / ``"rect"`` (F/D=1) / ``"gaussian"`` /
            custom. None → uniform full aperture (no taper).
        rx_FoverD : float or None, default None
            RX F-number. None → mirror ``FoverD`` (identical TX/RX aperture).
        rx_apodization_type : str or None, default None
            RX apodization type. None → mirror ``apodization_type``.
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
        mirror_rx = rx_FoverD is None and rx_apodization_type is None
        orig = (
            self.tx.delays.copy(),
            self.tx.apodization.copy(),
            self.rx.delays.copy(),
            self.rx.apodization.copy(),
        )
        try:
            self._apply_focus_apod(self.tx, focus, FoverD, apodization_type)
            # RX mirrors TX by default; otherwise build its own profile.
            self.rx.compute_delays(focus_mm=focus)
            if mirror_rx:
                self.rx.apodization = np.asarray(
                    self.tx.apodization, dtype=np.float32
                ).copy()
            else:
                self._apply_focus_apod(
                    self.rx,
                    focus,
                    FoverD if rx_FoverD is None else rx_FoverD,
                    apodization_type
                    if rx_apodization_type is None
                    else rx_apodization_type,
                )
            self._refresh_sub_elem_attributes()
            pts_m, amps = self._validate_scatterer_inputs(
                scatterer_positions_mm, amplitudes
            )
            line, coords = self._focused_sum_rf(pts_m, amps, downsampling=downsampling)
        finally:
            (
                self.tx.delays,
                self.tx.apodization,
                self.rx.delays,
                self.rx.apodization,
            ) = orig
            self._refresh_sub_elem_attributes()

        if output == "rf":
            return line.astype(np.float32), coords
        return np.abs(hilbert(line.astype(np.float64))).astype(np.float32), coords
