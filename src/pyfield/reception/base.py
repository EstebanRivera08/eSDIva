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
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import numpy as np
import pyvista as pv
from scipy.signal import decimate, hilbert

from pyfield.hsir.farfield_rect_patch import compute_h_sir
from pyfield.simulation_base import SimulationBase
from pyfield.utilities.helper_functions import (
    compute_sub_elem_attributes,
    compute_time_grid,
)

# Phases tracked in ``time_log`` (seconds) so the cost of the geometric physics is
# separated from the signal processing: "time_grid_s" (building the SIR sample axes),
# "sir_s" (the SIR / closed-form SIR-spectrum kernels — the physics core), and "fft_s"
# (the FFT-domain two-way convolution + excitation/IR/attenuation filtering).
_TIME_LOG_KEYS = ("time_grid_s", "sir_s", "fft_s")

# Output-size threshold above which the heavy methods warn / auto-decimate.
_SIZE_WARN_BYTES = 2 * 1024**3  # 2 GiB

# Pulse-echo fast path splits scatterers into depth bins (see _auto_depth_bins). This is
# both the floor (fewer scatterers than this → don't bin) and the target bin occupancy:
# the total cost (forward FFTs + per-bin SIR rebuild) is empirically minimised near this
# many scatterers per bin, so `_auto_depth_bins` aims for `P // _MIN_SCATTERERS_PER_BIN`.
_MIN_SCATTERERS_PER_BIN = 100


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
            "RX delays/apodization applied per receive element (no receive sum).",
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


class ReceptionBase(SimulationBase):
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
    # Per-phase timing (time_grid / SIR kernel / FFT convolution)
    # ------------------------------------------------------------------

    def _reset_time_log(self):
        """Zero the per-phase wall-clock log at the start of an RF computation."""
        self.time_log = dict.fromkeys(_TIME_LOG_KEYS, 0.0)

    @contextmanager
    def _timer(self, key):
        """Add the wall-clock time of the enclosed block to ``self.time_log[key]``."""
        if not hasattr(self, "time_log"):
            self._reset_time_log()
        t = time.perf_counter()
        try:
            yield
        finally:
            self.time_log[key] = self.time_log.get(key, 0.0) + (time.perf_counter() - t)

    def _timed_h_sir(self, *args, **kwargs):
        """``compute_h_sir`` timed into ``time_log["sir_s"]`` (the SIR kernel cost)."""
        with self._timer("sir_s"):
            return compute_h_sir(*args, **kwargs)

    def _fmt_time_log(self):
        """One-line ``time_grid/sir/fft`` breakdown of the last RF computation."""
        tl = self.time_log
        return (
            f"time_grid {tl['time_grid_s']:.2f}s, sir {tl['sir_s']:.2f}s, "
            f"fft {tl['fft_s']:.2f}s"
        )

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
        self._apply_settable(name, value)

    def _extract_rx_element_patches(self):
        """Pre-extract per-RX-element patch arrays."""
        return self._group_patches_by_element(
            int(self.rx.delays.shape[0]),
            self._rx_sub_el_idx,
            (
                self._rx_centers,
                self._rx_wx,
                self._rx_wy,
                self._rx_apod,
                self._rx_delays,
                self._rx_eu,
                self._rx_ev,
            ),
        )

    def _rx_groups(self, focused_sum):
        """Receive patch groups feeding the per-element / per-line SIR.

        ``focused_sum`` → ONE group holding every RX patch (the kernel sums it into a
        single beamformed line, Field II ``calc_scat``'s internal receive sum); else
        one group per receive element → raw per-channel RF. Each group is the tuple
        ``(centers, wx, wy, apod, delays, eu, ev)`` of that group's patch arrays.
        """
        if focused_sum:
            return [
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
        return self._extract_rx_element_patches()

    def _oneway_time_grid(self, points_m, aperture):
        """Far-field SIR sample grid for one aperture (``"tx"`` or ``"rx"``).

        Wraps ``compute_time_grid`` with that aperture's patch centres, largest patch
        dimensions and focusing delays. Returns ``(time_grid, t0, dt, T)``: the sampled
        time axis and its origin ``t0`` (s), step ``dt`` (s) and length ``T`` (samples),
        sized to cover the trapezoidal SIR of every scatterer.
        """
        if aperture == "tx":
            M, centers, wx_max, wy_max, delays = (
                self._tx_M,
                self._tx_centers,
                self._tx_wx_max,
                self._tx_wy_max,
                self.tx.delays,
            )
        else:
            M, centers, wx_max, wy_max, delays = (
                self._rx_M,
                self._rx_centers,
                self._rx_wx_max,
                self._rx_wy_max,
                self.rx.delays,
            )
        return compute_time_grid(
            points_m.shape[0],
            M,
            points_m,
            centers,
            wx_max,
            wy_max,
            self.c,
            self.fs,
            delays,
            verbose=False,
        )

    def _finalize(self, rf, pe_t0, dt, focused_sum, downsampling):
        """Beam-axis ``t0``, coords dict, and optional anti-aliased decimation.

        The pulse-echo origin ``pe_t0`` is shifted to the beam axis by subtracting the TX
        focusing bulk ``tx.delays.max()`` (the last-firing element's delay) so downstream
        beamforming needs no per-line correction; we also bakes the RX focus if any,
        so the RX bulk is subtracted too.

        For an elevation-focused (cylindrical-lens) aperture the time grid is referenced to
        the first-arriving rim, but the focused elevation echo peaks one lens transit later;
        each aperture's lens sag ``R − √(R² − (h/2)²)`` is added back as a propagation time
        (TX once, RX once) so the RF origin matches a lens-focused reference. Flat apertures
        have zero sag, so this is a no-op for them.
        """
        t0 = pe_t0 - float(np.max(self.tx.delays)) - float(np.max(self.rx.delays))
        t0 += (self.tx.elevation_lens_sag + self.rx.elevation_lens_sag) / self.c
        coords = {"t0": t0, "dt": dt}
        if downsampling is not None and int(downsampling) > 1:
            step = int(downsampling)
            rf = _anti_alias_decimate(rf, step)  # anti-aliased along last (time) axis
            coords["dt"] = dt * step
        return rf, coords

    def _accumulate_depth_bins(self, points_m, n_bins, per_bin_fn):
        """Sum per-depth-bin RF back onto one shared global sample lattice.

        Scatterers are ordered by distance from the transmit aperture centroid and split
        into ``n_bins`` depth groups; each spans a tight arrival window so its RF uses a
        short FFT. ``per_bin_fn(idx)`` returns ``(rf_bin, n0)`` — that bin's RF and the
        integer sample offset where it starts on the global lattice — and the bins simply
        add at ``n0`` (no resampling, since every bin shares the lattice).
        """
        center = np.asarray(self._tx_centers, dtype=np.float64).mean(axis=0)
        order = np.argsort(np.linalg.norm(points_m - center, axis=1))
        results = [per_bin_fn(idx) for idx in np.array_split(order, n_bins) if idx.size]
        n_out = results[0][0].shape[0]
        nt_total = max(off + r.shape[1] for r, off in results)
        rf = np.zeros((n_out, nt_total), dtype=np.float32)
        for r, off in results:
            rf[:, off : off + r.shape[1]] += r
        return rf

    @staticmethod
    def _snap_to_lattice(t0_nat, t0_global, dt):
        """Snap a bin's natural window origin onto the shared global sample lattice.

        Each depth bin has its own natural pulse-echo start ``t0_nat``, but all bins must
        add back onto ONE lattice (origin ``t0_global``, step ``dt``) so per-bin results
        combine at an integer sample offset with no resampling. This rounds ``t0_nat`` down
        to the nearest lattice sample and returns that integer offset ``n0``, the snapped
        origin ``t0_snap = t0_global + n0·dt``, and the sub-sample remainder
        ``shift = t0_nat - t0_snap ∈ [0, dt)`` (the spectral path absorbs ``shift`` into its
        TX reference phase so the product still lands at ``t0_snap``).

        Returns
        -------
        n0 : int
        t0_snap : float
        shift : float
        """
        n0 = int(np.floor((t0_nat - t0_global) / dt))
        t0_snap = t0_global + n0 * dt
        return n0, t0_snap, t0_nat - t0_snap

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
        if spread < 1.0:  # all scatterers at one depth — binning buys nothing.
            return 1
        # Each bin's FFT length covers its arrival window PLUS the fixed per-scatterer SIR
        # width (the focusing-delay + aperture spread, ~hundreds of samples — not a short
        # trapezoid). Finer binning shrinks only the window part of nfft, so the cost keeps
        # falling well past the old `spread/win_floor` estimate (which undershot to ~6 bins
        # and left PyField FFT-bound). Empirically the total (forward FFTs + per-bin SIR
        # rebuild) is minimised near ``_MIN_SCATTERERS_PER_BIN`` scatterers per bin; never
        # bin finer than the arrival spread (a sub-sample-thin bin buys nothing).
        return max(1, min(P // _MIN_SCATTERERS_PER_BIN, int(spread)))

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
    def show(
        self,
        scatterer_positions_mm=None,
        amplitudes=None,
        *,
        window_size=(900, 700),
        notebook=False,
        jupyter_backend=None,
        **kwargs,
    ):
        """Interactive 3-D preview of the pulse-echo setup (TX, RX, scatterers).

        Renders the transmit and receive apertures (their patch meshes, in mm)
        and the scatterer cloud, each point coloured by its scattering
        amplitude and faded in proportion to it — a quick visual check that
        positions, units and aperture poses are what the simulation will see.
        When TX and RX are the same object the aperture is drawn once.

        Parameters
        ----------
        scatterer_positions_mm : (N_scat, 3) array-like, optional
            Scatterer positions in mm. None draws only the apertures.
        amplitudes : (N_scat,) array-like, optional
            Scattering amplitude per point. None defaults to ones
            (all points fully opaque).
        window_size : (int, int)
            Pixel dimensions of the render window.
        notebook : bool
            Enable Jupyter notebook rendering.
        jupyter_backend : str, optional
            Backend string passed to PyVista (``'static'``, ``'trame'`` …).
        **kwargs
            Forwarded to the scatterer ``add_mesh`` call (e.g. ``point_size``).
        """
        plotter = pv.Plotter(window_size=window_size, notebook=notebook)

        if self.rx is self.tx:
            plotter.add_mesh(
                self.tx.get_mesh(),
                color="lightsteelblue",
                show_edges=True,
                label="TX = RX",
            )
        else:
            plotter.add_mesh(
                self.tx.get_mesh(),
                color="lightsteelblue",
                show_edges=True,
                label="TX",
            )
            plotter.add_mesh(
                self.rx.get_mesh(), color="salmon", show_edges=True, label="RX"
            )

        if scatterer_positions_mm is not None:
            points_m, amps = self._validate_scatterer_inputs(
                scatterer_positions_mm, amplitudes
            )
            cloud = pv.PolyData(np.asarray(points_m, dtype=np.float64) * 1e3)
            cloud["Amplitude"] = amps
            # Fade each point by |amplitude| so weak scatterers recede visually.
            a = np.abs(amps.astype(np.float64))
            peak = a.max() if a.size and a.max() > 0 else 1.0
            defaults = {
                "scalars": "Amplitude",
                "cmap": "viridis",
                "opacity": a / peak,
                "render_points_as_spheres": True,
                "point_size": 10.0,
                "scalar_bar_args": {"title": "Amplitude"},
                "label": "Scatterers",
            }
            for key, val in defaults.items():
                kwargs.setdefault(key, val)
            plotter.add_mesh(cloud, **kwargs)

        plotter.add_legend()
        plotter.add_axes()
        plotter.show_grid(
            font_size=10, xtitle="X (mm)", ytitle="Y (mm)", ztitle="Z (mm)"
        )
        if jupyter_backend is not None:
            plotter.show(jupyter_backend=jupyter_backend)
        else:
            plotter.show()

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
            Scatterer positions in mm.
        amplitudes : (N_scat,) numpy.ndarray
            Scattering amplitude of each scatterer.
        tx_events : list of dict
            Each dict has ``"delays"`` and/or ``"apodization"`` ``(E,)`` arrays.
        downsampling : int or None, default None
            Anti-aliased time decimation factor applied per event.

        Returns
        -------
        rf : (N_events, Erx, Nt) numpy.ndarray
            Per-event, per-receive-element RF (channels before time).
        coords : dict
            ``"t0"``/``"dt"`` of the first event, plus ``"t0_per_event"`` — an
            ``(N_events,)`` array of each event's beam-axis time origin.
            Events with differing focus have differing ``t0`` (the time grid
            depends on that event's delays); beamform each event with its own
            origin. ``dt`` is shared (one sampling rate); traces are
            zero-padded at the END to the common ``Nt``, so only the origin
            differs.
        """
        n_ev = len(tx_events)
        orig_delays = self.tx.delays.copy()
        orig_apod = self.tx.apodization.copy()
        results, coords_out, t0_events = [], None, []
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
                t0_events.append(coords_i["t0"])
        finally:
            self.tx.delays = orig_delays
            self.tx.apodization = orig_apod
            self._refresh_sub_elem_attributes()

        max_nt = max(r.shape[1] for r in results)
        n_rx = results[0].shape[0]
        rf_all = np.zeros((n_ev, n_rx, max_nt), dtype=np.float32)
        for i, r in enumerate(results):
            rf_all[i, :, : r.shape[1]] = r
        assert coords_out is not None
        coords_out["t0_per_event"] = np.asarray(t0_events, dtype=np.float64)
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
