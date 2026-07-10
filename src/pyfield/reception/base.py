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
from pyfield.plotting import add_transducer_mesh
from pyfield.simulation_base import SimulationBase
from pyfield.utilities.helper_functions import (
    compute_sub_elem_attributes,
    compute_time_grid,
    create_3D_spatial_grid_from_points,
)

# Phases tracked in ``time_log`` (seconds) so the cost of the geometric physics is
# separated from the signal processing: "time_grid_s" (building the SIR sample axes),
# "sir_s" (the SIR / closed-form SIR-spectrum kernels — the physics core), and "fft_s"
# (the FFT-domain two-way convolution + excitation/IR/attenuation filtering).
_TIME_LOG_KEYS = ("time_grid_s", "sir_s", "fft_s")

# Output-size threshold above which the heavy methods warn / auto-decimate.
_SIZE_WARN_BYTES = 2 * 1024**3  # 2 GiB

# Target number of scatterers per depth bin in the pulse-echo fast path (see
# _auto_depth_bins for why binning helps and how the count is chosen). It plays two
# roles: with fewer scatterers than this, binning is skipped entirely; above it, the
# automatic rule aims for roughly this many scatterers per bin — empirically the sweet
# spot where the savings from shorter per-bin time windows still outweigh the fixed
# per-bin overhead (SIR rebuild + FFT setup).
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
        """Compute the RF one depth bin at a time and sum onto one shared time axis.

        Scatterers are sorted by distance to the transmit aperture centroid (a proxy
        for their echo arrival time) and split into ``n_bins`` groups of equal size.
        ``per_bin_fn(idx)`` computes one group's RF on a time grid just long enough
        for that group's arrivals, and returns ``(rf_bin, n0)`` where ``n0`` is the
        integer sample index at which that grid starts on the shared global time axis
        (see `_snap_to_lattice`). Because every bin's samples land exactly on that
        shared axis, recombining is a plain addition into ``rf[:, n0:n0+len]`` —
        no interpolation or resampling anywhere.
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
        """Align one depth bin's time grid with the shared global time axis.

        All depth bins must add onto ONE common time axis (origin ``t0_global``,
        sample step ``dt``), but a bin's natural start time ``t0_nat`` (the earliest
        pulse-echo arrival in that bin) generally falls between two samples of that
        axis. This rounds ``t0_nat`` DOWN to the nearest sample of the shared axis.

        Time-domain paths simply build the bin's grid starting at ``t0_snap`` and
        ignore ``shift``; the spectral path applies ``shift`` as a phase ramp on the
        TX spectrum so the bin's RF still lands exactly on the shared samples.

        Returns
        -------
        n0 : int
            Integer sample index of the snapped start on the shared axis.
        t0_snap : float
            Snapped start time, ``t0_global + n0·dt`` (s).
        shift : float
            Sub-sample remainder ``t0_nat − t0_snap``, in ``[0, dt)`` (s).
        """
        n0 = int(np.floor((t0_nat - t0_global) / dt))
        t0_snap = t0_global + n0 * dt
        return n0, t0_snap, t0_nat - t0_snap

    def _auto_depth_bins(self, points_m, n_out):
        """Choose how many depth bins to split the scatterers into (1 = no binning).

        Why bin at all: each scatterer's echo occupies a short time window around its
        round-trip arrival ``2·|r|/c``, but a single computation over ALL scatterers
        must use a time grid spanning from the earliest to the latest arrival. With
        scatterers spread over centimetres of depth that grid is long and mostly
        empty, and its FFT length ``nfft`` (and, for the spectral form, the number of
        in-band frequency bins ∝ ``nfft``) grows with the depth span. Splitting the
        scatterers into depth-ordered groups lets each group use a grid only as long
        as its own arrival window — much shorter transforms, same summed RF.

        How the count is chosen:

        1. Don't bin when it cannot pay: fewer than ``_MIN_SCATTERERS_PER_BIN``
           scatterers, fewer than 2 output channels, or all scatterers at the same
           depth (arrival spread under one sample).
        2. Otherwise target ~``_MIN_SCATTERERS_PER_BIN`` scatterers per bin
           (``P // _MIN_SCATTERERS_PER_BIN``). More bins mean shorter windows but
           also more fixed per-bin overhead (SIR rebuild + FFT setup), and ``nfft``
           can never drop below the excitation length — so past this occupancy,
           extra bins are pure overhead. This target is the empirical balance.
        3. Never more bins than the arrival spread in samples: a bin thinner than
           one sample cannot shrink its window any further.
        """
        P = points_m.shape[0]
        if P < _MIN_SCATTERERS_PER_BIN or n_out < 2:
            return 1
        center = np.asarray(self._tx_centers, dtype=np.float64).mean(axis=0)
        arrival = 2.0 * np.linalg.norm(points_m - center, axis=1) / self.c
        spread = float(arrival.max() - arrival.min()) * self.fs  # samples
        if spread < 1.0:  # all scatterers at one depth — binning buys nothing.
            return 1
        return max(1, min(P // _MIN_SCATTERERS_PER_BIN, int(spread)))

    def _validate_scatterer_inputs(self, positions_mm, amplitudes):
        """Normalise and validate positions + amplitudes, return (points_m, amps).

        ``positions_mm`` may also be a grid dict (``x_extent``/``dx``/… keys in mm,
        exactly as `Emission` takes): the regular lattice of point targets it
        describes is built automatically — handy with ``per_scatterer=True`` to map
        the PSF across the field. A regular lattice is NOT a tissue phantom (its
        periodicity returns coherent lattice echoes, not speckle); for phantoms
        draw random scatterers, e.g. with
        [make_phantom][pyfield.utilities.phantom.make_phantom].
        """
        if isinstance(positions_mm, dict):
            # Grid dict → regular lattice of unit point targets (already metres).
            *_, points_m = create_3D_spatial_grid_from_points(positions_mm)
            points_m = points_m.astype(np.float32)
        else:
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
        TX_color="Delays",
        RX_color="Delays",
        TX_show_edges=False,
        RX_show_edges=False,
        TX_kwargs=None,
        RX_kwargs=None,
        window_size=(900, 700),
        notebook=False,
        jupyter_backend=None,
        scatterers_cmap="gray",
        legend=True,
        scale=1.0,
        save_path=None,
        off_screen=None,
        return_plotter=False,
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
        TX_color : str or tuple, default "Delays"
            Colour of the transmit aperture. Any PyVista colour (name, hex
            string or RGB tuple) paints the mesh uniformly. The special strings
            ``"Delays"`` / ``"Apodization"`` instead colour each patch by that
            beamforming quantity with a colour bar, as ``transducer.show()``
            does. When TX and RX show the SAME quantity, one shared colour bar
            (right side, common colour range) serves both; different
            quantities get the TX bar on the left and the RX bar on the right.
        RX_color : str or tuple, default "Delays"
            Same as ``TX_color``, for the receive aperture.
        TX_show_edges : bool, default False
            Draw the patch edges of the transmit aperture mesh.
        RX_show_edges : bool, default False
            Draw the patch edges of the receive aperture mesh.
        TX_kwargs : dict, optional
            Extra keyword arguments for the transmit aperture's
            ``add_transducer_mesh`` call (e.g. ``{"cmap": "viridis",
            "opacity": 0.5}``); they override the defaults above.
        RX_kwargs : dict, optional
            Same as ``TX_kwargs``, for the receive aperture.
        window_size : (int, int)
            Pixel dimensions of the render window.
        notebook : bool
            Enable Jupyter notebook rendering.
        jupyter_backend : str, optional
            Backend string passed to PyVista (``'static'``, ``'trame'`` …).
        scatterers_cmap : str, default "gray"
            Matplotlib colormap for the scatterer points (colour encodes each
            point's scattering amplitude).
        legend : bool, default True
            Draw the TX/RX/Scatterers legend. Note the colour bars belong to
            the ``"Delays"``/``"Apodization"`` aperture colouring — pass plain
            colours (e.g. ``TX_color="lightsteelblue"``) to remove them.
        scale : float, default 1.0
            Global resolution multiplier. Enlarges the render window, every font
            (grid, aperture tags, colour bars) and the saved screenshot together,
            so a higher value yields a larger, sharper image without changing the
            framing (e.g. ``scale=3`` for print figures).
        save_path : str or pathlib.Path, optional
            Screenshot file path (e.g. ``"setup.png"``). When given, the
            scene is rendered off-screen and saved instead of opening a
            window.
        off_screen : bool, optional
            Render without opening a window. ``None`` (default) renders off-screen
            only when ``save_path`` is given. Set ``True`` when ``return_plotter``
            is used to screenshot in a headless/batch run — otherwise the returned
            plotter has nothing rendered and ``plotter.screenshot()`` raises
            "Nothing to screenshot".
        return_plotter : bool, default False
            Return the PyVista ``Plotter`` object instead of showing or saving the
            scene. Useful for further customisation (e.g. camera position, extra
            meshes) before calling ``plotter.show()`` or ``plotter.screenshot()``.
            The axis grid is then NOT drawn, so the caller can apply their own
            ``plotter.show_grid(...)`` settings.
        **kwargs
            Forwarded to the scatterer ``add_mesh`` call (e.g. ``point_size``).

        Returns
        -------
        pyvista.Plotter or None
            The plotter (scene assembled, grid and rendering left to the
            caller) when ``return_plotter=True``; otherwise None.
        """
        if off_screen is None:
            off_screen = save_path is not None
        plotter = pv.Plotter(
            window_size=tuple(int(round(s * scale)) for s in window_size),
            notebook=notebook,
            off_screen=off_screen,
        )

        # One colour bar when both apertures show the SAME quantity: PyVista merges
        # scalar bars by title, so giving both meshes the same title shares the bar.
        # For "Delays" a common colour range is set so the shared bar is truthful
        # for both apertures ("Apodization" is already fixed to [0, 1]).
        shared = (
            self.rx is not self.tx
            and TX_color == RX_color
            and TX_color in ("Delays", "Apodization")
        )
        shared_clim = None
        if shared and TX_color == "Delays":
            d = np.concatenate([np.asarray(self.tx.delays), np.asarray(self.rx.delays)])
            shared_clim = [float(d.min()), float(d.max())]

        def _add_aperture(trans, color, show_edges, label, side, user_kwargs):
            # "Delays"/"Apodization" → colour patches by that beamforming scalar;
            # anything else is a uniform PyVista colour. Both go through
            # `add_transducer_mesh` so they share its lighting settings.
            opts = {"show_edges": show_edges, "label": label}
            if color in ("Delays", "Apodization"):
                title = "Delays (s)" if color == "Delays" else color
                if not shared:
                    title = f"{label} {title}"
                opts["scalars"] = color
                opts["scalar_bar_args"] = {
                    "title": title,
                    "title_font_size": int(20 * scale),
                    "label_font_size": int(18 * scale),
                    "vertical": True,
                    "position_x": 0.03 if side == "left" else 0.88,
                    "position_y": 0.35,
                    "height": 0.4,
                }
                if shared_clim is not None:
                    opts["clim"] = shared_clim
            else:
                opts["color"] = color
            opts.update(user_kwargs or {})
            mesh = trans.get_mesh()
            add_transducer_mesh(mesh, plotter=plotter, **opts)
            # Floating "TX"/"RX" tag just behind the aperture face (outside the
            # imaged field), so each probe is identifiable at a glance.
            xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
            tag_pos = np.array(mesh.center, dtype=np.float64)
            tag_pos[2] = zmin - 0.05 * max(xmax - xmin, ymax - ymin, 1.0)
            plotter.add_point_labels(
                [tag_pos],
                [label],
                font_size=int(16 * scale),
                bold=True,
                shape=None,
                show_points=False,
                always_visible=True,
            )

        if self.rx is self.tx:
            _add_aperture(
                self.tx, TX_color, TX_show_edges, "TX = RX", "right", TX_kwargs
            )
        else:
            # Shared bar sits on the right; otherwise TX bar left, RX bar right.
            _add_aperture(
                self.tx,
                TX_color,
                TX_show_edges,
                "TX",
                "right" if shared else "left",
                TX_kwargs,
            )
            _add_aperture(self.rx, RX_color, RX_show_edges, "RX", "right", RX_kwargs)

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
                "cmap": scatterers_cmap,
                "opacity": a / peak,
                "render_points_as_spheres": True,
                "point_size": 10.0 * scale,
                # Horizontal amplitude bar, bottom-centred, clear of the aperture
                # colour bars on the left/right edges.
                "scalar_bar_args": {
                    "title": "Amplitude",
                    "title_font_size": int(20 * scale),
                    "label_font_size": int(18 * scale),
                    "vertical": False,
                    "position_x": 0.35,
                    "position_y": 0.03,
                    "width": 0.3,
                },
                "label": "Scatterers",
            }
            for key, val in defaults.items():
                kwargs.setdefault(key, val)
            plotter.add_mesh(cloud, **kwargs)

        if legend:
            plotter.add_legend()
        plotter.add_axes(label_size=(0.1, 0.1, 0.1))
        # With return_plotter the axis grid is left to the caller (their own
        # show_grid settings would stack on top of this one).
        if not return_plotter:
            plotter.show_grid(
                font_size=int(10 * scale),
                xtitle="X (mm)",
                ytitle="Y (mm)",
                ztitle="Z (mm)",
            )

        plotter.camera.up = (0, 0, -1)  # ty: ignore[unresolved-attribute]
        if return_plotter:
            return plotter
        if save_path is not None:
            plotter.screenshot(str(save_path))
            plotter.close()
        elif jupyter_backend is not None:
            plotter.show(jupyter_backend=jupyter_backend)
        else:
            plotter.show()

    # ------------------------------------------------------------------
    def sequence_rf(
        self,
        scatterer_positions_mm,
        amplitudes,
        tx_events,
        *,
        downsampling=None,
        out_path=None,
        checkpoint_chunks=1,
    ):
        """Pulse-echo RF for a sequence of TX events (emission basis: PW/DW/...).

        Each event sets the TX delays/apodization, then ``pulse_echo_rf`` is run
        (summed over scatterers). Useful as the emission basis for matrix imaging.

        With ``out_path``, the sequence is CHECKPOINTED: each event's RF is
        written to disk (one compressed file per event + a manifest) the moment
        it finishes, and re-running the same call on the same folder skips the
        completed events and resumes from the first missing one. The manifest
        fingerprints the full simulation (probe, medium, excitation,
        scatterers, events); re-running with anything changed raises instead of
        silently mixing incompatible data.

        ``checkpoint_chunks`` bounds the work lost to a crash WITHIN one event:
        the RF is linear in the scatterers, so the cloud is split into chunks
        that are simulated and checkpointed one file at a time, then summed.
        Every chunk carries four zero-amplitude "grid sentinel" points (just
        nearer than the nearest scatterer and just farther than the farthest,
        per aperture) so all chunks of an event share one time grid and their
        sum is sample-exact.

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
        out_path : str or pathlib.Path or None, default None
            Checkpoint folder (created if missing). Events already completed
            there are skipped; each new event is written atomically on
            completion, so a crash costs at most the event in flight.
        checkpoint_chunks : int, default 1
            Number of scatterer chunks checkpointed per TX event (requires
            ``out_path``). With N chunks a crash costs at most 1/N of an
            event; pick it so one chunk takes ~10–15 minutes.

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

        Raises
        ------
        ValueError
            If ``checkpoint_chunks > 1`` without ``out_path``, or if ``tx``
            and ``rx`` are the same object while events set delays/apodization
            (RX weights are per receive channel, so the event's TX weights
            would corrupt the receive traces — pass ``rx=tx.copy()``).
        TypeError
            If ``checkpoint_chunks > 1`` with a grid-dict scatterer input.
        """
        n_ev = len(tx_events)
        n_chunks = int(checkpoint_chunks)
        if n_chunks < 1:
            raise ValueError("checkpoint_chunks must be >= 1.")
        if n_chunks > 1 and out_path is None:
            raise ValueError(
                "checkpoint_chunks > 1 requires out_path — it exists to bound "
                "the work lost to a crash, which needs on-disk checkpoints."
            )
        if n_chunks > 1 and isinstance(scatterer_positions_mm, dict):
            raise TypeError(
                "checkpoint_chunks needs explicit scatterer positions, not a grid dict."
            )
        if self.rx is self.tx and any(
            "delays" in ev or "apodization" in ev for ev in tx_events
        ):
            # Reception applies RX delays/apodization PER RECEIVE CHANNEL, so
            # setting the event's TX weights on a shared object would also
            # time-shift/weight every receive trace — silently corrupting the RF.
            raise ValueError(
                "tx and rx are the same transducer object: per-event TX delays/"
                "apodization would also be applied on receive. Pass separate "
                "instances (e.g. rx=tx.copy())."
            )

        dataset, done = None, set()
        if out_path is not None:
            dataset = self._open_sequence_dataset(
                out_path,
                scatterer_positions_mm,
                amplitudes,
                tx_events,
                downsampling,
                n_chunks,
            )
            done = set(dataset.completed)
            if done and self.verbose:
                print(
                    f"Resuming {out_path}: {len(done)}/{n_ev * n_chunks} "
                    "checkpoint files done."
                )

        if n_chunks > 1:
            pos_mm = np.asarray(scatterer_positions_mm, dtype=np.float64)
            amp_arr = (
                np.ones(pos_mm.shape[0], dtype=np.float32)
                if amplitudes is None
                else np.asarray(amplitudes, dtype=np.float32)
            )
            sentinels_mm = self._grid_sentinels_mm(pos_mm)
            bounds = np.linspace(0, pos_mm.shape[0], n_chunks + 1).astype(int)

        orig_delays = self.tx.delays.copy()
        orig_apod = self.tx.apodization.copy()
        results, coords_out, t0_events = [], None, []
        try:
            for i, event in enumerate(tx_events):
                first = i * n_chunks
                if dataset is not None and all(
                    first + k in done for k in range(n_chunks)
                ):
                    continue  # already on disk from a previous run.
                if "delays" in event:
                    self.tx.delays = np.asarray(event["delays"], dtype=np.float32)
                if "apodization" in event:
                    self.tx.apodization = np.asarray(
                        event["apodization"], dtype=np.float32
                    )
                self._refresh_sub_elem_attributes()
                for k in range(n_chunks):
                    if dataset is not None and first + k in done:
                        continue
                    if n_chunks == 1:
                        pts_k, amp_k = scatterer_positions_mm, amplitudes
                    else:
                        sl = slice(bounds[k], bounds[k + 1])
                        pts_k = np.concatenate([pos_mm[sl], sentinels_mm])
                        amp_k = np.concatenate(
                            [amp_arr[sl], np.zeros(len(sentinels_mm), np.float32)]
                        )
                    if self.verbose:
                        tag = f", chunk {k + 1}/{n_chunks}" if n_chunks > 1 else ""
                        print(f"\n=== TX event {i + 1}/{n_ev}{tag} ===")
                    t_ev = time.perf_counter()
                    rf_i, coords_i = self.pulse_echo_rf(
                        pts_k, amp_k, downsampling=downsampling
                    )
                    if coords_out is None:
                        n_rx, nt = rf_i.shape
                        est = n_ev * n_rx * nt * 4
                        if est > _SIZE_WARN_BYTES and dataset is None:
                            warnings.warn(
                                f"sequence_rf output is ~{est / 1024**3:.1f} GiB "
                                f"({n_ev}×{n_rx}×{nt}); pass downsampling= to "
                                "shrink it.",
                                UserWarning,
                                stacklevel=2,
                            )
                        coords_out = coords_i
                    if dataset is not None:
                        dataset.write_event(
                            first + k,
                            rf_i,
                            coords_i["t0"],
                            coords_i["dt"],
                            duration_s=round(time.perf_counter() - t_ev, 2),
                            tx_event=i,
                            chunk=k,
                        )
                    else:
                        results.append(rf_i)
                        t0_events.append(coords_i["t0"])
        finally:
            self.tx.delays = orig_delays
            self.tx.apodization = orig_apod
            self._refresh_sub_elem_attributes()

        if dataset is not None:
            return dataset.load_all()

        max_nt = max(r.shape[1] for r in results)
        n_rx = results[0].shape[0]
        rf_all = np.zeros((n_ev, n_rx, max_nt), dtype=np.float32)
        for i, r in enumerate(results):
            rf_all[i, :, : r.shape[1]] = r
        assert coords_out is not None
        coords_out["t0_per_event"] = np.asarray(t0_events, dtype=np.float64)
        return rf_all, coords_out

    def _checkpointed_pulse_echo(
        self,
        scatterer_positions_mm,
        amplitudes,
        *,
        per_scatterer,
        downsampling,
        out_path,
        checkpoint_chunks,
    ):
        """Checkpointed single-shot pulse-echo = a one-event sequence.

        Backend for ``pulse_echo_rf(out_path=...)``: the current TX
        delays/apodization become an explicit event so the dataset fingerprint
        captures the focus state (``sequence_rf`` fingerprints its events; bare
        transducer state would go unchecked and a resume could silently mix
        two different focus settings).
        """
        if per_scatterer:
            raise ValueError(
                "per_scatterer=True cannot be checkpointed — the PSF is "
                "per point, there is nothing to sum in chunks."
            )
        event = {
            "delays": np.asarray(self.tx.delays, dtype=np.float32).copy(),
            "apodization": np.asarray(self.tx.apodization, dtype=np.float32).copy(),
        }
        rf, coords = self.sequence_rf(
            scatterer_positions_mm,
            amplitudes,
            [event],
            downsampling=downsampling,
            out_path=out_path,
            checkpoint_chunks=checkpoint_chunks,
        )
        return rf[0], {"t0": float(coords["t0"]), "dt": float(coords["dt"])}

    def _open_sequence_dataset(
        self,
        out_path,
        scatterer_positions_mm,
        amplitudes,
        tx_events,
        downsampling,
        checkpoint_chunks=1,
    ):
        """Open/create the checkpoint folder for ``sequence_rf``.

        The fingerprint covers everything that determines the RF: probe
        geometries (via their ``repr``), medium, sampling, excitation,
        scatterer cloud and the per-event delays/apodization. Any bit-level
        change in these makes the resume refuse, so a checkpoint can never be
        silently continued with different physics. The chunk count is part of
        the fingerprint too (chunk files are indexed by it), but only when
        chunking is on, so unchunked datasets keep their original fingerprint.
        """
        from pyfield.io import RFDataset

        config = {
            "tx": repr(self.tx),
            "rx": repr(self.rx),
            "fs": float(self.fs),
            "c": float(self.c),
            "downsampling": downsampling,
            "excitation": None
            if self.excitation is None
            else np.asarray(self.excitation),
            "scatterer_positions_mm": np.asarray(scatterer_positions_mm),
            "amplitudes": None if amplitudes is None else np.asarray(amplitudes),
            "tx_events": [
                {k: np.asarray(v) for k, v in ev.items()} for ev in tx_events
            ],
        }
        meta = {
            "n_events": len(tx_events) * checkpoint_chunks,
            "fs": float(self.fs),
            "c": float(self.c),
        }
        if checkpoint_chunks > 1:
            config["checkpoint_chunks"] = checkpoint_chunks
            meta["checkpoint_chunks"] = checkpoint_chunks
        return RFDataset(out_path, config, meta=meta)

    def _grid_sentinels_mm(self, pos_mm):
        """Four zero-amplitude points that pin the pulse-echo time grid.

        The RF time window is sized from the min/max patch-centre-to-scatterer
        distance of the simulated point set, so two different chunks of one
        phantom would get slightly different (non-sample-aligned) windows and
        their partial RFs could not be summed. Adding to EVERY chunk one point
        0.2 mm nearer than the nearest scatterer and one 0.2 mm farther than
        the farthest (for the TX and the RX aperture) forces one common window;
        with zero scattering amplitude they contribute nothing to the physics.

        Parameters
        ----------
        pos_mm : (N_scat, 3) numpy.ndarray
            Full scatterer cloud in mm.

        Returns
        -------
        (4, 3) numpy.ndarray
            Sentinel positions in mm.
        """
        pts = np.asarray(pos_mm, dtype=np.float64) * 1e-3
        out = []
        for centers in (self._tx_centers, self._rx_centers):
            c = np.asarray(centers, dtype=np.float64)
            c2 = np.einsum("ij,ij->i", c, c)
            best = (np.inf, 0, 0)
            worst = (-np.inf, 0, 0)
            # Chunked |p − c|² = |p|² + |c|² − 2 p·cᵀ keeps memory bounded.
            for s in range(0, pts.shape[0], 8192):
                p = pts[s : s + 8192]
                d2 = (
                    np.einsum("ij,ij->i", p, p)[:, None] + c2[None, :] - 2.0 * (p @ c.T)
                )
                ip, ic = np.unravel_index(np.argmin(d2), d2.shape)
                if d2[ip, ic] < best[0]:
                    best = (d2[ip, ic], s + ip, ic)
                ip, ic = np.unravel_index(np.argmax(d2), d2.shape)
                if d2[ip, ic] > worst[0]:
                    worst = (d2[ip, ic], s + ip, ic)
            for (d2v, ip, ic), push in ((best, -0.2e-3), (worst, +0.2e-3)):
                p, cc = pts[ip], c[ic]
                d = max(np.sqrt(max(d2v, 0.0)), 1e-9)
                # Move past the extreme point along the centre→point line.
                out.append(cc + (p - cc) * (d + push) / d)
        return np.asarray(out) * 1e3

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
        checkpoint_chunks=1,
    ):
        """Synthetic-aperture RF: each TX element/group fires alone, all RX record.

        Canonical diverging-wave emission basis for matrix imaging: every group
        fires FLAT (zero delay, unit apodization) — this OVERRIDES any TX delays/
        apodization set on the transducer. ``tx_groups="element"`` is full FMC
        (= Field II ``calc_scat_all``); ``int N`` uses N-element sub-apertures;
        a ``list[list[int]]`` gives custom groups.

        Each group is one TX event of ``sequence_rf``, so it shares that
        method's checkpointing: with ``out_path`` every group (or scatterer
        chunk, with ``checkpoint_chunks``) lands on disk the moment it
        finishes, a re-run resumes at the first missing file, and a changed
        setup refuses. Output ``Ntx_grp·Erx·Nt`` can be huge, hence the
        anti-aliased decimation (default 10×); a 10 s countdown precedes an
        in-RAM run so it can be aborted (a checkpointed run is always
        interruptible, so it starts straight away).

        Parameters
        ----------
        scatterer_positions_mm : (N_scat, 3) numpy.ndarray
            Scatterer positions in mm.
        amplitudes : (N_scat,) numpy.ndarray or None, default None
            Scattering coefficient at each position. None defaults to ones.
        tx_groups : str or int or list[list[int]], default "element"
            Transmit grouping: ``"element"`` fires each element alone (full
            FMC), ``int N`` fires N-element sub-apertures, ``list[list[int]]``
            fires custom element groups.
        decimation : int, default 10
            Anti-aliased time decimation factor.
        out_path : str or pathlib.Path or None, default None
            Checkpoint folder (an ``RFDataset``, created if missing): one
            compressed file per group, crash-safe and resumable.
        countdown : bool, default True
            Print a 10 s abortable countdown before an in-RAM computation
            (skipped if stdout is not a TTY or ``out_path`` is given).
        checkpoint_chunks : int, default 1
            Scatterer chunks checkpointed per group (requires ``out_path``);
            a crash then costs at most one chunk of one group.

        Returns
        -------
        rf : (Ntx_grp, Erx, Nt) numpy.ndarray
            Per-group, per-receive-element RF.
        coords : dict
            ``"t0"``/``"dt"`` of the decimated time axis (plus
            ``"t0_per_event"``, identical for all groups — they all fire flat).
        """
        n_el = int(self.tx.delays.shape[0])
        groups = self._resolve_tx_groups(tx_groups, n_el)
        events = []
        for g in groups:
            apod = np.zeros(n_el, dtype=np.float32)
            apod[list(g)] = 1.0
            events.append(
                {"delays": np.zeros(n_el, dtype=np.float32), "apodization": apod}
            )

        if out_path is None:
            # RAM-size estimate before committing: the pulse-echo window length
            # comes from the min/max patch↔scatterer travel times (same grid
            # for every group — all fire flat).
            points_m, _ = self._validate_scatterer_inputs(
                scatterer_positions_mm, amplitudes
            )
            *_, tx_T = self._oneway_time_grid(points_m, "tx")
            *_, rx_T = self._oneway_time_grid(points_m, "rx")
            nt_dec = -(-(tx_T + rx_T - 1) // max(int(decimation), 1))  # ceil
            est = len(groups) * int(self.rx.delays.shape[0]) * nt_dec * 4
            _countdown(est, "synthetic_aperture_rf", countdown)

        return self.sequence_rf(
            scatterer_positions_mm,
            amplitudes,
            events,
            downsampling=decimation,
            out_path=out_path,
            checkpoint_chunks=checkpoint_chunks,
        )

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
            Scatterer positions in mm.
        amplitudes : (N_scat,) numpy.ndarray or None, default None
            Scattering coefficient at each position. None defaults to ones.
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
            ``"t0"``/``"dt"`` of the line's time axis (beam-axis referenced).
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
