"""Delay-and-sum (DAS) beamforming for pulse-echo RF data."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from numba import njit, prange

from pyfield.utilities import to_dB


def DAS_focused_scanline(
    rf: npt.NDArray[np.floating],
    coords: dict,
    rx,
    focus_mm: list[float],
    c: float = 1540.0,
) -> npt.NDArray[np.float32]:
    """Delay-and-sum beamformer for a single focused scanline.

    Applies per-channel RX travel-time delays to align echoes from `focus_mm`
    and sums across all receive elements.  Suitable for static focused TX
    where transmit delays are already encoded in the RF data by `Reception`.

    Use this when you already have per-channel RF (e.g. from `pulse_echo_rf`) and
    want to beamform it externally. To build a focused line directly, prefer
    `Reception.scan_focusline`, which sums on receive inside the SIR kernel (one
    FFT pair, corner-time-resolution focus) instead of interpolating sampled RF.

    The delay for element *e* is ``Δt_e = (|r_f − r_e| − |r_f − r_ref|) / c``,
    where *r_ref* is the centre element position.  A positive Δt means the
    echo arrives later in that channel; the interpolation reads ahead by
    ``Δt / dt`` samples to re-align it.

    Parameters
    ----------
    rf : numpy.ndarray
        Raw channel RF data, shape ``(E_rx, Nt)`` (channels, time), as returned
        by `Reception`.
    coords : dict
        Timing info with keys ``"t0"`` (float, seconds) and ``"dt"``
        (float, seconds), as returned by `Reception`.
    rx : TransducerBase
        Receive transducer.  ``rx.element_centers`` provides element positions
        in metres, shape ``(E_rx, 3)``.
    focus_mm : list[float]
        Focal point ``[x, y, z]`` in mm for this scanline.
    c : float, default 1540.0
        Speed of sound (m/s).

    Returns
    -------
    numpy.ndarray
        Beamformed RF line, shape ``(Nt,)``, dtype float32.
    """
    focus_m = np.asarray(focus_mm, dtype=np.float64) * 1e-3
    dt = float(coords["dt"])

    rx_centers = rx.element_centers.astype(np.float64)  # (E_rx, 3) in metres
    dist_rx = np.linalg.norm(rx_centers - focus_m[np.newaxis, :], axis=1)  # (E_rx,)
    t_rx = dist_rx / c

    center_idx = rx_centers.shape[0] // 2
    delta_t = t_rx - t_rx[center_idx]  # positive = echo arrives later in that channel

    Nt = rf.shape[1]  # rf is (E_rx, Nt)
    sample_idx = np.arange(Nt, dtype=np.float64)
    rf_das = np.zeros(Nt, dtype=np.float64)

    for e in range(rf.shape[0]):
        # Echo from focus is at sample (i + Δt/dt) in channel e relative to
        # the centre channel at sample i.  Read ahead to align.
        rf_das += np.interp(
            sample_idx + delta_t[e] / dt,
            sample_idx,
            rf[e, :].astype(np.float64),
            left=0.0,
            right=0.0,
        )

    return rf_das.astype(np.float32)


@njit(parallel=True, fastmath=True, cache=True)
def _das_rca_kernel(
    rf,  # (Nev, Erx, Nt) float32
    t0_ev,  # (Nev,) time of first sample of each event (s)
    dt,  # sample period (s)
    sin_a,  # (Nev,) sin of each event's steering angle
    cos_a,  # (Nev,) cos of each event's steering angle
    xi_max,  # (Nev,) max TX-element projection on the steering direction (m)
    rx_u,  # (Erx,) row-centre coordinate along the row LONG axis (m)
    rx_v,  # (Erx,) row-centre coordinate ACROSS rows (the RX array axis, m)
    rx_z,  # (Erx,) row-centre axial coordinate (m)
    rx_half_len,  # half of the row length along its long axis (m)
    us,  # (Nu,) voxel coordinates along the TX array / row long axis (m)
    vs,  # (Nv,) voxel coordinates along the RX array axis (m)
    zs,  # (Nz,) voxel depths (m)
    c,  # speed of sound (m/s)
    inv_two_fnum,  # 1/(2·F#) — RX aperture half-width per unit depth
    use_hann,  # cosine-taper the RX aperture instead of a hard gate
    t_off,  # extra time offset (s), e.g. pulse-centre lag
):
    """Accumulate the RCA delay-and-sum volume; see `das_rca_volume`."""
    n_ev, n_rx, nt = rf.shape
    nu, nv, nz = us.size, vs.size, zs.size
    vol = np.zeros(nu * nv * nz, dtype=np.float32)

    for flat in prange(nu * nv * nz):  # ty: ignore[not-iterable]
        iu = flat // (nv * nz)
        iv = (flat // nz) % nv
        iz = flat % nz
        u, v, z = us[iu], vs[iv], zs[iz]

        acc = 0.0
        for e in range(n_ev):
            # TX: steered plane wave in the (u, z) plane. The wavefront passes
            # the voxel at (u·sinα + z·cosα − ξ_max)/c in the data's time frame
            # (the simulator subtracts the TX bulk delay = ξ_max projection).
            t_tx = (u * sin_a[e] + z * cos_a[e] - xi_max[e]) / c
            for r in range(n_rx):
                # Dynamic RX aperture: rows farther than z/(2·F#) from the
                # voxel (across the rows) contribute noise, not focus — gate.
                dv = v - rx_v[r]
                half_ap = z * inv_two_fnum
                if np.abs(dv) > half_ap:
                    continue
                # RX: echo arrives at the CLOSEST point of the long row
                # (stationary-phase arrival). Distance to the row treated as a
                # segment: only the part of |u − u_r| beyond the half-length
                # adds path.
                du = np.abs(u - rx_u[r]) - rx_half_len
                if du < 0.0:
                    du = 0.0
                dz = z - rx_z[r]
                t_rx = np.sqrt(du * du + dv * dv + dz * dz) / c

                s = (t_tx + t_rx + t_off - t0_ev[e]) / dt
                i0 = int(np.floor(s))
                if i0 < 0 or i0 >= nt - 1:
                    continue
                frac = s - i0
                val = rf[e, r, i0] * (1.0 - frac) + rf[e, r, i0 + 1] * frac
                if use_hann:
                    val *= 0.5 + 0.5 * np.cos(np.pi * dv / half_ap)
                acc += val
        vol[flat] = acc
    return vol.reshape(nu, nv, nz)


def das_rca_volume(
    rf: npt.NDArray[np.floating],
    coords: dict,
    *,
    angles_deg,
    tx_centers_mm,
    rx_centers_mm,
    rx_length_mm: float,
    grid_mm: dict,
    c: float = 1540.0,
    fnum: float = 1.0,
    rx_apodization: str = "hann",
    t_offset_s: float | None = None,
) -> tuple[npt.NDArray[np.float32], dict]:
    """3-D delay-and-sum for a row-column (RCA) plane-wave sequence.

    In RCA imaging one set of long parallel elements (the "columns") transmits
    plane waves steered in the plane containing the column-array axis, and the
    orthogonal set (the "rows") receives. Focusing is therefore one-way per
    direction: transmit compounding sharpens the image along the TX array
    axis, receive delay-and-sum along the RX array axis.

    Per voxel r = (u, v, z) — u along the TX array (= the rows' long axis),
    v along the RX array — and per event with steering angle α::

        t_tx = (u·sinα + z·cosα − ξ_max) / c      (plane-wave arrival)
        t_rx = |r − nearest point of row_r| / c    (echo back to row r)

    ``ξ_max = max_e(u_e·sinα + z_e·cosα)`` is the largest TX-element
    projection on the steering direction. It is subtracted because the
    simulator's ``t0`` is beam-axis referenced: the TX bulk delay
    (``delays.max()``, which for plane-wave delays ``(ξ_e − ξ_min)/c`` equals
    ``(ξ_max − ξ_min)/c``) is already removed from the time axis, leaving the
    wavefront crossing a point r at ``(ξ(r) − ξ_max)/c``.

    The sample at ``t_tx + t_rx`` is read from each row's trace (linear
    interpolation), weighted by a depth-dependent receive aperture
    (``|v − v_row| ≤ z/(2·F#)``, optionally Hann-tapered) and summed
    coherently over rows and angles.

    The TX array axis is inferred from ``tx_centers_mm`` (the horizontal axis
    of largest spread); the RX array axis is the orthogonal one. For the dual
    orientation (rows transmit, columns receive), call again with the
    swapped arrays and RF, and compound the two envelope volumes.

    Parameters
    ----------
    rf : (N_events, Erx, Nt) numpy.ndarray
        Per-event, per-receive-row RF, as returned by ``sequence_rf``.
    coords : dict
        ``"dt"`` and ``"t0_per_event"`` (or ``"t0"``) from ``sequence_rf``.
    angles_deg : (N_events,) array-like
        Steering angle of each transmitted plane wave (degrees, in the
        TX-array/z plane).
    tx_centers_mm : (Etx, 3) numpy.ndarray
        TX (column) element centres in mm, e.g. ``tx.element_centers * 1e3``.
    rx_centers_mm : (Erx, 3) numpy.ndarray
        RX (row) element centres in mm, in the same order as the RF channels.
    rx_length_mm : float
        Full length of each receive row along its long axis, in mm.
    grid_mm : dict
        Voxel grid: ``{"x_extent": [x0, xf], "y_extent": ..., "z_extent":
        ..., "dx": ..., "dy": ..., "dz": ...}`` in mm (same convention as the
        simulators' field grids).
    c : float, default 1540.0
        Speed of sound (m/s).
    fnum : float, default 1.0
        Receive F-number: rows within ``z/(2·fnum)`` of the voxel (across the
        rows) are summed.
    rx_apodization : {'hann', 'rect'}, default 'hann'
        Taper of the dynamic receive aperture.
    t_offset_s : float or None, default None
        Extra delay added to every sample lookup, to remove the axial bias of a
        band-limited pulse: the delay-and-sum reads the geometric round-trip
        time, but the two-way echo envelope peaks about half a pulse length
        later. When ``None`` (default) this lag is taken from
        ``coords["pulse_center_lag_s"]`` (the reception simulator stores it), so
        the correction is applied automatically; pass a float to override it (or
        ``0.0`` to disable it).

    Returns
    -------
    volume : (Nx, Ny, Nz) numpy.ndarray
        Beamformed RF volume (float32, coherent sum over rows and angles).
        Envelope-detect along z (e.g. Hilbert) before display.
    axes : dict
        ``"x_mm"``, ``"y_mm"``, ``"z_mm"`` — voxel-centre coordinates.

    Raises
    ------
    ValueError
        If ``rx_apodization`` is unknown or the event count mismatches.
    """
    if rx_apodization not in ("hann", "rect"):
        raise ValueError("rx_apodization must be 'hann' or 'rect'.")
    # The band-limited two-way pulse peaks about half its length after the
    # geometric arrival; unless overridden, apply the lag the reception stored.
    if t_offset_s is None:
        t_offset_s = float(coords.get("pulse_center_lag_s", 0.0))
    rf = np.ascontiguousarray(rf, dtype=np.float32)
    angles = np.deg2rad(np.asarray(angles_deg, dtype=np.float64))
    if rf.shape[0] != angles.size:
        raise ValueError(f"rf has {rf.shape[0]} events but {angles.size} angles.")

    tx_c = np.asarray(tx_centers_mm, dtype=np.float64) * 1e-3
    rx_c = np.asarray(rx_centers_mm, dtype=np.float64) * 1e-3

    # TX array axis = horizontal axis (x or y) along which the TX centres
    # spread; the RX array runs along the other one.
    spread = tx_c[:, :2].max(axis=0) - tx_c[:, :2].min(axis=0)
    u_ax = int(np.argmax(spread))  # 0 = x, 1 = y
    v_ax = 1 - u_ax

    def _axis_m(extent, step) -> np.ndarray:
        return np.arange(float(extent[0]), float(extent[1]), float(step)) * 1e-3

    xs = _axis_m(grid_mm["x_extent"], grid_mm["dx"])
    ys = _axis_m(grid_mm["y_extent"], grid_mm["dy"])
    zs = _axis_m(grid_mm["z_extent"], grid_mm["dz"])
    axes_xyz = (xs, ys, zs)
    us, vs = axes_xyz[u_ax], axes_xyz[v_ax]

    sin_a, cos_a = np.sin(angles), np.cos(angles)
    # Largest TX-element projection on each steering direction (the removed
    # TX bulk delay), per event.
    xi_max = (tx_c[:, u_ax][:, None] * sin_a + tx_c[:, 2][:, None] * cos_a).max(axis=0)

    t0_ev = np.asarray(
        coords.get("t0_per_event", np.full(angles.size, coords["t0"])),
        dtype=np.float64,
    )

    vol_uvz = _das_rca_kernel(
        rf,
        t0_ev,
        float(coords["dt"]),
        sin_a,
        cos_a,
        xi_max,
        np.ascontiguousarray(rx_c[:, u_ax]),
        np.ascontiguousarray(rx_c[:, v_ax]),
        np.ascontiguousarray(rx_c[:, 2]),
        float(rx_length_mm) * 1e-3 / 2.0,
        us,
        vs,
        zs,
        float(c),
        1.0 / (2.0 * float(fnum)),
        rx_apodization == "hann",
        float(t_offset_s),
    )
    # Kernel works in (u, v, z); present as (x, y, z).
    volume = vol_uvz if u_ax == 0 else vol_uvz.transpose(1, 0, 2)
    return volume, {"x_mm": xs * 1e3, "y_mm": ys * 1e3, "z_mm": zs * 1e3}


@njit(parallel=True, fastmath=True, cache=True)
def _das_general_kernel(
    rf,  # (Nev, Erx, Nt) float32
    t0_ev,  # (Nev,) time of first sample of each event (s)
    dt,  # sample period (s)
    mode,  # (Nev,) 0 = spherical wavefront, 1 = plane wavefront
    p0,  # (Nev,) source x (m) — spherical — or direction cosine nx — plane
    p1,  # (Nev,) source y (m) or ny
    p2,  # (Nev,) source z (m) or nz
    t_ref,  # (Nev,) TX time origin recovered from the event delays (s)
    focused,  # (Nev,) True when the source is IN FRONT of the array (z_vs > 0)
    rx_x,  # (Erx,) receive-element x (m)
    rx_y,  # (Erx,) receive-element y (m)
    rx_z,  # (Erx,) receive-element z (m)
    xs,  # (Nx,) voxel x coordinates (m)
    ys,  # (Ny,) voxel y coordinates (m)
    zs,  # (Nz,) voxel depths (m)
    c,  # speed of sound (m/s)
    inv_two_fnum,  # 1/(2·F#) — RX aperture half-radius per unit depth
    use_hann,  # cosine-taper the RX aperture instead of a hard gate
    t_off,  # extra time offset (s), e.g. pulse-centre lag
    use_cf,  # weight each voxel by its aperture coherence factor
):
    """Accumulate the general delay-and-sum volume; see `das_volume`."""
    n_ev, n_rx, nt = rf.shape
    nx, ny, nz = xs.size, ys.size, zs.size
    vol = np.zeros(nx * ny * nz, dtype=np.float32)

    for flat in prange(nx * ny * nz):  # ty: ignore[not-iterable]
        ix = flat // (ny * nz)
        iy = (flat // nz) % ny
        iz = flat % nz
        x, y, z = xs[ix], ys[iy], zs[iz]

        acc = 0.0
        sum_sq = 0.0
        n_used = 0
        for e in range(n_ev):
            if mode[e] == 0:
                # Spherical wavefront through/from the (virtual) source: it
                # crosses the voxel |r − r_vs|/c after (diverging) or before
                # (converging, i.e. focused TX above the focal depth) the
                # time origin t_ref recovered from the event's own delays.
                dxs = x - p0[e]
                dys = y - p1[e]
                dzs = z - p2[e]
                rad_vs = np.sqrt(dxs * dxs + dys * dys + dzs * dzs)
                if focused[e] and z < p2[e]:
                    t_tx = t_ref[e] - rad_vs / c
                else:
                    t_tx = t_ref[e] + rad_vs / c
            else:
                # Plane wavefront along unit direction n: crosses the voxel
                # at its projection r·n/c after the time origin.
                t_tx = t_ref[e] + (x * p0[e] + y * p1[e] + z * p2[e]) / c
            for r in range(n_rx):
                # Dynamic RX aperture: elements beyond z/(2·F#) laterally from
                # the voxel contribute noise, not focus — gate radially.
                dx = x - rx_x[r]
                dy = y - rx_y[r]
                rad = np.sqrt(dx * dx + dy * dy)
                half_ap = z * inv_two_fnum
                if rad > half_ap:
                    continue
                dz = z - rx_z[r]
                t_rx = np.sqrt(rad * rad + dz * dz) / c

                s = (t_tx + t_rx + t_off - t0_ev[e]) / dt
                i0 = int(np.floor(s))
                if i0 < 0 or i0 >= nt - 1:
                    continue
                frac = s - i0
                val = rf[e, r, i0] * (1.0 - frac) + rf[e, r, i0 + 1] * frac
                if use_hann:
                    val *= 0.5 + 0.5 * np.cos(np.pi * rad / half_ap)
                acc += val
                if use_cf:
                    sum_sq += val * val
                    n_used += 1
        if use_cf and n_used > 0 and sum_sq > 0.0:
            # Coherence factor CF = |Σs|²/(N·Σs²) ∈ [0, 1]: 1 when every
            # channel saw the same in-phase echo (a true reflector), → 0 for
            # incoherent clutter — multiplying the sum by CF suppresses
            # clutter without moving true echoes.
            vol[flat] = acc * (acc * acc / (n_used * sum_sq))
        else:
            vol[flat] = acc
    return vol.reshape(nx, ny, nz)


def das_volume(
    rf: npt.NDArray[np.floating],
    coords: dict,
    tx_events: list[dict],
    transducer,
    grid_mm: dict,
    *,
    c: float = 1540.0,
    fnum: float = 1.0,
    rx_apodization: str = "hann",
    t_offset_s: float | None = None,
    coherence_weight: bool = False,
) -> tuple[npt.NDArray[np.float32], dict]:
    """3-D delay-and-sum for any transmission basis (TX aperture = RX aperture).

    One beamformer for the four classic unfocused/focused transmit schemes.
    Each event dict carries the SAME ``"delays"``/``"apodization"`` arrays fed
    to ``sequence_rf``, plus ONE geometric key describing the transmitted
    wavefront:

    - ``"virtual_source_mm"``: ``[x, y, z]`` — spherical wavefront. ``z < 0``
      is a diverging wave (source behind the array), ``z > 0`` a focused
      transmit (the wave converges to the focus, then diverges — the
      virtual-source model of a focused beam), ``z ≈ 0`` a single-element /
      synthetic-aperture firing (source on the aperture).
    - ``"angles_deg"``: ``α`` or ``(θx, θy)`` — steered plane wave with
      direction ``n = [sin θx, sin θy, √(1 − sin²θx − sin²θy)]``.

    The transmit time origin is recovered from the event's own delays, so no
    delay-reference convention (min- vs max-referenced) needs to be assumed.
    The simulator's time axis starts with the TX bulk delay removed (its t0 is
    beam-axis referenced), so element ``e`` fires at
    ``τ_e = delays_e − max(delays)``. For a spherical event the wavefront
    obeys ``τ_e = t_ref ± |r_e − r_vs|/c`` (− for a source behind reaching the
    element, + for a focused wave leaving the element toward the focus), so::

        diverging / on-aperture:  t_ref = mean_e(τ_e − |r_e − r_vs|/c)
        focused (z_vs > 0):       t_ref = mean_e(τ_e + |r_e − r_vs|/c)

    (the mean is over apodization-active elements; it is exact when the delays
    were built from that source, and a least-squares fit otherwise). A plane
    wave analogously gives ``t_ref = mean_e(τ_e − r_e·n/c)``. The voxel's
    transmit arrival is then::

        spherical:  t_tx = t_ref ± |r − r_vs|/c    (− above a transmit focus)
        plane:      t_tx = t_ref + r·n/c

    and the echo returns over the direct path ``t_rx = |r − r_e|/c``. The
    sample at ``t_tx + t_rx`` is read from each channel (linear
    interpolation), weighted by a depth-dependent radial receive aperture
    (``|r_xy − r_e,xy| ≤ z/(2·F#)``, optionally Hann-tapered) and summed
    coherently over channels and events.

    Parameters
    ----------
    rf : (N_events, Erx, Nt) numpy.ndarray
        Per-event, per-channel RF, as returned by ``sequence_rf``.
    coords : dict
        ``"dt"`` and ``"t0_per_event"`` (or ``"t0"``) from ``sequence_rf``.
    tx_events : list[dict]
        One dict per event: ``"delays"`` ``(E,)`` (seconds; zeros if absent),
        optional ``"apodization"`` ``(E,)`` (selects the active elements for
        the time-origin fit), and ``"virtual_source_mm"`` or ``"angles_deg"``.
    transducer : TransducerBase
        The array (transmit = receive): ``element_centers`` (metres) provides
        both the RX channel positions (RF channel order) and the TX geometry
        for the time-origin recovery.
    grid_mm : dict
        Voxel grid: ``{"x_extent": [x0, xf], "y_extent": ..., "z_extent":
        ..., "dx": ..., "dy": ..., "dz": ...}`` in mm (same convention as the
        simulators' field grids).
    c : float, default 1540.0
        Speed of sound (m/s).
    fnum : float, default 1.0
        Receive F-number: elements within ``z/(2·fnum)`` laterally of the
        voxel are summed.
    rx_apodization : {'hann', 'rect'}, default 'hann'
        Taper of the dynamic receive aperture.
    t_offset_s : float or None, default None
        Extra delay added to every sample lookup, to remove the axial bias of a
        band-limited pulse: the delay-and-sum reads the geometric round-trip
        time, but the two-way echo envelope peaks about half a pulse length
        later. When ``None`` (default) this lag is taken from
        ``coords["pulse_center_lag_s"]`` — the reception simulator computes it
        from the drive and element impulse responses and stores it there — so
        the correction is applied automatically; pass a float to override it (or
        ``0.0`` to disable it).
    coherence_weight : bool, default False
        Multiply each voxel by its aperture coherence factor
        ``CF = |Σ s|² / (N·Σ s²)``: 1 for a true in-phase reflector, near 0
        for incoherent clutter.

    Returns
    -------
    volume : (Nx, Ny, Nz) numpy.ndarray
        Beamformed RF volume (float32, coherent sum over channels and events).
        Envelope-detect along z (e.g. Hilbert) before display.
    axes : dict
        ``"x_mm"``, ``"y_mm"``, ``"z_mm"`` — voxel-centre coordinates.

    Raises
    ------
    ValueError
        If ``rx_apodization`` is unknown, the event count mismatches, or an
        event carries neither/both geometric keys.
    """
    if rx_apodization not in ("hann", "rect"):
        raise ValueError("rx_apodization must be 'hann' or 'rect'.")
    # The band-limited two-way pulse peaks about half its length after the
    # geometric arrival; unless overridden, apply the lag the reception stored.
    if t_offset_s is None:
        t_offset_s = float(coords.get("pulse_center_lag_s", 0.0))
    rf = np.ascontiguousarray(rf, dtype=np.float32)
    n_ev = len(tx_events)
    if rf.shape[0] != n_ev:
        raise ValueError(f"rf has {rf.shape[0]} events but {n_ev} tx_events.")
    el = np.asarray(transducer.element_centers, dtype=np.float64)  # (E, 3) m

    mode = np.zeros(n_ev, dtype=np.int64)
    params = np.zeros((n_ev, 3), dtype=np.float64)
    t_ref = np.zeros(n_ev, dtype=np.float64)
    focused = np.zeros(n_ev, dtype=np.bool_)
    for i, ev in enumerate(tx_events):
        has_vs = "virtual_source_mm" in ev
        has_pw = "angles_deg" in ev
        if has_vs == has_pw:
            raise ValueError(
                f"event {i}: exactly one of 'virtual_source_mm' or "
                "'angles_deg' is required."
            )
        delays = np.asarray(ev.get("delays", np.zeros(el.shape[0])), np.float64)
        # Firing times in the data's frame: the simulator's t0 has the TX
        # bulk delay max(delays) already removed.
        tau = delays - delays.max()
        apod = np.asarray(ev.get("apodization", np.ones(el.shape[0])), np.float64)
        act = apod > 0
        if not act.any():
            raise ValueError(f"event {i}: apodization silences every element.")
        if has_vs:
            vs = np.asarray(ev["virtual_source_mm"], dtype=np.float64) * 1e-3
            d = np.linalg.norm(el[act] - vs, axis=1) / c
            focused[i] = vs[2] > 0.0
            # Focused: the wave leaves each element and meets at the focus
            # (t_ref = focus time). Diverging/on-aperture: the wave leaves the
            # source and reaches each element (t_ref = source firing time).
            t_ref[i] = (tau[act] + d).mean() if focused[i] else (tau[act] - d).mean()
            params[i] = vs
        else:
            ang = np.atleast_1d(np.deg2rad(np.asarray(ev["angles_deg"], np.float64)))
            sx = np.sin(ang[0])
            sy = np.sin(ang[1]) if ang.size > 1 else 0.0
            if sx * sx + sy * sy > 1.0:
                raise ValueError(f"event {i}: sin²θx + sin²θy > 1.")
            n = np.array([sx, sy, np.sqrt(1.0 - sx * sx - sy * sy)])
            mode[i] = 1
            t_ref[i] = (tau[act] - el[act] @ n / c).mean()
            params[i] = n

    def _axis_m(extent, step) -> np.ndarray:
        return np.arange(float(extent[0]), float(extent[1]), float(step)) * 1e-3

    xs = _axis_m(grid_mm["x_extent"], grid_mm["dx"])
    ys = _axis_m(grid_mm["y_extent"], grid_mm["dy"])
    zs = _axis_m(grid_mm["z_extent"], grid_mm["dz"])

    # Per-event start times; fall back to a shared t0 only when the per-event
    # key is absent (dict.get would evaluate coords["t0"] even when unused).
    t0_ev = np.asarray(
        coords["t0_per_event"]
        if "t0_per_event" in coords
        else np.full(n_ev, coords["t0"]),
        dtype=np.float64,
    )

    volume = _das_general_kernel(
        rf,
        t0_ev,
        float(coords["dt"]),
        mode,
        np.ascontiguousarray(params[:, 0]),
        np.ascontiguousarray(params[:, 1]),
        np.ascontiguousarray(params[:, 2]),
        t_ref,
        focused,
        np.ascontiguousarray(el[:, 0]),
        np.ascontiguousarray(el[:, 1]),
        np.ascontiguousarray(el[:, 2]),
        xs,
        ys,
        zs,
        float(c),
        1.0 / (2.0 * float(fnum)),
        rx_apodization == "hann",
        float(t_offset_s),
        bool(coherence_weight),
    )
    return volume, {"x_mm": xs * 1e3, "y_mm": ys * 1e3, "z_mm": zs * 1e3}


def envelope_db(
    rf: npt.NDArray[np.floating],
    vmin: float | None = None,
) -> npt.NDArray[np.float64]:
    """Compute log-compressed Hilbert envelope.

    Parameters
    ----------
    rf : numpy.ndarray
        RF signal, shape ``(Nt,)`` or ``(Nt, N_lines)``.
    vmin : float, optional
        Minimum linear amplitude floor before log conversion (fraction of peak).
        ``None`` defaults to ``1e-20`` (no effective floor).  To clip at
        −60 dB, pass ``vmin=10**(-60/20)`` ≈ ``0.001``.

    Returns
    -------
    numpy.ndarray
        Log-compressed envelope in dB (peak = 0 dB), same shape as `rf`.
    """
    from scipy.signal import hilbert

    env = np.abs(hilbert(np.asarray(rf, dtype=np.float64), axis=0))
    return to_dB(env, vmin=vmin)
