"""Delay-and-sum (DAS) beamforming for pulse-echo RF data."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from pyfield.utilities import to_dB


def das(
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

    The delay for element *e* is ``Δt_e = (|r_f − r_e| − |r_f − r_ref|) / c``,
    where *r_ref* is the centre element position.  A positive Δt means the
    echo arrives later in that channel; the interpolation reads ahead by
    ``Δt / dt`` samples to re-align it.

    Parameters
    ----------
    rf : numpy.ndarray
        Raw channel RF data, shape ``(Nt, E_rx)``, as returned by `Reception`.
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

    Nt = rf.shape[0]
    sample_idx = np.arange(Nt, dtype=np.float64)
    rf_das = np.zeros(Nt, dtype=np.float64)

    for e in range(rf.shape[1]):
        # Echo from focus is at sample (i + Δt/dt) in channel e relative to
        # the centre channel at sample i.  Read ahead to align.
        rf_das += np.interp(
            sample_idx + delta_t[e] / dt,
            sample_idx,
            rf[:, e].astype(np.float64),
            left=0.0,
            right=0.0,
        )

    return rf_das.astype(np.float32)


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
