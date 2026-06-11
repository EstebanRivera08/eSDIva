"""Per-element pulse-echo SIR computation.

Mirrors `transducer_sir_pe_sdi.py` but returns Δδ_pe for each RX element
separately rather than requiring a pre-filtered patch set.
"""

import numpy as np

from .transducer_sir_pe_sdi import compute_pe_sdi


def compute_pe_sdi_per_element(
    points,
    tx_centers,
    tx_wx,
    tx_wy,
    tx_apod,
    tx_delays,
    rx_centers,
    rx_wx,
    rx_wy,
    rx_apod,
    rx_delays,
    inv_c,
    t0,
    T,
    fs,
    dt,
    rx_sub_el_idx,
    n_rx_elements,
    *,
    tx_eu=None,
    tx_ev=None,
    rx_eu=None,
    rx_ev=None,
):
    """Compute the raw Δδ_pe delta train independently for each RX element.

    Parameters
    ----------
    points : (P, 3) float32
        Scatterer positions in metres.
    tx_centers : (M_tx, 3) float32
        TX patch centres in metres.
    tx_wx : (M_tx,) float32
        TX patch width in x (metres).
    tx_wy : (M_tx,) float32
        TX patch width in y (metres).
    tx_apod : (M_tx,) float32
        TX apodization per patch.
    tx_delays : (M_tx,) float32
        TX delay per patch (seconds).
    rx_centers : (M_rx, 3) float32
        All RX patch centres in metres.
    rx_wx : (M_rx,) float32
        RX patch width in x (metres).
    rx_wy : (M_rx,) float32
        RX patch width in y (metres).
    rx_apod : (M_rx,) float32
        RX apodization per patch.
    rx_delays : (M_rx,) float32
        RX delay per patch (seconds).
    inv_c : float
        Inverse speed of sound 1/c (s/m).
    t0 : float
        PE time grid start (seconds).
    T : int
        PE time axis length (samples).
    fs : float
        Sampling frequency (Hz).
    dt : float
        Time step (seconds).
    rx_sub_el_idx : (M_rx,) int32
        RX element index for each RX patch (0-based).
    n_rx_elements : int
        Number of RX elements.
    tx_eu : (M_tx, 3) float32 or None
        TX tangent u vectors.
    tx_ev : (M_tx, 3) float32 or None
        TX tangent v vectors.
    rx_eu : (M_rx, 3) float32 or None
        RX tangent u vectors.
    rx_ev : (M_rx, 3) float32 or None
        RX tangent v vectors.

    Returns
    -------
    delta_pe_per_elem : (P, n_rx_elements, T) float32
        Raw two-way pulse-echo delta train Δδ_pe per scatterer and RX element.
    """
    P = points.shape[0]
    out = np.zeros((P, n_rx_elements, T), dtype=np.float32)
    for e in range(n_rx_elements):
        mask = rx_sub_el_idx == e
        if not mask.any():
            continue
        rx_eu_e = rx_eu[mask] if rx_eu is not None else None
        rx_ev_e = rx_ev[mask] if rx_ev is not None else None
        delta_e = compute_pe_sdi(
            points,
            tx_centers,
            tx_wx,
            tx_wy,
            tx_apod,
            tx_delays,
            rx_centers[mask],
            rx_wx[mask],
            rx_wy[mask],
            rx_apod[mask],
            rx_delays[mask],
            inv_c,
            t0,
            T,
            fs,
            dt,
            tx_eu=tx_eu,
            tx_ev=tx_ev,
            rx_eu=rx_eu_e,
            rx_ev=rx_ev_e,
        )
        out[:, e, :] = delta_e  # (P, T)
    return out
