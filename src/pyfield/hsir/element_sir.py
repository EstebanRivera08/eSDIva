"""Per-element SIR computation.

Mirrors `farfield_rect_patch.py` but groups patches by element index, returning a
separate h_sir signal for each element rather than the sum over all patches.
"""

import numpy as np

from .farfield_rect_patch import compute_h_sir


def compute_h_sir_per_element(
    P,
    M,
    E,
    T,
    dt,
    time_grid,
    points,
    centers,
    wx,
    wy,
    inv_c,
    fs,
    apod,
    delays,
    sub_el_idx,
    eu=None,
    ev=None,
    method_flag=None,
):
    """Compute h_sir independently for each transducer element.

    Parameters
    ----------
    P : int
        Number of field points.
    M : int
        Total number of patches across all elements.
    E : int
        Number of elements.
    T : int
        Time axis length (samples).
    dt : float
        Time step (seconds).
    time_grid : (T,) float32
        Pre-computed time grid shared across all elements.
    points : (P, 3) float32
        Field point coordinates in metres.
    centers : (M, 3) float32
        Patch centre coordinates in metres.
    wx : (M,) float32
        Patch half-width in x (metres).
    wy : (M,) float32
        Patch half-width in y (metres).
    inv_c : float
        Inverse speed of sound 1/c (s/m).
    fs : float
        Sampling frequency (Hz).
    apod : (M,) float32
        Apodization weight per patch.
    delays : (M,) float32
        Delay per patch (seconds).
    sub_el_idx : (M,) int32
        Element index for each patch (0-based).
    eu : (M, 3) float32 or None
        Tangent unit vector u per patch. Passed to compute_h_sir.
    ev : (M, 3) float32 or None
        Tangent unit vector v per patch. Passed to compute_h_sir.
    method_flag : int or None
        0=naive, 1=sdi, None=auto. Forwarded to compute_h_sir.

    Returns
    -------
    h_per_elem : (P, E, T) float32
        SIR for each field point and element.
    """
    out = np.zeros((P, E, T), dtype=np.float32)
    for e in range(E):
        mask = sub_el_idx == e
        M_e = int(mask.sum())
        if M_e == 0:
            continue
        eu_e = eu[mask] if eu is not None else None
        ev_e = ev[mask] if ev is not None else None
        h_e, _ = compute_h_sir(
            P,
            M_e,
            T,
            dt,
            time_grid,
            points,
            centers[mask],
            wx[mask],
            wy[mask],
            inv_c,
            fs,
            apod[mask],
            delays[mask],
            method_flag,
            eu_e,
            ev_e,
        )
        out[:, e, :] = h_e  # (P, T)
    return out
