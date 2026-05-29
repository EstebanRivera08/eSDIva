"""Causal power-law attenuation transfer functions and distance helpers.

Standalone module — no dependency on Emission or Reception.  Both import
from here.

Physics reference: `.claude/rules/attenuation.md` and `physics-context.md §10`.

Notes
-----
**Unit convention**

* User-facing: ``alpha0`` in dB/(MHz^y·cm) — matches clinical literature.
* Internal: ``alpha0`` in Np/(Hz^y·m) — convert via ``convert_alpha0_to_nepers``.

**Attenuation model**

Causal power-law (Szabo 1994, Holm 2019).  Always includes Kramers-Kronig
dispersion — cost is zero, accuracy is strictly better than non-causal.

Frequency convention: formulas use linear frequency f [Hz], not angular
frequency omega.  The ``causal_attenuation_tf`` unit conversion uses
f-convention (no 2pi factor in the exponent), matching the dB/(MHz^y·cm)
user unit.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Unit conversion
# ---------------------------------------------------------------------------


def convert_alpha0_to_nepers(alpha0_dB: float, y: float) -> float:
    """Convert attenuation coefficient from dB/(MHz^y·cm) to Np/(Hz^y·m).

    Parameters
    ----------
    alpha0_dB : float
        Attenuation in dB/(MHz^y·cm) (user-facing unit).
    y : float
        Power-law exponent.

    Returns
    -------
    float
        Attenuation coefficient in Np/(Hz^y·m).

    Notes
    -----
    Conversion:
    ``alpha0_neper = alpha0_dB * 100 / (20 * log10(e) * 1e6^y)``

    * ``100``: 1/cm → 1/m (100 cm = 1 m).
    * ``20 * log10(e) ≈ 8.686``: dB → Np (1 Np = 8.686 dB).
    * ``(1e6)^y``: MHz^y → Hz^y.
    """
    return float(alpha0_dB) * 100.0 / (20.0 * np.log10(np.e) * (1e6 ** float(y)))


# ---------------------------------------------------------------------------
# Core transfer function
# ---------------------------------------------------------------------------


def causal_attenuation_tf(
    freqs_hz: np.ndarray,
    distances_m: np.ndarray,
    alpha0_dB: float,
    y: float,
    f0_hz: float,
) -> np.ndarray:
    """Causal power-law attenuation transfer function H_att(f, d).

    Absorption and Kramers–Kronig dispersion combined.

    General case (y ≠ 1):

    .. code-block:: text

        H(f, d) = exp(-α₀ |f|^y d) · exp(-j α₀ |f|^y tan(yπ/2) d)

    Special case (y = 1, O'Donnell 1981):

    .. code-block:: text

        H(f, d) = exp(-α₀ |f| d) · exp(-j (2α₀/π) f ln(|f|/f₀) d)

    Where α₀ is in Np/(Hz^y·m) (converted internally from dB/(MHz^y·cm)).

    Parameters
    ----------
    freqs_hz : (N_freq,) ndarray
        Frequency array in Hz (e.g. from ``numpy.fft.rfftfreq``).
    distances_m : ndarray, shape (...)
        Propagation distances in metres.  Any leading shape is accepted —
        the function broadcasts to return ``(..., N_freq)``.
    alpha0_dB : float
        Attenuation coefficient in dB/(MHz^y·cm).
        Pass ``0`` or ``None`` for no attenuation (returns all-ones array).
    y : float
        Power-law exponent (tissue: 1.0–1.3).
    f0_hz : float
        Reference frequency in Hz (transducer centre frequency).  Used only
        for the y = 1 logarithmic dispersion term.

    Returns
    -------
    numpy.ndarray
        Attenuation transfer function H, shape ``(..., N_freq)``,
        complex128.  ``|H| <= 1``.

    Notes
    -----
    * DC (f = 0): H = 1 (no attenuation at zero frequency — correct limit).
    * ``alpha0_dB = 0``: H = 1 everywhere (identity).
    """
    if alpha0_dB is None or alpha0_dB == 0:
        freqs = np.asarray(freqs_hz, dtype=np.float64)
        dist = np.asarray(distances_m, dtype=np.float64)
        return np.ones((*dist.shape, freqs.shape[0]), dtype=np.complex128)

    alpha0 = convert_alpha0_to_nepers(float(alpha0_dB), float(y))
    freqs = np.asarray(freqs_hz, dtype=np.float64)  # (N_freq,)
    dist = np.asarray(distances_m, dtype=np.float64)  # (...)
    freq_abs = np.abs(freqs)  # (N_freq,)

    # Broadcast: dist[..., np.newaxis] * freq_term[np.newaxis, ...]
    dist_e = dist[..., np.newaxis]  # (..., 1)

    if abs(float(y) - 1.0) < 1e-10:
        # Special case y = 1: logarithmic K-K dispersion (O'Donnell 1981).
        f0 = float(f0_hz)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_ratio = np.where(freq_abs > 0.0, np.log(freq_abs / f0), 0.0)
        absorption = np.exp(-alpha0 * freq_abs * dist_e)
        # f * ln(|f|/f0) → 0 at DC (handled by where above when freq=0 → log_ratio=0)
        phase = -(2.0 * alpha0 / np.pi) * freqs * log_ratio * dist_e
        H = absorption * np.exp(1j * phase)
    else:
        # General case y ≠ 1: Szabo 1994.
        freq_pow_y = freq_abs ** float(y)  # (N_freq,)
        absorption = np.exp(-alpha0 * freq_pow_y * dist_e)  # (..., N_freq)
        tan_term = np.tan(float(y) * np.pi / 2.0)
        # sign(f) ensures causal dispersion for negative-frequency components.
        phase = -alpha0 * np.sign(freqs) * freq_pow_y * tan_term * dist_e
        H = absorption * np.exp(1j * phase)

    return H  # complex128, shape (..., N_freq)


# ---------------------------------------------------------------------------
# Distance computation helpers
# ---------------------------------------------------------------------------


def compute_attenuation_distances(
    field_points_m: np.ndarray,
    transducer_center_m: np.ndarray,
    patch_centers_m: np.ndarray | None = None,
    mode: str = "per_point",
) -> np.ndarray:
    """Propagation distance for attenuation.

    Parameters
    ----------
    field_points_m : (P, 3) ndarray
        Field point coordinates in metres.
    transducer_center_m : (3,) ndarray
        Transducer geometric centre in metres.
    patch_centers_m : (M, 3) ndarray or None
        Patch centre coordinates in metres.  Required when ``mode="per_patch"``.
    mode : {"per_point", "per_patch"}
        Distance model:

        * ``"per_point"`` (fast, approximate): ``d_p = |r_p - r_tx_center|``,
          shape ``(P,)``.
        * ``"per_patch"`` (accurate near-field): ``d_{pm} = |r_p - r_m|``,
          shape ``(P, M)``.

    Returns
    -------
    numpy.ndarray
        Distances in metres.  Shape ``(P,)`` for ``per_point``, ``(P, M)``
        for ``per_patch``.

    Raises
    ------
    ValueError
        If ``mode="per_patch"`` and ``patch_centers_m`` is None, or if
        ``mode`` is unknown.
    """
    field_points_m = np.asarray(field_points_m, dtype=np.float64)  # (P, 3)
    transducer_center_m = np.asarray(transducer_center_m, dtype=np.float64)  # (3,)

    if mode == "per_point":
        return np.linalg.norm(field_points_m - transducer_center_m, axis=1)  # (P,)

    if mode == "per_patch":
        if patch_centers_m is None:
            raise ValueError("patch_centers_m required for mode='per_patch'.")
        patch_centers_m = np.asarray(patch_centers_m, dtype=np.float64)  # (M, 3)
        # (P, M, 3) → norm → (P, M)
        diff = field_points_m[:, np.newaxis, :] - patch_centers_m[np.newaxis, :, :]
        return np.linalg.norm(diff, axis=2)

    raise ValueError(f"mode must be 'per_point' or 'per_patch', got '{mode}'.")


def reduce_patch_distances_to_element(
    distances_pm: np.ndarray,
    sub_el_idx: np.ndarray,
    n_elements: int,
    reduce: str = "mean",
) -> np.ndarray:
    """Reduce per-patch distances (P, M) to per-element distances (P, E).

    Parameters
    ----------
    distances_pm : (P, M) ndarray
        Per-patch distances from ``compute_attenuation_distances(mode='per_patch')``.
    sub_el_idx : (M,) int32 ndarray
        Patch-to-element index from ``compute_sub_elem_attributes``.
    n_elements : int
        Total number of elements E.
    reduce : {"mean", "min", "max"}
        Reduction strategy over patches belonging to the same element.

        * ``"mean"``: average distance (good when patches cluster tightly).
        * ``"min"``: minimum distance (conservative — least attenuation).
        * ``"max"``: maximum distance (most attenuation).

    Returns
    -------
    numpy.ndarray
        One representative distance per field-point / element pair,
        shape ``(P, E)``.

    Raises
    ------
    ValueError
        If ``reduce`` is not one of the supported strategies.
    """
    distances_pm = np.asarray(distances_pm, dtype=np.float64)  # (P, M)
    sub_el_idx = np.asarray(sub_el_idx, dtype=np.int32)  # (M,)
    P = distances_pm.shape[0]
    result = np.zeros((P, n_elements), dtype=np.float64)

    if reduce not in ("mean", "min", "max"):
        raise ValueError(f"reduce must be 'mean', 'min', or 'max', got '{reduce}'.")

    for e in range(n_elements):
        mask = sub_el_idx == e
        if not mask.any():
            continue
        d_e = distances_pm[:, mask]  # (P, count_e)
        if reduce == "mean":
            result[:, e] = d_e.mean(axis=1)
        elif reduce == "min":
            result[:, e] = d_e.min(axis=1)
        else:
            result[:, e] = d_e.max(axis=1)

    return result  # (P, E)


def compute_reception_distances(
    scatterer_positions_m: np.ndarray,
    tx_center_m: np.ndarray,
    rx_element_centers_m: np.ndarray,
) -> np.ndarray:
    """Round-trip distances for per-element Reception attenuation.

    Two-path model:
    ``d_total[p, e] = |r_s_p - r_tx| + |r_s_p - r_rx_e|``

    TX path is isotropic (same for all RX elements).
    RX path is per-element (each element receives from a different distance).

    Parameters
    ----------
    scatterer_positions_m : (P, 3) ndarray
        Scatterer positions in metres.
    tx_center_m : (3,) ndarray
        TX transducer geometric centre in metres.
    rx_element_centers_m : (E_rx, 3) ndarray
        RX element centre positions in metres (one per element).

    Returns
    -------
    numpy.ndarray
        Total round-trip distance per scatterer-element pair (metres),
        shape ``(P, E_rx)``.  Feed directly into ``causal_attenuation_tf``
        to get H_att of shape ``(P, E_rx, N_freq)``.
    """
    scatterer_positions_m = np.asarray(
        scatterer_positions_m, dtype=np.float64
    )  # (P, 3)
    tx_center_m = np.asarray(tx_center_m, dtype=np.float64)  # (3,)
    rx_element_centers_m = np.asarray(
        rx_element_centers_m, dtype=np.float64
    )  # (E_rx, 3)

    # TX path: (P,)
    d_tx = np.linalg.norm(scatterer_positions_m - tx_center_m, axis=1)

    # RX path: (P, E_rx)
    diff = (
        scatterer_positions_m[:, np.newaxis, :]  # (P, 1, 3)
        - rx_element_centers_m[np.newaxis, :, :]  # (1, E_rx, 3)
    )  # (P, E_rx, 3)
    d_rx = np.linalg.norm(diff, axis=2)  # (P, E_rx)

    return d_tx[:, np.newaxis] + d_rx  # (P, E_rx)
