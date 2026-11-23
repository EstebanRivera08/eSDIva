"""Small signal helpers shared by simulations and their analysis."""

import numpy as np
import numpy.typing as npt


def tone_burst(fc_Hz: float, n_cycles: float, fs: float) -> npt.NDArray[np.float32]:
    """Hann-windowed sine burst: the standard short ultrasound pulse.

    `n_cycles` periods of a sine at the centre frequency `fc_Hz`, tapered by a
    Hann window so the spectrum has no sharp sidelobes. Fewer cycles give a
    shorter pulse (better axial resolution) and a wider bandwidth. Use it as a
    drive `excitation` or as a piezo `impulse_response`.

    Parameters
    ----------
    fc_Hz : float
        Centre frequency in Hz.
    n_cycles : float
        Pulse length in periods of `fc_Hz`.
    fs : float
        Sampling frequency in Hz; must match the simulator's `fs`.

    Returns
    -------
    (L,) numpy.ndarray
        The burst, float32, `L = ceil(n_cycles·fs/fc_Hz)` samples, peak ≤ 1.
    """
    t = np.arange(0, n_cycles / fc_Hz, 1.0 / fs)
    return (np.sin(2 * np.pi * fc_Hz * t) * np.hanning(t.size)).astype(np.float32)


def fwhm_mm(profile: np.ndarray, axis_mm: np.ndarray) -> float:
    """Full width at half maximum (−6 dB in amplitude) of a beam profile.

    The standard resolution measure: the distance between the two points where
    the amplitude falls to half its peak, linearly interpolated between
    samples. Pass an amplitude (|p| or an envelope), not a power or a dB curve.

    Parameters
    ----------
    profile : (N,) numpy.ndarray
        Amplitude profile with a single main lobe.
    axis_mm : (N,) numpy.ndarray
        Position of each sample in mm, increasing.

    Returns
    -------
    float
        Width in mm. If the lobe touches the edge of the profile, the edge is
        used, so the width is a lower bound.
    """
    p = np.abs(np.asarray(profile, dtype=float))
    x = np.asarray(axis_mm, dtype=float)
    i, half = int(np.argmax(p)), p.max() / 2
    left, right = i, i
    while left > 0 and p[left - 1] >= half:
        left -= 1
    while right < p.size - 1 and p[right + 1] >= half:
        right += 1
    x_left = (
        x[left]
        if left == 0
        else np.interp(half, p[left - 1 : left + 1], x[left - 1 : left + 1])
    )
    x_right = (
        x[right]
        if right == p.size - 1
        else np.interp(half, p[right : right + 2][::-1], x[right : right + 2][::-1])
    )
    return float(x_right - x_left)
