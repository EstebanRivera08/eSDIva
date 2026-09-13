"""Additive receiver noise for simulated RF.

eSDIva's RF is noiseless: the SIR model computes the echo an ideal receiver
would record, and every scatterer contributes exactly its Born-approximation
signal. Real channel data always sits on the thermal noise of the amplifier,
and a study whose conclusion depends on detectability — Doppler sensitivity,
contrast at depth, how many emissions must be averaged — cannot be answered
without it. This adds that noise explicitly, so it is a stated assumption of the
experiment rather than a property of the simulator.
"""

import numpy as np


def add_noise(rf, snr_db, *, reference=None, rng=None):
    """Add white Gaussian receiver noise at a given channel SNR.

    The noise is independent in every sample, channel and emission — the model
    of thermal noise in the receive electronics. Its level is set from a
    reference signal amplitude so that

        SNR_dB = 20 · log10(rms(reference) / sigma)

    and by default the reference is the RMS of `rf` itself, i.e. the SNR is
    quoted against the echo actually present.

    IMPORTANT when comparing sequences. Noise is a property of the RECEIVER, not
    of the transmit scheme, so two acquisitions being compared must be given the
    SAME noise level in absolute terms. Letting each derive its own sigma from
    its own RMS would quietly hand the weaker sequence a quieter amplifier and
    destroy the comparison. Compute `reference` once from one acquisition and
    pass that same value for all of them.

    Parameters
    ----------
    rf : numpy.ndarray
        Simulated RF of any shape, e.g. ``(Erx, Nt)`` from `pulse_echo_rf` or
        ``(N_events, Erx, Nt)`` from `sequence_rf`.
    snr_db : float
        Signal-to-noise ratio in dB, as an amplitude ratio (20·log10).
    reference : float, optional
        RMS amplitude the SNR is referenced to. None uses ``rms(rf)``. Pass an
        explicit value to give several acquisitions one common noise level.
    rng : numpy.random.Generator or int, optional
        Generator or seed, for reproducible noise.

    Returns
    -------
    numpy.ndarray
        `rf` plus noise, as float32.

    Examples
    --------
    One acquisition, 20 dB channel SNR::

        noisy = add_noise(rf, 20, rng=0)

    Two sequences compared fairly — one noise level, derived once::

        ref = float(np.sqrt((rf_a.astype(np.float64) ** 2).mean()))
        a = add_noise(rf_a, 6, reference=ref, rng=1)
        b = add_noise(rf_b, 6, reference=ref, rng=2)
    """
    rf = np.asarray(rf)
    generator = (
        rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    )
    if reference is None:
        reference = float(np.sqrt((rf.astype(np.float64) ** 2).mean()))
    sigma = float(reference) * 10.0 ** (-float(snr_db) / 20.0)
    noise = generator.normal(0.0, sigma, rf.shape)
    return (rf + noise).astype(np.float32)


def rf_rms(rf):
    """RMS amplitude of an RF array — the reference `add_noise` quotes SNR against.

    Compute this ONCE on one acquisition and pass it to `add_noise` as
    ``reference`` for every acquisition being compared, so they all sit on the
    same absolute noise floor.

    Parameters
    ----------
    rf : numpy.ndarray
        RF data of any shape.

    Returns
    -------
    float
        Root-mean-square amplitude over all samples.
    """
    return float(np.sqrt((np.asarray(rf).astype(np.float64) ** 2).mean()))
