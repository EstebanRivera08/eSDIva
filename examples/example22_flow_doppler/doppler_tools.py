"""
Doppler processing chain — demodulate, wall-filter, estimate.

eSDIva simulates the RF and beamforms it; turning an ensemble of beamformed
frames into a velocity is signal processing, not acoustics, so it lives here in
the example rather than in the package. The three functions are the conventional
processing chain and are named for it, so the steps map onto what any ultrasound
processing toolbox calls them:

    beamformed RF  ->  rf2iq  ->  wfilt  ->  iq2doppler  ->  velocity

The names are the conventional ones, and so are the signatures: `rf2iq(RF, Fs,
Fc)` demodulates to I/Q and estimates `Fc` itself when it is omitted, `wfilt`
removes the clutter, `iq2doppler` returns velocity and power. Code written
against another toolbox transfers with the argument order intact.

Each one is a few lines. Read them: there is nothing hidden, and every
assumption a Doppler number rests on is visible in this file.

WHAT EACH INTERMEDIATE OBJECT IS
--------------------------------
Four different things get loosely called "IQ" in ultrasound code. They are not
interchangeable, and a Doppler chain that mixes them up fails in ways that look
like physics:

    RF           real, band-pass, centred at fc.
    ANALYTIC     `scipy.signal.hilbert(rf)` - COMPLEX, still centred at fc.
                 Carries amplitude and phase. (The function name is misleading:
                 it returns the analytic signal, not the Hilbert transform.)
    BASEBAND IQ  analytic x exp(-j.2.pi.fc.t) - complex, centred at ZERO.
                 Carries amplitude and phase. This is what `rf2iq` returns, and
                 what quadrature demodulation means.
    ENVELOPE     |analytic| - REAL, phase DISCARDED. This is what a B-mode is
                 made of, and it is useless for Doppler: the phase is the signal.

The chain below runs on BASEBAND IQ throughout, never on the envelope.
"""

import numpy as np
from scipy.signal import hilbert


def rf2iq(rf, fs, fc=None, axis=-1, return_fc=False):
    """Demodulate a real RF signal to complex baseband IQ.

    Two operations, in this order: form the ANALYTIC signal with a Hilbert
    transform, then DOWN-MIX it by ``exp(-j·2π·fc·t)``. Together they are
    quadrature demodulation, and the result is baseband IQ - complex, centred at
    zero, carrying both amplitude and phase. It is NOT an envelope: ``abs()`` of
    this would throw the phase away, and the phase is the entire Doppler signal.

    An RF line is a real band-pass signal centred at `fc`. Its analytic signal
    (the Hilbert transform pair) carries the ECHO PHASE, and the down-mix shifts
    the carrier to zero so only the slowly varying envelope and phase remain. Demodulating does not change any velocity the
    ensemble encodes — at a fixed depth the down-mixing factor is the same
    constant for every emission — but baseband IQ is the representation the rest
    of the chain expects, and it makes the demodulation frequency explicit
    instead of implied.

    Run it AFTER beamforming, as the chain here does. The ordering is free -
    delay, the Hilbert transform and summation all commute along fast time - but
    beamforming signal that has ALREADY been down-mixed is not: baseband IQ has
    no carrier left, so aligning two channels then needs a delay AND a phase
    rotation ``exp(j·2π·fc·tau)``. Omitting the rotation sums the aperture with
    random carrier phase (measured: 67 % of the lumen power lost). `das_volume`
    is a real-valued delay-and-sum, so this ordering cannot make that mistake.

    Parameters
    ----------
    rf : numpy.ndarray
        Real signal, band-pass around `fc`.
    fs : float
        Sampling rate along `axis`, in Hz. For a BEAMFORMED line sampled in
        depth with step `dz`, one sample is a round trip of ``2·dz/c``, so
        ``fs = c/(2·dz)``.
    fc : float, optional
        Demodulation (carrier) frequency in Hz. Use the MEASURED centre
        frequency of the echo, not the probe's nominal rating. None estimates it
        from `rf` itself with `echo_center_frequency` over the full band, which
        is also the safest default here: it measures the carrier on the very
        signal being demodulated, which is the mistake this chain exists to
        avoid. MUST be given when `rf` is UNDERSAMPLED (bandpass sampling): the
        carrier is then aliased and cannot be recovered from the spectrum.
    axis : int, default -1
        Fast-time (depth) axis.
    return_fc : bool, default False
        Also return the carrier frequency used - useful when it was estimated,
        since the same value must be handed to `iq2doppler`.

    Returns
    -------
    iq : numpy.ndarray
        Complex baseband IQ, same shape as `rf`. Real part = in-phase (I),
        imaginary part = quadrature (Q).
    fc : float
        Carrier frequency in Hz. Returned only when ``return_fc=True``.

    Examples
    --------
    Carrier given, as in the shipped chain::

        iq = rf2iq(beamformed, fs_depth, f_echo)

    Carrier estimated from the data and handed back for the estimator::

        iq, fc = rf2iq(beamformed, fs_depth, return_fc=True)
        v, p = iq2doppler(wfilt(iq), prf, fc)
    """
    if fc is None:
        fc = echo_center_frequency(rf, 1.0 / fs, band=(0.0, fs / 2.0))
    analytic = hilbert(rf, axis=axis)
    t = np.arange(rf.shape[axis]) / fs
    shape = [1] * rf.ndim
    shape[axis] = -1
    iq = analytic * np.exp(-2j * np.pi * fc * t.reshape(shape))
    return (iq, float(fc)) if return_fc else iq


def echo_center_frequency(signal, dt, band=None):
    """Power-weighted centre frequency, in Hz, of whatever signal you pass.

    THIS IS NOT THE PROBE'S NOMINAL FREQUENCY, and using the nominal value is
    the most common way to get a Doppler velocity wrong. The pulse-echo chain
    band-passes the drive twice (transmit piezo, receive piezo) and the aperture
    adds its own frequency response, so the echo that comes back is centred
    somewhere else — and it drifts with depth as the aperture moves from near to
    far field. Since velocity scales as ``1/f``, a 10 % frequency error is a
    10 % velocity error on every voxel.

    MEASURE IT ON THE SIGNAL THE ESTIMATOR ACTUALLY PROCESSES. Delay-and-sum
    low-passes the data — it interpolates between RF samples and sums coherently
    across an aperture — so the BEAMFORMED signal is centred BELOW the raw
    channel RF. Measured on a 5 MHz linear array: 4.46 MHz on the channel data
    against 4.27 MHz after beamforming, and handing the estimator the channel
    figure biased every velocity low by 4.5 %. Passing the beamformed lines instead (with
    ``dt = 2·dz/c``, the round-trip time per depth sample) removed it — a
    plug-flow control went from 0.955 to 0.998 of its known velocity. It is also
    why Loupas's estimator takes its frequency from the very IQ it processes.

    The centroid is used rather than the spectral peak because the peak of a
    speckle spectrum is noisy, while the centroid barely moves when the
    integration band is changed.

    Parameters
    ----------
    signal : numpy.ndarray
        Signal with FAST TIME (or depth) along the last axis; leading axes are
        averaged over. Pass the BEAMFORMED lines when the frequency is destined
        for a Doppler estimator.
    dt : float
        Sampling interval along the last axis, in seconds. For beamformed lines
        sampled in depth that is the round-trip ``2·dz/c``, not the RF ``dt``.
    band : tuple[float, float], optional
        Integration band in Hz. None uses everything from DC to Nyquist, which
        is the safe default: a band must be wide enough to hold the WHOLE echo
        spectrum, and one hard-coded for a particular probe silently truncates a
        higher-frequency one. This had teeth - a band of (2, 10) MHz left over
        from a 5 MHz probe cut the top half off a 12.5 MHz echo and reported
        8.2 MHz instead of 10.5, which is a 28 % velocity error. Narrow it only
        to reject a known out-of-band artefact, and scale it to the probe
        (e.g. ``(0.3 * fc, 1.7 * fc)``) rather than writing absolute numbers.

    Returns
    -------
    float
        Centre frequency in Hz.
    """
    signal = np.asarray(signal)
    spec = np.abs(np.fft.rfft(signal, axis=-1)) ** 2
    spec = spec.reshape(-1, spec.shape[-1]).mean(axis=0)
    f = np.fft.rfftfreq(signal.shape[-1], dt)
    lo, hi = (0.0, f[-1]) if band is None else band
    keep = (f > lo) & (f < hi)
    return float((f[keep] * spec[keep]).sum() / spec[keep].sum())


def wfilt(iq, method="poly", order=0, axis=0):
    """Wall (clutter) filter: remove the echo of stationary tissue.

    Tissue outshines blood by 40-60 dB, so the flow signal is buried until the
    stationary part is removed. What makes that possible is that a still
    scatterer returns the SAME complex value in every emission, i.e. it sits at
    zero slow-time frequency, while moving blood does not.

    - ``"poly"`` fits and subtracts a polynomial of `order` along slow time.
      Order 0 subtracts the ensemble mean and removes only what is perfectly
      still; order 1-2 also removes a slow drift, which is what tissue motion
      from breathing or vessel-wall pulsation looks like over a short ensemble.
      Higher orders eat into slow flow, so raising it trades clutter rejection
      for low-velocity sensitivity.
    - ``"eig"`` drops the `order`+1 strongest singular components of the
      ensemble. Clutter is both strong and spatially coherent, so it collects in
      the first few components; this adapts to moving tissue in a way a fixed
      polynomial cannot, at the cost of assuming clutter dominates the energy.

    Parameters
    ----------
    iq : numpy.ndarray
        Complex IQ ensemble, slow time along `axis`.
    method : {'poly', 'eig'}, default 'poly'
        Filter family.
    order : int, default 0
        Polynomial degree, or the number of suppressed components minus one.
    axis : int, default 0
        Slow-time (emission) axis.

    Returns
    -------
    numpy.ndarray
        The ensemble with the clutter component removed.
    """
    iq = np.moveaxis(iq, axis, 0)
    n_ens = iq.shape[0]
    if method == "poly":
        n = np.arange(n_ens) - (n_ens - 1) / 2.0
        basis = np.vstack([n**k for k in range(order + 1)]).T
        # Projector onto the clutter subspace; subtracting it leaves the rest.
        proj = basis @ np.linalg.pinv(basis)
        out = iq - np.tensordot(proj, iq, axes=(1, 0))
    elif method == "eig":
        flat = iq.reshape(n_ens, -1)
        u, s, vh = np.linalg.svd(flat, full_matrices=False)
        s = s.copy()
        s[: order + 1] = 0.0
        out = (u * s) @ vh
        out = out.reshape(iq.shape)
    else:
        raise ValueError(f"unknown wall-filter method {method!r}; use 'poly' or 'eig'.")
    return np.moveaxis(out, 0, axis)


def iq2doppler(iq, prf, fc, c=1540.0, axis=0):
    """Mean axial velocity per voxel from the lag-one slow-time autocorrelation.

    Between two emissions a scatterer receding at ``v_z`` lengthens the round
    trip by ``2·v_z/PRF``, turning the echo phase by ``4π·fc·v_z/(c·PRF)``.
    Averaging the lag-one products BEFORE taking the angle weights every sample
    by its own amplitude, which is what makes the estimate hold up in speckle
    where individual samples can be near zero.

    Sign convention: POSITIVE velocity means receding from the probe. A receding
    target lengthens the path and so turns the phase negative, hence the minus.

    Only the AXIAL component is recoverable. Flow at an angle `theta` to the
    beam returns ``v·cos(theta)``, and flow across the beam returns nothing at
    all — a Doppler image of a perpendicular vessel is legitimately blank.

    The estimate wraps beyond ``v_nyquist = c·PRF/(4·fc)``, where the
    inter-emission phase leaves ±π: faster flow folds and reads with the wrong
    sign.

    Parameters
    ----------
    iq : numpy.ndarray
        Wall-filtered complex IQ ensemble, slow time along `axis`.
    prf : float
        Emission rate in Hz — the slow-time sampling rate.
    fc : float
        Centre frequency of the received echo in Hz (measured, not nominal).
    c : float, default 1540.0
        Speed of sound in m/s.
    axis : int, default 0
        Slow-time (emission) axis.

    Returns
    -------
    velocity : numpy.ndarray
        Mean axial velocity in m/s, with `axis` removed.
    power : numpy.ndarray
        Mean power of the filtered ensemble — the colour-Doppler power that says
        where there is any flow signal worth trusting a velocity from.
    """
    iq = np.moveaxis(iq, axis, 0)
    r1 = np.sum(np.conj(iq[:-1]) * iq[1:], axis=0)
    velocity = -c * prf / (4 * np.pi * fc) * np.angle(r1)
    return velocity, (np.abs(iq) ** 2).mean(axis=0)


def doppler_spectrum(iq_gate, prf, fc, c=1540.0, nfft=256):
    """Velocity spectrum of one sample volume (spectral / pulsed-wave Doppler).

    The slow-time signal at one voxel is a sum over every scatterer in the
    sample volume, each turning the phase at its own rate, so its spectrum IS
    the velocity distribution there. A parabolic (laminar) profile spreads power
    from zero up to the axis peak; a plug profile would give one narrow line.

    The ensemble is short, so the transform is zero-padded: 24 raw emissions
    resolve only ~3 cm/s, which is too coarse to read an edge off. Padding
    interpolates the same information onto a readable axis — it adds no
    resolution, it just stops the envelope landing between bins.

    Parameters
    ----------
    iq_gate : (n_emissions,) numpy.ndarray
        Wall-filtered complex IQ at one voxel.
    prf : float
        Emission rate in Hz.
    fc : float
        Measured echo centre frequency in Hz.
    c : float, default 1540.0
        Speed of sound in m/s.
    nfft : int, default 256
        Zero-padded transform length.

    Returns
    -------
    velocity_axis : (nfft,) numpy.ndarray
        Axial velocity in m/s, ascending (positive = receding).
    power : (nfft,) numpy.ndarray
        Spectral power, normalised to a peak of 1.
    """
    n_ens = len(iq_gate)
    spec = np.fft.fftshift(np.abs(np.fft.fft(iq_gate * np.hanning(n_ens), nfft)) ** 2)
    # Negated to match the sign convention of `iq2doppler`: a receding target
    # turns the phase negative, so its energy sits at negative slow-time
    # frequency and must be flipped to read as a positive velocity.
    v_axis = -np.fft.fftshift(np.fft.fftfreq(nfft, 1 / prf)) * c / (2 * fc)
    order = np.argsort(v_axis)
    spec = spec[order]
    return v_axis[order], spec / spec.max()


def max_velocity_envelope(v_axis, spectrum, threshold=0.5):
    """Fastest velocity carrying real power — the clinical peak-velocity read.

    For a laminar profile the spectrum runs from zero up to the axis peak, so
    the MEAN velocity sits well below the peak and it is this fast edge that
    corresponds to the maximum flow speed. The threshold is a fraction of the
    spectral peak (0.5 = -3 dB); a lower one reaches further into the noise.

    Returns NaN when nothing clears the threshold on the positive side.
    """
    fast = v_axis[(v_axis > 0) & (spectrum >= threshold)]
    return float(fast.max()) if fast.size else float("nan")


if __name__ == "__main__":
    # Self-check on a synthetic ensemble whose answer is known exactly: a
    # 5 MHz tone burst moving at a fixed axial velocity between emissions.
    fs_, fc_, prf_, c_, v_ = 40e6, 5e6, 5e3, 1540.0, 0.10
    n_, nt_ = 16, 512
    t_ = np.arange(nt_) / fs_
    tau_ = 2 * v_ * np.arange(n_) / (prf_ * c_)  # round-trip delay per emission
    gate_ = np.exp(-(((t_ - 6e-6) / 4e-7) ** 2))
    rf_ = np.array([gate_ * np.cos(2 * np.pi * fc_ * (t_ - d)) for d in tau_])

    iq_, fc_est = rf2iq(rf_, fs_, return_fc=True)
    assert abs(fc_est / fc_ - 1) < 0.02, f"carrier estimate off: {fc_est / 1e6:.3f} MHz"

    # Same burst at 12.5 MHz. This is the case a band hard-coded for a 5 MHz
    # probe gets wrong - it would report ~8 MHz and scale every velocity by 1.5.
    hf_ = 12.5e6
    rf_hf = np.array([gate_ * np.cos(2 * np.pi * hf_ * (t_ - d)) for d in tau_])
    fc_hf = echo_center_frequency(rf_hf, 1.0 / fs_)
    assert abs(fc_hf / hf_ - 1) < 0.02, (
        f"high-frequency carrier off: {fc_hf / 1e6:.3f} MHz"
    )
    assert np.allclose(iq_, rf2iq(rf_, fs_, fc_est)), "return_fc changed the result"

    v_est = iq2doppler(iq_, prf_, fc_est, c=c_)[0][nt_ // 2]
    assert abs(v_est / v_ - 1) < 0.03, f"velocity off: {v_est * 100:.2f} cm/s"

    # The down-mix must not change the velocity: its factor depends only on
    # depth, so it is identical in every emission and cancels in the lag-one
    # product. Guards against someone "simplifying" it into the estimator.
    v_ana = iq2doppler(hilbert(rf_, axis=-1), prf_, fc_est, c=c_)[0][nt_ // 2]
    assert np.isclose(v_est, v_ana), "down-mix is not phase-neutral"

    print(
        f"OK  fc {fc_est / 1e6:.3f} MHz (true {fc_ / 1e6:.1f})   "
        f"v {v_est * 100:.2f} cm/s (true {v_ * 100:.1f})"
    )
