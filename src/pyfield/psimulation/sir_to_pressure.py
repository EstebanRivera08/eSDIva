"""Convert a spatial impulse response (SIR) to a pressure field."""

import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from pyfield.utilities.helper_functions import (
    reshape_to_mapped_points,
)

from .attenuation import causal_attenuation_tf


def _next_pow2(n):
    return 1 << (int(n - 1).bit_length())


def from_sir_to_monochromatic_pressure(
    h_sir,
    x,
    y,
    z,
    fc,
    fs,
    *,
    alpha0=None,
    freq_power=1.0,
    f0_hz=None,
    distances_m=None,
    batch_size=2048,
    max_workers=None,
    verbose=False,
):
    """
    Compute the monochromatic pressure field from the SIR at a given frequency.

    Batches field points and evaluates the FFT of the SIR in parallel threads,
    then extracts the frequency bin closest to ``fc``.

    Parameters
    ----------
    h_sir : (T, P) numpy.ndarray
        Spatial impulse response sampled at ``fs`` for ``P`` field points.
    x : (Nx,) numpy.ndarray
        Grid coordinates along the lateral axis (metres).
    y : (Ny,) numpy.ndarray
        Grid coordinates along the elevation axis (metres).
    z : (Nz,) numpy.ndarray
        Grid coordinates along the axial axis (metres).
    fc : float
        Center frequency at which to evaluate the pressure field (Hz).
    fs : float
        Sampling frequency of the SIR (Hz).
    alpha0 : float or None, optional
        Attenuation coefficient in dB/(MHz^y·cm).  ``None`` = no attenuation.
    freq_power : float, optional
        Power-law exponent y for attenuation. Default 1.0.
    f0_hz : float or None, optional
        Reference frequency in Hz (required for y = 1 K-K dispersion).
    distances_m : (P,) array_like or None, optional
        Propagation distance per field point in metres.  Required when
        ``alpha0`` is not ``None``.
    batch_size : int, optional
        Number of field points processed per FFT batch. Default is 2048.
    max_workers : int, optional
        Maximum number of worker threads. ``None`` lets ``ThreadPoolExecutor``
        choose. Default is None.
    verbose : bool, optional
        If True, print timing information. Default is False.

    Returns
    -------
    (Nx, Ny, Nz) numpy.ndarray
        Monochromatic pressure field magnitude at ``fc`` on the 3D grid.
    """
    start_time = time.time()
    T, n_points = h_sir.shape
    # Use rfft (real-valued SIR): output is (T//2+1, batch) complex64, halving
    # FFT batch memory vs full fft. fc < fs/2 always holds for ultrasound.
    freq_vect = np.fft.rfftfreq(T, d=1.0 / fs)
    idx = int(np.argmin(np.abs(freq_vect - fc)))
    Hsir = np.zeros(n_points, dtype=np.float32)

    do_attenuation = alpha0 is not None and distances_m is not None
    if do_attenuation:
        distances_arr = np.asarray(distances_m)
        fc_actual = float(freq_vect[idx])

    # numpy promotes float32 h_sir → float64 before rfft → complex128 output (16 B).
    # Cap sequential batch so rfft output (T//2+1, batch) complex128 ≤ 8 MB.
    # Sequential loop (no ThreadPoolExecutor) avoids concurrent allocations on
    # top of the already-large h_sir array.
    T_rfft = T // 2 + 1
    batch_size = min(batch_size, max(1, int(8 * 1024 * 1024 // (T_rfft * 16))))

    for start in range(0, n_points, batch_size):
        end = min(start + batch_size, n_points)
        fft_batch = np.fft.rfft(h_sir[:, start:end], axis=0)  # (T//2+1, cols)
        vals = np.abs(fft_batch[idx, :])
        if do_attenuation:
            H_att = causal_attenuation_tf(
                np.array([fc_actual]),
                distances_arr[start:end],
                alpha0,
                freq_power,
                f0_hz,
            )  # (cols, 1)
            vals = vals * np.abs(H_att[:, 0]).astype(np.float32)
        Hsir[start:end] = vals

    if x is None or y is None or z is None:
        if verbose:
            print(
                f"Monochromatic pressure with shape {Hsir.shape} (no meshgrid) "
                f"computed from SIR in {time.time() - start_time:.2f} seconds..."
            )
        return Hsir
    else:
        Pressure_at_fc = reshape_to_mapped_points(x, y, z, Hsir)
        if verbose:
            print(
                f"Monochromatic pressure with shape {Hsir.shape} computed from SIR in"
                f" {time.time() - start_time:.2f} seconds..."
            )
    return Pressure_at_fc[0]


def from_sir_to_pressure(
    h_sir,
    x,
    y,
    z,
    fs,
    *,
    rho=1,
    excitation=None,
    alpha0=None,
    freq_power=1.0,
    f0_hz=None,
    distances_m=None,
    batch_size=2048,
    max_workers=None,
    verbose=False,
):
    """
    Compute the transient pressure field from the SIR and an excitation pulse.

    Batched FFT convolution in the frequency domain. Both the SIR columns and
    the derivative of the excitation are zero-padded to a common FFT length
    ``nfft >= T + L - 1``, producing linear convolution results that mimic
    ``scipy.signal.fftconvolve(..., mode='full')[:T]``.

    Parameters
    ----------
    h_sir : (T, P) numpy.ndarray
        Spatial impulse response sampled at ``fs`` for ``P`` field points.
    x : (Nx,) numpy.ndarray
        Grid coordinates along the lateral axis (metres).
    y : (Ny,) numpy.ndarray
        Grid coordinates along the elevation axis (metres).
    z : (Nz,) numpy.ndarray
        Grid coordinates along the axial axis (metres).
    fs : float
        Sampling frequency of the SIR (Hz).
    rho : float, optional
        Density of the propagation medium in kg/m^3. Default is 1.
    excitation : (L,) numpy.ndarray, optional
        Excitation pulse samples. If None, the SIR itself is returned as the
        pressure (identity excitation). Default is None.
    alpha0 : float or None, optional
        Attenuation coefficient in dB/(MHz^y·cm).  ``None`` = no attenuation
        (bit-identical output to the unmodified function). Default is None.
    freq_power : float, optional
        Power-law exponent y for attenuation. Default 1.0.
    f0_hz : float or None, optional
        Reference frequency in Hz (required for y = 1 K-K dispersion).
    distances_m : (P,) array_like or None, optional
        Propagation distance per field point in metres.  Required when
        ``alpha0`` is not ``None``.
    batch_size : int, optional
        Number of field points processed per FFT batch. Default is 2048.
    max_workers : int, optional
        Maximum number of worker threads. ``None`` lets ``ThreadPoolExecutor``
        choose. Default is None.
    verbose : bool, optional
        If True, print timing information. Default is False.

    Returns
    -------
    (Nt, Nx, Ny, Nz) numpy.ndarray
        Transient pressure field on the 3D grid across ``Nt`` time samples.
    """
    if excitation is not None:
        if isinstance(excitation, (tuple, list)):
            excitation = np.array(excitation, dtype=np.float32)
        if excitation.ndim != 1:
            raise ValueError("excitation must be a 1D array.")

    try:
        start_time = time.time()
        T, n_points = h_sir.shape

        if excitation is None:
            Pressure_flat = h_sir
        else:
            # Backward diff d/dt matches compute_dh SDI convention: (v[n]-v[n-1])*fs.
            # Using forward diff (diff(exc)) introduces a 1-sample phase shift relative
            # to the per-element path that uses dh (backward-diff-based) directly.
            derivative_excitation = np.diff(excitation, prepend=0) * fs
            L = derivative_excitation.shape[0]
            # choose FFT length for linear convolution
            nfft = _next_pow2(T + L - 1)
            # compute frequency representation of excitation derivative (rfft)
            fft_dExcitation = np.fft.rfft(derivative_excitation, n=nfft)[:, np.newaxis]

            # Attenuation setup — only computed when alpha0 is provided.
            do_attenuation = alpha0 is not None and distances_m is not None
            if do_attenuation:
                distances_arr = np.asarray(distances_m)
                freqs_att = np.fft.rfftfreq(nfft, d=1.0 / fs)

            Pressure_flat = np.zeros((T, n_points), dtype=np.float32)

            def _process_batch(start):
                """Convolve one batch of SIR columns with the excitation derivative."""
                end = min(start + batch_size, n_points)
                cols = end - start
                # pad h_sir columns to nfft
                h_pad = np.zeros((nfft, cols), dtype=np.float64)
                h_pad[:T, :] = h_sir[:, start:end].astype(np.float64, copy=False)
                # rfft, multiply, irfft
                H_batch = np.fft.rfft(h_pad, axis=0)
                # broadcast multiply: (nfreqs, cols) * (nfreqs, 1)
                fft_Pressure = H_batch * fft_dExcitation
                if do_attenuation:
                    H_att = causal_attenuation_tf(
                        freqs_att,
                        distances_arr[start:end],
                        alpha0,
                        freq_power,
                        f0_hz,
                    )  # (cols, N_freq)
                    fft_Pressure = fft_Pressure * H_att.T  # (N_freq, cols)
                outputfft = np.fft.irfft(fft_Pressure, n=nfft, axis=0)
                # replicate previous behaviour: take first T samples (causal alignment)
                Pressure_batch = np.abs(outputfft[:T, :])
                return start, end, Pressure_batch

            # Parallel loop: collect blocks and write to Pressure_flat
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for start, end, out in executor.map(
                    _process_batch, range(0, n_points, batch_size)
                ):
                    Pressure_flat[:, start:end] = out.astype(np.float32, copy=False)

    except Exception as e:
        raise ValueError(f"Error FFT processing: {e}")

    # Reshape back to mapped points and scale by rho
    if x is None or y is None or z is None:
        if verbose:
            print(
                f"Pressure with shape {Pressure_flat.shape} (no meshgrid) computed from SIR in"
                f" {time.time() - start_time:.2f} seconds..."
            )
        return Pressure_flat
    else:
        pressure_field = reshape_to_mapped_points(x, y, z, Pressure_flat) * rho
        if verbose:
            print(
                f"Pressure with shape {pressure_field.shape} computed from SIR"
                f" in {time.time() - start_time:.2f} seconds..."
            )
        return pressure_field
