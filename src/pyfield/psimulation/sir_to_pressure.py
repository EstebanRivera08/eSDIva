"""Convert a spatial impulse response (SIR) to a pressure field."""

import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from pyfield.utilities.helper_functions import (
    reshape_to_mapped_points,
)


def _next_pow2(n):
    return 1 << (int(n - 1).bit_length())


def from_sir_to_monochromatic_pressure(
    h_sir, x, y, z, fc, fs, *, batch_size=2048, max_workers=None, verbose=False
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
    n_points = h_sir.shape[1]
    # Frequency vector
    freq_vect = np.linspace(0, fs, h_sir.shape[0], endpoint=False)
    idx = np.argmin((freq_vect - fc) ** 2)
    Hsir = np.zeros(n_points, dtype=np.float32)  # FFT(h_sir) at fc

    def _process_batch(start):
        """Evaluate ``|FFT(h_sir)|`` at ``fc`` for one batch of field points."""
        end = min(start + batch_size, n_points)
        fft_batch = np.fft.fft(h_sir[:, start:end], axis=0)
        return start, end, np.abs(fft_batch[idx, :])

    # Parallel loop
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for start, end, vals in executor.map(
            _process_batch, range(0, n_points, batch_size)
        ):
            Hsir[start:end] = vals

    # Reshape back to 3D grid
    Pressure_at_fc = reshape_to_mapped_points(x, y, z, Hsir)
    if verbose:
        print(
            f"Monochromatic pressure with shape {Pressure_at_fc.shape} computed from SIR in {time.time() - start_time:.2f} seconds..."
        )
    return Pressure_at_fc[0, :, :, :]


def from_sir_to_pressure(
    h_sir,
    x,
    y,
    z,
    fs,
    *,
    rho=1,
    excitation=None,
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
    batch_size : int, optional
        Number of field points processed per FFT batch. Default is 2048.
    max_workers : int, optional
        Maximum number of worker threads. ``None`` lets ``ThreadPoolExecutor``
        choose. Default is None.
    verbose : bool, optional
        If True, print timing information. Default is False.

    Returns
    -------
    (Nx, Ny, Nz, T) numpy.ndarray
        Transient pressure field on the 3D grid across ``T`` time samples.
    """
    # allow excitation to be None (no excitation -> identity)

    if excitation is None:
        excitation = None
    else:
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
            # derivative of excitation with respect to time: d/dt = fs * diff(samples)
            derivative_excitation = np.diff(excitation) * fs
            L = derivative_excitation.shape[0]
            # choose FFT length for linear convolution
            nfft = _next_pow2(T + L - 1)
            # compute frequency representation of excitation derivative (rfft)
            fft_dExcitation = np.fft.rfft(derivative_excitation, n=nfft)[:, np.newaxis]

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
                # broadcast multiply: (nfreqs, cols) * (nfreqs,1)
                fft_Pressure = H_batch * fft_dExcitation
                outputfft = np.fft.irfft(fft_Pressure, n=nfft, axis=0)
                # replicate previous behaviour: take first T samples (causal alignment)
                Pressure_flat = np.abs(outputfft[:T, :])
                return start, end, Pressure_flat

            # Parallel loop: collect blocks and write to Pressure_flat
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for start, end, out in executor.map(
                    _process_batch, range(0, n_points, batch_size)
                ):
                    Pressure_flat[:, start:end] = out.astype(np.float32, copy=False)

    except Exception as e:
        raise ValueError(f"Error FFT processing: {e}")

    # Reshape back to mapped points and scale by rho
    try:
        pressure_field = reshape_to_mapped_points(x, y, z, Pressure_flat) * rho
    except Exception as e:
        raise ValueError(f"Error reshaping pressure field: {e}")
    if verbose:
        print(
            f"Pressure with shape {pressure_field.shape} computed from SIR in {time.time() - start_time:.2f} seconds..."
        )
    return pressure_field
