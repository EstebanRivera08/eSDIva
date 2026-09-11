"""Measure, on THIS machine, how long a planned acquisition will take.

A pulse-echo simulation's cost is dominated by the number of scatterers times
the number of emissions, but the constant of proportionality is a property of
the machine — core count, memory bandwidth, BLAS threading — and varies by well
over an order of magnitude between a laptop and a compute node. Timings quoted
in a paper, a README or by a colleague are therefore useless for planning YOUR
run. `estimate_sequence_runtime` times a few short probe simulations with the
transducers, medium and excitation you are actually going to use, and projects
from those.
"""

import os
import platform
import time

import numpy as np

# Warm-up size: large enough to exercise the real kernels, small enough to be
# instant. The first call also pays the numba JIT compilation, which would
# otherwise be charged to the first timed probe and inflate every estimate.
_WARMUP_SCATTERERS = 32


def _machine_summary():
    """One-line description of the machine the estimate applies to."""
    threads = os.environ.get("NUMBA_NUM_THREADS") or os.environ.get("OMP_NUM_THREADS")
    return {
        "platform": f"{platform.system()} {platform.machine()}",
        "processor": platform.processor() or "unknown",
        "cpu_count": os.cpu_count(),
        "thread_env": threads or "default",
    }


def estimate_sequence_runtime(
    sim,
    scatterer_positions_mm,
    n_emissions,
    *,
    probe_fractions=(0.05, 0.15),
    amplitudes=None,
    verbose=True,
):
    """Project the wall time of a planned sequence, measured on this machine.

    Times two short pulse-echo runs on random SUBSETS of the real scatterer
    cloud, fits the per-emission cost as ``t(N) = fixed + per_scatterer·N``, and
    extrapolates to the full cloud and the planned number of emissions. Because
    the probes use the actual transducers, medium, excitation and phantom, the
    result carries this machine's real constant rather than someone else's.

    The cost per scatterer is not perfectly constant — it grows with cloud size
    before levelling off, as depth binning starts to pay off — so an estimate
    extrapolated from very small fractions runs OPTIMISTIC. Probe with the
    largest fraction you can afford to wait for; the default pair costs about
    a fifth of one full emission.

    Parameters
    ----------
    sim : Reception or ReceptionConventional
        The configured simulator — same transducers, medium and excitation the
        real run will use. Its `method` matters: kernels differ in cost.
    scatterer_positions_mm : (N_scat, 3) or (N_events, N_scat, 3) numpy.ndarray
        The phantom you intend to simulate. A moving cloud is probed at its
        first emission; every emission costs the same.
    n_emissions : int
        Total emissions planned — for a compounded Doppler sequence this is
        ``n_angles × n_frames``, NOT the number of frames.
    probe_fractions : tuple[float, float], default (0.05, 0.15)
        Fractions of the cloud to time. Two points determine the fixed cost and
        the per-scatterer slope.
    amplitudes : (N_scat,) numpy.ndarray, optional
        Scattering amplitudes. Only the COUNT of scatterers affects timing, so
        this changes nothing; accepted so the call mirrors the real one.
    verbose : bool, default True
        Print the measurement and the projection.

    Returns
    -------
    dict
        ``"seconds_per_emission"``, ``"total_seconds"``, ``"total_hours"``,
        ``"per_scatterer_s"``, ``"fixed_s"``, ``"n_scatterers"``,
        ``"n_emissions"``, ``"probes"`` (the timed points) and ``"machine"``.

    Examples
    --------
    Size a 10-angle, 200-frame ultrafast Doppler block before committing::

        est = estimate_sequence_runtime(sim, phantom_mm, n_emissions=10 * 200)
        if est["total_hours"] > 12:
            print("overnight job - use out_path= to checkpoint it")
    """
    pos = np.asarray(scatterer_positions_mm)
    if pos.ndim == 3:  # moving cloud: every emission costs the same
        pos = pos[0]
    n_scat = pos.shape[0]
    n_emissions = int(n_emissions)

    # Absorb the numba JIT cost, which is paid once per process and would
    # otherwise be billed to the first probe.
    sim.pulse_echo_rf(pos[:_WARMUP_SCATTERERS])

    rng = np.random.default_rng(0)
    probes = []
    for frac in probe_fractions:
        n = max(int(round(frac * n_scat)), _WARMUP_SCATTERERS)
        n = min(n, n_scat)
        idx = rng.choice(n_scat, size=n, replace=False)
        t0 = time.perf_counter()
        sim.pulse_echo_rf(pos[idx])
        probes.append((n, time.perf_counter() - t0))

    (n_a, t_a), (n_b, t_b) = probes[0], probes[-1]
    if n_b == n_a:  # degenerate (tiny cloud): fall back to a flat rate
        per_scat, fixed = t_b / max(n_b, 1), 0.0
    else:
        per_scat = (t_b - t_a) / (n_b - n_a)
        fixed = t_a - per_scat * n_a
    per_emission = max(fixed + per_scat * n_scat, min(t_a, t_b))
    total = per_emission * n_emissions

    result = {
        "seconds_per_emission": per_emission,
        "total_seconds": total,
        "total_hours": total / 3600.0,
        "per_scatterer_s": per_scat,
        "fixed_s": fixed,
        "n_scatterers": n_scat,
        "n_emissions": n_emissions,
        "probes": probes,
        "machine": _machine_summary(),
    }

    if verbose:
        m = result["machine"]
        print(
            f"Runtime estimate on this machine ({m['platform']}, "
            f"{m['cpu_count']} cores, threads={m['thread_env']})"
        )
        for n, t in probes:
            print(f"  probe {n:>7,d} scatterers : {t:8.2f} s")
        print(f"  -> {per_emission:.2f} s per emission at {n_scat:,d} scatterers")
        print(f"  -> {n_emissions:,d} emissions = {_fmt_duration(total)}")
        if total > 6 * 3600:
            print("  Long run: pass out_path= to sequence_rf so a crash resumes.")
        if max(probe_fractions) < 0.1:
            print("  NOTE: probed on small subsets - the true cost per scatterer")
            print("  grows with cloud size, so this projection is optimistic.")
    return result


def _fmt_duration(seconds):
    """Human-readable duration: seconds, minutes, hours or days."""
    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 5400:
        return f"{seconds / 60:.1f} min"
    if seconds < 48 * 3600:
        return f"{seconds / 3600:.1f} h"
    return f"{seconds / 86400:.1f} days"
