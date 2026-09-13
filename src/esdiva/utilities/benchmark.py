"""Measure, on THIS machine, how long a planned simulation will take.

A simulation's cost is dominated by the number of points it evaluates — scatterers for
pulse-echo RF, field points for an emitted field — times the number of emissions (or
fields) planned, but the constant of proportionality is a property of the machine —
core count, memory bandwidth, BLAS threading — and varies by well over an order of
magnitude between a laptop and a compute node. Timings quoted in a paper, a README or
by a colleague are therefore useless for planning YOUR run.
`estimate_sequence_runtime` times a few short probe simulations with the transducers,
medium and excitation you are actually going to use, and projects from those.
"""

import os
import platform
import time

import numpy as np

from .helper_functions import create_3D_spatial_grid_from_points

# Warm-up size: large enough to exercise the real kernels, small enough to be
# instant. The first call also pays the numba JIT compilation, which would
# otherwise be charged to the first timed probe and inflate every estimate.
_WARMUP_POINTS = 32


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
    points_mm,
    n_emissions,
    *,
    probe_fractions=(0.05, 0.15),
    amplitudes=None,
    verbose=True,
):
    """Project the wall time of a planned run, measured on this machine.

    Times two short runs of the configured simulator on random SUBSETS of the real
    points — scatterers for `Reception` (pulse-echo RF), field points for `Emission`
    (a pressure field) — fits the cost of one run as ``t(N) = fixed + per_point·N``,
    and extrapolates to all the points and the planned number of runs. Because the
    probes use the actual transducers, medium, excitation and grid, the result carries
    this machine's real constant rather than someone else's.

    The cost per point is not perfectly constant — it grows with the point count
    before levelling off, as depth binning starts to pay off — so an estimate
    extrapolated from very small fractions runs OPTIMISTIC. Probe with the largest
    fraction you can afford to wait for; the default pair costs about a fifth of one
    full run.

    Parameters
    ----------
    sim : Emission, Reception or ReceptionConventional
        The configured simulator — same transducers, medium, excitation and mode
        (monochromatic/transient, `method`) the real run will use.
    points_mm : dict or (N, 3) or (N_events, N, 3) numpy.ndarray
        The points to simulate, in mm: a phantom for reception (a moving cloud is
        probed at its first emission; every emission costs the same), or a field
        grid dict / point array for emission.
    n_emissions : int
        Total runs planned: emissions for reception (a compounded Doppler sequence is
        ``n_angles × n_frames``, NOT the number of frames), fields for emission (e.g.
        one per steering angle or focus).
    probe_fractions : tuple[float, float], default (0.05, 0.15)
        Fractions of the points to time. Two points determine the fixed cost and the
        per-point slope.
    amplitudes : (N,) numpy.ndarray, optional
        Scattering amplitudes. Only the COUNT of points affects timing, so this changes
        nothing; accepted so the call mirrors the real one.
    verbose : bool, default True
        Print the measurement and the projection.

    Returns
    -------
    dict
        ``"seconds_per_emission"``, ``"total_seconds"``, ``"total_hours"``,
        ``"per_point_s"``, ``"fixed_s"``, ``"n_points"``, ``"n_emissions"``,
        ``"probes"`` (the timed points) and ``"machine"``.

    Examples
    --------
    Size a 10-angle, 200-frame ultrafast Doppler block before committing::

        est = estimate_sequence_runtime(sim, phantom_mm, n_emissions=10 * 200)
        if est["total_hours"] > 12:
            print("overnight job - use out_path= to checkpoint it")

    Size a 3-D transient field sweep over 21 steering angles::

        est = estimate_sequence_runtime(Emission(tx, excitation=pulse), grid, 21)
    """
    if isinstance(points_mm, dict):
        points_mm = create_3D_spatial_grid_from_points(points_mm)[3] * 1e3
    pos = np.asarray(points_mm, dtype=np.float32)
    if pos.ndim == 3:  # moving cloud: every emission costs the same
        pos = pos[0]
    n_pts = pos.shape[0]
    n_emissions = int(n_emissions)
    run = getattr(sim, "pulse_echo_rf", sim)  # Reception core, or Emission.__call__
    unit = "scatterers" if hasattr(sim, "pulse_echo_rf") else "field points"

    # Absorb the numba JIT cost, which is paid once per process and would
    # otherwise be billed to the first probe.
    run(pos[:_WARMUP_POINTS])

    rng = np.random.default_rng(0)
    probes = []
    for frac in probe_fractions:
        n = min(max(int(round(frac * n_pts)), _WARMUP_POINTS), n_pts)
        idx = rng.choice(n_pts, size=n, replace=False)
        t0 = time.perf_counter()
        run(pos[idx])
        probes.append((n, time.perf_counter() - t0))

    (n_a, t_a), (n_b, t_b) = probes[0], probes[-1]
    if n_b == n_a:  # degenerate (tiny cloud): fall back to a flat rate
        per_pt, fixed = t_b / max(n_b, 1), 0.0
    else:
        per_pt = (t_b - t_a) / (n_b - n_a)
        fixed = t_a - per_pt * n_a
    per_emission = max(fixed + per_pt * n_pts, min(t_a, t_b))
    total = per_emission * n_emissions

    result = {
        "seconds_per_emission": per_emission,
        "total_seconds": total,
        "total_hours": total / 3600.0,
        "per_point_s": per_pt,
        "fixed_s": fixed,
        "n_points": n_pts,
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
            print(f"  probe {n:>7,d} {unit} : {t:8.2f} s")
        print(f"  -> {per_emission:.2f} s per run at {n_pts:,d} {unit}")
        print(f"  -> {n_emissions:,d} runs = {_fmt_duration(total)}")
        if total > 6 * 3600 and unit == "scatterers":
            print("  Long run: pass out_path= to sequence_rf so a crash resumes.")
        if max(probe_fractions) < 0.1:
            print("  NOTE: probed on small subsets - the true cost per point grows")
            print(f"  with the number of {unit}, so this projection is optimistic.")
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
