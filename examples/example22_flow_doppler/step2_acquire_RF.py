"""
Step 2 — acquire the three sequences.

Each arm is one `sequence_rf` call over a moving phantom, checkpointed to its own
folder. The three share a phantom definition, a probe, a pulse and a Doppler PRF;
they differ only in how the emissions are spent.

    A  conventional   N_LINES x E_SHORT   one focused transmit per line
    B  ultrafast      N_ANGLES x E_SHORT  compounded plane waves
    C  sensitivity    N_ANGLES x E_LONG   A's budget, spent on ensemble

THE ONE SUBTLETY: EACH ARM MOVES THE BLOOD AT ITS OWN RATE
-----------------------------------------------------------
The phantom is rebuilt per arm because "one emission" means different things.
The compounded arms fire all `N_ANGLES` angles of a frame back to back at the
emission PRF, so consecutive emissions are `1/(N_ANGLES·PRF_doppler)` apart.
The conventional arm sends one beam per slow-time sample, so its consecutive
emissions are `1/PRF_doppler` apart — and its events are ordered LINE-MAJOR
(a whole packet on one line, then the next), which is how a scanner dwells on a
line long enough to measure its velocity.

Getting this wrong would silently scale every velocity in one arm by `N_ANGLES`,
and the comparison would look like a physics result instead of a bookkeeping
error.

WHY `sequence_rf` AND NOT A HAND-WRITTEN LOOP
---------------------------------------------
The same RF is reachable by calling `pulse_echo_rf` once per emission and
stacking — identical physics. What the loop cannot give you is checkpointing (a
crash costs one emission, not the run), a config fingerprint that refuses to mix
two different flows into one store, and `t0_per_event` for the beamformer. With
~1560 emissions here, that matters.

Run with:
    uv run examples/example22_flow_doppler/step2_acquire_RF.py
"""

import time

from step1_define_flow_phantom import (
    ARMS,
    E_LONG,
    E_SHORT,
    N_EMISSIONS_A,
    N_EMISSIONS_B,
    N_EMISSIONS_C,
    PRF_DOPPLER,
    PRF_EMISSION,
    RF_DIRS,
    build_phantom,
    build_simulator,
    compound_events,
    describe,
    focused_events,
)

from esdiva.io import RFDataset
from esdiva.utilities import estimate_sequence_runtime

print("\n--- Example 22 - Step 2: acquisition (3 arms) ---\n")
describe()

sim, tx, rx = build_simulator()

# Per arm: (events, how many emissions, the rate the blood advances at).
# The conventional arm advances at PRF_DOPPLER because it fires one beam per
# slow-time sample; the compounded arms advance at the faster emission PRF.
plans = {
    "A": (focused_events(tx, E_SHORT), N_EMISSIONS_A, PRF_DOPPLER),
    "B": (compound_events(tx, E_SHORT)[0], N_EMISSIONS_B, PRF_EMISSION),
    "C": (compound_events(tx, E_LONG)[0], N_EMISSIONS_C, PRF_EMISSION),
}

total_start = time.perf_counter()
for arm, (events, n_em, prf) in plans.items():
    info = ARMS[arm]
    print(f"\n{'=' * 68}")
    print(
        f"ARM {arm} - {info['label']}: {n_em} emissions (ensemble {info['ensemble']})"
    )
    print("=" * 68)
    assert len(events) == n_em, f"arm {arm}: {len(events)} events != {n_em}"

    positions, amplitudes, truth = build_phantom(n_em, prf)
    print(
        f"Phantom {positions.shape}  "
        f"({truth['n_tissue']} tissue + {truth['n_blood']} blood)"
    )

    # Size it on THIS machine first. Cost per emission is a property of the
    # hardware, so a timing quoted from any other computer - including this
    # repo's docs - is not a prediction for yours.
    estimate_sequence_runtime(sim, positions, n_em)

    RF_DIRS[arm].parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    rf, coords = sim.sequence_rf(positions, amplitudes, events, out_path=RF_DIRS[arm])
    print(f"  rf {rf.shape} in {time.perf_counter() - t0:.1f} s")

print(f"\n{'=' * 68}")
print(f"All three arms done in {(time.perf_counter() - total_start) / 60:.1f} min")
for arm in plans:
    RFDataset(RF_DIRS[arm]).summary()
print("\nAcquisition complete - run step3_doppler_processing.py")
