"""
Step 2 — acquire the per-event RF with the Reception class (checkpointed).

Uses whatever step 1 selected (probe with its piezo impulse response,
phantom, diverging-wave sequence, drive burst) and runs the pulse-echo
simulation event by event:

- The probes carry their ``impulse_response`` (set in step 1's builders) and
  the bare drive is passed as ``excitation``: the Reception class folds
  ``exc ⊛ ir_tx ⊛ ir_rx`` in the frequency domain — the two-way band-limited
  pulse of a physical probe. Essential: without the impulse responses the
  elements are ideally broadband and the aperture impulse-response tails
  dominate the received spectrum (PSF +60 %, sidelobe skirt −22 → −10 dB).
- ``ReceptionSDI(method="spectral")``: at matrix channel counts the spectral
  pulse-echo kernel builds every receive channel's spectrum in one batched
  call — measured ~3x faster per event than the conventional path at 3025
  channels (and ~20x faster than Field II on the same machine).
- Every event (or scatterer chunk) is checkpointed to ``out/<scenario>/RF/``
  the moment it finishes (RF is linear in the scatterers, so chunk RFs sum
  sample-exactly). A crash costs one chunk; re-running resumes; a changed
  config (probe geometry, phantom, drive, sequence) refuses with a diff.
  CAVEAT: the fingerprint does NOT cover the impulse responses — if you
  change the pulse model, delete ``out/<scenario>/RF`` yourself.

Run with (pick the scenario in step 1 or via the SCENARIO env var):
    uv run examples/example21_rca_volume/step2_acquire_RF.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from step1_define_phantom_TX_RX import (
    C,
    DOWNSAMPLING,
    FS,
    RF_DIR,
    SC,
    SCENARIO,
    build_phantom,
    dw_events,
    excitation,
)

from pyfield.io import RFDataset
from pyfield.reception import ReceptionSDI

print(f"\n--- Example 21 · Step 2: acquisition, scenario '{SCENARIO}' ---\n")

scat_pos, scat_amp = build_phantom(SC)
print(
    f"{scat_pos.shape[0]:,d} scatterers ({SC['per_cell']}/cell at {SC['fc'] / 1e6:.0f} MHz)"
)
print(f"{len(SC['vs_mm'])} virtual sources at z = {SC['vs_mm'][0, 2]:.0f} mm")

# Separate TX/RX instances: reception applies RX delays per channel, so a
# shared object would leak each event's TX delays onto the receive channels.
tx, rx = SC["make_probe"](SC["fc"]), SC["make_probe"](SC["fc"])
sim = ReceptionSDI(
    tx,
    rx,
    c=C,
    fs=FS,
    excitation=excitation(SC["fc"]),
    method="spectral",
    verbose=True,
)

# The events carry their virtual source for the beamformer (step 3); the
# simulator — and the checkpoint fingerprint — only consume delays/apodization.
events = [
    {"delays": ev["delays"], "apodization": ev["apodization"]}
    for ev in dw_events(tx, SC["vs_mm"])
]

t_start = time.perf_counter()
rf, coords = sim.sequence_rf(
    scat_pos,
    scat_amp,
    events,
    downsampling=DOWNSAMPLING,
    out_path=RF_DIR,
    checkpoint_chunks=SC["checkpoint_chunks"],
)
t_total = time.perf_counter() - t_start
print(f"\nrf {rf.shape} stored in {RF_DIR}")
print(f"Acquisition wall time this run: {t_total / 3600:.2f} h")

RFDataset(RF_DIR).summary()
print("\nAcquisition complete — run step3_beamforming.py.")
