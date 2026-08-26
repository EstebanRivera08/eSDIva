"""
Example 20: Speckle Phantom — make_phantom() + focused B-mode of a cyst

Simulating tissue in pulse-echo means simulating SPECKLE: many sub-wavelength
scatterers at random positions whose echoes interfere. `make_phantom` builds
such a cloud from an echogenicity image — here the classic cyst phantom
(anechoic cyst + hyperechoic lesion in a speckle background).

  1. Draw an echogenicity map (2-D x-z image: 0 = cyst, 1 = background,
     4 = bright lesion)
  2. `make_phantom` — random positions + Gaussian amplitudes scaled by the map
  3. `sim.show()` — preview the cloud (points fade with |amplitude|, so the
     cyst appears as a hole and the lesion as a bright clot)
  4. B-mode: loop `scan_focusline` over lateral positions, align each line's
     time axis, log-compress

Run with:
    uv run examples/example20_phantom_simulation.py
"""

import matplotlib.pyplot as plt
import numpy as np
from config import FIG_FOLDER, SAVE_FIG

from sondi.reception import Reception
from sondi.transducers import LinearArrayTransducer
from sondi.utilities import align_to_common_time, make_phantom

# ============================================================================
# CONFIGURATION
# ============================================================================
C = 1540.0
FS = 100e6
FC = 5e6
PULSE_CYCLES = 2

# Phantom box (mm). Thin elevation slab: scatterers only need to fill the
# elevation beamwidth, and fewer scatterers per cell means faster simulation.
BOX = {"x_extent": [-12.0, 12.0], "y_extent": [-1.0, 1.0], "z_extent": [12.0, 38.0]}
# Fully developed speckle needs >= ~5-10 scatterers per resolution cell. Here the
# cell is ~0.6 mm (lateral, lambda*F#) x ~1.5 mm (elevation beamwidth at focus)
# x ~0.3 mm (half the 2-cycle pulse) ~= 0.27 mm^3, and the box is 1248 mm^3:
# 25000 scatterers ~= 5.4 per cell. Fewer and the texture and contrast numbers
# become artefacts of the particular random draw.
N_SCATTERERS = 25000

CYST_CENTER = (-4.0, 25.0)  # (x, z) mm — anechoic
CYST_RADIUS = 3.0
LESION_CENTER = (4.0, 25.0)  # (x, z) mm — hyperechoic
LESION_RADIUS = 2.0
LESION_GAIN = 4.0

N_LINES = 33
LINE_X = np.linspace(-8.0, 8.0, N_LINES)  # lateral scan positions (mm)
FOCUS_Z = 25.0  # transmit/receive focal depth (mm)

print("\n--- Example 20: Speckle phantom + focused B-mode ---\n")

if SAVE_FIG:
    FIG_FOLDER.mkdir(exist_ok=True)

# ============================================================================
# STEP 1: ECHOGENICITY MAP (2-D x-z image spanning the phantom box)
# ============================================================================
nx, nz = 256, 256
xm = np.linspace(*BOX["x_extent"], nx)
zm = np.linspace(*BOX["z_extent"], nz)
X, Z = np.meshgrid(xm, zm, indexing="ij")

emap = np.ones((nx, nz))
emap[(X - CYST_CENTER[0]) ** 2 + (Z - CYST_CENTER[1]) ** 2 < CYST_RADIUS**2] = 0.0
emap[(X - LESION_CENTER[0]) ** 2 + (Z - LESION_CENTER[1]) ** 2 < LESION_RADIUS**2] = (
    LESION_GAIN
)

# Show the echogenicity map
plt.figure(figsize=(5, 5))
plt.imshow(
    emap.T,
    extent=[*BOX["x_extent"], BOX["z_extent"][1], BOX["z_extent"][0]],
    cmap="gray",
    aspect="equal",
)
plt.title("Echogenicity map")
plt.xlabel("x (mm)")
plt.ylabel("Depth z (mm)")
if SAVE_FIG:
    plt.savefig(str(FIG_FOLDER / "ex20_phantom_map.png"), dpi=150)
plt.show()

# ============================================================================
# STEP 2: RANDOM SCATTERER CLOUD FROM THE MAP
# ============================================================================
# Positions are uniform in the box; each amplitude is N(0,1) × map(r). A
# regular grid would NOT work here: its periodicity returns coherent lattice
# echoes instead of speckle. For fully developed speckle aim for >= ~5-10
# scatterers per resolution cell (~ lambda·F# laterally x half pulse axially).
scat_pos, scat_amp = make_phantom(BOX, N_SCATTERERS, echogenicity_map=emap, seed=2026)
print(f"Phantom: {N_SCATTERERS} scatterers, |amp| in [0, {np.abs(scat_amp).max():.2f}]")

# ============================================================================
# STEP 3: PROBE + SIMULATOR + 3-D PREVIEW
# ============================================================================
tx = LinearArrayTransducer(
    n_elements=64,
    element_width_mm=0.25,
    element_height_mm=5.0,
    kerf_mm=0.05,
    no_sub_x=1,
    no_sub_y=4,
    frequency_Hz=FC,
)

t_pulse = np.arange(0, PULSE_CYCLES / FC, 1.0 / FS)
excitation = (np.sin(2 * np.pi * FC * t_pulse) * np.hanning(len(t_pulse))).astype(
    np.float32
)

# A physical probe band-passes the signal twice: drive ⊛ TX piezo impulse
# response ⊛ RX piezo impulse response. Without the IR the elements are ideally
# broadband and the aperture's low-frequency diffraction tails dominate the
# received spectrum — the PSF widens well beyond lambda*z/D and the sidelobe
# skirt fills the anechoic cyst. tx doubles as rx here, so one assignment
# applies the IR on both transmit and receive.
tx.impulse_response = excitation.copy()

sim = Reception(tx, tx, c=C, fs=FS, excitation=excitation, verbose=False)

# Preview: cyst shows as a hole (amplitude 0 → fully transparent), lesion as
# a bright clot. This is the check that the map, box and units line up.
sim.show(
    scat_pos,
    scat_amp,
    TX_color="blue",
    legend=False,
    save_path=str(FIG_FOLDER / "ex20_phantom_setup.png") if SAVE_FIG else None,
)

# ============================================================================
# STEP 4: B-MODE — ONE FOCUSED LINE PER LATERAL POSITION
# ============================================================================
# scan_focusline refocuses TX and RX on [x, 0, FOCUS_Z] and beamforms on
# receive inside the kernel — one envelope line per call. Each line has its
# OWN t0 (the delay profile changes with the lateral focus), so the lines are
# interpolated onto a common time axis before display.
lines = []
for i, xl in enumerate(LINE_X):
    env, coords = sim.scan_focusline(
        [xl, 0.0, FOCUS_Z],
        scat_pos,
        scat_amp,
        FoverD=2.0,
        apodization_type="hanning",
    )
    lines.append((env, coords))
    print(f"line {i + 1:2d}/{N_LINES}  x = {xl:+.1f} mm  ({env.shape[0]} samples)")

common_t, aligned = align_to_common_time(lines)
bmode = np.stack(aligned, axis=1)  # (Nt, N_LINES)

# ============================================================================
# STEP 5: LOG-COMPRESS + DISPLAY NEXT TO THE MAP
# ============================================================================
depth_mm = common_t * C / 2.0 * 1e3  # two-way time -> depth
bmode_db = 20 * np.log10(bmode / (bmode.max() + 1e-30) + 1e-6)

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 5), sharey=True)

ax0.imshow(
    emap.T,
    extent=[*BOX["x_extent"], BOX["z_extent"][1], BOX["z_extent"][0]],
    cmap="gray",
    aspect="equal",
)
ax0.set_title("Echogenicity map")
ax0.set_xlabel("x (mm)")
ax0.set_ylabel("Depth z (mm)")

im = ax1.imshow(
    bmode_db,
    extent=[LINE_X[0], LINE_X[-1], depth_mm[-1], depth_mm[0]],
    cmap="gray",
    vmin=-50,
    vmax=0,
    aspect="equal",
)
ax1.set_ylim(BOX["z_extent"][1], BOX["z_extent"][0])
ax1.set_title(f"B-mode ({N_LINES} focused lines, focus {FOCUS_Z:.0f} mm)")
ax1.set_xlabel("x (mm)")
plt.colorbar(im, ax=ax1, label="dB")
plt.tight_layout()

if SAVE_FIG:
    plt.savefig(str(FIG_FOLDER / "ex20_phantom_bmode.png"), dpi=150)

plt.show()

print("\nDone.")
