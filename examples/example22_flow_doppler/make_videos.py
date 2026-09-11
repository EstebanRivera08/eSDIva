"""
Two short videos built from the products steps 3 and 4 already stored.

Nothing is re-simulated: `doppler_results.npz` holds the beamformed IQ for every
slow-time sample, before AND after the wall filter, so the story is already on
disk as a movie rather than a still.

    video 1  "the flow you cannot see"    B-mode -> wall filter -> colour ->
                                          power -> zoom on the lumen + profile
    video 2  "two ways to spend a budget" conventional line-by-line against
                                          compounded plane waves, with the
                                          emission count and the frame rate

Rendered 1080 x 1350 (the 4:5 a feed gives the most room to), as MP4 and a
palette-optimised GIF. ffmpeg comes from the `imageio_ffmpeg` wheel already in
the venv, so there is nothing to install.

Run with:
    uv run examples/example22_flow_doppler/make_videos.py
"""

import json
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg
import matplotlib

matplotlib.use("Agg")
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
matplotlib.rcParams["animation.ffmpeg_path"] = FFMPEG

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FFMpegWriter, FuncAnimation  # noqa: E402
from scipy.signal import hilbert  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from step1_define_flow_phantom import (  # noqa: E402
    ANGLES_DEG,
    C,
    DX_MM,
    E_SHORT,
    GRID_MM,
    IQ_FILE,
    METRICS_FILE,
    N_ANGLES,
    N_LINES,
    PRF_DOPPLER,
    PRF_EMISSION,
    RF_DIRS,
    THETA_DEG,
    VESSEL_CENTER_MM,
    V_PEAK,
    compound_events,
    focused_events,
    line_positions_mm,
    make_probe,
)

from esdiva.beamforming import das_volume  # noqa: E402
from esdiva.io import RFDataset  # noqa: E402

FPS = 30
DPI = 135
W_PX, H_PX = 1080, 1350
FIGSIZE = (W_PX / DPI, H_PX / DPI)  # 4:5 - the tallest
# Instagram shows without cropping, and LinkedIn renders it unchanged.
OUT = Path(__file__).parent / "out"

INK, DIM, BG, ACCENT, COOL = "#f2f4f7", "#9aa4b2", "#0b0e13", "#ff7a45", "#3ba7ff"

plt.rcParams.update(
    {
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "savefig.facecolor": BG,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": DIM,
        "ytick.color": DIM,
        "font.size": 12,
    }
)


def _ease(t):
    """Smoothstep — linear fades look mechanical, this one breathes."""
    t = float(np.clip(t, 0.0, 1.0))
    return t * t * (3.0 - 2.0 * t)


def _db(x, floor=-45.0):
    a = np.abs(x)
    return np.maximum(20.0 * np.log10(a / a.max() + 1e-12), floor)


def _envelope(rf):
    """B-mode envelope of a REAL beamformed signal, depth along the last axis.

    Taking ``abs()`` of the beamformed RF is not a B-mode: the RF crosses zero
    twice per cycle, so its magnitude comes back striped at half the wavelength
    and reads as grain rather than speckle. The envelope is the magnitude of the
    ANALYTIC signal, which is what the rest of the example forms (`rf2iq` runs
    the same Hilbert transform before demodulating).
    """
    return np.abs(hilbert(np.asarray(rf, dtype=np.float64), axis=-1))


def _frame_axes(fig, extent, rect=(0.115, 0.40, 0.84, None)):
    """Image axes sized so the data keeps its true aspect inside `rect`."""
    span_x, span_z = extent[1] - extent[0], extent[2] - extent[3]
    left, bottom, w, h = rect
    if h is None:
        h = w * (W_PX / H_PX) * (span_z / span_x)
    ax = fig.add_axes([left, bottom, w, h])
    ax.set_xlabel("x (mm)", fontsize=11)
    ax.set_ylabel("z (mm)", fontsize=11)
    for sp in ax.spines.values():
        sp.set_color("#2a3140")
    return ax


def _save(fig, update, n_frames, name):
    """Visually lossless MP4, then a palette-optimised GIF made from it."""
    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000 / FPS)
    mp4 = OUT / f"{name}.mp4"
    anim.save(
        mp4,
        writer=FFMpegWriter(
            fps=FPS,
            codec="libx264",
            extra_args=["-crf", "18", "-pix_fmt", "yuv420p", "-preset", "slow"],
        ),
        dpi=DPI,
    )
    plt.close(fig)
    gif = OUT / f"{name}.gif"
    # Two-pass palette: a global GIF palette is far smaller AND far cleaner than
    # per-frame quantisation, which is what makes a naive GIF export enormous.
    vf = "fps=12,scale=600:-1:flags=lanczos"
    subprocess.run(
        [
            FFMPEG,
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(mp4),
            "-vf",
            f"{vf},palettegen=stats_mode=diff",
            str(OUT / "_pal.png"),
        ],
        check=True,
    )
    subprocess.run(
        [
            FFMPEG,
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(mp4),
            "-i",
            str(OUT / "_pal.png"),
            "-lavfi",
            f"{vf}[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=3",
            str(gif),
        ],
        check=True,
    )
    (OUT / "_pal.png").unlink(missing_ok=True)
    print(
        f"  {mp4.name}  {mp4.stat().st_size / 1e6:.1f} MB"
        f"   |   {gif.name}  {gif.stat().st_size / 1e6:.1f} MB"
    )


def _beamformed_for_video2():
    """Beamform what each transmit ACTUALLY produced, cached to disk.

    Nothing here is mocked up: the conventional panel shows the real beamformed
    line that each focused transmit returned, and the ultrafast panel shows the
    real single-angle image of each plane wave and their coherent sum. Showing a
    single plane wave next to the compound is the honest way to make the point -
    one angle alone is a poor image, and it is the SUM that synthesises a
    transmit focus everywhere.
    """
    cache = OUT / "video2_beamformed.npz"
    if cache.exists():
        return np.load(cache)

    print("  beamforming the per-line and per-angle images (cached after this)")
    probe = make_probe()
    conv_events = focused_events(probe, 1)  # one per line, un-repeated
    _, frame_events = compound_events(probe, 1)

    # --- conventional: one focused transmit -> one line -------------------
    rf, co = RFDataset(RF_DIRS["A"]).load_all()
    x_lines = line_positions_mm()
    cols = []
    for i, x_line in enumerate(x_lines):
        n = i * E_SHORT  # first emission of that line's packet
        line_grid = dict(GRID_MM, x_extent=[float(x_line), float(x_line) + DX_MM])
        vol, gc = das_volume(
            rf[n : n + 1],
            {"dt": co["dt"], "t0_per_event": co["t0_per_event"][n : n + 1]},
            [conv_events[i]],
            probe,
            line_grid,
            c=C,
            fnum=1.0,
        )
        cols.append(vol[0, 0, :])
    conv = np.stack(cols)  # (N_LINES, Nz)
    z_mm = gc["z_mm"]

    # --- ultrafast: each plane wave on its own, then their coherent sum ----
    rf_b, co_b = RFDataset(RF_DIRS["B"]).load_all()
    singles = []
    for k in range(N_ANGLES):
        vol, gc = das_volume(
            rf_b[k : k + 1],
            {"dt": co_b["dt"], "t0_per_event": co_b["t0_per_event"][k : k + 1]},
            [frame_events[k]],
            probe,
            GRID_MM,
            c=C,
            fnum=1.0,
        )
        singles.append(vol[:, 0, :])
    singles = np.stack(singles)  # (N_ANGLES, Nx, Nz)

    np.savez_compressed(
        cache,
        conv=conv,
        singles=singles,
        x_lines=x_lines,
        x_mm=gc["x_mm"],
        z_mm=z_mm,
    )
    return np.load(cache)


# ===========================================================================
# VIDEO 1 - the flow you cannot see
# ===========================================================================
def video_one():
    d = np.load(IQ_FILE)
    m = json.loads(METRICS_FILE.read_text())
    iq, iq_flow = d["C_iq"], d["C_iq_flow"]
    x, z = d["C_x_mm"], d["C_z_mm"]
    v_map, power, mask = d["C_v_map"] * 100, d["C_power"], d["C_mask"]
    n_slow = iq.shape[0]
    # Cropped to +/-5 mm so the frame is square: 10 mm across against 11 mm
    # deep. The vessel crosses the whole of it, so nothing of the story is lost.
    XCROP = 5.0
    extent = [-XCROP, XCROP, z[-1], z[0]]
    keep = (x >= -XCROP) & (x <= XCROP)
    x = x[keep]
    iq, iq_flow = iq[:, keep], iq_flow[:, keep]
    v_map, power, mask = v_map[keep], power[keep], mask[keep]

    bmode = np.stack([_db(iq[k]) for k in range(n_slow)])
    fl_db = np.stack([_db(iq_flow[k], floor=-30.0) for k in range(n_slow)])
    # Per-pixel opacity from the flow's own strength: noise stays invisible and
    # the lumen glows, which is what makes the wall filter read as a reveal
    # rather than a purple wash over the whole box.
    fl_a = np.clip((fl_db + 26.0) / 26.0, 0.0, 1.0) ** 1.6
    p_db = 10 * np.log10(power / power.max() + 1e-12)
    vlim = float(np.nanmax(np.abs(np.where(mask, v_map, np.nan))))

    acts = np.cumsum([75, 70, 80, 60, 110, 45])
    n_frames = int(acts[-1])

    fig = plt.figure(figsize=FIGSIZE)
    # Square image up top, a working band underneath - never dead space.
    ax = _frame_axes(fig, extent, rect=(0.260, 0.470, 0.480, 0.422))
    im_b = ax.imshow(
        bmode[0].T,
        extent=extent,
        cmap="gray",
        vmin=-45,
        vmax=0,
        aspect="equal",
        interpolation="bilinear",
    )
    im_f = ax.imshow(
        fl_db[0].T,
        extent=extent,
        cmap="afmhot",
        vmin=-30,
        vmax=0,
        aspect="equal",
        interpolation="bilinear",
    )
    im_f.set_alpha(np.zeros_like(fl_a[0].T))
    im_v = ax.imshow(
        np.where(mask, v_map, np.nan).T,
        extent=extent,
        cmap="coolwarm",
        vmin=-vlim,
        vmax=vlim,
        aspect="equal",
        interpolation="nearest",
        alpha=0.0,
    )
    im_p = ax.imshow(
        np.where(mask, p_db, np.nan).T,
        extent=extent,
        cmap="hot",
        vmin=-20,
        vmax=0,
        aspect="equal",
        interpolation="nearest",
        alpha=0.0,
    )

    title = fig.text(0.5, 0.955, "", ha="center", fontsize=23, fontweight="bold")
    sub = fig.text(0.5, 0.918, "", ha="center", fontsize=13.5, color=DIM)
    fig.text(
        0.115,
        0.026,
        "eSDIva  ·  example 22  ·  flow & Doppler",
        fontsize=11,
        color="#5b6472",
    )

    axp = fig.add_axes([0.19, 0.100, 0.66, 0.195])
    axp.set_visible(False)

    # The same band first carries the cost trade, so the viewer meets the
    # headline claim before the profile lands on it.
    axt = fig.add_axes([0.19, 0.100, 0.66, 0.195])
    axt.set_facecolor(BG)
    em = {a: m["arms"][a]["n_emissions"] for a in ("A", "B")}
    mean = {a: m["arms"][a]["v_roi_mean_m_s"] * 100 for a in ("A", "B")}
    axt.barh([1, 0], [em["A"], em["B"]], height=0.5, color=[COOL, ACCENT])
    for yy, a, lab in ((1, "A", "conventional"), (0, "B", "ultrafast")):
        axt.text(em["A"] * 0.02, yy + 0.33, lab, fontsize=12, color=INK, va="bottom")
        # One label per bar, always past its end: the ultrafast bar is 8x
        # shorter, so anything placed inside it collides with the text.
        axt.text(
            em[a] + em["A"] * 0.03,
            yy,
            f"{em[a]} emissions   ·   mean {mean[a]:.1f} cm/s",
            fontsize=11,
            color=DIM,
            va="center",
        )
    axt.set_xlim(0, em["A"] * 1.75)
    axt.set_ylim(-0.5, 1.75)
    axt.set_yticks([])
    axt.set_xticks([])
    for sp in axt.spines.values():
        sp.set_visible(False)
    axt.set_title(
        f"same answer, {em['A'] / em['B']:.0f}× fewer emissions",
        fontsize=13,
        color=INK,
        pad=6,
    )
    # The band under the image carries a key that changes with the act, so it
    # is never dead space: a slow-time ticker first, then the colour scale.
    axc = fig.add_axes([0.30, 0.392, 0.40, 0.018])
    axc.set_visible(False)
    ticker = fig.text(
        0.5, 0.396, "", ha="center", fontsize=13, color=DIM, family="monospace"
    )

    def _key(mappable, label, vmin, vmax):
        axc.set_visible(True)
        axc.clear()
        grad = np.linspace(vmin, vmax, 256)[None, :]
        axc.imshow(grad, aspect="auto", cmap=mappable, extent=[vmin, vmax, 0, 1])
        axc.set_yticks([])
        axc.set_xlabel(label, fontsize=11, color=DIM, labelpad=4)
        axc.tick_params(labelsize=10)
        for sp in axc.spines.values():
            sp.set_color("#2a3140")

    r_meas, r_true = d["C_profile_measured"] * 100, d["C_profile_true"] * 100
    rr = np.linspace(0, 1, len(r_meas) + 1)
    rr = 0.5 * (rr[:-1] + rr[1:])

    x0, x1, z0, z1 = extent[0], extent[1], extent[3], extent[2]
    zx, zz = 2.8, 2.8  # half-window of the zoom, mm

    def update(f):
        k = f % n_slow
        im_b.set_data(bmode[k].T)
        im_f.set_data(fl_db[k].T)

        if f < acts[0]:
            title.set_text("Blood is almost invisible")
            sub.set_text("B-mode of a vessel tilted 60° to the beam")
            im_f.set_alpha(np.zeros_like(fl_a[k].T))
            im_v.set_alpha(0.0)
            im_p.set_alpha(0.0)
            axc.set_visible(False)
            ticker.set_text(f"slow-time sample {k + 1:>2}/{n_slow}")
        elif f < acts[1]:
            t = _ease((f - acts[0]) / (acts[1] - acts[0]))
            title.set_text("Remove everything that does not move")
            sub.set_text("A wall filter deletes the stationary echo")
            im_b.set_alpha(1.0 - 0.85 * t)
            im_f.set_alpha(fl_a[k].T * t)
            ticker.set_text(f"slow-time sample {k + 1:>2}/{n_slow}")
        elif f < acts[2]:
            t = _ease((f - acts[1]) / (acts[2] - acts[1]))
            title.set_text("Colour Doppler — how fast")
            sub.set_text(f"phase gained between emissions   ±{vlim:.0f} cm/s")
            im_b.set_alpha(0.15 + 0.85 * t)
            im_f.set_alpha(fl_a[k].T * (1.0 - t))
            im_v.set_alpha(t)
            ticker.set_text("")
            _key("coolwarm", "axial velocity (cm/s)", -vlim, vlim)
        elif f < acts[3]:
            t = _ease((f - acts[2]) / (acts[3] - acts[2]))
            title.set_text("Power Doppler — how much")
            sub.set_text("flow energy instead of velocity; it cannot alias")
            im_v.set_alpha(1.0 - t)
            im_p.set_alpha(t)
            _key("hot", "flow power (dB)", -20, 0)
        else:
            t = _ease(min((f - acts[3]) / (acts[4] - acts[3]), 1.0))
            title.set_text("And the profile is right")
            sub.set_text(
                f"{V_PEAK * 100:.0f} cm/s peak at {THETA_DEG:.0f}°  —  "
                f"the axis reads {m['arms']['C']['psf_core_ratio']:.2f}× truth"
            )
            im_p.set_alpha(1.0 - 0.5 * t)
            im_v.set_alpha(0.5 * t)
            axc.set_visible(False)
            ticker.set_text("")
            axt.set_visible(False)
            cx, cz = VESSEL_CENTER_MM[0], VESSEL_CENTER_MM[2]
            ax.set_xlim(x0 + (cx - zx - x0) * t, x1 + (cx + zx - x1) * t)
            ax.set_ylim(z1 + (cz + zz - z1) * t, z0 + (cz - zz - z0) * t)
            if t > 0.4:
                axp.set_visible(True)
                axp.clear()
                axp.set_facecolor(BG)
                n = max(int(len(rr) * (t - 0.4) / 0.45), 2)
                axp.plot(rr, r_true, "--", color=DIM, lw=2.2, label="true")
                axp.plot(
                    rr[:n],
                    r_meas[:n],
                    "-o",
                    color=ACCENT,
                    lw=2.8,
                    ms=5,
                    label="measured",
                )
                axp.set(
                    xlim=(0, 1),
                    ylim=(0, max(r_true) * 1.3),
                    xlabel="r / R    (0 = axis, 1 = wall)",
                )
                axp.set_ylabel("cm/s", fontsize=10)
                axp.legend(fontsize=10, frameon=False, loc="lower left")
                axp.tick_params(labelsize=9)
                for sp in axp.spines.values():
                    sp.set_color("#2a3140")
        return ()

    print("video 1 - the flow you cannot see")
    _save(fig, update, n_frames, "ex22_video1_flow")


# ===========================================================================
# VIDEO 2 - two ways to spend a budget
# ===========================================================================
def video_two():
    """Conventional line-by-line against compounding, using the real images."""
    b = _beamformed_for_video2()
    conv, singles = b["conv"], b["singles"]
    x, z = b["x_mm"], b["z_mm"]
    extent = [x[0], x[-1], z[-1], z[0]]
    nx, nz = singles.shape[1], singles.shape[2]

    # dB on a shared reference, so the panels are comparable frame to frame:
    # a single plane wave really is dimmer and coarser than the compound.
    # Envelope-detect FIRST, then log — and keep one shared reference so the
    # single-angle image is honestly dimmer than the compound rather than each
    # being auto-normalised to look equally bright.
    conv_env = _envelope(conv)
    sing_env = _envelope(singles)
    cum_env = _envelope(np.cumsum(singles, axis=0))  # sum coherently, then detect
    ref = cum_env[-1].max()

    def _to_db(a):
        return np.maximum(20 * np.log10(a / ref + 1e-12), -45)

    conv_db, sing_db, cum_db = _to_db(conv_env), _to_db(sing_env), _to_db(cum_env)

    em_conv, em_fast = N_LINES * E_SHORT, N_ANGLES * E_SHORT
    t_conv, t_fast = em_conv / PRF_DOPPLER, em_fast / PRF_EMISSION
    # The sequence's own angles - verified against the event delays, which
    # encode asin(c * d(delay)/dx) and match these to 0.01 deg.
    angles = ANGLES_DEG

    acts = np.cumsum([35, 160, 50, 130, 110, 50, 80])
    n_frames = int(acts[-1])

    fig = plt.figure(figsize=FIGSIZE)
    ax = _frame_axes(fig, extent, rect=(0.115, 0.455, 0.84, None))
    im = ax.imshow(
        np.full((nz, nx), -45.0),
        extent=extent,
        cmap="gray",
        vmin=-45,
        vmax=0,
        aspect="equal",
        interpolation="bilinear",
    )
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(z[-1], z[0])
    beam = ax.fill([0, 0, 0, 0], [0, 0, 0, 0], color=ACCENT, alpha=0.0, zorder=3)[0]
    wave = ax.plot([], [], color=COOL, lw=3.0, alpha=0.0, zorder=3)[0]

    title = fig.text(0.5, 0.955, "", ha="center", fontsize=23, fontweight="bold")
    sub = fig.text(0.5, 0.918, "", ha="center", fontsize=13.5, color=DIM)
    counter = fig.text(
        0.5, 0.300, "", ha="center", fontsize=17, color=INK, family="monospace"
    )
    verdict = fig.text(0.5, 0.205, "", ha="center", fontsize=15, color=ACCENT)
    punch = fig.text(0.5, 0.120, "", ha="center", fontsize=14, color=COOL)
    fig.text(
        0.115,
        0.026,
        "eSDIva  ·  example 22  ·  flow & Doppler",
        fontsize=11,
        color="#5b6472",
    )

    z_focus = VESSEL_CENTER_MM[2]
    blank = np.full((nz, nx), -45.0)

    def update(f):
        if f < acts[0]:
            title.set_text("Two ways to spend your emissions")
            sub.set_text("same phantom, same probe, same beamformer")
            for t_ in (counter, verdict, punch):
                t_.set_text("")
            im.set_data(blank)
            beam.set_alpha(0.0)
            wave.set_alpha(0.0)
        elif f < acts[1]:  # conventional, line by line, REAL lines
            t = (f - acts[0]) / (acts[1] - acts[0])
            i = min(int(t * N_LINES), N_LINES - 1)
            img = blank.copy()
            img[:, : i + 1] = conv_db[: i + 1].T
            im.set_data(img)
            xl = float(x[i])
            beam.set_xy(
                np.array(
                    [
                        [xl - 2.2, z[0]],
                        [xl + 2.2, z[0]],
                        [xl + 0.45, z_focus],
                        [xl - 0.45, z_focus],
                    ]
                )
            )
            beam.set_alpha(0.45)
            wave.set_alpha(0.0)
            title.set_text("Conventional: one focused line at a time")
            sub.set_text(f"{E_SHORT} emissions on that line, then step across")
            counter.set_text(
                f"line {i + 1:>3}/{N_LINES}    "
                f"emissions {(i + 1) * E_SHORT:>4}/{em_conv}"
            )
            verdict.set_text("")
        elif f < acts[2]:
            im.set_data(conv_db.T)
            beam.set_alpha(0.0)
            title.set_text("Conventional: one focused line at a time")
            sub.set_text("every line is beamformed from its own transmit")
            counter.set_text(f"{em_conv} emissions per colour image")
            verdict.set_text(
                f"one image every {t_conv * 1e3:.0f} ms   →   "
                f"{1 / t_conv:.0f} images per second"
            )
        elif f < acts[3]:  # each plane wave on its own - the real single-angle image
            t = (f - acts[2]) / (acts[3] - acts[2])
            i = min(int(t * N_ANGLES), N_ANGLES - 1)
            im.set_data(sing_db[i].T)
            beam.set_alpha(0.0)
            # Sweep the wavefront through the box during each angle's dwell, so
            # it reads as a travelling emission rather than a depth marker.
            frac = (t * N_ANGLES) % 1.0
            # A wavefront is PERPENDICULAR to the propagation direction
            # (sin0, cos0), so along it dz/dx = -tan(theta). With +theta the
            # far-x elements fire later, so the wavefront leans the other way.
            s_ = -np.tan(np.deg2rad(angles[i]))
            zc = z[0] + (z[-1] - z[0]) * min(frac * 1.5, 1.0)
            wave.set_data(np.array([x[0], x[-1]]), zc + s_ * np.array([x[0], x[-1]]))
            wave.set_alpha(0.85)
            title.set_text("One plane wave lights everything")
            sub.set_text("the whole box from a single emission — but soft")
            counter.set_text(
                f"angle {i + 1}/{N_ANGLES}   "
                f"({angles[i]:+.0f}°)    emission {i + 1}/{N_ANGLES}"
            )
            verdict.set_text("no transmit focus yet — dim, and coarse")
        elif f < acts[4]:  # coherent compounding, sharpening for real
            t = (f - acts[3]) / (acts[4] - acts[3])
            i = min(int(t * N_ANGLES), N_ANGLES - 1)
            im.set_data(cum_db[i].T)
            wave.set_alpha(0.0)
            title.set_text("Compounding: sum them coherently")
            sub.set_text("adding the angles synthesises a transmit focus everywhere")
            counter.set_text(f"{i + 1}/{N_ANGLES} angles summed")
            verdict.set_text("brighter and sharper with every angle")
        elif f < acts[5]:
            im.set_data(cum_db[-1].T)
            wave.set_alpha(0.0)
            title.set_text("Ultrafast: 9 angles, one frame")
            sub.set_text("every voxel imaged on every emission")
            counter.set_text(f"{em_fast} emissions per colour image")
            verdict.set_text(
                f"one image every {t_fast * 1e3:.1f} ms   →   "
                f"{1 / t_fast:.0f} images per second"
            )
        else:
            im.set_data(cum_db[-1].T)
            title.set_text("Same answer, a fraction of the cost")
            sub.set_text("")
            counter.set_text(f"{em_conv}  →  {em_fast} emissions")
            verdict.set_text(f"{em_conv / em_fast:.0f}× fewer emissions")
            punch.set_text(
                f"{t_conv / t_fast:.0f}× faster — which is what buys the long ensemble"
            )
        return ()

    print("video 2 - two ways to spend a budget")
    _save(fig, update, n_frames, "ex22_video2_sequences")


if __name__ == "__main__":
    if not IQ_FILE.exists():
        raise SystemExit(f"No results at {IQ_FILE} - run step3 first.")
    video_one()
    video_two()
    print(f"\nWritten to {OUT}")
