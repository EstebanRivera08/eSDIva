"""Plotting helpers for the advanced case studies.

`animate_xam` turns the three linear pulses of a cross-amplitude-modulation (xAM)
sequence into a movie: the X-wave and its two halves propagating, the linear
combination TX1 - TX2 - TX3 that a tissue (linear) scatterer returns, and the
quadratic combination TX1² - TX2² - TX3² that a bubble returns.
"""

import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter

INK, DIM, BG = "#f2f4f7", "#9aa4b2", "#0b0e13"
DARK = {
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "savefig.facecolor": BG,
    "text.color": INK,
    "axes.labelcolor": INK,
    "axes.edgecolor": "#2a3140",
    "xtick.color": DIM,
    "ytick.color": DIM,
}


def _ease(t):
    """Smoothstep: a fade that starts and ends at rest, not a mechanical ramp."""
    t = float(np.clip(t, 0.0, 1.0))
    return t * t * (3.0 - 2.0 * t)


def animate_xam(
    tx1,
    tx2,
    tx3,
    path=None,
    *,
    n_frames=120,
    blend_s=1.5,
    hold_s=2.5,
    db_range=30.0,
    fps=20,
):
    """Five-panel movie of an xAM sequence: the pulses in time, then their RMS maps.

    The panels show, on the same XZ plane:

    a) TX1, both half-apertures: the full X-wave, two plane waves crossing.
    b) TX2, left half only.  c) TX3, right half only.
    d) Linear, TX1 - TX2 - TX3. A linear scatterer's echo is e ∝ p, so the
       three echoes combine like the three fields. By superposition
       TX1 = TX2 + TX3, so this is zero everywhere and at every instant.
    e) Non-linear, TX1² - TX2² - TX3². In the simplest non-linear model,
       e ∝ p + α·p², the linear part cancels as in d) and α·(TX1² - TX2² - TX3²)
       = 2α·TX2·TX3 is left. It is non-zero only where both plane waves are
       present at the same point at the same instant: the sound sheet travelling
       down the bisector. A footnote on the figure states this argument.

    Every frame is in dB, 20·log10 of the magnitude: a linear scale hides the
    weak edges of the beams that matter here. a-d are referenced to the TX1
    peak, so d reads at its true (vanishing) level; e, in Pa², to its own peak
    (20·log10 because the bubble echo AMPLITUDE is ∝ p²). After the transient
    each panel blends into its time-RMS map √⟨·²⟩, in dB on the same references.
    Both stages share one colour map and one dB scale, so the blend shows the
    energy each wave leaves behind rather than a change of colour code.

    Parameters
    ----------
    tx1, tx2, tx3 : Field
        Transient pressure of the three pulses, ``(Nt, Nx, 1, Nz)`` on one grid and
        one time axis (same delays, only the apodization differs).
    path : str or pathlib.Path, optional
        Save the movie there: ``.mp4`` (H.264 through the ffmpeg of the
        ``imageio-ffmpeg`` package) or ``.gif``. Without it the animation is
        returned for ``plt.show()``.
    n_frames : int, default: 120
        Time frames shown, spread evenly over the whole transient.
    blend_s : float, default: 1.5
        Duration of the blend from the last pulse frame to the RMS maps (s).
    hold_s : float, default: 2.5
        How long the RMS maps stay on screen (s).
    db_range : float, default: 30.0
        Displayed dynamic range (dB).
    fps : int, default: 20
        Frames per second.

    Returns
    -------
    matplotlib.animation.FuncAnimation
        The animation (keep a reference until it is shown or saved).
    """
    titles = [
        "a) TX1: left + right",
        "b) TX2: left half",
        "c) TX3: right half",
        "d) Linear\nTX1 − TX2 − TX3",
        "e) Non-linear\nTX1² − TX2² − TX3²",
    ]
    note = (
        "Assumed toy model (not simulated) of the echo e of a scatterer driven "
        "by pressure p:  linear  e ∝ p;  non-linear  e ∝ p + α·p².\n"
        "xAM fires TX1 = TX2 + TX3, so  e1 − e2 − e3 ∝ TX1 − TX2 − TX3 = 0  "
        "for a linear scatterer,  while a non-linear one keeps  "
        "α·(TX1² − TX2² − TX3²) = 2α·TX2·TX3."
    )
    fields = [tx1, tx2, tx3, tx1 - tx2 - tx3, tx1**2 - tx2**2 - tx3**2]
    movies = [np.abs(np.asarray(f, dtype=float)[:, :, 0, :]) for f in fields]
    rms = [np.sqrt(np.mean(m**2, axis=0)) for m in movies]
    peaks = [movies[0].max()] * 4 + [movies[4].max()]
    rms_refs = [rms[0].max()] * 4 + [rms[4].max()]

    def db(a, ref):
        return np.maximum(20 * np.log10(a / ref + 1e-12), -db_range)

    rms_db = [db(r, ref) for r, ref in zip(rms, rms_refs)]

    coords = tx1.coords
    x, z = coords["x"], coords["z"]
    t_us = (coords["t0"] + np.arange(tx1.shape[0]) * coords["dt"]) * 1e6
    ks = np.linspace(0, tx1.shape[0] - 1, n_frames).astype(int)
    n_blend, n_hold = round(blend_s * fps), round(hold_s * fps)
    last = [db(m[ks[-1]], p) for m, p in zip(movies, peaks)]

    with plt.rc_context(DARK):
        # Portrait 3 + 2 grid so the movie fits a phone held upright; the free
        # sixth cell holds the shared colour bar.
        fig, ax = plt.subplot_mosaic(
            [["a", "b", "c"], ["d", "e", "cbar"]],
            figsize=(8, 10.5),
            constrained_layout=True,
        )
        axes = [ax[k] for k in "abcde"]
        images = []
        for i, (ax_i, title, m, p) in enumerate(zip(axes, titles, movies, peaks)):
            im = ax_i.imshow(
                db(m[0], p).T,
                extent=[x[0], x[-1], z[-1], z[0]],
                aspect="equal",
                cmap="inferno",
                vmin=-db_range,
                vmax=0,
                interpolation="bilinear",
            )
            ax_i.set(xlabel="x (mm)", ylabel="z (mm)" if i in (0, 3) else None)
            ax_i.set_title(title, fontsize=11)
            images.append(im)
        # One shared colour map and dB scale for every panel and both stages.
        ax["cbar"].axis("off")
        fig.colorbar(
            images[0],
            cax=ax["cbar"].inset_axes([0.3, 0.1, 0.12, 0.8]),
            label="dB  (a–d re TX1 peak, e re own)",
        )
        head = fig.suptitle(" ", fontsize=14, fontweight="bold")
        fig.supxlabel(
            "\n".join(textwrap.fill(line, 80) for line in note.split("\n")),
            fontsize=9,
            color=DIM,
        )
        # Solve the layout once; re-solving it on every frame triples render time.
        fig.canvas.draw()
        fig.set_layout_engine("none")

    def draw(f):
        if f < n_frames:  # Pulses propagating.
            k = ks[f]
            for im, m, p in zip(images, movies, peaks):
                im.set_data(db(m[k], p).T)
            head.set_text(f"Instantaneous |p|   ·   t = {t_us[k]:.2f} µs")
        else:  # Blend into the time-RMS maps, then hold them.
            t = _ease((f - n_frames) / max(n_blend, 1))
            for im, a, b in zip(images, last, rms_db):
                im.set_data(((1 - t) * a + t * b).T)
            head.set_text("RMS over time: the energy each wave leaves behind")
        return [*images, head]

    anim = FuncAnimation(
        fig, draw, frames=n_frames + n_blend + n_hold, interval=1000 / fps
    )
    if path is not None:
        path = Path(path)
        if path.suffix == ".mp4":
            import imageio_ffmpeg

            plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
            writer = FFMpegWriter(
                fps=fps, codec="libx264", extra_args=["-pix_fmt", "yuv420p"]
            )
        else:
            writer = PillowWriter(fps=fps)
        anim.save(path, writer=writer)
        plt.close(fig)
    return anim
