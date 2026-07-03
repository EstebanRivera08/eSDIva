"""Shared save/export helpers for plotting functions.

``save_path`` is a **directory** (combined with ``file_name``) or a full file
path — anything with an extension is treated as the output file directly.
``file_name`` always **includes the extension** (e.g. ``"field.png"``).
Directory creation is handled here, not by callers.
"""

from __future__ import annotations

from pathlib import Path


def _resolve_export_path(save_path, file_name: str) -> Path:
    """Resolve the output file, creating its directory if needed.

    ``save_path`` with a file extension is used as the full output path
    (``file_name`` ignored); otherwise it is a directory joined with
    ``file_name``.
    """
    p = Path(save_path)
    if p.suffix:
        p.parent.mkdir(parents=True, exist_ok=True)
        return p
    p.mkdir(parents=True, exist_ok=True)
    return p / file_name


def save_matplotlib_animation(
    ani,
    save_path,
    file_name: str,
    *,
    fps: int = 30,
    dpi: int = 150,
) -> Path | None:
    """Save a :class:`~matplotlib.animation.FuncAnimation` to disk.

    For extensions in ``{.mp4, .avi, .mov}`` ffmpeg is used directly.
    For anything else ffmpeg is tried first, falling back to pillow (GIF).

    Parameters
    ----------
    ani : matplotlib.animation.FuncAnimation
        Animation object to save.
    save_path : str or pathlib.Path
        Output directory.
    file_name : str
        File name with extension (e.g. ``"anim.mp4"``).
    fps : int, default: 30
        Frame rate.
    dpi : int, default: 150
        Resolution in dots per inch.

    Returns
    -------
    pathlib.Path or None
        Path that was actually written, or ``None`` on failure.
    """
    path = _resolve_export_path(save_path, file_name)
    ext = path.suffix.lower()

    if ext in {".mp4", ".avi", ".mov"}:
        try:
            ani.save(str(path), writer="ffmpeg", fps=fps, dpi=dpi)
            print(f"Video saved: {path.resolve()}")
            return path
        except Exception as exc:
            print(f"Video export failed: {exc}")
            return None

    # Non-video extension: try ffmpeg first, fall back to pillow GIF
    try:
        ani.save(str(path), writer="ffmpeg", fps=fps, dpi=dpi)
        print(f"Video saved: {path.resolve()}")
        return path
    except Exception as exc_ffmpeg:
        gif_path = path.with_suffix(".gif")
        try:
            ani.save(str(gif_path), writer="pillow", fps=fps)
            print(f"GIF saved (ffmpeg unavailable): {gif_path.resolve()}")
            return gif_path
        except Exception as exc_gif:
            print(f"Export failed: {exc_ffmpeg} | {exc_gif}")
            return None


def save_pyvista_screenshot(
    plotter,
    save_path,
    file_name: str,
    *,
    transparent_background: bool = True,
) -> Path | None:
    """Save a PyVista plotter screenshot.

    Parameters
    ----------
    plotter : pyvista.Plotter
        Plotter to capture.
    save_path : str or pathlib.Path
        Output directory.
    file_name : str
        File name with extension (e.g. ``"screenshot.png"``).
    transparent_background : bool, default: True
        Use a transparent background in the screenshot.

    Returns
    -------
    pathlib.Path or None
        Saved path, or ``None`` on failure.
    """
    path = _resolve_export_path(save_path, file_name)
    try:
        plotter.screenshot(str(path), transparent_background=transparent_background)
        print(f"\nScreenshot saved to: {path.resolve()}")
        return path
    except Exception as exc:
        print(f"Screenshot export failed: {exc}")
        return None


def save_pyvista_movie(
    plotter,
    save_path,
    file_name: str,
    update_fn,
    frame_indices,
    *,
    fps: int = 30,
) -> Path | None:
    """Record a PyVista animation by iterating *frame_indices*.

    *update_fn(idx)* is called for each frame index before writing.
    ``.gif`` uses ``plotter.open_gif``; other extensions use
    ``plotter.open_movie``.

    Parameters
    ----------
    plotter : pyvista.Plotter
        Plotter to record from.
    save_path : str or pathlib.Path
        Output directory.
    file_name : str
        File name with extension (e.g. ``"movie.mp4"``).
    update_fn : callable
        Called as ``update_fn(idx)`` for each frame before writing.
    frame_indices : iterable of int
        Sequence of frame indices to iterate over.
    fps : int, default: 30
        Frame rate for video formats.

    Returns
    -------
    pathlib.Path or None
        Saved path, or ``None`` on failure.
    """
    path = _resolve_export_path(save_path, file_name)
    try:
        if path.suffix.lower() == ".gif":
            plotter.open_gif(str(path))
        else:
            plotter.open_movie(str(path), framerate=fps)

        for idx in frame_indices:
            update_fn(idx)
            plotter.write_frame()

        plotter.close()
        print(f"\nVideo saved to: {path.resolve()}")
        return path
    except Exception as exc:
        print(f"Movie export failed: {exc}")
        return None
