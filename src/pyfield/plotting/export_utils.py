"""Shared save/export helpers for plotting functions.

Convention
---------
- ``save_path`` is always a **directory** (or ``None`` to skip saving).
- ``file_name`` always **includes the extension** (e.g. ``"field.png"``).
- Directory creation is handled here, not by callers.
"""

from __future__ import annotations

from pathlib import Path


def _resolve_export_path(save_path, file_name: str) -> Path:
    """Join *save_path* and *file_name*, creating the directory if needed."""
    out_dir = Path(save_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / file_name


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

    Returns the path that was actually written, or ``None`` on failure.
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

    Returns the saved path, or ``None`` on failure.
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

    Returns the saved path, or ``None`` on failure.
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
