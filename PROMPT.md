
I want to standardize the saving logic across all plotting functions.  
The following structure is close to what I want (it would need to be modified for 3D and
pyvista):

```python
if save_path:
    save_path = pathlib.Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    video_path = save_path / file_name

    if video_path.suffix.lower() not in {".mp4", ".avi", ".mov"}:
        try:
            ani.save(str(video_path), writer="ffmpeg", fps=fps, dpi=150)
            print(f"Video saved: {video_path.resolve()}")
        except Exception as e:
            gif_path = video_path.with_suffix(".gif")
            try:
                ani.save(str(gif_path), writer="pillow", fps=fps)
                print(f"GIF saved: {gif_path.resolve()}")
            except Exception as e2:
                print(f"Export failed: {e} | {e2}")
    else:
        try:
            ani.save(str(video_path), writer="ffmpeg", fps=fps, dpi=150)
            print(f"Video saved: {video_path.resolve()}")
        except Exception as e:
            print(f"Video export failed: {e}")
```

The idea is:

- If the user provides a filename with a supported video extension (`.mp4`, `.avi`, `.mov`), try saving directly with ffmpeg.
- If the extension is missing or unsupported, attempt saving as a video first, and if that fails, fall back to GIF (which is less error‑prone).
- This logic should be **shared by all plotting functions**, so consider implementing a **single (or 2 if not integrable between matplot and pyvista) reusable helper function** that handles:
  - path creation  
  - extension detection  
  - ffmpeg save attempt  
  - GIF fallback  
  - consistent error reporting  

The goal is to avoid duplicated save logic and ensure all plotting functions behave consistently.


