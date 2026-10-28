"""One-figure diagnostic of a single-element transducer's emitted field."""

from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def visualize_element(
    transducer,
    *,
    c_m_s: float = 1540.0,
    fs: float = 100e6,
    max_angle_deg: float = 70.0,
    arc_mm: float = 35.0,
    axial_max_mm: float = 80.0,
    xz_half_width_mm: float = 30.0,
    xz_depth_mm: float = 60.0,
    dx_mm: float = 0.5,
    db_floor: float = -40.0,
    reference: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    off_screen: bool = True,
) -> plt.Figure:
    """Four-panel diagnostic of a mono-element transducer's monochromatic field.

    Runs a single monochromatic :class:`esdiva.emission.Emission` at the element's
    centre frequency and lays out: (1) the patch mesh, (2) the ``|P|`` directivity
    (dB) on an arc, (3) the normalised on-axis ``|P|`` vs depth, (4) the ``xz``
    pressure plane (dB). All fields share the same Emission, so the panels agree.

    Parameters
    ----------
    transducer : esdiva.transducers.TransducerBase
        A single-element transducer (circular, radial-profile, ...).
    c_m_s : float, default: 1540.0
        Sound speed in m/s (sets the wavelength and the near-field marker).
    fs : float, default: 1e8
        Sampling frequency passed to the Emission.
    max_angle_deg : float, default: 70.0
        Half-span of the directivity arc.
    arc_mm : float, default: 35.0
        Radius of the directivity arc in mm.
    axial_max_mm : float, default: 80.0
        Depth span of the on-axis profile in mm.
    xz_half_width_mm : float, default: 30.0
        Half-width (x) of the xz plane in mm.
    xz_depth_mm : float, default: 60.0
        Depth (z) of the xz plane in mm.
    dx_mm : float, default: 0.5
        Grid spacing of the xz plane in mm.
    db_floor : float, default: -40.0
        Lower dB limit of the xz colour scale.
    reference : tuple of (ndarray, ndarray), optional
        ``(angle_deg, directivity_db)`` target overlaid on the directivity panel.
    title : str, optional
        Figure suptitle.
    save_path : str, optional
        If given, the figure is written here.
    off_screen : bool, default: True
        Render the mesh screenshot off-screen (no interactive window).

    Returns
    -------
    matplotlib.figure.Figure
        The assembled four-panel figure.
    """
    import pyvista as pv

    from esdiva.emission import Emission

    sim = Emission(transducer, fs=fs, c=c_m_s, monochromatic=True, verbose=False)

    def magnitude(points_mm: np.ndarray) -> np.ndarray:
        return np.abs(sim(points_mm)[0]).squeeze()

    # Directivity on an arc at constant radius, peak-normalised.
    angle = np.linspace(-max_angle_deg, max_angle_deg, 361)
    th = np.deg2rad(angle)
    arc_pts = np.stack([arc_mm * np.sin(th), np.zeros_like(th), arc_mm * np.cos(th)], 1)
    p_arc = magnitude(arc_pts)
    dir_db = 20 * np.log10(p_arc / p_arc.max())
    pos = angle >= 0
    below = np.where(dir_db[pos] <= -6.0)[0]
    half_angle = float(angle[pos][below[0]]) if below.size else float("nan")

    # On-axis |P| vs depth, normalised.
    z_axis = np.linspace(0.5, axial_max_mm, 400)
    axial = magnitude(np.stack([np.zeros_like(z_axis), np.zeros_like(z_axis), z_axis], 1))
    axial /= axial.max()

    # xz pressure plane from one grid Emission (same machinery, shared conventions).
    grid = {
        "x_extent": [-xz_half_width_mm, xz_half_width_mm],
        "y_extent": [0, 0],
        "z_extent": [0, xz_depth_mm],
        "dx": dx_mm,
        "dy": 0,
        "dz": dx_mm,
    }
    p_grid, coords = sim(grid)
    x, z = coords["x"], coords["z"]
    p = np.abs(p_grid).squeeze()  # (Nx, Nz), y is singleton
    plane = (20 * np.log10(np.clip(p / p.max(), 10 ** (db_floor / 20), None))).T  # (Nz, Nx)

    # Element mesh screenshot.
    pl = pv.Plotter(off_screen=off_screen, window_size=(700, 700))
    pl.add_mesh(transducer.get_mesh(), show_edges=True, scalars="Apodization", cmap="cool", clim=[0, 1])
    pl.view_isometric()
    pl.reset_camera()
    pl.camera.zoom(1.2)
    mesh_img = pl.screenshot(return_img=True)
    pl.close()

    fig, ax = plt.subplots(1, 4, figsize=(22, 5))
    ax[0].imshow(mesh_img)
    ax[0].axis("off")
    ax[0].set_title("element mesh")

    if reference is not None:
        ax[1].plot(reference[0], reference[1], "k--", lw=1, label="reference")
    ax[1].plot(angle, dir_db, "C0", lw=2, label=f"−6 dB at {half_angle:.0f}°")
    ax[1].axhline(-6, color="r", ls=":")
    ax[1].set(xlabel="angle (deg)", ylabel="dB", ylim=(-20, 1),
              xlim=(-max_angle_deg, max_angle_deg), title="directivity")
    ax[1].legend()

    ax[2].plot(z_axis, axial, "C0", lw=2)
    diameter_mm = getattr(transducer, "diameter", None)
    if diameter_mm is not None:
        lambda_mm = c_m_s / transducer.fc * 1e3
        near_field = (diameter_mm * 1e3) ** 2 / (4 * lambda_mm)  # D²/4λ
        ax[2].axvline(near_field, color="r", ls=":", label="near-field N = D²/4λ")
        ax[2].legend()
    ax[2].set(xlabel="z (mm)", ylabel="|P|", title="on-axis field")
    ax[2].grid(alpha=0.3)

    im = ax[3].pcolormesh(x, z, plane, cmap="jet", shading="auto", vmin=db_floor, vmax=0)
    if diameter_mm is not None:
        r = diameter_mm * 1e3 / 2
        ax[3].plot([-r, r], [0, 0], "w", lw=3)
    ax[3].set(xlabel="x (mm)", ylabel="z (mm)", title="xz field (dB)")
    ax[3].invert_yaxis()
    fig.colorbar(im, ax=ax[3], label="|P| (dB)")

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130)
    return fig
