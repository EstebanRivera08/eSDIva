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
    xz_half_width_mm: float = 30.0,
    xz_depth_mm: float = 60.0,
    dx_mm: float = 0.5,
    db_floor: float = -40.0,
    reference: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    reference_profile: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    profile_ylim: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    off_screen: bool = True,
) -> plt.Figure:
    """Four-panel diagnostic of a mono-element transducer's monochromatic field.

    Runs a single monochromatic :class:`esdiva.emission.Emission` at the element's
    centre frequency and lays out a 2×2 grid: the top row is (1) the surface profile
    ``z(r)`` read from the element's own patches and (2) the ``|P|`` directivity (dB)
    on an arc; the bottom row is (3) the patch mesh (3-D render) and (4) the ``xz``
    pressure plane (dB). In panels (1) and (2) the found curve is drawn solid red and
    an optional reference dotted black, so a fitted surface/beam can be compared
    against its target at a glance.

    Parameters
    ----------
    transducer : esdiva.transducers.TransducerBase
        A single-element transducer (circular, radial-profile, ...).
    c_m_s : float, default: 1540.0
        Sound speed in m/s (sets the wavelength).
    fs : float, default: 1e8
        Sampling frequency passed to the Emission.
    max_angle_deg : float, default: 70.0
        Half-span of the directivity arc.
    arc_mm : float, default: 35.0
        Radius of the directivity arc in mm.
    xz_half_width_mm : float, default: 30.0
        Half-width (x) of the xz plane in mm.
    xz_depth_mm : float, default: 60.0
        Depth (z) of the xz plane in mm.
    dx_mm : float, default: 0.5
        Grid spacing of the xz plane in mm.
    db_floor : float, default: -40.0
        Lower dB limit of the xz colour scale.
    reference : tuple of (ndarray, ndarray), optional
        ``(angle_deg, directivity_db)`` target overlaid dotted on the directivity panel.
    reference_profile : tuple of (ndarray, ndarray), optional
        ``(r_mm, z_um)`` reference surface overlaid dotted on the profile panel.
    profile_ylim : tuple of (float, float), optional
        Fixed ``(z_min, z_max)`` limits (µm) for the profile panel. Pin it to stop the
        axis rescaling between frames when animating a fit, so only the curve moves.
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

    # Surface profile z(r) read straight off the patch centres (apex pinned to z=0).
    centres = np.array(transducer.sub_quad_verts).mean(axis=1)  # (M, 3)
    r_mm = np.hypot(centres[:, 0], centres[:, 1]) * 1e3
    z_um = centres[:, 2] * 1e6
    order = np.argsort(r_mm)
    r_mm, z_um = r_mm[order], z_um[order] - z_um.max()

    # Directivity on an arc at constant radius, peak-normalised.
    angle = np.linspace(-max_angle_deg, max_angle_deg, 361)
    th = np.deg2rad(angle)
    arc_pts = np.stack([arc_mm * np.sin(th), np.zeros_like(th), arc_mm * np.cos(th)], 1)
    p_arc = np.abs(sim(arc_pts)[0]).squeeze()
    dir_db = 20 * np.log10(p_arc / p_arc.max())
    pos = angle >= 0
    below = np.where(dir_db[pos] <= -6.0)[0]
    half_angle = float(angle[pos][below[0]]) if below.size else float("nan")

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
    plane = (
        20 * np.log10(np.clip(p / p.max(), 10 ** (db_floor / 20), None))
    ).T  # (Nz, Nx)

    # Element mesh screenshot.
    pl = pv.Plotter(off_screen=off_screen, window_size=(700, 700))
    pl.add_mesh(
        transducer.get_mesh(),
        show_edges=True,
        scalars="Apodization",
        cmap="cool",
        clim=[0, 1],
    )
    pl.view_isometric()
    pl.reset_camera()
    pl.camera.zoom(1.2)
    mesh_img = pl.screenshot(return_img=True)
    pl.close()

    # 2×2 grid: top row profile + directivity, bottom row 3-D mesh + xz field.
    fig, ax = plt.subplots(2, 2, figsize=(12, 10))

    if reference_profile is not None:
        ax[0, 0].plot(
            reference_profile[0], reference_profile[1], "k:", lw=1.5, label="reference"
        )
    ax[0, 0].plot(r_mm, z_um, "r", lw=2, label="found")
    ax[0, 0].set(xlabel="r (mm)", ylabel="z (µm)", title="surface profile z(r)")
    if profile_ylim is not None:
        ax[0, 0].set_ylim(profile_ylim)
    ax[0, 0].legend()
    ax[0, 0].grid(alpha=0.3)

    if reference is not None:
        ax[0, 1].plot(reference[0], reference[1], "k:", lw=1.5, label="reference")
    ax[0, 1].plot(angle, dir_db, "r", lw=2, label=f"found (−6 dB at {half_angle:.0f}°)")
    ax[0, 1].axhline(-6, color="0.7", ls=":")
    ax[0, 1].set(
        xlabel="angle (deg)",
        ylabel="dB",
        ylim=(-20, 1),
        xlim=(-max_angle_deg, max_angle_deg),
        title="directivity",
    )
    ax[0, 1].legend()

    ax[1, 0].imshow(mesh_img)
    ax[1, 0].axis("off")
    ax[1, 0].set_title("element mesh")

    im = ax[1, 1].pcolormesh(
        x, z, plane, cmap="jet", shading="auto", vmin=db_floor, vmax=0
    )
    diameter_mm = getattr(transducer, "diameter", None)
    if diameter_mm is not None:
        r = diameter_mm * 1e3 / 2
        ax[1, 1].plot([-r, r], [0, 0], "w", lw=3)
    ax[1, 1].set(xlabel="x (mm)", ylabel="z (mm)", title="xz field (dB)")
    ax[1, 1].invert_yaxis()
    fig.colorbar(im, ax=ax[1, 1], label="|P| (dB)")

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130)
    return fig
