"""Random scatterer phantoms for pulse-echo (speckle) simulation."""

import numpy as np
from scipy.ndimage import map_coordinates


def make_phantom(extents_mm, n_scatterers, echogenicity_map=None, seed=None):
    """Random scatterer cloud for a speckle phantom.

    Tissue in pulse-echo simulation is modelled as many sub-wavelength scatterers
    at RANDOM positions: their echoes interfere to produce speckle. A regular
    lattice is unsuitable for this — its spatial periodicity returns coherent
    lattice echoes instead of speckle. Positions are drawn uniformly in the box,
    and each amplitude is a zero-mean Gaussian draw scaled by the local
    echogenicity::

        amplitude_i = N(0, 1) · map(r_i)

    so an anechoic region (map = 0) returns no echo and a region with map = g
    has echo energy ∝ g². For fully developed speckle use at least ~5-10
    scatterers per resolution cell (cell ≈ λ·F# laterally × half the pulse
    length axially).

    Parameters
    ----------
    extents_mm : dict or (3, 2) array-like
        Phantom box in mm: ``{"x_extent": [x0, xf], "y_extent": [y0, yf],
        "z_extent": [z0, zf]}`` (same keys as the simulators' grid dicts,
        ``_mm``-suffixed variants accepted) or ``[[x0, xf], [y0, yf], [z0, zf]]``.
    n_scatterers : int
        Number of scatterers to draw.
    echogenicity_map : numpy.ndarray, optional
        Relative scattering strength over the box, sampled at each scatterer
        position by linear interpolation. 2-D ``(Nx, Nz)`` = an x-z image
        (uniform along y, the usual imaging-plane case) or 3-D
        ``(Nx, Ny, Nz)``. Values are relative: 0 = anechoic (cyst),
        1 = background, >1 = hyperechoic. None = uniform speckle.
    seed : int, optional
        Seed for reproducible draws.

    Returns
    -------
    positions_mm : (n_scatterers, 3) numpy.ndarray
        Scatterer positions in mm (float32), ready for the `Reception` classes.
    amplitudes : (n_scatterers,) numpy.ndarray
        Scattering amplitudes (float32).

    Raises
    ------
    ValueError
        If ``echogenicity_map`` is not 2-D or 3-D.
    """
    if isinstance(extents_mm, dict):
        suffix = "_mm" if "x_extent_mm" in extents_mm else ""
        box = np.array(
            [extents_mm[f"{ax}_extent{suffix}"] for ax in "xyz"], dtype=np.float64
        )
    else:
        box = np.asarray(extents_mm, dtype=np.float64).reshape(3, 2)

    rng = np.random.default_rng(seed)
    pos = rng.uniform(box[:, 0], box[:, 1], size=(int(n_scatterers), 3))
    amps = rng.standard_normal(pos.shape[0])

    if echogenicity_map is not None:
        m = np.asarray(echogenicity_map, dtype=np.float64)
        if m.ndim == 2:
            axes = (0, 2)  # x-z image, uniform along elevation (y).
        elif m.ndim == 3:
            axes = (0, 1, 2)
        else:
            raise ValueError(
                f"echogenicity_map must be 2-D (Nx, Nz) or 3-D (Nx, Ny, Nz), "
                f"got ndim={m.ndim}."
            )
        # Fractional pixel index of each scatterer in the map: the map's first
        # pixel sits at the box's low edge, its last at the high edge. A
        # degenerate axis (zero extent) maps every scatterer to pixel 0.
        span = np.where(box[:, 1] > box[:, 0], box[:, 1] - box[:, 0], 1.0)
        frac = (pos - box[:, 0]) / span
        coords = [frac[:, ax] * (m.shape[i] - 1) for i, ax in enumerate(axes)]
        amps *= map_coordinates(m, coords, order=1, mode="nearest")

    return pos.astype(np.float32), amps.astype(np.float32)
