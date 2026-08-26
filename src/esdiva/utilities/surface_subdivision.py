"""Parametric surface subdivision into flat rectangular patch approximations.

Ultrasound simulators based on the spatial impulse response (SIR) method
decompose a transducer aperture into a mosaic of small **flat rectangular
piston patches**.  The SIR of the full aperture is the superposition of the
individual piston SIRs.  This module provides a general tool to build such a
mosaic for **any smooth (C1) parametric surface**, not just flat disks.

The key physical requirement is that every patch must be a genuine rectangle
(not a parallelogram or trapezoid), so the analytical piston-SIR formula
remains valid.  On a curved surface this is achieved by constructing each
patch as a rectangle in the **local tangent plane** at the patch centre.

Notes
-----
The module uses a uniform Cartesian parameter-space grid.  Each patch is
constructed in the local tangent plane with arc-length extents, so it
correctly represents the physical size on the curved surface.

At high curvature, adjacent tangent-plane patches leave small wedge-shaped
gaps (inherent to the flat-piston approximation).  Gaps shrink quadratically
with resolution — increase ``n_u``/``n_v`` for better coverage.

**Tuning parameters**:

- ``n_u``, ``n_v`` -- resolution; increase until ``coverage`` is acceptable.
- ``border_refine`` -- subdivision factor for boundary cells.

Examples
--------
Spherical bowl (concave transducer)::

    import numpy as np
    from esdiva.utilities.surface_subdivision import subdivide_parametric_surface

    R, R_ap = 20e-3, 8e-3

    def bowl(x, y):
        z = R - np.sqrt(max(R**2 - x**2 - y**2, 0.0))
        return np.array([x, y, z])

    frames = subdivide_parametric_surface(
        bowl,
        u_range=(-R_ap, R_ap),
        v_range=(-R_ap, R_ap),
        n_u=30, n_v=30,
        inside_fn=lambda x, y: x**2 + y**2 <= R_ap**2,
    )
"""

from __future__ import annotations

import warnings
from typing import Callable, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Spherical-cap subdivision (ring-based tiling)
# ---------------------------------------------------------------------------


def subdivide_spherical_cap(
    R: float,
    theta_max: float,
    n_rings: int,
    *,
    concave: bool = True,
    normal_sign: float = 1.0,
    ratio_big_patches: float = 0.85,
    refine_factor: int = 3,
) -> dict:
    """Tile a spherical cap with flat rectangular patches on concentric rings.

    Each ring at polar angle ``θ_i`` contains a variable number of azimuthal
    patches chosen so that every patch is approximately square (arc-length
    aspect ratio ≈ 1).  This avoids the Cartesian-parameterisation singularity
    ``∂z/∂x → ∞`` that occurs near the rim of deep bowls.

    Parameters
    ----------
    R : float
        Sphere radius in metres.
    theta_max : float
        Half-angle from pole to rim in radians.  ``theta_max = π/2`` gives a
        full hemisphere.
    n_rings : int
        Number of concentric rings from pole to rim.
    concave : bool
        If ``True`` (default), the bowl opens toward +z (pole at z = -sag, rim at
        z = 0).  If ``False``, the dome bulges toward +z (apex at z = +sag,
        rim at z = 0).
    normal_sign : float
        Multiplier for the outward normal.  Default ``+1.0``.
    ratio_big_patches : float
        Fraction of rings (from the outer rim inward) that use coarse
        resolution.  The remaining innermost rings are each replaced by
        ``refine_factor`` thinner sub-rings, reducing patch overlap at the
        pole.  ``1.0`` disables refinement.  Default ``0.85``.
    refine_factor : int
        Each refined inner ring is replaced by this many thinner sub-rings.
        Default ``3``.

    Notes
    -----
    Z-convention: rim is always at z = 0.
    Concave bowl: pole at z = -sag (below rim), opens toward +z.
    Convex dome: apex at z = +sag (above rim).

    Returns
    -------
    dict
        Same keys as :func:`subdivide_parametric_surface`: ``corners``,
        ``centers``, ``normals``, ``tangents_u``, ``tangents_v``, ``wu``,
        ``wv``, ``el_idx``, ``coverage``.
    """
    sag = R * (1.0 - np.cos(theta_max))
    dtheta = theta_max / n_rings

    corners_list: List[np.ndarray] = []
    centers_list: List[np.ndarray] = []
    normals_list: List[np.ndarray] = []
    tu_list: List[np.ndarray] = []
    tv_list: List[np.ndarray] = []
    wu_list: List[float] = []
    wv_list: List[float] = []

    # Build ring schedule: (theta_center, dtheta_ring) for each ring.
    # Outer rings (coarse) + inner rings (refined near the pole).
    n_coarse = max(1, round(n_rings * ratio_big_patches))
    n_center = n_rings - n_coarse  # innermost rings to refine

    ring_schedule: List[Tuple[float, float]] = []
    for i in range(n_rings):
        if i < n_center and refine_factor > 1:
            # Refine this inner ring into refine_factor thinner sub-rings
            dt_fine = dtheta / refine_factor
            for k in range(refine_factor):
                theta_c = i * dtheta + (k + 0.5) * dt_fine
                ring_schedule.append((theta_c, dt_fine))
        else:
            ring_schedule.append(((i + 0.5) * dtheta, dtheta))

    for theta_c, dt_ring in ring_schedule:
        sin_tc = np.sin(theta_c)
        cos_tc = np.cos(theta_c)

        # Number of azimuthal patches for near-square aspect ratio
        n_azi = max(3, round(2.0 * np.pi * sin_tc / dt_ring))
        dphi = 2.0 * np.pi / n_azi

        # Arc-length patch extents
        wu = R * dt_ring
        wv = R * sin_tc * dphi

        for j in range(n_azi):
            phi_j = (j + 0.5) * dphi
            cos_p = np.cos(phi_j)
            sin_p = np.sin(phi_j)

            # --- Surface position ---
            x = R * sin_tc * cos_p
            y = R * sin_tc * sin_p
            if concave:
                z = R * (np.cos(theta_max) - cos_tc)  # rim at z=0, pole at z=-sag
            else:
                z = sag - R * (1.0 - cos_tc)  # rim at z=0, apex at z=+sag
            cen = np.array([x, y, z])

            # --- Analytical tangent vectors (unit length) ---
            if concave:
                tu = np.array([cos_tc * cos_p, cos_tc * sin_p, sin_tc])
            else:
                tu = np.array([cos_tc * cos_p, cos_tc * sin_p, -sin_tc])
            tv = np.array([-sin_p, cos_p, 0.0])

            # Normal from cross product (already orthogonal)
            n_vec = normal_sign * np.cross(tu, tv)
            n_len = float(np.linalg.norm(n_vec))
            n_vec = n_vec / max(n_len, 1e-30)

            # --- Corners in tangent plane ---
            wu_half = wu * 0.5
            wv_half = wv * 0.5
            c00 = cen - wu_half * tu - wv_half * tv
            c10 = cen + wu_half * tu - wv_half * tv
            c11 = cen + wu_half * tu + wv_half * tv
            c01 = cen - wu_half * tu + wv_half * tv
            corners_list.append(np.array([c00, c10, c11, c01], dtype=np.float64))

            centers_list.append(cen)
            normals_list.append(n_vec)
            tu_list.append(tu)
            tv_list.append(tv)
            wu_list.append(wu)
            wv_list.append(wv)

    M = len(centers_list)

    wu_arr = np.array(wu_list, dtype=np.float32)
    wv_arr = np.array(wv_list, dtype=np.float32)
    total_patch_area = float(np.sum(wu_arr * wv_arr)) if M > 0 else 0.0

    # Exact theoretical area of the spherical cap
    theoretical_area = 2.0 * np.pi * R * R * (1.0 - np.cos(theta_max))
    coverage = total_patch_area / theoretical_area if theoretical_area > 0.0 else 0.0

    # Overlap warning: check if any patch is too large relative to R
    if M > 0:
        max_wu = float(np.max(wu_arr))
        max_wv = float(np.max(wv_arr))
        max_ratio = max(max_wu, max_wv) / R
        if max_ratio > 0.3:
            warnings.warn(
                f"Patch overlap detected near the pole (max patch/R ratio = "
                f"{max_ratio:.2f}). To reduce overlap: increase no_sub, "
                f"decrease ratio_big_patches, or increase refine_factor.",
                UserWarning,
                stacklevel=2,
            )

    print(
        f"  Patches: {M}"
        f"  |  Coverage: {coverage * 100:.1f}%"
        f"  (patch area {total_patch_area * 1e6:.2f} mm²,"
        f" theoretical {theoretical_area * 1e6:.2f} mm²)"
    )

    return {
        "corners": corners_list,
        "centers": np.array(centers_list, dtype=np.float64),
        "normals": np.array(normals_list, dtype=np.float64),
        "tangents_u": np.array(tu_list, dtype=np.float64),
        "tangents_v": np.array(tv_list, dtype=np.float64),
        "wu": wu_arr,
        "wv": wv_arr,
        "el_idx": [0] * M,
        "coverage": coverage,
    }


def subdivide_parametric_surface(
    surface_fn: Callable[[float, float], np.ndarray],
    u_range: Tuple[float, float],
    v_range: Tuple[float, float],
    n_u: int,
    n_v: int,
    *,
    inside_fn: Optional[Callable[[float, float], bool]] = None,
    accept_fn: Optional[Callable[[float, float], bool]] = None,
    border_refine: int = 3,
    normal_sign: float = 1.0,
    normalize_patch_size: bool = False,
) -> dict:
    """Subdivide a C1 parametric surface into flat rectangular patches.

    Each patch is a genuine flat rectangle in the local tangent plane at the
    patch centre.  Patch dimensions ``(wu, wv)`` represent the physical width
    and height of that piston element in metres; they are used directly by the
    SIR kernel (``farfield_rect_patch``).

    Parameters
    ----------
    surface_fn : callable
        ``(u: float, v: float) -> ndarray(3,)`` — surface position in metres.
        Must be C1 (continuously differentiable) within the parameter range.
    u_range : (float, float)
        Parameter-space extent ``(u_min, u_max)`` for the first axis.
    v_range : (float, float)
        Parameter-space extent ``(v_min, v_max)`` for the second axis.
    n_u : int
        Number of coarse patches along the u-direction.
    n_v : int
        Number of coarse patches along the v-direction.
    inside_fn : callable, optional
        ``(u: float, v: float) -> bool`` — aperture mask used for
        coarse/boundary/outside classification.  Cells entirely outside are
        discarded; boundary cells (mixed inside/outside corners) are
        subdivided into ``border_refine²`` sub-patches.  If ``None`` all
        cells are accepted.
    accept_fn : callable, optional
        ``(u: float, v: float) -> bool`` — acceptance mask for refined
        sub-patches.  Only sub-patches whose centre satisfies ``accept_fn``
        are kept.  Defaults to ``inside_fn`` when ``None``.  This allows
        using a smaller circle for ``inside_fn`` (to force more border
        cells to be refined) while accepting patches up to a larger boundary.
    border_refine : int
        Subdivision factor for boundary cells.  Default ``3`` (9 sub-patches
        per boundary cell).
    normal_sign : float
        Multiplier for the outward normal direction (``+1.0`` or ``-1.0``).
        The raw normal is ``∂r/∂u × ∂r/∂v``; flip with ``-1.0`` if it points
        into the medium instead of away from it.  Default ``+1.0``.
    normalize_patch_size : bool
        When ``True`` the patch half-extents are set to ``ddu/2`` and ``ddv/2``
        (the parameter-space step), ignoring the local Jacobian stretch factor.
        For arc-length parameterisations (where ``u`` is already in metres) this
        produces uniform patches across the aperture.  Default ``False``.

    Returns
    -------
    dict
        Patch mosaic with keys ``corners``, ``centers``, ``normals``,
        ``tangents_u``, ``tangents_v``, ``wu``, ``wv``, ``el_idx``,
        and ``coverage``.
    """
    # Default accept_fn to inside_fn when not provided
    if accept_fn is None:
        accept_fn = inside_fn

    u0, u1 = u_range
    v0, v1 = v_range

    # ------------------------------------------------------------------
    # Uniform Cartesian parameter-space grid
    # ------------------------------------------------------------------
    u_edges = np.linspace(u0, u1, n_u + 1)
    v_edges = np.linspace(v0, v1, n_v + 1)

    # ------------------------------------------------------------------
    # Patch accumulation
    # ------------------------------------------------------------------
    corners_list: List[np.ndarray] = []
    centers_list: List[np.ndarray] = []
    normals_list: List[np.ndarray] = []
    tu_list: List[np.ndarray] = []
    tv_list: List[np.ndarray] = []
    wu_list: List[float] = []
    wv_list: List[float] = []

    def _add_patch(uc: float, vc: float, ddu: float, ddv: float) -> None:
        # --- centre on the curved surface ---
        cen = surface_fn(uc, vc)

        # --- local tangent frame via central finite differences ---
        eps_u = ddu * 0.001
        eps_v = ddv * 0.001
        drdu = (surface_fn(uc + eps_u, vc) - surface_fn(uc - eps_u, vc)) / (2.0 * eps_u)
        drdv = (surface_fn(uc, vc + eps_v) - surface_fn(uc, vc - eps_v)) / (2.0 * eps_v)

        # Limit maximum tangent length to avoid extreme patch overlap at singularities.
        len_u = float(np.linalg.norm(drdu))
        len_v = float(np.linalg.norm(drdv))
        if len_u > 1.5:  # limit to 150%
            len_u = 1.5
        if len_v > 1.5:
            len_v = 1.5

        tu = drdu / max(len_u, 1e-30)
        tv_raw = drdv / max(len_v, 1e-30)
        # Gram-Schmidt: orthogonalise tv against tu
        tv_orth = tv_raw - float(np.dot(tv_raw, tu)) * tu
        tv = tv_orth / max(float(np.linalg.norm(tv_orth)), 1e-30)

        n_vec = normal_sign * np.cross(tu, tv)
        n_vec /= max(float(np.linalg.norm(n_vec)), 1e-30)

        # --- arc-length half-extents ---
        if normalize_patch_size:
            wu_half = ddu * 0.5
            wv_half = ddv * 0.5
        else:
            wu_half = len_u * (ddu * 0.5)
            wv_half = len_v * (ddv * 0.5)

        # --- flat rectangle in the local tangent plane ---
        c00 = cen - wu_half * tu - wv_half * tv
        c10 = cen + wu_half * tu - wv_half * tv
        c11 = cen + wu_half * tu + wv_half * tv
        c01 = cen - wu_half * tu + wv_half * tv
        corners_list.append(np.array([c00, c10, c11, c01], dtype=np.float64))

        centers_list.append(cen)
        tu_list.append(tu)
        tv_list.append(tv)
        normals_list.append(n_vec)
        wu_list.append(wu_half * 2.0)
        wv_list.append(wv_half * 2.0)

    # ------------------------------------------------------------------
    # Main loop over the grid
    # ------------------------------------------------------------------
    for i in range(n_u):
        u0c, u1c = u_edges[i], u_edges[i + 1]
        uc_c = 0.5 * (u0c + u1c)

        for j in range(n_v):
            v0c, v1c = v_edges[j], v_edges[j + 1]
            vc_c = 0.5 * (v0c + v1c)

            ddu = u1c - u0c
            ddv = v1c - v0c

            if inside_fn is None:
                _add_patch(uc_c, vc_c, ddu, ddv)
                continue

            assert accept_fn is not None  # always paired with inside_fn
            # Classify cell by its four parameter-space corners
            in_flags = [
                inside_fn(u0c, v0c),
                inside_fn(u1c, v0c),
                inside_fn(u1c, v1c),
                inside_fn(u0c, v1c),
            ]
            n_in = sum(in_flags)

            if n_in == 4:
                # Interior cell — add at coarse resolution
                _add_patch(uc_c, vc_c, ddu, ddv)
            elif n_in == 0:
                # All corners outside inside_fn.  Check accept_fn:
                # if all corners also outside accept_fn → truly exterior.
                # Otherwise the cell straddles the annulus → refine it.
                acc_flags = [
                    accept_fn(u0c, v0c),
                    accept_fn(u1c, v0c),
                    accept_fn(u1c, v1c),
                    accept_fn(u0c, v1c),
                ]
                if not any(acc_flags):
                    continue  # entirely outside aperture
                # Fall through to refinement below
                sdu = ddu / border_refine
                sdv = ddv / border_refine
                for si in range(border_refine):
                    for sj in range(border_refine):
                        suc = u0c + (si + 0.5) * (u1c - u0c) / border_refine
                        svc = v0c + (sj + 0.5) * (v1c - v0c) / border_refine
                        if accept_fn(suc, svc):
                            _add_patch(suc, svc, sdu, sdv)
            else:
                # Boundary cell — subdivide and keep sub-patches inside
                sdu = ddu / border_refine
                sdv = ddv / border_refine
                for si in range(border_refine):
                    for sj in range(border_refine):
                        suc = u0c + (si + 0.5) * (u1c - u0c) / border_refine
                        svc = v0c + (sj + 0.5) * (v1c - v0c) / border_refine
                        if accept_fn(suc, svc):
                            _add_patch(suc, svc, sdu, sdv)

    M = len(centers_list)

    # ------------------------------------------------------------------
    # Coverage statistics
    # ------------------------------------------------------------------
    wu_arr = np.array(wu_list, dtype=np.float32)
    wv_arr = np.array(wv_list, dtype=np.float32)
    total_patch_area = float(np.sum(wu_arr * wv_arr)) if M > 0 else 0.0

    # Theoretical surface area: numerical integration of ||∂r/∂u × ∂r/∂v||
    n_sa = max(n_u, n_v, 20)
    dsu = (u1 - u0) / n_sa
    dsv = (v1 - v0) / n_sa
    eps_u_sa, eps_v_sa = dsu * 0.01, dsv * 0.01
    theoretical_area = 0.0
    for _i in range(n_sa):
        _uc = u0 + (_i + 0.5) * dsu
        for _j in range(n_sa):
            _vc = v0 + (_j + 0.5) * dsv
            # Use accept_fn (actual aperture boundary) not inside_fn (coarse-cell
            # inner zone), so the reference area matches the accepted patch region.
            if accept_fn is not None and not accept_fn(_uc, _vc):
                continue
            _drdu = (
                surface_fn(_uc + eps_u_sa, _vc) - surface_fn(_uc - eps_u_sa, _vc)
            ) / (2.0 * eps_u_sa)
            _drdv = (
                surface_fn(_uc, _vc + eps_v_sa) - surface_fn(_uc, _vc - eps_v_sa)
            ) / (2.0 * eps_v_sa)
            theoretical_area += (
                float(np.linalg.norm(np.cross(_drdu, _drdv))) * dsu * dsv
            )

    coverage = total_patch_area / theoretical_area if theoretical_area > 0.0 else 0.0

    # Coverage warning: flag if patch area significantly exceeds surface area
    if coverage > 1.02:
        warnings.warn(
            f"Patch coverage is {coverage:.1%} (> 102%).  This may indicate patch "
            "overlap.  Increase no_sub_diameter or decrease ratio_big_patches.",
            UserWarning,
            stacklevel=2,
        )

    print(
        f"  Patches: {M}"
        f"  |  Coverage: {coverage * 100:.1f}%"
        f"  (patch area {total_patch_area * 1e6:.2f} mm²,"
        f" theoretical {theoretical_area * 1e6:.2f} mm²)"
    )

    return {
        "corners": corners_list,
        "centers": np.array(centers_list, dtype=np.float64),
        "normals": np.array(normals_list, dtype=np.float64),
        "tangents_u": np.array(tu_list, dtype=np.float64),
        "tangents_v": np.array(tv_list, dtype=np.float64),
        "wu": wu_arr,
        "wv": wv_arr,
        "el_idx": [0] * M,
        "coverage": coverage,
    }
