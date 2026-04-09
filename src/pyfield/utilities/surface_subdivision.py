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
The module uses two grid strategies selected automatically by the maximum
arc-length amplification measured along the parameter centre lines.

**Low curvature** (``max ||dr/du|| <= 1.01``):
    Uniform Cartesian grid with full arc-length patch size.

**High curvature** (``max ||dr/du|| > 1.01``):
    Arc-length adapted grid where cell boundaries correspond to uniform
    arc-length intervals on the surface.

**Tuning parameters**:

- ``n_u``, ``n_v`` -- resolution; increase until ``coverage`` is acceptable.
- ``patch_fill`` -- fraction of arc-length spacing used as patch width
  (high-curvature mode only).
- ``max_patch_scale`` -- rejection threshold for steep patches.
- ``border_refine`` -- subdivision factor for boundary cells.

Examples
--------
Spherical bowl (concave transducer)::

    import numpy as np
    from pyfield.utilities.surface_subdivision import subdivide_parametric_surface

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

from typing import Callable, List, Optional, Tuple

import numpy as np


def _arclen_adapted_edges(
    metric_fn: Callable[[float], float],
    s0: float,
    s1: float,
    n_cells: int,
) -> Tuple[np.ndarray, float]:
    """
    Compute ``n_cells + 1`` parameter edge values uniformly spaced in arc-length.

    Uses a fine numerical quadrature (``max(20 × n_cells, 500)`` intervals)
    to build the cumulative arc-length, then interpolates the inverse mapping
    to find parameter values at uniform arc-length targets.

    Parameters
    ----------
    metric_fn : callable
        ``s → ||dr/ds||`` — the local arc-length scaling (metric) along the
        parameter axis at value *s*.
    s0, s1 : float
        Parameter-space extent.
    n_cells : int
        Number of arc-length-uniform cells desired.

    Returns
    -------
    edges : ndarray(n_cells + 1,)
        Parameter values of cell edges.  ``edges[0] == s0``,
        ``edges[-1] == s1``; spacing is uniform in arc-length.
    total_arclen : float
        Total arc-length of the curve from ``s0`` to ``s1``.
    """
    n_fine = max(n_cells * 20, 500)
    s_fine = np.linspace(s0, s1, n_fine + 1)
    cumlen = np.zeros(n_fine + 1)
    for k in range(n_fine):
        s_mid = 0.5 * (s_fine[k] + s_fine[k + 1])
        cumlen[k + 1] = cumlen[k] + metric_fn(s_mid) * (s_fine[k + 1] - s_fine[k])
    total_arclen = float(cumlen[-1])
    s_targets = np.linspace(0.0, total_arclen, n_cells + 1)
    return np.interp(s_targets, cumlen, s_fine), total_arclen


def subdivide_parametric_surface(
    surface_fn: Callable[[float, float], np.ndarray],
    u_range: Tuple[float, float],
    v_range: Tuple[float, float],
    n_u: int,
    n_v: int,
    *,
    inside_fn: Optional[Callable[[float, float], bool]] = None,
    border_refine: int = 3,
    normal_sign: float = 1.0,
    max_patch_scale: float = 1.5,
    patch_fill: float = 1,
    curvature_threshold: float = 1.1,  # max 10% curvature
) -> dict:
    """Subdivide a C1 parametric surface into flat rectangular patches.

    Each accepted patch is a genuine flat rectangle in the local tangent plane
    at the patch centre.  Patch dimensions ``(wu, wv)`` represent the physical
    width and height of that piston element in metres; they are used directly
    by the SIR kernel (``farfield_rect_patch``).

    The function first measures the maximum arc-length amplification along the
    parameter centre lines and selects either a uniform Cartesian grid
    (low curvature) or an arc-length adapted grid (high curvature).

    Parameters
    ----------
    surface_fn : callable
        ``(u: float, v: float) → ndarray(3,)`` — surface position in metres.
        Must be C1 (continuously differentiable) within the parameter range.
    u_range : (float, float)
        Parameter-space extent ``(u_min, u_max)`` for the first axis.
    v_range : (float, float)
        Parameter-space extent ``(v_min, v_max)`` for the second axis.
    n_u : int
        Number of coarse patches along the u-direction.  In high-curvature
        mode the arc-length adapted grid still uses exactly ``n_u`` cells.
    n_v : int
        Number of coarse patches along the v-direction.
    inside_fn : callable, optional
        ``(u: float, v: float) → bool`` — aperture mask.  Cells entirely
        outside are discarded; boundary cells (mixed inside/outside corners)
        are subdivided into ``border_refine²`` sub-patches and only the
        sub-patches whose centre is inside are kept.  If ``None`` all cells
        are accepted.
    border_refine : int
        Subdivision factor for boundary cells.  Default ``3`` (9 sub-patches
        per boundary cell).
    normal_sign : float
        Multiplier for the outward normal direction (``+1.0`` or ``-1.0``).
        The raw normal is ``∂r/∂u × ∂r/∂v``; flip with ``-1.0`` if it points
        into the medium instead of away from it.  Default ``+1.0``.
    max_patch_scale : float
        Maximum allowed local arc-length amplification.  Patches where
        ``||∂r/∂u|| > max_patch_scale × 1`` (measured at the patch centre
        using the cell size as reference) are rejected and leave holes.
        Reduce from the default ``3.0`` for cleaner hole edges on strongly
        curved surfaces; increase to keep more coverage at the cost of larger
        patch-size variation.
    patch_fill : float
        *High-curvature mode only.*  Fraction of the arc-length cell spacing
        used as the full patch width: ``wu = patch_fill × arc_spacing``, where
        ``arc_spacing = total_arclen / n_u``.

        - ``1.0`` — patches exactly touch (maximum coverage, zero gap).
        - ``0.75`` — default; patches cover 75 % of arc-length spacing,
          leaving a 25 % gap per edge.  Coverage ≈ ``patch_fill²`` for
          uniformly curved surfaces (≈ 56 % at default).
        - ``0.5`` — 50 % gap; conservative, no risk of overlap even on very
          coarse grids.

        Has no effect in low-curvature mode (full arc-length is always used).
    curvature_threshold : float, optional
        Maximum metric value below which the surface is considered low
        curvature.  Default 1.1.

    Returns
    -------
    dict
        Patch mosaic with keys ``corners``, ``centers``, ``normals``,
        ``tangents_u``, ``tangents_v``, ``wu``, ``wv``, ``el_idx``,
        ``coverage``, and ``n_rejected``.
    """
    u0, u1 = u_range
    v0, v1 = v_range

    du_nominal = (u1 - u0) / n_u
    dv_nominal = (v1 - v0) / n_v

    # ------------------------------------------------------------------
    # Curvature detection — sample metric along the parameter centre lines
    # ------------------------------------------------------------------
    v_center = 0.5 * (v0 + v1)
    u_center = 0.5 * (u0 + u1)
    eps_ref = max(u1 - u0, v1 - v0) * 1e-4

    def _metric_u(u: float) -> float:
        drdu = (
            surface_fn(u + eps_ref, v_center) - surface_fn(u - eps_ref, v_center)
        ) / (2.0 * eps_ref)
        return float(np.linalg.norm(drdu))

    def _metric_v(v: float) -> float:
        drdv = (
            surface_fn(u_center, v + eps_ref) - surface_fn(u_center, v - eps_ref)
        ) / (2.0 * eps_ref)
        return float(np.linalg.norm(drdv))

    _sample_u = [u0 + (k + 0.5) * du_nominal for k in range(min(n_u, 10))]
    _sample_v = [v0 + (k + 0.5) * dv_nominal for k in range(min(n_v, 10))]
    _max_metric = max(
        max(_metric_u(u) for u in _sample_u),
        max(_metric_v(v) for v in _sample_v),
    )

    _high_curvature = _max_metric > curvature_threshold

    # ------------------------------------------------------------------
    # Build parameter-space grid edges
    # ------------------------------------------------------------------
    if _high_curvature:
        print(
            "High curvature detected (max ||∂r/∂u|| = {:.2f} > {:.2f})".format(
                _max_metric, curvature_threshold
            )
        )
        # Arc-length adapted: centres are uniformly spaced on the surface.
        # Metric sampled along the centre lines is a good 1-D approximation
        # for rotationally symmetric surfaces (exact at the rim, worst case).
        u_edges, _ = _arclen_adapted_edges(_metric_u, u0, u1, n_u)
        v_edges, _ = _arclen_adapted_edges(_metric_v, v0, v1, n_v)
    else:
        # Low curvature: uniform Cartesian grid — already optimal.
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
    rejected_count: List[int] = [0]

    def _add_patch(uc: float, vc: float, ddu: float, ddv: float) -> None:
        # --- centre on the curved surface ---
        cen = surface_fn(uc, vc)

        # --- local tangent frame via central finite differences ---
        eps_u = ddu * 0.01
        eps_v = ddv * 0.01
        drdu = (surface_fn(uc + eps_u, vc) - surface_fn(uc - eps_u, vc)) / (2.0 * eps_u)
        drdv = (surface_fn(uc, vc + eps_v) - surface_fn(uc, vc - eps_v)) / (2.0 * eps_v)

        len_u = float(np.linalg.norm(drdu))
        len_v = float(np.linalg.norm(drdv))

        tu = drdu / max(len_u, 1e-30)
        tv_raw = drdv / max(len_v, 1e-30)
        # Gram-Schmidt: orthogonalise tv against tu
        tv_orth = tv_raw - float(np.dot(tv_raw, tu)) * tu
        tv = tv_orth / max(float(np.linalg.norm(tv_orth)), 1e-30)

        n_vec = normal_sign * np.cross(tu, tv)
        n_vec /= max(float(np.linalg.norm(n_vec)), 1e-30)

        # --- arc-length half-extents (used for rejection test only) ---
        wu_half_arc = len_u * (ddu * 0.5)
        wv_half_arc = len_v * (ddv * 0.5)

        # reject if local curvature makes the flat-rectangle approximation
        # too poor (arc-length much larger than the parameter cell)
        if wu_half_arc > max_patch_scale * (
            ddu * 0.5
        ) or wv_half_arc > max_patch_scale * (ddv * 0.5):
            rejected_count[0] += 1
            return

        # --- patch half-extents ---
        # Low-curvature mode: use full arc-length extent so adjacent tilted
        # rectangles share edges to first order (the second-order overlap from
        # surface tilt is < 0.01 % at this curvature level).
        # High-curvature mode: scale by patch_fill so the flat rectangle fits
        # within the arc-length cell without physically overlapping its
        # neighbours.  patch_fill = 1.0 → touching; patch_fill < 1 → uniform
        # gap proportional to (1 − patch_fill).
        if _high_curvature:
            wu_half = patch_fill * wu_half_arc
            wv_half = patch_fill * wv_half_arc
        else:
            wu_half = wu_half_arc
            wv_half = wv_half_arc

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
                continue  # entirely outside aperture
            else:
                # Boundary cell — subdivide and keep sub-patches inside
                sdu = ddu / border_refine
                sdv = ddv / border_refine
                for si in range(border_refine):
                    for sj in range(border_refine):
                        suc = u0c + (si + 0.5) * (u1c - u0c) / border_refine
                        svc = v0c + (sj + 0.5) * (v1c - v0c) / border_refine
                        if inside_fn(suc, svc):
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
            if inside_fn is not None and not inside_fn(_uc, _vc):
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
    n_rejected = rejected_count[0]

    print(
        f"  Patches: {M} accepted / {M + n_rejected} attempted"
        + (f", {n_rejected} rejected (oversized)" if n_rejected > 0 else "")
        + f"  |  Coverage: {coverage * 100:.1f}%"
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
        "n_rejected": n_rejected,
    }
