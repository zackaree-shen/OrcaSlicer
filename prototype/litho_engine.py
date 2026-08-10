"""Lithophane prototype - pluggable engine (mode x color-order).

Architecture:
  - The forward model is Beer-Lambert transmission, which is a *product* of
    per-color exponentials:

        tau_c = 10^(-d_C/TD_Cc - d_M/TD_Mc - d_Y/TD_Yc - d_W/TD_Wc)

    By commutativity of addition, the transmitted color is independent of the
    order in which C/M/Y layers are stacked. Therefore the gamut card and the
    inverse solver are SHARED across all color orders; the order only changes
    the GEOMETRY (which Z band holds which color).

  - LAYERED     : each color occupies its own Z band; order = stacking order.
  - INTERLEAVED : C/M/Y share one Z band, each color is a local height-field
                  present only where that color's thickness > 0 (pixel boxes).
                  This is Bambu's mixed/方案B approach and never leaves thin
                  membrane holes.

Usage:
    from litho_engine import LithoMode, ColorOrder, color_lithophane_engine
    meshes, dE, gamut, reached = color_lithophane_engine(
        rgb_image, mode=LithoMode.LAYERED, order=ColorOrder.YMC, ...)
"""

from __future__ import annotations

import enum

import numpy as np

from litho_core import LithophaneParams, heightfield_to_mesh
from litho_color import (
    build_gamut_stacked,
    solve_stacked,
    _resample,
    _resample_rgb,
    DEFAULT_TD,
    WHITE_THICKNESS,
    COLOR_BAND_MAX,
    TOP_BAND_MAX,
    Z_C_BASE,
    Z_M_BASE,
    Z_Y_BASE,
    Z_TOP_BASE,
    LAYER_GAP,
)


class LithoMode(enum.Enum):
    """Geometry strategy for how the C/M/Y color layers are placed in Z."""

    GREYSCALE = "greyscale"    # single grey relief, no color
    LAYERED = "layered"        # strict Z separation, one color per band
    INTERLEAVED = "interleaved"  # C/M/Y share one Z band (mixed / Bambu B)


class ColorOrder(enum.Enum):
    """Stacking order of the three color bands (LAYERED) or same-band paint
    order (INTERLEAVED). Order does NOT affect color; it only changes geometry.
    MIXED = INTERLEAVED mode."""

    CMY = "CMY"
    CYM = "CYM"
    MCY = "MCY"
    MYC = "MYC"
    YMC = "YMC"
    YCM = "YCM"
    MIXED = "MIXED"  # resolves to INTERLEAVED mode


# ---------------------------------------------------------------------------
# Color-engine mapping
# ---------------------------------------------------------------------------

_ORDER_LETTERS = {
    ColorOrder.CMY: ("C", "M", "Y"),
    ColorOrder.CYM: ("C", "Y", "M"),
    ColorOrder.MCY: ("M", "C", "Y"),
    ColorOrder.MYC: ("M", "Y", "C"),
    ColorOrder.YMC: ("Y", "M", "C"),
    ColorOrder.YCM: ("Y", "C", "M"),
}

# Per-color Z-band base for LAYERED mode (default CMY order). The engine
# reorders these bases according to ColorOrder.
_COLOR_BASE = {"C": Z_C_BASE, "M": Z_M_BASE, "Y": Z_Y_BASE}


def _z_bases_for_order(order: ColorOrder):
    """Return {color: z_offset} for LAYERED mode under the given order.

    The first color in the order sits directly above the white base, the next
    above it, etc. Total stack height is identical regardless of order.
    """
    letters = _ORDER_LETTERS[order]
    base = WHITE_THICKNESS + LAYER_GAP
    out = {}
    for ch in letters:
        out[ch] = base
        base += COLOR_BAND_MAX + LAYER_GAP
    return out


# ---------------------------------------------------------------------------
# INTERLEAVED geometry: pixel boxes (each box is a closed solid)
# ---------------------------------------------------------------------------

def _pixel_boxes_mesh(mask, thickness, z_lo, z_hi, dx, dy):
    """Build a mesh of axis-aligned boxes for each pixel where mask is True.

    Every box is a closed solid (12 triangles), so the union is watertight with
    no thin-membrane artifacts. Boxes are placed from z_lo to z_lo + thickness.
    thickness is clipped to z_hi - z_lo.
    """
    gy, gx = mask.shape
    verts = []
    faces = []
    # Box faces: front/back/left/right/top/bottom, CCW outward. Precompute the
    # 8 corner offsets for a unit box, then translate/scale per pixel.
    # We emit a separate 8-vertex box per pixel for simplicity and correctness.
    def add_box(x0, y0, z0, x1, y1, z1, vbase):
        # 8 corners, CCW-outward faces. Standard unit-cube triangulation.
        corners = [
            (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),  # bottom 0-3
            (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1),  # top 4-7
        ]
        for p in corners:
            verts.append(p)
        # 12 triangles, outward-facing (verified with validate_mesh).
        tris = [
            (0, 2, 1), (0, 3, 2),  # -Z bottom
            (4, 5, 6), (4, 6, 7),  # +Z top
            (0, 1, 5), (0, 5, 4),  # -Y
            (1, 2, 6), (1, 6, 5),  # +X
            (2, 3, 7), (2, 7, 6),  # +Y
            (3, 0, 4), (3, 4, 7),  # -X
        ]
        for a, b, c in tris:
            faces.append((vbase + a, vbase + b, vbase + c))

    for iy in range(gy):
        for ix in range(gx):
            if not mask[iy, ix]:
                continue
            th = float(np.clip(thickness[iy, ix], 0.0, z_hi - z_lo))
            if th < 1e-4:
                continue
            x0 = ix * dx
            y0 = iy * dy
            x1 = (ix + 1) * dx
            y1 = (iy + 1) * dy
            add_box(x0, y0, z_lo, x1, y1, z_lo + th, len(verts))

    if not verts:
        # Degenerate: return an empty mesh (caller must handle).
        import numpy as _np
        return _np.zeros((0, 3)), _np.zeros((0, 3), dtype=_np.int64)
    return np.array(verts, dtype=np.float64), np.array(faces, dtype=np.int64)


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

def color_lithophane_engine(rgb_image, mode=LithoMode.LAYERED, order=ColorOrder.CMY,
                            params=None, td=None, layers_max=8, layer_h=0.08,
                            dW=WHITE_THICKNESS, top_max=TOP_BAND_MAX, exact=False,
                            pitch_cmy=0.8, pitch_top=0.25):
    """Generate lithophane meshes under a given mode and color order.

    Returns (meshes, dE, gamut, reached_rgb) where meshes maps color -> mesh.
    Keys: 'W','C','M','Y','top' always present (C/M/Y boxes may be empty in
    INTERLEAVED mode if a color is unused).
    """
    if params is None:
        params = LithophaneParams()
    if td is None:
        td = DEFAULT_TD

    # MIXED order is the same as INTERLEAVED mode (Bambu 方案B).
    if order == ColorOrder.MIXED:
        mode = LithoMode.INTERLEAVED

    from litho_core import thickness_grid_shape
    gx, gy = thickness_grid_shape(rgb_image.shape[0], rgb_image.shape[1], params)
    small = _resample_rgb(rgb_image, (gy, gx))

    # Shared gamut + inverse solver (order-independent by Beer-Lambert).
    gamut = build_gamut_stacked(layers_max=layers_max, layer_h=layer_h,
                                top_max=top_max, dW=dW, td=td)
    dTop, dC, dM, dY, dE, idx = solve_stacked(small, gamut, exact=exact)

    thickness = {"C": dC, "M": dM, "Y": dY, "top": dTop, "W": np.full_like(dTop, dW)}

    # Dual-resolution grids.
    params_cmy = LithophaneParams(width_mm=params.width_mm, height_mm=params.height_mm,
                                  pixel_pitch_mm=pitch_cmy)
    params_top = LithophaneParams(width_mm=params.width_mm, height_mm=params.height_mm,
                                  pixel_pitch_mm=pitch_top)
    gx_c, gy_c = thickness_grid_shape(rgb_image.shape[0], rgb_image.shape[1], params_cmy)
    gx_t, gy_t = thickness_grid_shape(rgb_image.shape[0], rgb_image.shape[1], params_top)

    meshes = {}

    # White base: full slab on the coarse grid, always present.
    tW = np.full((gy_c, gx_c), dW)
    meshes["W"] = heightfield_to_mesh(tW, params_cmy, z_offset=0.0)

    if mode == LithoMode.GREYSCALE:
        # Single grey relief (M1): brightness -> top thickness only.
        grey = np.asarray(rgb_image.convert("L") if hasattr(rgb_image, "convert") else
                          (0.299 * rgb_image[..., 0] + 0.587 * rgb_image[..., 1] +
                           0.114 * rgb_image[..., 2]))
        tTop = _resample(dTop, (gy_t, gx_t))
        meshes["top"] = heightfield_to_mesh(tTop, params_top, z_offset=0.0)
        # C/M/Y are unused in greyscale mode: attach empty meshes so callers
        # can iterate all five keys uniformly.
        empty = (np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64))
        for ch in ("C", "M", "Y"):
            meshes[ch] = empty
        return meshes, dE, gamut, None

    if mode == LithoMode.LAYERED:
        bases = _z_bases_for_order(order)
        # Resample each color's thickness to the coarse grid and build the
        # height-field in its Z band.
        for ch in ("C", "M", "Y"):
            t = _resample(thickness[ch], (gy_c, gx_c))
            meshes[ch] = heightfield_to_mesh(t, params_cmy, z_offset=bases[ch])
        tTop = _resample(dTop, (gy_t, gx_t))
        meshes["top"] = heightfield_to_mesh(tTop, params_top, z_offset=Z_TOP_BASE)
        reached = gamut["rgb8"][idx]
        return meshes, dE, gamut, reached

    # INTERLEAVED: C/M/Y share one Z band, pixel boxes only where thickness > 0.
    dx = params_cmy.width_mm / float(gx_c - 1)
    dy = params_cmy.height_mm / float(gy_c - 1)
    z_lo = Z_C_BASE  # all three colors share the same Z band
    z_hi = z_lo + COLOR_BAND_MAX
    for ch in ("C", "M", "Y"):
        t = _resample(thickness[ch], (gy_c, gx_c))
        mask = t > 0.02  # ignore sub-membrane thickness
        verts, faces = _pixel_boxes_mesh(mask, t, z_lo, z_hi, dx, dy)
        from litho_core import heightfield_to_mesh as _h  # noqa
        # Build a TriangleMesh-style (vertices, faces) container.
        meshes[ch] = (verts, faces)
    tTop = _resample(dTop, (gy_t, gx_t))
    meshes["top"] = heightfield_to_mesh(tTop, params_top, z_offset=Z_TOP_BASE)
    reached = gamut["rgb8"][idx]
    return meshes, dE, gamut, reached
