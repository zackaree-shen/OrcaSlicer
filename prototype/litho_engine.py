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

    # MIXED is a valid *order label* ONLY for INTERLEAVED mode (Bambu 方案B).
    # LAYERED + MIXED is a user error (LAYERED has exactly 6 CMY permutations);
    # do NOT silently reroute it to INTERLEAVED, otherwise LAYERED/MIXED and
    # INTERLEAVED/MIXED produce identical geometry and confuse the user.
    if order == ColorOrder.MIXED and mode != LithoMode.INTERLEAVED:
        raise ValueError(
            f"ColorOrder.MIXED only applies to INTERLEAVED mode, "
            f"got mode={mode.value}. For LAYERED use one of the 6 CMY orders.")

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

    # Floor for height-field layers. A 0-thickness pixel in a height-field mesh
    # makes front/back vertices coincide and produces degenerate side-wall
    # triangles (fails validate_mesh's degenerate check). 1e-3 mm is far below
    # any real layer height so it does not affect the print.
    MIN_THICKNESS = 1e-3

    meshes = {}

    # White base: full slab on the coarse grid, always present.
    tW = np.full((gy_c, gx_c), dW)
    meshes["W"] = heightfield_to_mesh(tW, params_cmy, z_offset=0.0)

    if mode == LithoMode.GREYSCALE:
        # Single grey relief (M1): brightness -> thickness, using the real
        # greyscale pipeline (litho_core.thickness_map), NOT the color solver's
        # dTop. dark pixel -> thick (less backlight), white -> thin.
        from litho_core import thickness_map
        grey = (0.299 * rgb_image[..., 0] + 0.587 * rgb_image[..., 1] +
                0.114 * rgb_image[..., 2]).astype(np.uint8)
        g = _resample(grey.astype(np.float64), (gy_t, gx_t))
        g = np.clip(g, 0, 255).astype(np.uint8)
        h = thickness_map(g, params_top)   # (gy_t, gx_t) thickness in mm
        meshes["top"] = heightfield_to_mesh(h, params_top, z_offset=0.0)
        # C/M/Y are unused in greyscale mode: attach empty meshes so callers
        # can iterate all five keys uniformly.
        empty = (np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64))
        for ch in ("C", "M", "Y"):
            meshes[ch] = empty
        # dE is not meaningful for pure greyscale (no color matching).
        dE = np.zeros((gy_t, gx_t))
        return meshes, dE, gamut, None

    if mode == LithoMode.LAYERED:
        bases = _z_bases_for_order(order)
        # Resample each color's thickness to the coarse grid and build the
        # height-field in its Z band.
        for ch in ("C", "M", "Y"):
            t = np.maximum(_resample(thickness[ch], (gy_c, gx_c)), MIN_THICKNESS)
            meshes[ch] = heightfield_to_mesh(t, params_cmy, z_offset=bases[ch])
        tTop = np.maximum(_resample(dTop, (gy_t, gx_t)), MIN_THICKNESS)
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
        meshes[ch] = (verts, faces)

    # The top relief layer's bottom must follow the actual C/M/Y fill height
    # (z_lo + max(dC,dM,dY)) per pixel, otherwise it floats in air over pixels
    # whose color boxes are shorter than the full band. Use a variable-bottom
    # band mesh: bottom = fill top, top = bottom + dTop.
    #
    # The color boxes live on the COARSE grid; the top relief is on the FINE
    # grid. A bilinear upsample of the coarse fill would dilute box tops at box
    # edges and re-open small gaps, so we upsample the fill by NEAREST neighbor:
    # every fine point takes the fill of the coarse cell it falls in, which is
    # guaranteed >= that cell's actual box top. The band mesh bottom is then a
    # staircase that never dips below the material below it.
    from litho_core import heightfield_band_mesh
    dC_c = _resample(dC, (gy_c, gx_c))
    dM_c = _resample(dM, (gy_c, gx_c))
    dY_c = _resample(dY, (gy_c, gx_c))
    fill_coarse = z_lo + np.maximum.reduce([dC_c, dM_c, dY_c])   # (gy_c, gx_c)

    def _nearest_upsample(a, shape):
        gy_f, gx_f = shape
        gy_c0, gx_c0 = a.shape
        iy = np.clip(np.round(np.linspace(0, gy_c0 - 1, gy_f)).astype(int), 0, gy_c0 - 1)
        ix = np.clip(np.round(np.linspace(0, gx_c0 - 1, gx_f)).astype(int), 0, gx_c0 - 1)
        return a[np.ix_(iy, ix)]

    fill_fine = _nearest_upsample(fill_coarse, (gy_t, gx_t))
    bot = fill_fine + LAYER_GAP                       # never below the color boxes
    topf = bot + np.maximum(_resample(dTop, (gy_t, gx_t)), MIN_THICKNESS)
    meshes["top"] = heightfield_band_mesh(bot, topf, params_top)
    reached = gamut["rgb8"][idx]
    return meshes, dE, gamut, reached
