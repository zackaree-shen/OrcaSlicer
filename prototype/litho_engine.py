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
    STACKED = "stacked"        # Bambu-style: per-pixel C->M->Y continuous
                               # stacked column, zero gap, never floats
    OVERLAP = "overlap"        # C/M/Y same-base overlapping boxes + MAX color
                               # model (what overlapping geometry actually
                               # prints) — gamut-collapsed comparison mode


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


def _band_height(layer_h, layers_max):
    """Height of one C/M/Y color band in mm (layers_max * layer_h)."""
    return layers_max * layer_h


def _z_bases_for_order(order: ColorOrder, dW=WHITE_THICKNESS,
                       layer_h=0.08, layers_max=8):
    """Return {color: z_offset} for LAYERED mode under the given order.

    The first color in the order sits directly above the white base, the next
    above it, etc. Total stack height is identical regardless of order.
    Z positions are computed from the actual layer height so they stay valid
    when the user changes layer_h (e.g. 0.2 mm default).
    """
    letters = _ORDER_LETTERS[order]
    band = _band_height(layer_h, layers_max)
    base = dW + LAYER_GAP
    out = {}
    for ch in letters:
        out[ch] = base
        base += band + LAYER_GAP
    return out


def _color_band_bounds(dW=WHITE_THICKNESS, layer_h=0.08, layers_max=8):
    """Return (z_lo, z_hi) of the shared C/M/Y band (INTERLEAVED mode)."""
    z_lo = dW + LAYER_GAP
    z_hi = z_lo + _band_height(layer_h, layers_max)
    return z_lo, z_hi


def _top_base(dW=WHITE_THICKNESS, layer_h=0.08, layers_max=8):
    """Z offset where the top relief band starts (above the C/M/Y bands)."""
    z_lo, z_hi = _color_band_bounds(dW, layer_h, layers_max)
    return z_hi + LAYER_GAP


# ---------------------------------------------------------------------------
# INTERLEAVED geometry: pixel boxes (each box is a closed solid)
# ---------------------------------------------------------------------------

def _pixel_boxes_mesh(mask, thickness, z_lo, z_hi, dx, dy):
    """Build a mesh of axis-aligned boxes for each pixel where mask is True.

    Every box is a closed solid (12 triangles), so the union is watertight with
    no thin-membrane artifacts. Boxes are placed from z_lo to z_lo + thickness.
    z_lo and z_hi may be scalars or per-pixel arrays (thickness is clamped to
    z_hi - z_lo per pixel). Used for both same-base boxes and stacked columns
    where each color segment starts at a per-pixel height.
    """
    gy, gx = mask.shape
    if np.isscalar(z_lo):
        z_lo_arr = np.full((gy, gx), float(z_lo))
    else:
        z_lo_arr = np.asarray(z_lo, dtype=np.float64)
    if np.isscalar(z_hi):
        z_hi_arr = np.full((gy, gx), float(z_hi))
    else:
        z_hi_arr = np.asarray(z_hi, dtype=np.float64)
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

    # Build (gx-1)*(gy-1) boxes, one per CELL, so the union covers exactly
    # [0, (gx-1)*dx] x [0, (gy-1)*dy] — matching the W base and top relief
    # extents. Building gx*gy boxes would extend half a cell beyond the model
    # and offset the color layers 0.5 mm in +x/+y (adversarial finding #7).
    for iy in range(gy - 1):
        for ix in range(gx - 1):
            if not mask[iy, ix]:
                continue
            lo = z_lo_arr[iy, ix]
            hi = z_hi_arr[iy, ix]
            th = float(np.clip(thickness[iy, ix], 0.0, hi - lo))
            if th < 1e-4:
                continue
            x0 = ix * dx
            y0 = iy * dy
            x1 = (ix + 1) * dx
            y1 = (iy + 1) * dy
            add_box(x0, y0, lo, x1, y1, lo + th, len(verts))

    if not verts:
        # Degenerate: return an empty mesh (caller must handle).
        import numpy as _np
        return _np.zeros((0, 3)), _np.zeros((0, 3), dtype=_np.int64)
    return np.array(verts, dtype=np.float64), np.array(faces, dtype=np.int64)


def _nearest_upsample(a, shape):
    """Nearest-neighbour upsample of a (gy_c, gx_c) array to (gy_f, gx_f).

    Used to carry a coarse-grid field (e.g. color-column fill height) onto the
    fine top-relief grid without bilinear dilution at box edges.
    """
    gy_f, gx_f = shape
    gy_c0, gx_c0 = a.shape
    iy = np.clip(np.round(np.linspace(0, gy_c0 - 1, gy_f)).astype(int), 0, gy_c0 - 1)
    ix = np.clip(np.round(np.linspace(0, gx_c0 - 1, gx_f)).astype(int), 0, gx_c0 - 1)
    return a[np.ix_(iy, ix)]


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

    # MIXED is a valid *order label* ONLY for the same-base modes
    # (INTERLEAVED / OVERLAP, Bambu 方案B). LAYERED + MIXED is a user error
    # (LAYERED has exactly 6 CMY permutations); do NOT silently reroute it.
    if order == ColorOrder.MIXED and mode not in (LithoMode.INTERLEAVED, LithoMode.OVERLAP):
        raise ValueError(
            f"ColorOrder.MIXED only applies to INTERLEAVED/OVERLAP modes, "
            f"got mode={mode.value}. For LAYERED use one of the 6 CMY orders.")

    from litho_core import thickness_grid_shape
    gx, gy = thickness_grid_shape(rgb_image.shape[0], rgb_image.shape[1], params)
    small = _resample_rgb(rgb_image, (gy, gx))

    # Gamut + inverse solver. INTERLEAVED/STACKED/LAYERED use the stacked
    # (sum, Beer-Lambert product) model; OVERLAP uses the max model matching
    # the same-base overlapping geometry it actually prints.
    if mode == LithoMode.OVERLAP:
        from litho_color import build_gamut_overlap
        gamut = build_gamut_overlap(layers_max=layers_max, layer_h=layer_h,
                                    top_max=top_max, dW=dW, td=td)
    else:
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
        bases = _z_bases_for_order(order, dW=dW, layer_h=layer_h, layers_max=layers_max)
        # Top relief sits above the LAST color band (not above the shared band
        # used by INTERLEAVED).
        last_base = max(bases.values())
        top_z = last_base + _band_height(layer_h, layers_max) + LAYER_GAP
        # Resample each color's thickness to the coarse grid and build the
        # height-field in its Z band.
        for ch in ("C", "M", "Y"):
            t = np.maximum(_resample(thickness[ch], (gy_c, gx_c)), MIN_THICKNESS)
            meshes[ch] = heightfield_to_mesh(t, params_cmy, z_offset=bases[ch])
        tTop = np.maximum(_resample(dTop, (gy_t, gx_t)), MIN_THICKNESS)
        meshes["top"] = heightfield_to_mesh(tTop, params_top, z_offset=top_z)
        reached = gamut["rgb8"][idx]
        return meshes, dE, gamut, reached

    if mode == LithoMode.STACKED:
        # Bambu-style: per-pixel C->M->Y continuous stacked column.
        # Each pixel is ONE continuous column from z_lo to z_lo + dC + dM + dY,
        # with C/M/Y occupying contiguous sub-segments (zero internal gap; air
        # transmits 1.0 so the Beer-Lambert product is unchanged). Every column
        # starts at the same z_lo (on the white base) -> never floats.
        #
        # We emit each color as its own pixel-box layer within the shared band
        # so the slicer can assign per-color extruders by Z range, matching
        # Bambu's structure (color boxes stacked within one low Z band).
        from litho_core import heightfield_band_mesh
        dx = params_cmy.width_mm / float(gx_c - 1)
        dy = params_cmy.height_mm / float(gy_c - 1)
        # Color band starts right above the white base.
        z_lo = dW + LAYER_GAP
        dC_c = _resample(dC, (gy_c, gx_c))
        dM_c = _resample(dM, (gy_c, gx_c))
        dY_c = _resample(dY, (gy_c, gx_c))
        # C column: [z_lo, z_lo + dC]
        # M column: [z_lo + dC, z_lo + dC + dM]   (0-thickness skipped)
        # Y column: [z_lo + dC + dM, z_lo + dC + dM + dY]
        c_top = z_lo + dC_c
        m_top = c_top + dM_c
        y_top = m_top + dY_c
        # Continuous height fields (NOT per-pixel boxes). STACKED keeps the
        # physically-correct per-pixel C->M->Y stacked columns: C sits on the
        # base, M on C's top, Y on M's top — each is a band mesh whose bottom
        # follows the previous color's top (variable-bottom). This eliminates
        # the fragmented-island appearance while preserving sum-model physics.
        MIN_FLOOR = 1e-3
        meshes["C"] = heightfield_to_mesh(np.maximum(dC_c, MIN_FLOOR), params_cmy, z_offset=z_lo)
        meshes["M"] = heightfield_band_mesh(c_top, np.maximum(m_top, c_top + MIN_FLOOR), params_cmy)
        meshes["Y"] = heightfield_band_mesh(m_top, np.maximum(y_top, m_top + MIN_FLOOR), params_cmy)
        # Top relief: bottom follows the tallest column (y_top where present,
        # else m_top / c_top / z_lo), never floats.
        fill_coarse = np.maximum.reduce([y_top, m_top, c_top,
                                         np.full_like(c_top, z_lo)])
        fill_fine = _nearest_upsample(fill_coarse, (gy_t, gx_t))
        bot = fill_fine + LAYER_GAP
        topf = bot + np.maximum(_resample(dTop, (gy_t, gx_t)), MIN_THICKNESS)
        meshes["top"] = heightfield_band_mesh(bot, topf, params_top)
        reached = gamut["rgb8"][idx]
        return meshes, dE, gamut, reached

    # INTERLEAVED and OVERLAP share the same same-base overlapping geometry:
    # C/M/Y height fields all start at z_lo and have different heights, so
    # they overlap in Z. INTERLEAVED matches it with the sum color card
    # (preview only — printing collapses to max); OVERLAP matches it with the
    # segstack color card (what the overlapping geometry actually prints).
    # Fall-through from both modes reaches this block.
    dx = params_cmy.width_mm / float(gx_c - 1)
    dy = params_cmy.height_mm / float(gy_c - 1)
    z_lo, z_hi = _color_band_bounds(dW, layer_h, layers_max)  # shared C/M/Y band
    MIN_FLOOR = 1e-3
    for ch in ("C", "M", "Y"):
        t = _resample(thickness[ch], (gy_c, gx_c))
        # Continuous height field (NOT per-pixel boxes) starting at the shared
        # z_lo; tiny floor at 0-thickness pixels. Eliminates fragmented islands
        # and keeps face count low (2 tris/cell), matching Bambu's geometry.
        meshes[ch] = heightfield_to_mesh(np.maximum(t, MIN_FLOOR), params_cmy, z_offset=z_lo)

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

    fill_fine = _nearest_upsample(fill_coarse, (gy_t, gx_t))
    bot = fill_fine + LAYER_GAP                       # never below the color boxes
    topf = bot + np.maximum(_resample(dTop, (gy_t, gx_t)), MIN_THICKNESS)
    meshes["top"] = heightfield_band_mesh(bot, topf, params_top)
    reached = gamut["rgb8"][idx]
    return meshes, dE, gamut, reached
