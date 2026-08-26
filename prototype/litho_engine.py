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
    refine_dtop_surface,
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
    BAMBU = "bambu"            # Bambu reference: ONE complete white model
                               # (base slab + relief plate merged into a
                               # single part) carries detail + brightness;
                               # C/M/Y are color channels in a Z band between
                               # the two white volumes, overlapping each
                               # other (nC/nM/nY stacked layers). W and CMY
                               # do not overlap in Z (bilinear fill follow).
                               # White grid finer than CMY (0.22 vs 0.44mm).


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


def _flip_mesh_y(mesh, height_mm):
    """Mirror mesh along the build-plate Y axis so the top view reads like the
    source image (image top -> +Y). Preserves outward face winding."""
    verts, faces = mesh
    if len(verts) == 0:
        return verts, faces
    v = verts.copy()
    v[:, 1] = height_mm - v[:, 1]
    # Mirror reverses orientation; swap two vertex indices per triangle to keep
    # faces outward-facing (and positive signed volume).
    f = faces[:, [0, 2, 1]].copy()
    return v, f


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

def color_lithophane_engine(rgb_image, mode=LithoMode.LAYERED, order=ColorOrder.CMY,
                            params=None, td=None, layers_max=8, layer_h=0.08,
                            dW=WHITE_THICKNESS, top_max=TOP_BAND_MAX, exact=False,
                            pitch_cmy=0.30, pitch_top=0.15, smooth_top=True,
                            carve="concave", sharpen=2.0, contrast=1.5,
                            tone_map=True, surface_refine=False, detail_level=1.0,
                            c_strength=1.0, m_strength=1.0, y_strength=1.0,
                            w_strength=1.0, flip_y=True):
    """Generate lithophane meshes under a given mode and color order.

    sharpen/contrast: image preprocessing applied to the luminance channel
    BEFORE solving (hue-preserving). Improves edge sharpness / detail in the
    relief. Default sharpen=2.0, contrast=1.5 (mild).

    detail_level: 0.0 = maximum smoothing / region merging (sacrifices fine
    detail for printability); 1.0 = preserve as much detail as possible. Maps
    to edge-aware refinement strength and morphological clean-up radius.

    c/m/y/w_strength: per-color effective density correction. >1 makes that
    color "stronger" in the model, so the solver uses less of it. Use this to
    compensate for filament color cast (e.g. y_strength=1.2 to reduce yellow).

    flip_y: if True (default), mirror the mesh along the build-plate Y axis so
    the top view reads like the original image (image top -> +Y). Set False to
    keep the legacy orientation where image top maps to -Y.

    Returns (meshes, dE, gamut, reached_rgb) where meshes maps color -> mesh.
    Keys: 'W','C','M','Y','top' always present (C/M/Y boxes may be empty in
    INTERLEAVED mode if a color is unused).
    """
    if params is None:
        params = LithophaneParams()
    if td is None:
        td = DEFAULT_TD

    # Per-color density correction (printer-profile style). Applied once here
    # and used for gamut + forward evaluation so solver and honest dE agree.
    from litho_color import correct_td
    td = correct_td(td, c=c_strength, m=m_strength, y=y_strength, w=w_strength)

    # detail_level maps to a controlled detail/smoothness trade-off.
    # 0.0 -> heavy smoothing + region merging; 1.0 -> crisp detail.
    detail_level = float(np.clip(detail_level, 0.0, 1.0))
    edge_alpha = 0.20 + 0.60 * detail_level          # 0.20 .. 0.80
    morph_radius = int(round(3.0 * (1.0 - detail_level)))  # 3 .. 0
    refine_gamma = 0.10 + 0.12 * (1.0 - detail_level)      # 0.22 .. 0.10
    refine_iter = int(round(20 + 40 * (1.0 - detail_level)))  # 60 .. 20

    # MIXED is a valid *order label* ONLY for the same-base modes
    # (INTERLEAVED / OVERLAP / BAMBU, Bambu 方案B). LAYERED + MIXED is a user
    # error (LAYERED has exactly 6 CMY permutations); do NOT silently reroute.
    if order == ColorOrder.MIXED and mode not in (
            LithoMode.INTERLEAVED, LithoMode.OVERLAP, LithoMode.BAMBU):
        raise ValueError(
            f"ColorOrder.MIXED only applies to INTERLEAVED/OVERLAP/BAMBU modes, "
            f"got mode={mode.value}. For LAYERED use one of the 6 CMY orders.")

    from litho_core import thickness_grid_shape
    # P1a-v2 (iteration 45): dTop is a DIRECT monotone field anchored to the
    # image's own dynamic range (anchored_dtop_field) — NOT the v1
    # "re-tone-the-RGB" route (tone_mapping_preprocess), which absolute
    # Beer-Lambert inversion + window clip crushed 84.4% of a dark cartoon to
    # dTop == top_max (giant flat plateaus). Keep an unprocessed resample of
    # the ORIGINAL image for honest dE reporting (adversarial review B3: the
    # old dE was computed against the tone-mapped image — a self-flattering
    # metric).
    _p1 = tone_map and mode in (LithoMode.BAMBU, LithoMode.OVERLAP)
    if _p1:
        _p1_orig = rgb_image.copy()
    # Image preprocessing (sharpen + contrast on luminance, hue-preserving).
    # Applied BEFORE solving so the solver sees crisper edges.
    if sharpen > 0 or abs(contrast - 1.0) > 1e-6:
        from litho_color import preprocess_image
        rgb_image = preprocess_image(rgb_image, sharpen=sharpen, contrast=contrast)
    gx, gy = thickness_grid_shape(rgb_image.shape[0], rgb_image.shape[1], params)
    small = _resample_rgb(rgb_image, (gy, gx))

    # Gamut + inverse solver. INTERLEAVED/STACKED/LAYERED use the stacked
    # (sum, Beer-Lambert product) model; OVERLAP uses the max model matching
    # the same-base overlapping geometry it actually prints. BAMBU uses a
    # dedicated card matching its reference geometry: white = thin base
    # (z_lo=0.2, like the reference) + relief (dTop), and the C/M/Y band is
    # capped at the reference band height (band=0.7) so the white relief
    # plate ALWAYS sits above the color band (never punched through).
    if mode == LithoMode.OVERLAP:
        # OVERLAP now uses the same high-detail P1 solver path as BAMBU
        # (anchored_dtop_field + resolve_cmy_for_dtop) so its surface detail
        # matches BAMBU-like quality, while the mesh still prints as thin
        # overlapping C/M/Y bands plus a white relief that follows the fill top.
        gamut = build_gamut_stacked(layers_max=layers_max, layer_h=layer_h,
                                    top_max=top_max, dW=dW, td=td)
    elif mode == LithoMode.BAMBU:
        # Reference (measured lithophane_谢bro_U1): C/M/Y band ~[0.2, 0.9]
        # (height 0.7), white relief above it up to ~2.28. We fix z_lo=0.2
        # and band=0.7 so the white plate bottom (= z_lo + band + LAYER_GAP)
        # is always above the color band top (= z_lo + band). The white
        # "base" seen by the Beer-Lambert model is z_lo (thin bottom slab),
        # NOT z_lo + band (that volume is the C/M/Y band, not white).
        # (that volume is the C/M/Y band, not white).
        z_lo = 0.2
        band = 0.7
        gamut = build_gamut_stacked(
            layers_max=max(1, int(round(band / layer_h))), layer_h=layer_h,
            top_max=top_max, dW=z_lo, td=td)
    else:
        gamut = build_gamut_stacked(layers_max=layers_max, layer_h=layer_h,
                                    top_max=top_max, dW=dW, td=td)
    # P1 path (BAMBU + tone_map) replaces _smooth_top_resolve entirely:
    # dTop is the anchored monotone FIELD (computed from the same preprocessed
    # image the CMY solve sees — adversarial review M2: v1 tone-mapped BEFORE
    # sharpen/contrast, twisting its own anchors), then spike surgery cleans
    # residual outlier gradients. Solve runs raw (smooth_top=False).
    dTop, dC, dM, dY, dE, idx = solve_stacked(small, gamut, exact=exact,
                                              smooth_top=smooth_top and not _p1)

    if _p1:
        from litho_color import (anchored_dtop_field, resolve_cmy_for_dtop,
                                 spike_surgery, forward_stacked,
                                 xyz_to_lab, linear_to_xyz, srgb8_to_linear,
                                 _dE2000_pair)
        # (a) monotone anchored dTop from the SAME image the solver saw.
        dTop = anchored_dtop_field(small, td_w=td["W"][0], top_max=top_max)
        # (a2) edge-aware surface refinement: smooth flats, preserve edges.
        # detail_level drives the smoothing strength; surface_refine=False
        # disables it entirely and falls back to legacy spike surgery.
        if surface_refine:
            dTop = refine_dtop_surface(dTop, small, edge_alpha=edge_alpha,
                                       gamma=refine_gamma, n_iter=refine_iter)
        # (b) re-solve C/M/Y with dTop pinned (kills the CMY-lattice sawtooth
        # where solver-chosen dTop snapped back to top_max across cells).
        flat_lab = xyz_to_lab(linear_to_xyz(
            srgb8_to_linear(small).reshape(-1, 3))).reshape(-1, 3)
        dC, dM, dY, dE, idx = resolve_cmy_for_dtop(flat_lab, gamut, dTop)
        # (c) legacy spike cleanup only when the new refinement is off.
        if not surface_refine:
            dTop = spike_surgery(dTop, t_sigma=1.5, iterations=2, top_max=top_max)
        # (d) HONEST dE: forward-model the ACTUAL printed geometry (surgery
        # may shift dTop within the band) against the ORIGINAL image, not the
        # preprocessed target (adversarial review B3).
        _fwd_dW = dW  # gamut white base (0.2 mm for thin overlap / BAMBU)
        _tau = forward_stacked(dTop, dC, dM, dY, td=td, dW=_fwd_dW).reshape(-1, 3)
        _lab_p = xyz_to_lab(linear_to_xyz(
            np.clip(_tau, 0, 1))).reshape(-1, 3)
        _orig_small = _resample_rgb(_p1_orig, small.shape[:2])
        _lab_t = xyz_to_lab(linear_to_xyz(
            srgb8_to_linear(_orig_small))).reshape(-1, 3)
        _sub = np.linspace(0, _lab_t.shape[0] - 1,
                           min(20000, _lab_t.shape[0])).astype(int)
        _med = float(np.median(_dE2000_pair(_lab_t[_sub], _lab_p[_sub])))
        dE = np.full_like(dTop, _med)

    # Apply edge-aware refinement to non-BAMBU modes as well (they do not use
    # the P1 anchored-dTop path above).
    if surface_refine and not _p1:
        dTop = refine_dtop_surface(dTop, small, edge_alpha=edge_alpha,
                                   gamma=refine_gamma, n_iter=refine_iter)

    # Morphological open-close to merge tiny fragments / suppress residual
    # spikes after refinement. detail_level drives the radius.
    if morph_radius > 0:
        from litho_color import morph_smooth
        dTop = morph_smooth(dTop, radius=morph_radius, iterations=1)

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
        if flip_y:
            meshes = {k: _flip_mesh_y(v, params.height_mm) for k, v in meshes.items()}
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
        if flip_y:
            meshes = {k: _flip_mesh_y(v, params.height_mm) for k, v in meshes.items()}
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
        # BILINEAR upsample of the coarse fill top onto the fine grid. A
        # height-field top IS its bilinear surface, so bilinear sampling is
        # exact. (_nearest_upsample samples coarse VERTICES, which can sit
        # below the bilinear surface between vertices -> the top relief would
        # dip into the C/M/Y columns below and interfere at slicing time;
        # measured up to -1.06mm on the test image.)
        fill_fine = _resample(fill_coarse, (gy_t, gx_t))
        bot = fill_fine + LAYER_GAP
        topf = bot + np.maximum(_resample(dTop, (gy_t, gx_t)), MIN_THICKNESS)
        meshes["top"] = heightfield_band_mesh(bot, topf, params_top)
        reached = gamut["rgb8"][idx]
        if flip_y:
            meshes = {k: _flip_mesh_y(v, params.height_mm) for k, v in meshes.items()}
        return meshes, dE, gamut, reached

    if mode == LithoMode.BAMBU:
        # Bambu reference geometry (measured from lithophane_谢bro_U1):
        #   - W is ONE complete white model: a thin base slab [0, z_lo] plus a
        #     full-coverage white relief plate above the color band. The two
        #     white volumes are merged into a single mesh (single part, white
        #     extruder) so the user sees ONE white body carrying both the
        #     detail (relief) and the brightness (thickness).
        #   - C/M/Y are COLOR CHANNELS: one overlapping Z band between the
        #     white base and the white relief plate (reference CMY band
        #     ~[0.2, 0.9]). Each position is nC/nM/nY stacked color layers;
        #     the three channels overlap (interfere).
        #   - Reference dimensions (measured): z_lo = 0.2 (thin white bottom),
        #     band = 0.7 (C/M/Y height), white relief up to ~2.28. The band is
        #     FIXED (not layers_max*layer_h) so the white plate bottom
        #     (= z_lo + band + LAYER_GAP) always sits ABOVE the color band top
        #     (= z_lo + band): the slicer's part-order clipping can only hollow
        #     the band region, never the white relief above it -> no 镂空.
        #   - White thickness seen by the Beer-Lambert model is z_lo + dTop
        #     (base slab + relief); the C/M/Y band volume is color, NOT white
        #     (no double counting — the gamut above was built with dW=z_lo).
        #   - The white relief plate bottom is sampled BILINEARLY from the
        #     coarse CMY fill top (a height-field top IS the bilinear surface),
        #     so the plate never dips into the color band (which would let the
        #     slicer's part-order clipping hollow out W) and never floats.
        z_lo = 0.2
        band = 0.7
        MIN_FLOOR = 1e-3
        from litho_core import heightfield_band_mesh

        # --- C/M/Y color channels: same-base overlapping band ---
        dC_c = _resample(dC, (gy_c, gx_c))
        dM_c = _resample(dM, (gy_c, gx_c))
        dY_c = _resample(dY, (gy_c, gx_c))
        for ch, t in (("C", dC_c), ("M", dM_c), ("Y", dY_c)):
            meshes[ch] = heightfield_to_mesh(np.maximum(t, MIN_FLOOR), params_cmy,
                                             z_offset=z_lo)

        # --- White base slab (coarse grid) ---
        tW_base = np.full((gy_c, gx_c), z_lo)
        base_v, base_f = heightfield_to_mesh(tW_base, params_cmy, z_offset=0.0)

        # --- White relief plate (fine grid), bottom follows CMY fill top ---
        fill_coarse = z_lo + np.maximum.reduce([dC_c, dM_c, dY_c])   # coarse fill top
        # BILINEAR upsample of the coarse fill top onto the fine grid — this
        # is the exact CMY height-field surface, not a vertex sample.
        fill_fine = _resample(fill_coarse, (gy_t, gx_t))
        bot = np.maximum(fill_fine, z_lo + band) + LAYER_GAP  # >= band top + gap
        dTop_fine = np.maximum(_resample(dTop, (gy_t, gx_t)), MIN_THICKNESS)
        if carve == "concave":
            # 凹刻（用户修正后的正确语义）：
            # 顶面随 dTop 起伏（凸起状，不是平面！）——暗处（dTop 大）顶面高，
            # 亮处（dTop 小）顶面低（"向下刻蚀"）。整体看着是凸起的浮雕。
            # 与凸刻的区别：底面完全平整（固定在 CMY 带顶+gap，不跟随 fill），
            # 像从一块平整底板上凸起的浮雕；凸刻底面跟随 CMY fill（有起伏）。
            flat_bot = z_lo + band + LAYER_GAP
            relief_bot = np.full_like(dTop_fine, flat_bot)
            topf = flat_bot + dTop_fine
        else:
            # 凸刻（阳刻）：底面跟随 fill，顶面 = bot + dTop
            topf = bot + dTop_fine
            relief_bot = bot
        relief_v, relief_f = heightfield_band_mesh(relief_bot, topf, params_top)

        # --- Merge base + relief into ONE white mesh (single W part) ---
        Wv = np.vstack([base_v, relief_v])
        Wf = np.vstack([base_f, relief_f + len(base_v)])
        meshes["W"] = (Wv, Wf)

        # No separate top layer — the brightness relief lives in W.
        empty = (np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64))
        meshes["top"] = empty
        reached = gamut["rgb8"][idx]
        if flip_y:
            meshes = {k: _flip_mesh_y(v, params.height_mm) for k, v in meshes.items()}
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
    # The color fields live on the COARSE grid; the top relief is on the FINE
    # grid. The coarse fill top is upsampled BILINEARLY: a height-field top IS
    # its bilinear surface, so this is exact. (_nearest_upsample samples
    # coarse vertices, which can sit below the bilinear surface between
    # vertices, so the relief would dip into the C/M/Y fields and interfere at
    # slicing time — measured up to -1.15mm on the test image. The previous
    # "nearest is guaranteed >= box top" comment was wrong.)
    from litho_core import heightfield_band_mesh
    dC_c = _resample(dC, (gy_c, gx_c))
    dM_c = _resample(dM, (gy_c, gx_c))
    dY_c = _resample(dY, (gy_c, gx_c))
    fill_coarse = z_lo + np.maximum.reduce([dC_c, dM_c, dY_c])   # (gy_c, gx_c)

    fill_fine = _resample(fill_coarse, (gy_t, gx_t))
    bot = fill_fine + LAYER_GAP                       # never below the color fields
    topf = bot + np.maximum(_resample(dTop, (gy_t, gx_t)), MIN_THICKNESS)
    meshes["top"] = heightfield_band_mesh(bot, topf, params_top)
    reached = gamut["rgb8"][idx]
    if flip_y:
        meshes = {k: _flip_mesh_y(v, params.height_mm) for k, v in meshes.items()}
    return meshes, dE, gamut, reached
