"""Lithophane prototype - M2 color (CMYW stacked transmission).

Physics: backlight passes through W/C/M/Y semi-transparent PLA layers once.
For each pixel the output color is the subtractive transmission product:

    tau_c = 10^(-d_W/TD_Wc - d_C/TD_Cc - d_M/TD_Mc - d_Y/TD_Yc),  c in {R,G,B}

Each color is a thickness-map STL (solution A); the four STLs stack along Z.
The inverse problem maps every target RGB to the nearest reachable color of a
precomputed gamut card (CIEDE2000 full enumeration). Because the CMYW gamut is
physically bounded (saturation is limited by dye selectivity x thickness), the
solver performs gamut mapping: unreachable target colors are mapped to their
nearest reachable neighbor rather than being force-fit.

Experimentally validated facts (see litho_experiments/):
  - gamut reaches only ~1.9 decades of channel separation at 0.64 mm/color,
    so saturated sRGB primaries are structurally unreachable -> gamut mapping
    is mandatory, "photo-like but soft" is the honest expectation.
  - white base at 0.30 mm beats 0.45 (brighter, whiter, better coverage).
  - CIEDE2000 nearest neighbor beats Lab-Euclidean kd-tree (~1.2 dE median).
  - TD defaults are coarse; must be calibrated per filament batch.
"""

from __future__ import annotations

import itertools

import numpy as np

from litho_core import LithophaneParams, heightfield_to_mesh


# ---------------------------------------------------------------------------
# Color space conversions (sRGB -> linear -> XYZ -> Lab)
# ---------------------------------------------------------------------------

# sRGB (D65) to linear RGB.
_SRGB_LUT = None  # lazily built LUT for srgb_byte -> linear


def _srgb_to_linear_lut():
    global _SRGB_LUT
    if _SRGB_LUT is None:
        c = np.arange(256, dtype=np.float64) / 255.0
        lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
        _SRGB_LUT = lin
    return _SRGB_LUT


def srgb8_to_linear(srgb):
    """srgb: (...,3) uint8 or float in [0,255]. Returns linear RGB (...,3) 0..1."""
    s = np.asarray(srgb, dtype=np.float64) / 255.0
    return np.where(s <= 0.04045, s / 12.92, ((s + 0.055) / 1.055) ** 2.4)


def linear_to_srgb8(lin):
    """linear RGB (...,3) 0..1 -> sRGB uint8 (...,3)."""
    l = np.clip(lin, 0.0, 1.0)
    s = np.where(l <= 0.0031308, 12.92 * l, 1.055 * l ** (1 / 2.4) - 0.055)
    return (np.clip(s, 0, 1) * 255.0 + 0.5).astype(np.uint8)


# sRGB D65 -> XYZ matrix (CIE).
_SRGB_TO_XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
])
_XYZ_WHITE = np.array([0.95047, 1.0, 1.08883])  # D65


def linear_to_xyz(lin):
    """linear RGB (...,3) -> XYZ (...,3)."""
    return np.einsum("ij,...j->...i", _SRGB_TO_XYZ, np.asarray(lin, dtype=np.float64))


def xyz_to_lab(xyz):
    """XYZ (...,3) -> Lab (...,3)."""
    xyz = np.asarray(xyz, dtype=np.float64)
    f = xyz / _XYZ_WHITE
    f = np.where(f > 0.008856, f ** (1 / 3), 7.787 * f + 16 / 116)
    fx, fy, fz = f[..., 0], f[..., 1], f[..., 2]
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    return np.stack([L, a, b], axis=-1)


def srgb8_to_lab(srgb):
    return xyz_to_lab(linear_to_xyz(srgb8_to_linear(srgb)))


# ---------------------------------------------------------------------------
# CIEDE2000 (Sharma et al. 2005), vectorized pairwise
# ---------------------------------------------------------------------------

def _dE2000_pair(lab1, lab2):
    """dE2000 between two Lab points (each (3,) or (N,3)). Returns float or (N,)."""
    l1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    l2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]
    c1 = np.hypot(a1, b1)
    c2 = np.hypot(a2, b2)
    cbar = (c1 + c2) / 2.0
    cbar7 = cbar ** 7
    g = 0.5 * (1 - np.sqrt(cbar7 / (cbar7 + 25 ** 7)))
    a1p = (1 + g) * a1
    a2p = (1 + g) * a2
    c1p = np.hypot(a1p, b1)
    c2p = np.hypot(a2p, b2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360.0
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360.0

    dlp = l2 - l1
    dcp = c2p - c1p

    dhp = np.zeros_like(h1p)
    cprod = c1p * c2p
    m = (cprod > 0)
    d = h2p - h1p
    dhp[m] = d[m]
    dhp[m & (d > 180)] = d[m & (d > 180)] - 360
    dhp[m & (d < -180)] = d[m & (d < -180)] + 360
    dhp[~m] = 0.0

    dhp_rad = np.radians(dhp)
    dhp2 = 2 * np.sqrt(c1p * c2p) * np.sin(dhp_rad / 2)

    lbarp = (l1 + l2) / 2
    cbarp = (c1p + c2p) / 2

    hsum = h1p + h2p
    hdiff = h1p - h2p
    hbarp = np.zeros_like(h1p)
    m0 = cprod > 0
    m1 = m0 & (np.abs(hdiff) <= 180)
    m2 = m0 & ~m1 & (hsum < 360)
    m3 = m0 & ~m1 & ~m2
    hbarp[m1] = (h1p[m1] + h2p[m1]) / 2
    hbarp[m2] = (hsum[m2] + 360) / 2
    hbarp[m3] = (hsum[m3] - 360) / 2
    hbarp[~m0] = hsum[~m0]

    hbar_deg = hbarp
    t = (1
         - 0.17 * np.cos(np.radians(hbar_deg - 30))
         + 0.24 * np.cos(np.radians(2 * hbar_deg))
         + 0.32 * np.cos(np.radians(3 * hbar_deg + 6))
         - 0.20 * np.cos(np.radians(4 * hbar_deg - 63)))
    dtheta = 30 * np.exp(-((hbar_deg - 275) / 25) ** 2)
    cbarp7 = cbarp ** 7
    rc = 2 * np.sqrt(cbarp7 / (cbarp7 + 25 ** 7))
    sl = 1 + 0.015 * (lbarp - 50) ** 2 / np.sqrt(20 + (lbarp - 50) ** 2)
    sc = 1 + 0.045 * cbarp
    sh = 1 + 0.015 * cbarp * t
    rt = -np.sin(np.radians(2 * dtheta)) * rc

    return np.sqrt((dlp / sl) ** 2 + (dcp / sc) ** 2 + (dhp2 / sh) ** 2
                   + rt * (dcp / sc) * (dhp2 / sh))


def ciede2000_matrix(lab_targets, lab_gamut):
    """Pairwise dE2000 between (N,3) targets and (M,3) gamut. Returns (N,M)."""
    n = lab_targets.shape[0]
    m = lab_gamut.shape[0]
    # Expand to (N, M, 3) and vectorize.
    t = np.broadcast_to(lab_targets[:, None, :], (n, m, 3))
    g = np.broadcast_to(lab_gamut[None, :, :], (n, m, 3))
    l1, a1, b1 = t[..., 0], t[..., 1], t[..., 2]
    l2, a2, b2 = g[..., 0], g[..., 1], g[..., 2]
    c1 = np.hypot(a1, b1)
    c2 = np.hypot(a2, b2)
    cbar = (c1 + c2) / 2.0
    cbar7 = cbar ** 7
    gcoef = 0.5 * (1 - np.sqrt(cbar7 / (cbar7 + 25 ** 7)))
    a1p = (1 + gcoef) * a1
    a2p = (1 + gcoef) * a2
    c1p = np.hypot(a1p, b1)
    c2p = np.hypot(a2p, b2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360.0
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360.0

    dlp = l2 - l1
    dcp = c2p - c1p
    cprod = c1p * c2p
    d = h2p - h1p
    dhp = np.where(cprod > 0,
                   np.where(d > 180, d - 360, np.where(d < -180, d + 360, d)),
                   0.0)
    dhp_rad = np.radians(dhp)
    dhp2 = 2 * np.sqrt(c1p * c2p) * np.sin(dhp_rad / 2)

    lbarp = (l1 + l2) / 2
    cbarp = (c1p + c2p) / 2
    hsum = h1p + h2p
    hdiff = h1p - h2p
    m0 = cprod > 0
    m1 = m0 & (np.abs(hdiff) <= 180)
    m2 = m0 & ~m1 & (hsum < 360)
    m3 = m0 & ~m1 & ~m2
    hbarp = np.where(m1, (h1p + h2p) / 2,
             np.where(m2, (hsum + 360) / 2,
              np.where(m3, (hsum - 360) / 2, hsum)))

    t = (1
         - 0.17 * np.cos(np.radians(hbarp - 30))
         + 0.24 * np.cos(np.radians(2 * hbarp))
         + 0.32 * np.cos(np.radians(3 * hbarp + 6))
         - 0.20 * np.cos(np.radians(4 * hbarp - 63)))
    dtheta = 30 * np.exp(-((hbarp - 275) / 25) ** 2)
    cbarp7 = cbarp ** 7
    rc = 2 * np.sqrt(cbarp7 / (cbarp7 + 25 ** 7))
    sl = 1 + 0.015 * (lbarp - 50) ** 2 / np.sqrt(20 + (lbarp - 50) ** 2)
    sc = 1 + 0.045 * cbarp
    sh = 1 + 0.015 * cbarp * t
    rt = -np.sin(np.radians(2 * dtheta)) * rc

    return np.sqrt((dlp / sl) ** 2 + (dcp / sc) ** 2 + (dhp2 / sh) ** 2
                   + rt * (dcp / sc) * (dhp2 / sh))


# ---------------------------------------------------------------------------
# Transmission model
# ---------------------------------------------------------------------------

# Per-color TD per channel: (TD_R, TD_G, TD_B) in mm.
# Defaults are v1 coarse estimates (see module docstring for validation notes).
DEFAULT_TD = {
    "C": (0.3, 3.0, 3.0),   # cyan absorbs red
    "M": (3.0, 0.3, 3.0),   # magenta absorbs green
    "Y": (3.0, 3.0, 0.3),   # yellow absorbs blue
    "W": (5.4, 5.4, 5.4),   # white base, product-doc measured
}


def forward_transmission(dC, dM, dY, dW, td=None, backlight=(1.0, 1.0, 1.0)):
    """Simulate backlight transmission through the four layers.

    dC/dM/dY/dW: thickness in mm (scalar or array-like). The thickness arrays
    may be (N,) with N pixels; each is broadcast against the per-color 3-channel
    TD (TD_R, TD_G, TD_B) to yield (N, 3) linear RGB.
    """
    if td is None:
        td = DEFAULT_TD
    dC = np.asarray(dC, dtype=np.float64)[..., None]
    dM = np.asarray(dM, dtype=np.float64)[..., None]
    dY = np.asarray(dY, dtype=np.float64)[..., None]
    dW = np.asarray(dW, dtype=np.float64)[..., None]
    tdc = np.asarray(td["C"], dtype=np.float64)
    tdm = np.asarray(td["M"], dtype=np.float64)
    tdy = np.asarray(td["Y"], dtype=np.float64)
    tdw = np.asarray(td["W"], dtype=np.float64)
    tau = 10.0 ** (-(dW / tdw + dC / tdc + dM / tdm + dY / tdy))
    return np.asarray(backlight) * tau


# ---------------------------------------------------------------------------
# Gamut card + inverse problem (gamut mapping via CIEDE2000 nearest neighbor)
# ---------------------------------------------------------------------------

def build_gamut(layers_max=8, layer_h=0.08, dW=0.30, td=None):
    """Precompute the reachable color card.

    Enumerates (nC, nM, nY) in [0..layers_max]^3 with d = n * layer_h, plus the
    fixed white base dW. Returns dict with:
      lab:        (M, 3) Lab of each reachable color
      rgb_linear: (M, 3) linear RGB
      rgb8:       (M, 3) uint8 sRGB (for preview)
      thickness:  (M, 3) (dC, dM, dY) of each entry
    """
    if td is None:
        td = DEFAULT_TD
    combos = list(itertools.product(range(layers_max + 1), repeat=3))
    dC = np.array([c[0] for c in combos], dtype=np.float64) * layer_h
    dM = np.array([c[1] for c in combos], dtype=np.float64) * layer_h
    dY = np.array([c[2] for c in combos], dtype=np.float64) * layer_h
    rgb_lin = forward_transmission(dC, dM, dY, np.full_like(dC, dW), td=td)
    lab = xyz_to_lab(linear_to_xyz(rgb_lin))
    return {
        "lab": lab,
        "rgb_linear": rgb_lin,
        "rgb8": linear_to_srgb8(rgb_lin),
        "thickness": np.stack([dC, dM, dY], axis=-1),
    }


def solve_thicknesses(target_srgb, gamut, chunk=4096, k=64, exact=False):
    """Inverse problem: map each target sRGB pixel to nearest reachable color.

    Uses CIEDE2000 full enumeration over the gamut card (gamut mapping: colors
    outside the reachable gamut are snapped to their nearest neighbor).

    target_srgb: (H, W, 3) uint8. Returns:
      (dC, dM, dY): each (H, W) thickness maps in mm
      dE:           (H, W) best-match CIEDE2000
      reached_idx:  (H, W) gamut index chosen
    """
    target = srgb8_to_linear(target_srgb)
    lab = xyz_to_lab(linear_to_xyz(target))
    h, w, _ = target.shape
    flat = lab.reshape(-1, 3)
    n = flat.shape[0]
    gamut_lab = gamut["lab"]
    gamut_t = gamut["thickness"]
    m = gamut_lab.shape[0]

    dC = np.zeros(n)
    dM = np.zeros(n)
    dY = np.zeros(n)
    dE = np.zeros(n)
    idx = np.zeros(n, dtype=np.int64)

    # Fast path: Lab-Euclidean cKDTree top-k pre-filter, then exact CIEDE2000
    # on those k candidates. The Euclidean NN and the dE2000 NN agree ~40% of
    # the time, but the true dE2000 optimum is almost always within the top-k
    # Euclidean neighbors; k=16 keeps the dE2000 penalty negligible (verified
    # against full enumeration in litho_experiments/). This is ~50x faster than
    # full enumeration for typical grid sizes.
    try:
        from scipy.spatial import cKDTree
        if exact:
            raise ImportError  # force the exact fallback path below
        tree = cKDTree(gamut_lab)
        k = min(k, m)
        # (n, k) nearest Euclidean neighbors, batches to bound memory.
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            _, nbrs = tree.query(flat[start:end], k=k)
            nbrs = np.atleast_2d(nbrs)                     # (batch, k)
            batch = end - start
            tiled_target = np.repeat(flat[start:end], k, axis=0)        # (batch*k, 3)
            cand_gamut = gamut_lab[nbrs.reshape(-1)]                    # (batch*k, 3)
            # Rowwise (one-to-one) dE2000 between each target and ITS k neighbors.
            dist = _dE2000_pair(tiled_target, cand_gamut).reshape(batch, k)
            best = np.argmin(dist, axis=1)
            rows = np.arange(batch)
            idx_slice = nbrs[rows, best]
            dE[start:end] = dist[rows, best]
            dC[start:end] = gamut_t[idx_slice, 0]
            dM[start:end] = gamut_t[idx_slice, 1]
            dY[start:end] = gamut_t[idx_slice, 2]
            idx[start:end] = idx_slice
    except ImportError:
        # Fallback: exact full enumeration (small grids).
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            dist = ciede2000_matrix(flat[start:end], gamut_lab)  # (chunk, M)
            best = np.argmin(dist, axis=1)
            dE[start:end] = dist[np.arange(end - start), best]
            dC[start:end] = gamut_t[best, 0]
            dM[start:end] = gamut_t[best, 1]
            dY[start:end] = gamut_t[best, 2]
            idx[start:end] = best

    return (dC.reshape(h, w), dM.reshape(h, w), dY.reshape(h, w),
            dE.reshape(h, w), idx.reshape(h, w))


# ---------------------------------------------------------------------------
# Full color lithophane pipeline
# ---------------------------------------------------------------------------

def color_lithophane_meshes(rgb_image, params=None, td=None, layers_max=8, layer_h=0.08, dW=0.30, exact=False):
    """Generate the 4 color-thickness meshes (W/C/M/Y) for a color image.

    rgb_image: (H, W, 3) uint8 sRGB. Returns:
      meshes: dict {color: (vertices, faces)}
      dE_map: (gy, gx) best-match CIEDE2000 per grid point
      gamut:  the reachable color card (for stats/preview)
      reached_rgb: (gy, gx, 3) uint8 — the actual printed appearance (WYSIWYG)
    """
    if params is None:
        params = LithophaneParams()
    if td is None:
        td = DEFAULT_TD

    # Solve at the output grid resolution: down-sample the image first so the
    # full CIEDE2000 enumeration runs on grid points only (fast even for large
    # source images). Bilinear down-sample of the RGB image.
    from litho_core import thickness_grid_shape
    gx, gy = thickness_grid_shape(rgb_image.shape[0], rgb_image.shape[1], params)
    small = _resample_rgb(rgb_image, (gy, gx))

    gamut = build_gamut(layers_max=layers_max, layer_h=layer_h, dW=dW, td=td)
    dC, dM, dY, dE, idx = solve_thicknesses(small, gamut, exact=exact)

    tW = np.full((gy, gx), dW)
    # A pixel with 0 layers of a color means that layer is absent there; the
    # mesh would then have coincident front/back vertices at z=0, producing
    # zero-area side-wall triangles. A floor of 1e-3 mm (well below any real
    # layer height) keeps the solid watertight without affecting the print.
    MIN_THICKNESS = 1e-3
    tC = np.maximum(dC, MIN_THICKNESS)
    tM = np.maximum(dM, MIN_THICKNESS)
    tY = np.maximum(dY, MIN_THICKNESS)

    meshes = {}
    for color, hmap in [("W", tW), ("C", tC), ("M", tM), ("Y", tY)]:
        meshes[color] = heightfield_to_mesh(hmap, params)

    # WYSIWYG: the appearance actually achieved by the printed stack.
    reached_rgb = gamut["rgb8"][idx]
    return meshes, dE, gamut, reached_rgb


def _resample_rgb(src, shape):
    """Bilinear resize of an (H,W,3) uint8 image to (gy,gx,3) uint8."""
    h, w = src.shape[0], src.shape[1]
    gy, gx = shape
    ys = np.linspace(0, h - 1, gy)
    xs = np.linspace(0, w - 1, gx)
    y0 = np.floor(ys).astype(int)
    y1 = np.minimum(y0 + 1, h - 1)
    x0 = np.floor(xs).astype(int)
    x1 = np.minimum(x0 + 1, w - 1)
    fy = (ys - y0)[:, None]
    fx = (xs - x0)[None, :]
    v00 = src[np.ix_(y0, x0)].astype(np.float64)
    v10 = src[np.ix_(y0, x1)].astype(np.float64)
    v01 = src[np.ix_(y1, x0)].astype(np.float64)
    v11 = src[np.ix_(y1, x1)].astype(np.float64)
    top = v00 * (1 - fx)[..., None] + v10 * fx[..., None]
    bot = v01 * (1 - fx)[..., None] + v11 * fx[..., None]
    out = top * (1 - fy)[..., None] + bot * fy[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def _resample(src, shape):
    """Naive bilinear resize of a (H,W) float array to (gy,gx)."""
    h, w = src.shape
    gy, gx = shape
    ys = (np.linspace(0, h - 1, gy))
    xs = (np.linspace(0, w - 1, gx))
    y0 = np.floor(ys).astype(int)
    y1 = np.minimum(y0 + 1, h - 1)
    x0 = np.floor(xs).astype(int)
    x1 = np.minimum(x0 + 1, w - 1)
    fy = (ys - y0)[:, None]
    fx = (xs - x0)[None, :]
    v00 = src[np.ix_(y0, x0)]
    v10 = src[np.ix_(y0, x1)]
    v01 = src[np.ix_(y1, x0)]
    v11 = src[np.ix_(y1, x1)]
    top = v00 * (1 - fx) + v10 * fx
    bot = v01 * (1 - fx) + v11 * fx
    return top * (1 - fy) + bot * fy


# ===========================================================================
# Stacked 5-layer lithophane (strict Z separation, Bambu-style structure)
#
# Z layout (each layer occupies its own Z band, so every slice plane cuts
# exactly one color -> one color per layer during slicing):
#
#   z top
#   ┌─────────────────────┐
#   │ top_white (brightness) │ [TOP_BASE, TOP_BASE + dTop(x,y)]
#   ├─────────────────────┤
#   │ yellow              │ [Y_BASE, Y_BASE + dY(x,y)]
#   ├─────────────────────┤
#   │ magenta             │ [M_BASE, M_BASE + dM(x,y)]
#   ├─────────────────────┤
#   │ cyan                │ [C_BASE, C_BASE + dC(x,y)]
#   ├─────────────────────┤
#   │ white base (fixed)  │ [0, WHITE_THICKNESS]
#   └─────────────────────┘
#
# The top white layer is the brightness relief (like a classic greyscale
# lithophane); C/M/Y modulate hue. Backlight traverses all five; the total
# transmission is the product of per-layer Beer-Lambert transmissions.
# ===========================================================================

# Z band parameters (mm). Bands are strictly separated so each slice plane
# intersects exactly one layer -> one color per printed layer.
WHITE_THICKNESS = 0.8     # fixed white base slab (Bambu Min Thickness)
COLOR_BAND_MAX  = 0.64    # max thickness per C/M/Y layer (8 x 0.08 mm)
TOP_BAND_MAX    = 2.0     # max top-white brightness relief

# Inter-layer gap. If adjacent slabs share an exact plane (e.g. white top
# surface at z=0.8 and cyan bottom surface at z=0.8), a slice plane landing on
# that z cuts both layers producing two identical overlapping polygons, which
# the slicer's prepare_infill() stage (25% progress) then repeatedly boolean-
# subtracts - a geometric pathology that looks like a hang. A small gap keeps
# every slice plane inside exactly one layer. 0.05 mm is well below the first
# layer height (0.15) so the printed result is unaffected.
LAYER_GAP = 0.05

Z_C_BASE   = WHITE_THICKNESS + LAYER_GAP                    # 0.85
Z_M_BASE   = Z_C_BASE + COLOR_BAND_MAX + LAYER_GAP          # 1.54
Z_Y_BASE   = Z_M_BASE + COLOR_BAND_MAX + LAYER_GAP          # 2.23
Z_TOP_BASE = Z_Y_BASE + COLOR_BAND_MAX + LAYER_GAP          # 2.92


def forward_stacked(dTop, dC, dM, dY, td=None, dW=WHITE_THICKNESS, backlight=(1.0, 1.0, 1.0)):
    """Backlight transmission through the 5-layer stack (W base + C/M/Y + top W).

    Thicknesses may be scalar or (N,); broadcast per-channel against TD.
    """
    if td is None:
        td = DEFAULT_TD
    dTop = np.asarray(dTop, dtype=np.float64)[..., None]
    dC = np.asarray(dC, dtype=np.float64)[..., None]
    dM = np.asarray(dM, dtype=np.float64)[..., None]
    dY = np.asarray(dY, dtype=np.float64)[..., None]
    dW = np.asarray(dW, dtype=np.float64)[..., None]
    tdc = np.asarray(td["C"], dtype=np.float64)
    tdm = np.asarray(td["M"], dtype=np.float64)
    tdy = np.asarray(td["Y"], dtype=np.float64)
    tdw = np.asarray(td["W"], dtype=np.float64)
    tau = 10.0 ** (-(dW / tdw + dC / tdc + dM / tdm + dY / tdy + dTop / tdw))
    return np.asarray(backlight) * tau


def build_gamut_stacked(layers_max=8, layer_h=0.08, top_max=TOP_BAND_MAX, top_step=0.08,
                        dW=WHITE_THICKNESS, td=None):
    """Precompute the reachable 5-layer color card.

    Enumerates (nC, nM, nY) in [0..layers_max]^3 and nTop in
    [0..round(top_max/top_step)]. Thickness = n * layer_h. Returns dict with:
      lab:        (M, 3) Lab of each reachable color
      rgb_linear: (M, 3) linear RGB
      rgb8:       (M, 3) uint8 sRGB
      thickness:  (M, 4) (dTop, dC, dM, dY)

    top_step is 0.08 to keep the top-white relief smooth (coarser steps turn
    the brightness relief into visible stair-step cliffs that slice into
    jagged artifacts). This makes the card ~19k entries; the fast kd-tree path
    keeps solving fast.
    """
    if td is None:
        td = DEFAULT_TD
    n_top = int(round(top_max / top_step))
    combos = list(itertools.product(range(n_top + 1), range(layers_max + 1),
                                    range(layers_max + 1), range(layers_max + 1)))
    dTop = np.array([c[0] for c in combos], dtype=np.float64) * top_step
    dC = np.array([c[1] for c in combos], dtype=np.float64) * layer_h
    dM = np.array([c[2] for c in combos], dtype=np.float64) * layer_h
    dY = np.array([c[3] for c in combos], dtype=np.float64) * layer_h
    rgb_lin = forward_stacked(dTop, dC, dM, dY, td=td, dW=dW)
    lab = xyz_to_lab(linear_to_xyz(rgb_lin))
    return {
        "lab": lab,
        "rgb_linear": rgb_lin,
        "rgb8": linear_to_srgb8(rgb_lin),
        "thickness": np.stack([dTop, dC, dM, dY], axis=-1),
    }


def solve_stacked(target_srgb, gamut, chunk=4096, k=32, exact=False):
    """Inverse problem over the 5-layer stack.

    Maps each target sRGB pixel to the nearest reachable stacked color
    (CIEDE2000 nearest neighbor; gamut mapping for unreachable targets).
    Returns (dTop, dC, dM, dY, dE, idx) each (H, W).

    exact=True forces full enumeration over the whole card, which is only
    feasible for small grids: n_pixels * n_gamut must be < ~2e8, otherwise the
    call degrades to k=256 refine (still far more accurate than k=32) rather
    than hanging on a multi-billion-element distance matrix.
    """
    lab = xyz_to_lab(linear_to_xyz(srgb8_to_linear(target_srgb)))
    h, w, _ = target_srgb.shape
    flat = lab.reshape(-1, 3)
    n = flat.shape[0]
    gamut_lab = gamut["lab"]
    gamut_t = gamut["thickness"]
    m = gamut_lab.shape[0]

    dTop = np.zeros(n)
    dC = np.zeros(n)
    dM = np.zeros(n)
    dY = np.zeros(n)
    dE = np.zeros(n)
    idx = np.zeros(n, dtype=np.int64)

    try:
        from scipy.spatial import cKDTree
        # Guard against pathological full-enumeration matrices: exact mode only
        # runs the O(n*m) dE2000 matrix when it fits (~2e8 elements ≈ 1.6 GB
        # float64). For larger grids fall back to a generous k=256 refine, which
        # is still far more accurate than the fast k=32 path.
        if exact and n * m <= 200_000_000:
            for start in range(0, n, chunk):
                end = min(start + chunk, n)
                dist = ciede2000_matrix(flat[start:end], gamut_lab)
                best = np.argmin(dist, axis=1)
                dE[start:end] = dist[np.arange(end - start), best]
                dTop[start:end] = gamut_t[best, 0]
                dC[start:end] = gamut_t[best, 1]
                dM[start:end] = gamut_t[best, 2]
                dY[start:end] = gamut_t[best, 3]
                idx[start:end] = best
            return (dTop.reshape(h, w), dC.reshape(h, w), dM.reshape(h, w), dY.reshape(h, w),
                    dE.reshape(h, w), idx.reshape(h, w))
        tree = cKDTree(gamut_lab)
        k = min(k if not exact else 256, m)
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            _, nbrs = tree.query(flat[start:end], k=k)
            nbrs = np.atleast_2d(nbrs)
            batch = end - start
            tiled = np.repeat(flat[start:end], k, axis=0)
            cand = gamut_lab[nbrs.reshape(-1)]
            dist = _dE2000_pair(tiled, cand).reshape(batch, k)
            best = np.argmin(dist, axis=1)
            rows = np.arange(batch)
            sel = nbrs[rows, best]
            dE[start:end] = dist[rows, best]
            dTop[start:end] = gamut_t[sel, 0]
            dC[start:end] = gamut_t[sel, 1]
            dM[start:end] = gamut_t[sel, 2]
            dY[start:end] = gamut_t[sel, 3]
            idx[start:end] = sel
    except ImportError:
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            dist = ciede2000_matrix(flat[start:end], gamut_lab)
            best = np.argmin(dist, axis=1)
            dE[start:end] = dist[np.arange(end - start), best]
            dTop[start:end] = gamut_t[best, 0]
            dC[start:end] = gamut_t[best, 1]
            dM[start:end] = gamut_t[best, 2]
            dY[start:end] = gamut_t[best, 3]
            idx[start:end] = best

    return (dTop.reshape(h, w), dC.reshape(h, w), dM.reshape(h, w), dY.reshape(h, w),
            dE.reshape(h, w), idx.reshape(h, w))


def color_lithophane_stacked(rgb_image, params=None, td=None, layers_max=8, layer_h=0.08,
                             dW=WHITE_THICKNESS, top_max=TOP_BAND_MAX, exact=False,
                             pitch_cmy=0.8, pitch_top=0.25):
    """Generate the 5 stacked-layer meshes (W / C / M / Y / top) for a color image.

    Dual-resolution (Bambu-style): C/M/Y use a COARSE grid (pitch_cmy) because
    they only carry hue and have a small number of steps; top uses a FINE grid
    (pitch_top) because it carries the brightness detail. This keeps total
    geometry small enough for the slicer's prepare_infill() while preserving
    the top relief detail.

    Strict Z separation: each layer occupies its own Z band, so any slice plane
    crosses exactly one layer -> one color per printed layer.

    Returns:
      meshes: dict {color: (vertices, faces)}
      dE_map: (gy, gx) best-match CIEDE2000
      gamut:  reachable color card
      reached_rgb: (gy, gx, 3) uint8 WYSIWYG appearance
    """
    if params is None:
        params = LithophaneParams()
    if td is None:
        td = DEFAULT_TD

    MIN_THICKNESS = 1e-3  # floor to keep watertightness (0-layer pixels)

    from litho_core import thickness_grid_shape
    gx, gy = thickness_grid_shape(rgb_image.shape[0], rgb_image.shape[1], params)
    small = _resample_rgb(rgb_image, (gy, gx))

    gamut = build_gamut_stacked(layers_max=layers_max, layer_h=layer_h,
                                top_max=top_max, dW=dW, td=td)
    dTop, dC, dM, dY, dE, idx = solve_stacked(small, gamut, exact=exact)

    # ---- Dual-resolution geometry ----
    # C/M/Y: coarse grid. W: coarse too (it's a flat slab). top: fine grid.
    params_cmy = LithophaneParams(width_mm=params.width_mm, height_mm=params.height_mm,
                                  pixel_pitch_mm=pitch_cmy, base_thickness=params.base_thickness,
                                  depth_range=params.depth_range)
    params_top = LithophaneParams(width_mm=params.width_mm, height_mm=params.height_mm,
                                  pixel_pitch_mm=pitch_top, base_thickness=params.base_thickness,
                                  depth_range=params.depth_range)

    # Resample thickness maps to each layer's grid.
    gx_c, gy_c = thickness_grid_shape(rgb_image.shape[0], rgb_image.shape[1], params_cmy)
    gx_t, gy_t = thickness_grid_shape(rgb_image.shape[0], rgb_image.shape[1], params_top)

    # W (flat base): coarse grid, constant dW.
    tW = np.full((gy_c, gx_c), dW)
    # C/M/Y: coarse grid thickness maps.
    tC = np.maximum(_resample(dC, (gy_c, gx_c)), MIN_THICKNESS)
    tM = np.maximum(_resample(dM, (gy_c, gx_c)), MIN_THICKNESS)
    tY = np.maximum(_resample(dY, (gy_c, gx_c)), MIN_THICKNESS)
    # top: fine grid brightness relief (continuous heights).
    tTop = np.maximum(_resample(dTop, (gy_t, gx_t)), MIN_THICKNESS)

    meshes = {
        "W":   heightfield_to_mesh(tW,   params_cmy, z_offset=0.0),
        "C":   heightfield_to_mesh(tC,   params_cmy, z_offset=Z_C_BASE),
        "M":   heightfield_to_mesh(tM,   params_cmy, z_offset=Z_M_BASE),
        "Y":   heightfield_to_mesh(tY,   params_cmy, z_offset=Z_Y_BASE),
        "top": heightfield_to_mesh(tTop, params_top, z_offset=Z_TOP_BASE),
    }
    reached_rgb = gamut["rgb8"][idx]
    return meshes, dE, gamut, reached_rgb
