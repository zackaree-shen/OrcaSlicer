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
    "C": (0.5, 3.0, 3.0),   # cyan absorbs red
    "M": (3.0, 0.5, 3.0),   # magenta absorbs green
    "Y": (3.0, 3.0, 0.5),   # yellow absorbs blue
    "W": (5.4, 5.4, 5.4),   # white base, product-doc measured
}


def correct_td(td, c=1.0, m=1.0, y=1.0, w=1.0):
    """Return a TD dict with per-color effective density scaled.

    A strength > 1 makes the color "act stronger" in the Beer-Lambert model,
    so the solver uses *less* of it to reach the same target. This is the
    printable equivalent of an ICC profile correction: it compensates for
    filament batches that are denser or weaker than the default TD assumes,
    without assuming anything about the input image's neutral balance.

    Values are clipped to a small positive floor to avoid division by zero.
    """
    def scale(t, s):
        return tuple(np.clip(np.asarray(t, dtype=np.float64) / max(float(s), 1e-6), 1e-6, None))
    return {
        "C": scale(td["C"], c),
        "M": scale(td["M"], m),
        "Y": scale(td["Y"], y),
        "W": scale(td["W"], w),
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


def forward_segstack(dTop, dC, dM, dY, td=None, dW=WHITE_THICKNESS, backlight=(1.0, 1.0, 1.0)):
    """OVERLAP forward model — segment-stack (segstack), order-dependent.

    The slicer resolves same-base overlapping C/M/Y parts by PART ORDER
    (clip_multipart_objects, PrintObjectSlice.cpp): with export order
    W,C,M,Y,top, the LAST part (Y) wins the overlap. The printed column per
    pixel is therefore a *segment stack* in part order:
        Y occupies [0, dY)
        M occupies [dY, max(dY, dM))   (only the part above Y)
        C occupies [max(dY,dM), max(dY,dM,dC))
    and the transmitted color is the product of Beer-Lambert transmissions of
    each segment. This is what a same-base overlapping geometry ACTUALLY
    prints (verified by reading the slicer source; the naive 'max column wins'
    model is wrong — the winner is the last part, not the tallest).

    Part order is fixed to W,C,M,Y,top (our export order). To compare
    algorithms under different orders, permute dC/dM/dY before calling.
    """
    if td is None:
        td = DEFAULT_TD
    dC = np.asarray(dC, dtype=np.float64)[..., None]
    dM = np.asarray(dM, dtype=np.float64)[..., None]
    dY = np.asarray(dY, dtype=np.float64)[..., None]
    dTop = np.asarray(dTop, dtype=np.float64)[..., None]
    tdc = np.asarray(td["C"], dtype=np.float64)
    tdm = np.asarray(td["M"], dtype=np.float64)
    tdy = np.asarray(td["Y"], dtype=np.float64)
    tdw = np.asarray(td["W"], dtype=np.float64)

    # Segment thicknesses in part order Y, M, C (bottom-up).
    dY_seg = dY
    dM_seg = np.maximum(dM - dY, 0.0)
    dC_seg = np.maximum(dC - np.maximum(dY, dM), 0.0)
    exponent = (dW / tdw + dTop / tdw
                + dY_seg / tdy + dM_seg / tdm + dC_seg / tdc)
    tau = 10.0 ** (-exponent)
    return np.asarray(backlight) * tau


def build_gamut_overlap(layers_max=8, layer_h=0.08, top_max=TOP_BAND_MAX, top_step=0.08,
                        dW=WHITE_THICKNESS, td=None):
    """OVERLAP color card (segstack model, order-dependent W,C,M,Y,top).

    Same enumeration as stacked but the forward model uses forward_segstack
    (part-order segment stack, matching what overlapping geometry actually
    prints). Returns the same dict shape as build_gamut_stacked so
    solve_stacked is reusable. Card is less degenerate than max but still
    order-fixed.
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
    rgb_lin = forward_segstack(dTop, dC, dM, dY, td=td, dW=dW)
    lab = xyz_to_lab(linear_to_xyz(rgb_lin))
    return {
        "lab": lab,
        "rgb_linear": rgb_lin,
        "rgb8": linear_to_srgb8(rgb_lin),
        "thickness": np.stack([dTop, dC, dM, dY], axis=-1),
    }


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


def preprocess_image(rgb, sharpen=0.5, contrast=1.3):
    """Denoise (clean) + edge-aware sharpen (sharp) + contrast on LUMINANCE.

    Pipeline: median+gaussian denoise -> edge-weighted unsharp mask ->
    contrast stretch. Hue-preserving. Applied BEFORE solving so the solver
    sees crisp, clean edges -> dTop relief has finer, noise-free detail.
    """
    if sharpen <= 0 and abs(contrast - 1.0) < 1e-6:
        return rgb
    from scipy.ndimage import gaussian_filter, median_filter, sobel
    rgb_f = rgb.astype(np.float64)
    Y = 0.299 * rgb_f[..., 0] + 0.587 * rgb_f[..., 1] + 0.114 * rgb_f[..., 2]
    # 1. Denoise: median (salt-pepper) + light gaussian (high-freq noise).
    Y = 0.5 * Y + 0.25 * median_filter(Y, size=3) + 0.25 * gaussian_filter(Y, sigma=0.5)
    # 2. Edge-aware sharpen: strong at edges (preserve contours), weak in flats.
    if sharpen > 0:
        Y_blur = gaussian_filter(Y, sigma=1.0)
        gx = np.abs(sobel(Y)); gy = np.abs(sobel(Y))
        edge_w = np.clip(np.hypot(gx, gy) / (np.percentile(np.hypot(gx, gy), 90) + 1e-6), 0, 1)
        Y = Y + sharpen * (Y - Y_blur) * (0.3 + 0.7 * edge_w)
    # 3. Contrast stretch.
    if abs(contrast - 1.0) > 1e-6:
        Y_mean = Y.mean()
        Y = Y_mean + contrast * (Y - Y_mean)
    Y = np.clip(Y, 0, 255)
    ratio = np.clip(Y / (0.299 * rgb_f[..., 0] + 0.587 * rgb_f[..., 1] +
                         0.114 * rgb_f[..., 2] + 1e-6), 0, 3)
    return np.clip(rgb_f * ratio[..., None], 0, 255).astype(np.uint8)


def tone_mapping_preprocess(rgb, td_w=5.4, d_w=0.5, top_max=1.5):
    """P1a: Beer-Lambert tone mapping (iteration 41 research).

    Linear luminance->thickness crushes shadows (measured: Y=0-64 -> only
    14 perceptual levels). Invert the physics instead: d ∝ -log10(luminance)
    so optical density is linear. Academic: Alexa & Matusik TOG 2010.
    """
    rgb_f = rgb.astype(np.float64) / 255.0
    lin = np.where(rgb_f <= 0.04045, rgb_f / 12.92, ((rgb_f + 0.055) / 1.055) ** 2.4)
    lum = np.clip(0.299 * lin[..., 0] + 0.587 * lin[..., 1] + 0.114 * lin[..., 2], 1e-4, 1.0)
    d_target = np.clip(-td_w * np.log10(lum) - d_w, 0, top_max)
    Y_tone = (1 - d_target / top_max) * 255.0
    Y_orig = 0.299 * rgb_f[..., 0] + 0.587 * rgb_f[..., 1] + 0.114 * rgb_f[..., 2]
    ratio = np.clip(Y_tone / 255.0 / (Y_orig + 1e-6), 0, 3)
    return (np.clip(rgb_f * ratio[..., None], 0, 1) * 255).astype(np.uint8)


def spike_surgery(dT, t_sigma=1.5, iterations=5, top_max=1.5):
    """P1b: local spike surgery (Kerber-inspired outlier gradient removal).

    Replaces outlier-gradient pixels (NN degeneracy spikes) with their 3x3
    neighborhood median. Local surgery preserves the global thickness
    distribution (unlike Poisson rebuilds which drifted dE to 2.9-9.2).
    The result is a CONTINUOUS field — geometry should use it directly
    (not the discrete card entries).
    """
    from scipy.ndimage import median_filter
    h = np.asarray(dT, dtype=np.float64).copy()
    for _ in range(iterations):
        hy, hx = np.gradient(h)
        gmag = np.hypot(hy, hx)
        outlier = gmag > t_sigma * gmag.std()
        if not outlier.any():
            break
        h = np.where(outlier, median_filter(h, size=3), h)
    return np.clip(h, 0, top_max)


def refine_dtop_surface(dTop, guide_rgb, n_iter=20, kappa=0.08, gamma=0.15,
                        edge_alpha=0.5, edge_percentile=95):
    """Edge-aware surface refinement for dTop.

    Goals:
      - Remove small spikes / high-frequency noise in flat regions, producing a
        smooth surface.
      - Preserve strong luminance edges so object boundaries and thin text stay
        crisp ("棱角封面").
      - Recover a controlled amount of the original detail at those edges so
        fine features (e.g. readable letters) are not blurred away.

    Uses Perona-Malik anisotropic diffusion with an image-guide edge map. The
    diffusion coefficient is derived from the *input image* luminance gradient,
    not from dTop itself, so noise in dTop is treated as noise and image edges
    are treated as structure. After diffusion a soft edge mask adds back some
    of the original dTop detail at the preserved edges only.

    Parameters
    ----------
    dTop : (H, W) ndarray
        Raw thickness relief field (e.g. output of anchored_dtop_field).
    guide_rgb : (H, W, 3) uint8
        Preprocessed RGB image used as the structure guide.
    n_iter, kappa, gamma : Perona-Malik controls.
        gamma must be <= 0.25 for explicit-scheme stability.
    edge_alpha : amount of original detail added back at strong edges.
        0 = fully smoothed, 1 = original detail fully preserved at edges.
    edge_percentile : percentile used to normalize the guide edge magnitude.
    """
    guide = np.asarray(guide_rgb, dtype=np.float64)
    if guide.ndim == 3:
        Y = (0.299 * guide[..., 0] + 0.587 * guide[..., 1]
             + 0.114 * guide[..., 2]) / 255.0
    else:
        Y = guide / 255.0

    # Edge magnitude from guide luminance, normalized to [0, 1].
    gy, gx = np.gradient(Y)
    edge_mag = np.hypot(gy, gx)
    denom = np.percentile(edge_mag, edge_percentile) + 1e-9
    edge_mag = np.clip(edge_mag / denom, 0.0, 1.0)

    # Edge-stopping coefficient: ~1 in flat regions (smooth freely),
    # ~0 at strong edges (do not blur across them).
    c = np.exp(-(edge_mag / kappa) ** 2)

    img = np.asarray(dTop, dtype=np.float64).copy()
    top_max = float(img.max())
    for _ in range(n_iter):
        dn = np.zeros_like(img)
        ds = np.zeros_like(img)
        de = np.zeros_like(img)
        dw = np.zeros_like(img)
        dn[:-1, :] = img[1:, :] - img[:-1, :]
        ds[1:, :] = img[:-1, :] - img[1:, :]
        de[:, :-1] = img[:, 1:] - img[:, :-1]
        dw[:, 1:] = img[:, :-1] - img[:, 1:]
        img += gamma * c * (dn + ds + de + dw)
        img = np.clip(img, 0.0, top_max)

    # Soft edge mask: high at strong edges, low in flat regions.
    edge_mask = 1.0 - np.exp(-(edge_mag / 0.30) ** 2)
    out = img + edge_alpha * edge_mask * (dTop - img)
    return np.clip(out, 0.0, top_max)


def morph_smooth(dTop, radius=1, iterations=1):
    """Morphological open-close to suppress tiny spikes and merge fragments.

    Uses scipy.ndimage.grey_opening then grey_closing with a square element of
    size ``2*radius+1``. A radius of 1 (3x3) removes/isolates single-pixel
    spikes and merges single-pixel gaps while keeping larger features intact.
    This is meant as a post-process *after* edge-aware diffusion: it cleans the
    remaining small fragmented bumps that diffusion left behind.

    Parameters
    ----------
    dTop : (H, W) ndarray
        Relief thickness field in mm.
    radius : int
        Structuring-element half-size; 0 disables smoothing.
    iterations : int
        Number of open-close passes.

    Returns
    -------
    ndarray
        Smoothed dTop, clipped to the original [min, max] range.
    """
    if radius <= 0 or iterations <= 0:
        return dTop
    from scipy.ndimage import grey_opening, grey_closing
    h = np.asarray(dTop, dtype=np.float64)
    lo, hi = float(h.min()), float(h.max())
    size = 2 * int(radius) + 1
    for _ in range(int(iterations)):
        h = grey_opening(h, size=(size, size))
        h = grey_closing(h, size=(size, size))
    return np.clip(h, lo, hi)


def anchored_dtop_field(rgb, td_w=5.4, top_max=2.0, p_low=0.5, p_high=99.5):
    """P1a-v2: monotone luminance->dTop field via optical-density-domain CDF
    equalization (iteration 45; supersedes tone_mapping_preprocess).

    v1 (tone_mapping_preprocess) did ABSOLUTE Beer-Lambert inversion
    (d = -td_w*log10(L) - d_w, clipped to [0, top_max]). Measured on a dark
    cartoon where 79.6% of pixels sit below the white window
    (tau < 10^(-(dW+top_max)/td_w) = 0.391): the clip crushed 84.4% of pixels
    to dTop == top_max -> giant flat plateaus ("details lost" report), and
    the ratio-encoded re-map could not represent the fix either (3x cap).

    v2 maps the pixel's RANK in the optical-density domain (-td_w*log10(L))
    to thickness: every printable layer (top_max/layer_h = 11 at 0.2mm)
    carries an equal pixel count — measured layer histogram 81.4% on ONE
    layer -> [5, 10, ..., 10, 5]% perfectly uniform; saturation 84.4%->0.5%.
    Histogram equalization over the density domain is the standard
    lithophane/bas-relief practice to maximize usable layers (ItsLithy-style
    equalization; adaptive range mapping per Hufstedler 2020, image dynamic
    range -> printable window).

    Mid-rank (ties share one rank): a large constant background must map to
    ONE dTop value — naive argsort().argsort() ranks would paint a linear
    gradient INSIDE the plateau (adversarial self-check, iteration 45).

    The -d_w offset cancels in ranking, so it is intentionally absent
    (iteration 45 adversarial review, m1).

    Guards (adversarial review B4):
      - L clamped to [1e-4, 1] before log10 (pure-black pixels would give
        log10(0) = -inf).
      - narrow-dynamic images (unique-value span < 1e-3) return a constant
        mid-height field (rank is uniform anyway; explicit guard keeps the
        contract obvious and NaN-free).
    """
    lin = srgb8_to_linear(rgb if getattr(rgb, "dtype", None) == np.uint8
                          else np.asarray(rgb, dtype=np.uint8))
    L = np.clip(lin.mean(axis=-1), 1e-4, 1.0)
    d_od = -td_w * np.log10(L)
    vals, inv, counts = np.unique(d_od, return_inverse=True, return_counts=True)
    if len(vals) < 2 or vals[-1] - vals[0] < 1e-3:
        return np.full(L.shape, top_max / 2.0)
    cdf = np.cumsum(counts) - 0.5 * counts          # mid-rank per unique value
    rank = cdf[inv] / d_od.size                     # (0, 1), ties share rank
    return rank.reshape(L.shape) * top_max


def resolve_cmy_for_dtop(flat_lab, gamut, dTop, top_tol=0.10, k=64, chunk=8192):
    """Re-solve (dC,dM,dY) with dTop pinned to an externally-given field.

    Iteration 45 v2: dTop comes from anchored_dtop_field (monotone), NOT from
    the NN solver. This removes the CMY-lattice sawtooth measured on a gray
    ramp: a 0.2mm CMY step = 0.53 decades of neutral density, but dTop can
    only compensate 0.407 decades inside one lattice cell, so solver-chosen
    dTop snapped back to top_max across cells (non-monotone, 3 large jumps).

    Band-constrained pick among the k nearest-Lab card neighbors: entries
    with |card_dTop - dTop| <= top_tol first (lowest dE wins), band-nearest
    fallback when empty (same guard as _smooth_top_resolve).
    top_tol=0.10 = 1.25x card top_step 0.08 (build_gamut_stacked default);
    continuous dTop values land at most half a step (0.04) from a card entry,
    so 0.10 admits the two nearest entries without re-admitting degeneracy.

    Returns (dC, dM, dY, dE, idx) each shaped like dTop.
    """
    from scipy.spatial import cKDTree
    dTop = np.asarray(dTop, dtype=np.float64)
    h, w = dTop.shape
    gamut_t = gamut["thickness"]
    gamut_lab = gamut["lab"]
    tree = cKDTree(gamut_lab)
    kk = min(k, len(gamut_lab))
    n = flat_lab.shape[0]
    dTop_f = dTop.ravel()
    res_t = np.empty((n, 4)); res_dE = np.empty(n); res_idx = np.empty(n, dtype=np.int64)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        _, nbrs = tree.query(flat_lab[s:e], k=kk)
        nbrs = np.atleast_2d(nbrs)
        kk2 = nbrs.shape[1]
        tiled = np.repeat(flat_lab[s:e], kk2, axis=0)
        cand = gamut_lab[nbrs.reshape(-1)]
        dist = _dE2000_pair(tiled, cand).reshape(e - s, kk2)
        dtop_cand = gamut_t[nbrs][..., 0]
        dtop_target = dTop_f[s:e, None]
        in_band = np.abs(dtop_cand - dtop_target) <= top_tol
        n_in = in_band.sum(axis=1)
        dist_band = np.where(in_band, dist, 1e9)
        empty = n_in == 0
        if empty.any():
            dtop_dist = np.abs(dtop_cand - dtop_target)
            fb = np.argmin(dtop_dist, axis=1)
            for r in np.where(empty)[0]:
                dist_band[r, fb[r]] = dist[r, fb[r]]
        best = np.argmin(dist_band, axis=1)
        rows = np.arange(e - s)
        res_t[s:e] = gamut_t[nbrs[rows, best]]
        res_dE[s:e] = dist_band[rows, best]
        res_idx[s:e] = nbrs[rows, best]
    return (res_t[:, 1].reshape(h, w), res_t[:, 2].reshape(h, w),
            res_t[:, 3].reshape(h, w), res_dE.reshape(h, w), res_idx.reshape(h, w))


def solve_stacked(target_srgb, gamut, chunk=4096, k=32, exact=False,
                  smooth_top=False, top_tol=0.5):
    """Inverse problem over the 5-layer stack.

    Maps each target sRGB pixel to the nearest reachable stacked color
    (CIEDE2000 nearest neighbor; gamut mapping for unreachable targets).
    Returns (dTop, dC, dM, dY, dE, idx) each (H, W).

    exact=True forces full enumeration over the whole card, which is only
    feasible for small grids: n_pixels * n_gamut must be < ~2e8, otherwise the
    call degrades to k=256 refine (still far more accurate than k=32) rather
    than hanging on a multi-billion-element distance matrix.

    smooth_top=True removes most of the "spike" artifact on the white relief
    top: the per-pixel nearest-neighbor maps nearly-identical pixels to
    different card entries (the (dTop,dC,dM,dY)->color map is degenerate), so
    dTop flip-flops by up to ~2.0mm between adjacent pixels. The pass
    degeneracy-weights a Gaussian-smooth of dTop and re-solves (dC,dM,dY) with
    dTop held within +/-top_tol of the blended value.

    top_tol is the BALANCED trade-off (iteration 27 adversarial verdict):
      - 0.08 (old default): dE explodes (photo 0.09 -> 1.86) because the
        smoothing excludes the true optimum; this was the real reason the
        anti-spike path looked broken.
      - 0.5 (default): dE is preserved EXACTLY (photo 0.088 -> 0.092), content
        is kept (siglap 0.36 -> 0.39, the raw-NN "detail" was ~82% noise), and
        spikes drop ~40% (max 2.0 -> 1.34, 20% cliff edges gone). Residual
        grad p95 ~0.44-0.73 is the honest "light spikes" trade for full
        detail+color; p95<0.2 is structurally unreachable without doubling dE.
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
            dTop = dTop.reshape(h, w); dC = dC.reshape(h, w)
            dM = dM.reshape(h, w); dY = dY.reshape(h, w)
            dE = dE.reshape(h, w); idx = idx.reshape(h, w)
            if smooth_top:
                _smooth_top_resolve(dTop, dC, dM, dY, dE, idx, flat, gamut,
                                    top_tol=top_tol, k=256)
            return (dTop, dC, dM, dY, dE, idx)
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
        dTop = dTop.reshape(h, w); dC = dC.reshape(h, w)
        dM = dM.reshape(h, w); dY = dY.reshape(h, w)
        dE = dE.reshape(h, w); idx = idx.reshape(h, w)
        if smooth_top:
            _smooth_top_resolve(dTop, dC, dM, dY, dE, idx, flat, gamut,
                                top_tol=top_tol, k=k)
        return (dTop, dC, dM, dY, dE, idx)
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


def _guided_filter(p, I, r=8, eps=0.05):
    """Guided filter (He et al. 2010) — edge-preserving smoothing.

    Uses guide image I (luminance) to distinguish real edges (high local
    variance -> keep dTop) from noise (low variance -> smooth).
    """
    from scipy.ndimage import uniform_filter
    def _box(x, rad):
        return uniform_filter(x, size=2 * rad + 1, mode='reflect')
    mean_I = _box(I, r); mean_p = _box(p, r)
    cov_Ip = _box(I * p, r) - mean_I * mean_p
    var_I = _box(I * I, r) - mean_I * mean_I
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    return _box(a, r) * I + _box(b, r)


def _dtop_anisotropic_diffusion(dT, n_iter=15, kappa=0.3, gamma=0.1):
    """Perona-Malik anisotropic diffusion ON dTop directly.

    Removes NN degeneracy flip-flop noise in flat regions (clean) while
    preserving real thickness jumps at edges (sharp). The noise is in the
    SOLVER OUTPUT (dTop), not the input image, so image-space denoising
    doesn't help — must denoise dTop.
    """
    img = dT.copy()
    for _ in range(n_iter):
        dn = np.zeros_like(img); ds = np.zeros_like(img)
        de = np.zeros_like(img); dw = np.zeros_like(img)
        dn[:-1] = img[1:] - img[:-1]; ds[1:] = img[:-1] - img[1:]
        de[:, :-1] = img[:, 1:] - img[:, :-1]; dw[:, 1:] = img[:, :-1] - img[:, 1:]
        grad = np.sqrt(dn**2 + ds**2 + de**2 + dw**2)
        c = 1.0 / (1.0 + (grad / kappa)**2)
        img += gamma * (c*dn + c*ds + c*de + c*dw)
    return img


def _smooth_top_resolve(dTop, dC, dM, dY, dE, idx, flat_lab, gamut,
                        top_tol=0.08, k=64, smooth_sigma=1.0, guide_r=4, guide_eps=0.005):
    """EXPERIMENTAL post-solve spatial-consistency pass (white-relief anti-spike).

    NOTE (iteration 26): this approach is superseded by default-off. Gaussian
    smoothing of dTop removes the flip-flop spikes but ALSO destroys the fine
    relief detail (measured dTop Laplacian 0.41 -> 0.011 vs Bambu 0.275; raw
    detail 0.30 already matches Bambu). The spike is rooted in CARD DEGENERACY
    (a gray target has 3-17 near-equal-dE card candidates spanning up to 2.0mm
    of dTop; the NN picks arbitrarily), and dTop is the ONLY fine-resolution
    channel (CMY prints on the 0.8mm coarse grid) — so no post-hoc smoothing
    can remove spikes without removing detail. Keep for API callers who prefer
    smoothness over detail; the real fix belongs in the card/solver layer
    (Bambu-style monotone gray->dTop map + finer top_step).

    Steps:
      1. Degeneracy-weighted blend of dTop and its Gaussian-smooth (sigma=3):
         smooth hard in high-degeneracy regions (kill flip-flop), keep the
         forced detail in low-degeneracy regions.
      2. Re-solve (dC,dM,dY) with dTop held within +/-top_tol of the blended
         value, so color (a joint function of all four channels) survives.
    """
    from scipy.ndimage import gaussian_filter
    from scipy.spatial import cKDTree
    h, w = dTop.shape
    chunk = 8192
    gamut_t = gamut["thickness"]
    gamut_lab = gamut["lab"]
    n = flat_lab.shape[0]
    tree = cKDTree(gamut_lab)
    kk = min(k, len(gamut_lab))

    # Chroma-aware partitioned tolerance for the CMY re-solve below.
    chroma = np.sqrt(flat_lab[:, 1] ** 2 + flat_lab[:, 2] ** 2).reshape(h, w)
    tol_map = np.clip(0.1 + (chroma - 5.0) / 15.0 * (top_tol - 0.1), 0.1, top_tol)

    # Pipeline D (measured best 'clean + sharp'):
    # 1. Anisotropic diffusion ON dTop (15 iter) — kills degeneracy flip-flop
    #    noise in flat regions (clean) while preserving real edges (sharp).
    # 2. Guided filter (r=4, eps=0.005) — edge-preserving smoothing using
    #    luminance L* as guide; sharpens contour alignment with image.
    # 3. Second diffusion (10 iter) — final cleanup pass for residual noise.
    # Result: spike p95 0.429 -> 0.180 (-58%), detail lap 0.320 -> 0.252
    # (6x original Gaussian 0.042), dE unchanged.
    guide = flat_lab[:, 0].reshape(h, w) / 100.0
    dTop_d1 = _dtop_anisotropic_diffusion(dTop, n_iter=15, kappa=0.3, gamma=0.1)
    dTop_gf = _guided_filter(dTop_d1 / 2.0, guide, r=guide_r, eps=guide_eps) * 2.0
    dTop_s = _dtop_anisotropic_diffusion(dTop_gf, n_iter=10, kappa=0.3, gamma=0.08)
    # Re-solve (dC,dM,dY) with dTop in [dTop_s-tol, dTop_s+tol] (per-pixel tol).
    dTop_s_f = dTop_s.ravel()
    tol_f = tol_map.ravel()
    res_t = np.empty((n, 4)); res_dE = np.empty(n); res_idx = np.empty(n, dtype=np.int64)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        _, nbrs = tree.query(flat_lab[s:e], k=kk)
        nbrs = np.atleast_2d(nbrs)
        kk2 = nbrs.shape[1]
        tiled = np.repeat(flat_lab[s:e], kk2, axis=0)
        cand = gamut_lab[nbrs.reshape(-1)]
        dist = _dE2000_pair(tiled, cand).reshape(e - s, kk2)
        cand_t = gamut_t[nbrs]
        dtop_cand = cand_t[..., 0]
        dtop_target = dTop_s_f[s:e, None]
        tol_pix = tol_f[s:e, None]
        in_band = np.abs(dtop_cand - dtop_target) <= tol_pix
        # Guard: if NO candidate falls in the band (possible when the smoothed
        # dTop sits between card steps outside the top-k neighbors), fall back
        # to the band's nearest candidate instead of a 1e9 empty-argmin.
        n_in = in_band.sum(axis=1)
        dist_band = np.where(in_band, dist, 1e9)
        empty = n_in == 0
        if empty.any():
            # fallback: nearest dTop distance wins
            dtop_dist = np.abs(dtop_cand - dtop_target)
            fb = np.argmin(dtop_dist, axis=1)
            for r in np.where(empty)[0]:
                dist_band[r, fb[r]] = dist[r, fb[r]]
        best = np.argmin(dist_band, axis=1)
        rows = np.arange(e - s)
        sel = nbrs[rows, best]
        res_t[s:e] = cand_t[rows, best]
        res_dE[s:e] = dist_band[rows, best]
        res_idx[s:e] = sel
    # In-place update: keep smoothed dTop, take re-solved C/M/Y.
    dTop[:, :] = dTop_s
    dC[:, :] = res_t[:, 1].reshape(h, w)
    dM[:, :] = res_t[:, 2].reshape(h, w)
    dY[:, :] = res_t[:, 3].reshape(h, w)
    dE[:, :] = res_dE.reshape(h, w)
    idx[:, :] = res_idx.reshape(h, w)


def color_lithophane_stacked(rgb_image, params=None, td=None, layers_max=8, layer_h=0.08,
                             dW=WHITE_THICKNESS, top_max=TOP_BAND_MAX, exact=False,
                             pitch_cmy=0.30, pitch_top=0.15):
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
