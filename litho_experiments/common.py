"""Common model functions for CMYW lithophane numerical experiments.

Model (as stated in the proposed scheme):
  tau_c = 10^(-d_W/TD_Wc - d_C/TD_Cc - d_M/TD_Mc - d_Y/TD_Yc),  c in {R,G,B}
  output linear RGB = backlight(white) x (tau_R, tau_G, tau_B)
"""
import numpy as np
from scipy.spatial import cKDTree

# ---------------------------------------------------------------------------
# Proposed default TD values (mm per decade of transmission), indexed [R,G,B]
# ---------------------------------------------------------------------------
TD_DEFAULT = {
    'W': np.array([5.4, 5.4, 5.4]),
    'C': np.array([0.3, 3.0, 3.0]),
    'M': np.array([3.0, 0.3, 3.0]),
    'Y': np.array([3.0, 3.0, 0.3]),
}

D_W_BASE = 0.45          # fixed white base thickness [mm]
STEP     = 0.08          # one layer thickness [mm]
N_LEVELS = 9             # 0..8 layers
MAX_THICK = STEP * (N_LEVELS - 1)   # 0.64 mm


def trans(thickness, td):
    """Transmission 10^(-d/TD) (vectorized over thickness and/or td)."""
    t = np.asarray(thickness, dtype=float)
    td = np.asarray(td, dtype=float)
    if t.ndim == 1 and td.ndim == 1:
        t = t[:, None]           # (n,1) x (3,) -> (n,3)
    return np.power(10.0, -(t / td))


def forward_linear(dC, dM, dY, dW=D_W_BASE, td=None):
    """Linear RGB after backlight passes W/C/M/Y stack."""
    if td is None:
        td = TD_DEFAULT
    return (trans(dW, td['W'])
            * trans(dC, td['C'])
            * trans(dM, td['M'])
            * trans(dY, td['Y']))


# ---------------------------------------------------------------------------
# Colorimetry
# ---------------------------------------------------------------------------
def srgb_encode(lin):
    lin = np.clip(lin, 0.0, 1.0)
    return np.where(lin <= 0.0031308, lin * 12.92, 1.055 * np.power(lin, 1 / 2.4) - 0.055)


def srgb_decode(s):
    s = np.clip(s, 0.0, 1.0)
    return np.where(s <= 0.04045, s / 12.92, np.power((s + 0.055) / 1.055, 2.4))


M_XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
])
D65 = np.array([0.95047, 1.0, 1.08883])


def linear_to_lab(lin):
    """Linear RGB (assumed D65 white = 1,1,1) -> CIE Lab."""
    xyz = np.asarray(lin, dtype=float) @ M_XYZ.T
    xyz = xyz / D65
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)


def srgb_to_lab(s):
    return linear_to_lab(srgb_decode(s))


def ciede2000(lab1, lab2):
    """Vectorized CIEDE2000. lab1/lab2 shape (...,3) in CIE Lab."""
    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]
    C1 = np.hypot(a1, b1)
    C2 = np.hypot(a2, b2)
    Cbar = (C1 + C2) / 2
    G = 0.5 * (1 - np.sqrt(Cbar ** 7 / (Cbar ** 7 + 25 ** 7)))
    ap1 = (1 + G) * a1
    ap2 = (1 + G) * a2
    Cp1 = np.hypot(ap1, b1)
    Cp2 = np.hypot(ap2, b2)
    hp1 = np.degrees(np.arctan2(b1, ap1)) % 360
    hp2 = np.degrees(np.arctan2(b2, ap2)) % 360
    dLp = L2 - L1
    dCp = Cp2 - Cp1
    dhp = np.where(Cp1 * Cp2 == 0, 0.0,
                   np.where(np.abs(hp2 - hp1) <= 180, hp2 - hp1,
                            np.where(hp2 - hp1 > 180, hp2 - hp1 - 360, hp2 - hp1 + 360)))
    dHp = 2 * np.sqrt(Cp1 * Cp2) * np.sin(np.radians(dhp) / 2)
    Lbar = (L1 + L2) / 2
    Cbarp = (Cp1 + Cp2) / 2
    hpbar = np.where(Cp1 * Cp2 == 0, hp1 + hp2,
                     np.where(np.abs(hp1 - hp2) <= 180, (hp1 + hp2) / 2,
                              np.where((hp1 + hp2) < 360, (hp1 + hp2 + 360) / 2,
                                       (hp1 + hp2 - 360) / 2)))
    T = (1 - 0.17 * np.cos(np.radians(hpbar - 30))
         + 0.24 * np.cos(np.radians(2 * hpbar))
         + 0.32 * np.cos(np.radians(3 * hpbar + 6))
         - 0.20 * np.cos(np.radians(4 * hpbar - 63)))
    dtheta = 30 * np.exp(-((hpbar - 275) / 25) ** 2)
    RC = 2 * np.sqrt(Cbarp ** 7 / (Cbarp ** 7 + 25 ** 7))
    SL = 1 + 0.015 * (Lbar - 50) ** 2 / np.sqrt(20 + (Lbar - 50) ** 2)
    SC = 1 + 0.045 * Cbarp
    SH = 1 + 0.015 * Cbarp * T
    RT = -np.sin(np.radians(2 * dtheta)) * RC
    de = np.sqrt((dLp / SL) ** 2 + (dCp / SC) ** 2 + (dHp / SH) ** 2
                 + RT * (dCp / SC) * (dHp / SH))
    return de


# ---------------------------------------------------------------------------
# Color card
# ---------------------------------------------------------------------------
def make_card(levels=None, td=None, dW=D_W_BASE):
    """All (dC,dM,dY) combos -> thickness, linear RGB, sRGB, Lab."""
    if td is None:
        td = TD_DEFAULT
    if levels is None:
        levels = np.arange(N_LEVELS) * STEP
    dC, dM, dY = np.meshgrid(levels, levels, levels, indexing='ij')
    dC = dC.ravel(); dM = dM.ravel(); dY = dY.ravel()
    lin = forward_linear(dC, dM, dY, dW=dW, td=td)
    srgb = srgb_encode(lin)
    lab = linear_to_lab(lin)
    return dC, dM, dY, lin, srgb, lab


def nearest_card(card_lab, t_lab):
    """Lab-Euclidean nearest neighbor in card (same method as the scheme)."""
    tree = cKDTree(card_lab)
    _, idx = tree.query(np.asarray(t_lab, dtype=float))
    return idx


def coverage_stats(card_lab, card_srgb, targets_srgb, tree=None):
    """For each target sRGB: nearest card -> predicted sRGB -> DeltaE2000.
    Returns de array (and predicted sRGB)."""
    t_lab = srgb_to_lab(targets_srgb)
    idx = nearest_card(card_lab, t_lab)
    pred_srgb = card_srgb[idx]
    de = ciede2000(t_lab, srgb_to_lab(pred_srgb))
    return de, pred_srgb


def summarize(de, name=""):
    de = np.asarray(de)
    print(f"  [{name}] n={len(de)}  median={np.median(de):.2f}  "
          f"mean={de.mean():.2f}  p90={np.percentile(de, 90):.2f}  "
          f"p95={np.percentile(de, 95):.2f}  max={de.max():.2f}")
    for thr in (3, 6, 10, 15):
        frac = 100.0 * np.mean(de <= thr)
        print(f"      dE<= {thr:>2}: {frac:6.2f}%")
    return de
