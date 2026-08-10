"""Experiment 4: layer budget. Is 8 layers (max 0.64 mm) enough?

Tests:
  a) Coverage with 8/12/16 layers at same 0.08 mm step -> does adding
     thickness headroom fix the out-of-gamut errors?
  b) Where would clipped thicknesses occur? Simulate real inversion:
     unconstrained continuous thickness solve (per-pixel, Newton) for
     random sRGB, count pixels needing d > 0.64 mm.
  c) Truncation error: colors whose required d exceeds max -> dE after clamp.
"""
import numpy as np
import sys
sys.path.insert(0, '.')
from common import (TD_DEFAULT, D_W_BASE, STEP, make_card, coverage_stats,
                    summarize, srgb_decode, srgb_to_lab, forward_linear,
                    linear_to_lab, ciede2000, srgb_encode, nearest_card)

rng = np.random.default_rng(11)
targets = rng.random((10000, 3))
t_lin = srgb_decode(targets)

print("=" * 78)
print("EXP4: layer budget (levels at 0.08 mm step)")
print("=" * 78)

# (a) coverage vs #layers
for nl in (9, 13, 17):                      # 0..8, 0..12, 0..16 layers
    lv = np.arange(nl) * STEP
    dC, dM, dY, lin, csrgb, clab = make_card(lv)
    de, pred = coverage_stats(clab, csrgb, targets)
    print(f"\n[a] {nl-1:2d} layers (max {(nl-1)*STEP:.2f} mm), card size {len(dC)}:")
    summarize(de, f"{nl-1} layers")

# (b) unconstrained continuous thickness solve via Newton (per pixel, 3 ch -> 3 unknowns)
#     solve linear RGB target exactly: -log10(tau_c) = sum d_i/TD_ic
print("\n[b] Continuous unconstrained solve (Newton, exact Beer-Lambert):")
# A d = v   where A[i,dye] = 1/TD_{dye,ch}  (ch x dye), d dye x 1, v = log10(1/target_lin) - dW terms
td = TD_DEFAULT
dyes = ['C', 'M', 'Y']
A = np.array([[1.0 / td[k][c] for k in dyes] for c in range(3)])   # 3ch x 3dye
wbase = np.full(3, -np.log10(10 ** (-D_W_BASE / td['W'][0])))
# vector over pixels: v = -log10(t_lin) - wbase
v = -np.log10(np.clip(t_lin, 1e-9, 1)) - wbase[None, :]            # n x 3ch
d_uncon = np.linalg.solve(A, v.T).T                                # n x 3 dye thickness
neg = (d_uncon < -1e-6).sum(axis=0)
over = (d_uncon > 0.64 + 1e-6).sum(axis=0)
print(f"  needed thickness distribution (mm) over 10000 random sRGB:")
for j, nm in enumerate(dyes):
    d = d_uncon[:, j]
    print(f"    {nm}: min={d.min():.3f}  p10={np.percentile(d,10):.3f}  "
          f"median={np.median(d):.3f}  p90={np.percentile(d,90):.3f}  "
          f"max={d.max():.3f}   #<0: {neg[j]}  #>0.64: {over[j]}")

# (c) truncation: clamp negative to 0 and >0.64 to 0.64, compute dE
d_clamped = np.clip(d_uncon, 0.0, 0.64)
lin_clamped = forward_linear(d_clamped[:, 0], d_clamped[:, 1], d_clamped[:, 2])
lab_t = srgb_to_lab(targets)
lab_clamped = linear_to_lab(lin_clamped)
de_clamp = ciede2000(lab_t, lab_clamped)
print("\n[c] Truncation error (unconstrained solve clamped to [0, 0.64]):")
summarize(de_clamp, "clamped continuous")

# (c2) with 12 and 16 layers (max 0.88 / 1.2):
for maxd in (0.88, 1.20):
    d_c2 = np.clip(d_uncon, 0.0, maxd)
    l2 = forward_linear(d_c2[:, 0], d_c2[:, 1], d_c2[:, 2])
    de2 = ciede2000(lab_t, linear_to_lab(l2))
    print(f"\n[c] Truncation error with max {maxd:.2f} mm ({int(maxd/STEP)} layers):")
    summarize(de2, f"max {maxd:.2f}")

# (d) which colors need more than 0.64mm? map the demand
print("\n[d] Demand for thickness >0.64mm by target region:")
needs_more = (d_uncon.max(axis=1) > 0.64)
lab = lab_t[needs_more]
hue = np.degrees(np.arctan2(lab[:, 2], lab[:, 1])) % 360
print(f"  {needs_more.sum()}/10000 random sRGB need > 0.64mm in some dye "
      f"({100*needs_more.mean():.1f}%)")
print(f"  mean target L* of those: {lab[:,0].mean():.1f}  vs all: {lab_t[:,0].mean():.1f}")
bins = np.arange(0, 361, 30)
for k in range(12):
    m = (hue >= bins[k]) & (hue < bins[k+1])
    if m.sum():
        print(f"    hue [{bins[k]:3d},{bins[k+1]:3d}): {m.sum()}")
