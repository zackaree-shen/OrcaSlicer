"""Experiment 1: forward-model self-consistency and the 729-color card.

Checks (per the proposed scheme):
  a) all-zero C/M/Y -> output should be white (backlight attenuated by W base)
  b) single-color max thickness -> C/M/Y at 0.64mm should be strongly tinted
  c) Lab distribution / coverage shape of the card
  d) distinctness of the 729 card entries (quantization collapses)
"""
import numpy as np
import sys
sys.path.insert(0, '.')
from common import (TD_DEFAULT, D_W_BASE, STEP, N_LEVELS, MAX_THICK,
                    forward_linear, srgb_encode, srgb_decode,
                    linear_to_lab, srgb_to_lab, make_card, ciede2000)

levels = np.arange(N_LEVELS) * STEP
dC, dM, dY, lin, srgb, lab = make_card(levels)

print("=" * 78)
print("EXP1: forward model & 729-card self-consistency")
print("=" * 78)
print(f"card size: {len(dC)}  (levels={N_LEVELS}^3, step={STEP}, max={MAX_THICK})")

# --- white point (all zero) ---
i0 = np.argmin(dC + dM + dY)
print("\n[1a] White point: dC=dM=dY=0, dW=0.45")
print(f"  linear RGB = {lin[i0].round(4)}")
print(f"  sRGB       = {srgb[i0].round(4)}  -> 8-bit {(srgb[i0]*255).round(0).astype(int)}")
print(f"  Lab        = {lab[i0].round(2)}")
print(f"  (W base alone attenuates each channel to 10^(-0.45/5.4)= {10**(-0.45/5.4):.4f}, L*= {lab[i0][0]:.1f})")
white_target = srgb_to_lab(np.array([[1., 1., 1.]]))
print(f"  dE2000 vs pure sRGB white (100,0,0): {ciede2000(lab[i0:i0+1], white_target)[0]:.2f}")

# --- single-dye maxima ---
print("\n[1b] Single-dye at max thickness (others 0):")
for name, d in [("C", (MAX_THICK, 0, 0)), ("M", (0, MAX_THICK, 0)), ("Y", (0, 0, MAX_THICK))]:
    l = forward_linear(*d)
    s = srgb_encode(l)
    L = linear_to_lab(l)[0]
    print(f"  {name} max {MAX_THICK}mm: linear={l.round(4)}  sRGB={s.round(3)}"
          f"  (8bit {(s*255).round(0).astype(int)})  Lab={L.round(1)}")

# --- the 8 vertices of the reachable set ---
print("\n[1c] Reachable-set vertices (0 / max thickness per dye):")
verts = []
for a in (0., MAX_THICK):
    for b in (0., MAX_THICK):
        for c in (0., MAX_THICK):
            l = forward_linear(a, b, c)
            s = srgb_encode(l)
            verts.append((a, b, c, l, s))
for a, b, c, l, s in verts:
    print(f"  C={a:.2f} M={b:.2f} Y={c:.2f} -> sRGB {s.round(3)} (8bit {(s*255).round(0).astype(int)})  L*={linear_to_lab(l)[0]:.1f}")

# --- card Lab distribution ---
print("\n[1d] Card Lab distribution:")
L = lab[:, 0]; a = lab[:, 1]; b = lab[:, 2]
C = np.hypot(a, b)
hue = np.degrees(np.arctan2(b, a)) % 360
print(f"  L*: min={L.min():.1f}  median={np.median(L):.1f}  max={L.max():.1f}")
print(f"  a*: min={a.min():.1f}  max={a.max():.1f}   b*: min={b.min():.1f}  max={b.max():.1f}")
print(f"  chroma: min={C.min():.1f}  median={np.median(C):.1f}  max={C.max():.1f}")
for th in (70, 80, 90):
    print(f"  fraction of card with L* > {th}: {100*np.mean(L > th):.1f}%")
for th in (60, 80, 100):
    print(f"  fraction of card with chroma > {th}: {100*np.mean(C > th):.1f}%")
print(f"  hue coverage: min={hue.min():.1f} max={hue.max():.1f}  "
      f"fraction within any 30deg bin... (see table below)")
import collections
bins = np.arange(0, 361, 30)
cnt, _ = np.histogram(hue, bins)
print(f"  hue histogram (30-deg bins, 0=red/yellow axis): {cnt}")

# --- distinctness in 8-bit sRGB (quantization collapse) ---
print("\n[1e] Distinctness of the 729 card entries:")
u8 = np.round(srgb * 255).astype(int)
_, counts = np.unique(u8, axis=0, return_counts=True)
print(f"  unique in 8-bit sRGB: {len(counts)} / 729")
print(f"  duplicated entries: {729 - len(counts)}")
if len(counts):
    dup = counts[counts > 1]
    if len(dup):
        print(f"  max duplicates per color: {dup.max()},  #colors with >1 card entry: {(counts>1).sum()}")
# distinct in float Lab?
print(f"  unique in float Lab: {np.unique(lab, axis=0).shape[0]} / 729")

# --- nearest-neighbour spacing of the card (grid coarseness) ---
from scipy.spatial import cKDTree
tree = cKDTree(lab)
d2, _ = tree.query(lab, k=2)
spacing = d2[:, 1]
print(f"\n[1f] Card grid spacing in Lab (euclid, to 2nd nearest): "
      f"median={np.median(spacing):.1f}  p90={np.percentile(spacing,90):.1f}  max={spacing.max():.1f}")

# --- dark end: max all -> near black? ---
i_all = np.argmax(dC + dM + dY)
print(f"\n[1g] All dyes max: sRGB={srgb[i_all].round(4)}  Lab={lab[i_all].round(2)}")
