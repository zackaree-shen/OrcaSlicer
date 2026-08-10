"""Experiment 2: how much sRGB gamut does the 729 CMYW card cover?

Nearest-neighbour in Lab (exactly the scheme's inverse step), then
forward-evaluate predicted sRGB, report DeltaE2000 vs target.
Sampling: uniform 8x8x8=512 8-bit cube + 10000 uniform-random sRGB.
"""
import numpy as np
import sys
sys.path.insert(0, '.')
from common import make_card, coverage_stats, summarize, srgb_to_lab, ciede2000

levels = np.arange(9) * 0.08
dC, dM, dY, lin, card_srgb, card_lab = make_card(levels)

def run(targets, name):
    print(f"\n### Coverage of {name} ({len(targets)} colors) by 729-card nearest neighbor")
    de, pred = coverage_stats(card_lab, card_srgb, targets)
    summarize(de, name)
    return de, pred

# --- 8^3=512 grid (avoid near-black duplicates, but keep all for realism) ---
grid = (np.arange(8)[:, None, None] / 7.0)
r = np.broadcast_to(grid, (8, 8, 8)); g = np.broadcast_to(grid.T, (8, 8, 8)); b = np.broadcast_to(grid.T.T, (8, 8, 8))
targets_grid = np.stack([r.ravel(), g.ravel(), b.ravel()], axis=-1)
de_grid, pred_grid = run(targets_grid, "8x8x8 grid")

rng = np.random.default_rng(42)
targets_rand = rng.random((10000, 3))
de_rand, pred_rand = run(targets_rand, "10000 random sRGB")

# --- worst 10 targets (combine both sets) ---
targets_all = np.concatenate([targets_grid, targets_rand])
de_all = np.concatenate([de_grid, de_rand])
pred_all = np.concatenate([pred_grid, pred_rand])
worst = np.argsort(de_all)[-10:][::-1]
print("\n### Worst-10 target colors (sRGB) and their DeltaE2000 to best card entry:")
for i in worst:
    t = targets_all[i]
    print(f"  target sRGB8=({(t*255).round(0).astype(int)})  "
          f"dE={de_all[i]:.2f}  predicted sRGB8=({(pred_all[i]*255).round(0).astype(int)})  "
          f"target Lab={srgb_to_lab(t[None,:])[0].round(1)}")

# --- which regions are bad: table by lightness and hue of target ---
print("\n### Coverage by target lightness (random set):")
t_lab = srgb_to_lab(targets_rand)
for lo, hi in [(0, 25), (25, 50), (50, 75), (75, 90), (90, 101)]:
    m = (t_lab[:, 0] >= lo) & (t_lab[:, 0] < hi)
    if m.sum():
        print(f"  target L* in [{lo},{hi}): n={m.sum():5d}  median dE={np.median(de_rand[m]):.2f}  "
              f"p90={np.percentile(de_rand[m], 90):.2f}  <=3: {100*np.mean(de_rand[m]<=3):.0f}%  <=6: {100*np.mean(de_rand[m]<=6):.0f}%")

print("\n### Coverage by target hue (random set, 30-deg bins):")
hue_t = np.degrees(np.arctan2(t_lab[:, 2], t_lab[:, 1])) % 360
bins = np.arange(0, 361, 30)
for k in range(12):
    m = (hue_t >= bins[k]) & (hue_t < bins[k + 1])
    if m.sum():
        print(f"  hue [{bins[k]:3d},{bins[k+1]:3d}): n={m.sum():5d}  median dE={np.median(de_rand[m]):.2f}  "
              f"p90={np.percentile(de_rand[m], 90):.2f}  <=3: {100*np.mean(de_rand[m]<=3):.0f}%  <=6: {100*np.mean(de_rand[m]<=6):.0f}%")

# --- how far is the gamut boundary from sRGB primaries? (can we get saturated colors?) ---
print("\n### Distance from best card entry for saturated sRGB primaries:")
for nm, s in [("Red(255,0,0)", [1, 0, 0]), ("Green(0,255,0)", [0, 1, 0]), ("Blue(0,0,255)", [0, 0, 1]),
              ("Yellow(255,255,0)", [1, 1, 0]), ("Cyan(0,255,255)", [0, 1, 1]), ("Magenta(255,0,255)", [1, 0, 1]),
              ("White", [1, 1, 1]), ("Black", [0, 0, 0]),
              ("Orange(255,128,0)", [1, .5, 0]), ("Purple(128,0,255)", [.5, 0, 1]), ("Teal(0,128,255)", [0, .5, 1])]:
    t = np.array([s])
    de, pred = coverage_stats(card_lab, card_srgb, t)
    print(f"  {nm:16s} target={(np.array(s)*255).round(0).astype(int)}  dE={de[0]:6.2f}  "
          f"best card sRGB8=({(pred[0]*255).round(0).astype(int)})")
