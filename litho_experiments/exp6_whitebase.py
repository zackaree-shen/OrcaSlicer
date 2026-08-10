"""Experiment 6: white base thickness. dW = 0.3 / 0.45 / 0.6.

Measures:
  a) white point Lab and its dE to pure sRGB white for each dW
  b) whole-card mean L*, and coverage stats over random sRGB
  c) does dW=0.3 make colors brighter (better highlights) but lose
     contrast / make black too weak? Compare black-point.
"""
import numpy as np
import sys
sys.path.insert(0, '.')
from common import (make_card, coverage_stats, summarize, srgb_to_lab,
                    ciede2000, forward_linear, srgb_encode, linear_to_lab,
                    TD_DEFAULT)

rng = np.random.default_rng(5)
targets = rng.random((10000, 3))
t_lab = srgb_to_lab(targets)
levels = np.arange(9) * 0.08
white = np.array([[1., 1., 1.]])
w_lab = srgb_to_lab(white)

print("=" * 78)
print("EXP6: white base role (dW sweep)")
print("=" * 78)
for dW in (0.30, 0.45, 0.60, 0.90):
    dC, dM, dY, lin, csrgb, clab = make_card(levels, dW=dW)
    iw = np.argmin(dC + dM + dY)
    de_w = ciede2000(clab[iw:iw+1], w_lab)[0]
    ib = np.argmax(dC + dM + dY)
    de, pred = coverage_stats(clab, csrgb, targets)
    print(f"\n### dW={dW:.2f} mm")
    print(f"  white point sRGB8=({(csrgb[iw]*255).round(0).astype(int)})  "
          f"L*={clab[iw][0]:.1f}  dE2000 vs pure white={de_w:.2f}")
    print(f"  black point sRGB8=({(csrgb[ib]*255).round(0).astype(int)})  L*={clab[ib][0]:.1f}")
    print(f"  card median L*={np.median(clab[:,0]):.1f}  frac L*>80: {100*np.mean(clab[:,0]>80):.1f}%")
    summarize(de, f"dW={dW:.2f}")
    # luminance contrast range (max L* - min L*)
    print(f"  card L* range: {clab[:,0].min():.1f} .. {clab[:,0].max():.1f} "
          f"(span {clab[:,0].max()-clab[:,0].min():.1f})")
