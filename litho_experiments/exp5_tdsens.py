"""Experiment 5: TD sensitivity. How much does the C TD_R default matter?

Variants: C TD_R in {0.2, 0.3, 0.5, 1.0}; also C TD_G/TD_B in {3.0, 1.0, 10.0}.
Measure:
  a) change in the 729-card in linear RGB / Lab (mean |dLab|, white point shift)
  b) change in inverse results: re-run NN inversion for fixed random targets,
     compare recovered (dC,dM,dY) and final dE vs the baseline.
"""
import numpy as np
import sys
sys.path.insert(0, '.')
from common import (TD_DEFAULT, make_card, coverage_stats, summarize,
                    srgb_to_lab, ciede2000, nearest_card, linear_to_lab,
                    srgb_decode)

rng = np.random.default_rng(3)
targets = rng.random((10000, 3))
levels = np.arange(9) * 0.08

base_td = TD_DEFAULT
_, _, _, base_lin, base_srgb, base_lab = make_card(levels, td=base_td)
base_idx = nearest_card(base_lab, srgb_to_lab(targets))
base_de, _ = coverage_stats(base_lab, base_srgb, targets)
print(f"Baseline C TD_R=0.3: median dE={np.median(base_de):.2f} p90={np.percentile(base_de,90):.2f}")

variants = {
    "C TD_R=0.2": {**base_td, 'C': np.array([0.2, 3.0, 3.0])},
    "C TD_R=0.5": {**base_td, 'C': np.array([0.5, 3.0, 3.0])},
    "C TD_R=1.0": {**base_td, 'C': np.array([1.0, 3.0, 3.0])},
    "C TD_G=1.0 (weaker cross-abs)": {**base_td, 'C': np.array([0.3, 1.0, 3.0])},
    "C TD_G=10 (stronger cross-abs)": {**base_td, 'C': np.array([0.3, 10.0, 3.0])},
    "C TD_R=0.6&TD_G=0.6 (overlap)": {**base_td, 'C': np.array([0.6, 0.6, 3.0])},
}
for nm, td in variants.items():
    dC, dM, dY, lin, srgb, lab = make_card(levels, td=td)
    # a) card change
    dlab = np.abs(lab - base_lab)
    dlin = np.abs(lin - base_lin)
    print(f"\n### {nm}")
    print(f"  card change: mean|dLin|={dlin.mean():.4f} max|dLin|={dlin.max():.4f}  "
          f"mean|dLab|={dlab.mean():.2f} max|dLab|={dlab.max():.2f}")
    # white point
    w = np.argmin(dC + dM + dY)
    print(f"  white point Lab: {lab[w].round(2)} (baseline {base_lab[0].round(2)})")
    # C primary at max
    im = np.argmax(dC)  # max C, 0 M, 0 Y (first occurrence)
    print(f"  C-max 8-bit sRGB: {(srgb[im]*255).round(0).astype(int)} "
          f"(baseline {(base_srgb[im]*255).round(0).astype(int)})")
    # b) inverse change
    idx = nearest_card(lab, srgb_to_lab(targets))
    # recovered thickness difference vs baseline (only where both are NN matches)
    rec_v = np.stack([dC[idx], dM[idx], dY[idx]], axis=-1)
    rec_b = np.stack([np.full(len(targets), levels[0]),]*3, axis=-1)  # placeholder
    dCb = (np.argmin(np.abs(levels - base_td['C'][0])))*0.08  # not meaningful; skip
    # compare final color quality
    de, _ = coverage_stats(lab, srgb, targets)
    print(f"  coverage with this TD: median dE={np.median(de):.2f} "
          f"(vs baseline {np.median(base_de):.2f})  p90={np.percentile(de,90):.2f}")
    # how many pixels changed thickness assignment vs baseline card?
    changed = (idx != base_idx).mean()
    print(f"  fraction of pixels whose NN card entry changed vs baseline: {100*changed:.1f}%")
