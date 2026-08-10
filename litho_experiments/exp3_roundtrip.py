"""Experiment 3: inverse problem round-trip precision.

(a) On-card: for each of the 729 card colors, invert (kd-tree NN) and
    forward again -> dE should be ~0 if table lookup is exact.
(b) Thickness recovery: check the NN actually returns the exact (dC,dM,dY).
(c) Off-card: uniform random sRGB targets -> forward-roundtrip dE
    (this is dominated by discretization, quantified in exp2).
(d) Quantization level needed to reach dE<2 median (sweep levels).
"""
import numpy as np
import sys
sys.path.insert(0, '.')
from common import (make_card, coverage_stats, summarize, srgb_to_lab,
                    ciede2000, forward_linear, srgb_encode, linear_to_lab)

levels = np.arange(9) * 0.08
dC, dM, dY, lin, card_srgb, card_lab = make_card(levels)

# (a) on-card round trip
from common import nearest_card
idx = nearest_card(card_lab, card_lab)          # NN of each card point in itself
recovered = np.stack([dC[idx], dM[idx], dY[idx]], axis=-1)
orig = np.stack([dC, dM, dY], axis=-1)
match = np.all(recovered == orig, axis=1)
print(f"[3a] On-card inversion exact-thickness recovery: {100*np.mean(match):.1f}% "
      f"({np.sum(match)}/729)")
if not match.all():
    bad = np.where(~match)[0]
    print("  mismatched examples:", [(orig[i], recovered[i]) for i in bad[:5]])

# predicted color = the card color itself -> dE 0 by construction
pred_lin = lin[idx]
de_oncard = ciede2000(card_lab, linear_to_lab(pred_lin))
print(f"[3a] On-card roundtrip dE2000: max={de_oncard.max():.2e} (machine-precision expected)")

# (b) off-card roundtrip error (discretization dominated)
rng = np.random.default_rng(7)
targets = rng.random((10000, 3))
de_off, pred = coverage_stats(card_lab, card_srgb, targets)
print("[3b] Off-card roundtrip dE2000 (10000 random sRGB):")
summarize(de_off, "random off-card")

# (c) how much of the error is quantization vs gamut limitation?
#     Reverse: take card colors, move to a random layer step, invert, forward.
#     This isolates discretization for achievable colors.
print("\n[3c] Discretization only (perturb achievable colors, then invert):")
# pick random achievable linear colors = card colors perturbed slightly
i = rng.integers(0, len(dC), 3000)
base_lin = lin[i]
noise = rng.uniform(-0.004, 0.004, base_lin.shape)     # tiny linear perturbation
t = np.clip(base_lin + noise, 0, 1)
t_srgb = srgb_encode(t)
de_disc, pred_disc = coverage_stats(card_lab, card_srgb, t_srgb)
summarize(de_disc, "perturbed-achievable")

# (d) what does grid coarseness contribute in Lab-distance terms?
from scipy.spatial import cKDTree
tree = cKDTree(card_lab)
d2, _ = tree.query(card_lab, k=2)
print(f"\n[3d] Card grid 2nd-nearest Lab distance: median={np.median(d2[:,1]):.2f} "
      f"p90={np.percentile(d2[:,1],90):.2f} max={d2[:,1].max():.2f}")
# average dE of random colors to nearest card, split by which error source:
# The 10000-random median 5.45 includes both discretization and out-of-gamut.
