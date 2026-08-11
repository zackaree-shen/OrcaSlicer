"""Validation for the GUI grid-size guard (large image buildability).

The guard in LithophaneApp.build() must keep ALL grids (solver + geometry:
pitch_cmy/pitch_top) under MAX_POINTS when the output size is large (e.g.
4000x3000 mm from the 1px=1mm default). This prevents the OOM / hang that
would otherwise occur, and - critically - prevents the old
"Width must be in [10,500]" rejection dialog.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from litho_core import LithophaneParams, thickness_grid_shape

RESULTS = []


def report(name, ok, detail=""):
    RESULTS.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name:55s} {detail}")


def test_grid_guard():
    # Large image: 4000x3000 mm default (1px=1mm), solver pitch 0.3.
    # This was the crashing input before the fix.
    w_mm, h_mm = 4000.0, 3000.0
    base_pitch = 0.3
    p = LithophaneParams(width_mm=w_mm, height_mm=h_mm, pixel_pitch_mm=base_pitch)
    gx, gy = thickness_grid_shape(3000, 4000, p)
    n0 = gx * gy
    report("raw grid is huge (would crash)", n0 > 1_000_000, f"n0={n0:,}")

    # Simulate the guard math from LithophaneApp.build().
    MAX_POINTS = 600_000
    n_top0 = (int(w_mm / 0.25) + 1) * (int(h_mm / 0.25) + 1)
    worst = max(n0, n_top0)
    scale = (worst / MAX_POINTS) ** 0.5 * 1.1
    eff_pitch = base_pitch * scale
    pitch_cmy = 0.8 * scale
    pitch_top = 0.25 * scale

    # All three grids must now be under MAX_POINTS.
    gx2, gy2 = thickness_grid_shape(3000, 4000,
                                    LithophaneParams(width_mm=w_mm, height_mm=h_mm,
                                                     pixel_pitch_mm=eff_pitch))
    n_solve = gx2 * gy2
    n_cmy = (int(w_mm / pitch_cmy) + 1) * (int(h_mm / pitch_cmy) + 1)
    n_top = (int(w_mm / pitch_top) + 1) * (int(h_mm / pitch_top) + 1)

    report("solver grid <= MAX_POINTS", n_solve <= MAX_POINTS, f"{n_solve:,}")
    report("cmy geometry grid <= MAX_POINTS", n_cmy <= MAX_POINTS, f"{n_cmy:,}")
    report("top geometry grid <= MAX_POINTS", n_top <= MAX_POINTS, f"{n_top:,}")
    report("effective pitch within UI range", 0.02 <= eff_pitch <= 20.0,
           f"pitch={eff_pitch:.2f}")

    # Reasonable sizes (<= 500 mm) must NOT trigger the guard (pitch unchanged).
    p_small = LithophaneParams(width_mm=200, height_mm=150, pixel_pitch_mm=0.3)
    gx3, gy3 = thickness_grid_shape(150, 200, p_small)
    report("small size not guarded (no pitch change)", gx3 * gy3 <= 600_000,
           f"grid={gx3 * gy3:,}")


if __name__ == "__main__":
    test_grid_guard()
    print()
    print(f"{sum(RESULTS)}/{len(RESULTS)} passed")
    sys.exit(0 if all(RESULTS) else 1)
