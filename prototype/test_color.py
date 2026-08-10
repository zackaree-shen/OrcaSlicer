"""Validation for the M2 color lithophane (CMYW stacked transmission)."""

from __future__ import annotations

import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from litho_color import (
    DEFAULT_TD, build_gamut, ciede2000_matrix, forward_transmission,
    linear_to_srgb8, solve_thicknesses, srgb8_to_linear, srgb8_to_lab,
    xyz_to_lab,
)
from litho_core import LithophaneParams, heightfield_to_mesh, validate_mesh

RESULTS = []


def report(name, ok, detail):
    RESULTS.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name:50s} {detail}")


def test_color_space():
    # White sRGB -> Lab ~ (100, 0, 0); known values from sRGB spec.
    lab_white = srgb8_to_lab(np.array([[[255, 255, 255]]]))
    report("sRGB white -> Lab ~(100,0,0)",
           abs(lab_white[0, 0, 0] - 100) < 0.5 and abs(lab_white[0, 0, 1]) < 0.5 and abs(lab_white[0, 0, 2]) < 0.5,
           f"L={lab_white[0,0,0]:.2f}")

    lab_black = srgb8_to_lab(np.array([[[0, 0, 0]]]))
    report("sRGB black -> Lab ~(0,0,0)",
           lab_black[0, 0, 0] < 1.0, f"L={lab_black[0,0,0]:.2f}")

    # dE2000 self-consistency.
    a = np.array([[50, 20, 30]])
    report("dE2000 self-consistency dE(a,a)=0", float(ciede2000_matrix(a, a)[0, 0]) < 1e-9,
           f"dE={float(ciede2000_matrix(a, a)[0, 0]):.2e}")

    # Known pair: lab (50,2.6772,-79.7751) vs (50,0,-82.7485) => dE=2.0425 (Sharma dataset)
    p1 = np.array([[50.0, 2.6772, -79.7751]])
    p2 = np.array([[50.0, 0.0, -82.7485]])
    dE = float(ciede2000_matrix(p1, p2)[0, 0])
    report("dE2000 vs Sharma dataset (2.0425)", abs(dE - 2.0425) < 0.001, f"dE={dE:.4f}")


def test_forward_model():
    # Zero color thickness -> white (only base W attenuates).
    rgb = forward_transmission(0, 0, 0, 0.30, td=DEFAULT_TD)
    rgb8 = linear_to_srgb8(rgb[np.newaxis, np.newaxis, :])
    report("0 thickness -> near-white", rgb8[0, 0, 1] > 230 and abs(int(rgb8[0,0,0]) - int(rgb8[0,0,2])) < 12,
           f"rgb8={tuple(rgb8[0,0])}")

    # Max cyan thickness -> should be cyan-tinted (B > R).
    rgb = forward_transmission(0.64, 0, 0, 0.30, td=DEFAULT_TD)
    rgb8 = linear_to_srgb8(rgb[np.newaxis, np.newaxis, :])
    report("max C -> cyan tint (B>R)", int(rgb8[0,0,2]) > int(rgb8[0,0,0]),
           f"rgb8={tuple(rgb8[0,0])}")

    # Max Y -> yellow tint (R>B).
    rgb = forward_transmission(0, 0, 0.64, 0.30, td=DEFAULT_TD)
    rgb8 = linear_to_srgb8(rgb[np.newaxis, np.newaxis, :])
    report("max Y -> yellow tint (R>B)", int(rgb8[0,0,0]) > int(rgb8[0,0,2]),
           f"rgb8={tuple(rgb8[0,0])}")


def test_gamut_and_solver():
    gamut = build_gamut(layers_max=8, layer_h=0.08, dW=0.30)
    n = gamut["lab"].shape[0]
    report("gamut size = 729", n == 729, f"M={n}")

    # Reachable coverage: sample sRGB cube, measure gamut-mapped dE2000.
    rng = np.random.default_rng(0)
    targets = rng.integers(0, 256, size=(2000, 1, 3))
    dC, dM, dY, dE, _ = solve_thicknesses(targets, gamut, chunk=512)
    dE_flat = dE.ravel()
    med = float(np.median(dE_flat))
    p90 = float(np.percentile(dE_flat, 90))
    frac6 = float(np.mean(dE_flat <= 6))
    frac3 = float(np.mean(dE_flat <= 3))
    # Honest reporting: report the numbers regardless of pass/fail threshold.
    # Acceptable if median <= 6 and >=40% within 6 (soft-color expectation).
    ok = med <= 6.0 and frac6 >= 0.40
    report(f"gamut coverage: median dE={med:.2f}, p90={p90:.2f}, <=3:{frac3:.0%}, <=6:{frac6:.0%}",
           ok, f"(soft-color expectation)")

    # Thickness range sanity.
    report("solver thickness within [0, 0.64]",
           dC.min() >= 0 and dC.max() <= 0.64 + 1e-9 and dM.max() <= 0.64 + 1e-9 and dY.max() <= 0.64 + 1e-9,
           f"dC=[{dC.min():.2f},{dC.max():.2f}]")

    return gamut


def test_end_to_end(gamut):
    # Color photo-like image.
    h, w = 360, 480
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    # Gradient + saturated stripes + gray area (tests gamut extremes).
    r = np.clip(0.6 * x / w + 0.3 * np.sin(y / 18), 0, 1)
    g = np.clip(0.6 * (1 - x / w) + 0.3 * np.cos(y / 22), 0, 1)
    b = np.clip(0.5 + 0.4 * np.sin(x / 30) * np.cos(y / 26), 0, 1)
    img = np.stack([r, g, b], axis=-1)
    img = (img * 255).astype(np.uint8)
    Image.fromarray(img, "RGB").save("_color_sample.png")

    params = LithophaneParams(width_mm=144, height_mm=108, pixel_pitch_mm=0.3)
    from litho_color import color_lithophane_meshes
    meshes, dE, _, reached = color_lithophane_meshes(img, params=params)

    # All 4 meshes must be closed solids with positive volume.
    all_ok = True
    ztops = []
    for color in ["W", "C", "M", "Y"]:
        verts, faces = meshes[color]
        v = validate_mesh(verts, faces)
        ok = v["open_edges"] == 0 and v["volume"] > 0 and v["degenerate"] == 0
        all_ok &= ok
        ztops.append(float(verts[:, 2].max()))
        print(f"  mesh {color}: V={v['num_vertices']} F={v['num_faces']} "
              f"vol={v['volume']:,.1f} open={v['open_edges']} zmax={verts[:,2].max():.2f}")
    report("4 color meshes watertight + positive volume", all_ok, "")

    # Z-order check: each mesh occupies its own Z band (W < C < M < Y conceptually,
    # but each is a full solid from 0..zmax; slicing assigns layers by Z height).
    report("Z stacking: Y thickest > M > C > W on the color bands",
           ztops[3] >= ztops[2] >= ztops[1] >= ztops[0],
           f"ztop={[round(z,2) for z in ztops]}")

    # dE map average on this photo-like image.
    report(f"photo-like dE median={float(np.median(dE)):.2f}",
           float(np.median(dE)) <= 6.0, "")

    os.remove("_color_sample.png")
    return meshes


if __name__ == "__main__":
    test_color_space()
    gamut = test_gamut_and_solver()
    test_end_to_end(gamut)
    print()
    print(f"{sum(RESULTS)}/{len(RESULTS)} passed")
    sys.exit(0 if all(RESULTS) else 1)
