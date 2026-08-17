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


def test_smooth_top_antispike():
    """smooth_top=True (balanced, top_tol=0.5) greatly reduces the white-relief
    spike artifact while preserving color+dE: a smooth gray ramp's dTop max
    adjacent step drops by >50% vs raw NN. Regression for the 'W 顶层尖刺' bug
    (nearest-neighbor degeneracy caused ±1.7mm jumps).

    Note (iteration 27 verdict): max_step < 0.2 is NOT the target — that is
    structurally unreachable without doubling dE. The honest balance is
    dE-preserving + spikes roughly halved (raw 1.6 -> smooth ~0.8) + full
    detail (lap ~0.28 ≈ Bambu 0.275)."""
    from litho_color import build_gamut_stacked, solve_stacked
    gamut = build_gamut_stacked(layers_max=8, layer_h=0.1, dW=0.2)
    # Smooth gray ramp L=255..0 (left bright, right dark).
    ramp = np.stack([np.linspace(255, 0, 256, dtype=np.uint8)] * 3, axis=-1)
    ramp = ramp[None, :, :]  # (1, 256, 3)

    dT_raw = None
    for sm in (False, True):
        dT, dC, dM, dY, dE, idx = solve_stacked(ramp, gamut, smooth_top=sm)
        dT = dT[0]
        g = np.abs(np.diff(dT))
        if sm:
            # Balanced: spikes roughly halved vs raw NN.
            g_raw_max = np.abs(np.diff(dT_raw)).max() if dT_raw is not None else 1.6
            report(f"smooth_top={sm}: gray-ramp max dTop step < raw (halved)",
                   g.max() < g_raw_max * 0.6, f"max={g.max():.3f} raw_max={g_raw_max:.3f}")
        else:
            dT_raw = dT
            g_raw = np.abs(np.diff(dT)).max()
            report(f"smooth_top={sm}: raw NN flip-flops (max step > 0.3)",
                   g_raw > 0.3, f"max_step={g_raw:.3f}")


def test_anchored_dtop_field():
    """Iteration 45 v2: OD-domain CDF-equalized dTop field.

    Regression guards for the adversarial findings:
      B1/B2: v1 absolute-inversion clip crushed 84.4% of a dark cartoon to
             dTop == top_max (flat plateaus) -> saturation must stay < 5%.
      B4:    pure-black pixels (log10(0) = -inf) and narrow-dynamic images
             (0/0 NaN) must be guarded.
      ties:  a constant plateau must map to ONE dTop value (mid-rank), not a
             linear gradient inside the plateau.
      monotonicity: darker pixel -> thicker white (field is monotone in L).
    """
    from litho_color import anchored_dtop_field
    TOP = 2.0

    # Dark cartoon stand-in: 80% of pixels below the white window (L<0.391
    # linear ~ sRGB<168), mimicking the measured 保险柜 image statistics.
    rng = np.random.default_rng(7)
    dark_bg = rng.integers(20, 120, (64, 64))                     # dark mass
    bright = rng.integers(170, 255, (16, 64))                     # 20% bright
    img = np.concatenate([np.concatenate([dark_bg, bright], 0)], -1)
    rgb = np.stack([img] * 3, axis=-1).astype(np.uint8)
    d = anchored_dtop_field(rgb, td_w=5.4, top_max=TOP)
    sat = (d >= TOP - 0.01).mean()
    report("anchored: dark-image saturation < 5%", sat < 0.05, f"sat={sat:.3f}")
    report("anchored: layer spread (>=8 of 11 layers used)",
           len(np.unique(np.round(d / 0.2).astype(int))) >= 8,
           f"layers={len(np.unique(np.round(d / 0.2).astype(int)))}")

    # Monotone in the BLURRED luminance (iteration 46: smooth_sigma makes
    # dTop a function of the local neighborhood, not the raw pixel — the
    # blur is what merges noise before ranking; monotonicity vs raw L no
    # longer holds by design).
    from scipy.ndimage import gaussian_filter
    lin_b = gaussian_filter(srgb8_to_linear(rgb).mean(-1), 0.8)
    order = np.argsort(lin_b.ravel())
    dv = d.ravel()[order]
    report("anchored: monotone in blurred luminance (brighter -> thinner)",
           bool(np.all(np.diff(dv) <= 1e-9)), f"max_up={np.diff(dv).max():.2e}")

    # Plateau: constant background maps to a single dTop value (mid-rank).
    # Check the interior AWAY from the bright strip: the blur's reach is
    # ~3px, rows near the strip legitimately differ.
    plat = np.full((32, 32, 3), 90, np.uint8)
    plat[:4] = 200                                                # bright strip
    dp = anchored_dtop_field(plat, td_w=5.4, top_max=TOP)
    bg = dp[10:]
    report("anchored: plateau maps to ONE value (no gradient artifact)",
           float(bg.max() - bg.min()) < 1e-12,
           f"spread={bg.max() - bg.min():.2e}")

    # Guards: pure-color image -> constant mid field, no NaN.
    solid = np.full((8, 8, 3), 128, np.uint8)
    ds = anchored_dtop_field(solid, td_w=5.4, top_max=TOP)
    report("anchored: narrow-dynamic guard (constant, finite)",
           bool(np.all(np.isfinite(ds)) and ds.std() < 1e-12),
           f"std={ds.std():.2e}")

    # Pure black pixels must not produce -inf/NaN.
    blk = np.zeros((8, 8, 3), np.uint8)
    blk[0, 0] = 255
    db = anchored_dtop_field(blk, td_w=5.4, top_max=TOP)
    report("anchored: pure-black guard (finite)",
           bool(np.all(np.isfinite(db))), "")


def test_resolve_cmy_for_dtop():
    """Iteration 45 v2: CMY re-solve respects the pinned dTop band.

    M1: solver-chosen dTop sawtoothed across CMY lattice cells; the pinned
    re-solve must return card entries within |dTop - target| <= tol (or the
    band-nearest entry when the band is empty).
    """
    from litho_color import (build_gamut_stacked, resolve_cmy_for_dtop,
                             srgb8_to_lab)
    gamut = build_gamut_stacked(layers_max=3, layer_h=0.2, top_max=2.0,
                                dW=0.2, td=DEFAULT_TD)
    ramp = np.stack([np.linspace(255, 0, 128, dtype=np.uint8)] * 3, -1)[None]
    lab = srgb8_to_lab(ramp).reshape(-1, 3)
    target = np.linspace(0, 2.0, 128).reshape(1, 128)             # pinned ramp
    dC, dM, dY, dE, idx = resolve_cmy_for_dtop(lab, gamut, target,
                                               top_tol=0.10)
    t = gamut["thickness"]
    got = t[idx][..., 0]
    dev = np.abs(got - target)
    # Non-band picks are allowed only as band-nearest fallbacks; their dTop
    # deviation must stay within tol + half of the card top_step (0.08).
    report("resolve_cmy: |dTop_card - target| <= tol+half-step",
           float(dev.max()) <= 0.10 + 0.05,
           f"max_dev={dev.max():.3f}")
    report("resolve_cmy: dE finite and mostly small",
           bool(np.all(np.isfinite(dE)) and np.median(dE) < 15.0),
           f"med_dE={np.median(dE):.2f}")


def test_desalt_and_smooth_sigma():
    """Iteration 46: rank equalizer spreads residual noise to full amplitude
    (adversarial B1: +-1 gray level -> dTop std ~0.55 on flat+noise). The
    pre-rank blur (smooth_sigma) + targeted salt removal must cut isolated
    print-perceptible spikes WITHOUT touching real edges.
    """
    from scipy.ndimage import binary_erosion, generate_binary_structure
    from litho_color import anchored_dtop_field, desalt_isolated_spikes, \
        spike_surgery

    def iso_pct(f, thr=0.3):
        g = np.zeros(f.shape, bool)
        gx = np.abs(np.diff(f, axis=1))
        gy = np.abs(np.diff(f, axis=0))
        g[:-1, :-1] = np.maximum(gx[:-1, :], gy[:, :-1]) > thr
        st = generate_binary_structure(2, 2)
        return float((g & ~binary_erosion(g, structure=st,
                                          border_value=0)).mean() * 100)

    # Textured scene close to the real cartoon statistics: dark mass with
    # incoherent texture + bright mass (no exact plateaus).
    rng = np.random.default_rng(7)
    dark = rng.integers(20, 120, (64, 64))
    bright = rng.integers(170, 255, (16, 64))
    img = np.stack([np.concatenate([dark, bright], 0)] * 3, -1).astype(np.uint8)

    f_raw = anchored_dtop_field(img, td_w=5.4, top_max=2.0, smooth_sigma=0.0)
    f_new = anchored_dtop_field(img, td_w=5.4, top_max=2.0, smooth_sigma=0.8)
    f_new = desalt_isolated_spikes(f_new, thr=0.3, rounds=2)
    f_new = spike_surgery(f_new, t_sigma=1.5, iterations=2, top_max=2.0)
    raw_pct, new_pct = iso_pct(f_raw), iso_pct(f_new)
    # Measured on the real cartoon: 11.0% -> 7.5%; synthetic is harsher
    # (fully incoherent texture). Assert a clear reduction, not a magic no.
    report("desalt+blur: isolated spikes clearly reduced",
           new_pct < raw_pct * 0.75, f"{raw_pct:.1f}% -> {new_pct:.1f}%")

    # Real edges survive: a dark/bright step must keep a >0.5mm jump across
    # the (blur-widened) transition after the full finalize chain.
    step = np.full((32, 32, 3), 60, np.uint8)
    step[:, 16:, :] = 200
    f_step = anchored_dtop_field(step, td_w=5.4, top_max=2.0, smooth_sigma=0.8)
    f_step = desalt_isolated_spikes(f_step)
    f_step = spike_surgery(f_step, t_sigma=1.5, iterations=2, top_max=2.0)
    row = f_step[16]
    jump = float(np.max(row[:14]) - np.min(row[18:]))
    report("desalt+blur: real edge jump preserved (>0.5mm)",
           jump > 0.5, f"jump={jump:.2f}mm")

    # Plateau guard still holds under the blur: the plateau INTERIOR (away
    # from the bright strip's blur reach) stays a single value.
    plat = np.full((32, 32, 3), 90, np.uint8)
    plat[:4] = 200
    dp = anchored_dtop_field(plat, td_w=5.4, top_max=2.0, smooth_sigma=0.8)
    interior = dp[10:]
    report("anchored: plateau interior still ONE value under blur",
           float(interior.max() - interior.min()) < 1e-12,
           f"spread={interior.max() - interior.min():.2e}")

    # Known limitation (documented, NOT asserted as pass): pure flat+noise
    # input has no structure to keep — the rank field is uniform noise and
    # the blur only decorrelates it (B1 reproduction measured std ~0.55 ->
    # ~0.44). No false assertion here.


if __name__ == "__main__":
    test_color_space()
    gamut = test_gamut_and_solver()
    test_end_to_end(gamut)
    test_smooth_top_antispike()
    test_anchored_dtop_field()
    test_resolve_cmy_for_dtop()
    test_desalt_and_smooth_sigma()
    print()
    print(f"{sum(RESULTS)}/{len(RESULTS)} passed")
    sys.exit(0 if all(RESULTS) else 1)
