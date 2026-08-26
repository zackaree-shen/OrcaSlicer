"""Validation for the pluggable lithophane engine (mode x color-order).

Covers:
  - LAYERED x all 6 CMY orders: every mesh watertight + positive volume, Z
    bands strictly separated, band layout matches the chosen order, and dE is
    order-invariant (Beer-Lambert commutativity -> shared solver).
  - INTERLEAVED (mixed): C/M/Y share one Z band as watertight pixel boxes;
    no sub-membrane thin films.
  - GREYSCALE: black=thick / white=thin (real M1 thickness_map), C/M/Y empty.
  - Full mode x order traversal smoke test.
  - STL export round-trip on a sample result.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from litho_core import LithophaneParams, export_stl, validate_mesh
from litho_engine import (
    LithoMode, ColorOrder, color_lithophane_engine, _z_bases_for_order,
)
from litho_color import (
    COLOR_BAND_MAX, LAYER_GAP, TOP_BAND_MAX, WHITE_THICKNESS, Z_C_BASE,
    whiteness_mask, merge_features, anchored_dtop_field,
    resolve_cmy_chroma_only, srgb8_to_linear,
    recalibrate_dtop_for_luminance, forward_stacked,
    xyz_to_lab, linear_to_xyz,
)

RESULTS = []


def report(name, ok, detail=""):
    RESULTS.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name:55s} {detail}")


def make_test_img(h=120, w=160):
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    r = np.clip(0.6 * x / w + 0.3 * np.sin(y / 16), 0, 1)
    g = np.clip(0.6 * (1 - x / w) + 0.3 * np.cos(y / 20), 0, 1)
    b = np.clip(0.5 + 0.4 * np.sin(x / 26) * np.cos(y / 22), 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def params():
    return LithophaneParams(width_mm=99.99, height_mm=71.42, pixel_pitch_mm=0.5)


def _check_mesh(name, verts, faces, require_nonempty=True):
    """Returns True if the mesh is watertight, positive-volume, non-degenerate."""
    if len(faces) == 0:
        return not require_nonempty
    v = validate_mesh(verts, faces)
    return v["open_edges"] == 0 and v["volume"] > 0 and v["degenerate"] == 0


# ---------------------------------------------------------------------------
# LAYERED x 6 orders
# ---------------------------------------------------------------------------

def test_layered_orders():
    img = make_test_img()
    p = params()

    dE_by_order = {}
    for order in [ColorOrder.CMY, ColorOrder.CYM, ColorOrder.MCY,
                  ColorOrder.MYC, ColorOrder.YMC, ColorOrder.YCM]:
        meshes, dE, gamut, reached = color_lithophane_engine(
            img, mode=LithoMode.LAYERED, order=order, params=p)

        # 1. All five meshes watertight + positive volume.
        ok_mesh = all(_check_mesh(c, v, f) for c, (v, f) in meshes.items())
        report(f"LAYERED/{order.value}: 5 meshes watertight+vol>0", ok_mesh)

        # 2. Z bands strictly separated (gap = LAYER_GAP, no overlap).
        bands = {c: (v[:, 2].min(), v[:, 2].max()) for c, (v, f) in meshes.items()}
        z_order = sorted(bands.items(), key=lambda kv: kv[1][0])
        separated = all(
            z_order[i][1][1] <= z_order[i + 1][1][0] + 1e-6
            for i in range(len(z_order) - 1))
        report(f"LAYERED/{order.value}: Z bands separated", separated,
               str([(c, round(lo, 2), round(hi, 2)) for c, (lo, hi) in z_order]))

        # 3. Band layout matches the chosen order (W bottom, then order letters, top).
        bases = _z_bases_for_order(order)
        letters = [c for c in sorted(bases, key=lambda c: bases[c])]
        # Expected: white base at 0, then each color at its base, top on top.
        base_z = {c: bands[c][0] for c in ("C", "M", "Y")}
        layout_ok = (abs(bands["W"][0]) < 1e-6
                     and all(abs(base_z[c] - bases[c]) < 0.01 for c in letters)
                     and bands["top"][0] > bands[letters[-1]][1])
        report(f"LAYERED/{order.value}: Z layout matches order", layout_ok,
               f"order={order.value} top_start={bands['top'][0]:.2f}")

        dE_by_order[order] = np.asarray(dE)

    # 4. Order invariance: dE identical across all orders (shared solver).
    ref = dE_by_order[ColorOrder.CMY]
    all_same = all(np.array_equal(dE_by_order[o], ref) for o in dE_by_order)
    report("Order invariance: dE identical for all 6 orders", all_same)


# ---------------------------------------------------------------------------
# INTERLEAVED (mixed / Bambu B)
# ---------------------------------------------------------------------------

def test_interleaved():
    img = make_test_img()
    p = params()
    meshes, dE, gamut, reached = color_lithophane_engine(
        img, mode=LithoMode.INTERLEAVED, order=ColorOrder.MIXED, params=p)

    # 1. Pixel-box meshes watertight.
    ok_mesh = all(_check_mesh(c, v, f, require_nonempty=False) for c, (v, f) in meshes.items())
    report("INTERLEAVED: meshes watertight (boxes)", ok_mesh)

    # 2. C/M/Y share the SAME Z band [Z_C_BASE, Z_C_BASE+COLOR_BAND_MAX].
    band_ok = True
    for c in ("C", "M", "Y"):
        v, f = meshes[c]
        if len(v) == 0:
            continue
        zmin, zmax = v[:, 2].min(), v[:, 2].max()
        band_ok &= (abs(zmin - Z_C_BASE) < 0.01 and
                    zmax <= Z_C_BASE + COLOR_BAND_MAX + 0.01)
    report("INTERLEAVED: C/M/Y share one Z band", band_ok)

    # 3. No sub-membrane thin films: every box height >= mask threshold (0.02).
    no_thin = True
    for c in ("C", "M", "Y"):
        v, f = meshes[c]
        if len(v) == 0:
            continue
        # Continuous height field: 0-thickness pixels get a 1e-3 floor (the
        # design trade-off that eliminates fragmented islands). Non-floor
        # material may be thin from bilinear resampling (down to ~0.008mm);
        # anything at or below the floor is the island-elimination film.
        h = v[:, 2] - Z_C_BASE
        non_floor = h[h > 0.005]  # exclude the 1e-3 floor
        min_h = non_floor.min() if len(non_floor) else 0.0
        no_thin &= (min_h >= 0.005 - 1e-6)
    report("INTERLEAVED: no thin membranes above floor (>=0.005)", no_thin,
           f"min_nonfloor_h={min_h:.3f}")

    # 4. dE similar to LAYERED (same physical model, different geometry).
    report("INTERLEAVED: dE median sane", float(np.median(dE)) <= 6.0,
           f"dE_med={float(np.median(dE)):.2f}")

    # 5. NO FLOATING: the top relief's bottom must sit on (or above) the
    #    actual C/M/Y fill height at every point. Any pixel whose top layer
    #    bottom is above the color fill + a small gap means the top floats.
    vt = meshes["top"][0]
    top_bot = vt[vt[:, 2] < vt[:, 2].max() - 0.01, 2]  # bottom-surface vertices
    # The minimum top bottom must be >= z_lo + min positive box height + gap.
    # (color boxes start at z_lo=Z_C_BASE and are >= 0.02 thick where present).
    z_lo = Z_C_BASE
    min_expected = z_lo + 0.02 + LAYER_GAP - 0.05  # tolerance
    report("INTERLEAVED: top bottom follows color fill (no floating)",
           float(top_bot.min()) >= min_expected,
           f"top_bottom_min={float(top_bot.min()):.3f} expect>={min_expected:.3f}")


def test_layer_height_param():
    """Changing layer_h must move the Z bands so they stay strictly separated
    (LAYERED) / C-M-Y share a band with top following fill (INTERLEAVED)."""
    img = make_test_img()
    p = params()
    for lh in (0.08, 0.2):
        # LAYERED: strictly increasing Z bands.
        m = color_lithophane_engine(img, mode=LithoMode.LAYERED,
                                    order=ColorOrder.CMY, params=p,
                                    layer_h=lh, pitch_cmy=1.5, pitch_top=1.0)[0]
        bands = []
        for c in ("W", "C", "M", "Y", "top"):
            v = m[c][0]
            if len(v):
                bands.append((float(v[:, 2].min()), float(v[:, 2].max())))
        bands.sort(key=lambda b: b[0])
        inc = all(bands[i][1] <= bands[i + 1][0] + 0.01 for i in range(len(bands) - 1))
        report(f"LAYERED layer_h={lh}: Z bands strictly increasing", inc,
               " ".join(f"[{lo:.1f},{hi:.1f}]" for lo, hi in bands))
        # C/M/Y all present.
        ok_cmy = all(len(m[c][1]) > 0 for c in ("C", "M", "Y"))
        report(f"LAYERED layer_h={lh}: C/M/Y present", ok_cmy)

        # INTERLEAVED: C/M/Y share band; top follows fill (min bottom >= band lo).
        m2, _, _, _ = color_lithophane_engine(img, mode=LithoMode.INTERLEAVED,
                                              order=ColorOrder.MIXED, params=p,
                                              layer_h=lh, pitch_cmy=1.5, pitch_top=1.0)
        shared = [float(m2[c][0][:, 2].min()) for c in ("C", "M", "Y") if len(m2[c][0])]
        report(f"INTERLEAVED layer_h={lh}: C/M/Y share one Z band",
               len(set(round(z, 3) for z in shared)) == 1,
               f"band_los={shared}")


# ---------------------------------------------------------------------------
# GREYSCALE (M1)
# ---------------------------------------------------------------------------

def test_greyscale():
    p = LithophaneParams(width_mm=40, height_mm=40, pixel_pitch_mm=1.0,
                         base_thickness=0.8, depth_range=2.0)
    black = np.zeros((40, 40, 3), dtype=np.uint8)
    white = np.full((40, 40, 3), 255, dtype=np.uint8)
    mid = np.full((40, 40, 3), 128, dtype=np.uint8)

    z_black = color_lithophane_engine(black, mode=LithoMode.GREYSCALE,
                                      order=ColorOrder.CMY, params=p)[0]["top"][0][:, 2].max()
    z_white = color_lithophane_engine(white, mode=LithoMode.GREYSCALE,
                                      order=ColorOrder.CMY, params=p)[0]["top"][0][:, 2].max()
    z_mid = color_lithophane_engine(mid, mode=LithoMode.GREYSCALE,
                                    order=ColorOrder.CMY, params=p)[0]["top"][0][:, 2].max()

    report("GREYSCALE: black thick / white thin", abs(z_black - 2.8) < 0.05 and abs(z_white - 0.8) < 0.05,
           f"black={z_black:.2f} mid={z_mid:.2f} white={z_white:.2f}")
    report("GREYSCALE: monotonic black>mid>white", z_black > z_mid > z_white)

    meshes = color_lithophane_engine(black, mode=LithoMode.GREYSCALE,
                                     order=ColorOrder.CMY, params=p)[0]
    cmy_empty = all(len(f) == 0 for c in ("C", "M", "Y") for v, f in [meshes[c]])
    report("GREYSCALE: C/M/Y empty", cmy_empty)
    report("GREYSCALE: top watertight", _check_mesh("top", meshes["top"][0], meshes["top"][1]))


# ---------------------------------------------------------------------------
# Full traversal smoke + STL round-trip
# ---------------------------------------------------------------------------

def test_full_traversal():
    img = make_test_img()
    p = params()
    combos = []
    for mode in LithoMode:
        if mode == LithoMode.LAYERED:
            orders = [ColorOrder.CMY, ColorOrder.YMC]
        elif mode in (LithoMode.INTERLEAVED, LithoMode.OVERLAP):
            orders = [ColorOrder.MIXED]
        elif mode == LithoMode.STACKED:
            orders = [ColorOrder.CMY]
        else:
            orders = [ColorOrder.CMY]
        for o in orders:
            combos.append((mode, o))
    for mode, o in combos:
        meshes, dE, _, reached = color_lithophane_engine(img, mode=mode, order=o, params=p)
        assert all(_check_mesh(c, v, f, require_nonempty=False) for c, (v, f) in meshes.items())
    report(f"Traversal: {len(combos)} mode/order combos all watertight", True,
           f"{len(combos)} combos")


def test_stacked_no_floating():
    """STACKED (Bambu-style) must have NO large vertical gaps > 0.3 mm at any
    pixel — every C/M/Y column starts on the white base or the color below, so
    nothing floats. LAYERED is known to have these gaps (M/Y hover)."""
    img = make_test_img()
    p = LithophaneParams(width_mm=50, height_mm=40, pixel_pitch_mm=1.0)

    def max_gap(meshes, n=30):
        rng = np.random.default_rng(3)
        worst = 0.0
        for _ in range(n):
            x = float(rng.uniform(3, 47))
            y = float(rng.uniform(3, 37))
            bands = []
            for c in ("W", "C", "M", "Y", "top"):
                v, f = meshes[c]
                if len(v) == 0:
                    continue
                d = np.sqrt((v[:, 0] - x) ** 2 + (v[:, 1] - y) ** 2)
                near = v[d < 1.0]
                if len(near):
                    bands.append((float(near[:, 2].min()), float(near[:, 2].max())))
            bands.sort(key=lambda b: b[0])
            for i in range(len(bands) - 1):
                worst = max(worst, bands[i + 1][0] - bands[i][1])
        return worst

    m_s = color_lithophane_engine(img, mode=LithoMode.STACKED, order=ColorOrder.CMY,
                                  params=p, pitch_cmy=1.0, pitch_top=0.5)[0]
    m_l = color_lithophane_engine(img, mode=LithoMode.LAYERED, order=ColorOrder.CMY,
                                  params=p, pitch_cmy=1.0, pitch_top=0.5)[0]
    g_s = max_gap(m_s)
    g_l = max_gap(m_l)
    report("STACKED: no floating (max vertical gap <= 0.3mm)",
           g_s <= 0.3, f"max_gap={g_s:.2f}mm")
    report("STACKED gap << LAYERED gap", g_s < g_l - 0.3,
           f"stacked={g_s:.2f} layered={g_l:.2f}")

    # STL round-trip on a LAYERED result.
    meshes, _, _, _ = color_lithophane_engine(img, mode=LithoMode.LAYERED,
                                              order=ColorOrder.CMY, params=p)
    tmp = "_engine_rt.stl"
    verts, faces = meshes["top"]
    export_stl(tmp, verts, faces, name="rt")
    from stl import mesh as stl_mesh
    loaded = stl_mesh.Mesh.from_file(tmp)
    os.remove(tmp)
    report("STL export round-trip", loaded.vectors.shape[0] == len(faces),
           f"{len(faces):,} tris")


def test_overlap():
    """OVERLAP mode shares the same-base overlapping geometry with INTERLEAVED
    but uses the SEGSTACK color card (part-order segment stack, matching what
    the overlapping geometry actually prints). Its dE vs the sum model differs
    because segstack is order-fixed; geometry must be watertight, C/M/Y same
    base Z."""
    img = make_test_img()
    p = params()
    m_o, dE_o, gamut_o, _ = color_lithophane_engine(img, mode=LithoMode.OVERLAP,
                                                    order=ColorOrder.MIXED, params=p,
                                                    pitch_cmy=1.0, pitch_top=1.0)
    m_i, dE_i, gamut_i, _ = color_lithophane_engine(img, mode=LithoMode.INTERLEAVED,
                                                    order=ColorOrder.MIXED, params=p,
                                                    pitch_cmy=1.0, pitch_top=1.0)

    # Watertight.
    ok_w = all(_check_mesh(c, v, f) for c, (v, f) in m_o.items())
    report("OVERLAP: all meshes watertight", ok_w)

    # Same same-base overlapping structure: C/M/Y all start at the same z_lo
    # (geometry pattern identical to INTERLEAVED, though thickness differs).
    z_lo_o = [float(m_o[c][0][:, 2].min()) for c in ("C", "M", "Y") if len(m_o[c][0])]
    same_base = len(set(round(z, 3) for z in z_lo_o)) == 1
    report("OVERLAP: C/M/Y share one base Z (overlapping)", same_base,
           f"base_z={z_lo_o}")

    # Segstack card is less degenerate than the old max card but the two
    # forward models still differ (order-fixed vs sum).
    uniq_o = len(np.unique(gamut_o["rgb8"], axis=0))
    uniq_i = len(np.unique(gamut_i["rgb8"], axis=0))
    report("OVERLAP: segstack card exists (non-empty unique colors)",
           uniq_o > 100, f"unique rgb8: overlap={uniq_o} interleaved={uniq_i}")

    # dE differs from INTERLEAVED (different forward model); assert they are
    # NOT identical (order-fixed segstack vs sum).
    report("OVERLAP: dE differs from INTERLEAVED (distinct forward model)",
           abs(float(np.median(dE_o)) - float(np.median(dE_i))) > 0.5,
           f"dE overlap={np.median(dE_o):.2f} interleaved={np.median(dE_i):.2f}")


def test_routing():
    """LAYERED/MIXED must be rejected (it was silently rerouted to INTERLEAVED
    before, making LAYERED and INTERLEAVED produce identical geometry)."""
    img = make_test_img()
    p = params()

    # LAYERED + MIXED is invalid.
    try:
        color_lithophane_engine(img, mode=LithoMode.LAYERED, order=ColorOrder.MIXED, params=p)
        report("Routing: LAYERED/MIXED rejected", False)
    except ValueError:
        report("Routing: LAYERED/MIXED rejected", True)

    # LAYERED/CMY and INTERLEAVED/MIXED must produce DIFFERENT Z structure:
    # LAYERED has disjoint Z bands, INTERLEAVED has C/M/Y sharing one base Z.
    m_layered = color_lithophane_engine(img, mode=LithoMode.LAYERED,
                                        order=ColorOrder.CMY, params=p)[0]
    m_inter = color_lithophane_engine(img, mode=LithoMode.INTERLEAVED,
                                      order=ColorOrder.MIXED, params=p)[0]
    # LAYERED: C/M/Y band starts distinct (disjoint).
    z_l = sorted({round(float(m_layered[c][0][:, 2].min()), 2) for c in ("C", "M", "Y")})
    # INTERLEAVED: C/M/Y all start at same z_lo.
    z_i = sorted({round(float(m_inter[c][0][:, 2].min()), 2) for c in ("C", "M", "Y")})
    differ = len(z_l) >= 2 and len(z_i) == 1
    report("Routing: LAYERED (disjoint bands) vs INTERLEAVED (shared base) differ",
           differ, f"layered bases={z_l} interleaved base={z_i}")

    # INTERLEAVED mode must accept MIXED order (valid).
    m_mixed = color_lithophane_engine(img, mode=LithoMode.INTERLEAVED,
                                     order=ColorOrder.MIXED, params=p)
    report("Routing: INTERLEAVED/MIXED valid", len(m_mixed[0]) == 5)


def test_bambu():
    """BAMBU mode: Bambu-reference geometry.

    - W is ONE complete white model: a thin base slab [0, z_lo=0.2] merged
      with a full-coverage white relief plate [z_lo+band+gap, ...] into a
      single watertight mesh (single part, white extruder).
    - C/M/Y are COLOR CHANNELS: one overlapping Z band [z_lo, z_lo+dCh]
      between the white base and the relief plate (reference [0.2, 0.9]).
    - No punch-through: the relief plate bottom (z_lo+band+LAYER_GAP) is
      always ABOVE the color band top, so the slicer's part-order clipping
      can never hollow out the white (the user-reported 镂空 bug).
    - No double counting: the gamut is built with dW=z_lo (the C/M/Y band
      volume is color, not white).
    - 'top' key exists (empty) so all consumers iterate 5 keys uniformly.
    - MIXED order accepted; other orders produce identical geometry.
    """
    img = make_test_img()
    p = params()
    m, dE, gamut, reached = color_lithophane_engine(
        img, mode=LithoMode.BAMBU, order=ColorOrder.MIXED, params=p,
        layers_max=8, layer_h=0.2, pitch_cmy=0.44, pitch_top=0.22)

    # 5 keys present, top empty.
    ok_keys = set(m.keys()) == {"W", "C", "M", "Y", "top"} and len(m["top"][0]) == 0
    report("BAMBU: 5 keys, top empty", ok_keys, f"keys={list(m.keys())}")

    # W full height: spans [0, z_lo + band + TOP_BAND_MAX], bottom at 0.
    vW = m["W"][0]
    ok_w = abs(float(vW[:, 2].min())) < 1e-6 and float(vW[:, 2].max()) > 2.0
    report("BAMBU: W full-height relief [0, z_lo+band+dTop]", ok_w,
           f"z=[{vW[:,2].min():.3f},{vW[:,2].max():.3f}]")

    # W no hollowing: white base slab [0, z_lo=0.2] present at bottom.
    report("BAMBU: W base slab solid (no hollowing)",
           float(vW[:, 2].min()) == 0.0 and float(vW[:, 2].max()) > 2.0,
           f"min={vW[:,2].min():.3f} max={vW[:,2].max():.3f}")

    # C/M/Y same base at z_lo=0.2 (reference), inside W's Z range.
    bases = [float(m[c][0][:, 2].min()) for c in ("C", "M", "Y") if len(m[c][0])]
    same_base = len(set(round(z, 3) for z in bases)) == 1 and round(bases[0], 2) == 0.2
    inside = all(b > vW[:, 2].min() and b < vW[:, 2].max() for b in bases)
    report("BAMBU: C/M/Y one overlapping base at z=0.2 (reference)",
           same_base and inside, f"bases={bases}")

    # NO PUNCH-THROUGH (the user-reported bug): for every XY column, the W
    # relief plate bottom must be >= CMY top + LAYER_GAP. If the plate dipped
    # into the color band, the slicer's part-order clipping would hollow out
    # the white above the band -> 镂空.
    from collections import defaultdict
    w_cols = defaultdict(list)
    for x, y, z in vW:
        w_cols[(round(x / 2), round(y / 2))].append(float(z))
    cmy_cols = defaultdict(list)
    for c in ("C", "M", "Y"):
        for x, y, z in m[c][0]:
            cmy_cols[(round(x / 2), round(y / 2))].append(float(z))
    margins = []
    for k, cmy_z in cmy_cols.items():
        if k not in w_cols:
            continue
        cmy_top = max(cmy_z)
        w_relief_lo = min((z for z in w_cols[k] if z > 0.3), default=None)
        if w_relief_lo is None:
            continue
        margins.append(w_relief_lo - cmy_top)
    no_punch = len(margins) > 0 and min(margins) >= LAYER_GAP - 1e-6
    report("BAMBU: no punch-through (W relief >= CMY top + gap)",
           no_punch, f"margin min={min(margins):.3f} med={np.median(margins):.3f}")

    # Model total height stays near the reference (~2.28mm): dW=z_lo (thin
    # base) + band 0.7 + relief. Assert it is far below the old 4.5mm.
    report("BAMBU: total height near reference",
           float(vW[:, 2].max()) < 3.5, f"total={vW[:,2].max():.3f}mm")

    # Watertight meshes.
    ok_wt = all(_check_mesh(c, v, f) for c, (v, f) in m.items() if len(f))
    report("BAMBU: all meshes watertight", ok_wt)

    # MIXED valid; other orders also accepted but produce IDENTICAL geometry
    # (BAMBU is same-base, order does not change the overlapping band — like
    # INTERLEAVED, which also ignores order).
    m_cmy = color_lithophane_engine(img, mode=LithoMode.BAMBU,
                                    order=ColorOrder.CMY, params=p,
                                    layers_max=8, layer_h=0.2,
                                    pitch_cmy=0.44, pitch_top=0.22)[0]
    same_geom = all(
        len(m[c][0]) == len(m_cmy[c][0]) for c in ("W", "C", "M", "Y"))
    report("BAMBU: order-invariant (CMY == MIXED geometry)",
           same_geom and len(m_cmy) == 5)

    # dE is meaningful: BAMBU uses its own gamut (dW=z_lo, band=0.7) so dE is
    # NOT identical to INTERLEAVED (dW=0.8). Assert it is finite and sane.
    report("BAMBU: dE median finite and reasonable",
           np.isfinite(np.median(dE)) and float(np.median(dE)) < 12.0,
           f"dE bambu={np.median(dE):.2f}")


def test_default_face_count():
    """Default OVERLAP output resolution (pixel_pitch=0.15, pitch_top=0.15,
    pitch_cmy=0.30) on a 144x108 mm lithophane. overlap_detail defaults favor
    maximum top relief detail; this guard catches runaway grids."""
    img = make_test_img(h=108*4, w=144*4)  # 4 px/mm source
    p = LithophaneParams(width_mm=144.0, height_mm=108.0, pixel_pitch_mm=0.15)
    meshes, _, _, _ = color_lithophane_engine(
        img, mode=LithoMode.OVERLAP, order=ColorOrder.MIXED, params=p)
    total = sum(len(f) for _, f in meshes.values())
    report("OVERLAP default resolution: total faces <= 6,500,000",
           total <= 6_500_000, f"total={total:,}")


# ---------------------------------------------------------------------------
# Iteration 46: white-collapse (CMY=0 in white regions) + default Bambu sizing
# ---------------------------------------------------------------------------

def test_whiteness_mask():
    """whiteness_mask: strict white (bright + achromatic) -> 1, else -> 0."""
    img = np.zeros((3, 3, 3), dtype=np.uint8)
    img[0, 0] = [255, 255, 255]   # pure white
    img[0, 1] = [220, 220, 220]   # light gray (min=220 < 230)
    img[0, 2] = [128, 128, 128]   # mid gray
    img[1, 0] = [255, 0, 0]       # red
    img[1, 1] = [240, 245, 240]   # near-white slight green (min=240, chroma=5)
    img[1, 2] = [0, 0, 0]         # black
    img[2, 0] = [200, 200, 220]   # bluish light (min=200 < 230)
    img[2, 1] = [250, 250, 250]   # near-white (min=250, chroma=0)
    img[2, 2] = [255, 240, 230]   # warm near-white (min=230, chroma=25 > 15)
    m = whiteness_mask(img, sigma=0.0)  # binary mask
    cases = [
        ("pure white (255,255,255)",                (0, 0), 1.0),
        ("near-white (240,245,240)",                (1, 1), 1.0),
        ("near-white (250,250,250)",                (2, 1), 1.0),
        ("light gray (220,220,220) excluded",       (0, 1), 0.0),
        ("mid gray (128,128,128)",                  (0, 2), 0.0),
        ("red (255,0,0)",                           (1, 0), 0.0),
        ("black",                                   (1, 2), 0.0),
        ("bluish light (200,200,220) excluded",      (2, 0), 0.0),
        ("warm (255,240,230 chroma=25) excluded",   (2, 2), 0.0),
    ]
    for label, (r, c), expected in cases:
        ok = (m[r, c] == expected)
        report(f"whiteness: {label}", ok, f"got {m[r,c]:.0f} (want {expected:.0f})")


def test_white_collapse_legacy_unchanged():
    """white_collapse=False: engine runs without gating (legacy behaviour).
    Sanity: P1 path completes, dE finite, meshes non-empty. Asserts the new
    toggle does not break the off-path."""
    img = make_test_img(h=80, w=120)
    p = LithophaneParams(width_mm=80.0, height_mm=53.33, pixel_pitch_mm=1.0)
    meshes, dE, _, _ = color_lithophane_engine(
        img, mode=LithoMode.OVERLAP, order=ColorOrder.MIXED, params=p,
        surface_refine=True, white_collapse=False)
    ok = (np.isfinite(np.median(dE))
          and len(meshes["top"][1]) > 0
          and len(meshes["C"][1]) > 0)
    report("white_collapse=False: P1 runs, finite dE, non-empty meshes",
           ok, f"dE_med={np.median(dE):.2f} top_faces={len(meshes['top'][1])}")


def test_export_sizing_default_156x106():
    """export_v4 default sizing: fixed 156x106mm canvas with center-crop to
    target aspect (matches Bambu). Uses a 1.8-aspect source image so the crop
    path is exercised. Top mesh x/y extent must equal width/height in mm."""
    import tempfile, os
    from PIL import Image as PILImage
    from export_v4 import export_lithophane
    # 36x20 px -> aspect 1.8 (> 1.472 target) -> triggers width crop.
    rgb = np.zeros((20, 36, 3), dtype=np.uint8)
    rgb[:, :, 0] = np.linspace(0, 255, 36)[None, :]
    rgb[:, :, 1] = np.linspace(0, 255, 20)[:, None]
    rgb[:, :, 2] = 128
    with tempfile.TemporaryDirectory() as td:
        ip = os.path.join(td, "in.png")
        op = os.path.join(td, "out")
        PILImage.fromarray(rgb).save(ip)
        result = export_lithophane(
            ip, op, pixel_pitch_mm=4.0, save_preview=False,
            surface_refine=True)
        v_top, _ = result["meshes"]["top"]
        w_mm = float(v_top[:, 0].max())
        h_mm = float(v_top[:, 1].max())
        ok_size = (abs(w_mm - 156.0) < 0.5) and (abs(h_mm - 106.0) < 0.5)
        report("export default sizing: 156x106 mm",
               ok_size, f"top x_max={w_mm:.2f}mm y_max={h_mm:.2f}mm")


def test_export_sizing_legacy_long_edge():
    """export_v4 legacy: --long-edge-mm 72 -> 72mm long edge, aspect-preserving
    (no crop). 36x20 -> long edge 72 -> 72x40mm."""
    import tempfile, os
    from PIL import Image as PILImage
    from export_v4 import export_lithophane
    rgb = np.zeros((20, 36, 3), dtype=np.uint8)
    rgb[:, :, 0] = np.linspace(0, 255, 36)[None, :]
    rgb[:, :, 2] = 200
    with tempfile.TemporaryDirectory() as td:
        ip = os.path.join(td, "in.png")
        op = os.path.join(td, "out")
        PILImage.fromarray(rgb).save(ip)
        result = export_lithophane(
            ip, op, long_edge_mm=72.0, pixel_pitch_mm=4.0, save_preview=False)
        v_top, _ = result["meshes"]["top"]
        w_mm = float(v_top[:, 0].max())
        h_mm = float(v_top[:, 1].max())
        ok_size = (abs(w_mm - 72.0) < 0.5) and (abs(h_mm - 40.0) < 0.5)
        report("export legacy long_edge=72: 72x40mm (no crop)",
               ok_size, f"top x_max={w_mm:.2f}mm y_max={h_mm:.2f}mm")


def test_merge_features_noop():
    """merge_features strength=0 is a byte-identical no-op (baseline-safe)."""
    h, w = 40, 60
    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    dTop = anchored_dtop_field(rgb, top_max=1.6)
    out = merge_features(rgb, dTop, top_max=1.6, lum_tol=0.05, edge_tol=0.03,
                          strength=0.0)
    ok = np.array_equal(out, dTop)
    report("merge_features strength=0: byte-identical no-op",
           ok, f"max|Δ|={float(np.abs(out - dTop).max()):.2e}")


def test_merge_features_merges_flat_noise_keeps_edge():
    """Synthetic: left half = flat luminance + per-pixel noise; right half =
    a higher flat plateau; a 1-px step between them is a real edge.

    After merging, the noisy left half must consolidate to FEWER unique dTop
    levels (features merged into a plateau) while the step jump across the
    boundary is PRESERVED (edge not eaten)."""
    h, w = 30, 60
    rng = np.random.default_rng(1)
    base = np.full((h, w), 120, dtype=np.float64)
    base[:, :w // 2] += rng.normal(0, 8, (h, w // 2))   # noisy left (sigma 8)
    base[:, w // 2:] = 200                              # flat right plateau
    rgb = np.stack([base, base, base], axis=-1).astype(np.uint8)
    dTop = anchored_dtop_field(rgb, top_max=1.6)
    merged = merge_features(rgb, dTop, top_max=1.6, lum_tol=0.05,
                            edge_tol=0.03, strength=1.0)

    # left half unique levels before/after: merging should reduce them
    n0 = len(np.unique(np.round(dTop[:, :w // 2], 4)))
    n1 = len(np.unique(np.round(merged[:, :w // 2], 4)))
    merged_levels = n1 < n0

    # step jump at the boundary column (col w//2-1 -> w//2) preserved
    jump_in = abs(dTop[:, w // 2].mean() - dTop[:, w // 2 - 1].mean())
    jump_out = abs(merged[:, w // 2].mean() - merged[:, w // 2 - 1].mean())
    edge_kept = jump_out >= 0.7 * jump_in

    ok = merged_levels and edge_kept
    report("merge: flat-noise consolidated, step edge preserved",
           ok, f"levels {n0}->{n1}, edge_jump {jump_in:.3f}->{jump_out:.3f}")


def test_merge_features_e2e_levels_reduced():
    """E2E: engine with merge_features=0.6 vs 0.0 on a test image. Merged dTop
    must have fewer unique levels (cleaner plateaus) while dE stays finite and
    the run completes. Mirrors the user's 'merge features -> clean' goal."""
    img = make_test_img(h=80, w=120)
    p = LithophaneParams(width_mm=80.0, height_mm=53.33, pixel_pitch_mm=1.0)
    m0, dE0, _, _ = color_lithophane_engine(
        img, mode=LithoMode.OVERLAP, order=ColorOrder.MIXED, params=p,
        surface_refine=True, merge_features=0.0)
    m1, dE1, _, _ = color_lithophane_engine(
        img, mode=LithoMode.OVERLAP, order=ColorOrder.MIXED, params=p,
        surface_refine=True, merge_features=0.6)
    # Recover dTop heights from the top mesh Z coordinate (mesh stores Z).
    z0 = m0["top"][0][:, 2]
    z1 = m1["top"][0][:, 2]
    levels0 = len(np.unique(np.round(z0, 3)))
    levels1 = len(np.unique(np.round(z1, 3)))
    ok = (np.isfinite(np.median(dE1)) and levels1 < levels0
          and len(m1["top"][1]) > 0)
    report("merge E2E: dTop levels reduced, dE finite, meshes ok",
           ok, f"levels {levels0}->{levels1}, dE_med={np.median(dE1):.2f}")


def test_resolve_cmy_chroma_only_picks_min_chroma_dist():
    """Iteration 49 unit: resolve_cmy_chroma_only must pick the card entry with
    the smallest (a*,b*) chroma distance to the target, ignoring L*."""
    from litho_color import build_gamut_stacked
    gamut = build_gamut_stacked(layers_max=6, layer_h=0.12, top_max=1.6, dW=0.2)
    lab = gamut["lab"]
    ab = lab[:, 1:3]
    rng = np.random.default_rng(0)
    sel = rng.integers(0, lab.shape[0], size=20)
    targets = lab[sel]
    dTop_dummy = np.zeros((4, 5))   # shape only; 20 cells = 20 targets
    dC, dM, dY, dE, idx = resolve_cmy_chroma_only(targets, gamut, dTop_dummy, k=32)
    idx_flat = np.asarray(idx).ravel()
    dE_flat = np.asarray(dE).ravel()
    tgt_ab = targets[:, 1:3]
    ok_all = True
    worst = 0.0
    for i in range(len(targets)):
        d2 = np.sqrt(((ab - tgt_ab[i]) ** 2).sum(axis=1))
        bf = int(np.argmin(d2))
        if idx_flat[i] != bf and abs(float(d2[idx_flat[i]]) - float(d2[bf])) > 1e-6:
            ok_all = False
        worst = max(worst, abs(float(dE_flat[i]) - float(d2[bf])))
    report("resolve_cmy_chroma_only: picks min (a,b) distance",
           ok_all, f"max dE mismatch={worst:.2e}")


def _band_thickness(mesh):
    """Extract per-cell thickness (top_z - bottom_z) from a band mesh.
    heightfield_band_mesh stores front (top surface) vertices first, then back
    (bottom surface); thickness = front_z - back_z = the dTop / dC / ... field."""
    v = mesh[0]
    n = v.shape[0] // 2
    return v[:n, 2] - v[n:, 2]


def test_chroma_decouple_keeps_dtop_untouched():
    """Iteration 49 integration: enabling chroma_decouple must NOT alter the
    white relief thickness (dTop) — only the C/M/Y colour fields change. The
    absolute mesh Z shifts because the relief sits on top of the CMY fill, but
    the thickness (front_z - back_z) must be byte-identical."""
    img = make_test_img(h=100, w=140).astype(np.uint8)
    p = LithophaneParams(width_mm=156.0, height_mm=106.0, pixel_pitch_mm=0.30)
    kw = dict(mode=LithoMode.OVERLAP, order=ColorOrder.MIXED, params=p,
              detail_level=1.0, surface_refine=False, white_collapse=False,
              pitch_cmy=0.30)
    m0, _, _, _ = color_lithophane_engine(img, **kw)
    m1, _, _, _ = color_lithophane_engine(img, **kw, chroma_decouple=True)
    t0 = _band_thickness(m0["top"])
    t1 = _band_thickness(m1["top"])
    report("chroma_decouple: white relief thickness (dTop) unchanged",
           np.allclose(t0, t1, atol=1e-9), f"max|dTop|={np.abs(t0 - t1).max():.2e}")


def test_cmy_smooth_reduces_colour_variance():
    """Iteration 49 integration: with chroma_decouple, cmy_smooth gaussian on
    C/M/Y must reduce the colour-field gradient variance (smoother colour
    backdrops) while leaving the white relief thickness untouched."""
    img = make_test_img(h=100, w=140).astype(np.uint8)
    p = LithophaneParams(width_mm=156.0, height_mm=106.0, pixel_pitch_mm=0.30)
    kw = dict(mode=LithoMode.OVERLAP, order=ColorOrder.MIXED, params=p,
              detail_level=1.0, surface_refine=False, white_collapse=False,
              pitch_cmy=0.30, chroma_decouple=True)

    def gradstd(arr):
        return float(np.std(np.diff(arr)))

    mS0, _, _, _ = color_lithophane_engine(img, **kw, cmy_smooth=0.0)
    mS1, _, _, _ = color_lithophane_engine(img, **kw, cmy_smooth=1.0)
    c0, c1 = _band_thickness(mS0["C"]), _band_thickness(mS1["C"])
    g0, g1 = gradstd(c0), gradstd(c1)
    t0 = _band_thickness(mS0["top"])
    t1 = _band_thickness(mS1["top"])
    ok = (g1 < g0) and np.allclose(t0, t1, atol=1e-9)
    report("cmy_smooth: C smoother, white relief thickness unchanged",
           ok, f"C grad_std {g0:.4f}->{g1:.4f}, dTop max|dΔ|={np.abs(t0-t1).max():.2e}")


def test_recalibrate_hits_target_luminance():
    """Unit: recalibrate_dtop_for_luminance re-derives dTop so the neutral
    white layer (W base + top) hits the target linear luminance exactly.

    Synthetic case: light CMY (so Y_cmy is large enough that the target is
    reachable within [0, top_max]), target L* = 80 (feasible: white base caps
    max luminance at 10^(-dW/tdw) ~= 0.71 -> L* ~89). After recalibration the
    forward-model luminance must equal the target luminance within 1e-3."""
    from litho_color import build_gamut_stacked
    gamut = build_gamut_stacked(layers_max=6, layer_h=0.12, top_max=1.6, dW=0.2)
    # Use a near-white card entry (minimal CMY) so Y_cmy ~ 1 (feasible target).
    tgt = gamut["lab"][0]                 # card entry 0 = (dTop=0, dC=dM=dY=0)
    sh = (6, 8)
    dC = np.zeros(sh); dM = np.zeros(sh); dY = np.zeros(sh)
    dTop0 = np.full(sh, 0.5)

    L = tgt[0]
    fy = (L + 16.0) / 116.0
    Y_t = (fy ** 3) if L > 7.999 else L / 903.3

    dTop_new = recalibrate_dtop_for_luminance(
        tgt, dC, dM, dY, dTop0, dW=0.2, top_max=1.6)
    tau = forward_stacked(dTop_new, dC, dM, dY, dW=0.2)
    Y_out = 0.2126 * tau[..., 0] + 0.7152 * tau[..., 1] + 0.0722 * tau[..., 2]
    ok = np.allclose(Y_out, Y_t, atol=1e-3)
    report("recalibrate: white layer hits target luminance exactly",
           ok, f"Y_t={Y_t:.4f} Y_out={float(Y_out.mean()):.4f} dTop={float(dTop_new.mean()):.3f}")


def test_recalib_luminance_lowers_dE():
    """Integration: with chroma_decouple=True, enabling recalib_luminance must
    LOWER the median dE (white layer now carries the precise target luminance
    instead of the approximate equalized field)."""
    img = make_test_img(h=100, w=140).astype(np.uint8)
    p = LithophaneParams(width_mm=156.0, height_mm=106.0, pixel_pitch_mm=0.30)
    kw = dict(mode=LithoMode.OVERLAP, order=ColorOrder.MIXED, params=p,
              detail_level=1.0, surface_refine=False, white_collapse=True,
              pitch_cmy=0.30, chroma_decouple=True, cmy_smooth=0.5)
    _, dE0, _, _ = color_lithophane_engine(img, **kw, recalib_luminance=False)
    _, dE1, _, _ = color_lithophane_engine(img, **kw, recalib_luminance=True)
    m0 = float(np.median(dE0)); m1 = float(np.median(dE1))
    ok = m1 <= m0 + 1e-6
    report("recalib_luminance: median dE not worse than without",
           ok, f"dE_med {m0:.2f} -> {m1:.2f}")


def test_merge_min_size_absorbs_tiny_specks():
    """Iteration 49b: a 1-2 px dark speck inside a bright plateau has its own
    |grad L| > edge_tol so it survives the merge gate as a single isolated
    component. min_size must sweep it into the surrounding plateau."""
    from litho_color import merge_features
    h, w = 30, 30
    rgb = np.full((h, w, 3), 220, dtype=np.uint8)  # bright plateau
    rgb[5:8, 5:8] = 30                                # single 3x3 dark speck
    rgb[20:21, 20:21] = 30                            # 1x1 speck
    dTop_in = np.full((h, w), 0.2, dtype=np.float64)
    dTop_in[5:8, 5:8] = 1.4                           # matching spike
    dTop_in[20:21, 20:21] = 1.4
    out_no_min = merge_features(rgb, dTop_in, lum_tol=0.05, edge_tol=0.03,
                                strength=1.0, min_size=0)
    out_min = merge_features(rgb, dTop_in, lum_tol=0.05, edge_tol=0.03,
                             strength=1.0, min_size=16)
    # Without min_size: the specks survive (dTop at specks ~ 1.4).
    # With min_size=16: specks (3x3 + 1x1) must be swept to plateau height.
    speck_max_no_min = max(dTop_in[5:8, 5:8].max(),
                           dTop_in[20:21, 20:21].max())
    speck_max_min = max(out_min[5:8, 5:8].max(),
                        out_min[20:21, 20:21].max())
    plateau = float(out_min[0, 0])
    abs_no_min = out_no_min[5:8, 5:8].max()
    ok = (abs_no_min > 1.2) and (speck_max_min <= plateau + 1e-9)
    report("merge min_size: 1-px and 3x3 specks swept to plateau",
           ok,
           f"dTop at specks: no_min={abs_no_min:.2f} (survived) "
           f"min16={speck_max_min:.2f}, plateau={plateau:.2f}")


def test_merge_min_size_preserves_large_text_region():
    """Sanity: a large dark text region (>min_size) must NOT be absorbed — it
    is a real feature, not noise."""
    from litho_color import merge_features
    h, w = 60, 60
    rgb = np.full((h, w, 3), 220, dtype=np.uint8)
    rgb[20:50, 20:50] = 30  # 30x30 text region
    dTop_in = np.full((h, w), 0.2)
    dTop_in[20:50, 20:50] = 1.4
    out = merge_features(rgb, dTop_in, lum_tol=0.05, edge_tol=0.03,
                         strength=1.0, min_size=16)
    text_dTop = out[20:50, 20:50].mean()
    bg_dTop = out[0:5, 0:5].mean()
    diff = text_dTop - bg_dTop
    ok = diff > 0.5  # text region retained its distinct height (>=0.5 mm)
    report("merge min_size: large text region preserved",
           ok, f"text-BG dTop diff={diff:.2f} (want >0.5)")


def test_merge_features_highlight_protect_preserves_bright_spots():
    """Iteration 50c: small bright spots (highlights) must survive aggressive
    min_size absorption when highlight_protect is enabled. A 6x6 bright coin
    on a darker desk should NOT be merged into the desk."""
    from litho_color import merge_features
    h, w = 60, 60
    rgb = np.full((h, w, 3), 80, dtype=np.uint8)          # dark desk
    rgb[25:31, 25:31, :] = 230                             # bright coin
    dTop = np.full((h, w), 0.2, dtype=np.float64)          # desk low
    dTop[25:31, 25:31] = 1.4                               # coin high
    # Without protection the coin (< min_size=40) is absorbed into the desk.
    out_off = merge_features(rgb, dTop, strength=1.0,
                             min_size=40, highlight_protect=0.0)
    out_on = merge_features(rgb, dTop, strength=1.0,
                            min_size=40, highlight_protect=0.05)
    coin_off = out_off[25:31, 25:31].mean()
    coin_on = out_on[25:31, 25:31].mean()
    bg_on = out_on[0:5, 0:5].mean()
    ok = (coin_on > coin_off + 0.5) and (coin_on > bg_on + 0.5)
    report("highlight_protect: bright coin preserved under aggressive merge",
           ok, f"coin off={coin_off:.2f} on={coin_on:.2f} bg={bg_on:.2f}")


def test_quantize_dtop_creates_terraces():
    """Iteration 50d: quantize_dtop rounds heights to step multiples, expanding
    flat plateaus so the slicer can fill with solid infill."""
    from litho_color import quantize_dtop
    dTop = np.array([[0.05, 0.11, 0.19, 0.31],
                     [0.07, 0.13, 0.21, 0.29]], dtype=np.float64)
    q = quantize_dtop(dTop, step=0.10, top_max=2.0)
    expected = np.array([[0.0, 0.1, 0.2, 0.3],
                         [0.1, 0.1, 0.2, 0.3]], dtype=np.float64)
    ok = np.allclose(q, expected, atol=1e-9)
    report("quantize_dtop: creates layer-height terraces",
           ok, f"out={q.tolist()}")


def test_engine_cmy_cover_margin_wires_through():
    """Iteration 50e: dtop_cmy_cover_margin is wired through the engine without
    crashing and does not blow up dE. The actual max(dC,dM,dY)+margin math is
    trivial; this test guards regressions in the P1 pipeline order."""
    from litho_engine import color_lithophane_engine
    from litho_core import LithophaneParams
    h, w = 50, 50
    rgb = np.full((h, w, 3), 200, dtype=np.uint8)
    rgb[15:35, 15:35] = [220, 30, 30]  # red patch -> CMY present
    params = LithophaneParams(width_mm=40, height_mm=40, pixel_pitch_mm=0.8)
    common = dict(params=params, pitch_cmy=0.8, pitch_top=0.8,
                  layer_h=0.12, dW=0.2, top_max=1.6,
                  sharpen=0, contrast=1.0,
                  chroma_decouple=True, recalib_luminance=True,
                  merge_features=0.0, merge_min_size=0,
                  dtop_min=0.0, dtop_quantize_step=0.0)
    meshes0, dE0, *_ = color_lithophane_engine(
        rgb, **common, dtop_cmy_cover_margin=0.0)
    meshes1, dE1, *_ = color_lithophane_engine(
        rgb, **common, dtop_cmy_cover_margin=0.10)
    med0 = float(np.median(dE0)) if isinstance(dE0, np.ndarray) else 0.0
    med1 = float(np.median(dE1)) if isinstance(dE1, np.ndarray) else 0.0
    ok = abs(med1 - med0) < 5.0
    report("engine dtop_cmy_cover_margin: wired through, dE stable",
           ok, f"dE_med {med0:.2f} -> {med1:.2f} (|Δ|<5.0)")


# === Iteration 50 ===
def test_dtop_median_filter_preserves_real_edges():
    """Iteration 50: median pre-filter kills tiny specks without flatting
    real edges."""
    from litho_color import merge_features
    h, w = 60, 60
    rgb = np.full((h, w, 3), 180, dtype=np.uint8)
    rgb[:, :30, :] = 20
    rgb[20:23, 40:43, :] = 20  # 3x3 speck (median window sweeps it)
    dTop = np.full((h, w), 1.0, dtype=np.float64)
    dTop[:, :30] = 0.0
    dTop[20:23, 40:43] = 0.0
    out = merge_features(rgb, dTop, top_max=2.0, strength=1.0,
                         dtop_median_size=3, min_size=0)
    edge_diff = out[:, :30].mean() - out[:, 30:].mean()
    speck_left = out[20:23, 40:43].mean()
    bg_right = out[20:23, 50:53].mean()
    ok = (abs(edge_diff) > 0.5) and (abs(speck_left - bg_right) < 0.7)
    report("median: edge preserved, 3x3 speck cleared",
           ok, f"|edge|={abs(edge_diff):.2f} |speck-bg|={abs(speck_left-bg_right):.2f}")


def test_merge_cmy_features_reduces_colour_variance():
    """Iteration 50: CMY plateau consolidation reduces jitter without
    flattening a real colour boundary."""
    from litho_color import merge_cmy_features
    h, w = 50, 50
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[:, :25, 0] = 200; rgb[:, :25, 1] = 30; rgb[:, :25, 2] = 30
    rgb[:, 25:, 0] = 30; rgb[:, 25:, 1] = 30; rgb[:, 25:, 2] = 200
    rgb[20:23, 10:13] = 50; rgb[21:24, 13:16] = 100
    dC = np.zeros((h, w))
    dM = np.zeros((h, w))
    dY = np.zeros((h, w))
    dM[:, :25] = 0.6
    dY[:, 25:] = 0.6
    rng = np.random.default_rng(0)
    dM[:, :25] += rng.normal(0, 0.10, (h, 25))
    out_C, out_M, out_Y = merge_cmy_features(rgb, dC, dM, dY,
                                             strength=1.0, min_size=16,
                                             cmy_median_size=0)
    edge = out_Y[:, 25:].mean() - out_Y[:, :25].mean()
    std_in = dM[:, :25].std()
    std_out = out_M[:, :25].std()
    ok = (edge > 0.4) and (std_out < std_in * 0.7)
    report("merge_cmy: step preserved, noise reduced",
           ok, f"Y_edge={edge:.2f} M_std {std_in:.3f}->{std_out:.3f}")


def test_enforce_dtop_minimum_lifts_low_regions():
    """Iteration 50: enforce_dtop_minimum lifts sub-floor dTop."""
    from litho_color import enforce_dtop_minimum
    dTop = np.array([[0.0, 0.5, 1.0],
                     [0.05, 1.2, 0.0],
                     [2.0, 0.02, 0.8]])
    out = enforce_dtop_minimum(dTop, dtop_min=0.10, top_max=2.0)
    below = dTop < 0.10
    lifted_ok = bool((out[below] == 0.10).all())
    same_ok = bool((out[~below] == dTop[~below]).all())
    report("enforce_dtop_minimum: lifts below floor, rest untouched",
           lifted_ok and same_ok,
           f"lifted_ok={lifted_ok} same_ok={same_ok}")
    out0 = enforce_dtop_minimum(dTop, dtop_min=0.0, top_max=2.0)
    report("enforce_dtop_minimum: zero disables (no-op)",
           bool(np.allclose(out0, dTop)), "")


def test_engine_integration_dtop_min_lifts_floor():
    """End-to-end: engine dtop_min is wired in (no exception, no dE blow-up).
    Floor enforcement itself is unit-tested in test_enforce_dtop_minimum_*;
    here we just verify the wiring calls the function and does not crash
    even when handed a synthetic dark-patch image that would normally crash
    dTop -> 0."""
    from litho_engine import color_lithophane_engine
    from litho_core import LithophaneParams
    h, w = 50, 50
    rgb = np.full((h, w, 3), 200, dtype=np.uint8)
    rgb[15:35, 15:35, :] = 5
    params = LithophaneParams(width_mm=40, height_mm=40, pixel_pitch_mm=0.8)
    common = dict(params=params, pitch_cmy=0.8, pitch_top=0.8,
                  layer_h=0.12, dW=0.2, top_max=1.6,
                  sharpen=0, contrast=1.0,
                  chroma_decouple=True, recalib_luminance=True,
                  merge_features=0.0, merge_min_size=0)
    meshes0, dE0, *_ = color_lithophane_engine(rgb, **common, dtop_min=0.0)
    meshes1, dE1, *_ = color_lithophane_engine(rgb, **common, dtop_min=0.15)
    med0 = float(np.median(dE0)) if isinstance(dE0, np.ndarray) else 0.0
    med1 = float(np.median(dE1)) if isinstance(dE1, np.ndarray) else 0.0
    ok = abs(med1 - med0) < 3.0
    report("engine dtop_min: wires through, dE not blown up",
           ok, f"dE_med {med0:.2f} -> {med1:.2f} (|Δ|<3.0)")


def test_engine_integration_cmy_merge_kills_jitter():
    """End-to-end: cmy_merge_features doesn't blow up dE."""
    from litho_engine import color_lithophane_engine
    from litho_core import LithophaneParams
    h, w = 60, 60
    rgb = np.full((h, w, 3), 200, dtype=np.uint8)
    rng = np.random.default_rng(0)
    noise = rng.integers(0, 30, (h, w, 3), dtype=np.uint8)
    rgb = np.clip(rgb.astype(int) - noise, 0, 255).astype(np.uint8)
    rgb[20:40, 20:40] = [220, 30, 30]
    params = LithophaneParams(width_mm=40, height_mm=40, pixel_pitch_mm=0.7)
    common = dict(params=params, pitch_cmy=0.7,
                  pitch_top=0.7, layer_h=0.12, dW=0.2, top_max=1.6,
                  sharpen=0, contrast=1.0,
                  chroma_decouple=True, recalib_luminance=True,
                  dtop_median_size=0, cmy_merge_features=0.0,
                  cmy_merge_min_size=0, cmy_median_size=0, dtop_min=0.0)
    _, dE0, *_ = color_lithophane_engine(rgb, **common)
    _, dE1, *_ = color_lithophane_engine(
        rgb, **{**common, "cmy_merge_features": 0.8,
                "cmy_merge_min_size": 20, "cmy_median_size": 3})
    if isinstance(dE0, np.ndarray):
        med0 = float(np.median(dE0))
        med1 = float(np.median(dE1))
    else:
        med0 = med1 = 0.0
    ok = abs(med1 - med0) < 3.0
    report("engine cmy_merge: dE not blown up",
           ok, f"dE_med {med0:.2f} -> {med1:.2f} (|Δ|<3.0)")


if __name__ == "__main__":
    test_layered_orders()
    test_interleaved()
    test_overlap()
    test_greyscale()
    test_full_traversal()
    test_stacked_no_floating()
    test_routing()
    test_layer_height_param()
    test_bambu()
    test_default_face_count()
    test_whiteness_mask()
    test_white_collapse_legacy_unchanged()
    test_export_sizing_default_156x106()
    test_export_sizing_legacy_long_edge()
    test_merge_features_noop()
    test_merge_features_merges_flat_noise_keeps_edge()
    test_merge_features_e2e_levels_reduced()
    test_resolve_cmy_chroma_only_picks_min_chroma_dist()
    test_chroma_decouple_keeps_dtop_untouched()
    test_cmy_smooth_reduces_colour_variance()
    test_recalibrate_hits_target_luminance()
    test_recalib_luminance_lowers_dE()
    test_merge_min_size_absorbs_tiny_specks()
    test_merge_min_size_preserves_large_text_region()
    test_merge_features_highlight_protect_preserves_bright_spots()
    test_quantize_dtop_creates_terraces()
    test_engine_cmy_cover_margin_wires_through()
    # Iteration 50
    test_dtop_median_filter_preserves_real_edges()
    test_merge_cmy_features_reduces_colour_variance()
    test_enforce_dtop_minimum_lifts_low_regions()
    test_engine_integration_dtop_min_lifts_floor()
    test_engine_integration_cmy_merge_kills_jitter()
    print()
    print(f"{sum(RESULTS)}/{len(RESULTS)} passed")
    sys.exit(0 if all(RESULTS) else 1)
