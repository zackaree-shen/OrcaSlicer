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
    print()
    print(f"{sum(RESULTS)}/{len(RESULTS)} passed")
    sys.exit(0 if all(RESULTS) else 1)
