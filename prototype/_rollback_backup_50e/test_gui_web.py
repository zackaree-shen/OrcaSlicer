"""Validation for the web GUI backend helpers (LithoApi._decimate / _mesh_json).

Covers:
  - _decimate passthrough for small meshes (n <= max_verts unchanged).
  - _decimate on a structured height-field band: index validity, size
    reduction, top AND bottom verts both referenced (no orphan vertices —
    regression for the "rotating past the back hides the model" bug), and
    Z stacking preserved (both top and bottom z survive at shared XY).
  - _mesh_json returns the {color: {verts, faces}} contract the frontend
    render3D() consumes.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from litho_gui_web import LithoApi

RESULTS = []


def report(name, ok, detail=""):
    RESULTS.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name:55s} {detail}")


def make_band_mesh(gx=200, gy=150, z_lo=0.5, z_hi=2.5):
    """Structured band mesh: n = 2*gx*gy verts (top grid + bottom grid),
    exactly sharing XY per column (like the engine's heightfield band)."""
    xs = np.linspace(0, 10, gx)
    ys = np.linspace(0, 10, gy)
    xx, yy = np.meshgrid(xs, ys)
    zz = z_lo + (z_hi - z_lo) * (0.5 + 0.5 * np.sin(xx) * np.cos(yy))  # always >= z_lo
    top = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)
    bot = np.stack([xx.ravel(), yy.ravel(), np.full(xx.size, z_lo)], axis=1)
    verts = np.vstack([top, bot])
    n = gx * gy
    faces = []
    for r in range(gy - 1):
        for c in range(gx - 1):
            a = r * gx + c
            b = a + 1
            c_ = (r + 1) * gx + c + 1
            d = (r + 1) * gx + c
            faces += [(a, b, c_), (a, c_, d)]
            faces += [(a + n, d + n, c_ + n), (a + n, c_ + n, b + n)]
    return verts, np.array(faces, dtype=np.int64)


def test_decimate_passthrough():
    verts, faces = make_band_mesh(gx=20, gy=15)  # 600 verts <= 12000
    out = LithoApi._decimate(verts, faces, max_verts=12000)
    v = np.array(out["verts"]).reshape(-1, 3)
    f = np.array(out["faces"]).reshape(-1, 3)
    report("decimate: small mesh passes through unchanged",
           len(v) == len(verts) and len(f) == len(faces) and np.array_equal(v, verts),
           f"v={len(v)}/{len(verts)} f={len(f)}/{len(faces)}")


def test_decimate_large_closed():
    verts, faces = make_band_mesh(gx=200, gy=150)  # 60000 verts
    out = LithoApi._decimate(verts, faces, max_verts=12000)
    v = np.array(out["verts"]).reshape(-1, 3)
    f = np.array(out["faces"]).reshape(-1, 3)
    reduced = len(v) < len(verts)
    report("decimate: large mesh is reduced", reduced, f"v={len(v)}/{len(verts)}")

    valid_idx = len(f) > 0 and int(f.min()) >= 0 and int(f.max()) < len(v)
    report("decimate: all face indices valid", valid_idx,
           f"f={len(f)} max_idx={int(f.max())} n_verts={len(v)}")

    # No orphan vertices: every kept vertex (top AND bottom) is referenced.
    used = np.unique(f)
    orphaned = len(v) - len(used)
    report("decimate: no orphan vertices (bottom referenced)", orphaned == 0,
           f"orphans={orphaned}")

    # Z stacking preserved: decimated mesh still has both top and bottom z.
    z_top = float(v[:, 2].max())
    z_bot = float(v[:, 2].min())
    report("decimate: Z stacking preserved (top>bottom, bottom==z_lo)",
           z_bot == 0.5 and z_top > 1.5,
           f"z=[{z_bot:.2f},{z_top:.2f}]")

    # Bottom faces exist: some triangle has all three verts at z_lo.
    v2 = np.abs(v[:, 2] - 0.5) < 1e-9
    has_bottom_tri = any(v2[t].all() for t in f[: min(len(f), 40000)])
    report("decimate: bottom faces emitted (closed shell)", has_bottom_tri)


def test_mesh_json_contract():
    api = LithoApi()
    verts, faces = make_band_mesh(gx=10, gy=10)
    meshes = {
        "W": (verts, faces),
        "C": (np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)),
    }
    out = api._mesh_json(meshes)
    ok = set(out.keys()) == {"W", "C"}
    ok &= isinstance(out["W"]["verts"], list) and len(out["W"]["verts"]) % 3 == 0
    ok &= isinstance(out["W"]["faces"], list) and len(out["W"]["faces"]) % 3 == 0
    ok &= out["C"]["verts"] == [] and out["C"]["faces"] == []
    report("_mesh_json: {color:{verts,faces}} contract + empty layers",
           ok, f"keys={list(out.keys())} W_v={len(out['W']['verts'])//3}")


if __name__ == "__main__":
    test_decimate_passthrough()
    test_decimate_large_closed()
    test_mesh_json_contract()
    print()
    print(f"{sum(RESULTS)}/{len(RESULTS)} passed")
    sys.exit(0 if all(RESULTS) else 1)
