"""Algorithm validation for the lithophane prototype.

Runs synthetic images through the full pipeline and checks:
  - closed mesh (0 open edges)
  - positive signed volume (outward winding)
  - no degenerate triangles
  - round-trips through numpy-stl
  - stability across a range of parameter combinations
"""

from __future__ import annotations

import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from litho_core import LithophaneParams, build_mesh, export_stl, load_greyscale, validate_mesh


def make_synthetic(width=480, height=360):
    """A photo-like test image: smooth gradient + radial blob + edges."""
    y, x = np.mgrid[0:height, 0:width].astype(np.float64)
    # Diagonal smooth gradient (tests bilinear sampling smoothness).
    grad = (x / width + y / height) / 2.0
    # Radial soft blob (tests curved height field).
    cx, cy = width * 0.65, height * 0.4
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) / (width * 0.28)
    blob = np.exp(-0.5 * r * r)
    # Sharp edge (tests hard transitions / quad correctness).
    edge = ((x > width * 0.3) & (x < width * 0.7) & (y > height * 0.2) & (y < height * 0.5)) * 1.0
    img = np.clip(grad * 0.5 + blob * 0.5 + edge * 0.3, 0, 1)
    return (img * 255).astype(np.uint8)


def check(grey, params, label):
    vertices, faces = build_mesh(grey, params)
    v = validate_mesh(vertices, faces)
    status = []
    status.append(f"vol={v['volume']:+.2f}")
    status.append(f"open={v['open_edges']}")
    status.append(f"deg={v['degenerate']}")
    status.append(f"V={v['num_vertices']} F={v['num_faces']}")
    ok = v["open_edges"] == 0 and v["volume"] > 0 and v["degenerate"] == 0
    print(f"{'PASS' if ok else 'FAIL'}  {label:45s} " + "  ".join(status))
    return ok, vertices, faces, v


def main():
    grey = make_synthetic()
    results = []

    # Baseline.
    p = LithophaneParams()
    results.append(check(grey, p, "default 144x108 pitch=0.2"))

    # Small footprint (fewer verts).
    p2 = LithophaneParams(width_mm=80, height_mm=60, pixel_pitch_mm=0.4)
    results.append(check(grey, p2, "small 80x60 pitch=0.4"))

    # Large footprint (more verts, stress memory/large-index).
    p3 = LithophaneParams(width_mm=200, height_mm=160, pixel_pitch_mm=0.08)
    results.append(check(grey, p3, "large 200x160 pitch=0.08"))

    # Extreme depths.
    p4 = LithophaneParams(base_thickness=0.2, depth_range=6.0)
    results.append(check(grey, p4, "extreme depth base=0.2 range=6"))

    # Thin base (near-degenerate white region).
    p5 = LithophaneParams(base_thickness=0.05, depth_range=1.0)
    results.append(check(grey, p5, "thin base=0.05"))

    # Square aspect, mirror on.
    p6 = LithophaneParams(width_mm=120, height_mm=120, mirror=True)
    results.append(check(grey, p6, "square 120x120 mirror"))

    # Tiny 1xN image (edge cases in bilinear).
    tiny = (grey[:2, :20])
    p7 = LithophaneParams()
    results.append(check(tiny, p7, "tiny image 20x2"))

    # All-white and all-black (extreme uniform).
    for name, val in [("all-white", 255), ("all-black", 0)]:
        uni = np.full((60, 80), val, dtype=np.uint8)
        results.append(check(uni, LithophaneParams(), name))

    # STL round-trip.
    vertices, faces = build_mesh(grey, p)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out_test.stl")
    fmt = export_stl(out, vertices, faces)
    from stl import mesh as stl_mesh
    loaded = stl_mesh.Mesh.from_file(out)
    print(f"PASS  STL round-trip ({fmt}): {loaded.vectors.shape[0]} tris loaded")
    os.remove(out)

    ok_all = all(r[0] for r in results)
    print()
    print("ALL PASS" if ok_all else "SOME FAILED")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
