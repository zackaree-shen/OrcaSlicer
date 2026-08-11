"""Validation for the reverse-import module (STL/3MF -> preview image).

Checks:
  - ray-cast thickness extraction is exact on pixel boxes (0 error)
  - 3MF part identity: single-letter names W/C/M/Y/top are mapped correctly
  - composite Z offsets are applied so layers land at their true heights
  - reconstruction runs end-to-end (3MF -> image) and produces a sane image
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from litho_reverse import (_color_key, extract_thickness_raycast, load_3mf_colors,
                           ray_z_spans, reconstruct_from_meshes, _ray_z_span)
from litho_engine import LithoMode, ColorOrder, color_lithophane_engine
from litho_core import LithophaneParams
from litho_3mf import assemble_lithophane_parts, write_3mf
from test_engine import make_test_img

RESULTS = []


def report(name, ok, detail=""):
    RESULTS.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name:55s} {detail}")


def test_raycast_box():
    """Vertical ray through a unit box returns the exact thickness."""
    # Box [0,1]x[0,1]x[0,0.5]: 2 top tris + 2 bottom tris.
    t1 = np.array([[0, 0, 0.5], [1, 0, 0.5], [1, 1, 0.5]], dtype=np.float64)
    t2 = np.array([[0, 0, 0.5], [1, 1, 0.5], [0, 1, 0.5]], dtype=np.float64)
    b1 = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]], dtype=np.float64)
    b2 = np.array([[0, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float64)
    tris = np.array([t1, t2, b1, b2])
    xs = np.array([0.5])
    ys = np.array([0.5])
    thick = extract_thickness_raycast(tris, xs, ys)
    report("ray-cast: box thickness exact", abs(thick[0, 0] - 0.5) < 1e-9,
           f"thick={thick[0,0]}")


def test_color_key():
    report("color key: single letters", _color_key("W") == "W" and _color_key("C") == "C"
           and _color_key("M") == "M" and _color_key("Y") == "Y" and _color_key("top") == "top")
    report("color key: descriptive", _color_key("litho_cyan.stl") == "C"
           and _color_key("jade white") == "W")


def test_3mf_roundtrip(tmp="_rev_rt.3mf"):
    img = make_test_img(h=30, w=40)
    p = LithophaneParams(width_mm=40, height_mm=30, pixel_pitch_mm=1.0)
    meshes, _, _, _ = color_lithophane_engine(img, mode=LithoMode.STACKED,
                                              order=ColorOrder.CMY, params=p,
                                              pitch_cmy=1.0, pitch_top=0.5)
    parts, offsets, names, extruders = assemble_lithophane_parts(meshes)
    write_3mf(tmp, parts, offsets, extruders, part_names=names)

    cmeshes = load_3mf_colors(tmp)
    report("3MF: all 5 colors recovered", set(cmeshes.keys()) == {"W", "C", "M", "Y", "top"},
           str(sorted(cmeshes.keys())))

    # Composite offsets applied: layer Z ranges distinct.
    ztops = []
    for c in ("W", "C", "M", "Y", "top"):
        if c in cmeshes and len(cmeshes[c]):
            ztops.append(float(cmeshes[c][:, :, 2].max()))
    report("3MF: Z offsets applied (layers distinct)", len(set(round(z, 1) for z in ztops)) >= 3,
           f"ztops={sorted(set(round(z, 1) for z in ztops))}")

    # End-to-end reconstruction produces a sane (non-black) image.
    recon = reconstruct_from_meshes(cmeshes, 40, 30, pixel_pitch=1.0)
    ok_shape = recon.shape == (30, 40, 3)
    ok_nonblack = float(recon.mean()) > 20
    report("reconstruct: sane image", ok_shape and ok_nonblack,
           f"shape={recon.shape} mean={recon.mean():.0f}")

    if os.path.exists(tmp):
        os.remove(tmp)


if __name__ == "__main__":
    test_raycast_box()
    test_color_key()
    test_3mf_roundtrip()
    print()
    print(f"{sum(RESULTS)}/{len(RESULTS)} passed")
    sys.exit(0 if all(RESULTS) else 1)
