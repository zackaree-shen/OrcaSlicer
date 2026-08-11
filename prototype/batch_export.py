"""Batch export: every lithophane algorithm -> 3MF + 5 STLs, in a layered dir.

Directory layout (each algorithm isolated, no cross-contamination):
    lithophane_exports/
      <mode>[_order]/        (one dir per algorithm)
        lithophane.3mf       (composite, one object / 5 parts)
        litho_W.stl  litho_C.stl  litho_M.stl  litho_Y.stl  litho_top.stl

Run:  python batch_export.py <image> <outdir>
"""

from __future__ import annotations

import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from litho_core import LithophaneParams, export_stl
from litho_engine import LithoMode, ColorOrder, color_lithophane_engine
from litho_3mf import assemble_lithophane_parts, write_3mf


def run_mode(rgb, mode, order, outdir, printer="Snapmaker U1"):
    """Generate one algorithm's meshes and write 3MF + 5 STLs into outdir."""
    os.makedirs(outdir, exist_ok=True)

    # Default printable size: long side 144mm, aspect-preserving (matches GUI).
    TARGET_LONG_MM = 144.0
    w_px, h_px = rgb.shape[1], rgb.shape[0]
    scale = TARGET_LONG_MM / max(w_px, h_px)
    params = LithophaneParams(width_mm=w_px * scale, height_mm=h_px * scale,
                              pixel_pitch_mm=0.3)

    meshes, dE, gamut, reached = color_lithophane_engine(
        rgb, mode=mode, order=order, params=params, exact=False)

    # --- 3MF composite export ---
    parts, offsets, names, extruders = assemble_lithophane_parts(meshes)
    from litho_gui import _BED_CENTERS_MM
    bc = _BED_CENTERS_MM.get(printer, (0.0, 0.0))
    write_3mf(os.path.join(outdir, "lithophane.3mf"), parts, offsets, extruders,
              part_names=names, printer_model=printer,
              printer_settings_id=f"{printer} (0.4 nozzle)",
              build_center_mm=(bc[0], bc[1], 0.0))

    # --- 5 per-color STLs (absolute Z, stack-aligned) ---
    stl_names = [("W", "white"), ("C", "cyan"), ("M", "magenta"),
                 ("Y", "yellow"), ("top", "top_white")]
    for key, name in stl_names:
        v, f = meshes[key]
        if len(f) == 0:
            continue
        export_stl(os.path.join(outdir, f"litho_{name}.stl"), v, f,
                   name=f"lithophane_{name}")

    return params, dE


def main():
    img_path = sys.argv[1]
    out_root = sys.argv[2]

    with Image.open(img_path) as im:
        rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)
    print(f"Image: {rgb.shape[1]}x{rgb.shape[0]} px")

    # All algorithms. LAYERED uses CMY order (the default); others MIXED/CMY.
    jobs = [
        ("greyscale",  LithoMode.GREYSCALE,   ColorOrder.CMY),
        ("layered_cmy", LithoMode.LAYERED,    ColorOrder.CMY),
        ("interleaved", LithoMode.INTERLEAVED, ColorOrder.MIXED),
        ("stacked",    LithoMode.STACKED,     ColorOrder.CMY),
        ("overlap",    LithoMode.OVERLAP,     ColorOrder.MIXED),
    ]

    for name, mode, order in jobs:
        outdir = os.path.join(out_root, name)
        try:
            params, dE = run_mode(rgb, mode, order, outdir)
            med = float(np.median(dE)) if dE is not None else float("nan")
            print(f"  {name:15s}: size {params.width_mm:.0f}x{params.height_mm:.0f}mm, "
                  f"dE_med={med:.2f}  -> {outdir}")
        except Exception as e:  # noqa: BLE001
            import traceback
            print(f"  {name:15s}: FAILED: {e}")
            traceback.print_exc()

    print("\nDone. See:", out_root)


if __name__ == "__main__":
    main()
