"""High-resolution detail test: verify whether finer pixel pitch recovers
detail lost in the default 0.3 mm grid."""

from __future__ import annotations

import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from litho_core import LithophaneParams, export_stl
from litho_engine import LithoMode, ColorOrder, color_lithophane_engine
from litho_3mf import assemble_lithophane_parts, write_3mf
from litho_gui import _BED_CENTERS_MM


IMG_PATH = r"Z:\selfDIr\壁纸\【哲风壁纸】保险柜-办公室-卡通.png"
OUT_ROOT = r"C:\Users\snapmaker\Desktop\bambu_v4_highdetail"
PRINTER = "Snapmaker U1"

# Finer grid: 4x pixels vs default 0.3 mm.
PIXEL_PITCH_MM = 0.15
PITCH_CMY = 0.30
PITCH_TOP = 0.15
TARGET_LONG_MM = 144.0


def run_export(rgb, outdir, surface_refine, label):
    os.makedirs(outdir, exist_ok=True)

    scale = TARGET_LONG_MM / max(rgb.shape[1], rgb.shape[0])
    params = LithophaneParams(width_mm=rgb.shape[1] * scale,
                              height_mm=rgb.shape[0] * scale,
                              pixel_pitch_mm=PIXEL_PITCH_MM)

    meshes, dE, gamut, reached = color_lithophane_engine(
        rgb, mode=LithoMode.BAMBU, order=ColorOrder.MIXED,
        params=params, exact=False,
        pitch_cmy=PITCH_CMY, pitch_top=PITCH_TOP,
        surface_refine=surface_refine)

    parts, offsets, names, extruders = assemble_lithophane_parts(meshes)
    bc = _BED_CENTERS_MM.get(PRINTER, (0.0, 0.0))
    write_3mf(os.path.join(outdir, "lithophane.3mf"), parts, offsets, extruders,
              part_names=names, printer_model=PRINTER,
              printer_settings_id=f"{PRINTER} (0.4 nozzle)",
              build_center_mm=(bc[0], bc[1], 0.0))

    stl_names = [("W", "white"), ("C", "cyan"), ("M", "magenta"),
                 ("Y", "yellow"), ("top", "top_white")]
    for key, name in stl_names:
        v, f = meshes[key]
        if len(f) == 0:
            continue
        export_stl(os.path.join(outdir, f"litho_{name}.stl"), v, f,
                   name=f"lithophane_{name}")

    med = float(np.median(dE)) if dE is not None else float("nan")
    print(f"  {label}: dE_med={med:.2f} -> {outdir}")
    return meshes, dE


def save_heightmap_png(dTop, path):
    lo, hi = dTop.min(), dTop.max()
    norm = np.clip((dTop - lo) / (hi - lo + 1e-9) * 255, 0, 255).astype(np.uint8)
    Image.fromarray(norm).save(path)


def main():
    with Image.open(IMG_PATH) as im:
        rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)
    print(f"Image: {rgb.shape[1]}x{rgb.shape[0]} px")
    print(f"High-detail grid: pixel_pitch={PIXEL_PITCH_MM}, pitch_cmy={PITCH_CMY}, pitch_top={PITCH_TOP}")

    run_export(rgb, os.path.join(OUT_ROOT, "hd_surface_refine"),
               surface_refine=True, label="HD + surface_refine")
    run_export(rgb, os.path.join(OUT_ROOT, "hd_no_refine"),
               surface_refine=False, label="HD + no_refine (raw)")

    # Quick preview of dTop resolution difference
    from litho_color import anchored_dtop_field, refine_dtop_surface, preprocess_image
    from litho_core import thickness_grid_shape
    params = LithophaneParams(width_mm=rgb.shape[1] * TARGET_LONG_MM / max(rgb.shape[1], rgb.shape[0]),
                              height_mm=rgb.shape[0] * TARGET_LONG_MM / max(rgb.shape[1], rgb.shape[0]),
                              pixel_pitch_mm=PIXEL_PITCH_MM)
    gx, gy = thickness_grid_shape(rgb.shape[0], rgb.shape[1], params)
    small = np.asarray(Image.fromarray(rgb).resize((gx, gy), Image.LANCZOS))
    proc = preprocess_image(small, sharpen=2.0, contrast=1.5)
    dTop_base = anchored_dtop_field(proc, td_w=5.4, top_max=2.0)
    dTop_refined = refine_dtop_surface(dTop_base, proc)
    save_heightmap_png(dTop_base, os.path.join(OUT_ROOT, "preview_dTop_hd_base.png"))
    save_heightmap_png(dTop_refined, os.path.join(OUT_ROOT, "preview_dTop_hd_refined.png"))

    print(f"\nDone. Output: {OUT_ROOT}")


if __name__ == "__main__":
    main()
