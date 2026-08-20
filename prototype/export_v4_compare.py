"""Export baseline (v0.45 spike_surgery) vs improved (edge-aware surface refine)
for side-by-side inspection."""

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
OUT_ROOT = r"C:\Users\snapmaker\Desktop\bambu_v4_improved"
PRINTER = "Snapmaker U1"


def run_export(rgb, outdir, surface_refine, label):
    os.makedirs(outdir, exist_ok=True)

    TARGET_LONG_MM = 144.0
    scale = TARGET_LONG_MM / max(rgb.shape[1], rgb.shape[0])
    params = LithophaneParams(width_mm=rgb.shape[1] * scale,
                              height_mm=rgb.shape[0] * scale,
                              pixel_pitch_mm=0.3)

    meshes, dE, gamut, reached = color_lithophane_engine(
        rgb, mode=LithoMode.BAMBU, order=ColorOrder.MIXED,
        params=params, exact=False,
        pitch_cmy=0.44, pitch_top=0.22,
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
    """Save dTop as a grayscale PNG for quick visual check."""
    lo, hi = dTop.min(), dTop.max()
    norm = np.clip((dTop - lo) / (hi - lo + 1e-9) * 255, 0, 255).astype(np.uint8)
    Image.fromarray(norm).save(path)


def main():
    with Image.open(IMG_PATH) as im:
        rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)
    print(f"Image: {rgb.shape[1]}x{rgb.shape[0]} px")

    # Baseline: current v0.45 (spike_surgery)
    meshes_base, dE_base = run_export(
        rgb, os.path.join(OUT_ROOT, "baseline_spike"),
        surface_refine=False, label="baseline (spike_surgery)")

    # Improved: edge-aware surface refinement
    meshes_improved, dE_improved = run_export(
        rgb, os.path.join(OUT_ROOT, "improved_edge_refine"),
        surface_refine=True, label="improved (edge-aware refine)")

    # Heightmap previews (top white relief)
    base_top = meshes_base["top"]
    improved_top = meshes_improved["top"]
    # BAMBU mode: top mesh is empty; relief lives in W part. Extract Z heights.
    # The W mesh in BAMBU is base + relief merged; relief vertices are those with
    # z > z_lo + band + gap (~0.95). For quick preview we just use the dTop field.
    # Recompute dTop fields by resampling for preview.
    from litho_color import anchored_dtop_field, refine_dtop_surface, preprocess_image
    from litho_core import thickness_grid_shape
    TARGET_LONG_MM = 144.0
    scale = TARGET_LONG_MM / max(rgb.shape[1], rgb.shape[0])
    params = LithophaneParams(width_mm=rgb.shape[1] * scale,
                              height_mm=rgb.shape[0] * scale,
                              pixel_pitch_mm=0.3)
    gx, gy = thickness_grid_shape(rgb.shape[0], rgb.shape[1], params)
    small = np.asarray(Image.fromarray(rgb).resize((gx, gy), Image.LANCZOS))
    proc = preprocess_image(small, sharpen=2.0, contrast=1.5)
    dTop_base = anchored_dtop_field(proc, td_w=5.4, top_max=2.0)
    dTop_improved = refine_dtop_surface(dTop_base, proc)
    save_heightmap_png(dTop_base, os.path.join(OUT_ROOT, "preview_dTop_baseline.png"))
    save_heightmap_png(dTop_improved, os.path.join(OUT_ROOT, "preview_dTop_improved.png"))

    print(f"\nDone. Output: {OUT_ROOT}")


if __name__ == "__main__":
    main()
