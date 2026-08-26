"""Main configurable CMYK lithophane exporter (v4).

Defaults combine BAMBU-like surface detail with an overlap-style thin stack:
  - Mode OVERLAP: thin white base + overlapping C/M/Y band + top relief.
  - Use the original input image as-is (no extra downscale).
  - pixel_pitch_mm = 0.15  (solver grid; letters stay readable).
  - pitch_top      = 0.15  (output mesh grid; same as solver to preserve maximum top detail).
  - pitch_cmy      = 0.30  (color layer mesh grid).
  - White base = 0.20 mm, color band = 6 * 0.12 = 0.72 mm, total ~2 mm.
  - detail_level = 1.0: max detail / max edge crispness (matches bambu_v4_overlap_detail).
  - y_strength = 1.0 (no yellow correction by default; tune per filament).

Example:
  python export_v4.py Z:/image.png -o C:/Users/me/Desktop/out
  python export_v4.py Z:/image.png -o C:/Users/me/Desktop/out --long-edge-mm 200 --pixel-pitch 0.10
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from litho_core import LithophaneParams, export_stl
from litho_engine import LithoMode, ColorOrder, color_lithophane_engine
from litho_3mf import assemble_lithophane_parts, write_3mf
from litho_gui import _BED_CENTERS_MM


STL_NAMES = [("W", "white"), ("C", "cyan"), ("M", "magenta"),
             ("Y", "yellow"), ("top", "top_white")]


def save_heightmap_png(dTop: np.ndarray, path: str) -> None:
    lo, hi = dTop.min(), dTop.max()
    norm = np.clip((dTop - lo) / (hi - lo + 1e-9) * 255, 0, 255).astype(np.uint8)
    Image.fromarray(norm).save(path)


def export_lithophane(
    image_path: str,
    outdir: str,
    *,
    printer: str = "Snapmaker U1",
    long_edge_mm: float = 144.0,
    pixel_pitch_mm: float = 0.15,
    pitch_top_mm: float | None = None,
    pitch_cmy_mm: float = 0.30,
    surface_refine: bool = True,
    detail_level: float = 1.0,
    c_strength: float = 1.0,
    m_strength: float = 1.0,
    y_strength: float = 1.0,
    w_strength: float = 1.0,
    dW: float = 0.20,
    top_max: float = 1.2,
    layers_max: int = 6,
    layer_h: float = 0.12,
    sharpen: float = 2.0,
    contrast: float = 1.5,
    save_preview: bool = True,
    flip_y: bool = True,
) -> dict:
    """Run the OVERLAP lithophane engine and export 3MF + per-color STLs.

    Returns the engine output dict for inspection.
    """
    os.makedirs(outdir, exist_ok=True)

    with Image.open(image_path) as im:
        rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)
    print(f"Input: {image_path}  ({rgb.shape[1]}x{rgb.shape[0]} px)")

    scale = long_edge_mm / max(rgb.shape[1], rgb.shape[0])
    params = LithophaneParams(
        width_mm=rgb.shape[1] * scale,
        height_mm=rgb.shape[0] * scale,
        pixel_pitch_mm=pixel_pitch_mm,
    )
    pitch_top = pitch_top_mm if pitch_top_mm is not None else 0.15

    print(f"Physical size: {params.width_mm:.1f} x {params.height_mm:.1f} mm")
    print(f"Grids: pixel_pitch={pixel_pitch_mm} mm, pitch_top={pitch_top} mm, pitch_cmy={pitch_cmy_mm} mm")

    meshes, dE, gamut, reached = color_lithophane_engine(
        rgb,
        mode=LithoMode.OVERLAP,
        order=ColorOrder.MIXED,
        params=params,
        exact=False,
        dW=dW,
        layers_max=layers_max,
        layer_h=layer_h,
        pitch_cmy=pitch_cmy_mm,
        pitch_top=pitch_top,
        top_max=top_max,
        sharpen=sharpen,
        contrast=contrast,
        surface_refine=surface_refine,
        detail_level=detail_level,
        c_strength=c_strength,
        m_strength=m_strength,
        y_strength=y_strength,
        w_strength=w_strength,
        flip_y=flip_y,
    )

    parts, offsets, names, extruders = assemble_lithophane_parts(meshes)
    bc = _BED_CENTERS_MM.get(printer, (0.0, 0.0))
    write_3mf(
        os.path.join(outdir, "lithophane.3mf"),
        parts, offsets, extruders,
        part_names=names,
        printer_model=printer,
        printer_settings_id=f"{printer} (0.4 nozzle)",
        build_center_mm=(bc[0], bc[1], 0.0),
    )

    for key, name in STL_NAMES:
        v, f = meshes[key]
        if len(f) == 0:
            continue
        export_stl(os.path.join(outdir, f"litho_{name}.stl"), v, f,
                   name=f"lithophane_{name}")

    if save_preview:
        # Recompute dTop preview at the same resolution for a quick visual check.
        from litho_color import anchored_dtop_field, refine_dtop_surface, preprocess_image
        from litho_core import thickness_grid_shape
        detail_level_c = float(np.clip(detail_level, 0.0, 1.0))
        edge_alpha = 0.20 + 0.60 * detail_level_c
        refine_gamma = 0.10 + 0.12 * (1.0 - detail_level_c)
        refine_iter = int(round(20 + 40 * (1.0 - detail_level_c)))
        gx, gy = thickness_grid_shape(rgb.shape[0], rgb.shape[1], params)
        small = np.asarray(Image.fromarray(rgb).resize((gx, gy), Image.LANCZOS))
        proc = preprocess_image(small, sharpen=sharpen, contrast=contrast)
        dTop = anchored_dtop_field(proc, td_w=5.4, top_max=top_max)
        if surface_refine:
            dTop = refine_dtop_surface(dTop, proc, edge_alpha=edge_alpha,
                                       gamma=refine_gamma, n_iter=refine_iter)
        save_heightmap_png(dTop, os.path.join(outdir, "preview_dTop.png"))

    med = float(np.median(dE)) if dE is not None else float("nan")
    print(f"dE median: {med:.2f}")
    print(f"Exported to: {outdir}")
    return {"meshes": meshes, "dE": dE, "gamut": gamut, "reached": reached}


def main():
    parser = argparse.ArgumentParser(
        description="Export high-detail CMYK lithophane to 3MF + STLs (v4)."
    )
    parser.add_argument("image", help="Input image path")
    parser.add_argument("-o", "--outdir", required=True, help="Output directory")
    parser.add_argument("--printer", default="Snapmaker U1", help="Printer model")
    parser.add_argument("--long-edge-mm", type=float, default=144.0,
                        help="Physical long edge size in mm (default: 144)")
    parser.add_argument("--pixel-pitch-mm", type=float, default=0.15,
                        help="Thickness grid pitch in mm; smaller = more detail (default: 0.15)")
    parser.add_argument("--pitch-top-mm", type=float, default=None,
                        help="Top relief mesh pitch (default: 0.15)")
    parser.add_argument("--pitch-cmy-mm", type=float, default=0.30,
                        help="Color layer mesh pitch (default: 0.30)")
    parser.add_argument("--no-surface-refine", action="store_true",
                        help="Disable edge-aware surface refinement")
    parser.add_argument("--detail-level", type=float, default=1.0,
                        help="0.0=max smooth/merge, 1.0=max detail (default: 0.5)")
    parser.add_argument("--cyan-strength", type=float, default=1.0,
                        help=">1 makes solver use less cyan (default: 1.0)")
    parser.add_argument("--magenta-strength", type=float, default=1.0,
                        help=">1 makes solver use less magenta (default: 1.0)")
    parser.add_argument("--yellow-strength", type=float, default=1.0,
                        help=">1 makes solver use less yellow (default: 1.0)")
    parser.add_argument("--white-strength", type=float, default=1.0,
                        help=">1 makes white act denser (default: 1.0)")
    parser.add_argument("--dW", type=float, default=0.20,
                        help="White base thickness in mm (default: 0.20)")
    parser.add_argument("--top-max", type=float, default=1.2,
                        help="Max top relief thickness in mm (default: 1.2)")
    parser.add_argument("--layers-max", type=int, default=6,
                        help="Number of color layers per channel (default: 6)")
    parser.add_argument("--layer-h", type=float, default=0.12,
                        help="Color layer height in mm (default: 0.12)")
    parser.add_argument("--sharpen", type=float, default=2.0,
                        help="Preprocess sharpening (default: 2.0)")
    parser.add_argument("--contrast", type=float, default=1.5,
                        help="Preprocess contrast (default: 1.5)")
    parser.add_argument("--no-preview", action="store_true",
                        help="Do not save preview_dTop.png")
    parser.add_argument("--legacy-orientation", action="store_true",
                        help="Keep legacy Y orientation (image top -> -Y)")
    args = parser.parse_args()

    export_lithophane(
        image_path=args.image,
        outdir=args.outdir,
        printer=args.printer,
        long_edge_mm=args.long_edge_mm,
        pixel_pitch_mm=args.pixel_pitch_mm,
        pitch_top_mm=args.pitch_top_mm,
        pitch_cmy_mm=args.pitch_cmy_mm,
        surface_refine=not args.no_surface_refine,
        detail_level=args.detail_level,
        c_strength=args.cyan_strength,
        m_strength=args.magenta_strength,
        y_strength=args.yellow_strength,
        w_strength=args.white_strength,
        dW=args.dW,
        top_max=args.top_max,
        layers_max=args.layers_max,
        layer_h=args.layer_h,
        sharpen=args.sharpen,
        contrast=args.contrast,
        save_preview=not args.no_preview,
        flip_y=not args.legacy_orientation,
    )


if __name__ == "__main__":
    main()
