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
    width_mm: float = 156.0,
    height_mm: float = 106.0,
    long_edge_mm: float | None = None,
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
    top_max: float = 1.6,
    layers_max: int = 6,
    layer_h: float = 0.12,
    sharpen: float = 2.0,
    contrast: float = 1.5,
    save_preview: bool = True,
    flip_y: bool = True,
    white_collapse: bool = True,
    merge_features: float = 0.0,
    merge_min_size: int = 0,
    chroma_decouple: bool = False,
    cmy_smooth: float = 0.0,
    recalib_luminance: bool = False,
    # Iteration 50: aggressive cleanup knobs for "precision vs smoothness"
    # balance on real-world images (cartoon, photo, text-on-busyness). All
    # default 0 / off so v1 (std-overlap-detail-v2) reproduces byte-for-byte
    # unless explicitly enabled.
    dtop_median_size: int = 0,         # odd >=3 = median pre-filter on dTop
    cmy_merge_features: float = 0.0,   # 0..1 plateau consolidation of CMY
    cmy_merge_min_size: int = 0,       # tiny CMY blocks absorption
    cmy_merge_chroma_tol: float = 8.0, # Lab (a*,b*) tol for CMY merge gate
    cmy_median_size: int = 0,          # odd >=3 = median pre-filter on CMY
    dtop_min: float = 0.0,             # min dTop (mm) — forces W always caps
    highlight_protect: float = 0.0,     # protect bright spots from merging
    dtop_quantize_step: float = 0.0,    # terrace quantize step (mm) for solid W
    dtop_cmy_cover_margin: float = 0.0, # ensure W top >= CMY top + margin
) -> dict:
    """Run the OVERLAP lithophane engine and export 3MF + per-color STLs.

    Sizing modes:
      - Default (long_edge_mm=None): fixed physical canvas width_mm x height_mm
        (156x106 mm to match the Bambu reference print), with CENTER-CROP of the
        source image to the target aspect ratio. This reproduces Bambu's crop
        behaviour so the same source image produces the same visible content.
      - Legacy (long_edge_mm=<value>): scale the source to make its long edge
        equal long_edge_mm while preserving aspect (no crop). Use this when you
        want the full source image at the given physical size.

    top_max default = 1.6 mm: combined with dW=0.20 + CMY band 6*0.12=0.72 mm,
    total print thickness is ~2.52 mm — matches the Bambu reference (2.5 mm).

    white_collapse (default True): in near-white source regions, zero out the
    C/M/Y layers so the top white relief carries the print alone. Matches Bambu
    behaviour and eliminates CMY speckle in white areas.
    """
    os.makedirs(outdir, exist_ok=True)

    with Image.open(image_path) as im:
        rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)
    print(f"Input: {image_path}  ({rgb.shape[1]}x{rgb.shape[0]} px)")

    if long_edge_mm is not None:
        # Legacy: aspect-preserving scale to the long edge.
        scale = long_edge_mm / max(rgb.shape[1], rgb.shape[0])
        phys_w = rgb.shape[1] * scale
        phys_h = rgb.shape[0] * scale
    else:
        # Fixed canvas with center-crop to the target aspect (Bambu-like).
        phys_w, phys_h = width_mm, height_mm
        tgt_aspect = width_mm / height_mm
        h_px, w_px = rgb.shape[:2]
        src_aspect = w_px / h_px
        if src_aspect > tgt_aspect + 1e-6:
            new_w = int(round(h_px * tgt_aspect))
            x0 = (w_px - new_w) // 2
            rgb = rgb[:, x0:x0 + new_w]
            print(f"Center-crop: {w_px}x{h_px} -> {new_w}x{h_px} px "
                  f"(aspect {tgt_aspect:.3f})")
        elif src_aspect < tgt_aspect - 1e-6:
            new_h = int(round(w_px / tgt_aspect))
            y0 = (h_px - new_h) // 2
            rgb = rgb[y0:y0 + new_h]
            print(f"Center-crop: {w_px}x{h_px} -> {w_px}x{new_h} px "
                  f"(aspect {tgt_aspect:.3f})")
    params = LithophaneParams(
        width_mm=phys_w, height_mm=phys_h,
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
        white_collapse=white_collapse,
        merge_features=merge_features,
        merge_min_size=merge_min_size,
        chroma_decouple=chroma_decouple,
        cmy_smooth=cmy_smooth,
        recalib_luminance=recalib_luminance,
        dtop_median_size=dtop_median_size,
        cmy_merge_features=cmy_merge_features,
        cmy_merge_min_size=cmy_merge_min_size,
        cmy_merge_chroma_tol=cmy_merge_chroma_tol,
        cmy_median_size=cmy_median_size,
        dtop_min=dtop_min,
        highlight_protect=highlight_protect,
        dtop_quantize_step=dtop_quantize_step,
        dtop_cmy_cover_margin=dtop_cmy_cover_margin,
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
    parser.add_argument("--width-mm", type=float, default=156.0,
                        help="Physical width in mm (default: 156, matches Bambu)")
    parser.add_argument("--height-mm", type=float, default=106.0,
                        help="Physical height in mm (default: 106, matches Bambu)")
    parser.add_argument("--long-edge-mm", type=float, default=None,
                        help="Legacy: physical long edge size in mm. If set, "
                             "overrides width/height and uses aspect-preserving "
                             "scale (no crop). Default: None (use width/height)")
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
    parser.add_argument("--top-max", type=float, default=1.6,
                        help="Max top relief thickness in mm (default: 1.6, "
                             "yields ~2.5mm total to match Bambu)")
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
    parser.add_argument("--white-collapse", dest="white_collapse",
                        action="store_true", default=True,
                        help="Zero CMY in white regions (default: on, matches Bambu)")
    parser.add_argument("--no-white-collapse", dest="white_collapse",
                        action="store_false",
                        help="Disable white-collapse; keep full CMY everywhere")
    parser.add_argument("--merge-features", dest="merge_features", type=float,
                        default=0.0,
                        help="White-layer feature-merge strength 0..1 (default 0 = off). "
                             "Consolidates the relief into clean plateaus; 0.4-0.7 recommended.")
    parser.add_argument("--merge-min-size", dest="merge_min_size", type=int,
                        default=0,
                        help="Absorb connected components smaller than this many "
                             "pixels into the largest 4-neighbouring component. "
                             "Only used with --merge-features > 0. Use to clean up "
                             "anti-aliased / JPEG noise specks that survived the "
                             "luminance+gradient gate. 0 disables (v1 behaviour). "
                             "16-30 is a sane range at 0.15-0.30 mm pitch.")
    parser.add_argument("--chroma-decouple", dest="chroma_decouple",
                        action="store_true", default=False,
                        help="CMY carries ONLY hue/saturation; the white relief "
                             "carries ALL luminance (K-style shading). Makes CMY "
                             "safe to smooth. Off by default (baseline behaviour).")
    parser.add_argument("--cmy-smooth", dest="cmy_smooth", type=float,
                        default=0.0,
                        help="Gaussian sigma (px) applied to C/M/Y colour fields "
                             "after solving. Only meaningful with --chroma-decouple "
                             "(CMY is then pure colour, so blurring cannot hurt "
                             "brightness/detail). 0.5-1.0 smooths colour backdrops.")
    parser.add_argument("--recalib-luminance", dest="recalib_luminance",
                        action="store_true", default=False,
                        help="With --chroma-decouple: re-derive dTop so the neutral "
                             "white layer hits the target L* exactly (cures the dE "
                             "explosion caused by decoupling). No-op without "
                             "--chroma-decouple.")
    # Iteration 50: precision-vs-smoothness balance knobs.
    parser.add_argument("--dtop-median-size", dest="dtop_median_size",
                        type=int, default=0,
                        help="Median filter window (px, odd >=3) applied to dTop "
                             "BEFORE feature merge. Edge-preserving; kills sub-"
                             "min_size specks AND surfaces real plateaus for the "
                             "lum_tol gate to latch onto. 3-5 recommended at 0.18 "
                             "mm pitch; 0 disables (v1 behaviour).")
    parser.add_argument("--cmy-merge-features", dest="cmy_merge_features",
                        type=float, default=0.0,
                        help="CMY plateau consolidation strength 0..1. Distinct "
                             "from --cmy-smooth (gaussian): this is edge-preserving "
                             "and forms CLEAN PLATEAUS — kills the 'messy background "
                             "walk' in slicer view. 0.5-0.8 typical. 0 disables.")
    parser.add_argument("--cmy-merge-min-size", dest="cmy_merge_min_size",
                        type=int, default=0,
                        help="Absorb CMY connected components smaller than this "
                             "many pixels into the largest 4-neighbouring "
                             "component. 16-30 recommended. 0 disables.")
    parser.add_argument("--cmy-merge-chroma-tol", dest="cmy_merge_chroma_tol",
                        type=float, default=8.0,
                        help="CIE Lab (a*,b*) tolerance for CMY merge gate "
                             "(default 8 = barely-perceptible chroma step).")
    parser.add_argument("--cmy-median-size", dest="cmy_median_size",
                        type=int, default=0,
                        help="Median filter window (px, odd >=3) applied to dC/dM/"
                             "dY before the CMY merge gate. 3-5 recommended; 0 "
                             "disables.")
    parser.add_argument("--dtop-min", dest="dtop_min", type=float, default=0.0,
                        help="Force dTop >= this thickness (mm) everywhere, so "
                             "the white relief ALWAYS caps the print. 0 disables "
                             "(v1 behaviour). 0.08-0.15 recommended with "
                             "--chroma-decouple to cure 'W cap missing' on dark "
                             "scenes.")
    parser.add_argument("--highlight-protect", dest="highlight_protect",
                        type=float, default=0.0,
                        help="Protect small bright spots from being merged into "
                             "darker neighbours by --merge-min-size. A pixel is a "
                             "'highlight' if its linear luminance exceeds the "
                             "local mean by this threshold (0..1). Any connected "
                             "component containing such a pixel is kept. "
                             "0 disables. 0.05-0.10 typical for coins/reflections.")
    parser.add_argument("--dtop-quantize-step", dest="dtop_quantize_step",
                        type=float, default=0.0,
                        help="Quantize the white-relief dTop to terraces of this "
                             "height (mm). This turns tiny continuous height "
                             "variations into discrete flat plateaus so the "
                             "slicer fills the white layer with solid infill "
                             "instead of porous micro-contours. 0 disables. "
                             "0.10-0.15 mm recommended (= 1-2 layer heights).")
    parser.add_argument("--dtop-cmy-cover-margin", dest="dtop_cmy_cover_margin",
                        type=float, default=0.0,
                        help="Ensure the front white relief is always at least "
                             "this much thicker than the tallest CMY layer at "
                             "each pixel. Cures 'W does not cover CMY' in slicer "
                             "views. 0 disables. 0.05-0.10 mm recommended.")
    args = parser.parse_args()

    export_lithophane(
        image_path=args.image,
        outdir=args.outdir,
        printer=args.printer,
        width_mm=args.width_mm,
        height_mm=args.height_mm,
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
        white_collapse=args.white_collapse,
        merge_features=args.merge_features,
        merge_min_size=args.merge_min_size,
        chroma_decouple=args.chroma_decouple,
        cmy_smooth=args.cmy_smooth,
        recalib_luminance=args.recalib_luminance,
        dtop_median_size=args.dtop_median_size,
        cmy_merge_features=args.cmy_merge_features,
        cmy_merge_min_size=args.cmy_merge_min_size,
        cmy_merge_chroma_tol=args.cmy_merge_chroma_tol,
        cmy_median_size=args.cmy_median_size,
        dtop_min=args.dtop_min,
        highlight_protect=args.highlight_protect,
        dtop_quantize_step=args.dtop_quantize_step,
        dtop_cmy_cover_margin=args.dtop_cmy_cover_margin,
    )


if __name__ == "__main__":
    main()
