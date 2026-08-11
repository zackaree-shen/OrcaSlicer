"""Lithophane Studio — modern GUI (pywebview + HTML/Three.js).

Python backend exposing a bridge (LithoApi) to the HTML/JS frontend, reusing
the existing engine (litho_engine / litho_3mf / litho_reverse). The frontend
in litho_web/ is a modern dark UI with a Three.js interactive 3D view.

Run:  python litho_gui_web.py
"""

from __future__ import annotations

import base64
import io
import os
import sys
import time
import threading

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from litho_core import LithophaneParams, validate_mesh
from litho_engine import LithoMode, ColorOrder, color_lithophane_engine
from litho_3mf import assemble_lithophane_parts, write_3mf
from litho_color import DEFAULT_TD, _resample_rgb, linear_to_srgb8

WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "litho_web")


class LithoApi:
    """Methods callable from the JS frontend via pywebview.api.*"""

    def __init__(self):
        self.rgb = None
        self.image_path = None
        self.last_meshes = None
        self.last_params = None
        self.out_dir = None

    # ------------------------------------------------------------ helpers
    def _img_b64(self, arr, max_side=800):
        """numpy (H,W,3) -> base64 PNG (downscaled for preview)."""
        im = Image.fromarray(arr)
        scale = min(max_side / im.width, max_side / im.height, 1.0)
        if scale < 1.0:
            im = im.resize((int(im.width * scale), int(im.height * scale)), Image.BILINEAR)
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    def _mesh_json(self, meshes):
        """meshes {color:(verts,faces)} -> {color:{verts:[...],faces:[...]}}"""
        out = {}
        for c, (v, f) in meshes.items():
            if len(f) == 0:
                out[c] = {"verts": [], "faces": []}
                continue
            out[c] = {"verts": np.asarray(v, dtype=float).ravel().tolist(),
                      "faces": np.asarray(f, dtype=int).ravel().tolist()}
        return out

    def _default_outdir(self):
        desk = os.path.join(os.path.expanduser("~"), "Desktop")
        base = os.path.join(desk, "lithophane_exports_web")
        os.makedirs(base, exist_ok=True)
        return base

    # ------------------------------------------------------------ JS API
    def pick_image(self):
        """Open file dialog, load image, return preview thumb + size."""
        import webview
        try:
            # pywebview doesn't ship a file dialog API; use tkinter's silently
            # (headless pick only, no main window).
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk(); root.withdraw()
            path = filedialog.askopenfilename(
                title="选择图片", filetypes=[("图片", "*.png *.jpg *.jpeg *.bmp"), ("All", "*.*")])
            root.destroy()
        except Exception:
            path = ""
        if not path:
            return {"ok": False, "error": "未选择"}
        try:
            with Image.open(path) as im:
                self.rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)
            self.image_path = path
            w, h = self.rgb.shape[1], self.rgb.shape[0]
            return {"ok": True, "thumb": self._img_b64(self.rgb),
                    "w": w, "h": h, "name": os.path.basename(path)}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def build(self, params: dict):
        """Generate meshes for the given params; return previews + 3D data."""
        if self.rgb is None:
            return {"ok": False, "error": "未选择图片"}
        try:
            p = LithophaneParams(
                width_mm=float(params["width_mm"]),
                height_mm=float(params["height_mm"]),
                pixel_pitch_mm=float(params.get("pitch", 0.3)),
                base_thickness=float(params.get("dwhite", 0.3)),
            )
            layer_h = float(params.get("layer_h", 0.2))
            layers = int(params.get("layers", 8))
            mode = LithoMode(params["mode"])
            order = ColorOrder(params.get("order", "CMY"))
            td = {
                "C": (float(params.get("td_c", 0.3)), 3.0, 3.0),
                "M": (3.0, float(params.get("td_m", 0.3)), 3.0),
                "Y": (3.0, 3.0, float(params.get("td_y", 0.3))),
                "W": DEFAULT_TD["W"],
            }
            # Grid guard: cap solver grid like the old GUI (keep interactive).
            from litho_core import thickness_grid_shape
            gx, gy = thickness_grid_shape(self.rgb.shape[0], self.rgb.shape[1], p)
            if gx * gy > 400_000:
                scale = ((gx * gy) / 400_000) ** 0.5
                p.pixel_pitch_mm = max(p.pixel_pitch_mm * scale, 0.3)
                pitch_cmy, pitch_top = 0.8 * scale, 0.25 * scale
            else:
                pitch_cmy, pitch_top = 0.8, 0.25

            t0 = time.time()
            meshes, dE, gamut, reached = color_lithophane_engine(
                self.rgb, mode=mode, order=order, params=p, td=td,
                layers_max=layers, layer_h=layer_h, exact=False,
                pitch_cmy=pitch_cmy, pitch_top=pitch_top)
            elapsed = time.time() - t0

            self.last_meshes = meshes
            self.last_params = (p, layers, layer_h, td, mode, order)

            total_faces = sum(len(f) for _, f in meshes.values())
            dE_med = float(np.median(dE)) if dE is not None else 0.0

            return {
                "ok": True,
                "reached_b64": self._img_b64(reached) if reached is not None else None,
                "meshes": self._mesh_json(meshes),
                "dE_med": round(dE_med, 2),
                "total_faces": total_faces,
                "elapsed": round(elapsed, 1),
                "size": f"{p.width_mm:.0f}×{p.height_mm:.0f}mm",
            }
        except Exception as e:
            import traceback; traceback.print_exc()
            return {"ok": False, "error": str(e)}

    def export_all(self, fmt: str = "BOTH"):
        """Export current build as 3MF and/or STLs to Desktop/lithophane_exports_web."""
        if self.last_meshes is None:
            return {"ok": False, "error": "先构建"}
        try:
            p, layers, layer_h, td, mode, order = self.last_params
            self.out_dir = os.path.join(self._default_outdir(), mode.value)
            os.makedirs(self.out_dir, exist_ok=True)

            if fmt in ("3MF", "BOTH"):
                parts, offsets, names, extruders = assemble_lithophane_parts(self.last_meshes)
                write_3mf(os.path.join(self.out_dir, "lithophane.3mf"), parts, offsets,
                          extruders, part_names=names, printer_model="Snapmaker U1",
                          printer_settings_id="Snapmaker U1 (0.4 nozzle)",
                          build_center_mm=(135.5, 136.0, 0.0))
            if fmt in ("STL", "BOTH"):
                from litho_core import export_stl
                stl_names = [("W","white"),("C","cyan"),("M","magenta"),
                             ("Y","yellow"),("top","top_white")]
                for key, name in stl_names:
                    v, f = self.last_meshes[key]
                    if len(f) == 0:
                        continue
                    export_stl(os.path.join(self.out_dir, f"litho_{name}.stl"), v, f,
                               name=f"lithophane_{name}")
            return {"ok": True, "dir": self.out_dir}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def reverse_import(self):
        """Pick a 3MF, reconstruct preview, return it for comparison."""
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk(); root.withdraw()
            path = filedialog.askopenfilename(title="选择 3MF",
                                              filetypes=[("3MF", "*.3mf")])
            root.destroy()
        except Exception:
            path = ""
        if not path:
            return {"ok": False, "error": "取消"}
        try:
            from litho_reverse import load_3mf_colors, reconstruct_from_meshes
            cm = load_3mf_colors(path)
            if "W" in cm and len(cm["W"]):
                v = cm["W"].reshape(-1, 3)
                w = float(v[:, 0].max() - v[:, 0].min())
                h = float(v[:, 1].max() - v[:, 1].min())
            else:
                w = h = 100.0
            recon = reconstruct_from_meshes(cm, w, h, pixel_pitch=max(w / 200, 0.5))
            return {"ok": True, "recon_b64": self._img_b64(recon)}
        except Exception as e:
            return {"ok": False, "error": str(e)}


def main():
    import webview
    api = LithoApi()
    window = webview.create_window(
        "Lithophane Studio",
        os.path.join(WEB_DIR, "index.html"),
        js_api=api,
        width=1280, height=800,
        min_size=(1000, 640),
        background_color="#0f1220",
    )
    webview.start()


if __name__ == "__main__":
    main()
